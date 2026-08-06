# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained convolution primitives for the XTTS-v2 HiFi-GAN decoder.

The GAN decoder is a deep chain of ``Conv1d`` + ``ConvTranspose1d`` layers (the
vocoder) plus a ``Conv2d`` SE-ResNet (the speaker encoder). ``ttnn`` has native
``conv1d``/``conv2d`` (with ``dilation``/``groups``) but no ``conv_transpose1d``,
so these primitives live here:

* :class:`TtConv1d`          — thin wrapper over ``ttnn.conv1d``.
* :class:`TtConvTranspose1d` — transpose conv expressed as a regular conv on the
  zero-stuffed input with a flipped, in/out-transposed kernel.
* :class:`TtConv2d`          — thin wrapper over ``ttnn.conv2d``.

Tensor convention: **channels-last** — ``[N, L, C]`` for 1D, flat ``[1, 1, N*H*W, C]``
(TILE) for 2D, the layouts ttnn's convs consume — avoiding per-layer transposes and
relayouts. Weights are PyTorch tensors (``Conv1d``: ``[out, in/groups, k]``;
``ConvTranspose1d``: ``[in, out, k]``; ``Conv2d``: ``[out, in/groups, kh, kw]``).

Defaults: **fp32 activations**, bf16 weights, HiFi4, ``fp32_dest_acc_en``. bf16
activations lose too much through this deep a chain (the ~36 residual adds + MRF
sums + tanh compound, and PCC drifts below 0.99 as the sequence lengthens); fp32
activations hold PCC ~0.999 at length. fp32 activations no longer OOM the wide
layers because conv1d auto-width-slices DRAM inputs — but they need a larger
``l1_small_size`` (32768) on the device. bf16 weights are kept: fp32 weights gave
no accuracy gain. Pass ``activations_dtype=ttnn.bfloat16`` for a faster, lower-
accuracy mode.
"""

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule

# The vocoder's conditioning-bias fold (TtConv1d.forward) prepares the combined bias on HOST via
# ttnn.from_device — a device->host READ that is fatal inside a ttnn trace capture. It is the faster
# path for eager execution (fused conv-bias epilogue, no full-length broadcast add), so it stays the
# DEFAULT. When capturing a trace, wrap the region in ``cond_bias_trace_safe()`` (or call
# ``set_cond_bias_trace_safe(True)``) to switch to the equivalent trace-safe post-conv device add.
_COND_BIAS_TRACE_SAFE = False


def set_cond_bias_trace_safe(flag: bool) -> bool:
    """Toggle the trace-safe conditioning-bias path; returns the previous value (for restore)."""
    global _COND_BIAS_TRACE_SAFE
    prev = _COND_BIAS_TRACE_SAFE
    _COND_BIAS_TRACE_SAFE = bool(flag)
    return prev


class cond_bias_trace_safe:
    """Context manager: force the trace-safe conditioning-bias add inside `with`, restore after."""

    def __enter__(self):
        self._prev = set_cond_bias_trace_safe(True)
        return self

    def __exit__(self, *exc):
        set_cond_bias_trace_safe(self._prev)
        return False


def _interleaved(x: ttnn.Tensor, shape, *, row_major: bool) -> ttnn.Tensor:
    """Bring a (possibly sharded) conv output to interleaved DRAM and reshape to
    ``shape`` so downstream ops consume it as ``[N, L, C]`` (or ``[N, H, W, C]``).

    ``to_memory_config(DRAM)`` is a cheap no-op when the conv already returns
    interleaved DRAM (the width-sliced path always does) and otherwise gathers an
    L1-sharded output. ``row_major=True`` additionally untilizes to ROW_MAJOR —
    needed by conv2d's downstream (speaker encoder) and by conv-transpose
    zero-stuffing. ``row_major=False`` keeps TILE, so a conv1d -> eltwise -> conv1d
    chain avoids the per-op untilize round-trip: ttnn.conv1d accepts a TILE
    interleaved input directly (verified PCC 1.0), and leaky_relu/add/mul/tanh all
    run in TILE, so the vocoder's deep conv chain never leaves tiled layout."""
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    if row_major:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(x, shape)


# Per-core resident-activation ceiling for the sharded conv chain. Kept below the smallest
# observed circular-buffer clash point (k=3 clashes ~64KB/core; k7/k11 have bigger CBs but
# were verified to fit at 48KB), so the chain's L1-resident tensors always leave room for the
# convs' circular buffers. The profiled decode lengths (latent_len<=32) sit at 32-48KB/core and
# stay sharded; longer sequences (the demo) exceed this and fall back to the interleaved path.
_SHARD_L1_BUDGET_BYTES = 48 * 1024


def _shard_height(device, nhw: int) -> int:
    grid = device.compute_with_storage_grid_size()
    ncores = int(grid.x) * int(grid.y)
    return math.ceil(math.ceil(nhw / ncores) / 32) * 32


def sharded_chain_fits_l1(device, length: int, channels: int, dtype_bytes: int = 4) -> bool:
    """Whether a HEIGHT_SHARDED activation of ``length x channels`` is small enough per core
    to keep the resblock chain resident in L1 without clashing the convs' circular buffers.
    Length-dependent, so it is checked at forward time (the same block shards at short decode
    lengths and falls back at long ones)."""
    return _shard_height(device, length) * channels * dtype_bytes <= _SHARD_L1_BUDGET_BYTES


def height_shard_l1(device, x: ttnn.Tensor, channels: int) -> ttnn.Tensor:
    """Bring a ``[N, L, C]`` (or ``[N, 1, L, C]``) TILE tensor to an L1 HEIGHT_SHARDED
    layout spread over the full compute grid, tile-aligned per core.

    This is the entry point for a sharded conv chain: once the activation is L1-sharded,
    ``ttnn.conv1d`` takes its L1 path (input already sharded -> no InterleavedToSharded;
    ``memory_config=None`` -> output stays sharded, no ShardedToInterleaved), so a chain of
    same-shape convs + eltwise stays in L1 and pays the reshard only once (here) and the
    gather only once (``_interleaved`` at chain exit).  Same-shape convs share this exact
    spec, so no per-conv re-derivation happens (verified PCC ~1.0)."""
    mem = ttnn.create_sharded_memory_config(
        shape=(_shard_height(device, x.shape[-2]), channels),
        core_grid=ttnn.CoreGrid(
            y=int(device.compute_with_storage_grid_size().y), x=int(device.compute_with_storage_grid_size().x)
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(x, mem)


def block_shard_grid(device, length: int, channels: int):
    """``(gx, gy, rows_per_core)`` for a BLOCK_SHARDED L1 placement, or ``None`` if the shape can't
    take one.

    BLOCK splits channels as well as rows, which is what makes it worth having alongside
    :func:`height_shard_l1`: on the vocoder's stage 0 (1120 x 256, fp32) height sharding is capped by
    tile alignment at a 32-row shard, i.e. only ceil(1120/32) = 35 of 110 cores and 32 KB/core
    resident. The block placement is 128 x 32 over 80 cores at 16 KB/core. Measured consequences:
    the k3 convs roughly halve (24 -> 13.4 us) as do their halos (7 -> 1.4 us), and -- the reason this
    exists at all -- **k7/k11 become buildable inside an L1-resident chain**, which height sharding
    simply cannot do (every height-sharded variant dies at program.cpp:170/176 on a circular-buffer
    clash, baseline config included).

    ``gx = channels / 32`` is NOT a free parameter. ttnn.conv1d silently RE-GRIDS a wider shard -- a
    4x10 / 128x64 input comes back as 8x10 / 128x32 -- which would desync the conv output from the
    chain activation and drop the residual add off its matching-spec L1 fast path. At one channel tile
    per grid column the conv hands back the exact spec it was given (verified identical shape, grid and
    orientation for k3 through k11). Only useful where channels/32 >= 2; stages 1-3 are narrower than
    stage 0 and would score fewer cores than height sharding, so they keep the height placement.
    """
    grid = device.compute_with_storage_grid_size()
    if channels % 32:
        return None
    gx, gy = channels // 32, int(grid.y)
    if gx < 2 or gx > int(grid.x):
        return None
    rows_per_core = math.ceil(math.ceil(length / gy) / 32) * 32
    if rows_per_core * gy < length:
        return None
    return gx, gy, rows_per_core


def block_chain_fits_l1(device, length: int, channels: int, dtype_bytes: int = 4) -> bool:
    """Whether a BLOCK_SHARDED activation of ``length x channels`` leaves the convs' circular buffers
    room in L1. Same budget as the height path, but the per-core tile is (rows_per_core x 32) rather
    than (shard_height x channels), so it clears the bar at shapes height sharding cannot."""
    plan = block_shard_grid(device, length, channels)
    if plan is None:
        return False
    _, _, rows_per_core = plan
    return rows_per_core * 32 * dtype_bytes <= _SHARD_L1_BUDGET_BYTES


def block_shard_l1(device, x: ttnn.Tensor, channels: int) -> ttnn.Tensor:
    """BLOCK_SHARDED counterpart of :func:`height_shard_l1` -- same contract (the returned tensor is
    the entry point of an L1-resident conv chain), different cut. See :func:`block_shard_grid`."""
    gx, gy, rows_per_core = block_shard_grid(device, x.shape[-2], channels)
    mem = ttnn.create_sharded_memory_config(
        shape=(rows_per_core, channels // gx),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(x, mem)


def _subpixel_weight(weight: torch.Tensor, bias: torch.Tensor | None, stride: int):
    """Fold a ``ConvTranspose1d`` weight ``[in, out, k]`` (HiFi-GAN padding
    ``(k - stride) // 2``) into ONE regular-conv weight ``[out*stride, in, Ic]`` with
    phase-major output channels (channel ``phi*out + o``) plus a symmetric padding, so
    ``conv1d(x, .)`` on the *un-stuffed* input followed by a length-interleave of the
    ``stride`` phase-channels reproduces the transpose conv exactly. This is the
    polyphase / sub-pixel identity: it avoids zero-stuffing (its pad/slice ops) and the
    conv MACs otherwise spent multiplying the inserted zeros. Proven against
    ``torch.nn.functional.conv_transpose1d`` (see scratch ``polyphase_verify.py``).

    Returns ``(weight_sp [out*stride, in, Ic], bias_sp [out*stride] | None, padding)``.
    """
    in_ch, out_ch, k = weight.shape
    pad_t = (k - stride) // 2
    phases = []  # (phase kernel [out, in, I], pad_left, pad_right)
    for phi in range(stride):
        j0 = (phi + pad_t) % stride
        idxs = list(range(j0, k, stride))  # taps contributing to this phase
        d = (phi + pad_t - j0) // stride
        w = torch.flip(weight[:, :, idxs], dims=[-1]).permute(1, 0, 2).contiguous()  # [out, in, I]
        phases.append((w, w.shape[-1] - 1 - d, d))
    pad_l = max(p[1] for p in phases)
    pad_r = max(p[2] for p in phases)
    assert pad_l == pad_r, f"expected symmetric common padding, got {pad_l} vs {pad_r}"
    ic = pad_l + pad_r + 1
    weight_sp = torch.zeros(stride * out_ch, in_ch, ic)
    for phi, (w, p_l, _) in enumerate(phases):
        off = pad_l - p_l  # align each phase kernel within the common window
        weight_sp[phi * out_ch : (phi + 1) * out_ch, :, off : off + w.shape[-1]] = w
    bias_sp = bias.repeat(stride) if bias is not None else None  # phase-major tiling
    return weight_sp, bias_sp, pad_l


class TtConv1d(LightweightModule):
    """1D convolution over a channels-last ``[N, L, C]`` device tensor.

    ``padding`` follows PyTorch semantics (symmetric); for a "same"-length dilated
    conv pass ``padding = dilation * (kernel_size - 1) // 2``.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_channels, in_channels // groups, kernel_size]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: ttnn.UnaryWithParam | None = None,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
        act_double_buffer: bool | None = None,
        weight_scale: float = 1.0,
        conv_config_overrides: dict | None = None,
    ):
        super().__init__()
        assert weight.dim() == 3, f"expected Conv1d weight [out, in/groups, k], got {tuple(weight.shape)}"
        out_channels, in_per_group, kernel_size = weight.shape
        # ``weight_scale`` scales the WEIGHT only (not the bias): conv is linear in its input, so a
        # constant output scale folds into the weight. Used to absorb HiFi-GAN's MRF mean (1/num_kernels)
        # into the *next* layer's weights — leaky_relu(c*x)==c*leaky_relu(x) for c>0, so the mean scale
        # commutes through the pre-activation and lands here, removing a per-stage ttnn.mul. The bias
        # (and any folded cond_bias) must stay unscaled, so it is applied to the un-scaled weight below.
        if weight_scale != 1.0:
            weight = weight * weight_scale

        self.device = device
        self.in_channels = in_per_group * groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activations_dtype = activations_dtype

        # ttnn.conv1d takes the raw PyTorch weight layout and preprocesses it on
        # first call; we cache the preprocessed device weight for reuse.
        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        # Un-preprocessed copy of the bias, kept on device as fp32/tiled, so a runtime
        # per-channel term can be folded into this conv's fused bias epilogue — see
        # ``forward``'s ``cond_bias``. Lets HiFi-GAN's conditioning add be absorbed into
        # the upsample conv instead of running as a separate full-length broadcast add.
        self._raw_bias_fp32 = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)
            self._raw_bias_fp32 = ttnn.from_torch(
                bias.reshape(1, 1, 1, -1).float(), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )
        # Pristine (host, un-preprocessed) copies, kept so a change of input length can re-derive
        # from them -- see ``forward``. This conv sees variable-length input by nature (the vocoder
        # decodes whatever the GPT produced), which is exactly the case the cache below breaks on.
        self._host_weight, self._host_bias = self.tt_weight, self.tt_bias
        self._prepared_for = None

        # No forced shard_layout: HEIGHT_SHARDED fails the DRAM slicer on the wide (1024-channel)
        # layers with short spatial extent; auto-sharding picks a valid layout per shape (PCC
        # ~0.9999). The sharded-chain mode (forward's ``keep_sharded``) needs no shard_layout
        # either — the conv takes its L1 path purely from being handed an already-L1-sharded
        # input. ``act_double_buffer`` (opt-in) is dropped only on the sharding-capable resblock
        # convs to fit their circular buffers alongside the resident sharded activations in L1.
        # ``activation`` (e.g. leaky_relu) is fused onto the conv output (post-bias),
        # so ``conv(x, activation=leaky_relu) == leaky_relu(conv(x))`` — used to fold
        # HiFi-GAN's between-conv activations into the producing conv.
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            activation=activation,
            **({"enable_act_double_buffer": act_double_buffer} if act_double_buffer is not None else {}),
        )
        # Optional per-conv scheduling overrides (perf-only Conv1dConfig fields: act_block_h_override,
        # force_split_reader, enable_*_double_buffer, enable_activation_reuse, ...). Bit-exact — they
        # change how the conv is tiled/streamed, not the math. Used by the conv-config sweep.
        if conv_config_overrides:
            for _k, _v in conv_config_overrides.items():
                setattr(self.conv_config, _k, _v)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor, cond_bias: ttnn.Tensor | None = None, keep_sharded: bool = False) -> ttnn.Tensor:
        batch_size, input_length, _ = x.shape
        # The prepared weights cached at the end of this function are only valid for the
        # parallelization ttnn.conv1d chose for *this* input length and memory config, and ttnn
        # cannot detect a stale one (see TtConv2d.forward for the mechanism and the measurement).
        # Re-derive from the pristine host copies whenever the signature moves.
        key = (batch_size, input_length, x.dtype, x.layout, x.memory_config())
        if key != self._prepared_for:
            self.tt_weight, self.tt_bias = self._host_weight, self._host_bias
        # ``cond_bias`` ([1,1,1,C], fp32) is a per-channel conditioning constant. Two equivalent
        # ways to apply it (identical math — a per-output-channel bias add):
        #   * EAGER (default, faster): fold it into the conv's bias so conv1d adds it in its fused
        #     epilogue — needs a host-prepared bias (from_device), a device->host READ.
        #   * TRACE-SAFE (_COND_BIAS_TRACE_SAFE): add it on device AFTER the conv, broadcasting over
        #     length. No host transfer, so it is legal inside a trace capture (from_device is fatal
        #     there). Set via cond_bias_trace_safe() around a trace region.
        fold = cond_bias is not None and not _COND_BIAS_TRACE_SAFE
        bias_tensor = self.tt_bias
        if fold:
            # Combine on device, then move to host so ttnn.conv1d prepares it through its normal
            # (host) bias path (a device unprepared bias makes conv pull it back to host anyway).
            combined = ttnn.to_layout(ttnn.add(self._raw_bias_fp32, cond_bias), ttnn.ROW_MAJOR_LAYOUT)
            bias_tensor = ttnn.from_device(combined)
            ttnn.deallocate(combined)
        out, out_length, [weight, bias] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=bias_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            batch_size=batch_size,
            input_length=input_length,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        if not fold:  # bias is the (prepared) base bias — cache it; when folding it is the combined bias
            self.tt_bias = bias
        self._prepared_for = key
        if keep_sharded:
            # Sharded-chain mode (input was L1-sharded): the L1 path already returns a
            # HEIGHT_SHARDED TILE output; reshape is a metadata op that preserves the sharding
            # (verified). No gather here — the chain owner gathers once at the end. cond_bias is
            # never used on these (resblock) convs, so the trace-safe add path below is unreachable.
            return ttnn.reshape(out, [batch_size, out_length, self.out_channels])
        # Keep TILE: the conv already emits TILE/interleaved-DRAM, and the whole
        # vocoder conv chain (+ its eltwise ops) consumes TILE, so we skip the
        # per-conv untilize->ROW_MAJOR round-trip.
        out = _interleaved(out, [batch_size, out_length, self.out_channels], row_major=False)
        if cond_bias is not None and not fold:  # trace-safe post-conv device add
            cb = ttnn.reshape(cond_bias, [1, 1, self.out_channels])
            if cb.dtype != out.dtype:
                cb = ttnn.typecast(cb, out.dtype)
            out = ttnn.add(out, cb)
        return out


class TtConvTranspose1d(LightweightModule):
    """``torch.nn.ConvTranspose1d`` with ``padding = (kernel_size - stride) // 2``
    (the HiFi-GAN upsampling convention, giving an exact ``stride``x upsample).

    Implemented as the polyphase / sub-pixel form of the transpose conv: ONE regular
    :class:`TtConv1d` with ``out*stride`` channels runs on the *un-stuffed* input, and
    its phase-major output channels are interleaved into the length dim (see
    :func:`_subpixel_weight`). This replaces the older zero-stuff-then-convolve scheme,
    which materialised a ``stride``x-inflated tensor (pad/slice TM ops) and spent most of
    the conv's MACs multiplying inserted zeros. Requires ``k - stride`` even (true for all
    XTTS upsample layers: k/stride = 16/8, 4/2).
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [in_channels, out_channels, kernel_size]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int,
        **conv_kwargs,
    ):
        super().__init__()
        assert weight.dim() == 3, f"expected ConvTranspose1d weight [in, out, k], got {tuple(weight.shape)}"
        in_channels, out_channels, kernel_size = weight.shape
        assert (kernel_size - stride) % 2 == 0, f"need (k - stride) even, got k={kernel_size}, stride={stride}"

        self.stride = stride
        self.out_channels = out_channels

        # Polyphase: a single conv on the un-stuffed input with out*stride channels,
        # then a length-interleave of those phase-channels reproduces the transpose conv.
        weight_sp, bias_sp, padding = _subpixel_weight(weight, bias, stride)
        self.conv = TtConv1d(device, weight_sp, bias_sp, stride=1, padding=padding, **conv_kwargs)

    def forward(self, x: ttnn.Tensor, cond_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        # Polyphase upsample: one conv (out*stride channels) on the un-stuffed input,
        # then interleave the phase-major channels into length. ``cond_bias`` (if given)
        # is a per-channel constant folded into the ups bias — a transpose conv adds its
        # bias per output channel post-conv, so it equals the HiFi-GAN conditioning add.
        # It is tiled ``stride``x to match the conv's out*stride channels.
        batch_size, input_length, _ = x.shape
        inner_cond = None
        if cond_bias is not None:
            inner_cond = ttnn.concat([cond_bias] * self.stride, dim=-1)  # [1,1,1,out*stride]
        z = self.conv(x, cond_bias=inner_cond)  # [N, L, out*stride], phase-major channels
        if inner_cond is not None:
            ttnn.deallocate(inner_cond)

        # Sub-pixel shuffle: [N, L, out*stride] -> [N, L*stride, out]. In row-major this
        # is a contiguous reinterpretation that lands phase phi of position q at output
        # index q*stride + phi (the transpose-conv output ordering).
        #
        # Two ways to spell it, BIT-EXACT to each other (maxdiff 0.0 on all four XTTS ups
        # shapes) but with very different device cost, because a TILE-layout reshape has to
        # gather each output tile from ``stride`` separate input column-blocks:
        #   * stride 2 -- ttnn.reshape straight on the TILE tensor wins big, and drops the
        #     untilize + retilize entirely: ups[2] 80.8 -> 56.4 us, ups[3] 138.9 -> 55.3 us.
        #   * stride 8 -- the same call LOSES (ups[1] 65.2 -> 93.6 us, ups[0] 39.3 -> 40.9),
        #     the 8-way tile gather costing more than a row-major round-trip, so those keep
        #     the untilize -> reshape -> retilize path.
        # Measured per-op under tracy on Blackhole; see the ups rows of the decoder report.
        shape = [batch_size, input_length * self.stride, self.out_channels]
        if self.stride <= 2:
            return ttnn.reshape(z, shape)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, shape)
        return ttnn.to_layout(z, ttnn.TILE_LAYOUT)


class TtConv2d(LightweightModule):
    """2D convolution over a **flat** channels-last ``[1, 1, N*H*W, C]`` TILE tensor —
    the exact layout ``ttnn.conv2d`` produces, so a conv -> eltwise -> conv chain never
    relayouts. The spatial extent travels beside the tensor (``forward`` takes
    ``input_height``/``input_width`` and returns the output's), because the flat form
    doesn't carry it.

    Keeping TILE (instead of untilizing to a ``[N, H, W, C]`` ROW_MAJOR view) is what the
    vocoder's conv1d chain already does, and it matters more here: an untilize + 4D
    reshape + retilize per conv cost ~200us each in the speaker encoder, and the 4D TILE
    form pads W to a tile per H row, inflating every eltwise op that follows.

    ``forward`` returns the conv's output **as produced** — L1-sharded, unless
    ``memory_config`` asks otherwise. That is not just cheaper by one gather: ttnn.conv2d
    picks its execution path from where the input lives (``determine_conv2d_execution_path``),
    and its DRAM path brackets the conv with a 4D unflatten + re-flatten of the activation
    — two full relayouts, ~180us per conv at the first stage. Handing the next conv an L1
    input keeps the whole chain on the L1 path, where those reshapes don't exist.

    ``stride``/``padding`` follow PyTorch semantics (symmetric). ``activation`` (e.g. relu)
    is fused onto the conv output post-bias, so ``conv(x, activation=relu) == relu(conv(x))``.
    Used by the speaker-encoder SE-ResNet (all 3x3 / 1x1 convs).
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_channels, in_channels // groups, kh, kw]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int = 1,
        padding: int = 1,
        activation: ttnn.UnaryWithParam | None = None,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
    ):
        super().__init__()
        assert weight.dim() == 4, f"expected Conv2d weight [out, in, kh, kw], got {tuple(weight.shape)}"
        out_channels, in_channels, kh, kw = weight.shape

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.activations_dtype = activations_dtype

        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)
        # The pristine (host, un-preprocessed) weights, kept for the lifetime of the module so a
        # shape change can re-derive from them -- see ``forward``. Costs one host-side copy of the
        # weights; before this they were dropped once the first call replaced them with the
        # prepared device tensor.
        self._host_weight, self._host_bias = self.tt_weight, self.tt_bias
        self._prepared_for = None

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            activation=activation,
            output_layout=ttnn.TILE_LAYOUT,
            # The halo's config tensors default to L1_SMALL, and they are cached per conv
            # program: the speaker encoder's 36 convs exhaust the 32 KB L1_SMALL region a
            # caller typically opens the device with (bank_manager OOM on the 1760 B
            # allocation). DRAM costs nothing measurable — they are read once per conv.
            config_tensors_in_dram=True,
            # Double-buffer both operand streams. Pure scheduling — it changes how the conv is
            # streamed, not the math — and it is worth the most exactly where the conv has least
            # to work with. Measured per speaker-encoder stage against the defaults: layer1
            # -0.1us, layer2 -1.0us, layer3 -4.5us, layer4 -14.8us. Layer4 dominates because its
            # 3x3 256->256 on a 101x8 image has only 216 output tiles to spread over cores (a
            # block shard caps at 8x9=72: grid_x is bounded by the 8 channel tiles, grid_y must
            # divide the 27 row tiles) against a 72-tile-deep K, so it is starved of overlap
            # rather than of parallelism.
            #
            # ``full_inner_dim=True`` belongs here on paper and is worth a further -7us at
            # layer4, but it SILENTLY COMPUTES THE WRONG ANSWER on small spatial extents: the
            # speaker encoder scores PCC 0.20-0.34 for mel_len < 128 with it on (0.999 without),
            # while mel_len >= 128 is unaffected. It is a scheduling flag, so that is a ttnn bug
            # rather than a numerics trade — do not re-enable it without a shape sweep.
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        input_height: int,
        input_width: int,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, int, int]:
        # Caching the prepared weights below is what keeps ttnn.conv2d's one-time weight
        # preprocessing (a host round-trip) off every call -- but they are only valid for the
        # parallelization the conv chose, which depends on the input shape and memory config.
        # ttnn cannot detect a stale one: is_valid_device_conv_weights checks only layout, rank,
        # out_channels and dtype (prepare_conv2d_weights.cpp:1069, in_channels is unused), so a
        # weight prepared for another shape passes, is used as-is, and the conv silently returns a
        # wrong answer -- no error, no warning. Measured on the speaker encoder: one module reused
        # across mel_len 200 -> 512 scored PCC 0.302 against 0.999 for a fresh one, and restoring
        # these pristine weights was what recovered it. So key the cache here and re-derive from
        # the host copies whenever the signature moves.
        key = (x.shape[0], input_height, input_width, x.dtype, x.layout, x.memory_config(), memory_config)
        if key != self._prepared_for:
            self.tt_weight, self.tt_bias = self._host_weight, self._host_bias
        out, (out_h, out_w), [weight, bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=x.shape[0],
            input_height=input_height,
            input_width=input_width,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            memory_config=memory_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        self.tt_bias = bias
        self._prepared_for = key
        # No relayout on the way out: the output is already the flat
        # [1, 1, N*out_h*out_w, C] TILE form the next conv / eltwise op consumes.
        return out, out_h, out_w
