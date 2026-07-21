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

Tensor convention: **channels-last** — ``[N, L, C]`` for 1D, ``[N, H, W, C]`` for
2D (ROW_MAJOR, on device), the layouts ttnn's convs consume — avoiding per-layer
transposes. Weights are PyTorch tensors (``Conv1d``: ``[out, in/groups, k]``;
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

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


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
    ):
        super().__init__()
        assert weight.dim() == 3, f"expected Conv1d weight [out, in/groups, k], got {tuple(weight.shape)}"
        out_channels, in_per_group, kernel_size = weight.shape

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

        # No forced shard_layout: HEIGHT_SHARDED fails the DRAM slicer on the wide
        # (1024-channel) layers with short spatial extent. Auto-sharding picks a
        # valid layout per shape and gives PCC ~0.9999.
        # ``activation`` (e.g. leaky_relu) is fused onto the conv output (post-bias),
        # so ``conv(x, activation=leaky_relu) == leaky_relu(conv(x))`` — used to fold
        # HiFi-GAN's between-conv activations into the producing conv.
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
            activation=activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor, cond_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        batch_size, input_length, _ = x.shape
        # ``cond_bias`` ([1,1,1,C], fp32, tiled) is a per-channel constant folded into the
        # bias so the conv adds it in its fused epilogue (free) instead of the caller
        # running a full-length broadcast add. It varies per call (depends on ``g``), so
        # the combined bias is rebuilt each call and never cached back onto ``self``.
        bias_tensor = self.tt_bias
        if cond_bias is not None:
            # Combine on device, then move to host so ttnn.conv1d prepares it through its
            # normal (host) bias path. A device-side unprepared bias makes conv pull it
            # back to host and reprocess anyway (with a warning); doing it explicitly is
            # the same tiny [1,1,1,C] transfer without the failed device-prepare attempt.
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
        if cond_bias is None:
            self.tt_bias = bias
        # Keep TILE: the conv already emits TILE/interleaved-DRAM, and the whole
        # vocoder conv chain (+ its eltwise ops) consumes TILE, so we skip the
        # per-conv untilize->ROW_MAJOR round-trip.
        return _interleaved(out, [batch_size, out_length, self.out_channels], row_major=False)


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
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = ttnn.reshape(z, [batch_size, input_length * self.stride, self.out_channels])
        return ttnn.to_layout(z, ttnn.TILE_LAYOUT)


class TtConv2d(LightweightModule):
    """2D convolution over a channels-last ``[N, H, W, C]`` device tensor.

    ``stride``/``padding`` follow PyTorch semantics (symmetric). Used by the
    speaker-encoder SE-ResNet (all 3x3 / 1x1 convs).
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_channels, in_channels // groups, kh, kw]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int = 1,
        padding: int = 1,
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

        self.conv_config = ttnn.Conv2dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, height, width, _ = x.shape
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
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        self.tt_bias = bias
        return _interleaved(out, [batch_size, out_h, out_w, self.out_channels], row_major=True)
