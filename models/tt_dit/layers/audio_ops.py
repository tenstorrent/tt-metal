# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Audio-flavored conv primitives for LTX-2 audio decode.

The mel-VAE decoder and the vocoder operate on small spectrogram and waveform
tensors that don't justify a native ``Conv1d`` / ``Conv2d`` op. We follow the
WAN-style ``ttnn.experimental.conv3d``-with-degenerate-axes pattern proven in
``models/tt_dit/models/transformers/wan2_2/s2v/auxi_blocks.py:CausalConv1d``:

- ``Conv2dViaConv3d``: ``ttnn.experimental.conv3d`` with kernel ``(1, kh, kw)``
  on an input laid out as ``(B, 1, H, W, C)``. Padding modes match
  ``torch.nn.Conv2d``: "zeros" (internal) or "causal-height"/"causal-width"
  (external asymmetric pad on the causal axis).
"""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger

import ttnn

from ..layers.module import Module, Parameter
from ..parallel.config import ParallelFactor
from ..parallel.manager import CCLManager
from ..utils.conv3d import _ntuple, aligned_channels, get_conv3d_config


def _t_neighbor_pad(
    x_BTC: ttnn.Tensor,
    *,
    pad_left: int,
    pad_right: int,
    parallel_config: ParallelFactor,
    ccl_manager: CCLManager,
    padding_mode: str = "zeros",
) -> ttnn.Tensor:
    """Halo exchange on the T axis (dim 1 in BTC layout).

    Mirrors `LTXCausalConv3d`'s H/W halo using `ccl_manager.neighbor_pad_persistent_buffer`.
    Pad sizes are in samples; boundary chips of the mesh get zero/replicate pad,
    interior chips exchange with neighbors via the configured mesh axis.

    Args:
        x_BTC: (B, T_per_device, C) ROW_MAJOR ttnn tensor, sharded along T on
            ``parallel_config.mesh_axis``.
        pad_left, pad_right: halo widths in T samples.
        parallel_config: ``ParallelFactor(factor, mesh_axis)`` describing the
            T-axis sharding.
        ccl_manager: CCLManager for semaphore + persistent-buffer reuse.
        padding_mode: "zeros" or "replicate".

    Returns:
        (B, T_per_device + pad_left + pad_right, C) ROW_MAJOR ttnn tensor.
    """
    if pad_left == 0 and pad_right == 0:
        return x_BTC
    if parallel_config is None or parallel_config.factor <= 1:
        # Local zero-pad fallback for the unsharded path (kept for symmetry).
        B, T, C = x_BTC.shape
        if pad_left > 0:
            zl = ttnn.zeros(
                (B, pad_left, C), dtype=x_BTC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=x_BTC.device()
            )
            x_BTC = ttnn.concat([zl, x_BTC], dim=1)
        if pad_right > 0:
            zr = ttnn.zeros(
                (B, pad_right, C), dtype=x_BTC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=x_BTC.device()
            )
            x_BTC = ttnn.concat([x_BTC, zr], dim=1)
        return x_BTC

    sem = ccl_manager.get_np_ping_pong_semaphore(parallel_config.mesh_axis)
    # `neighbor_pad_async` requires `num_links <= product_of_outer_dims_before_sharded_dim`.
    # For (B, T, C) sharded on T (dim=1), outer dim is just B — usually 1.
    outer_dims = 1
    for i in range(1):  # iterate over dims [0..dim-1] = [0]
        outer_dims *= x_BTC.shape[i]
    num_links = max(1, min(outer_dims, ccl_manager.num_links))
    return ccl_manager.neighbor_pad_persistent_buffer(
        x_BTC,
        dims=[1],
        pad_left=[pad_left],
        pad_right=[pad_right],
        padding_mode=padding_mode,
        axes=[parallel_config.mesh_axis],
        neighbor_sems=[sem],
        num_links=[num_links],
    )


class Conv2dViaConv3d(Module):
    """2D conv implemented as ``ttnn.experimental.conv3d`` with kernel ``(1, kh, kw)``.

    Operates on a tensor laid out as ``(B, 1, H, W, C)`` in ROW_MAJOR. Output has
    the same layout. Single-device only (no halo) — the audio decoder runs on
    one chip; the cost of sharding ~MB-class spectrograms across the mesh is
    dominated by CCL overhead.

    Padding modes:
    - ``"zeros"`` — symmetric zero pad handled by conv3d internal padding.
    - ``"causal_height"`` — asymmetric pad on H (entire ``kh-1`` on the front,
      none on the back), zero pad on W. Matches ``CausalConv2d`` with
      ``causality_axis=HEIGHT`` in the LTX-2 audio reference.
    - ``"causal_width"`` — same idea, swapped axes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding_mode: str = "zeros",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if padding_mode not in ("zeros", "causal_height", "causal_width"):
            raise ValueError(f"padding_mode must be zeros/causal_height/causal_width, got {padding_mode!r}")

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        kh, kw = _ntuple(kernel_size, 2)
        sh, sw = _ntuple(stride, 2)
        self.kernel_size = (1, kh, kw)
        self.stride = (1, sh, sw)
        self.padding_mode = padding_mode
        self.pad_h = kh - 1
        self.pad_w = kw - 1
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Internal vs external padding split: for zeros mode the conv3d kernel
        # symmetrically pads on H and W. For causal modes we pad externally on
        # the causal axis and pass internal pad=0 on that axis (mirrors
        # WanCausalConv3d's temporal-pad split).
        if padding_mode == "zeros":
            self.internal_padding = (0, kh // 2, kw // 2)
        elif padding_mode == "causal_height":
            self.internal_padding = (0, 0, kw // 2)
        else:  # causal_width
            self.internal_padding = (0, kh // 2, 0)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        from models.common.utility_functions import is_blackhole

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() or dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map ``Conv2d.weight (Cout, Cin, kh, kw)`` → conv3d-prepared ``(d, Cout)``."""
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            w = state["weight"]
            # Torch conv2d weight: (Cout, Cin, kh, kw). Reshape to 5D for conv3d:
            # (Cout, Cin, 1, kh, kw) — degenerate T=1 axis matches kernel_size[0]=1.
            assert w.dim() == 4, f"expected 4D Conv2d weight, got {tuple(w.shape)}"
            w_5d = w.unsqueeze(2).contiguous()

            if self.out_channels != self.unpadded_out_channels:
                pad_co = self.out_channels - self.unpadded_out_channels
                w_5d = torch.nn.functional.pad(w_5d, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_co))
                if "bias" in state:
                    state["bias"] = torch.nn.functional.pad(state["bias"], (0, pad_co))

            weight_tt = ttnn.from_torch(w_5d, dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt,
                C_in_block=self.conv_config.C_in_block,
                device=self.mesh_device,
            )
            state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BHWC``: ``(B, H, W, C)`` ROW_MAJOR. Internally pads to 5D and calls conv3d."""
        assert x_BHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {x_BHWC.layout}"
        B, H, W, C = x_BHWC.shape

        # Causal external pad on the chosen axis.
        if self.padding_mode == "causal_height" and self.pad_h > 0:
            # Replicate-pad the front of H by self.pad_h rows. Matches F.pad
            # behaviour for asymmetric causal Conv2d, with zeros on the front.
            B_, H_, W_, C_ = x_BHWC.shape
            pad_tensor_shape = (B_, self.pad_h, W_, C_)
            zero_pad = ttnn.zeros(
                pad_tensor_shape, dtype=x_BHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )
            x_BHWC = ttnn.concat([zero_pad, x_BHWC], dim=1)
        elif self.padding_mode == "causal_width" and self.pad_w > 0:
            B_, H_, W_, C_ = x_BHWC.shape
            pad_tensor_shape = (B_, H_, self.pad_w, C_)
            zero_pad = ttnn.zeros(
                pad_tensor_shape, dtype=x_BHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )
            x_BHWC = ttnn.concat([zero_pad, x_BHWC], dim=2)

        # Add degenerate T=1 axis: (B, H, W, C) → (B, 1, H, W, C).
        x_5d = ttnn.reshape(x_BHWC, (x_BHWC.shape[0], 1, x_BHWC.shape[1], x_BHWC.shape[2], x_BHWC.shape[3]))

        out_5d = ttnn.experimental.conv3d(
            input_tensor=x_5d,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Strip the degenerate T=1 axis: (B, 1, H_out, W_out, C_out) → (B, H_out, W_out, C_out).
        out = ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[2], out_5d.shape[3], out_5d.shape[4]))
        return out


class Conv1dViaConv3d(Module):
    """1D conv implemented as ``ttnn.experimental.conv3d`` with kernel ``(k, 1, 1)``.

    Operates on ``(B, T, 1, 1, C)`` ROW_MAJOR. Mirrors the s2v ``CausalConv1d``
    pattern from ``models/tt_dit/models/transformers/wan2_2/s2v/auxi_blocks.py``.
    Padding modes:
    - ``"zeros"`` — internal symmetric zero pad (handles ``padding="same"``).
    - ``"causal"`` — external front-only zero pad by ``k-1`` along T.

    T-axis sharding: pass ``parallel_config=ParallelFactor(factor, mesh_axis)``
    and a ``ccl_manager`` to fracture inputs/outputs along T across the mesh.
    The conv padding gets moved from internal to external halo exchange via
    ``ccl_manager.neighbor_pad_persistent_buffer`` (same trick LTXCausalConv3d
    uses for H/W in vae_ltx).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        if padding_mode not in ("zeros", "causal"):
            raise ValueError(f"padding_mode must be zeros/causal, got {padding_mode!r}")
        if dilation != 1:
            # We model "padding=same with dilation" by treating effective_kernel = (k-1)*d+1.
            pass

        sharded = parallel_config is not None and parallel_config.factor > 1
        if sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size_t = kernel_size
        self.dilation = dilation
        self.kernel_size = (kernel_size, 1, 1)
        self.stride = (stride, 1, 1)
        self.padding_mode = padding_mode
        self.bias_enabled = bias
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # effective kernel extent for "same" padding with dilation
        eff_k = (kernel_size - 1) * dilation + 1

        if sharded:
            # Sharded: zero internal T padding, do halo exchange in forward().
            self.internal_padding = (0, 0, 0)
            self.external_pad_front = 0  # subsumed by halo
            if padding_mode == "zeros":
                self.halo_pad_left = eff_k // 2
                self.halo_pad_right = eff_k // 2
            else:  # causal
                self.halo_pad_left = eff_k - 1
                self.halo_pad_right = 0
        elif padding_mode == "zeros":
            # Symmetric pad on T inside the kernel.
            self.internal_padding = (eff_k // 2, 0, 0)
            self.external_pad_front = 0
            self.halo_pad_left = 0
            self.halo_pad_right = 0
        else:  # causal — pad k-1 at front externally
            self.internal_padding = (0, 0, 0)
            self.external_pad_front = eff_k - 1
            self.halo_pad_left = 0
            self.halo_pad_right = 0

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        if bias:
            self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        else:
            self.bias = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map torch ``Conv1d.weight (Cout, Cin, k)`` → conv3d-prepared ``(d, Cout)``."""
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            w = state["weight"]
            assert w.dim() == 3, f"expected 3D Conv1d weight, got {tuple(w.shape)}"
            w_5d = w.unsqueeze(-1).unsqueeze(-1).contiguous()  # (Cout, Cin, k, 1, 1)

            if self.out_channels != self.unpadded_out_channels:
                pad_co = self.out_channels - self.unpadded_out_channels
                w_5d = torch.nn.functional.pad(w_5d, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_co))
                if "bias" in state:
                    state["bias"] = torch.nn.functional.pad(state["bias"], (0, pad_co))

            weight_tt = ttnn.from_torch(w_5d, dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt,
                C_in_block=self.conv_config.C_in_block,
                device=self.mesh_device,
            )
            state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])
        if "bias" in state and self.bias is not None:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTC``: ``(B, T, C)`` ROW_MAJOR. Returns ``(B, T_out, C_out)``.

        When ``parallel_config.factor > 1``, ``T`` is the *per-device* time
        extent (fractured on ``parallel_config.mesh_axis``); the halo exchange
        adds boundary context from neighbor chips before the conv.
        """
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT

        if self.parallel_config is not None and self.parallel_config.factor > 1:
            # Halo exchange replaces the local external_pad_front for sharded inputs.
            x_BTC = _t_neighbor_pad(
                x_BTC,
                pad_left=self.halo_pad_left,
                pad_right=self.halo_pad_right,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="zeros",
            )
        elif self.external_pad_front > 0:
            B, T, C = x_BTC.shape
            zero_pad = ttnn.zeros(
                (B, self.external_pad_front, C),
                dtype=x_BTC.get_dtype(),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
            )
            x_BTC = ttnn.concat([zero_pad, x_BTC], dim=1)

        # (B, T_pad, C) → (B, T_pad, 1, 1, C).
        x_5d = ttnn.reshape(x_BTC, (x_BTC.shape[0], x_BTC.shape[1], 1, 1, x_BTC.shape[2]))

        out_5d = ttnn.experimental.conv3d(
            input_tensor=x_5d,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        # (B, T_out, 1, 1, C_out) → (B, T_out, C_out).
        return ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[1], out_5d.shape[4]))


class Snake(Module):
    """``y = x + (1 / (α + ε)) · sin(α · x)²``. α has shape ``(1, 1, C)`` per channel."""

    def __init__(
        self,
        channels: int,
        *,
        alpha_logscale: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale  # If True, learned param is log(α); collapse at load time.
        self.eps = 1e-9
        self.alpha = Parameter(total_shape=[1, 1, channels], device=mesh_device, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "alpha" in state:
            a = state["alpha"]
            if self.alpha_logscale:
                a = torch.exp(a)
            state["alpha"] = a.reshape(1, 1, -1).contiguous()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        a = self.alpha.data
        ax = ttnn.multiply(x_BTC, a)
        s = ttnn.sin(ax)
        s2 = ttnn.multiply(s, s)
        inv = ttnn.reciprocal(ttnn.add(a, self.eps))
        return ttnn.add(x_BTC, ttnn.multiply(s2, inv))


class SnakeBeta(Module):
    """``y = x + (1 / (β + ε)) · sin(α · x)²``. α, β both ``(1, 1, C)`` learned."""

    def __init__(
        self,
        channels: int,
        *,
        alpha_logscale: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.eps = 1e-9
        self.alpha = Parameter(total_shape=[1, 1, channels], device=mesh_device, dtype=dtype)
        self.beta = Parameter(total_shape=[1, 1, channels], device=mesh_device, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("alpha", "beta"):
            if name in state:
                t = state[name]
                if self.alpha_logscale:
                    t = torch.exp(t)
                state[name] = t.reshape(1, 1, -1).contiguous()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        a = self.alpha.data
        b = self.beta.data
        ax = ttnn.multiply(x_BTC, a)
        s = ttnn.sin(ax)
        s2 = ttnn.multiply(s, s)
        inv = ttnn.reciprocal(ttnn.add(b, self.eps))
        return ttnn.add(x_BTC, ttnn.multiply(s2, inv))


def make_kaiser_sinc_kernel(taps: int, half_width: int, ratio: int, beta: float = 6.0) -> torch.Tensor:
    """Build the BigVGAN-v2 anti-aliasing kaiser-sinc kernel of shape ``(1, 1, taps)``."""
    import math

    if taps % 2 == 0:
        raise ValueError(f"taps must be odd, got {taps}")
    # Kaiser-windowed sinc: same recipe as bigvgan_v2/activations/anti_alias.py
    n = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2.0
    sinc = torch.where(
        n == 0, torch.tensor(1.0, dtype=torch.float64), torch.sin(math.pi * n / ratio) / (math.pi * n / ratio)
    )
    # Kaiser window
    alpha = beta
    win = torch.special.modified_bessel_i0(
        alpha * torch.sqrt(1.0 - (2 * n / (taps - 1)) ** 2)
    ) / torch.special.modified_bessel_i0(torch.tensor(alpha, dtype=torch.float64))
    kernel = (sinc * win).float()
    kernel = kernel / kernel.sum() * ratio  # normalize so DC=ratio (upsampler) or 1 (downsampler)
    return kernel.reshape(1, 1, taps)


class Activation1d(Module):
    """Anti-aliased activation: ``UpSample1d → activation → DownSample1d`` (BigVGAN v2).

    Up/down filters are non-learnable kaiser-sinc kernels baked at __init__ time;
    they're materialized as fixed ``Conv1dViaConv3d`` weights (no torch state — we
    construct them and bypass ``_prepare_torch_state``).
    """

    def __init__(
        self,
        channels: int,
        activation: Module,
        *,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.activation = activation
        # NOTE: up_kernel_size / down_kernel_size are BigVGAN defaults; the kaiser-
        # sinc weights are constants — we treat them as Parameters with on-device
        # initialization in ``_init_filters`` rather than going through Module.load.
        # For Stage B we'll bake them at construct time once the weight shape is known.
        # Placeholder: store ratios + sizes; real implementation in Stage B.
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.up_kernel_size = up_kernel_size
        self.down_kernel_size = down_kernel_size
        self.mesh_device = mesh_device
        self.dtype = dtype

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # TODO Stage B: up-sample → activation → down-sample with the kaiser-sinc
        # kernels. Pure passthrough for now to keep imports unblocked.
        return self.activation(x_BTC)
