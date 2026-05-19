# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around ``ttnn.conv1d`` / ``ttnn.conv_transpose2d`` for the VAE.

All convolutions in the Oobleck decoder share the same input layout convention:

    ``[batch, time, channels]`` (row-major)

Conv1d weights are stored in PyTorch's ``[out, in, k]`` layout; ConvTranspose1d
weights are stored in PyTorch's ``[in, out, k]`` layout. Both get fed through
the standard TTNN ``prepare_conv_*`` helpers which expect rank-4 layouts.

For conv_transpose, we use ``ttnn.conv_transpose2d`` with ``H=1`` and treat the
time axis as ``W``. PyTorch's ``ConvTranspose1d`` and TTNN's ``conv_transpose2d``
implement the same formula, so the time-axis output length matches.
"""

from __future__ import annotations

import numpy as np

from .._ttnn import get_ttnn
from ..math_perf_env import ace_step_linear_l1_memory_config, ace_step_reshape_kwargs


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


def _to_float32_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        pass
    return np.asarray(arr, dtype=np.float32)


class TtConv1d:
    """Owns a ``ttnn.conv1d`` op + its prepared weights/bias for one Conv1d layer."""

    def __init__(
        self,
        *,
        weight_host,
        bias_host,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        device,
        activation_dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

        self.activation_dtype = activation_dtype or getattr(ttnn, "bfloat16", None)
        self.weights_dtype = weights_dtype or getattr(ttnn, "bfloat16", None)
        if self.activation_dtype is None or self.weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16; supply activation_dtype/weights_dtype")
        if math_fidelity is None:
            math_fidelity = ttnn.MathFidelity.HiFi2

        self._vae_conv_perf = True
        # Only kernel_size==1 convs (residual 1×1 projections) safely fit in L1 on Blackhole.
        # Any kernel_size > 1 conv overflows the static CB region regardless of dilation:
        # L1 buffer @ 139072 vs CB end @ 139328 (256-byte clash at act_block_h_override=32).
        # Larger kernels need bigger CBs that push the activation buffer into the CB region.
        self._l1_mem = ace_step_linear_l1_memory_config(ttnn) if (self.kernel_size == 1) else None

        w = _to_float32_numpy(weight_host)
        if w.ndim != 3 or w.shape[0] != self.out_channels or w.shape[1] != self.in_channels:
            raise ValueError(
                f"Unexpected Conv1d weight shape {w.shape}; "
                f"expected ({self.out_channels}, {self.in_channels}, {self.kernel_size})"
            )
        if int(w.shape[2]) != self.kernel_size:
            raise ValueError(f"Conv1d kernel mismatch: got {w.shape[2]}, expected {self.kernel_size}")

        self._weight_host_tt = ttnn.as_tensor(w, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if bias_host is None:
            self._bias_host_tt = None
            self._has_bias = False
        else:
            b = _to_float32_numpy(bias_host).reshape(1, 1, 1, -1)
            self._bias_host_tt = ttnn.as_tensor(b, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._has_bias = True

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=None,
            deallocate_activation=bool(self._vae_conv_perf),
            # act_block_h_override=32 forces the minimum activation block height (must be a multiple of 32).
            # Smaller value → smaller per-core circular buffers → fits within Blackhole L1 (1.5 MB).
            # Default (0) lets TTNN auto-pick, which chooses a large block that overflows L1 for
            # the high-channel-count layers in the Oobleck decoder (up to 2048 ch).
            act_block_h_override=32,
            # Store persistent conv config tensors in DRAM instead of L1_SMALL to free up the small arena.
            config_tensors_in_dram=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._packed_for: tuple[int, int] | None = None
        self._weight_dev = None
        self._bias_dev = None

    def _input_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _output_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _maybe_l1(self, x):
        # Always move to the target memory so L1 Snake outputs are explicitly placed in DRAM
        # for k>1 convs. Leaving the tensor as-is is fragile: L1 allocator fragmentation can
        # place a Snake output inside the static CB region of the conv program, causing a
        # "L1 buffer clashes with CB" crash (e.g. L1 buffer @ 133120 vs CB end @ 139328).
        target = self._l1_mem if self._l1_mem is not None else self.ttnn.DRAM_MEMORY_CONFIG
        return self.ttnn.to_memory_config(x, target)

    def _ensure_packed(self, batch_size: int, input_length: int) -> None:
        ttnn = self.ttnn
        if self._packed_for == (batch_size, input_length) and self._weight_dev is not None:
            return
        input_mem = self._input_memory_config()
        self._weight_dev = ttnn.prepare_conv_weights(
            weight_tensor=self._weight_host_tt,
            input_memory_config=input_mem,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            has_bias=self._has_bias,
            groups=1,
            device=self.device,
            input_dtype=self.activation_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        if self._has_bias:
            self._bias_dev = ttnn.prepare_conv_bias(
                bias_tensor=self._bias_host_tt,
                input_memory_config=input_mem,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_height=1,
                input_width=input_length,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                dilation=(1, self.dilation),
                device=self.device,
                input_dtype=self.activation_dtype,
                groups=1,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
            )
        self._packed_for = (batch_size, input_length)

    def __call__(self, x):
        """Run conv1d on a ``[B, T, C]`` row-major tensor.

        Returns a ``[B, T_out, out_channels]`` row-major tensor.
        """
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        if len(x.shape) != 3:
            raise ValueError(f"TtConv1d expects rank-3 [B,T,C], got {x.shape}")
        b = int(x.shape[0])
        t = int(x.shape[1])
        c = int(x.shape[-1])
        if c != self.in_channels:
            raise ValueError(f"Conv1d input channels mismatch: got {c}, expected {self.in_channels}")

        x = self._maybe_l1(x)
        self._ensure_packed(b, t)
        ret = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self._weight_dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self._bias_dev,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=b,
            input_length=t,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.activation_dtype,
            memory_config=self._output_memory_config(),
        )
        out, out_length = ret
        out = ttnn.squeeze(out, 0)
        out = ttnn.reshape(out, (b, out_length, self.out_channels), **_sr)
        # L1 conv returns TILE; normalize to ROW_MAJOR so time/channel dims match PyTorch semantics
        # and residual / slice logic sees the true logical length.
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


class TtConvTranspose1d:
    """Owns a ``ttnn.conv_transpose2d`` op for one ConvTranspose1d layer.

    PyTorch ``ConvTranspose1d`` weight layout is ``[in, out, k]``; TTNN's
    ``conv_transpose2d`` expects ``[in, out, kH, kW]`` (matching PyTorch's
    ``ConvTranspose2d``), so we add an ``H=1`` axis on host.
    """

    def __init__(
        self,
        *,
        weight_host,
        bias_host,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        device,
        activation_dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.activation_dtype = activation_dtype or getattr(ttnn, "bfloat16", None)
        self.weights_dtype = weights_dtype or getattr(ttnn, "bfloat16", None)
        if self.activation_dtype is None or self.weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16; supply activation_dtype/weights_dtype")
        if math_fidelity is None:
            math_fidelity = ttnn.MathFidelity.HiFi2

        self._vae_conv_perf = True
        # conv_transpose has no dilation → smaller CB requirement → L1 interleaved is safe on Blackhole.
        self._l1_mem = ace_step_linear_l1_memory_config(ttnn)

        w = _to_float32_numpy(weight_host)
        if w.ndim != 3 or w.shape[0] != self.in_channels or w.shape[1] != self.out_channels:
            raise ValueError(
                f"Unexpected ConvTranspose1d weight shape {w.shape}; "
                f"expected ({self.in_channels}, {self.out_channels}, {self.kernel_size})"
            )
        if int(w.shape[2]) != self.kernel_size:
            raise ValueError(f"ConvTranspose1d kernel mismatch: got {w.shape[2]}, expected {self.kernel_size}")

        w4 = w.reshape(self.in_channels, self.out_channels, 1, self.kernel_size)
        self._weight_host_tt = ttnn.as_tensor(w4, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if bias_host is None:
            self._bias_host_tt = None
            self._has_bias = False
        else:
            b = _to_float32_numpy(bias_host).reshape(1, 1, 1, -1)
            self._bias_host_tt = ttnn.as_tensor(b, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._has_bias = True

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=None,
            deallocate_activation=bool(self._vae_conv_perf),
            act_block_h_override=32,
            config_tensors_in_dram=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._weight_dev = self._weight_host_tt
        self._bias_dev = self._bias_host_tt
        self._uploaded = False

    def _output_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _maybe_l1(self, x):
        # Always move to the target memory — same rationale as TtConv1d._maybe_l1.
        target = self._l1_mem if self._l1_mem is not None else self.ttnn.DRAM_MEMORY_CONFIG
        return self.ttnn.to_memory_config(x, target)

    def __call__(self, x):
        """Run conv_transpose2d on a ``[B, T, C]`` row-major tensor.

        Returns ``[B, T_out, out_channels]`` row-major with
        ``T_out = (T-1)*stride - 2*padding + kernel`` (PyTorch formula).
        """
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtConvTranspose1d expects rank-3 [B,T,C], got {x.shape}")
        b = int(x.shape[0])
        t = int(x.shape[1])
        c = int(x.shape[-1])
        if c != self.in_channels:
            raise ValueError(f"ConvT1d input channels mismatch: got {c}, expected {self.in_channels}")

        x = self._maybe_l1(x)
        # TTNN conv_transpose2d expects [B, H, W, C] NHWC; map T -> W with H=1.
        x4 = ttnn.unsqueeze(x, 1)  # [B, 1, T, C]

        out, [self._weight_dev, self._bias_dev] = ttnn.conv_transpose2d(
            input_tensor=x4,
            weight_tensor=self._weight_dev,
            bias_tensor=self._bias_dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=b,
            input_height=1,
            input_width=t,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_weights_and_bias=True,
            mirror_kernel=True,
            dtype=self.activation_dtype,
            memory_config=self._output_memory_config(),
        )
        # conv_transpose2d returns NHWC [B, 1, T_out, out_channels] (rank-4)
        out = ttnn.squeeze(out, 1)
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
