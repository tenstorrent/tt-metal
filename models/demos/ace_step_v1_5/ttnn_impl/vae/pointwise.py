# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""1×1 Conv1d as ``ttnn.linear`` (LoFi + ``bfloat8_b`` weights + L1 activations)."""

from __future__ import annotations

from .._ttnn import get_ttnn
from ..math_perf_env import (
    ace_step_dense_linear_program_config,
    ace_step_init_vae_conv_compute_kernel_config,
    ace_step_linear_weight_dtype,
    ace_step_vae_activation_memory_config,
)
from .conv1d import _to_float32_numpy


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


class TtVaePointwise1d:
    """Pointwise (kernel 1) channel mixing via ``ttnn.linear`` instead of ``ttnn.conv1d``."""

    def __init__(
        self,
        *,
        weight_host,
        bias_host,
        in_channels: int,
        out_channels: int,
        device,
        activation_dtype=None,
        weights_dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.activation_dtype = activation_dtype or getattr(ttnn, "bfloat16", None)
        if self.activation_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16; supply activation_dtype")

        w = _to_float32_numpy(weight_host)
        if w.ndim == 3 and int(w.shape[2]) == 1:
            w = w.reshape(self.out_channels, self.in_channels)
        if w.shape != (self.out_channels, self.in_channels):
            raise ValueError(f"Pointwise weight expected ({self.out_channels}, {self.in_channels}), got {w.shape}")

        self._l1_mc = ace_step_vae_activation_memory_config(ttnn)
        self._linear_ck = ace_step_init_vae_conv_compute_kernel_config(device)
        w_dtype = ace_step_linear_weight_dtype(ttnn, weights_dtype or self.activation_dtype)

        w_mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        self.weight_tt = ttnn.as_tensor(
            w,
            device=device,
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem,
        )
        if bias_host is None:
            self.bias_tt = None
        else:
            b = _to_float32_numpy(bias_host).reshape(1, 1, 1, -1)
            self.bias_tt = ttnn.as_tensor(
                b,
                device=device,
                dtype=w_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=w_mem,
            )

    def _linear_kwargs(self, *, batch_size: int, seq_len: int) -> dict:
        kw: dict = {}
        if self._linear_ck is not None:
            kw["compute_kernel_config"] = self._linear_ck
        pc = ace_step_dense_linear_program_config(
            self.device,
            seq_len=int(seq_len),
            in_dim=self.in_channels,
            out_dim=self.out_channels,
            batch_size=int(batch_size),
        )
        if pc is not None:
            kw["program_config"] = pc
        if self._l1_mc is not None:
            kw["memory_config"] = self._l1_mc
        return kw

    def __call__(self, x):
        """``[B, T, C_in]`` row-major → ``[B, T, C_out]`` row-major DRAM."""
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtVaePointwise1d expects rank-3 [B,T,C], got {x.shape}")
        b = int(x.shape[0])
        t = int(x.shape[1])
        c = int(x.shape[-1])
        if c != self.in_channels:
            raise ValueError(f"Pointwise input channels mismatch: got {c}, expected {self.in_channels}")

        l1_mc = self._l1_mc
        dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        _til_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, **_til_kw)

        out = ttnn.linear(
            x,
            self.weight_tt,
            bias=self.bias_tt,
            transpose_b=True,
            **self._linear_kwargs(batch_size=b, seq_len=t),
        )
        if len(out.shape) == 4:
            out = ttnn.squeeze(out, 1)
        _rm_kw = {"memory_config": dram_mc} if dram_mc is not None else {}
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, **_rm_kw)
