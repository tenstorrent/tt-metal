# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Toto 2.0: precision policies, uploads, per-token linears and norms.

Per-token activations are flat, ``(1, 1, tokens, D)``, so every linear folds the whole batch into M. Rows of
``S_pad`` tokens (a whole number of tiles) are viewed as ``(R, 1, S_pad, D)`` around time attention; both views
share one buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.toto2_forecast.tt.preprocessing import TILE, pad_to_tile  # noqa: F401


@dataclass(frozen=True)
class Toto2Precision:
    """``accuracy``: bf16 weights, HiFi2 matmuls. ``performance``: bfloat8_b weights, LoFi matmuls, with one
    per-role exception: SDPA stays at HiFi2 (a negligible share of the FLOPs; LoFi SDPA cost 22m at context 8192
    0.9905 -> 0.9947 normalized PCC). Both: bf16 activations, RMSNorm at HiFi4 with fp32 accumulation, RoPE at
    HiFi4. ``fp32_acc`` turns on fp32 dest accumulation in the matmuls and SDPA."""

    name: str = "accuracy"
    bf8_weights: bool = False
    lofi: bool = False
    sdpa_lofi: bool = False
    fp32_acc: bool = False

    @classmethod
    def accuracy(cls) -> "Toto2Precision":
        return cls()

    @classmethod
    def performance(cls) -> "Toto2Precision":
        return cls(name="performance", bf8_weights=True, lofi=True)

    @classmethod
    def named(cls, name: str) -> "Toto2Precision":
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def weight_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16

    def _config(self, math_fidelity, fp32_dest_acc_en: bool = False):
        import ttnn

        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def matmul_config(self):
        import ttnn

        fidelity = ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2
        return self._config(fidelity, fp32_dest_acc_en=self.fp32_acc)

    def sdpa_config(self):
        import ttnn

        fidelity = ttnn.MathFidelity.LoFi if self.sdpa_lofi else ttnn.MathFidelity.HiFi2
        return self._config(fidelity, fp32_dest_acc_en=self.fp32_acc)

    def norm_config(self):
        import ttnn

        return self._config(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def rope_config(self):
        import ttnn

        # Below HiFi4 the +-1 rotation matmul drops input mantissa bits.
        return self._config(ttnn.MathFidelity.HiFi4)


def upload(device, t: torch.Tensor, *, dtype=None, layout=None):
    """Host tensor -> device DRAM (bf16 tiles by default)."""
    import ttnn

    dtype = ttnn.bfloat16 if dtype is None else dtype
    host_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(
        t.detach().to(host_dtype).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_linear(device, weight: torch.Tensor, dtype, *, pad_in: int | None = None, pad_out: int | None = None):
    """``nn.Linear`` weight ``(out, in)`` -> device ``(in, out)``, zero-padded to ``pad_in`` x ``pad_out``."""
    w = weight.t()
    k, n = w.shape
    w = torch.nn.functional.pad(w, (0, (pad_out or n) - n, 0, (pad_in or k) - k))
    return upload(device, w, dtype=dtype)


def upload_bias(device, bias: torch.Tensor | None, *, pad_out: int | None = None):
    if bias is None:
        return None
    b = torch.nn.functional.pad(bias, (0, (pad_out or bias.shape[0]) - bias.shape[0]))
    return upload(device, b.reshape(1, -1))


def linear(x, weight, precision: Toto2Precision, *, bias=None, activation=None):
    """``x @ weight (+ bias)``, then ``activation`` ("silu")."""
    import ttnn

    return ttnn.linear(
        x,
        weight,
        bias=bias,
        activation=activation,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.matmul_config(),
    )


def rms_norm(x, eps: float, precision: Toto2Precision):
    """Weightless RMSNorm, statistics accumulated in fp32."""
    import ttnn

    return ttnn.rms_norm(
        x, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=precision.norm_config()
    )


def scaled_residual(f, x, beta: float):
    """``f + beta * x``; consumes ``f``, borrows ``x``."""
    import ttnn

    scaled = ttnn.multiply(x, beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(f, scaled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(scaled)
    ttnn.deallocate(f)
    return out


def take_slots(x, rows: int, seq: int, start: int, count: int):
    """Flat ``(1, 1, rows * seq, D)`` -> ``(1, 1, rows * count, D)``: token slots ``[start, start + count)`` of every
    row; ``x`` is borrowed."""
    import ttnn

    d = x.shape[-1]
    picked = ttnn.slice(ttnn.reshape(x, (rows, seq, d)), (0, start, 0), (rows, start + count, d))
    return ttnn.reshape(picked, (1, 1, rows * count, d))
