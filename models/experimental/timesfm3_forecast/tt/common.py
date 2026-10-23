# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for TimesFM-3: precision policies, uploads, linears and norms.

Per-token activations are flat, ``(1, 1, rows * padded_seq, D)``, so every linear folds the whole batch
into M; ``padded_seq`` is tile-aligned, so the ``(rows, padded_seq, D)`` view around attention is free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TILE = 32


def pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class TtPrecision:
    """``accuracy``: bf16 weights, HiFi2 matmuls. ``performance``: bfloat8_b weights, LoFi matmuls.
    Both keep bf16 activations, and HiFi4 with fp32 accumulation for the norms and RoPE."""

    name: str = "accuracy"
    bf8_weights: bool = False
    lofi: bool = False
    # Tokenizer (patch encoder) matmuls: None = as the other matmuls, else "hifi4" or "hifi4_fp32"
    # (PLAN section 9: check the patch encoder alone at HiFi4 on the long context).
    embed_fidelity: str | None = None

    @classmethod
    def accuracy(cls) -> "TtPrecision":
        return cls()

    @classmethod
    def performance(cls) -> "TtPrecision":
        return cls(name="performance", bf8_weights=True, lofi=True)

    @classmethod
    def by_name(cls, name: str) -> "TtPrecision":
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def weight_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16

    @staticmethod
    def _config(math_fidelity, fp32_dest_acc_en: bool):
        import ttnn

        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def matmul_config(self):
        import ttnn

        return self._config(ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2, False)

    def embed_config(self):
        import ttnn

        if self.embed_fidelity is None:
            return self.matmul_config()
        return self._config(ttnn.MathFidelity.HiFi4, self.embed_fidelity == "hifi4_fp32")

    def sdpa_config(self):
        import ttnn

        return self._config(ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2, False)

    def norm_config(self):
        import ttnn

        return self._config(ttnn.MathFidelity.HiFi4, True)


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


def upload_linear(device, weight: torch.Tensor, dtype):
    """``nn.Linear`` weight ``(out, in)`` -> device ``(in, out)``, the layout ``ttnn.linear`` reads."""
    return upload(device, weight.t(), dtype=dtype)


def upload_vector(device, v: torch.Tensor):
    """A norm gain or bias ``(n,)`` -> device ``(1, n)`` bf16 tiles."""
    return upload(device, v.reshape(1, -1))


def linear(x, weight, precision: TtPrecision, *, bias=None, activation=None, compute_kernel_config=None):
    import ttnn

    return ttnn.linear(
        x,
        weight,
        bias=bias,
        activation=activation,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.matmul_config() if compute_kernel_config is None else compute_kernel_config,
    )


def rms_norm(x, eps: float, precision: TtPrecision, weight=None):
    import ttnn

    return ttnn.rms_norm(
        x,
        epsilon=eps,
        weight=weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.norm_config(),
    )


def to_rows(x, rows: int, seq: int):
    """Flat ``(1, 1, rows * seq, D)`` -> ``(rows, seq, D)`` (a view: ``seq`` is tile-aligned)."""
    import ttnn

    return ttnn.reshape(x, (rows, seq, x.shape[-1]))


def to_flat(x):
    """``(rows, seq, D)`` or ``(rows, 1, seq, D)`` -> flat ``(1, 1, rows * seq, D)``."""
    import ttnn

    shape = tuple(x.shape)
    rows_seq = 1
    for s in shape[:-1]:
        rows_seq *= s
    return ttnn.reshape(x, (1, 1, rows_seq, shape[-1]))


def take_slots(x, rows: int, seq: int, start: int, count: int):
    """Flat ``(1, 1, rows * seq, D)`` -> ``(1, 1, rows * count, D)``: token slots ``[start, start + count)``
    of every row; ``x`` is borrowed."""
    import ttnn

    d = x.shape[-1]
    picked = ttnn.slice(to_rows(x, rows, seq), (0, start, 0), (rows, start + count, d))
    out = ttnn.reshape(picked, (1, 1, rows * count, d))
    return out
