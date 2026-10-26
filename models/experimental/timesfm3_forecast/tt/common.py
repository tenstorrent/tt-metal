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


WEIGHT_ROLES = ("tokenizer", "seq_qkv", "seq_wo", "var_qkv", "var_wo", "w_ov", "ffn", "head")


@dataclass(frozen=True)
class TtPrecision:
    """Per-role precision policy. Activations are bf16 everywhere; norms and RoPE run HiFi4 with fp32 dest
    accumulation in every policy.

    - ``matmul_fidelity`` / ``matmul_fp32_acc``: every linear (tokenizer, projections, FFN, head);
    - ``sdpa_fidelity`` / ``sdpa_fp32_acc``: both attentions;
    - ``bf8_weights``: bfloat8_b for every weight role except those in ``bf16_roles`` (``WEIGHT_ROLES``);
    - ``embed_fidelity``: an override for the tokenizer matmuls (None = as ``matmul_*``).

    ``accuracy`` and ``performance`` are the contract's named policies; variants come from
    ``dataclasses.replace``.
    """

    name: str = "accuracy"
    bf8_weights: bool = False
    bf16_roles: tuple = ()
    matmul_fidelity: str = "hifi2"
    matmul_fp32_acc: bool = False
    sdpa_fidelity: str = "hifi2"
    sdpa_fp32_acc: bool = False
    embed_fidelity: str | None = None  # None | "hifi4" | "hifi4_fp32"
    bf16_role_fidelity: str | None = None  # matmul fidelity of the ``bf16_roles`` (None: as ``matmul_fidelity``)

    @classmethod
    def accuracy(cls) -> "TtPrecision":
        """bf16 weights, HiFi2 with fp32 dest accumulation on every matmul and both SDPAs (G1 re-raise:
        ``doc/optimize/iter_00/accumulation_sweep.md``, variant A1; without the fp32 accumulation the forecast
        band of narrow intermittent rows shrank to 0.40 of the reference)."""
        return cls(matmul_fp32_acc=True, sdpa_fp32_acc=True)

    @classmethod
    def performance(cls) -> "TtPrecision":
        """bfloat8_b weights on the QKV projections only, HiFi2 with fp32 dest accumulation everywhere (R1-0, D3
        condition 1: variant P6b of ``doc/optimize/iter_00/r1_0/accumulation_sweep.md``, the cheapest variant that
        passes every performance floor; bf8 on any other role, or LoFi, collapses the intermittent rows)."""
        return cls(
            name="performance",
            bf8_weights=True,
            bf16_roles=tuple(r for r in WEIGHT_ROLES if r not in ("seq_qkv", "var_qkv")),
            matmul_fidelity="hifi2",
            matmul_fp32_acc=True,
            sdpa_fidelity="hifi2",
            sdpa_fp32_acc=True,
        )

    @classmethod
    def by_name(cls, name: str) -> "TtPrecision":
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def describe(self) -> dict:
        """The per-role settings, as recorded in bench records and the GATE."""
        return {
            "weights": "bfloat8_b" if self.bf8_weights else "bf16",
            "bf16_roles": list(self.bf16_roles),
            "matmul": f"{self.matmul_fidelity}, fp32_acc={self.matmul_fp32_acc}",
            "bf16_role_matmul": self.bf16_role_fidelity or "as matmul",
            "sdpa": f"{self.sdpa_fidelity}, fp32_acc={self.sdpa_fp32_acc}",
            "embed": self.embed_fidelity or "as matmul",
            "norms_rope": "hifi4, fp32_acc=True",
        }

    def weight_dtype(self, role: str | None = None):
        import ttnn

        if role is not None and role not in WEIGHT_ROLES:
            raise ValueError(f"unknown weight role {role}")
        return ttnn.bfloat8_b if self.bf8_weights and role not in self.bf16_roles else ttnn.bfloat16

    @staticmethod
    def _fidelity(name: str):
        import ttnn

        return {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[
            name
        ]

    @staticmethod
    def _config(math_fidelity, fp32_dest_acc_en: bool):
        import ttnn

        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def matmul_config(self, role: str | None = None):
        fidelity = self.matmul_fidelity
        if role in self.bf16_roles and self.bf16_role_fidelity is not None:
            fidelity = self.bf16_role_fidelity
        return self._config(self._fidelity(fidelity), self.matmul_fp32_acc)

    def embed_config(self):
        import ttnn

        if self.embed_fidelity is None:
            return self.matmul_config()
        return self._config(ttnn.MathFidelity.HiFi4, self.embed_fidelity == "hifi4_fp32")

    def sdpa_config(self):
        return self._config(self._fidelity(self.sdpa_fidelity), self.sdpa_fp32_acc)

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


# R1-4 (``doc/optimize/iter_00/r1_4/matmul_sweep.json``): the fastest config per weight kind, at M = 65536 and 16384.
# R2-3 (``doc/optimize/iter_01/r2_3/qkv_sweep.json``): QKV re-tuned, bf16 4.72 vs 5.07 ms, bf8 3.95 vs 4.18 ms.
MATMUL_MINIMAL_QKV = ("mm", (16, 4, 8, 4, 1))  # QKV (N = 4608), bf16 and bf8 weights
MATMUL_MINIMAL = ("mm", (16, 8, 4, 4, 1))  # every other per-token linear (K, N <= 1536)


def _matmul_choice(weight, k_tiles: int, n_tiles: int):
    if n_tiles > 48:
        return MATMUL_MINIMAL_QKV
    return MATMUL_MINIMAL


def linear(x, weight, precision: TtPrecision, *, bias=None, activation=None, compute_kernel_config=None, tuned=True):
    """``x @ weight``; per-token linears with M >= 512 rows take the R1-4 program config (``tuned``)."""
    import ttnn

    from models.experimental.timesfm3_forecast.tt import program_configs as pc

    kernel = precision.matmul_config() if compute_kernel_config is None else compute_kernel_config
    shape = x.padded_shape
    m = 1
    for i in range(len(shape) - 1):
        m *= shape[i]
    if tuned and bias is None and m >= 512:
        m_tiles, k_tiles, n_tiles = m // TILE, x.padded_shape[-1] // TILE, weight.padded_shape[-1] // TILE
        kind, arg = _matmul_choice(weight, k_tiles, n_tiles)
        act = pc.fused_activation(activation)
        if kind == "mm":
            grid = x.device().compute_with_storage_grid_size()
            return ttnn.experimental.minimal_matmul(
                x,
                weight,
                fused_activation=act,
                config=pc.minimal_config((grid.x, grid.y), m_tiles, k_tiles, n_tiles, arg),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kernel,
            )
        cfg = pc.matmul_2d_config((11, 10), m_tiles, k_tiles, n_tiles, fused_activation=act, out_block_cap=arg)
        return ttnn.linear(
            x, weight, program_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel
        )
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        activation=activation,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel,
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
