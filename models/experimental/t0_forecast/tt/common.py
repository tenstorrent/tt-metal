# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for t0: precision policies, uploads, per-token linears and norms.

Per-token activations stay flat, ``(1, 1, rows * S_pad, d)`` with ``S_pad`` the patch count padded to a whole
tile per row, so every linear folds the whole batch into M; they are viewed as ``(rows, S_pad, d)`` around
attention, and the two views share one buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TILE = 32


def pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class TtT0Precision:
    """``accuracy`` (default): bf16 weights, HiFi2 matmuls; ``performance``: bfloat8_b weights, LoFi.

    Both accumulate matmuls in fp32 (fp32 dest) and run norm statistics at HiFi4 with fp32 dest:
    with bf16 accumulation one smooth series reached only pcc_normalized 0.990 (0.9995 with fp32), and fp32
    accumulation was also faster at the paper shape (1.056 vs 1.166 s device; jobs 395, 398).
    Per-role exception in both: the patch encoder (``embed``) runs at HiFi4. At HiFi2 the far read-out patches
    of the long-context shape drift (pcc_normalized 0.99870, last patch 0.9966); HiFi4 on ``embed`` alone gives
    0.99987, and HiFi4 on any other role changes nothing (job 417). The patch encoder is ~0.7 % of the forward.

    Fidelity and accumulation per role (accuracy / performance; every matmul and SDPA has fp32 dest accumulation):
        embed (patch encoder)            HiFi4 / HiFi4, fp32 acc
        qkv, wv, wo, ff1, ff2            HiFi2 / LoFi,  fp32 acc
        decoder                          HiFi2 / LoFi,  fp32 acc, fp32 output
        SDPA (time and group)            HiFi2 / LoFi,  fp32 acc
        RMSNorm, QK-norm, out_norm       HiFi4, fp32 acc
        RoPE rotation, head cumsum       HiFi4, fp32 acc (exact 0 / +-1 matrices); softplus on fp32
    Measured: bf16 against fp32 accumulation on every matmul, jobs 395 and 398 (above)."""

    name: str = "accuracy"
    bf8_weights: bool = False
    lofi: bool = False
    hifi4: bool = False  # every matmul at HiFi4 (sweep variant)
    fp32_acc: bool = True  # fp32 dest accumulation in matmuls
    hifi4_roles: frozenset = frozenset()  # matmul roles at HiFi4: embed, qkv, wv, wo, ff1, ff2, decoder
    fp32_residual: bool = False  # keep the residual stream in fp32 (norms emit bf16 for the matmuls)
    bf16_roles: frozenset = frozenset()  # matmul roles whose weights stay bf16 when ``bf8_weights`` (sweep variant)

    @classmethod
    def accuracy(cls) -> "TtT0Precision":
        return cls(hifi4_roles=frozenset({"embed"}))

    @classmethod
    def performance(cls) -> "TtT0Precision":
        return cls(name="performance", bf8_weights=True, lofi=True, hifi4_roles=frozenset({"embed"}))

    def replaced(self, **changes) -> "TtT0Precision":
        from dataclasses import replace

        return replace(self, **changes)

    @classmethod
    def named(cls, name: str) -> "TtT0Precision":
        variants = {
            "accuracy_bf16acc": cls(name="accuracy_bf16acc", fp32_acc=False),
            "accuracy_hifi4": cls(name="accuracy_hifi4", hifi4=True),
            "performance_bf16acc": cls(name="performance_bf16acc", bf8_weights=True, lofi=True, fp32_acc=False),
        }
        if name in variants:
            return variants[name]
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def weight_dtype(self, role: str | None = None):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights and role not in self.bf16_roles else ttnn.bfloat16

    def compute_kernel_config(self, *, math_fidelity=None, fp32_acc: bool = False, role: str | None = None):
        import ttnn

        if math_fidelity is None:
            math_fidelity = ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2
            if self.hifi4 or role in self.hifi4_roles:
                math_fidelity = ttnn.MathFidelity.HiFi4
            fp32_acc = fp32_acc or self.fp32_acc
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )

    def norm_kernel_config(self):
        import ttnn

        return self.compute_kernel_config(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_acc=True)

    def exact_kernel_config(self):
        """HiFi4 with fp32 accumulation, for the +-1 / 0-1 matrices (RoPE rotation, head cumsum)."""
        import ttnn

        return self.compute_kernel_config(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_acc=True)


def upload(device, t: torch.Tensor, *, dtype=None, layout=None):
    """Host tensor -> device DRAM (bf16 tiles by default)."""
    import ttnn

    dtype = ttnn.bfloat16 if dtype is None else dtype
    # torch rounds fp32 -> bf16 the same way from_torch does, and much faster.
    host = t.detach().to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32).contiguous()
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_linear(device, weight: torch.Tensor, dtype):
    """``nn.Linear`` weight ``(out, in)`` -> device ``(in, out)``, the layout ``ttnn.linear`` reads."""
    return upload(device, weight.t(), dtype=dtype)


def upload_bias(device, bias: torch.Tensor):
    return upload(device, bias.reshape(1, -1))


def upload_norm(device, gain: torch.Tensor):
    """RMSNorm gain ``(n,)`` -> row-major ``(1, 1, n / 32, 32)`` bf16, the layout ``ttnn.rms_norm`` reads."""
    import ttnn

    return upload(device, gain.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)


def linear(x, weight, precision: TtT0Precision, *, role: str, bias=None, activation=None, dtype=None):
    import ttnn

    return ttnn.linear(
        x,
        weight,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(role=role),
    )


def linear_minimal(x, weight, precision: TtT0Precision, *, role: str, blocks: tuple, bias=None, activation=None):
    """``x @ weight (+ bias)`` on ``ttnn.experimental.minimal_matmul`` with ``blocks`` = (M, K, N, subblock_h,
    subblock_w) caps (``program_configs.minimal_config``); ``x`` is flat ``(1, 1, M, K)`` in DRAM."""
    import ttnn

    from models.experimental.t0_forecast.tt import program_configs as pc

    grid = x.device().compute_with_storage_grid_size()
    m_tiles = x.padded_shape[-2] // TILE
    k_tiles, n_tiles = weight.padded_shape[-2] // TILE, weight.padded_shape[-1] // TILE
    return ttnn.experimental.minimal_matmul(
        x,
        weight,
        bias_tensor=bias,
        fused_activation=pc.fused_activation(activation),
        config=pc.minimal_config((grid.x, grid.y), m_tiles, k_tiles, n_tiles, blocks),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(role=role),
    )


def linear_residual(x, weight, residual, ones, precision: TtT0Precision, *, role: str, blocks: tuple, bias=None):
    """``residual + x @ weight (+ bias)`` with the add in the ``minimal_matmul`` epilogue
    (``dit_minimal_matmul_addcmul_fused``, ``ones`` the ``(1, N)`` multiplier) when the weight dtype matches the
    residual's; otherwise matmul + ADD. ``residual`` is borrowed."""
    import ttnn

    from models.experimental.t0_forecast.tt import program_configs as pc

    if weight.dtype != residual.dtype:
        out = linear(x, weight, precision, role=role, bias=bias)
        y = add(residual, out)
        ttnn.deallocate(out)
        return y
    grid = x.device().compute_with_storage_grid_size()
    m_tiles = x.padded_shape[-2] // TILE
    k_tiles, n_tiles = weight.padded_shape[-2] // TILE, weight.padded_shape[-1] // TILE
    return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
        x,
        weight,
        1.0,
        residual,
        ones,
        bias_tensor=bias,
        config=pc.minimal_config((grid.x, grid.y), m_tiles, k_tiles, n_tiles, blocks),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(role=role),
    )


def upload_ones(device, n: int):
    """``(1, 1, 1, n)`` bf16 ones: the multiplier of ``linear_residual``."""
    return upload_bias(device, torch.ones(n))


def rms_norm(x, gain, eps: float, precision: TtT0Precision):
    """RMSNorm with a learned gain; always returns bf16 (an fp32 residual stream is normed into bf16)."""
    import ttnn

    out = ttnn.rms_norm(
        x,
        epsilon=eps,
        weight=gain,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.norm_kernel_config(),
    )
    if out.dtype != ttnn.bfloat16:
        narrow = ttnn.typecast(out, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        out = narrow
    return out


def add(a, b):
    import ttnn

    return ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
