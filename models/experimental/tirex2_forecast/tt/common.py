# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for TiRex-2: precision policy, uploads, matmul configs.

Layout. The residual stream is batch-major, ``[rows, 1, seq, 512]`` in bf16, with ``rows`` padded to a multiple
of 32 (dummy all-NaN rows) and ``seq`` = 320 (285 tokens padded to five mLSTM chunks of 64). The sLSTM time
loop runs time-major, ``[seq, 1, rows, *]``, so every step reads one tile-aligned ``[rows, 2048]`` slab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TILE = 32


def pad_to(n: int, m: int = TILE) -> int:
    return -(-n // m) * m


@dataclass(frozen=True)
class Precision:
    """Precision policy.

    ``accuracy``: bf16 weights and activations, HiFi2 dense matmuls, HiFi4 for the sLSTM input projection.
    ``performance``: bfp8 weights, LoFi, bf16 activations.
    In both: the recurrent states (mLSTM C, n, m; sLSTM c, n, m) and the exponential gating are fp32; the
    sLSTM recurrent matmul runs at HiFi4 with ``R`` in bf16 (documented exception); the tiny mLSTM gate
    projection (512 -> 8) keeps fp32 weights; norms run at HiFi4 with fp32 accumulation."""

    name: str = "accuracy"
    bfp8_weights: bool = False
    lofi: bool = False

    @classmethod
    def accuracy(cls) -> "Precision":
        return cls()

    @classmethod
    def performance(cls) -> "Precision":
        return cls(name="performance", bfp8_weights=True, lofi=True)

    @classmethod
    def by_name(cls, name: str) -> "Precision":
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def weight_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bfp8_weights else ttnn.bfloat16

    def matmul_config(self):
        """Dense projections: HiFi2 (accuracy) or LoFi (performance), fp32 accumulation."""
        import ttnn

        return kernel_config(ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2)

    def gate_config(self):
        """The sLSTM input projection ``W_in``, whose outputs feed the exponential gates: HiFi4 in ``accuracy``
        (HiFi2 there is the largest single error source: worst paper-row median error 0.20 -> 0.05 of the
        scale), LoFi in ``performance``."""
        import ttnn

        return kernel_config(ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi4)


def kernel_config(fidelity=None, fp32_acc: bool = True, packer_l1_acc: bool = True):
    import ttnn

    fidelity = ttnn.MathFidelity.HiFi4 if fidelity is None else fidelity
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=packer_l1_acc
    )


def exact_config():
    """Recurrent and normalization math: HiFi4, fp32 accumulation."""
    import ttnn

    return kernel_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)


def upload(device, t: torch.Tensor, dtype=None, layout=None, memory_config=None):
    """Host tensor -> device (DRAM, tile layout, bf16 unless given)."""
    import ttnn

    dtype = ttnn.bfloat16 if dtype is None else dtype
    host = t.detach().contiguous()
    host = host.to(torch.bfloat16) if dtype == ttnn.bfloat16 else host.float()
    if device is None:  # host tensor (e.g. the source of a copy into a traced input)
        return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT if layout is None else layout)
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
    )


def upload_weight(device, w: torch.Tensor, dtype):
    """``[in, out]`` matrix -> ``[1, 1, in, out]`` on device."""
    return upload(device, w.reshape(1, 1, *w.shape), dtype=dtype)


def upload_vector(device, v: torch.Tensor, dtype=None):
    """``[n]`` -> ``[1, 1, 1, n]`` (bias, norm weight)."""
    return upload(device, v.reshape(1, 1, 1, -1), dtype=dtype)


def bmm_config(device, m: int, k: int, n: int):
    """Program config for a batched ``[N, 1, m, k] @ [N, 1, k, n]`` with small tile-aligned ``m, k, n``.

    The default batched path is ~20x slower for these shapes (64 x 64 x 64: 38 ms -> 1.8 ms at N = 20480).
    """
    import ttnn

    grid = device.compute_with_storage_grid_size()
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    sub_w = max(d for d in range(1, min(nt, 4) + 1) if nt % d == 0)
    sub_h = max(d for d in range(1, mt + 1) if mt % d == 0 and d * sub_w <= 4)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=kt,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=mt,
        per_core_N=nt,
    )


def bmm(device, a, b, dtype=None, transpose_b: bool = False, config=None):
    """Batched small matmul with an explicit program config (fp32 output by default)."""
    import ttnn

    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-2] if transpose_b else b.shape[-1]
    return ttnn.matmul(
        a,
        b,
        transpose_b=transpose_b,
        dtype=ttnn.float32 if dtype is None else dtype,
        compute_kernel_config=exact_config() if config is None else config,
        program_config=bmm_config(device, pad_to(m), pad_to(k), pad_to(n)),
    )


def num_tiles(shape) -> int:
    *outer, h, w = shape
    return math.prod(outer) * (pad_to(h) // TILE) * (pad_to(w) // TILE)
