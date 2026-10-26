# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for TiRex-2: precision policy, uploads, matmul configs.

Layout. The residual stream is batch-major, ``[rows, 1, seq, 512]`` in bf16, with ``rows`` padded to a multiple
of 32 (dummy all-NaN rows) and ``seq`` = 320 (285 tokens padded to five mLSTM chunks of 64). The sLSTM time
loop runs time-major, ``[seq, 1, rows, *]``, so every step reads one tile-aligned ``[rows, 2048]`` slab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

TILE = 32


def pad_to(n: int, m: int = TILE) -> int:
    return -(-n // m) * m


ROLES = ("embed", "mlstm", "w_in", "ffn", "variate", "head")


@dataclass(frozen=True)
class Precision:
    """Precision policy, per matmul role (``ROLES``).

    ``accuracy``: bf16 weights and activations; every dense matmul at HiFi4. (Round 0 first ran HiFi2 with only
    ``w_in``, the sLSTM input projection that feeds the exponential gates, at HiFi4; the G1 sweep measured HiFi4
    on every dense role at +0.4 % device time for active-row PCC 0.959 -> 0.995 and a worst median error of
    0.165 -> 0.118 of the scale: ``doc/optimize/iter_00/precision_sweep.md``.)
    ``performance``: bfp8 weights, bf16 activations; ``w_in`` bf16 at HiFi4, embed / mLSTM / head at HiFi2, FFN
    and variate at LoFi (R1-0 sweep, ``doc/optimize/iter_01/r1_0/precision_sweep.md``: LoFi everywhere narrowed the
    band, spread_ratio p01 0.867; this policy 0.938 for +0.4 % device time).
    In every policy:
    - the recurrent states (mLSTM C, n, m; sLSTM c, n, m) and the exponential gating are fp32;
    - the sLSTM recurrent matmul ``y @ R`` runs at HiFi4 with ``R`` in bf16 (documented exception) and fp32
      output; its input ``y`` is cast to bf16 unless ``y_fp32``;
    - the mLSTM gate projection (512 -> 8 per head) keeps fp32 weights, HiFi4, fp32 output;
    - norms and the chunkwise mLSTM matmuls run at HiFi4 with fp32 accumulation;
    - **accumulation:** every matmul accumulates in fp32 (``fp32_dest_acc_en=True``); dense matmuls also use
      ``packer_l1_acc=True``. ``fp32_acc=False`` turns fp32 dest accumulation off for the dense roles only (for
      the measured comparison).

    ``hifi4_roles`` forces HiFi4 on dense roles; ``hifi2_roles`` lifts LoFi roles to HiFi2; ``bf16_roles`` keeps a role's weights bf16 under bfp8.
    """

    name: str = "accuracy"
    bfp8_weights: bool = False
    lofi: bool = False
    hifi4_roles: frozenset = frozenset(ROLES)
    bf16_roles: frozenset = frozenset()
    hifi2_roles: frozenset = frozenset()
    fp32_acc: bool = True
    y_fp32: bool = False

    @classmethod
    def accuracy(cls) -> "Precision":
        return cls()

    @classmethod
    def performance(cls) -> "Precision":
        return cls(
            name="performance",
            bfp8_weights=True,
            lofi=True,
            hifi4_roles=frozenset({"w_in"}),
            bf16_roles=frozenset({"w_in"}),
            hifi2_roles=frozenset({"embed", "mlstm", "head"}),
        )

    @classmethod
    def performance_r0(cls) -> "Precision":
        """The round-0 performance policy (LoFi everywhere, bfp8 weights), kept for comparisons."""
        return cls(name="performance_r0", bfp8_weights=True, lofi=True, hifi4_roles=frozenset())

    @classmethod
    def accuracy_r0(cls) -> "Precision":
        """The first round-0 accuracy policy (HiFi2 dense, HiFi4 ``w_in``), kept for back-to-back comparisons."""
        return cls(name="accuracy_r0", hifi4_roles=frozenset({"w_in"}))

    @classmethod
    def by_name(cls, name: str) -> "Precision":
        """``accuracy`` / ``performance``, optionally with variant suffixes joined by ``+``:
        ``hifi4=<role,..|all>``, ``hifi2=<role,..|all>`` (HiFi2 instead of LoFi), ``bf16=<role,..|all>``, ``y32``, ``acc16`` (no fp32 dest accumulation).
        """
        base, *mods = name.split("+")
        p = {
            "accuracy": cls.accuracy,
            "accuracy_r0": cls.accuracy_r0,
            "performance": cls.performance,
            "performance_r0": cls.performance_r0,
        }[base]()
        kw = {}
        for m in mods:
            key, _, val = m.partition("=")
            roles = frozenset(ROLES) if val == "all" else frozenset(r for r in val.split(",") if r)
            if key == "hifi4":
                kw["hifi4_roles"] = p.hifi4_roles | roles
            elif key == "hifi2":
                kw["hifi2_roles"] = p.hifi2_roles | roles
            elif key == "bf16":
                kw["bf16_roles"] = p.bf16_roles | roles
            elif key == "y32":
                kw["y_fp32"] = True
            elif key == "acc16":
                kw["fp32_acc"] = False
            else:
                raise ValueError(f"unknown precision variant {m!r}")
        return replace(p, name=name, **kw)

    def weight_dtype(self, role: str | None = None):
        import ttnn

        return ttnn.bfloat8_b if self.bfp8_weights and role not in self.bf16_roles else ttnn.bfloat16

    def matmul_config(self, role: str | None = None):
        """Dense projections of ``role``: HiFi4 if listed in ``hifi4_roles``, else HiFi2 (accuracy) or LoFi."""
        import ttnn

        if role in self.hifi4_roles:
            fidelity = ttnn.MathFidelity.HiFi4
        else:
            lofi = self.lofi and role not in self.hifi2_roles
            fidelity = ttnn.MathFidelity.LoFi if lofi else ttnn.MathFidelity.HiFi2
        return kernel_config(fidelity, fp32_acc=self.fp32_acc)

    def gate_config(self):
        """The sLSTM input projection ``W_in`` (role ``w_in``): HiFi4 in ``accuracy`` (at HiFi2 it was the largest
        single error source: worst paper-row median error 0.20 -> 0.05 of the scale), and HiFi4 with bf16 weights in
        ``performance`` too (R1-0, D4)."""
        return self.matmul_config("w_in")


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
