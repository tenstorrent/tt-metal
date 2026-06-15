# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GatherIn0FatLinear -- weight-stationary 2D matmul on the gather_in0 substrate
(deep-plan_15 SCOPE-A).

This is the NEW path: ``Matmul1D`` with ``gather_in0=True`` via
``MatmulMultiCoreReuseMultiCast1DProgramConfig``. The op-code emitted is
``MatmulDeviceOperation`` (MM_NATIVE) -- NOT ``MatmulDecodeDeviceOperation``. The op
keeps in1 (the WEIGHT) L1-RESIDENT WIDTH_SHARDED across the full K-tile-divisible grid
(set_globally_allocated_address inside matmul_multicore_reuse_mcast_1d_program_factory.cpp);
it rings the ACTIVATION (in0, cb_in2). The weight is staged ONCE (resident cache keyed on
id(weight)) and reused across calls -- ``buffer_address()`` is queryable + stable.

The KERNEL-visible lever is FAT-FILL: ``out_subblock_h>1`` runs the native systolic FPU
rectangle fill. The :1119 validator gate
(``out_subblock_w == per_core_N || out_subblock_h == 1``) is REAL dataflow correctness on
the gather path (NO re-stride writer); we honor it by deriving a GATE-LEGAL fat subblock
(``ow == per_core_N`` so ``oh`` may be > 1). For per_core_N>2 this only admits oh=1 (the thin
baseline) -- those shapes need the SCOPE-B re-stride writer and are out of scope here.

Probe-confirmed (deep-plan_15 Phase-0): SigLIP fc1 oh=1 ow=4 PCC 0.999969, in1 resident
stable; oh=2/oh=4 FATAL :1119 at per_core_N=4. SigLIP o / VLM o (per_core_N=1) and VLM qkv
(per_core_N=2) admit a legal fat config.

Pure-ttnn ``__call__`` (no torch.* in the compute path). PyTorch used only in __init__
(weight prep). Per-model (pi05_chunked_mmdecode), a benchmark/feasibility instrument.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn

TT_METAL_COMMIT = "e4500c1fae97c103b16fc24fc7010b852992a9e6"

BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16
TILE = 32

# DST register tile capacity (Blackhole): 8 bf16 tiles; fp32_dest_acc halves to 4.
_DST_CAP_BF16 = 8
_DST_CAP_FP32 = 4


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _rect_grid(num_cores: int, gx: int):
    """Largest rectangle (x, y) with x*y == num_cores and x <= gx (probe rule)."""
    x = gx
    while x > 0 and num_cores % x != 0:
        x -= 1
    return (x, num_cores // x) if x else None


def gather_legal_fat_subblock(out_block_h: int, per_core_N: int, dst_cap: int):
    """Largest-area (oh, ow) honoring BOTH:
      * DST cap: oh*ow <= dst_cap
      * tiling: out_block_h % oh == 0 and per_core_N % ow == 0
      * :1119 GATHER gate: ow == per_core_N OR oh == 1
    Walk descending oh (fat-M first), with ow == per_core_N when it fits the cap, else ow
    that divides per_core_N with oh==1. Graceful-degrade to (1,1)-ish thin baseline."""
    # Prefer fat-M: ow == per_core_N (gate-legal), oh as large as the cap + tiling allow.
    if per_core_N <= dst_cap:
        max_oh = dst_cap // per_core_N
        for oh in range(min(max_oh, out_block_h), 0, -1):
            if out_block_h % oh == 0:
                return (oh, per_core_N)
    # per_core_N > dst_cap (or nothing fit fat): thin oh=1, widest ow dividing both N and cap.
    for ow in range(min(per_core_N, dst_cap), 0, -1):
        if per_core_N % ow == 0:
            return (1, ow)
    return (1, 1)


def derive_gather_grid(K: int, num_cores_hint: int, gx: int):
    """Pick a K-tile-divisible core count <= grid that the gather ring needs
    (in0_block_w = K_pad/num_cores/TILE). Prefer the hint when K_tiles % hint == 0."""
    Kt = K // TILE
    # Candidates: divisors of Kt that form a rectangle under gx, descending toward the hint.
    cands = [c for c in range(min(num_cores_hint, Kt), 0, -1) if Kt % c == 0 and _rect_grid(c, gx)]
    return cands[0] if cands else 1


class GatherIn0FatLinear:
    """One [K, N] projection via native gather_in0 (Matmul1D) with a resident
    WIDTH_SHARDED-L1 in1 weight + a GATE-LEGAL fat-fill subblock.

    Lifecycle: weight prep (torch) in __init__; ``stage()`` materializes the resident
    WIDTH_SHARDED-L1 weight ONCE (cached by id(weight)); pure-ttnn ``__call__``."""

    # process-wide resident-in1 cache: (id(weight_torch), shape, dtype, num_cores) -> tt tensor
    _resident_cache: dict = {}

    def __init__(
        self,
        device,
        weight_KN_torch,
        *,
        m_rows: int,
        num_cores: int,
        weight_dtype=BF8,
        out_dtype=BF16,
        pad_n: Optional[int] = None,
        pad_k: Optional[int] = None,
        role: str = "generic",
        resident_weight: bool = True,
        fp32_dest_acc: bool = True,
        oh_override: Optional[int] = None,
        ow_override: Optional[int] = None,
    ):
        self.device = device
        self.role = role
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.m_rows = int(m_rows)
        self.resident_weight = bool(resident_weight)
        self.fp32_dest_acc = bool(fp32_dest_acc)
        self._weight_id = id(weight_KN_torch)

        w = weight_KN_torch
        K0, N0 = int(w.shape[0]), int(w.shape[1])
        self.K_orig, self.N_orig = K0, N0
        if pad_n and pad_n > N0:
            w = torch.nn.functional.pad(w, (0, pad_n - N0))
        if pad_k and pad_k > K0:
            w = torch.nn.functional.pad(w, (0, 0, 0, pad_k - K0))
        K, N = int(w.shape[0]), int(w.shape[1])

        grid = device.compute_with_storage_grid_size()
        self.gx = int(grid.x)
        self.num_cores = derive_gather_grid(K, num_cores, self.gx)
        self.grid = _rect_grid(self.num_cores, self.gx)
        assert self.grid, (self.num_cores, self.gx)

        # Width-shard K and N across num_cores; round each per-core slice up to a tile.
        self.K_per = _round_up(math.ceil(K / self.num_cores), TILE)
        self.K_pad = self.K_per * self.num_cores
        self.N_per = _round_up(math.ceil(N / self.num_cores), TILE)
        self.N_pad = self.N_per * self.num_cores
        if self.K_pad != K:
            w = torch.nn.functional.pad(w, (0, 0, 0, self.K_pad - K))
        if self.N_pad != N:
            w = torch.nn.functional.pad(w, (0, self.N_pad - N))
        self.K, self.N = self.K_pad, self.N_pad
        self._w_KN = w.to(torch.bfloat16).contiguous()

        self.in0_block_w = self.K_pad // self.num_cores // TILE
        self.out_block_h = _round_up(self.m_rows, TILE) // TILE  # per_core_M (M-tiles)
        self.per_core_N = self.N_pad // self.num_cores // TILE  # out_block_w
        self.M_pad = _round_up(self.m_rows, TILE)

        dst_cap = _DST_CAP_FP32 if self.fp32_dest_acc else _DST_CAP_BF16
        if oh_override is not None and ow_override is not None:
            self.out_subblock_h, self.out_subblock_w = int(oh_override), int(ow_override)
        else:
            self.out_subblock_h, self.out_subblock_w = gather_legal_fat_subblock(
                self.out_block_h, self.per_core_N, dst_cap
            )
        self.dst_cap = dst_cap

        self.crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid[0] - 1, self.grid[1] - 1))}
        )
        self.in0_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.crs, [self.M_pad, self.K_per], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.in1_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.crs, [self.K_pad, self.N_per], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.out_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.crs, [self.M_pad, self.N_per], ttnn.ShardOrientation.ROW_MAJOR),
        )

        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=self.fp32_dest_acc,
            packer_l1_acc=False,
            dst_full_sync_en=True,
        )

        self.program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.grid,
            in0_block_w=self.in0_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            per_core_M=self.out_block_h,
            per_core_N=self.per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
        )

        self._w_tt = None
        self._resident_addr = None
        self._staged = False
        self.stage()

    def stage(self):
        if self._staged:
            return
        key = (self._weight_id, tuple(self._w_KN.shape), str(self.weight_dtype), self.num_cores)
        if self.resident_weight and key in GatherIn0FatLinear._resident_cache:
            self._w_tt = GatherIn0FatLinear._resident_cache[key]
        else:
            self._w_tt = ttnn.from_torch(
                self._w_KN,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.weight_dtype,
                memory_config=self.in1_mc,
            )
            if self.resident_weight:
                GatherIn0FatLinear._resident_cache[key] = self._w_tt
        self._resident_addr = int(self._w_tt.buffer_address())
        self._w_KN = None
        self._staged = True

    def buffer_address(self) -> int:
        return int(self._w_tt.buffer_address())

    def describe(self) -> dict:
        return dict(
            role=self.role,
            K=self.K,
            N=self.N,
            K_orig=self.K_orig,
            N_orig=self.N_orig,
            num_cores=self.num_cores,
            grid=self.grid,
            in0_block_w=self.in0_block_w,
            per_core_M=self.out_block_h,
            per_core_N=self.per_core_N,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            dst_cap=self.dst_cap,
            fp32_dest_acc=self.fp32_dest_acc,
            resident=self.resident_weight,
            in1_addr=hex(self._resident_addr or 0),
            m_rows=self.m_rows,
            weight_dtype=str(self.weight_dtype),
        )

    def device_calls_per_call(self) -> int:
        return 1  # ONE gather Matmul1D row per __call__

    def __call__(self, x_dev):
        """y = x_dev @ W. Pure-ttnn. Expects [1,1,M,K] or [M,K]; returns the same rank,
        N-width sharded L1. Re-stages weight every call when resident_weight is False (Leg X)."""
        if not self.resident_weight:
            # Leg X: re-DMA + re-shard the weight every call (isolates residency-saved glue).
            w_re = ttnn.from_torch(
                ttnn.to_torch(self._w_tt) if self._w_KN is None else self._w_KN,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.weight_dtype,
                memory_config=self.in1_mc,
            )
            b = w_re
        else:
            b = self._w_tt
        orig = list(x_dev.shape)
        K = self.K
        # Pad K columns if needed.
        if orig[-1] != K:
            x_dev = ttnn.pad(x_dev, [(0, 0)] * (len(orig) - 1) + [(0, K - orig[-1])], 0.0)
        # Shard in0 WIDTH_SHARDED-L1 (the gather ring rings this).
        a_sh = ttnn.to_memory_config(x_dev, self.in0_mc)
        y = ttnn.matmul(
            a_sh,
            b,
            program_config=self.program_config,
            memory_config=self.out_mc,
            compute_kernel_config=self.ckc,
            dtype=self.out_dtype,
        )
        ttnn.deallocate(a_sh)
        if not self.resident_weight:
            ttnn.deallocate(b)
        return y
