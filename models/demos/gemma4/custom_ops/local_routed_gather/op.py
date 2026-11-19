# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``LocalRoutedGather``: the token side of the local routed-expert prefill (``tt/experts/local_routed_prefill.py``)
as one program: from a chunk's hidden rows and top-k routing, replicated on every chip of a whole-expert TP mesh,
every chip lays out the (token, slot) pairs routed to ITS experts expert-major and writes the combine's per-token row
index and weights. No fabric: every chip already holds every token.

The cores split the chunk's tokens (contiguous 32-token units, ``plan_units``). Each core scans its tokens' top-k ids
and counts its pairs per local expert; the counts go to the leader core, which lays out the tile-aligned expert
regions (prefix sum of the per-expert totals rounded up to 32, in local-expert order) and hands every core its first
row per expert (region start + the pairs of that expert on the cores before it), so the rows of an expert region are
the expert's tokens in ascending order. Each core then copies its tokens' hidden rows into their buffer rows and
writes, per token, the buffer row of every local slot (``SKIP_ROW`` elsewhere) and the slot weights (the router's
value on local slots, zero elsewhere): ``post_combine_reduce`` with ``combine_row_index`` then sums the weighted
expert rows per token and reads nothing for the other chips' slots.

Contract (per chip; the same program runs on every chip with its own ``idx`` table):
| tensor | shape / dtype / layout | role |
|---|---|---|
| ``hidden`` | ``[1, 1, T_real, H]`` bf16 ROW_MAJOR DRAM (one page per row) | the chunk's rows, replicated |
| ``ids`` | ``[1, 1, T_real, K]`` uint16 TILE DRAM | top-k global expert ids (ids >= E: no expert, e.g. a sentinel pad) |
| ``values`` | ``[1, 1, T_real, K]`` bf16 TILE DRAM | the top-k routing weights |
| ``idx`` | ``[E_loc]`` uint32 ROW_MAJOR DRAM | local slot -> global expert id |
| ``buffer`` (out) | ``[M, H]`` bf16 ROW_MAJOR DRAM | the GeGLU op's input: expert ``idx[e]``'s rows at ``[offsets[idx[e]], + counts[idx[e]])``; other rows untouched |
| ``counts``, ``offsets`` (out) | ``[1, E]`` uint32 ROW_MAJOR DRAM | per GLOBAL expert id (zero for experts not on this chip) |
| ``row_index`` (out) | ``[1, 1, T, K]`` uint32 ROW_MAJOR DRAM | buffer row of every local (token, slot), ``SKIP_ROW`` elsewhere |
| ``weights`` (out) | ``[1, 1, T, K]`` bf16 ROW_MAJOR DRAM | the slot's routing weight on local slots, 0 elsewhere |

``T`` is the chunk's row bucket (``T_real`` rounded up; one program per bucket, ``T_real`` a runtime argument): tokens
past ``T_real`` route nowhere. ``M = T * K + 32 * E_loc`` is the
drop-free capacity (every token's K slots on this chip, one tile of alignment per local expert); outputs have fixed
shapes and the program reads nothing on the host, so a call is trace-safe.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _up64(v: int) -> int:
    return -(-int(v) // 64) * 64


class LocalRoutedGather:
    SEM = 0
    MAX_CORES = 64  # the leader's per-core count / base tables and its runtime args (two NoC coords per core)
    MAX_LOCAL_EXPERTS = 64  # the kernel's (local expert << 16 | rank) pair codes
    MAX_EXPERTS = 256  # the kernel's one-byte global id -> local slot table
    NR = 8  # tokens per copy batch (NR hidden rows staged)
    SKIP_ROW = 0xFFFFFFFF  # post_combine_reduce's default skip row: never read

    @staticmethod
    def capacity(T: int, K: int, E_loc: int) -> int:
        """Buffer rows ``M``: every token's K slots on this chip plus one tile of region alignment per local expert."""
        return int(T) * int(K) + TILE * int(E_loc)

    @classmethod
    def supports(cls, E: int, E_loc: int, K: int, H: int) -> bool:
        """Whether the kernel takes this expert layout (the CI's reduced-expert configurations included): at most
        ``MAX_EXPERTS`` global and ``MAX_LOCAL_EXPERTS`` local experts, 1 <= K <= 16 (the slots in one tile face
        column), a 64-byte multiple hidden row."""
        E, E_loc, K, H = int(E), int(E_loc), int(K), int(H)
        return (
            0 < E <= cls.MAX_EXPERTS
            and 0 < E_loc <= min(E, cls.MAX_LOCAL_EXPERTS)
            and 1 <= K <= 16
            and (2 * H) % 64 == 0
        )

    @classmethod
    def plan_units(cls, T: int, grid_cores: int):
        """``(n_cores, starts, counts)``: the ``T / 32`` token units dealt contiguously over ``min(units, grid,
        MAX_CORES)`` cores, the counts differing by at most one. Pure host logic (unit-tested)."""
        assert T % TILE == 0 and T > 0, T
        units = T // TILE
        n = max(1, min(units, int(grid_cores), cls.MAX_CORES))
        q, r = divmod(units, n)
        counts = [q + (1 if i < r else 0) for i in range(n)]
        starts = [sum(counts[:i]) for i in range(n)]
        return n, starts, counts

    @staticmethod
    def golden(hidden: torch.Tensor, ids: torch.Tensor, values: torch.Tensor, idx, E: int):
        """torch reference of one chip. ``hidden`` [T, H], ``ids`` / ``values`` [T, K] -> (buffer rows {row: token},
        counts [E], offsets [E], row_index [T, K] (``SKIP_ROW`` off-chip), weights [T, K]). Rows of an expert region
        hold its tokens in ascending order; regions tile-aligned in local-expert order."""
        T, K = ids.shape
        counts, offsets = torch.zeros(E, dtype=torch.int64), torch.zeros(E, dtype=torch.int64)
        row_index = torch.full((T, K), LocalRoutedGather.SKIP_ROW, dtype=torch.int64)
        weights = torch.zeros(T, K, dtype=values.dtype)
        rows = {}
        run = 0
        for g in [int(v) for v in idx]:
            hits = (ids == g).nonzero().tolist()  # (t, k) in token order
            counts[g], offsets[g] = len(hits), run
            for i, (t, k) in enumerate(hits):
                row_index[t, k] = run + i
                weights[t, k] = values[t, k]
                rows[run + i] = t
            run += (len(hits) + TILE - 1) // TILE * TILE
        return rows, counts, offsets, row_index, weights

    def __init__(self, T: int, E: int, E_loc: int, K: int, H: int):
        assert T % TILE == 0 and T > 0, T
        assert self.supports(E, E_loc, K, H), f"unsupported expert layout E {E} E_loc {E_loc} K {K} H {H}"
        self.T, self.E, self.E_loc, self.K, self.H = T, E, E_loc, K, H
        self.M = self.capacity(T, K, E_loc)

    def op(self, hidden: ttnn.Tensor, ids: ttnn.Tensor, values: ttnn.Tensor, idx: ttnn.Tensor):
        T, E, E_loc, K, H, M = self.T, self.E, self.E_loc, self.K, self.H, self.M
        T_real = int(hidden.shape[-2])
        assert 0 < T_real <= T, f"{T_real} rows on a {T}-row gather"
        assert hidden.layout == ttnn.ROW_MAJOR_LAYOUT and hidden.dtype == ttnn.bfloat16, "hidden: bf16 ROW_MAJOR"
        assert [int(v) for v in hidden.shape][-2:] == [T_real, H], f"hidden {hidden.shape} != [.., {T_real}, {H}]"
        for t, name, dt in ((ids, "ids", ttnn.uint16), (values, "values", ttnn.bfloat16)):
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == dt, f"{name}: {dt} TILE, got {t.dtype} {t.layout}"
            assert [int(v) for v in t.shape][-2:] == [T_real, K], f"{name} {t.shape} != [.., {T_real}, {K}]"
        assert (
            idx.layout == ttnn.ROW_MAJOR_LAYOUT and idx.dtype == ttnn.uint32 and int(idx.shape[-1]) == E_loc
        ), idx.shape
        device = hidden.device()
        dram = ttnn.DRAM_MEMORY_CONFIG
        rm = ttnn.ROW_MAJOR_LAYOUT
        buffer = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H]), ttnn.bfloat16, rm, device, dram)
        counts = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, rm, device, dram)
        offsets = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, rm, device, dram)
        row_index = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, K]), ttnn.uint32, rm, device, dram)
        weights = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, K]), ttnn.bfloat16, rm, device, dram)
        gx, gy = compute_grid(device)
        n, starts, unit_counts = self.plan_units(T, gx * gy)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        noc = [device.worker_core_from_logical_core(c) for c in cores]
        units_max = max(unit_counts)
        ri_page, w_page = row_index.buffer_aligned_page_size(), weights.buffer_aligned_page_size()
        vec_page = counts.buffer_aligned_page_size()
        assert offsets.buffer_aligned_page_size() == vec_page
        cnt_bytes = _up64(E_loc * 4)
        scratch = (
            256
            + _up64(E_loc * 4)
            + 2 * units_max * 2048
            + _up64(units_max * TILE * K * 4)
            + 2 * cnt_bytes
            + 2 * n * cnt_bytes
            + 2 * _up64(vec_page)
            + self.NR * (_up64(ri_page) + _up64(w_page) + _up64(2 * H))
        )  # the kernel's scratch layout (64-aligned regions)
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
                ],
            )
        ]
        ct = [0, E, E_loc, K, T, H, M, n, units_max, self.SEM, noc[0].x, noc[0].y, self.NR]
        ct += [ri_page, w_page, vec_page, self.SKIP_ROW]
        tensors = [hidden, ids, values, idx, buffer, counts, offsets, row_index, weights]
        for t in tensors:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        coords = []
        for c in noc:
            coords += [c.x, c.y]
        addrs = [t.buffer_address() for t in tensors]
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            rt[core.x][core.y] = addrs + [i, starts[i], unit_counts[i], T_real] + (coords if i == 0 else [])
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "local_routed_gather.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        sems = [ttnn.SemaphoreDescriptor(self.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op(tensors, program)
        return buffer, counts, offsets, row_index, weights
