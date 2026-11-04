# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``ExpertGatherSmall``: the dispatch side of the routed experts for a small prefill chunk (<= 256 tokens) whose
hidden rows and dense routing matrix are replicated on every chip of the dispatch group, as one program.

Instead of the DeepSeek token dispatch over the fabric, every chip picks from its own copy of the chunk the rows
routed to ITS experts: core ``e`` (one per local expert) scans column ``idx[e]`` of the routing matrix, the cores
exchange their counts through the leader core, which lays out the tile-aligned expert regions, and each core
copies its tokens' rows into its region of the GeGLU buffer and writes its column tiles of the weight matrix
``W`` (``W[t, m] = routing weight`` of token ``t`` at buffer row ``m``, zero elsewhere). The combine is then the
matmul ``W @ expert_out`` ([T, M] x [M, H]) on the compute engine and one all-reduce of the per-chip partials
over the dispatch axis (``tt/experts/dispatch_prefill.py``, ``GEMMA4_SMALL_ROW_EXPERTS=1``).

Contract (per chip; the same program runs on every chip with its own ``idx`` table):
| tensor | shape / dtype / layout | role |
|---|---|---|
| ``hidden`` | ``[1, 1, T, H]`` bf16 ROW_MAJOR DRAM (one page per row) | the chunk's rows, replicated |
| ``routing`` | ``[1, 1, T, E]`` bf16 TILE DRAM | the dense routing matrix (weights at the selected experts, zeros elsewhere), replicated |
| ``idx`` | ``[E_loc]`` uint32 ROW_MAJOR DRAM | local slot -> global expert id (``TtRoutedExpert``'s table) |
| ``buffer`` (out) | ``[M, H]`` bf16 ROW_MAJOR DRAM | the GeGLU op's input: expert ``idx[e]``'s rows at ``[offsets[idx[e]], + counts[idx[e]])``; rows outside the regions untouched |
| ``counts``, ``offsets`` (out) | ``[1, E]`` uint32 ROW_MAJOR DRAM | per GLOBAL expert id (zero for experts not on this chip); regions tile-aligned in local-expert order |
| ``w`` (out) | ``[1, 1, T, M]`` bf16 TILE DRAM | the combine weights: ``W[t, m]`` = the token's routing weight at its buffer row, zero elsewhere (every column tile written) |

``M = T * K + 32 * (E_loc - 1)`` is the drop-free capacity (every token's K experts on this chip, one tile of
alignment per local expert); it is fixed at construction (``ExpertGatherSmall(T, E, E_loc, K, H)``) and the
outputs have fixed shapes, so a call is trace-safe (no host reads, no data-dependent shapes).
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid
from models.demos.gemma4.tt.mesh_orientation import decode_flag

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class ExpertGatherSmall:
    CB_ROUTING, CB_SCRATCH, CB_TILE = 0, 1, 2
    SEM = 0
    MAX_TOKENS = 256
    MAX_LOCAL_EXPERTS = 64  # the kernel's fixed scratch regions: idx table 256 B, offset table 512 B (static_asserts)
    ROWS_BATCH = 16  # hidden rows copied per batch (NR reads, one barrier, NR writes, one barrier)

    @staticmethod
    def enabled() -> bool:
        """``GEMMA4_SMALL_ROW_EXPERTS`` through the decode-fusion resolver (env wins, else the build's profile: on
        for the 4-chip Blackhole MoE build, off elsewhere): the small-row expert path for short prefill chunks."""
        return decode_flag("GEMMA4_SMALL_ROW_EXPERTS") == "1"

    @staticmethod
    def capacity(T: int, K: int, E_loc: int) -> int:
        return T * K + TILE * (E_loc - 1)

    @classmethod
    def supports(cls, E: int, E_loc: int, K: int, H: int) -> bool:
        """Whether the gather kernel takes this expert layout: a tile multiple of experts, a local share of them of
        at most ``MAX_LOCAL_EXPERTS`` (the kernel's fixed idx / offset-table scratch regions), 1 <= K <= E, and a
        hidden row that fits the kernel's L1 staging. The plan declines (shared dispatch path) instead of the
        constructor asserting on a reduced-expert configuration."""
        E, E_loc, K, H = int(E), int(E_loc), int(K), int(H)
        return (
            E > 0
            and E % TILE == 0
            and 0 < E_loc <= min(E, cls.MAX_LOCAL_EXPERTS)
            and 1 <= K <= E
            and H % TILE == 0
            and 2 * H <= 8192
        )

    def __init__(self, T: int, E: int, E_loc: int, K: int, H: int):
        assert T % TILE == 0 and 0 < T <= self.MAX_TOKENS, f"T {T}: a multiple of 32 up to {self.MAX_TOKENS}"
        assert self.supports(E, E_loc, K, H), f"unsupported expert layout E {E} E_loc {E_loc} K {K} H {H}"
        self.T, self.E, self.E_loc, self.K, self.H = T, E, E_loc, K, H
        self.NR = min(self.ROWS_BATCH, T)
        self.M = self.capacity(T, K, E_loc)
        assert self.M % TILE == 0 and self.M >= T * K, "drop-free buffer contract"

    @staticmethod
    def golden(hidden: torch.Tensor, routing: torch.Tensor, idx, T: int, K: int, E: int):
        """torch reference. hidden [T, H], routing [T, E] (float), idx list of global ids ->
        (buffer [M, H] with the regions filled (other rows zero), counts [E], offsets [E], w [T, M])."""
        E_loc = len(idx)
        M = ExpertGatherSmall.capacity(T, K, E_loc)
        buffer = torch.zeros(M, hidden.shape[-1], dtype=hidden.dtype)
        w = torch.zeros(T, M, dtype=routing.dtype)
        counts, offsets = torch.zeros(E, dtype=torch.int64), torch.zeros(E, dtype=torch.int64)
        run = 0
        for e, g in enumerate(idx):
            toks = torch.nonzero(routing[:, g] != 0).flatten()
            counts[g], offsets[g] = len(toks), run
            for i, t in enumerate(toks.tolist()):
                buffer[run + i] = hidden[t]
                w[t, run + i] = routing[t, g]
            run += (len(toks) + TILE - 1) // TILE * TILE
        return buffer, counts, offsets, w

    def op(self, hidden: ttnn.Tensor, routing: ttnn.Tensor, idx: ttnn.Tensor):
        T, E, E_loc, H, M = self.T, self.E, self.E_loc, self.H, self.M
        assert hidden.layout == ttnn.ROW_MAJOR_LAYOUT and hidden.dtype == ttnn.bfloat16, "hidden: bf16 ROW_MAJOR"
        assert [int(v) for v in hidden.shape] == [1, 1, T, H], f"hidden {hidden.shape} != [1, 1, {T}, {H}]"
        assert routing.layout == ttnn.TILE_LAYOUT and routing.dtype == ttnn.bfloat16, "routing: bf16 TILE"
        assert [int(v) for v in routing.shape] == [1, 1, T, E], f"routing {routing.shape} != [1, 1, {T}, {E}]"
        assert (
            idx.layout == ttnn.ROW_MAJOR_LAYOUT and idx.dtype == ttnn.uint32 and int(idx.shape[-1]) == E_loc
        ), idx.shape
        device = hidden.device()
        dram = ttnn.DRAM_MEMORY_CONFIG
        buffer = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, dram)
        counts = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, dram)
        offsets = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, dram)
        w = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, M]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)
        gx, gy = compute_grid(device)  # (x, y), capped by GEMMA4_TEST_GRID
        assert E_loc <= gx * gy, f"{E_loc} local experts exceed the {gx}x{gy} grid"
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(E_loc)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        noc = [device.worker_core_from_logical_core(c) for c in cores]
        tile_bytes = TILE * TILE * 2
        rows_t, cols_e_t, m_t = T // TILE, E // TILE, M // TILE
        scratch_bytes = (
            256 + T * 8 + E_loc * 64 + 64 + 512 + 2 * E * 4 + self.NR * H * 2
        )  # the kernel's scratch layout (64-aligned regions, NR staged rows)
        scratch_bytes = (scratch_bytes + 63) // 64 * 64

        def cb(idx_, total, page, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx_, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(self.CB_ROUTING, rows_t * tile_bytes, tile_bytes),
            cb(self.CB_SCRATCH, scratch_bytes, scratch_bytes),
            cb(self.CB_TILE, (rows_t + 1) * tile_bytes, tile_bytes),  # the zero tile + one staged column of W
        ]
        ct = [
            self.CB_ROUTING,
            self.CB_SCRATCH,
            self.CB_TILE,
            E,
            E_loc,
            T,
            H,
            M,
            rows_t,
            cols_e_t,
            m_t,
            self.SEM,
            noc[0].x,
            noc[0].y,
            self.NR,
        ]
        for t in (hidden, routing, idx, buffer, counts, offsets, w):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        coords = []
        for c in noc:
            coords += [c.x, c.y]
        rt = ttnn.RuntimeArgs()
        for e, core in enumerate(cores):
            rt[core.x][core.y] = [
                hidden.buffer_address(), routing.buffer_address(), idx.buffer_address(), buffer.buffer_address(),
                counts.buffer_address(), offsets.buffer_address(), w.buffer_address(), e,
            ] + coords  # fmt: skip
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "expert_gather_small.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        sems = [ttnn.SemaphoreDescriptor(self.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([hidden, routing, idx, buffer, counts, offsets, w], program)
        return buffer, counts, offsets, w
