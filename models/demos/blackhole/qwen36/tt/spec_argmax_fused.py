# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_ARGMAX_FUSED (default 0): the spec iteration's argmax tails as two small kernels and ONE all-gather.

What it replaces. Both spec argmax tails take the vocab-sharded LM-head output (each device holds [1,1,R,vocab/TP]):
  * verify (QWEN36_VERIFY_SHARDED_ARGMAX, R = B*T rows, bf16): Qwen36Model.shard_argmax_max_rows (untilize, the
    multi-core ROW_MAJOR ttnn.argmax, then the shard max a second time: pad + grid reshape + 2 ttnn.max + reshape) and
    _dl_greedy_token_row_batched (2 all-gathers and ~30 one-tile slice / max / compare / typecast ops);
  * drafter (QWEN36_DRAFT_SHARDED_ARGMAX, R = users, fp32): Qwen36MTP._argmax_sharded (the same argmax + max pair,
    2 all-gathers, a second argmax, ~12 one-tile ops).
On the 32K profile these cost ~0.72 ms per verify and ~0.50 ms per draft step (2.2 ms per K=3 iteration).

What runs instead (both tails, same output contracts):
  1. kernels/argmax_shard_local.cpp, multi-core: reads the TILE shard in place and writes, per row, the order value of
     its max and that max's lowest local column into one uint32 [32,32] tile (per-core scan, then a two-level merge
     over the grid rows);
  2. ONE ttnn.all_gather of that tile (dim 3) -> [1,1,32,32*TP];
  3. kernels/argmax_shard_pick.cpp, one core: per row the device with the strictly greatest value, scanning devices
     upward, id = device * vocab/TP + local column, written as the tail's own output tensor.
Replay time of the tail alone in a trace, 8-chip ring, 32K-row shapes (tests/test_spec_argmax_fused.py::
test_fused_argmax_trace): verify (32 rows bf16) 710 -> 62 us, drafter step (8 rows fp32) 491 -> 34 us.

Exactness. The id of every row is torch.argmax over the full vocab row (the lowest id among the maxima), the rule the
shipped tails implement: ranking is on raw bits (order value |v| or -|v|, +0 == -0), so it is exact in the logits'
own dtype (bf16 verify, fp32 drafter) and needs no float reduce at all. Rows are independent. NaN is the only input
where it can differ (see the kernel header). Flag off: no instance is built, so no program and no buffer.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"
LOCAL_KERNEL = str(_KERNELS / "argmax_shard_local.cpp")
PICK_KERNEL = str(_KERNELS / "argmax_shard_pick.cpp")

LANES = 0  # output [1,1,1,32] uint32 RM, lane r = row r's id, lanes >= R = 0 (the verify ids row)
ROWS = 1  # output [1,1,R,1] uint32 RM, one id per row (draft_argmax)

_CB_IN0, _CB_IN1, _CB_PART1, _CB_PART2, _CB_OUT = 0, 1, 2, 3, 4  # stage 1
_SLOT_BYTES = 32 * 16  # stage-1 partial slot: 32 rows x (order, col, 0, 0)
_TILE = 32


def all_gather_dim3(t):
    """The collective both tails use for the one stage-1 tile: ttnn.all_gather along dim 3 into DRAM (13.5 us for one
    uint32 tile on the 8-chip ring, about half of all_gather_async with the model's tt_ccl semaphores)."""
    return ttnn.all_gather(t, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def spec_argmax_fused_enabled():
    """QWEN36_SPEC_ARGMAX_FUSED=1 routes the verify and drafter argmax tails through FusedShardArgmax (default 0)."""
    return bool(int(os.environ.get("QWEN36_SPEC_ARGMAX_FUSED", "0")))


def _row_split(rows):
    """Rows of the BRISC half and the NCRISC half: [0, ceil(R/2)) and [ceil(R/2), R) (the second is empty at R = 1)."""
    h = (rows + 1) // 2
    return [(0, h), (h, rows)]


def _stage_tile_stride(lo, hi, esz):
    """Bytes one tile occupies in a RISC's staging buffer (mirror of TILE_STRIDE in argmax_shard_local.cpp)."""
    frb = 16 * esz
    ra = 64 // frb

    def chunk(a, b):
        if b <= a:
            return 0
        al, ah = a // ra * ra, (b + ra - 1) // ra * ra
        return (ah - al) * frb

    ca = chunk(min(lo, 16), min(hi, 16))
    cb = chunk(max(lo, 16) - 16, max(hi, 16) - 16)
    return 2 * (ca + cb)


class FusedShardArgmax:
    """The fused per-shard argmax of one vocab-sharded LM head (no device state: every call allocates its own outputs).

    ``gather(t)`` all-gathers the [1,1,32,32] uint32 TILE stage-1 tile along dim 3 across the TP devices in mesh order
    (device d's tile lands at tile column d, as the vocab shards are laid out) and may free ``t``; default
    all_gather_dim3. ``max_cores`` caps the stage-1 grid (QWEN36_SPEC_ARGMAX_FUSED_CORES, default the full compute
    grid)."""

    _unsupported = set()  # reasons already logged by ids() (process-wide: one line per reason)

    def __init__(self, mesh_device, vocab_size, num_devices, gather=all_gather_dim3, max_cores=None):
        self.mesh = mesh_device
        self.num_devices = int(num_devices)
        self.vocab_size = int(vocab_size)
        assert self.vocab_size % self.num_devices == 0, (vocab_size, num_devices)
        self.shard = self.vocab_size // self.num_devices
        self.gather = gather
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = ttnn.CoreCoord(grid.x, grid.y)
        cap = grid.x * grid.y
        env = int(os.environ.get("QWEN36_SPEC_ARGMAX_FUSED_CORES", "0") or 0)
        if max_cores is None and env > 0:
            max_cores = env
        self.max_cores = cap if not max_cores else max(1, min(int(max_cores), cap))
        # NOC coordinates of the level-1 merge leader of every grid row (core (0, y)); row 0's is the root.
        self.leaders = []
        for y in range(grid.y):
            c = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, y))
            self.leaders += [int(c.x), int(c.y)]
        self._plans = {}

    # ------------------------------------------------------------------------------------------------------------- #
    def supports(self, logits):
        """None if ``logits`` (one device's [1,1,R,vocab/TP] shard) can take the fused path, else the reason."""
        shape = tuple(int(s) for s in logits.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
            return f"shape {shape} is not [1,1,R,W]"
        if not 1 <= shape[2] <= _TILE:
            return f"{shape[2]} rows (need 1..32)"
        if shape[3] != self.shard:
            return f"width {shape[3]} != vocab shard {self.shard}"
        if logits.layout != ttnn.TILE_LAYOUT:
            return "not TILE layout"
        if logits.dtype not in (ttnn.bfloat16, ttnn.float32):
            return f"dtype {logits.dtype}"
        if logits.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return "not interleaved"
        return None

    def _plan(self, rows, width, padded_width, fp32):
        """Static (address-independent) part of the stage-1 program: core split, CB and semaphore descriptors."""
        key = (rows, width, padded_width, fp32)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        wt = padded_width // _TILE
        n = min(self.max_cores, wt)
        tpc = -(-wt // n)
        n = -(-wt // tpc)  # no core without tiles
        crs = ttnn.num_cores_to_corerangeset(n, self.grid, True)  # row-wise: core c = y * grid.x + x
        ng = -(-n // self.grid.x)  # grid rows in use
        esz = 4 if fp32 else 2
        halves = [(h, lo, hi) for h, (lo, hi) in enumerate(_row_split(rows)) if hi > lo]
        cbs = []
        for h, lo, hi in halves:
            size = tpc * _stage_tile_stride(lo, hi, esz) + 64
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=crs,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=(_CB_IN0, _CB_IN1)[h], data_format=ttnn.uint32, page_size=size
                        )
                    ],
                )
            )
        part = (
            (_CB_PART1, self.grid.x * _SLOT_BYTES),
            (_CB_PART2, ng * _SLOT_BYTES),
            (_CB_OUT, 64 + 4096 + _SLOT_BYTES),
        )
        for idx, size in part:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=crs,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)
                    ],
                )
            )
        # ids 0/1: level-1 post counters of the two halves, 2/3: level-2 (all four, even when one half is idle)
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(4)]
        plan = dict(n=n, ng=ng, tpc=tpc, wt=wt, crs=crs, halves=halves, cbs=cbs, sems=sems)
        self._plans[key] = plan
        return plan

    def local(self, logits):
        """Stage 1: [1,1,32,32] uint32 TILE (DRAM) per device; row r < R: lane 0 = order value of the shard's row max,
        lane 1 = its lowest local column. ``logits`` is not freed."""
        rows, width = int(logits.shape[2]), int(logits.shape[3])
        padded_width = int(logits.padded_shape[3])
        fp32 = logits.dtype == ttnn.float32
        p = self._plan(rows, width, padded_width, fp32)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, _TILE, _TILE]), ttnn.uint32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        accessors = list(ttnn.TensorAccessorArgs(logits).get_compile_time_args()) + list(
            ttnn.TensorAccessorArgs(out).get_compile_time_args()
        )
        common = [logits.buffer_address(), out.buffer_address()] + self.leaders[: 2 * p["ng"]]
        kernels = []
        for h, lo, hi in p["halves"]:
            cta = [
                rows,
                width,
                p["wt"],
                int(fp32),
                int(self.grid.x),
                p["n"],
                p["tpc"],
                lo,
                hi,
                h,
                (_CB_IN0, _CB_IN1)[h],
                _CB_PART1,
                _CB_PART2,
                _CB_OUT,
            ] + accessors
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=LOCAL_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=p["crs"],
                    compile_time_args=cta,
                    common_runtime_args=common,
                    config=ttnn.ReaderConfigDescriptor() if h == 0 else ttnn.WriterConfigDescriptor(),
                )
            )
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=p["sems"], cbs=p["cbs"])
        return ttnn.generic_op([logits, out], desc)

    def pick(self, gathered, rows, mode):
        """Stage 2 on one core: the ids, from the all-gathered stage-1 tiles [1,1,32,32*TP] (not freed)."""
        assert mode in (LANES, ROWS), mode
        nd = self.num_devices
        assert int(gathered.shape[3]) == _TILE * nd, f"gathered width {gathered.shape[3]} != 32 x {nd}"
        shape = [1, 1, 1, _TILE] if mode == LANES else [1, 1, rows, 1]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        out_page = int(out.buffer_aligned_page_size()) if mode == ROWS else 0
        assert mode == LANES or 4 <= out_page <= 64, f"ROWS output page {out_page} B"
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        in_size = nd * 2048 + 64
        out_size = (128 if mode == LANES else rows * 64) + 64
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)],
            )
            for idx, size in ((0, in_size), (1, out_size))
        ]
        cta = [rows, nd, self.shard, mode, 0, 1, out_page]
        cta += list(ttnn.TensorAccessorArgs(gathered).get_compile_time_args())
        cta += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        kernel = ttnn.KernelDescriptor(
            kernel_source=PICK_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=cta,
            common_runtime_args=[gathered.buffer_address(), out.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.generic_op([gathered, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))

    def __call__(self, logits, mode):
        """Global argmax ids of every row of the vocab-sharded ``logits`` (not freed), in the ``mode`` contract."""
        rows = int(logits.shape[2])
        tile = self.local(logits)
        gathered = self.gather(tile)
        ttnn.deallocate(tile)  # a no-op if the gather already freed it
        ids = self.pick(gathered, rows, mode)
        ttnn.deallocate(gathered)
        return ids

    def ids(self, logits, mode):
        """__call__, or None (logged once per reason) when ``logits`` cannot take the fused path: the caller then runs
        the tail it replaces, so the flag never turns an unsupported shape into an error."""
        why = self.supports(logits)
        if why is not None:
            if why not in self._unsupported:
                self._unsupported.add(why)
                logger.warning(f"QWEN36_SPEC_ARGMAX_FUSED: shipped argmax tail kept for this call ({why})")
            return None
        return self(logits, mode)
