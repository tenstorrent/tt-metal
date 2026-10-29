# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_gather3`` (``QWEN38_GATE_GATHER=1``, default off): the GR gate program and the block-input gather as ONE program.

``gate_gather`` = ``gr_read.gate3.low_rank_gate3`` (the same kernels, cores and math, copied unchanged) whose ten gate worker
cores, besides writing the block tile to its DRAM page as today, hand the tile's valid rows (rows < ``rows`` of faces 0 and 1)
to a staging slot on the gather transport core (6, 0), which runs ``row_gather2``'s barrier-free packet / landing-buffer /
unpack program as extra kernels of the same launch and writes the gathered ``[1, 1, rows, 2560]`` tensor into the consumer's
L1 layout.  The consumer's layout is the GR module's ``gather_memory_config`` (set by the layer); GR.read leaves it in
``CONTEXT`` for the gate dispatch, the gate records (block, gathered) in ``GATHERED`` and ``row_gather2.row_gather`` hands the
gathered tensor to the block's gather call (same memory config) instead of running its program.  Any mismatch falls back to the
ordinary gather.  Bitwise to gate3 + row_gather2 (rows 1, 2).
"""

from __future__ import annotations

import atexit
import os

import ttnn

from ...contracts import TP_SIZE
from .. import gr_fold
from .. import program as fp
from .. import gr_read
from ..gr_read import gate3 as g3
from ..row_gather2 import CHUNK_PITCH, prepare as prepare_gather

NAME = "row_gather3"
ENV = "QWEN38_GATE_GATHER"
PHASE = "rowgather2"  # the gather's semaphore pair and landing buffer are row_gather2's: one parity counter for every gather
TRANSPORT_KERNEL = fp.kernel_source(NAME, "transport.cpp")
WRITER_FWD = fp.kernel_source(NAME, "writer_fwd.cpp")
SEM_GATE, SEM_GO, SEM_SCRATCH, SEM_DONE, SEM_ARRIVE = 0, 1, 2, 3, 4
ROWS_MAX = 2
CONTEXT: dict = {"memory_config": None}
GATHERED: dict = {}  # id(block) -> (block, gathered tensor, memory config)
CALLS = {"fused": 0, "handed": 0}
_STAGE: dict[int, dict] = {}


def _report() -> None:
    if CALLS["fused"] or CALLS["handed"]:
        print(f"GATE_GATHER_CALLS program_builds={CALLS['fused']} handed_to_gather={CALLS['handed']}", flush=True)


atexit.register(_report)


def enabled_by_env(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "0") == "1"


def set_context(memory_config) -> None:
    """GR.read's call: the consumer layout the next gate dispatch should gather into (None: no gather)."""

    CONTEXT["memory_config"] = memory_config if enabled_by_env() else None


def take_context():
    memory_config = CONTEXT["memory_config"]
    CONTEXT["memory_config"] = None
    return memory_config


def take_gathered(hidden, memory_config):
    """row_gather2.row_gather's hook: the gathered tensor the gate program wrote for ``hidden``, else None."""

    entry = GATHERED.pop(id(hidden), None)
    if entry is None:
        return None
    block, gathered, cfg = entry
    if block is hidden and cfg == memory_config:
        CALLS["handed"] += 1
        return gathered
    return None


def prepare(mesh, tiles: int = 10) -> dict:
    """row_gather2's landing buffer + semaphores and the persistent staging slot ([tiles x 2 chunks x 64 B]) on core (6, 0)."""

    state = prepare_gather(mesh, tiles)
    if id(mesh) not in _STAGE:
        core = gr_fold.TRANSPORT["partials"][0]
        words = tiles * 2 * CHUNK_PITCH // 4
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, words], ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        stage = ttnn.zeros(
            [1, words], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=memory
        )
        _STAGE[id(mesh)] = {"tensor": stage, "addr": int(stage.buffer_address())}
    return {**state, "stage_addr": _STAGE[id(mesh)]["addr"]}


def admits(gathered_partials, rows: int, memory_config) -> bool:
    try:
        return (
            enabled_by_env()
            and memory_config is not None
            and 1 <= rows <= ROWS_MAX
            and TP_SIZE == 8
            and g3.enabled(rows)
            and int(g3.HIDDEN_TILES) == 10
        )
    except Exception:
        return False


def gate_gather(gathered_partials, normalized, up, memory_config, *, matmul: str = "chain"):
    """``low_rank_gate3`` + the gather as one program; returns (block, injection) like ``low_rank_gate3``."""

    shape = tuple(gathered_partials.shape)
    if (
        len(shape) != 4
        or shape[:2] != (TP_SIZE, 1)
        or shape[3] != g3.PARTIAL_WIDTH
        or gathered_partials.dtype != g3.FP32
    ):
        raise ValueError(f"GR gathered partials must be fp32 [{TP_SIZE}, 1, rows, {g3.PARTIAL_WIDTH}], got {shape}")
    rows = shape[2]
    g3._expect(normalized, (1, 1, rows, g3.FLAT_WIDTH), g3.BF16, "GR normalized row")
    g3._expect(up, (1, 1, g3.PARTIAL_WIDTH, g3.FLAT_WIDTH), g3.BF16, "GR up")
    mesh = gathered_partials.device()
    CALLS["fused"] += 1
    tiles = int(g3.HIDDEN_TILES)
    prep = prepare(mesh, tiles)
    block = fp.allocate((1, 1, rows, g3.LOCAL_HIDDEN), g3.BF16, ttnn.TILE_LAYOUT, mesh)
    injection = fp.allocate((1, 1, rows, g3.BRANCHES), g3.BF16, ttnn.TILE_LAYOUT, mesh)
    gathered = fp.allocate((1, 1, rows, g3.LOCAL_HIDDEN * TP_SIZE), g3.BF16, ttnn.TILE_LAYOUT, mesh, memory_config)

    # ---- the gate program of gate3.low_rank_gate3, unchanged except the writer (writer_fwd.cpp) --------------------------------
    t = g3.LOW_RANK_TILES_PER_CORE
    n_producers = g3.PARTIAL_TILES // t
    inject_core, inject_local = divmod(g3.INJECT_TILE, t)
    n_tiles = g3._tiles_wide(up)
    workers, producers, rect = g3._rectangle(mesh, g3.GATE_WIDTH, g3.GATE_HEIGHT, rows_above=n_producers, wrap=False)
    transports = gr_fold.TRANSPORT["partials"][:1]
    if any((w.x, w.y) == (transports[0].x, transports[0].y) for w in workers + producers):
        raise RuntimeError("the gather transport core is inside the gate's cores")
    plain = [c for i, c in enumerate(producers) if i != inject_core]
    inject = [producers[inject_core]]
    w_set, p_set, all_set = g3._core_set(workers), g3._core_set(producers), g3._core_set(workers + producers)
    plain_set, inject_set = g3._core_set(plain), g3._core_set(inject)
    tcores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    BF16, FP32, TILE_BF16, TILE_FP32, BRANCHES = g3.BF16, g3.FP32, g3.TILE_BF16, g3.TILE_FP32, g3.BRANCHES
    PARTIAL_TILES = g3.PARTIAL_TILES
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, t * TP_SIZE, p_set),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, p_set),
        fp.cb_descriptor(2, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(17, BF16, TILE_BF16, 1, inject_set),
        fp.cb_descriptor(8, BF16, TILE_BF16, PARTIAL_TILES, all_set),
        fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, w_set),
        fp.cb_descriptor(10, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(11, BF16, TILE_BF16, 1, w_set),
        fp.cb_descriptor(12, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(13, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(14, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(15, FP32, TILE_FP32, BRANCHES, w_set),
        fp.cb_descriptor(18, BF16, TILE_BF16, 1, w_set),
    ]
    tw = mesh.worker_core_from_logical_core(transports[0])

    def gate_kernels(rank):
        reader = g3._reader(
            p_set,
            [(gathered_partials, 0)],
            [(g3.CONST_ZERO, 1)],
            [
                (core, ([g3._stream(gathered_partials, t, TP_SIZE, c * t, PARTIAL_TILES, 1, t * TP_SIZE)], [0]))
                for c, core in enumerate(producers)
            ],
        )
        computes = [
            fp.compute_kernel(g3.LOWRANK3, plain_set, [t, TP_SIZE, t, 16, 17], fp32_dest=True),
            fp.compute_kernel(g3.LOWRANK3, inject_set, [t, TP_SIZE, inject_local, 16, 17], fp32_dest=True),
        ]
        senders = [
            g3._mcast_writer(
                plain_set,
                [(core, (c * t, rect, (0, 1), (0, 0, 1, 1))) for c, core in enumerate(producers) if c != inject_core],
                src_cb=16,
                dst_cb=8,
                tiles=t,
                extra=(injection, g3.NONE_CB),
            ),
            g3._mcast_writer(
                inject_set,
                [(producers[inject_core], (inject_core * t, rect, (0, 1), (1, 0, 1, 1)))],
                src_cb=16,
                dst_cb=8,
                tiles=t,
                extra=(injection, 17),
            ),
        ]
        receiver = g3._mcast_reader(
            w_set,
            [(normalized, 10), (up, 9)],
            [
                (
                    core,
                    [
                        g3._stream(normalized, 1, BRANCHES, j, g3.HIDDEN_TILES, 0, BRANCHES),
                        g3._stream(up, BRANCHES, PARTIAL_TILES, j, n_tiles, g3.HIDDEN_TILES, BRANCHES * PARTIAL_TILES),
                    ],
                )
                for j, core in enumerate(workers)
            ],
            recv_cb=8,
            recv_tiles=PARTIAL_TILES,
            senders=n_producers,
            zero_cb=11,
        )
        gate_k = fp.compute_kernel(
            g3.GATE3,
            w_set,
            [PARTIAL_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, g3.UP_SPILL if matmul == "chain" else 0, 15],
            fp32_dest=True,
            unpack_to_dest_fp32=(10, 12, 13),
        )
        # the gate's writer with the hand-off: writer.cpp's stream plus the chunks to the staging slot and the arrival step
        compile_args = [1, 18, 0, 0]
        for _ in range(3):
            compile_args += fp.accessor_args(block)
        compile_args += [rows, SEM_ARRIVE]
        writer = fp.writer_kernel(
            WRITER_FWD,
            w_set,
            compile_args,
            [
                (core, [block.buffer_address(), 1, j, 1, 1, tw.x, tw.y, prep["stage_addr"]])
                for j, core in enumerate(workers)
            ],
        )
        return [reader, *computes, *senders, receiver, gate_k, writer]

    # ---- the gather transport as extra kernels on core (6, 0) ------------------------------------------------------------------
    spec = gr_fold.Transport(
        gathered,
        block,
        0,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE,
        transports,
        semaphore_ids=(SEM_GO, SEM_SCRATCH, SEM_DONE),
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=0,
        kernel=TRANSPORT_KERNEL,
        extra_args=(rows, prep["land_addr"], prep["data1"], fp.TILE_BYTES[g3.BF16], prep["stage_addr"], SEM_ARRIVE),
    )
    semaphores = [fp.semaphore_descriptor(SEM_GATE, all_set)] + [
        fp.semaphore_descriptor(s, tcores) for s in (SEM_GO, SEM_SCRATCH, SEM_DONE)
    ]
    semaphores.append(fp.semaphore_descriptor(SEM_ARRIVE, ttnn.CoreRangeSet([*tcores.ranges(), *w_set.ranges()])))
    meta = fp.program_meta(
        NAME,
        "gate_gather",
        rows,
        reads=(gathered_partials, normalized, up),
        writes=(block, injection),
        l1_bytes=len(workers) * PARTIAL_TILES * TILE_BF16 + TP_SIZE * tiles * 2 * CHUNK_PITCH,
        flops=(TP_SIZE + 3) * rows * g3.PARTIAL_WIDTH
        + 2 * rows * g3.PARTIAL_WIDTH * g3.FLAT_WIDTH
        + 3 * rows * g3.FLAT_WIDTH,
        cores=len(workers) + len(producers) + 1,
        outputs=((block, 3), (injection, None)),
    )
    fp.run_program(
        [gathered_partials, normalized, up, block, injection, gathered],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=gate_kernels, links=1),
        meta=meta,
    )
    fp.stamp_topology(gathered, block)
    GATHERED[id(block)] = (block, gathered, memory_config)
    return block, injection
