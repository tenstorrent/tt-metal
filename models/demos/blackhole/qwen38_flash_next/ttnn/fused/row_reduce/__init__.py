# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_reduce``: the per-block output reduce-scatter of a decode row as one program, moving only the valid rows.

Stands in for ``ttnn.reduce_scatter(x, dim=3, cluster_axis=<line>, memory_config=DRAM, topology=Linear)`` (GDN, QSA) and
for the line ``reduce_scatter_minimal_async`` of ``tt_all_reduce`` (MoE) of a local partial ``[1, 1, rows, W]`` TILE
tensor (``rows`` 1..2 valid rows of one padded tile row, bf16; ``rows`` = 1 for fp32) -> ``[1, 1, rows, W / TP_SIZE]``,
device d holding the columns ``[W / TP_SIZE * d, W / TP_SIZE * (d + 1))`` of the sum, in a DRAM interleaved TILE tensor.
Every device sends the valid rows of its whole partial as ONE packet per direction on the line (the packed row mode of
``row_gather``, ``gr_fold/kernels/transport_phase.h``'s transport) into the peers' L1; each device then sums its own
slice from the TP_SIZE partials in the summation tree of the stock line reduce-scatter (``kernels/compute.cpp``), one
rounding per add in the stock op's precision.  Tolerance class BITWISE for the rows < ``rows`` (the other rows of the
output tiles are not meaningful).  ``QWEN38_ROW_REDUCE=1`` switches the program on (default off: ``admits`` is False
and a caller of ``resolve_admitted`` takes the stock op); a partial outside the contract takes the stock op too.
The first call per mesh creates the transport's global semaphores: make it eagerly, before any trace capture.
"""

from __future__ import annotations

import atexit
import os
import sys

import ttnn

from ...contracts import TP_SIZE
from .. import gr_fold
from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "row_reduce"
ENV = "QWEN38_ROW_REDUCE"
PHASE = "rowreduce"
TRANSPORT_KERNEL = fp.kernel_source(NAME, "transport.cpp")
COMPUTE_KERNEL = fp.kernel_source(NAME, "compute.cpp")
CB_IN, PACK_CB, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT = range(8)
LINKS = 1
ROWS_MAX = {ttnn.bfloat16: 2, ttnn.float32: 1}
READ_ALIGN = 64  # the DRAM read alignment of Blackhole: transport.cpp reads each chunk into a 64 B aligned scratch
# MIRROR flips which neighbour direction is the "forward" chain of the tree (the stock op's order, established by
# tests/test_fused_row_reduce_device.py); QWEN38_ROW_REDUCE_MIRROR=1 selects the flipped tree for that test.
# QWEN38_ROW_REDUCE_L1_DIRECT=1 (default off): a partial in L1 is read straight into the staging slot (16 B alignment), no
# bounce of a 32 B chunk through the aligned scratch
L1_DIRECT = os.environ.get("QWEN38_ROW_REDUCE_L1_DIRECT", "0") == "1"
MIRROR = int(os.environ.get("QWEN38_ROW_REDUCE_MIRROR", "0"))
_SEMAPHORES: dict[int, bool] = {}
CALLS: dict[
    str, int
] = {}  # row_reduce() launches (program builds, eager or captured) by calling function, for the A/B logs


def _report_calls() -> None:
    if CALLS:
        print(f"ROW_REDUCE_CALLS total={sum(CALLS.values())} {dict(sorted(CALLS.items()))}", flush=True)
    if WRITE_CALLS["program"] or WRITE_CALLS["fused"]:
        print(
            f"ROW_REDUCE_WRITE_CALLS program_builds={WRITE_CALLS['program']} handed_to_gr_write_hook={WRITE_CALLS['fused']}",
            flush=True,
        )


atexit.register(_report_calls)


def _payload() -> int:
    """The fabric packet payload (bytes, a multiple of 16): the default 4352 B takes 2 packets per direction for one bf16
    row (5,120 B), 3 for two rows or one fp32 row (10,240 B); one packet with a payload >= that (fabric_router_config).
    """

    return int(ttnn.get_tt_fabric_max_payload_size_bytes()) // 16 * 16


def enabled_by_env(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "1") != "0"  # default on since x33 A/B (-0.6 ms)


def version(environ=None) -> int:
    """QWEN38_ROW_REDUCE: 2 = the barrier-free program (``kernels/transport2.cpp``), anything else enabled = v1."""

    return (
        1 if (os.environ if environ is None else environ).get(ENV, "2") == "1" else 2
    )  # default v2 (x41 A/B -0.35 ms)


TRANSPORT2_KERNEL = fp.kernel_source(NAME, "transport2.cpp")
TRANSPORT3_KERNEL = fp.kernel_source(NAME, "transport3.cpp")
COMPUTE3_KERNEL = fp.kernel_source(NAME, "compute3.cpp")
CB_SBF, CB_A, CB_COEF, CB_RES, CB_UPD, CB_NEW = range(8, 14)
HELPER_KERNEL = fp.kernel_source(NAME, "helper.cpp")
SEM_HELP = 3
BRANCHES = 4
HELPER_CORES = tuple(ttnn.CoreCoord(6 + i, 1) for i in range(BRANCHES))  # the gr_write math of branch i
PHASE2 = "rowreduce2"
_STATE2: dict[int, dict] = {}


def prepare2(mesh, tiles: int = 80) -> dict:
    """v2's persistent landing buffer ([call counter 16 B][2 parity sets x TP_SIZE slots of ``tiles`` x 2 x 64 B]) on the
    transport core of every device and its two parity data semaphores, created once per mesh before any trace capture.
    """

    state = _STATE2.get(id(mesh))
    if state is not None:
        if state["tiles"] != tiles:
            raise ValueError(f"row_reduce v2 landing buffer was made for {state['tiles']} input tiles, got {tiles}")
        return state
    grid = mesh.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sem0, sem1 = ttnn.create_global_semaphore(mesh, all_cores, 0), ttnn.create_global_semaphore(mesh, all_cores, 0)
    slot = tiles * 2 * READ_ALIGN
    words = (16 + 2 * TP_SIZE * slot) // 4
    core = gr_fold.TRANSPORT["partials"][0]
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    land = ttnn.zeros([1, words], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=memory)
    state = {
        "tiles": tiles,
        "sems": (sem0, sem1),
        "data1": int(ttnn.get_global_semaphore_address(sem1)),
        "land": land,
        "land_addr": int(land.buffer_address()),
        "slot": slot,
    }
    gr_fold.line_semaphores(mesh).addresses[PHASE2] = (
        int(ttnn.get_global_semaphore_address(sem0)),
        int(ttnn.get_global_semaphore_address(sem0)),
    )
    _STATE2[id(mesh)] = state
    return state


def _row_reduce_v2(partial, out, rows, tiles, out_tiles, chunk, mesh):
    state = prepare2(mesh, tiles)
    tile_bytes = fp.TILE_BYTES[partial.dtype]
    pitch = chunk if (chunk % READ_ALIGN == 0 or fp.in_l1(partial)) else READ_ALIGN
    if out_tiles * 2 * pitch > tile_bytes:
        raise ValueError("row_reduce v2: the dense tile of a slice does not fit a tile")
    total = tiles * 2 * pitch
    payload = _payload()
    transports = gr_fold.TRANSPORT["partials"][:LINKS]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    spec = gr_fold.Transport(
        out,
        partial,
        CB_IN,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE2,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=TRANSPORT2_KERNEL,
        extra_args=(
            rows,
            PACK_CB,
            pitch,
            CB_FS,
            CB_BS,
            CB_F,
            CB_B,
            CB_G,
            CB_OUT,
            out_tiles,
            MIRROR,
            state["land_addr"],
            state["data1"],
            payload,
            state["slot"],
        ),
    )
    dtype = partial.dtype
    cbs = [
        fp.cb_descriptor(CB_IN, dtype, tile_bytes, TP_SIZE, cores),
        fp.cb_descriptor(PACK_CB, dtype, total, 1, cores),
        fp.cb_descriptor(CB_FS, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_BS, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_F, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_B, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_G, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_OUT, dtype, tile_bytes, 1, cores),
    ]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]

    def compute(rank):
        logical = TP_SIZE - 1 - rank if MIRROR else rank
        return [
            fp.compute_kernel(
                COMPUTE_KERNEL,
                cores,
                [CB_IN, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT, 1, TP_SIZE],
                [(transports[0], [logical])],
                fp32_dest=dtype == ttnn.float32,
            )
        ]

    meta = fp.program_meta(
        NAME,
        "row_reduce2",
        rows,
        reads=(partial,),
        writes=(out,),
        l1_bytes=total + (TP_SIZE - 1) * total,
        flops=(TP_SIZE - 1) * out_tiles * fp.TILE * fp.TILE,
        cores=LINKS,
        outputs=((out, 3),),
    )
    fp.run_program(
        [partial, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=compute, links=LINKS),
        meta=meta,
    )
    return fp.stamp_topology(out, partial, 3)


def _geometry(partial):
    """(rows, tiles, out tiles per device, chunk bytes, packed bytes per tile, landing slot bytes, read scratch bytes)."""

    shape = tuple(int(v) for v in partial.shape)
    rows = shape[2]
    tiles = int(partial.padded_shape[3]) // fp.TILE
    tile_bytes = fp.TILE_BYTES[partial.dtype]
    chunk = rows * (tile_bytes // 64)  # rows [0, rows) of one 16-wide face
    packed = 2 * chunk
    slot = -(-(tiles * packed) // 16) * 16
    scratch = tiles * 2 * READ_ALIGN if chunk % READ_ALIGN else 0  # the bounce buffer of a 32 B chunk
    return rows, tiles, tiles // TP_SIZE, chunk, packed, slot, scratch


def admits(partial, *, cluster_axis=1, memory_config=None) -> bool:
    """The contract: ``[1, 1, rows, W]`` TILE on the TP line, bf16 with rows 1..2 or fp32 with rows 1, W / 32 a
    multiple of TP_SIZE, DRAM interleaved output.  The packed rows of the whole partial go out in as many fabric packets as the payload needs.
    """

    if not enabled_by_env():
        return False
    try:
        shape = tuple(int(v) for v in partial.shape)
        padded = tuple(int(v) for v in partial.padded_shape)
        if (
            cluster_axis != gr_fold.TP_AXIS
            or len(shape) != 4
            or shape[:2] != (1, 1)
            or partial.dtype not in ROWS_MAX
            or not 1 <= shape[2] <= ROWS_MAX[partial.dtype]
            or padded[2] != fp.TILE
            or padded[3] % (fp.TILE * TP_SIZE)
            or partial.layout != ttnn.TILE_LAYOUT
            or tuple(int(v) for v in partial.device().shape) != (1, TP_SIZE)
            or (memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG)
        ):
            return False
        gr_fold.line(partial.device())
        return _payload() >= 16
    except Exception:
        return False


def composed(partial, *, cluster_axis=1, memory_config=None):
    return ttnn.reduce_scatter(
        partial,
        dim=3,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
        topology=ttnn.Topology.Linear,
    )


def prepare(mesh) -> None:
    """Create the program's global semaphores (before any trace capture); v2: also its persistent landing buffer."""

    _semaphores(mesh)
    if version() == 2:
        prepare2(mesh, 80)


def _semaphores(mesh) -> None:
    """The (barrier, data) global semaphore pair of this program's own phase, registered in the line transport's table
    (``gr_fold.LineSemaphores``: ``transport_mesh_program`` looks the pair up by phase name)."""

    table = gr_fold.line_semaphores(mesh)
    if PHASE in table.addresses:
        return
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    barrier, data = ttnn.create_global_semaphore(mesh, cores, 0), ttnn.create_global_semaphore(mesh, cores, 0)
    table.handles[PHASE] = (barrier, data)
    table.addresses[PHASE] = (
        int(ttnn.get_global_semaphore_address(barrier)),
        int(ttnn.get_global_semaphore_address(data)),
    )


def row_reduce(partial, *, cluster_axis=1, memory_config=None):
    shape = tuple(int(v) for v in partial.shape)
    if not admits(partial, cluster_axis=cluster_axis, memory_config=memory_config):
        raise ValueError(f"row_reduce does not admit {shape} {partial.dtype} {partial.layout}")
    frame = sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_name in ("row_reduce_or_reduce_scatter", "_row_reduce_call"):
        frame = frame.f_back
    caller = frame.f_code.co_name
    CALLS[caller] = CALLS.get(caller, 0) + 1
    mesh = partial.device()
    rows, tiles, out_tiles, chunk, packed, slot, scratch = _geometry(partial)
    tile_bytes = fp.TILE_BYTES[partial.dtype]
    out = fp.allocate((1, 1, rows, shape[3] // TP_SIZE), partial.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    if version() == 2:
        state = PENDING["state"]
        if state is not None and _write_admits(partial, state):
            PENDING["state"] = None
            new_residual = _row_reduce_write(partial, out, rows, tiles, out_tiles, chunk, mesh, state)
            WRITTEN[id(out)] = (out, state, new_residual)
            return out
        return _row_reduce_v2(partial, out, rows, tiles, out_tiles, chunk, mesh)
    _semaphores(mesh)
    transports = gr_fold.TRANSPORT["partials"][:LINKS]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    payload = _payload()
    l1_input = int(L1_DIRECT and fp.in_l1(partial))
    spec = gr_fold.Transport(
        out,
        partial,
        CB_IN,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=TRANSPORT_KERNEL,
        extra_args=(
            rows,
            PACK_CB,
            slot,
            CB_FS,
            CB_BS,
            CB_F,
            CB_B,
            CB_G,
            CB_OUT,
            out_tiles,
            MIRROR,
            tiles * packed,
            payload,
            l1_input,
        ),
    )
    dtype = partial.dtype
    cbs = [
        fp.cb_descriptor(CB_IN, dtype, tile_bytes, TP_SIZE, cores),
        fp.cb_descriptor(PACK_CB, dtype, (TP_SIZE + 2) * slot + 2 * scratch, 1, cores),
        fp.cb_descriptor(CB_FS, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_BS, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_F, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_B, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_G, dtype, tile_bytes, 2, cores),
        fp.cb_descriptor(CB_OUT, dtype, tile_bytes, 1, cores),
    ]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]

    def compute(rank):
        logical = TP_SIZE - 1 - rank if MIRROR else rank
        return [
            fp.compute_kernel(
                COMPUTE_KERNEL,
                cores,
                [CB_IN, CB_FS, CB_BS, CB_F, CB_B, CB_G, CB_OUT, 1, TP_SIZE],
                [(transports[0], [logical])],
                fp32_dest=dtype == ttnn.float32,
            )
        ]

    meta = fp.program_meta(
        NAME,
        "row_reduce",
        rows,
        reads=(partial,),
        writes=(out,),
        l1_bytes=2 * tiles * packed + (TP_SIZE - 1) * tiles * packed,
        flops=(TP_SIZE - 1) * out_tiles * fp.TILE * fp.TILE,
        cores=LINKS,
        outputs=((out, 3),),
    )
    fp.run_program(
        [partial, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=compute, links=LINKS),
        meta=meta,
    )
    return fp.stamp_topology(out, partial, 3)


def row_reduce_or_reduce_scatter(partial, *, cluster_axis=1, memory_config=None):
    """The call sites' entry: the packed program within its contract, the stock reduce-scatter otherwise."""

    if admits(partial, cluster_axis=cluster_axis, memory_config=memory_config):
        return row_reduce(partial, cluster_axis=cluster_axis, memory_config=memory_config)
    return composed(partial, cluster_axis=cluster_axis, memory_config=memory_config)


register(
    FusedKernel(
        name=NAME,
        replaces="ttnn.reduce_scatter(dim=3, Linear) of the per-block decode output partial (gdn / qsa out-projection, "
        "moe tt_all_reduce), one program with only the valid rows crossing the fabric and the stock summation tree",
        tolerance=BITWISE,
        fused=row_reduce,
        composed=composed,
        gate=None,
        admits=admits,
    )
)


# ------------------------------------------------------------------------------------------------ the GR write folded in
# QWEN38_RR_WRITE=1 (default off, needs the v2 reduce): the block output's reduce also produces the new residual
# ``residual + bf16(block * injection_b)`` that gr_write would (kernels/transport3.cpp + compute3.cpp).  The reduce sites are
# unchanged: GR.read / read_rows leave the matching (residual, injection) state in PENDING (gr.py), the first admitted reduce
# after it takes it, and GR.write / write_rows hand back the program's new residual for that block output (WRITTEN) instead
# of running gr_write.
RR_WRITE_ENV = "QWEN38_RR_WRITE"
PENDING: dict = {"state": None}
WRITTEN: dict = {}  # id(block output) -> (block output, state, new residual)
WRITE_CALLS = {"fused": 0, "program": 0}


def write_enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(RR_WRITE_ENV, "1") == "1" and version(
        environ
    ) == 2  # default on (x50 -0.39 ms)


def set_pending(state) -> None:
    PENDING["state"] = state if write_enabled() else None


def _write_admits(partial, state) -> bool:
    try:
        residual, injection = state.residual, state.injection
        rows = int(partial.shape[2])
        return (
            write_enabled()
            and tuple(int(v) for v in residual.shape) == (1, BRANCHES, rows, int(partial.shape[3]) // TP_SIZE)
            and tuple(int(v) for v in injection.shape) == (1, 1, rows, BRANCHES)
            and partial.dtype
            == ttnn.bfloat16  # an fp32 partial's block reaches GR.write through QSA's typecast: no hand-off key
            and residual.dtype == ttnn.bfloat16
            and injection.dtype == ttnn.bfloat16
            and residual.layout == ttnn.TILE_LAYOUT
            and injection.layout == ttnn.TILE_LAYOUT
            and not fp.in_l1(residual)
            and not fp.in_l1(injection)
            and int(partial.shape[3]) // TP_SIZE // fp.TILE == 10
        )
    except Exception:
        return False


def take_written(block_output, state):
    """GR.write's hook: the new residual the reduce program wrote for ``block_output`` under ``state``, else None."""

    entry = WRITTEN.pop(id(block_output), None)
    if entry is None:
        return None
    block, held, new_residual = entry
    if block is block_output and held is state:
        WRITE_CALLS["fused"] += 1
        return new_residual
    return None


def _row_reduce_write(partial, out, rows, tiles, out_tiles, chunk, mesh, state):
    WRITE_CALLS["program"] += 1
    prep = prepare2(mesh, tiles)
    residual, injection = state.residual, state.injection
    new_residual = fp.allocate(
        (1, BRANCHES, rows, out_tiles * fp.TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    tile_bytes = fp.TILE_BYTES[partial.dtype]
    is_fp32 = partial.dtype == ttnn.float32
    pitch = chunk if (chunk % READ_ALIGN == 0 or fp.in_l1(partial)) else READ_ALIGN
    total = tiles * 2 * pitch
    transports = gr_fold.TRANSPORT["partials"][:LINKS]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    payload = _payload()
    bf = fp.TILE_BYTES[ttnn.bfloat16]
    helper_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in HELPER_CORES])
    every = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in (*transports, *HELPER_CORES)])
    helper_xy = [(w.x, w.y) for w in (mesh.worker_core_from_logical_core(c) for c in HELPER_CORES)]
    spec = gr_fold.Transport(
        out,
        partial,
        CB_IN,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE2,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=TRANSPORT3_KERNEL,
        # the residual / injection / new residual addresses ride in the consumer slots of the runtime phase block
        consumers=(
            (residual.buffer_address(), injection.buffer_address()),
            (new_residual.buffer_address(), 0),
            *helper_xy,
        ),
        extra_args=(
            rows,
            PACK_CB,
            pitch,
            CB_FS,
            CB_BS,
            CB_F,
            CB_B,
            CB_G,
            CB_OUT,
            out_tiles,
            MIRROR,
            prep["land_addr"],
            prep["data1"],
            payload,
            prep["slot"],
            CB_SBF,
            CB_A,
            CB_COEF,
            CB_RES,
            CB_NEW,
            int(is_fp32),
            SEM_HELP,
            *fp.accessor_args(residual),
            *fp.accessor_args(injection),
            *fp.accessor_args(new_residual),
        ),
    )
    dtype = partial.dtype
    # every CB on the reduce core AND the helper cores (same set, same addresses: the block tile is written into the
    # helpers' CB_A from the reduce core)
    cbs = [
        fp.cb_descriptor(CB_IN, dtype, tile_bytes, TP_SIZE, every),
        fp.cb_descriptor(PACK_CB, dtype, total + bf, 1, every),
        fp.cb_descriptor(CB_FS, dtype, tile_bytes, 2, every),
        fp.cb_descriptor(CB_BS, dtype, tile_bytes, 2, every),
        fp.cb_descriptor(CB_F, dtype, tile_bytes, 2, every),
        fp.cb_descriptor(CB_B, dtype, tile_bytes, 2, every),
        fp.cb_descriptor(CB_G, dtype, tile_bytes, 2, every),
        fp.cb_descriptor(CB_OUT, dtype, tile_bytes, 1, every),
        fp.cb_descriptor(CB_SBF, ttnn.bfloat16, bf, 1, every),
        fp.cb_descriptor(CB_A, ttnn.bfloat16, bf, 1, every),
        fp.cb_descriptor(CB_COEF, ttnn.bfloat16, bf, 1, every),
        fp.cb_descriptor(CB_RES, ttnn.bfloat16, bf, 1, every),
        fp.cb_descriptor(CB_UPD, ttnn.bfloat16, bf, 1, every),
        fp.cb_descriptor(CB_NEW, ttnn.bfloat16, bf, 1, every),
    ]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]
    semaphores.append(fp.semaphore_descriptor(SEM_HELP, every))
    acc = [*fp.accessor_args(residual), *fp.accessor_args(injection), *fp.accessor_args(new_residual)]
    addresses = [residual.buffer_address(), injection.buffer_address(), new_residual.buffer_address()]

    def compute(rank):
        logical = TP_SIZE - 1 - rank if MIRROR else rank
        args = [
            CB_IN,
            CB_FS,
            CB_BS,
            CB_F,
            CB_B,
            CB_G,
            CB_OUT,
            1,
            TP_SIZE,
            CB_SBF,
            CB_A,
            CB_COEF,
            CB_RES,
            CB_UPD,
            CB_NEW,
            int(is_fp32),
        ]
        helper_args = [CB_A, CB_COEF, CB_RES, CB_NEW, rows, out_tiles, SEM_HELP, *acc]
        return [
            fp.compute_kernel(
                COMPUTE3_KERNEL,
                cores,
                [*args, 0],
                [(transports[0], [logical])],
                fp32_dest=is_fp32,
                unpack_to_dest_fp32=(CB_OUT,) if is_fp32 else (),
            ),  # the narrowing reads the fp32 sum exactly, as ttnn.typecast
            fp.compute_kernel(
                COMPUTE3_KERNEL, helper_set, [*args, 1], [(c, [0]) for c in HELPER_CORES], fp32_dest=False
            ),
            fp.reader_kernel(
                HELPER_KERNEL, helper_set, helper_args, [(c, [*addresses, i]) for i, c in enumerate(HELPER_CORES)]
            ),
            fp.writer_kernel(
                HELPER_KERNEL, helper_set, helper_args, [(c, [*addresses, i]) for i, c in enumerate(HELPER_CORES)]
            ),
        ]

    meta = fp.program_meta(
        NAME,
        "row_reduce_write",
        rows,
        reads=(partial, residual, injection),
        writes=(out, new_residual),
        l1_bytes=total + (TP_SIZE - 1) * total,
        flops=(TP_SIZE - 1) * fp.TILE * fp.TILE + 2 * BRANCHES * fp.TILE * fp.TILE,
        cores=LINKS,
        outputs=((out, 3),),
    )
    fp.run_program(
        [partial, residual, injection, out, new_residual],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, kernels=compute, links=LINKS),
        meta=meta,
    )
    new_residual.update_tensor_topology(
        residual.tensor_topology()
    )  # as gr_write: generic_op leaves the allocation's placement
    fp.stamp_topology(out, partial, 3)
    return new_residual
