# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-core Metal 2.0 ProgramSpec probes: unicast ring, raw mcast, helper mcast, negatives."""

from pathlib import Path

import ttnn
from ttnn.mcast_spec import McastFamily

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

DFB_RECV = "recv"
SCRATCH_STAGE = "stage"
SEM_ARRIVED = "arrived"
SEM_SPACE = "space"
SEM_READY = "ready"
TP_IN = "in"
TP_OUT = "out"
K_MOVER = "mover"
K_SENDER = "sender"
K_RECEIVER = "receiver"
K_WRITER = "writer"


def _row(cols):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, 0))])


def _cores(cols):
    return [ttnn.CoreCoord(x, 0) for x in range(cols)]


def _virt(device, core):
    v = device.worker_core_from_logical_core(core)
    return int(v.x), int(v.y)


def _alloc_like(inp):
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(inp.shape, inp.dtype, ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.INTERLEAVED),
        inp.device(),
    )


# ---------------------------------------------------------------------------- probe 1: unicast ring


def ring_rotate(inp, cols, tiles_per_core, num_entries, *, use_credit=True):
    """out[core i] = in[core i-1]. Each core unicasts its tiles into its successor's recv DFB."""
    device = inp.device()
    out = _alloc_like(inp)
    tile_bytes = inp.buffer_page_size()
    cores = _cores(cols)
    grid = _row(cols)

    mover = ttnn.KernelSpec(
        unique_id=K_MOVER,
        source=str(KERNEL_DIR / "ring_mover.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        scratchpad_bindings=[ttnn.ScratchpadBinding(SCRATCH_STAGE, SCRATCH_STAGE)],
        semaphore_bindings=[
            ttnn.SemaphoreBinding(SEM_ARRIVED, SEM_ARRIVED),
            ttnn.SemaphoreBinding(SEM_SPACE, SEM_SPACE),
        ],
        compile_time_args={"use_credit": int(use_credit)},
        runtime_arg_schema=ttnn.RuntimeArgSchema(
            runtime_arg_names=["first_page", "num_tiles", "next_x", "next_y", "prev_x", "prev_y"]
        ),
    )
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "tile_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )

    spec = ttnn.ProgramSpec(
        name="xc_ring_rotate",
        kernels=[mover, writer],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(
                unique_id=DFB_RECV, entry_size=tile_bytes, num_entries=num_entries, data_format=inp.dtype
            )
        ],
        semaphores=[
            ttnn.SemaphoreSpec(unique_id=SEM_ARRIVED, target_nodes=grid),
            ttnn.SemaphoreSpec(unique_id=SEM_SPACE, target_nodes=grid),
        ],
        scratchpads=[ttnn.ScratchpadSpec(unique_id=SCRATCH_STAGE, size_per_node=tile_bytes)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="ring", kernels=[K_MOVER, K_WRITER], target_nodes=grid)],
    )

    nxt = {c: _virt(device, cores[(i + 1) % cols]) for i, c in enumerate(cores)}
    prv = {c: _virt(device, cores[(i - 1) % cols]) for i, c in enumerate(cores)}
    first = {c: i * tiles_per_core for i, c in enumerate(cores)}

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_MOVER,
                runtime_arg_values={
                    "first_page": first,
                    "num_tiles": {c: tiles_per_core for c in cores},
                    "next_x": {c: nxt[c][0] for c in cores},
                    "next_y": {c: nxt[c][1] for c in cores},
                    "prev_x": {c: prv[c][0] for c in cores},
                    "prev_y": {c: prv[c][1] for c in cores},
                },
            ),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={"first_page": first, "num_tiles": {c: tiles_per_core for c in cores}},
            ),
        ]
    )
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})


# ---------------------------------------------------------------- probe 2: raw mcast, two kernels


def _raw_mcast_spec(inp, out, cols, *, shared_writer, dfb_producers="both", writer_nodes=None):
    """Sender KernelSpec on core 0, receiver KernelSpec on cores 1..N-1, both PRODUCER of `recv`.

    `dfb_producers` selects the negative variants:
      "both"        -> legal: sender covers node 0, receiver covers nodes 1..N-1
      "sender_only" -> the honest cross-node wiring: producer only where the data is read
      "none"        -> no producer at all
    """
    device = inp.device()
    tile_bytes = inp.buffer_page_size()
    cores = _cores(cols)
    grid = _row(cols)
    send_node = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    recv_nodes = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(cols - 1, 0))])

    sender_dfb = [ttnn.producer_of(DFB_RECV, DFB_RECV)] if dfb_producers in ("both", "sender_only") else []
    receiver_dfb = [ttnn.producer_of(DFB_RECV, DFB_RECV)] if dfb_producers == "both" else []

    sender = ttnn.KernelSpec(
        unique_id=K_SENDER,
        source=str(KERNEL_DIR / "mcast_sender.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=sender_dfb,
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        scratchpad_bindings=[ttnn.ScratchpadBinding(SCRATCH_STAGE, SCRATCH_STAGE)],
        semaphore_bindings=[ttnn.SemaphoreBinding(SEM_READY, SEM_READY)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(
            runtime_arg_names=["page", "dests_incl", "dests_excl", "x_start", "y_start", "x_end", "y_end"]
        ),
    )
    receiver = ttnn.KernelSpec(
        unique_id=K_RECEIVER,
        source=str(KERNEL_DIR / "mcast_receiver.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=receiver_dfb,
        semaphore_bindings=[ttnn.SemaphoreBinding(SEM_READY, SEM_READY)],
    )
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "tile_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )

    if shared_writer:
        # `writer` listed in BOTH work units -- does target_nodes compose to the union?
        work_units = [
            ttnn.WorkUnitSpec(name="send", kernels=[K_SENDER, K_WRITER], target_nodes=send_node),
            ttnn.WorkUnitSpec(name="recv", kernels=[K_RECEIVER, K_WRITER], target_nodes=recv_nodes),
        ]
    else:
        work_units = [
            ttnn.WorkUnitSpec(name="send", kernels=[K_SENDER], target_nodes=send_node),
            ttnn.WorkUnitSpec(name="recv", kernels=[K_RECEIVER], target_nodes=recv_nodes),
            ttnn.WorkUnitSpec(name="write", kernels=[K_WRITER], target_nodes=writer_nodes or grid),
        ]

    spec = ttnn.ProgramSpec(
        name="xc_raw_mcast",
        kernels=[sender, receiver, writer],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=DFB_RECV, entry_size=tile_bytes, num_entries=1, data_format=inp.dtype)
        ],
        semaphores=[ttnn.SemaphoreSpec(unique_id=SEM_READY, target_nodes=grid)],
        scratchpads=[ttnn.ScratchpadSpec(unique_id=SCRATCH_STAGE, size_per_node=tile_bytes)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=work_units,
    )

    # Mcast rectangle: the WHOLE row in VIRTUAL coords (loopback data mcast), plus TWO hand-counted
    # fan-outs for that one rectangle -- the data mcast counts the sender, the semaphore mcast does not.
    lo = _virt(device, ttnn.CoreCoord(0, 0))
    hi = _virt(device, ttnn.CoreCoord(cols - 1, 0))
    c0 = ttnn.CoreCoord(0, 0)

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_SENDER,
                runtime_arg_values={
                    "page": {c0: 0},
                    "dests_incl": {c0: cols},
                    "dests_excl": {c0: cols - 1},
                    "x_start": {c0: lo[0]},
                    "y_start": {c0: lo[1]},
                    "x_end": {c0: hi[0]},
                    "y_end": {c0: hi[1]},
                },
            ),
            ttnn.KernelRunArgs(kernel=K_RECEIVER),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={
                    "first_page": {c: i for i, c in enumerate(cores)},
                    "num_tiles": {c: 1 for c in cores},
                },
            ),
        ]
    )
    return spec, run_args


def raw_mcast(inp, cols, *, shared_writer=True):
    out = _alloc_like(inp)
    spec, run_args = _raw_mcast_spec(inp, out, cols, shared_writer=shared_writer)
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})


# ------------------------------------------------------- probe 3: same mcast via McastFamily helper


def family_mcast(inp, cols):
    device = inp.device()
    out = _alloc_like(inp)
    tile_bytes = inp.buffer_page_size()
    cores = _cores(cols)
    grid = _row(cols)

    mcast = McastFamily(
        device,
        grid,
        "bcast",
        shape=ttnn.Mcast1DShape.PerRow,
        sender_index=0,
        config=ttnn.McastConfig(noc=ttnn.NOC.NOC_0),
    )

    mover = ttnn.KernelSpec(
        unique_id=K_MOVER,
        source=str(KERNEL_DIR / "mcast_family_mover.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["page", "is_sender"]),
    )
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "tile_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )

    spec = ttnn.ProgramSpec(
        name="xc_family_mcast",
        kernels=[mover, writer],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=DFB_RECV, entry_size=tile_bytes, num_entries=1, data_format=inp.dtype)
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="row", kernels=[K_MOVER, K_WRITER], target_nodes=grid)],
    )

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_MOVER,
                runtime_arg_values={
                    "page": {c: 0 for c in cores},
                    "is_sender": {c: int(mcast.is_sender(c)) for c in cores},
                },
            ),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={
                    "first_page": {c: i for i, c in enumerate(cores)},
                    "num_tiles": {c: 1 for c in cores},
                },
            ),
        ]
    )
    mcast.attach(spec, run_args, kernels=[K_MOVER], cores=cores)
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})


# ---------------------------------------------------------------------------- negatives (host-side)


def raw_mcast_variant(inp, cols, *, shared_writer, dfb_producers="both", writer_nodes=None):
    out = _alloc_like(inp)
    spec, run_args = _raw_mcast_spec(
        inp, out, cols, shared_writer=shared_writer, dfb_producers=dfb_producers, writer_nodes=writer_nodes
    )
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})


def sem_placement_probe(inp, cols, *, sem_nodes, declare_sem=True):
    """A one-tile local passthrough whose kernels bind a semaphore they only bump locally.

    The semaphore's target_nodes is a knob: does the host check that the nodes binding a semaphore
    are the nodes it was placed on?  Nothing here blocks on the semaphore, so a wrong answer shows
    up as a host error or as nothing at all -- never as a hang.
    """
    out = _alloc_like(inp)
    tile_bytes = inp.buffer_page_size()
    cores = _cores(cols)
    grid = _row(cols)

    reader = ttnn.KernelSpec(
        unique_id=K_MOVER,
        source=str(KERNEL_DIR / "sem_touch_reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        semaphore_bindings=[ttnn.SemaphoreBinding(SEM_READY, SEM_READY)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "tile_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )

    spec = ttnn.ProgramSpec(
        name="xc_sem_placement",
        kernels=[reader, writer],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=DFB_RECV, entry_size=tile_bytes, num_entries=1, data_format=inp.dtype)
        ],
        semaphores=[ttnn.SemaphoreSpec(unique_id=SEM_READY, target_nodes=sem_nodes)] if declare_sem else [],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_MOVER, K_WRITER], target_nodes=grid)],
    )
    pages = {c: i for i, c in enumerate(cores)}
    ones = {c: 1 for c in cores}
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_MOVER, runtime_arg_values={"first_page": pages, "num_tiles": ones}),
            ttnn.KernelRunArgs(kernel=K_WRITER, runtime_arg_values={"first_page": pages, "num_tiles": ones}),
        ]
    )
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})


# ------------------------------------------- probe 6: is a DFB at the same L1 address on every node?

DFB_PAD = "pad"
K_REP_LEAN = "rep_lean"
K_REP_FAT = "rep_fat"
K_WR_LEAN = "wr_lean"
K_WR_FAT = "wr_fat"
REPORT_BYTES = 128


def address_report(device, cols, *, asymmetric):
    """Every node writes its own `recv`/`stage` L1 addresses to its output row.

    asymmetric=True gives node 0 the resource set {recv, stage} and nodes 1.. the set
    {pad, recv, stage}, with `pad` declared FIRST. If DFB placement is per-node, `recv` moves.
    """
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, cols, REPORT_BYTES // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ),
        device,
    )
    cores = _cores(cols)
    grid = _row(cols)
    node0 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rest = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(cols - 1, 0))])

    def reporter(uid, source, dfbs):
        return ttnn.KernelSpec(
            unique_id=uid,
            source=str(KERNEL_DIR / source),
            hw_config=ttnn.create_reader_dm_config(),
            dfb_bindings=[ttnn.producer_of(d, d) for d in dfbs],
            scratchpad_bindings=[ttnn.ScratchpadBinding(SCRATCH_STAGE, SCRATCH_STAGE)],
        )

    def wr(uid, source, dfbs, nodes):
        return ttnn.KernelSpec(
            unique_id=uid,
            source=str(KERNEL_DIR / source),
            hw_config=ttnn.create_writer_dm_config(),
            dfb_bindings=[ttnn.consumer_of(d, d) for d in dfbs],
            tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
            runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["page"]),
        )

    pad_dfb = ttnn.DataflowBufferSpec(
        unique_id=DFB_PAD, entry_size=REPORT_BYTES, num_entries=4, data_format=ttnn.uint32
    )
    recv_dfb = ttnn.DataflowBufferSpec(
        unique_id=DFB_RECV, entry_size=REPORT_BYTES, num_entries=1, data_format=ttnn.uint32
    )

    if asymmetric:
        kernels = [
            reporter(K_REP_LEAN, "addr_report_lean.cpp", [DFB_RECV]),
            reporter(K_REP_FAT, "addr_report_fat.cpp", [DFB_PAD, DFB_RECV]),
            wr(K_WR_LEAN, "addr_writer_lean.cpp", [DFB_RECV], node0),
            wr(K_WR_FAT, "addr_writer_fat.cpp", [DFB_PAD, DFB_RECV], rest),
        ]
        # `pad` is declared FIRST so a per-node allocator would push `recv` down on the fat nodes.
        dfbs = [pad_dfb, recv_dfb]
        work_units = [
            ttnn.WorkUnitSpec(name="lean", kernels=[K_REP_LEAN, K_WR_LEAN], target_nodes=node0),
            ttnn.WorkUnitSpec(name="fat", kernels=[K_REP_FAT, K_WR_FAT], target_nodes=rest),
        ]
        wr_pages = [
            (K_WR_LEAN, {ttnn.CoreCoord(0, 0): 0}),
            (K_WR_FAT, {c: i for i, c in enumerate(cores) if i > 0}),
        ]
    else:
        kernels = [
            reporter(K_REP_LEAN, "addr_report_lean.cpp", [DFB_RECV]),
            wr(K_WR_LEAN, "addr_writer_lean.cpp", [DFB_RECV], grid),
        ]
        dfbs = [recv_dfb]
        work_units = [ttnn.WorkUnitSpec(name="all", kernels=[K_REP_LEAN, K_WR_LEAN], target_nodes=grid)]
        wr_pages = [(K_WR_LEAN, {c: i for i, c in enumerate(cores)})]

    spec = ttnn.ProgramSpec(
        name="xc_address_report",
        kernels=kernels,
        dataflow_buffers=dfbs,
        scratchpads=[ttnn.ScratchpadSpec(unique_id=SCRATCH_STAGE, size_per_node=REPORT_BYTES)],
        tensor_parameters=[ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec)],
        work_units=work_units,
    )
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[ttnn.KernelRunArgs(kernel=k, runtime_arg_values={"page": v}) for k, v in wr_pages]
    )
    # ttnn.generic_op rejects a single-tensor io list ("must contain at least one input tensor and
    # one output tensor"), so a write-only program needs a dummy input it never binds.
    dummy = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.INTERLEAVED
        ),
        device,
    )
    return ttnn.generic_op([dummy, out], spec, run_args, {TP_OUT: 1})


# --------------------------------------------------- probe 7: what the named bindings make impossible


def passthrough(inp, cols, *, writer_accessor=DFB_RECV, declare_dfb=True):
    """Local DRAM->DFB->DRAM passthrough with knobs for the two name-level mistakes."""
    out = _alloc_like(inp)
    tile_bytes = inp.buffer_page_size()
    cores = _cores(cols)
    grid = _row(cols)

    reader = ttnn.KernelSpec(
        unique_id=K_MOVER,
        source=str(KERNEL_DIR / "passthrough_reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_RECV, DFB_RECV)],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "tile_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        # tile_writer.cpp says `dfb::recv`; this is the accessor name the host promises it.
        dfb_bindings=[ttnn.consumer_of(DFB_RECV, writer_accessor)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["first_page", "num_tiles"]),
    )

    spec = ttnn.ProgramSpec(
        name="xc_passthrough",
        kernels=[reader, writer],
        dataflow_buffers=(
            [ttnn.DataflowBufferSpec(unique_id=DFB_RECV, entry_size=tile_bytes, num_entries=2, data_format=inp.dtype)]
            if declare_dfb
            else []
        ),
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_MOVER, K_WRITER], target_nodes=grid)],
    )
    pages = {c: i for i, c in enumerate(cores)}
    ones = {c: 1 for c in cores}
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_MOVER, runtime_arg_values={"first_page": pages, "num_tiles": ones}),
            ttnn.KernelRunArgs(kernel=K_WRITER, runtime_arg_values={"first_page": pages, "num_tiles": ones}),
        ]
    )
    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})
