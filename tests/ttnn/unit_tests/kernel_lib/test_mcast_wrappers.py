# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mcast1D/Mcast2D attachment, receiver routing and source/consumer ordering."""

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    KERNEL_DIR,
    TILE_BYTES,
    core_set,
    make_cb,
    tile_pattern,
    run_wrapper_case,
)


def _run_transfer(
    device,
    *,
    recv_rect,
    sender_logical=(5, 5),
    payload_tiles=1,
    n_iters=1,
    handshake=False,
    noc=0,
    counter=False,
    caller_managed=False,
    control=False,
    control_value=0,
    ack_subset=None,
):
    """Separate sender/receiver kernels, including receivers on the opposite NoC."""
    (x0, y0), (x1, y1) = recv_rect
    receivers = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    size = device.compute_with_storage_grid_size()
    if any(x >= size.x or y >= size.y for x, y in receivers + [sender_logical]):
        pytest.skip("requires a larger worker grid")
    assert sender_logical not in receivers
    if ack_subset is not None:
        assert handshake and not control and n_iters == 1 and 0 < ack_subset < len(receivers)
    signal = ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag
    helper = ttnn.Mcast2D(
        device,
        core_set(receivers),
        ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(*sender_logical)),
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=signal,
            ack_count_override=ack_subset,
        ),
    )
    payload = tile_pattern(payload_tiles).reshape(1, 1, 32, 32 * payload_tiles)
    input_tensor = ttnn.from_torch(payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pages, page_bytes = (1, 32) if control else (payload_tiles, TILE_BYTES)
    output_shape = [len(receivers), 8] if control else [len(receivers), 1, 32, 32 * payload_tiles]
    dtype, layout = (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if control else (ttnn.bfloat16, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dtype, layout, device, ttnn.DRAM_MEMORY_CONFIG
    )
    participants = core_set(receivers + [sender_logical])
    descriptor = ttnn.ProgramDescriptor(
        cbs=[make_cb(i, participants, pages=pages, page_bytes=page_bytes, dtype=dtype) for i in (0, 1)]
    )
    sender_args = ttnn.RuntimeArgs()
    sender_args[sender_logical[0]][sender_logical[1]] = [input_tensor.buffer_address(), 0]
    defines = [("MCAST_TEST_CONTROL", "1")] if control else []
    named = [("control_value", control_value)] if control else []
    sender = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set([sender_logical]),
        compile_time_args=[0, 1, pages, page_bytes, n_iters, int(not caller_managed)]
        + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        named_compile_time_args=named,
        defines=defines,
        runtime_args=sender_args,
        config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
    )
    kernels = [sender]
    # A partial-ACK family owns the channel; passive receivers adopt only its data-ready ID.
    batches = [receivers] if ack_subset is None else [receivers[:ack_subset], receivers[ack_subset:]]
    for batch in batches:
        args = ttnn.RuntimeArgs()
        for x, y in batch:
            args[x][y] = [output_tensor.buffer_address(), receivers.index((x, y)) * pages]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set(batch),
                compile_time_args=[1, pages, page_bytes, n_iters]
                + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()),
                named_compile_time_args=named,
                defines=defines,
                runtime_args=args,
                config=ttnn.ReaderConfigDescriptor() if noc else ttnn.WriterConfigDescriptor(),
            )
        )
    helper.attach(descriptor, "mcast", kernels[:2])
    if ack_subset is not None:
        offset = dict(sender.named_compile_time_args)["mcast_ct_offset"]
        passive = ttnn.Mcast2D(
            device,
            core_set(receivers),
            ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(*sender_logical)),
            ttnn.McastConfig(
                noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
                handshake=False,
                data_ready=signal,
                sem_ids=[sender.compile_time_args[offset + 2]],
            ),
        )
        passive.attach(descriptor, "mcast", kernels[2:])
    descriptor.kernels = kernels
    actual = ttnn.to_torch(ttnn.generic_op([input_tensor, output_tensor], descriptor))
    if control:
        expected = n_iters if counter else control_value or 1
        assert torch.equal(actual.to(torch.int64), torch.full((len(receivers), 8), expected, dtype=torch.int64))
    else:
        for index, core in enumerate(receivers):
            assert torch.equal(actual[index], payload[0]), core


def _run_sender_loopback(device, rect_len, payload_tiles, n_iters):
    """1xrect_len column rect at x=0; sender = (0,0); receivers = (0,1)..(0,rect_len-1)."""
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    R = rect_len
    in_shape = [1, 1, 32, 32 * payload_tiles]
    payload = torch.arange(0, payload_tiles * 1024, dtype=torch.float32).reshape(in_shape).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [R, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]
    full_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, R - 1))])
    sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    has_receivers = R > 1
    mc = ttnn.Mcast2D(
        device,
        full_crs,
        ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(0, 0)),
        ttnn.McastConfig(handshake=False, base_sem_id=0),
    )
    cb_src, cb_dst, cb_result = (0, 1, 16)
    cbs = [
        make_cb(cb_src, full_crs, pages=payload_pages, page_bytes=page_bytes, dtype=ttnn.bfloat16),
        make_cb(cb_dst, full_crs, pages=payload_pages, page_bytes=page_bytes, dtype=ttnn.bfloat16),
        make_cb(cb_result, sender_crs, pages=payload_pages, page_bytes=page_bytes, dtype=ttnn.bfloat16),
    ]
    sender_ct = [cb_src, cb_dst]
    sender_ct += [payload_pages, page_bytes, n_iters, cb_result]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[0][0] = [input_tensor.buffer_address(), 0] + [output_tensor.buffer_address(), 0]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_loopback_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_loopback_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=[cb_dst, cb_result, payload_pages],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )
    kernels = [sender_k, compute_k]
    if has_receivers:
        recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, R - 1))])
        recv_ct = [cb_dst]
        recv_ct += [payload_pages, page_bytes, n_iters]
        recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        recv_rt = ttnn.RuntimeArgs()
        for j in range(1, R):
            recv_rt[0][j] = [output_tensor.buffer_address(), j * payload_pages]
        recv_k = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=recv_crs,
            compile_time_args=recv_ct,
            runtime_args=recv_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels.append(recv_k)
    pd = ttnn.ProgramDescriptor(cbs=cbs)
    mc.attach(pd, "mcast", [sender_k, recv_k] if has_receivers else [sender_k])
    pd.kernels = kernels
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(R, 1, 32, 32 * payload_tiles)
    for jj in range(R):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"Sender loopback: shard {jj} mismatch (jj==0 is the sender's own loopback copy)"
    logger.info(f"Sender loopback R={R} pt={payload_tiles} N={n_iters}: PASS")


def _run_rotating_line(
    device, span, payload_tiles, receiver_span=None, sender_indices=None, data_ready_mode=ttnn.McastDataReady.Flag
):
    receiver_span = span if receiver_span is None else receiver_span
    sender_indices = list(range(span)) if sender_indices is None else sender_indices
    assert len(sender_indices) == span
    N = max(receiver_span, max(sender_indices) + 1)
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(receiver_span - 1, 0))])
    participant_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N - 1, 0))])
    in_shape = [N, 1, 32, 32 * payload_tiles]
    payload = torch.zeros(in_shape, dtype=torch.float32)
    for i in range(N):
        payload[i] = float(i + 1)
    payload = payload.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [N * span, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]
    if sender_indices == list(range(span)) and receiver_span == span:
        mc = ttnn.Mcast1D(
            device,
            receiver_grid,
            ttnn.Mcast1DShape.PerRow,
            ttnn.Mcast1DRotatingSenderConfig(),
            ttnn.McastConfig(data_ready=data_ready_mode),
        )
    else:
        sender_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(sender, 0), ttnn.CoreCoord(sender, 0)) for sender in sender_indices]
        )
        mc = ttnn.Mcast1D(
            device,
            receiver_grid,
            ttnn.Mcast1DShape.PerRow,
            ttnn.Mcast1DRotatingSenderConfig(sender_grid=sender_grid),
            ttnn.McastConfig(data_ready=data_ready_mode),
        )
    cb = 0
    cbs = [make_cb(cb, participant_grid, pages=payload_pages, page_bytes=page_bytes, dtype=ttnn.bfloat16)]
    ct = [cb] + [span, payload_pages, page_bytes]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for X in range(N):
        rt[X][0] = [
            input_tensor.buffer_address(),
            X * payload_pages,
            output_tensor.buffer_address(),
            X * span * payload_pages,
        ]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_rotating_line.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=participant_grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(cbs=cbs)
    mc.attach(pd, "mcast", [k])
    assert k.compile_time_args[dict(k.named_compile_time_args)["mcast_ct_offset"] + 6] == span
    pd.kernels = [k]
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(N * span, 1, 32, 32 * payload_tiles)
    for c in range(N):
        for r, sender in enumerate(sender_indices):
            if c >= receiver_span and c != sender:
                continue
            assert torch.equal(
                torch_out[c * span + r].to(torch.float32), payload[sender].to(torch.float32)
            ), f"core {c} round {r}: expected sender core {sender} (const {sender + 1}) -> coord-order / data-path bug"
    logger.info(
        f"ROTATING-LINE (helper-driven) participants={N} receivers={receiver_span} senders={sender_indices} pt={payload_tiles}: PASS"
    )


def _run_fixed_line(
    device, grid_cols, grid_rows, num_blocks, payload_tiles, *, starting_sender_index=0, sender_placement=None
):
    GC, GR, NB = (grid_cols, grid_rows, num_blocks)
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GC - 1, GR - 1))])
    in_shape = [GR, NB, 32, 32 * payload_tiles]
    payload = torch.zeros(in_shape, dtype=torch.float32)
    for y in range(GR):
        for b in range(NB):
            payload[y, b] = float(y * NB + b + 1)
    payload = payload.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [GC * GR * NB, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]
    mc = ttnn.Mcast1D(
        device,
        grid,
        ttnn.Mcast1DShape.PerRow,
        ttnn.Mcast1DFixedSenderConfig(
            starting_sender_index=starting_sender_index,
            sender_placement=sender_placement if sender_placement is not None else ttnn.Mcast1DSenderPlacement.Uniform,
        ),
        ttnn.McastConfig(),
    )
    cb = 0
    cbs = [make_cb(cb, grid, pages=payload_pages, page_bytes=page_bytes, dtype=ttnn.bfloat16)]
    ct = [cb] + [NB, payload_pages, page_bytes]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for Y in range(GR):
        for X in range(GC):
            rt[X][Y] = [
                input_tensor.buffer_address(),
                Y * NB * payload_pages,
                output_tensor.buffer_address(),
                (Y * GC + X) * NB * payload_pages,
            ]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_fixed_line.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    pd = ttnn.ProgramDescriptor(cbs=cbs)
    mc.attach(pd, "mcast", [k])
    assert (
        k.compile_time_args[dict(k.named_compile_time_args)["mcast_ct_offset"] :][6] == 0
    ), "fixed mode has no rotating span"
    if sender_placement == ttnn.Mcast1DSenderPlacement.Diagonal:
        for Y in range(GR):
            expected_sender = ttnn.CoreCoord((starting_sender_index + Y) % GC, Y)
            assert (
                k.runtime_args[expected_sender.x][expected_sender.y][
                    dict(k.named_compile_time_args)["mcast_rt_offset"] :
                ][-2]
                & 1
            ), f"row {Y}: expected diagonal sender {expected_sender}"
    pd.kernels = [k]
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(GC * GR * NB, 32, 32 * payload_tiles)
    for Y in range(GR):
        for X in range(GC):
            for b in range(NB):
                slot = (Y * GC + X) * NB + b
                assert torch.equal(
                    torch_out[slot].to(torch.float32), payload[Y, b].to(torch.float32)
                ), f"core ({X},{Y}) block {b}: expected const {Y * NB + b + 1} -> rect / sender-coord / data-path bug"
    logger.info(
        f"FIXED-LINE (helper-driven) GC={GC} GR={GR} NB={NB} pt={payload_tiles}: PASS ({GC * GR * NB} slots correct)"
    )


def test_smoke(device):
    _run_transfer(device, recv_rect=((0, 0), (0, 1)), sender_logical=(1, 1))


def test_caller_managed_source_l1(device):
    _run_transfer(device, recv_rect=((0, 0), (0, 1)), n_iters=4, caller_managed=True)


@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize(
    "rect,pages,rounds",
    [
        (((0, 0), (0, 1)), 1, 1),
        (((0, 0), (0, 7)), 4, 8),
        (((0, 0), (3, 1)), 4, 8),
    ],
    ids=["small", "column", "rectangle"],
)
def test_opposite_noc_transfer(device, counter, noc, rect, pages, rounds):
    _run_transfer(device, recv_rect=rect, payload_tiles=pages, n_iters=rounds, noc=noc, counter=counter)


@pytest.mark.parametrize("handshake", [False, True], ids=["unacknowledged", "handshaked"])
@pytest.mark.parametrize("noc", [0, 1])
def test_control_counter_accumulation(device, handshake, noc):
    # No per-round host/caller barrier: receivers may observe accumulated Counter signals.
    _run_transfer(
        device, recv_rect=((0, 0), (0, 7)), control=True, counter=True, n_iters=32, handshake=handshake, noc=noc
    )


@pytest.mark.parametrize("control_value", [0, 2], ids=["default-valid", "ignore-batch"])
def test_control_flag_value(device, control_value):
    _run_transfer(device, recv_rect=((0, 0), (0, 1)), control=True, handshake=True, control_value=control_value)


@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_split_count(device, payload_tiles):
    _run_transfer(device, recv_rect=((0, 0), (0, 3)), payload_tiles=payload_tiles, handshake=True, ack_subset=2)


def test_split_count_across_bh_non_worker_columns(device):
    _run_transfer(device, recv_rect=((0, 0), (8, 0)), handshake=True, ack_subset=2)


@pytest.mark.parametrize("payload_tiles", [1, 4, 16])
def test_sender_loopback(device, payload_tiles):
    _run_sender_loopback(device, rect_len=4, payload_tiles=payload_tiles, n_iters=32)


def test_sender_loopback_local_copy(device):
    _run_sender_loopback(device, rect_len=1, payload_tiles=1, n_iters=1)


def test_rotating_line_smoke(device):
    _run_rotating_line(device, span=2, payload_tiles=1)


def test_rotating_line_counter_smoke(device):
    _run_rotating_line(device, span=2, payload_tiles=1, data_ready_mode=ttnn.McastDataReady.Counter)


def test_rotating_line_outside_sender(device):
    _run_rotating_line(device, span=2, payload_tiles=1, receiver_span=2, sender_indices=[0, 2])


def test_fixed_line_smoke(device):
    _run_fixed_line(device, grid_cols=2, grid_rows=1, num_blocks=2, payload_tiles=1)


def test_fixed_line_diagonal(device):
    _run_fixed_line(
        device,
        grid_cols=4,
        grid_rows=4,
        num_blocks=3,
        payload_tiles=1,
        starting_sender_index=0,
        sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal,
    )


def test_fixed_line_diagonal_wraparound(device):
    _run_fixed_line(
        device,
        grid_cols=8,
        grid_rows=8,
        num_blocks=2,
        payload_tiles=1,
        starting_sender_index=5,
        sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal,
    )


@pytest.mark.parametrize("span,payload_tiles", [(4, 1), (8, 4)])
def test_rotating_line(device, span, payload_tiles):
    _run_rotating_line(device, span=span, payload_tiles=payload_tiles)


@pytest.mark.parametrize("cols,rows,blocks,pages", [(4, 2, 1, 4), (8, 4, 3, 1), (8, 4, 3, 4)])
def test_fixed_line(device, cols, rows, blocks, pages):
    _run_fixed_line(device, grid_cols=cols, grid_rows=rows, num_blocks=blocks, payload_tiles=pages)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("caller_managed", [False, True])
@pytest.mark.parametrize(
    "width,senders,rotating", [(1, [0], False), (2, [0], False), (2, [0, 2], True), (9, [0], False)]
)
def test_alternating_prepared_payload(device, noc, caller_managed, width, senders, rotating):
    run_wrapper_case(device, width=width, senders=senders, rotating=rotating, noc=noc, caller_managed=caller_managed)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize(
    "width,senders,rotating,caller_managed",
    [
        (1, [0], False, True),
        (2, [0], False, False),
        (2, [2], False, True),
        (2, [0, 2], True, True),
    ],
)
def test_prepared_control(device, noc, counter, width, senders, rotating, caller_managed):
    run_wrapper_case(
        device,
        width=width,
        senders=senders,
        rotating=rotating,
        noc=noc,
        counter=counter,
        control=True,
        alternating=False,
        caller_managed=caller_managed,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_prepared_no_handshake(device, noc, counter):
    run_wrapper_case(
        device, width=2, senders=[2], rotating=False, noc=noc, counter=counter, caller_managed=True, handshake=False
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("control", [False, True])
def test_mixed_local_only_sender_turn(device, noc, control):
    run_wrapper_case(device, width=1, senders=[0, 1], rotating=True, noc=noc, control=control, caller_managed=True)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("width", [1, 2])
def test_single_sender_rotating_config(device, noc, counter, control, width):
    run_wrapper_case(device, width=width, senders=[0], rotating=True, noc=noc, counter=counter, control=control)


@pytest.mark.parametrize("kind", ["row", "column"])
@pytest.mark.parametrize("rotating", [False, True])
def test_line_wrapper_shared_kernel(device, kind, rotating):
    run_wrapper_case(device, width=2, senders=[0, 1] if rotating else [0], rotating=rotating, kind=kind)
