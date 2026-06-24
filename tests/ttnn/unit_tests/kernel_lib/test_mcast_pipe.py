# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Unit test for the materialized `Pipe` helper (ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp).
# Ported from the tune-helper bake-off harness (now test_mcast_pipe_bakeoff.py): same program
# shape + coverage matrix, but the mcast+handshake block is driven through Pipe::send/receive
# instead of the raw object-API bake-off kernels.
#
# What changed vs the bake-off:
#   * The fence is BAKED IN to flush — there is no barrier variant to test (the helper has no
#     barrier knob, by design), and linking is always on (no unlinked variant).
#   * Variants exercise the helper's actual axes: the data-ready signal (Flag default | Counter
#     knob) and PRE_HANDSHAKE.
#   * Loopback is NOT a knob: the Pipe infers it at runtime from sender-in-rect. So these tests
#     double as the inference gate —
#       sender out-of-rect (test_coverage/test_smoke) -> Pipe must infer a plain mcast,
#       sender in-rect     (test_f3_loopback)         -> Pipe must infer loopback,
#       rect 1x1 self-only (test_f3_degenerate)       -> area==1 -> Pipe must collapse to a local copy.
#
# Green here == the helper reproduces the bake-off WINNERS' behavior bit-exact, with no hang,
# AND infers the right multicast mode purely from geometry + the active-core count.
#
import torch
import pytest
import ttnn
from loguru import logger

TILE_BYTES = 32 * 32 * 2  # bf16 tile
ACK_EQUALS_FANOUT = 0xFFFFFFFF  # SenderPipe sentinel: consumer-ack count == the EXCLUDE fan-out (dense)


def _virt(device, lx, ly):
    """Logical worker -> virtual NoC coords (firmware-dependent; never hardcode the offset)."""
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
    return c.x, c.y


KERNEL_DIR = "tests/ttnn/unit_tests/kernel_lib/kernels"


def _defines(staging_counter):
    return [
        ("STAGING_COUNTER", str(staging_counter)),
    ]


# (name, staging_counter)  — LINK is no longer a knob (always linked), so the old "flag_unlinked"
# variant is gone (it would be identical to flag_linked).
VARIANTS = {
    "flag_linked": (0,),  # canonical clean-spine path: Flag + linked pair + flush
    "counter": (1,),  # DataReadySignal::Counter knob (atomic-barrier fence)
}


def _run_pipe(
    device, variant, recv_rect, sender_logical, payload_tiles, n_iters, pre_handshake, sender_noc=0, ack_count=None
):
    """recv_rect = ((rx0,ry0),(rx1,ry1)) logical; sender_logical = (sx,sy) logical.
    sender_noc selects which NoC the sender mcasts on: 0 (reader) or 1 (writer). On NoC1 the
    hardware needs start=high-corner / end=low-corner; the test always passes the rect in
    CANONICAL (low->high) order, so a green NoC1 run proves McastRect<NOC_ID> owns the per-NoC corner
    swap (precomputed in its ctor; the old verbatim-passthrough would mis-encode the rect on NoC1)."""
    (rx0, ry0), (rx1, ry1) = recv_rect
    sx, sy = sender_logical
    (stage_c,) = VARIANTS[variant]
    # one reader + one writer; swap which side is which so the sender lands on the requested NoC.
    sender_cfg = ttnn.WriterConfigDescriptor() if sender_noc == 1 else ttnn.ReaderConfigDescriptor()
    recv_cfg = ttnn.ReaderConfigDescriptor() if sender_noc == 1 else ttnn.WriterConfigDescriptor()

    nrx, nry = rx1 - rx0 + 1, ry1 - ry0 + 1
    num_recv = nrx * nry
    # The mcast fan-out is now derived from the rect area inside the helper. The sender passes only the
    # consumer-ack count; ACK_EQUALS_FANOUT means "ack == EXCLUDE fan-out" (here sender is out-of-rect, so
    # that == num_recv, every rect core acks). A test may override to drive the split-count case.
    consumer_ack_count = ACK_EQUALS_FANOUT if ack_count is None else ack_count

    page_bytes = TILE_BYTES
    payload_pages = payload_tiles

    # ---- tensors ----
    in_shape = [1, 1, 32, 32 * payload_tiles]
    payload = torch.arange(0, payload_tiles * 1024, dtype=torch.float32).reshape(in_shape).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [num_recv, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    # ---- core sets ----
    recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(rx0, ry0), ttnn.CoreCoord(rx1, ry1))])
    sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy))])
    union_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(rx0, ry0), ttnn.CoreCoord(rx1, ry1)),
            ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy)),
        ]
    )

    # ---- virtual mcast rectangle ----
    vx0, vy0 = _virt(device, rx0, ry0)
    vx1, vy1 = _virt(device, rx1, ry1)
    sender_vx, sender_vy = _virt(device, sx, sy)

    # ---- CBs (both on union so index->addr map is identical across all cores) ----
    cb_src, cb_dst = 0, 1
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_src, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_dst, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]

    # ---- semaphores ----
    DATA_READY, CONSUMED = 0, 1
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=union_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=union_crs, initial_value=0),
    ]

    # ---- sender kernel ----
    sender_ct = [
        cb_src,
        cb_dst,
        DATA_READY,
        CONSUMED,
        consumer_ack_count,
        payload_pages,
        page_bytes,
        n_iters,
        int(pre_handshake),
    ]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        defines=_defines(stage_c),
        runtime_args=sender_rt,
        config=sender_cfg,
    )

    # ---- receiver kernel ----
    recv_ct = [cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, n_iters, int(pre_handshake)]
    recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    recv_rt = ttnn.RuntimeArgs()
    j = 0
    for ry in range(ry0, ry1 + 1):
        for rx in range(rx0, rx1 + 1):
            recv_rt[rx][ry] = [output_tensor.buffer_address(), j * payload_pages, sender_vx, sender_vy]
            j += 1
    recv_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
        defines=_defines(stage_c),
        runtime_args=recv_rt,
        config=recv_cfg,
    )

    pd = ttnn.ProgramDescriptor(kernels=[sender_k, recv_k], semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)

    torch_out = ttnn.to_torch(output).reshape(num_recv, 1, 32, 32 * payload_tiles)
    ok = True
    for jj in range(num_recv):
        if not torch.equal(torch_out[jj].to(torch.float32), payload[0].to(torch.float32)):
            logger.error(f"receiver {jj} payload mismatch")
            ok = False
    assert ok, f"variant={variant}: not all receiver shards bit-exact"
    logger.info(f"variant={variant} rect={recv_rect} sender={sender_logical} N={n_iters}: PASS")


# ---------- SMOKE: single cell, canonical flag+linked, 1x2 rect, sender out-of-rect ----------
def test_smoke(device):
    _run_pipe(
        device,
        variant="flag_linked",
        recv_rect=((0, 0), (0, 1)),
        sender_logical=(1, 1),
        payload_tiles=1,
        n_iters=1,
        pre_handshake=False,
    )


# ---------- coverage: helper variants, sender out-of-rect ----------
RECTS = {
    "1x2": ((0, 0), (0, 1)),
    "1x8": ((0, 0), (0, 7)),
    "4x2": ((0, 0), (3, 1)),
}
SENDER = (5, 5)
COVERAGE_VARIANTS = ["flag_linked", "counter"]


@pytest.mark.parametrize("variant", COVERAGE_VARIANTS)
@pytest.mark.parametrize("rect_name", list(RECTS.keys()))
@pytest.mark.parametrize("n_iters", [1, 8])
@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_coverage(device, variant, rect_name, n_iters, payload_tiles):
    _run_pipe(
        device,
        variant=variant,
        recv_rect=RECTS[rect_name],
        sender_logical=SENDER,
        payload_tiles=payload_tiles,
        n_iters=n_iters,
        pre_handshake=False,
    )


# ---------- NoC1 corner-ordering: McastRect must own the per-NoC start/end swap ----------
# Sender mcasts on NoC1, where the hardware wants start=high-corner / end=low-corner. The rect is
# passed in CANONICAL (low->high) order regardless of NoC, so a PASS proves the Pipe derives the
# routing-correct ordering from NOC_ID via McastRect<NOC_ID> (corners precomputed in its ctor). With
# the old verbatim-passthrough this would mis-encode the rect on NoC1 (degenerate box -> wrong/hang).
@pytest.mark.parametrize("variant", COVERAGE_VARIANTS)
@pytest.mark.parametrize("rect_name", list(RECTS.keys()))
def test_noc1_sender_corner_order(device, variant, rect_name):
    _run_pipe(
        device,
        variant=variant,
        recv_rect=RECTS[rect_name],
        sender_logical=SENDER,
        payload_tiles=4,
        n_iters=8,
        pre_handshake=False,
        sender_noc=1,
    )


# ---------- PRE_HANDSHAKE confirmation (provisional item: consumer-wait-inside-send vs reused dest) ----------
# Drives the multi-round reused-dest protocol with PRE_HANDSHAKE=true. This is the in-context
# stand-in for "matmul in0": the sender refills its source each round and the consumed-wait inside
# send() must gate the mcast (not the fill). Green confirms the wait-inside-send ordering.
@pytest.mark.parametrize("rect_name", ["1x2", "1x8"])
@pytest.mark.parametrize("n_iters", [1, 8])
def test_pre_handshake(device, rect_name, n_iters):
    _run_pipe(
        device,
        variant="flag_linked",
        recv_rect=RECTS[rect_name],
        sender_logical=SENDER,
        payload_tiles=4,
        n_iters=n_iters,
        pre_handshake=True,
    )


# ========== RUNTIME FAN-OUT: the fan-out is derived from the runtime rect area, not a template arg ==========
# With NUM_ACTIVE_RECEIVER_CORES removed, the sender binary bakes NO recipient count — the fan-out comes
# purely from McastRect::area(), and the rect corners are runtime args. So the SAME compiled kernel must
# broadcast correctly to rects of DIFFERENT areas chosen at runtime. This sweeps three distinct,
# non-power-of-2-friendly areas (2, 8, 8-as-4x2) through one build; a PASS proves the fan-out adapts from
# the runtime area alone (a stale compile-time count would mis-size num_dests and corrupt/hang).
@pytest.mark.parametrize("rect_name", ["1x2", "1x8", "4x2"])
def test_runtime_fanout(device, rect_name):
    _run_pipe(
        device,
        variant="flag_linked",
        recv_rect=RECTS[rect_name],
        sender_logical=SENDER,
        payload_tiles=2,
        n_iters=4,
        pre_handshake=False,
        ack_count=None,  # ACK_EQUALS_FANOUT: ack tracks the runtime fan-out too
    )


# ============ SPLIT COUNT: fan-out (area) > consumer-ack count (the D2 regression) ============
# Models conv-WS / dram-sharded / conv-1D-weights: the mcast box has cores that RECEIVE the broadcast
# but do NOT ack. The data fan-out is the full rect area (4); only a subset (2) acks. The sender must
# wait on the SEPARATE ack count, not the fan-out. A 1x4 box: 2 acking receivers (PRE_HANDSHAKE=true) +
# 2 non-acking (PRE_HANDSHAKE=false). All 4 receive data; the sender passes consumer_ack_count=2.
#
# THIS IS THE REGRESSION the split fixes: with the dense default (ack == fan-out == 4) the sender would
# wait for 4 acks while only 2 arrive -> HANG (the round-1 conv-1D-weights hang, report.md D2). The
# explicit ack=2 makes it pass. n_iters=1 keeps the mixed-handshake level-flag protocol unambiguous.
def _run_split_count(device, payload_tiles):
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    # 1x4 receiver box at x=0; sender out-of-rect at (5,5).
    rx0, ry0, rx1, ry1 = 0, 0, 0, 3
    sx, sy = 5, 5
    area = (rx1 - rx0 + 1) * (ry1 - ry0 + 1)  # 4
    num_recv = area
    ack_subset = 2  # only the first 2 box cores ack; fan-out (4) > ack (2)

    in_shape = [1, 1, 32, 32 * payload_tiles]
    payload = torch.arange(0, payload_tiles * 1024, dtype=torch.float32).reshape(in_shape).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [num_recv, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(rx0, ry0), ttnn.CoreCoord(rx1, ry1))])
    sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy))])
    union_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(rx0, ry0), ttnn.CoreCoord(rx1, ry1)),
            ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy)),
        ]
    )

    vx0, vy0 = _virt(device, rx0, ry0)
    vx1, vy1 = _virt(device, rx1, ry1)
    sender_vx, sender_vy = _virt(device, sx, sy)

    cb_src, cb_dst = 0, 1
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_src, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_dst, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]
    DATA_READY, CONSUMED = 0, 1
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=union_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=union_crs, initial_value=0),
    ]

    # ---- sender: PRE_HANDSHAKE=true, fan-out=area(4) derived from rect, explicit ack=2 ----
    sender_ct = [cb_src, cb_dst, DATA_READY, CONSUMED, ack_subset, payload_pages, page_bytes, 1, int(True)]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        defines=_defines(0),
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- receivers: first `ack_subset` ack (PRE_HANDSHAKE=true); the rest receive but don't ack ----
    kernels = [sender_k]
    for j in range(num_recv):
        ry = ry0 + j  # column box, so the j-th core is (rx0, ry0+j)
        acks = j < ack_subset
        recv_ct = [cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, 1, int(acks)]
        recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        recv_rt = ttnn.RuntimeArgs()
        recv_rt[rx0][ry] = [output_tensor.buffer_address(), j * payload_pages, sender_vx, sender_vy]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(rx0, ry), ttnn.CoreCoord(rx0, ry))]),
                compile_time_args=recv_ct,
                defines=_defines(0),
                runtime_args=recv_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )

    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(num_recv, 1, 32, 32 * payload_tiles)
    for jj in range(num_recv):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"split-count: receiver {jj} mismatch (fan-out={area}, ack={ack_subset})"
    logger.info(
        f"SPLIT-COUNT fan-out={area} ack={ack_subset} pt={payload_tiles}: PASS (data on all {area}, sender waited {ack_subset})"
    )


@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_split_count(device, payload_tiles):
    _run_split_count(device, payload_tiles=payload_tiles)


# ================== F3: sender IN rect, INCLUDE_SRC loopback (bake-off winner) ==================
# Clean setup (no same-core two-kernel hang): the sender (column corner) runs the F3 sender
# kernel and writes its OWN shard; the other column cores run the plain receiver kernel.
def _run_f3(device, rect_len, payload_tiles, n_iters):
    """1xrect_len column rect at x=0; sender = (0,0); receivers = (0,1)..(0,rect_len-1)."""
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    R = rect_len
    # sender is IN the rect (INCLUDE_SRC loopback): the helper derives the fan-out from area(R) — excl =
    # R-1 (the OTHER cores), incl = R (adds +1 for the sender's own self-copy). R==1 is the degenerate
    # self-only case (area==1, excl==0) the Pipe collapses to a local copy. PRE_HANDSHAKE=false here, so
    # this slot is a now-ignored padding arg (kept only to preserve the f3 sender's CT-arg indices).
    pad_unused = R - 1

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
    has_receivers = R > 1  # degenerate (R==1) self-only: no receiver cores, sender does a local copy

    vx0, vy0 = _virt(device, 0, 0)
    vx1, vy1 = _virt(device, 0, R - 1)
    sender_vx, sender_vy = vx0, vy0

    cb_src, cb_dst = 0, 1
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=full_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_src, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=full_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_dst, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]
    DATA_READY, CONSUMED = 0, 1
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=full_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=full_crs, initial_value=0),
    ]

    # sender kernel (writes its own shard 0)
    sender_ct = [cb_src, cb_dst, DATA_READY, pad_unused, payload_pages, page_bytes, n_iters]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[0][0] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1, output_tensor.buffer_address(), 0]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_f3_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    kernels = [sender_k]
    # receiver kernel on the other column cores (shards 1..R-1); none in the degenerate case
    if has_receivers:
        recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, R - 1))])
        recv_ct = [cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, n_iters, 0]
        recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        recv_rt = ttnn.RuntimeArgs()
        for j in range(1, R):
            recv_rt[0][j] = [output_tensor.buffer_address(), j * payload_pages, sender_vx, sender_vy]
        recv_k = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=recv_crs,
            compile_time_args=recv_ct,
            defines=[("STAGING_COUNTER", "0")],
            runtime_args=recv_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels.append(recv_k)

    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(R, 1, 32, 32 * payload_tiles)
    for jj in range(R):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"F3 INCLUDE_SRC: shard {jj} mismatch (jj==0 is the sender's own loopback copy)"
    logger.info(f"F3 INCLUDE_SRC R={R} pt={payload_tiles} N={n_iters}: PASS")


@pytest.mark.parametrize("payload_tiles", [1, 4, 16])
def test_f3_loopback(device, payload_tiles):
    _run_f3(device, rect_len=4, payload_tiles=payload_tiles, n_iters=1)


# Degenerate guard: rect_len==1 => area==1 self-only (excl==0). The Pipe must collapse the
# loopback to a local copy (else the raw loopback hangs). Only the sender core participates.
def test_f3_degenerate(device):
    _run_f3(device, rect_len=1, payload_tiles=1, n_iters=1)


# ============ ROTATING-ROLE regression ============
# Two cores ping-pong over a SINGLE shared data_ready cell: each is a sender on some iters and a
# receiver on others. A core's receiver turn clears the shared cell, so its next sender turn must
# re-assert VALID before broadcasting it -- otherwise it broadcasts a stale INVALID and the partner's
# receive() hangs. Sender is always out of its 1x1 partner rect (no loopback).
def _run_rotating(device, payload_tiles, n_iters):
    assert n_iters % 2 == 0, "ping-pong needs an even iter count"
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    recv_per_core = n_iters // 2
    num_recv_total = 2 * recv_per_core

    # core A = (0,0) role 0 (sends on even iters); core B = (0,1) role 1 (sends on odd iters)
    A, B = (0, 0), (0, 1)

    in_shape = [1, 1, 32, 32 * payload_tiles]
    payload = torch.arange(0, payload_tiles * 1024, dtype=torch.float32).reshape(in_shape).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = [num_recv_total, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    union_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(*A), ttnn.CoreCoord(*A)),
            ttnn.CoreRange(ttnn.CoreCoord(*B), ttnn.CoreCoord(*B)),
        ]
    )

    cb_src, cb_dst = 0, 1
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_src, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=union_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_dst, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]

    # data_ready: host-init 0 (the pipes own its per-side init); consumer_ready: host-init 0 (a
    # remote-incremented counter, MUST be host-owned per the helper contract).
    DATA_READY, CONSUMED = 0, 1
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=union_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=union_crs, initial_value=0),
    ]

    va = _virt(device, *A)
    vb = _virt(device, *B)

    def _kernel_for(role, partner_virt, out_start_pages):
        ct = [cb_src, cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, n_iters]
        ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        core = A if role == 0 else B
        rt[core[0]][core[1]] = [
            input_tensor.buffer_address(),
            0,
            partner_virt[0],
            partner_virt[1],
            output_tensor.buffer_address(),
            out_start_pages,
            role,
        ]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))])
        # role 0 runs on the reader NoC, role 1 on the writer NoC (one kernel per core)
        cfg = ttnn.ReaderConfigDescriptor() if role == 0 else ttnn.WriterConfigDescriptor()
        return ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/pipe_rotating.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )

    # A's partner is B; B's partner is A. A writes its received blocks to DRAM slots starting at 0;
    # B starts after A's region.
    kA = _kernel_for(role=0, partner_virt=vb, out_start_pages=0)
    kB = _kernel_for(role=1, partner_virt=va, out_start_pages=recv_per_core * payload_pages)

    pd = ttnn.ProgramDescriptor(kernels=[kA, kB], semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)

    torch_out = ttnn.to_torch(output).reshape(num_recv_total, 1, 32, 32 * payload_tiles)
    for jj in range(num_recv_total):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"rotating-role: received block {jj} mismatch"
    logger.info(f"ROTATING-ROLE pt={payload_tiles} N={n_iters}: PASS (no hang, all blocks bit-exact)")


# n_iters>=2 is enough: a core's first receiver turn clobbers the shared cell, so its second
# (sender) turn is the first to broadcast a stale flag without M12b. Larger N stresses repetition.
@pytest.mark.parametrize("payload_tiles", [1, 4])
@pytest.mark.parametrize("n_iters", [2, 4, 8])
def test_rotating_role(device, payload_tiles, n_iters):
    _run_rotating(device, payload_tiles=payload_tiles, n_iters=n_iters)
