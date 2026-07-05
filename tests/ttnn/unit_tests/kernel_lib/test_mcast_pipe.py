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


# The data-ready signal now rides the mcast WIRE (McastConfig.data_ready), not a kernel #define — the
# kernel's McastArgs reads it out of the flags word, exactly like it reads pre_handshake. LINK is not a
# knob (always linked), so the old "flag_unlinked" variant is gone (identical to flag_linked).
VARIANTS = {
    "flag_linked": ttnn.McastDataReady.Flag,  # canonical clean-spine path: Flag + linked pair + flush
    "counter": ttnn.McastDataReady.Counter,  # Counter signal (atomic-barrier fence)
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
    data_ready_mode = VARIANTS[variant]
    # one reader + one writer; swap which side is which so the sender lands on the requested NoC.
    sender_cfg = ttnn.WriterConfigDescriptor() if sender_noc == 1 else ttnn.ReaderConfigDescriptor()
    recv_cfg = ttnn.ReaderConfigDescriptor() if sender_noc == 1 else ttnn.WriterConfigDescriptor()

    nrx, nry = rx1 - rx0 + 1, ry1 - ry0 + 1
    num_recv = nrx * nry
    # ttnn.Mcast2D owns the fan-out (from the rect area) and the ack count. num_active=0 => the dense
    # default (every rect core acks == the fan-out); a test may override to drive the split-count case.
    num_active = 0 if ack_count is None else ack_count

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

    # ---- mcast helper: owns sems + logical->virtual rect (+ NoC1 corner swap) + per-core args ----
    # Sender is out-of-rect here, so Mcast2D runs in separate-sender mode (fan-out == area). pre_handshake
    # and the data-ready signal now ride the wire (McastConfig.handshake / .data_ready) — no kernel knobs.
    mc = ttnn.Mcast2D(
        device,
        recv_crs,
        ttnn.CoreCoord(sx, sy),
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_1 if sender_noc == 1 else ttnn.NOC.NOC_0,
            base_sem_id=0,
            handshake=pre_handshake,
            data_ready=data_ready_mode,
        ),
        num_active=num_active,
    )

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

    # ---- semaphores: the helper creates data_ready + consumer_ready on the rect ∪ {sender} set ----
    semaphores = mc.owned_semaphores()

    # ---- sender kernel ----
    # CT: [cb_src, cb_dst] + McastArgs block [active, data_ready, consumer_ready, num_active, flags] + scalars.
    # pre_handshake + signal are in the mcast block (flags word) now — no separate pre_handshake CT word.
    sender_ct = [cb_src, cb_dst]
    sender_ct += list(mc.compile_time_args())
    sender_ct += [payload_pages, page_bytes, n_iters]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    # RT: [input_addr, start_id] + the dest rect (virtual, NOC-ordered) the helper emits for the sender.
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0] + list(mc.runtime_args(ttnn.CoreCoord(sx, sy)))
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=sender_cfg,
    )

    # ---- receiver kernel ----
    recv_ct = [cb_dst]
    recv_ct += list(mc.compile_time_args())
    recv_ct += [payload_pages, page_bytes, n_iters]
    recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    recv_rt = ttnn.RuntimeArgs()
    j = 0
    for ry in range(ry0, ry1 + 1):
        for rx in range(rx0, rx1 + 1):
            # RT: [output_addr, shard_start] + the sender coords the helper emits for this receiver.
            recv_rt[rx][ry] = [output_tensor.buffer_address(), j * payload_pages] + list(
                mc.runtime_args(ttnn.CoreCoord(rx, ry))
            )
            j += 1
    recv_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
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
        ack_count=None,  # num_active=0 (dense): ack tracks the runtime fan-out too
    )


# ============ SPLIT COUNT: fan-out (area) > consumer-ack count (the D2 regression) ============
# Models conv-WS / dram-sharded / conv-1D-weights: the mcast box has cores that RECEIVE the broadcast
# but do NOT ack. The data fan-out is the full rect area (4); only a subset (2) acks. The sender must
# wait on the SEPARATE ack count, not the fan-out. A 1x4 box: 2 acking receivers (pre_handshake=true) +
# 2 non-acking (pre_handshake=false). All 4 receive data; the acking config carries num_active=2 as the
# ack count.
#
# pre_handshake rides the mcast WIRE, but ONE Mcast2D object serves the whole family (the layernorm
# idiom: one semantic mcast, each face picks its own handshake per kernel). The flags word's
# pre_handshake bit is chosen at compile_time_args() time: the sender + the acking receivers splice
# mc.compile_time_args() (handshake=True -> pre_handshake set); the non-acking receivers splice
# mc.compile_time_args(pre_handshake=False). All share the one data_ready + consumer_ready pair; only the
# acking cores touch consumer_ready. (No second mcast object for the same semantic mcast.)
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

    # ONE Mcast2D (handshake=True, num_active=2): the sender waits on 2 acks though it broadcasts to all
    # 4 (fan-out == area). Every receiver rides THIS object; the acking vs non-acking split is a per-kernel
    # pre_handshake bit on compile_time_args() below, not a second object.
    mc = ttnn.Mcast2D(device, recv_crs, ttnn.CoreCoord(sx, sy), ttnn.McastConfig(base_sem_id=0), num_active=ack_subset)

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
    # ---- semaphores: helper creates data_ready + consumer_ready on rect ∪ {sender} ----
    semaphores = mc.owned_semaphores()

    # ---- sender: pre_handshake=true (from mc's wire); fan-out=area(4) from the rect, but the wire's
    #      num_active=2 is the ack subset the sender waits on (the D2 split-count regression) ----
    sender_ct = [cb_src, cb_dst]
    sender_ct += list(mc.compile_time_args())
    sender_ct += [payload_pages, page_bytes, 1]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0] + list(mc.runtime_args(ttnn.CoreCoord(sx, sy)))
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- receivers: first `ack_subset` get pre_handshake=true (ack); the rest pre_handshake=false
    #      (receive the data but don't ack) — ONE mc object, the bit chosen per kernel. Same data_ready. ----
    kernels = [sender_k]
    for j in range(num_recv):
        ry = ry0 + j  # column box, so the j-th core is (rx0, ry0+j)
        acks = j < ack_subset
        recv_ct = [cb_dst]
        recv_ct += list(mc.compile_time_args(pre_handshake=acks))
        recv_ct += [payload_pages, page_bytes, 1]
        recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        recv_rt = ttnn.RuntimeArgs()
        recv_rt[rx0][ry] = [output_tensor.buffer_address(), j * payload_pages] + list(
            mc.runtime_args(ttnn.CoreCoord(rx0, ry))
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(rx0, ry), ttnn.CoreCoord(rx0, ry))]),
                compile_time_args=recv_ct,
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
    # Sender is IN the rect (INCLUDE_SRC loopback): Mcast2D runs in fully-inside mode and derives the
    # fan-out from area(R) — excl = R-1 (the other cores), incl = R (+1 for the sender's own self-copy).
    # R==1 is the degenerate self-only case (area==1) the Pipe collapses to a local copy. Handshake is
    # off here (PRE_HANDSHAKE=false), so Mcast2D creates only the data_ready semaphore.

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

    # Mcast2D fully-inside (sender at the column corner, in the rect); no handshake -> data_ready only.
    mc = ttnn.Mcast2D(device, full_crs, ttnn.CoreCoord(0, 0), ttnn.McastConfig(handshake=False, base_sem_id=0))

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
    # ---- semaphores: no handshake -> Mcast2D creates just data_ready, on the rect ----
    semaphores = mc.owned_semaphores()

    # sender kernel (writes its own shard 0)
    # CT: [cb_src, cb_dst] + McastArgs block [active, data_ready, consumer_ready(UNUSED), num_active, flags].
    # handshake=False -> flags pre_handshake bit clear, so the sender/receiver run without the ack.
    sender_ct = [cb_src, cb_dst]
    sender_ct += list(mc.compile_time_args())
    sender_ct += [payload_pages, page_bytes, n_iters]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    # RT: [input_addr, start_id] + the full rect (incl. this sender -> loopback) + [output_addr, self_shard].
    sender_rt[0][0] = (
        [input_tensor.buffer_address(), 0]
        + list(mc.runtime_args(ttnn.CoreCoord(0, 0)))
        + [output_tensor.buffer_address(), 0]
    )
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
        recv_ct = [cb_dst]
        recv_ct += list(mc.compile_time_args())
        recv_ct += [payload_pages, page_bytes, n_iters]
        recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        recv_rt = ttnn.RuntimeArgs()
        for j in range(1, R):
            recv_rt[0][j] = [output_tensor.buffer_address(), j * payload_pages] + list(
                mc.runtime_args(ttnn.CoreCoord(0, j))
            )
        recv_k = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/pipe_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=recv_crs,
            compile_time_args=recv_ct,
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


# ======== ROTATING LINE via the host helper (Mcast1D) + the unified McastArgs decoder ========
# End-to-end proof of the rotating WIRE (not just the pipe's shared-cell mechanics above):
# ttnn.Mcast1D(PerRow, rotating_sender=True) emits the semaphores, CT, and the per-core RT block (full-line rect
# + ordered per-round sender coords); pipe_rotating_line.cpp decodes it with ONE
# McastArgs<1,5,SPAN> (owns both arg lists) and runs the N-core rotating line -- the 1D mirror of block-sharded
# matmul in0. Each core i holds a distinct constant shard (i+1). Over N rounds core r broadcasts its
# shard to the whole line; every core records what it saw per round to DRAM. The check
# output[c*N + r] == shard(r) validates BOTH the data path AND that the receiver indexes the sender
# coords in the RIGHT ORDER -- a mis-ordered coord list would hand core c shard(r' != r).
def _run_rotating_line(device, span, payload_tiles):
    N = span
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles

    # the line = a single row of N cores at y=0
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N - 1, 0))])

    # input: N distinct shards (outer row i = core i's shard, constant i+1: bf16-exact and distinct).
    in_shape = [N, 1, 32, 32 * payload_tiles]
    payload = torch.zeros(in_shape, dtype=torch.float32)
    for i in range(N):
        payload[i] = float(i + 1)
    payload = payload.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # output: N*N shards -> for each core c, N slots (one per round).
    out_shape = [N * N, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    # ---- the host helper owns sems + CT + per-core RT for the rotating line ----
    mc = ttnn.Mcast1D(device, grid, ttnn.Mcast1DShape.PerRow, 0, ttnn.McastConfig(rotating_sender=True))
    assert mc.num_senders() == N, f"expected {N} sender rounds, got {mc.num_senders()}"
    semaphores = mc.owned_semaphores()

    cb = 0  # one CB per core: mcast source (in place) + landing region
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]

    # CT: [cb] + McastArgs<1,5,span> block (5 words) + [span, payload_pages, page_bytes] + TA(in) + TA(out)
    ct = [cb] + list(mc.compile_time_args()) + [N, payload_pages, page_bytes]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # RT: [in_addr, in_start, out_addr, out_start, my_index] + McastArgs<1,5,span> RT block (rect + coords)
    rt = ttnn.RuntimeArgs()
    for X in range(N):
        core = ttnn.CoreCoord(X, 0)
        rt[X][0] = [
            input_tensor.buffer_address(),
            X * payload_pages,
            output_tensor.buffer_address(),
            X * N * payload_pages,
            X,
        ] + list(mc.runtime_args(core))

    k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_rotating_line.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)

    torch_out = ttnn.to_torch(output).reshape(N * N, 1, 32, 32 * payload_tiles)
    for c in range(N):
        for r in range(N):
            assert torch.equal(
                torch_out[c * N + r].to(torch.float32), payload[r].to(torch.float32)
            ), f"core {c} round {r}: expected shard {r} (const {r + 1}) -> coord-order / data-path bug"
    logger.info(f"ROTATING-LINE (helper-driven) N={N} pt={payload_tiles}: PASS ({N * N} slots correct, order verified)")


# Smoke: the smallest line (2 cores, 1 tile, one round each way). Run this FIRST to shake out compile
# / wire errors before the full sweep.
def test_rotating_line_smoke(device):
    _run_rotating_line(device, span=2, payload_tiles=1)


@pytest.mark.parametrize("span", [2, 4, 8])
@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_rotating_line(device, span, payload_tiles):
    _run_rotating_line(device, span=span, payload_tiles=payload_tiles)


# ======== FIXED edge-sender LINE via the host helper (Mcast1D) + the unified McastArgs decoder ========
# The fixed-mode counterpart of test_rotating_line: the 2D dual-mcast matmul in0 shape. On a GC x GR
# grid, ttnn.Mcast1D(PerRow, sender_index=0) makes each ROW an independent per-row family -- the col-0 core
# is the fixed sender, the rest of the row receives. The sender streams `num_blocks` blocks of its row
# (the matmul K-block loop): each block is staged from DRAM and multicast, and every receiver receives
# each block. pipe_fixed_line.cpp decodes the wire with ONE McastArgs<1,5> (owns both arg lists;
# sender() reads its rect off RT, receiver().receive() reads the sender coords off RT) and -- because
# the role is fixed for the whole loop -- builds the pipe ONCE above the block loop. Each (row Y, block b) holds a distinct constant (Y*NB + b + 1);
# the check output[(Y*GC + X)*NB + b] == block(Y, b) proves the data path across the loop AND that the
# helper emits the right per-row rect / sender coords for every row independently.
def _run_fixed_line(device, grid_cols, grid_rows, num_blocks, payload_tiles):
    GC, GR, NB = grid_cols, grid_rows, num_blocks
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GC - 1, GR - 1))])

    # input: GR rows x NB blocks, each a distinct constant (Y*NB + b + 1): bf16-exact & distinct.
    in_shape = [GR, NB, 32, 32 * payload_tiles]
    payload = torch.zeros(in_shape, dtype=torch.float32)
    for y in range(GR):
        for b in range(NB):
            payload[y, b] = float(y * NB + b + 1)
    payload = payload.to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # output: NB slots per core -> core (X,Y) block b writes slot (Y*GC + X)*NB + b.
    out_shape = [GC * GR * NB, 1, 32, 32 * payload_tiles]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    # ---- the host helper owns sems + CT + per-core RT for every per-row family at once ----
    mc = ttnn.Mcast1D(device, grid, ttnn.Mcast1DShape.PerRow, 0, ttnn.McastConfig())  # fixed edge sender
    assert mc.num_senders() == 1, "fixed mode has a single sender per line"
    semaphores = mc.owned_semaphores()

    cb = 0
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        ),
    ]

    # CT: [cb] + McastArgs<1,5> block (5 words) + [num_blocks, payload_pages, page_bytes] + TA(in) + TA(out)
    ct = [cb] + list(mc.compile_time_args()) + [NB, payload_pages, page_bytes]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # RT: [in_addr, in_start, out_addr, out_start, is_sender] + McastArgs<1,5> RT block (sender rect | sender coords)
    rt = ttnn.RuntimeArgs()
    for Y in range(GR):
        for X in range(GC):
            core = ttnn.CoreCoord(X, Y)
            rt[X][Y] = [
                input_tensor.buffer_address(),
                Y * NB * payload_pages,  # row Y's first block (sender only; unused by receivers)
                output_tensor.buffer_address(),
                (Y * GC + X) * NB * payload_pages,
                int(mc.is_sender(core)),
            ] + list(mc.runtime_args(core))

    k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_fixed_line.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=semaphores, cbs=cbs)
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


# Smoke: smallest active fixed line (1 sender + 1 receiver, single row), with a >1 block loop so the
# build-once hoist is actually exercised.
def test_fixed_line_smoke(device):
    _run_fixed_line(device, grid_cols=2, grid_rows=1, num_blocks=2, payload_tiles=1)


@pytest.mark.parametrize("grid_cols,grid_rows", [(2, 1), (4, 2), (8, 4)])
@pytest.mark.parametrize("num_blocks", [1, 3])
@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_fixed_line(device, grid_cols, grid_rows, num_blocks, payload_tiles):
    _run_fixed_line(
        device, grid_cols=grid_cols, grid_rows=grid_rows, num_blocks=num_blocks, payload_tiles=payload_tiles
    )


# ======== RAW PIPE (no host helper): SenderPipe + ReceiverPipe constructed BY HAND in the kernel ========
# Every other test in this file drives the mcast through the host helpers (Mcast1D/Mcast2D own the
# semaphores + logical->virtual coord math + the wire) and decodes it kernel-side with McastArgs. This
# test bypasses ALL of that: it hand-creates the two semaphores, hand-converts logical->virtual coords,
# and hand-wires the CT/RT that the raw kernels (pipe_raw_sender.cpp / pipe_raw_receiver.cpp) read to
# spell SenderPipe / ReceiverPipe themselves. It exercises the pipe primitives in isolation from the
# host arg-emitter -- the layer-1 (pipe mechanics) coverage, restored in the raw style of commit
# 85bd88fc045 (before the host helper existed).
#
# Case: a 1x2 receiver rect with the sender OUT of the rect (plain no-loopback mcast), PRE_HANDSHAKE on
# (so the consumer-ready ack round-trips), Flag data-ready signal, over 2 rounds.
def _run_raw_pipe(device, recv_rect, sender_logical, payload_tiles, n_iters, pre_handshake):
    (rx0, ry0), (rx1, ry1) = recv_rect
    sx, sy = sender_logical
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    num_recv = (rx1 - rx0 + 1) * (ry1 - ry0 + 1)

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

    # ---- BY HAND: logical->virtual mcast rectangle + sender coords (no host helper to do it) ----
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

    # ---- BY HAND: the two semaphores on the union set (init 0; consumer_ready MUST be host-owned) ----
    DATA_READY, CONSUMED = 0, 1
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=union_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=union_crs, initial_value=0),
    ]

    # ---- sender kernel: dense ack (ACK_EQUALS_FANOUT -> the sender waits on the full EXCLUDE fan-out) ----
    sender_ct = [
        cb_src,
        cb_dst,
        DATA_READY,
        CONSUMED,
        ACK_EQUALS_FANOUT,
        payload_pages,
        page_bytes,
        n_iters,
        int(pre_handshake),
    ]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_raw_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- receiver kernel: sender coords go to the ReceiverPipe ctor (RT 2,3) ----
    recv_ct = [cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, n_iters, int(pre_handshake)]
    recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    recv_rt = ttnn.RuntimeArgs()
    j = 0
    for ry in range(ry0, ry1 + 1):
        for rx in range(rx0, rx1 + 1):
            recv_rt[rx][ry] = [output_tensor.buffer_address(), j * payload_pages, sender_vx, sender_vy]
            j += 1
    recv_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_raw_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
        runtime_args=recv_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    pd = ttnn.ProgramDescriptor(kernels=[sender_k, recv_k], semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)

    torch_out = ttnn.to_torch(output).reshape(num_recv, 1, 32, 32 * payload_tiles)
    for jj in range(num_recv):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"raw-pipe: receiver {jj} payload mismatch"
    logger.info(f"RAW-PIPE (no helper) rect={recv_rect} sender={sender_logical} N={n_iters}: PASS")


def test_raw_pipe_no_helper(device):
    _run_raw_pipe(
        device,
        recv_rect=((0, 0), (0, 1)),
        sender_logical=(5, 5),
        payload_tiles=1,
        n_iters=2,
        pre_handshake=True,
    )
