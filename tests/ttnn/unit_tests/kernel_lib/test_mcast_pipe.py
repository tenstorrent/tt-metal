# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Unit test for the materialized `Pipe` helper (ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp).
# Ported from the tune-helper bake-off harness (now test_mcast_pipe_bakeoff.py): same program
# shape + coverage matrix, but the mcast+handshake block is driven through Pipe::send/receive
# instead of the raw object-API bake-off kernels.
#
# What changed vs the bake-off:
#   * F1 fence is BAKED IN to flush — there is no barrier variant to test (the helper has no
#     barrier knob, by design). The "f1_*"/"f4_unlinked" loser variants are gone.
#   * Variants now exercise the helper's actual axes: STAGING (Flag default | Counter knob),
#     LINK (linked default | unlinked fallback), PRE_HANDSHAKE.
#   * EXCLUDE_SRC vs INCLUDE_SRC is NO LONGER a knob (Round 2): the Pipe infers it at runtime
#     from sender-in-rect. So these tests double as the inference gate —
#       sender out-of-rect (test_coverage/test_smoke) -> Pipe must infer EXCLUDE,
#       sender in-rect     (test_f3_loopback)         -> Pipe must infer INCLUDE_SRC loopback,
#       num_active_cores==1 (test_f3_degenerate)      -> Pipe must collapse to a local copy.
#
# Green here == the helper reproduces the bake-off WINNERS' behavior bit-exact, with no hang,
# AND infers the right multicast mode purely from geometry + the active-core count.
#
import torch
import pytest
import ttnn
from loguru import logger

TILE_BYTES = 32 * 32 * 2  # bf16 tile


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
    "counter": (1,),  # Staging::Counter knob (atomic-barrier fence)
}


def _run_pipe(device, variant, recv_rect, sender_logical, payload_tiles, n_iters, pre_handshake, sender_noc=0):
    """recv_rect = ((rx0,ry0),(rx1,ry1)) logical; sender_logical = (sx,sy) logical.
    sender_noc selects which NoC the sender mcasts on: 0 (reader) or 1 (writer). On NoC1 the
    hardware needs start=high-corner / end=low-corner; the test always passes the rect in
    CANONICAL (low->high) order, so a green NoC1 run proves McastRect.start_end_for_noc() owns
    the per-NoC corner swap (the old verbatim-passthrough would mis-encode the rect on NoC1)."""
    (rx0, ry0), (rx1, ry1) = recv_rect
    sx, sy = sender_logical
    (stage_c,) = VARIANTS[variant]
    # one reader + one writer; swap which side is which so the sender lands on the requested NoC.
    sender_cfg = ttnn.WriterConfigDescriptor() if sender_noc == 1 else ttnn.ReaderConfigDescriptor()
    recv_cfg = ttnn.ReaderConfigDescriptor() if sender_noc == 1 else ttnn.WriterConfigDescriptor()

    nrx, nry = rx1 - rx0 + 1, ry1 - ry0 + 1
    num_recv = nrx * nry
    # sender out-of-rect: every rect core is an active receiver, so active-cores == rect area.
    num_active_cores = num_recv

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
        num_active_cores,
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
# passed in CANONICAL (low->high) order regardless of NoC, so a PASS proves the Pipe re-derives the
# routing-correct ordering from noc_index via McastRect.start_end_for_noc(). With the old
# verbatim-passthrough this would mis-encode the rect on NoC1 (degenerate box -> wrong/hang).
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


# ================== F3: sender IN rect, INCLUDE_SRC loopback (bake-off winner) ==================
# Clean setup (no same-core two-kernel hang): the sender (column corner) runs the F3 sender
# kernel and writes its OWN shard; the other column cores run the plain receiver kernel.
def _run_f3(device, rect_len, payload_tiles, n_iters):
    """1xrect_len column rect at x=0; sender = (0,0); receivers = (0,1)..(0,rect_len-1)."""
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    R = rect_len
    # sender is IN the rect (INCLUDE_SRC loopback); num_active_receiver_cores is the RECIPIENT count
    # = the OTHER R-1 cores (the helper adds +1 for the sender's own self-copy). R==1 is the
    # degenerate self-only case (0 recipients) the Pipe collapses to a local copy.
    num_active_cores = R - 1

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
    sender_ct = [cb_src, cb_dst, DATA_READY, num_active_cores, payload_pages, page_bytes, n_iters]
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


# Degenerate guard: rect_len==1 => num_active_cores==1 self-only. The Pipe must collapse INCLUDE_SRC
# loopback to a local copy (else the raw loopback hangs). Only the sender core participates.
def test_f3_degenerate(device):
    _run_f3(device, rect_len=1, payload_tiles=1, n_iters=1)
