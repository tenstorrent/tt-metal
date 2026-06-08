# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# tune-helper `mcast_pipe` style bake-off harness (Blackhole p150a, throwaway-grade).
#
# Drives the matched object-API bake-off kernels (bakeoff_mcast_sender/receiver.cpp) over a
# topology matrix to decide F1 (flush vs barrier), F3 (loopback), F2 (flag vs counter) by
# measurement. Correctness = each receiver's DRAM shard == the broadcast payload, bit-exact.
#
# HARDCODED virtualization offset for THIS machine (Blackhole p150a, COORDINATE_VIRTUALIZATION):
#   worker logical (lx, ly) -> virtual NoC (lx + 1, ly + 2)   [VIRTUAL_TENSIX_START_X/Y]
# Not portable across arches / harvesting layouts — bake-off scaffold only.

import os
import torch
import pytest
import ttnn
from loguru import logger

VIRT_X, VIRT_Y = 1, 2  # Blackhole p150a worker virtualization offset
TILE_BYTES = 32 * 32 * 2  # bf16 tile

KERNEL_DIR = "tests/ttnn/unit_tests/kernel_lib/kernels"


def _defines(fence_barrier, staging_counter, loopback_include, linked):
    return [
        ("FENCE_BARRIER", str(fence_barrier)),
        ("STAGING_COUNTER", str(staging_counter)),
        ("LOOPBACK_INCLUDE", str(loopback_include)),
        ("LINKED", str(linked)),
    ]


# (name, fence_barrier, staging_counter, loopback_include, linked)
VARIANTS = {
    "f1_flush": (0, 0, 0, 0),
    "f1_barrier": (1, 0, 0, 0),
    "f2_flag": (0, 0, 0, 0),
    "f2_counter": (0, 1, 0, 0),
    "f3_exclude": (0, 0, 0, 0),
    "f3_include": (0, 0, 1, 0),
    "f4_linked": (0, 0, 0, 1),
    "f4_unlinked": (1, 0, 0, 0),
}


def _run_pipe(device, variant, recv_rect, sender_logical, payload_tiles, n_iters, pre_handshake):
    """recv_rect = ((rx0,ry0),(rx1,ry1)) logical; sender_logical = (sx,sy) logical."""
    (rx0, ry0), (rx1, ry1) = recv_rect
    sx, sy = sender_logical
    fence_b, stage_c, loop_i, linked = VARIANTS[variant]

    nrx, nry = rx1 - rx0 + 1, ry1 - ry0 + 1
    num_recv = nrx * nry
    sender_in_rect = (rx0 <= sx <= rx1) and (ry0 <= sy <= ry1)
    # num_dests: INCLUDE counts self (only meaningful if sender in rect); EXCLUDE never counts self
    num_dests = num_recv if (loop_i == 0 or not sender_in_rect) else num_recv  # rect cores; self handled by HW

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
    union_crs = (
        recv_crs.merge(sender_crs)
        if hasattr(recv_crs, "merge")
        else ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(rx0, ry0), ttnn.CoreCoord(rx1, ry1)),
                ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx, sy)),
            ]
        )
    )

    # ---- virtual mcast rectangle ----
    vx0, vy0, vx1, vy1 = rx0 + VIRT_X, ry0 + VIRT_Y, rx1 + VIRT_X, ry1 + VIRT_Y
    sender_vx, sender_vy = sx + VIRT_X, sy + VIRT_Y

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
        num_dests,
        payload_pages,
        page_bytes,
        n_iters,
        int(pre_handshake),
    ]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sx][sy] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/bakeoff_mcast_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        defines=_defines(fence_b, stage_c, loop_i, linked),
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
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
        kernel_source=f"{KERNEL_DIR}/bakeoff_mcast_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
        defines=_defines(fence_b, stage_c, loop_i, linked),
        runtime_args=recv_rt,
        config=ttnn.WriterConfigDescriptor(),
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


# ---------- SMOKE: single cell, F1 flush, 1x2 rect, sender out-of-rect, N=1 ----------
def test_smoke_flush(device):
    _run_pipe(
        device,
        variant="f1_flush",
        recv_rect=((0, 0), (0, 1)),
        sender_logical=(1, 1),
        payload_tiles=1,
        n_iters=1,
        pre_handshake=False,
    )


# ---------- PASS 1 coverage: F1 / F2 / F4, sender out-of-rect ----------
# rects chosen to span shapes; sender (5,5) is outside all of them.
RECTS = {
    "1x2": ((0, 0), (0, 1)),
    "1x8": ((0, 0), (0, 7)),
    "4x2": ((0, 0), (3, 1)),
}
SENDER = (5, 5)
COVERAGE_VARIANTS = ["f1_flush", "f1_barrier", "f2_flag", "f2_counter", "f4_linked", "f4_unlinked"]


@pytest.mark.parametrize("variant", COVERAGE_VARIANTS)
@pytest.mark.parametrize("rect_name", list(RECTS.keys()))
@pytest.mark.parametrize("n_iters", [1, 8])
@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_coverage(device, variant, rect_name, n_iters, payload_tiles):
    # pre_handshake stays False (use-case knob, settled on paper); this is what stresses the
    # flag-vs-counter stale-retrigger hazard at N>1.
    _run_pipe(
        device,
        variant=variant,
        recv_rect=RECTS[rect_name],
        sender_logical=SENDER,
        payload_tiles=payload_tiles,
        n_iters=n_iters,
        pre_handshake=False,
    )


# ================== F3 bake-off: sender IN rect, INCLUDE_SRC vs EXCLUDE_SRC + local self-copy ==================
# Clean setup (no same-core two-kernel hang): the sender (column corner) runs ONLY the F3 sender
# kernel and writes its OWN output shard; the other column cores run the plain receiver kernel.
def _run_f3(device, mode_include, rect_len, payload_tiles, n_iters):
    """1xrect_len column rect at x=0; sender = (0,0); receivers = (0,1)..(0,rect_len-1)."""
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    R = rect_len
    num_dests = R if mode_include else (R - 1)

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
    recv_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, R - 1))])

    vx0, vy0, vx1, vy1 = 0 + VIRT_X, 0 + VIRT_Y, 0 + VIRT_X, (R - 1) + VIRT_Y
    sender_vx, sender_vy = 0 + VIRT_X, 0 + VIRT_Y

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
    f3_defines = [("LOOPBACK_INCLUDE", "1" if mode_include else "0")]

    # sender kernel (writes its own shard 0)
    sender_ct = [cb_src, cb_dst, DATA_READY, num_dests, payload_pages, page_bytes, n_iters]
    sender_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    sender_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[0][0] = [input_tensor.buffer_address(), 0, vx0, vy0, vx1, vy1, output_tensor.buffer_address(), 0]
    sender_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/bakeoff_f3_sender.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        defines=f3_defines,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # receiver kernel on the other column cores (shards 1..R-1)
    recv_ct = [cb_dst, DATA_READY, CONSUMED, payload_pages, page_bytes, n_iters, 0]
    recv_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    recv_rt = ttnn.RuntimeArgs()
    for j in range(1, R):
        recv_rt[0][j] = [output_tensor.buffer_address(), j * payload_pages, sender_vx, sender_vy]
    recv_k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/bakeoff_mcast_receiver.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
        defines=[("STAGING_COUNTER", "0")],
        runtime_args=recv_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    pd = ttnn.ProgramDescriptor(kernels=[sender_k, recv_k], semaphores=semaphores, cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)
    torch_out = ttnn.to_torch(output).reshape(R, 1, 32, 32 * payload_tiles)
    for jj in range(R):
        assert torch.equal(
            torch_out[jj].to(torch.float32), payload[0].to(torch.float32)
        ), f"F3 mode_include={mode_include}: shard {jj} mismatch (jj==0 is the sender's own copy)"
    logger.info(f"F3 include={mode_include} R={R} pt={payload_tiles} N={n_iters}: PASS")


@pytest.mark.parametrize("mode_include", [True, False], ids=["include", "excludecopy"])
@pytest.mark.parametrize("payload_tiles", [1, 4, 16])
def test_f3_coverage(device, mode_include, payload_tiles):
    _run_f3(device, mode_include=mode_include, rect_len=4, payload_tiles=payload_tiles, n_iters=1)


@pytest.mark.parametrize("mode_include", [True, False], ids=["include", "excludecopy"])
@pytest.mark.parametrize("payload_tiles", [1, 4, 16, 64], ids=lambda p: f"pt{p}")
def test_f3_perf(device, mode_include, payload_tiles):
    # 8-core column, N=8; sweep payload to find any INCLUDE-vs-EXCLUDE+copy crossover.
    _run_f3(device, mode_include=mode_include, rect_len=8, payload_tiles=payload_tiles, n_iters=8)


# ---------- F3 loopback (OLD, scoped out): sender INSIDE the rect, same-core two-kernel hang ----------
# SCOPED OUT (Step ★ R6): forcing the sender to also be a receiver puts the sender (NCRISC)
# and receiver (BRISC) kernels on the SAME core — the rotating-sender/role-flip hybrid. This
# HANGS with a dispatch/program-construction exception (not a NoC timeout), confirming the
# two-roles-on-one-core shape is out of scope for a single Pipe object (needs the two-Pipe
# refactor). INCLUDE_SRC itself is proven in production (census C1/C2: rms_sender,
# ln_post_allgather). Skipped so it doesn't hang the device on a routine run.
@pytest.mark.skip(reason="R6 same-core sender+receiver hybrid: out of scope (Step ★), hangs device")
def test_f3_loopback_self(device):
    # sender (0,0) is one of the 2 rect cores; INCLUDE_SRC must deliver to self + the other.
    _run_pipe(
        device,
        variant="f3_include",
        recv_rect=((0, 0), (0, 1)),
        sender_logical=(0, 0),
        payload_tiles=1,
        n_iters=1,
        pre_handshake=False,
    )


# ---------- PASS 2 perf: amplifying cell (1x8 rect, N=8, payload 4), one variant per run ----------
# Run under tracy:  python -m tracy -r -m pytest <this>::test_perf -k "<variant>"
@pytest.mark.parametrize("variant", ["f1_flush", "f1_barrier", "f2_flag", "f2_counter", "f4_linked"])
def test_perf(device, variant):
    _run_pipe(
        device,
        variant=variant,
        recv_rect=((0, 0), (0, 7)),  # 1x8 = 8 receivers (barrier waits 8 ACKs)
        sender_logical=SENDER,
        payload_tiles=4,
        n_iters=8,
        pre_handshake=False,
    )
