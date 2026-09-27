# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct SenderPipe/ReceiverPipe construction without a host topology or argument decoder."""

import torch
import ttnn
from loguru import logger

TILE_BYTES = 32 * 32 * 2
KERNEL_DIR = "tests/ttnn/unit_tests/kernel_lib/kernels"


def _virt(device, lx, ly):
    core = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
    return core.x, core.y


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

    # ---- sender kernel: explicit receiver count for both fan-out and acknowledgments ----
    sender_ct = [
        cb_src,
        cb_dst,
        DATA_READY,
        CONSUMED,
        num_recv,
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
