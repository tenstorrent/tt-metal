# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fabric mux throughput: NC client cores of chip (0, 0) of a 2 x 4 tray send NPKT packets of PB bytes each through one
fabric mux (one link) into the same cores of chip (0, 1), a fused write + atomic inc a packet. The receivers stamp their
first and last arrival; GB/s = all bytes over the receiving chip's span. WS_WARM (default 20) warm-up + WS_MEAS
(default 5) measured launches a case."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
fabric_mux = ttnn.experimental.fabric_mux


def launch(mesh, NC, nbuf, NPKT, PB, flush, mux=(6, 0)):
    cores = [(c, y) for c in (0, 1) for y in range(10)][:NC]
    words = lambda n: sharded_l1(mesh, cores, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    slots = words(2 * PB // 16)  # source on the sender, the 2 landing slots on the receiver (same address)
    cnt = words(1)
    st = words(4)
    cfg = fabric_mux.Config(
        num_full_size_channels=NC,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=nbuf,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    ctype = fabric_mux.ChannelType.FULL_SIZE
    dst = node(mesh, (0, 1))
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    cl_rt, mx_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for x, y in cores:
        xy = noc(mesh, x, y)
        cl_rt[x][y] = [mesh_id, int(dst.chip_id), xy[0] | xy[1] << 8, slots.buffer_address(), st.buffer_address()]
        rv_rt[x][y] = [st.buffer_address()]
    mx_rt[mux[0]][mux[1]] = []
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    client = ttnn.KernelDescriptor(
        kernel_source=kpath("mux_bench_client.cpp"),
        core_ranges=crs(cores),
        compile_time_args=[NPKT, PB, slots.buffer_address(), cnt.buffer_address(), int(flush)]
        + list(fabric_mux.client_compile_time_args(num_clients=NC, channel_type=ctype, config=cfg)),
        runtime_args=cl_rt,
        defines=FAB_DEFS,
        config=dm(1),
    )
    muxk = ttnn.KernelDescriptor(
        kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        core_ranges=crs([mux]),
        compile_time_args=list(cfg.kernel_compile_time_args()),
        runtime_args=mx_rt,
        defines=FAB_DEFS,
        config=dm(0),
        opt_level=fabric_mux.KernelBuildOptLevel.O3,
    )
    p = ttnn.ProgramDescriptor(kernels=[client, muxk], semaphores=[], cbs=[])
    p.kernels[1].runtime_args[mux[0]][mux[1]].extend(
        cfg.kernel_runtime_args(
            source_node_id=node(mesh, (0, 0)),
            destination_node_id=dst,
            link_index=0,
            program_descriptor=p,
            mux_logical_core=ttnn.CoreCoord(*mux),
        )
    )
    mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*mux))
    master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*cores[0]))
    term_sem = None
    for i, (x, y) in enumerate(cores):
        args = fabric_mux.client_runtime_args(
            connection_valid=True,
            is_termination_master=i == 0,
            channel_type=ctype,
            mux_virtual_core=mux_vc,
            client_index=i,
            client_logical_core=ttnn.CoreCoord(x, y),
            config=cfg,
            program_descriptor=p,
            termination_master_virtual_core=master_vc,
            termination_master_semaphore_id=term_sem,
        )
        if i == 0:
            term_sem = args[10]
        p.kernels[0].runtime_args[x][y].extend(args)
    recv = ttnn.KernelDescriptor(
        kernel_source=kpath("mux_bench_recv.cpp"),
        core_ranges=crs(cores),
        compile_time_args=[NPKT, cnt.buffer_address()],
        runtime_args=rv_rt,
        config=dm(0),
    )
    q = ttnn.ProgramDescriptor(kernels=[recv], semaphores=[], cbs=[])
    mpd = ttnn.MeshProgramDescriptor()
    for coord, prog in (((0, 0), p), ((0, 1), q)):
        mc = ttnn.MeshCoordinate(*coord)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
    ttnn.synchronize_device(mesh)
    ttnn.generic_op([slots, cnt, st], mpd)
    ttnn.synchronize_device(mesh)
    sts = ttnn.get_device_tensors(st)
    s_recv = (ttnn.to_torch(sts[1]).to(torch.int64) & 0xFFFFFFFF).reshape(NC, 16)
    s_send = (ttnn.to_torch(sts[0]).to(torch.int64) & 0xFFFFFFFF).reshape(NC, 16)
    span = int(s_recv[:, 3].max() - s_recv[:, 2].min())
    send = int((s_send[:, 1] - s_send[:, 0]).max())
    return span, send


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize(
    "NC,nbuf,flush", [(20, 2, 1), (20, 3, 1), (20, 2, 0), (10, 4, 1), (4, 4, 1), (1, 4, 1)],
    ids=["c20b2", "c20b3", "c20b2_noflush", "c10b4", "c4b4", "c1b4"],
)
def test_mux_bench(mesh_device, NC, nbuf, flush):
    PB, NPKT = 24 * 576, 300
    clk = aiclk_mhz(0)
    rows = []
    for i in range(WARM + MEAS):
        span, send = launch(mesh_device, NC, nbuf, NPKT, PB, flush)
        if i >= WARM:
            rows.append((span, send))
    gbs = [NC * NPKT * PB / (r[0] / clk * 1e3) for r in rows]
    logger.info(
        f"MUX {NC} clients, {nbuf} buffers a channel, flush {flush}: {WARM} warm-up + {MEAS} measured, "
        f"{statistics.median(gbs):.1f} GB/s median (min {min(gbs):.1f}, max {max(gbs):.1f}) into the next chip, "
        f"{NC * NPKT} packets of {PB} B; longest client send loop {statistics.median(r[1] for r in rows) / clk:.0f} us"
    )
