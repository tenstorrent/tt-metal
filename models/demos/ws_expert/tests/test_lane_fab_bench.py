# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Weights-only fabric bench on the lane design's link pattern (ws_lane.py on the 2 x 4 tray): the same group cores,
mux cores, cycle and links as the lane, every chip at once, each group core sending NPKT packets of 24 bfp4 tiles
(13,824 B) to the same core of the next chip in each active direction (kernels/lane_fab_client.cpp: the lane's
non-blocking send, fused write + atomic inc), without the lane's schedule, slots, credits or cut-through forwarding.
GB/s a link direction = 20 x NPKT x PB over the span from the first to the last arrival at the receiving group's cores.
WS_WARM (default 5) warm-up + WS_MEAS (default 5) measured launches a case."""

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
from models.demos.ws_expert.ws_lane import LCfg, lane_cores, lanes, mux_core

WARM = int(os.environ.get("WS_WARM", "5"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
fabric_mux = ttnn.experimental.fabric_mux
PB = 24 * 576


def launch(mesh, groups_on, dmask, NPKT, flush, nbuf=2, PB=PB, inc=1):
    """One launch; returns per (chip, group, direction) the receiving span (cycles) and per chip the longest send loop
    and the mean full-mux polls a core."""
    R, Cn = tuple(mesh.shape)
    G = 2
    ln = lanes((R, Cn), LCfg())
    groups = {g: lane_cores(g, G) for g in groups_on}
    C = [c for g in groups_on for c in groups[g]]
    muxes = [mux_core(g, d, G) for g in groups_on for d in range(2)]
    words = lambda n: sharded_l1(mesh, C, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    src = words(PB // 16)
    slots = words(4 * PB // 16)
    cnt = words(2)
    st = words(4)
    mcfg = fabric_mux.Config(
        num_full_size_channels=20,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=nbuf,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    ctype = fabric_mux.ChannelType.FULL_SIZE
    client_ct = list(fabric_mux.client_compile_time_args(num_clients=20, channel_type=ctype, config=mcfg))
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc = (r, c)
            nxt = {}
            for g in groups_on:
                cyc = ln[g][0]
                i = cyc.index(me_rc)
                nxt[g] = (cyc[(i + 1) % len(cyc)], cyc[(i - 1) % len(cyc)])
            kernels = []
            cl_idx = {}
            for g in groups_on:
                rt, rv = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for x, y in groups[g]:
                    xy = noc(mesh, x, y)
                    rt[x][y] = [
                        mesh_id,
                        int(node(mesh, nxt[g][0]).chip_id),
                        int(node(mesh, nxt[g][1]).chip_id),
                        xy[0] | xy[1] << 8,
                        st.buffer_address(),
                    ]
                    rv[x][y] = [st.buffer_address()]
                cl_idx[g] = len(kernels)
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("lane_fab_client.cpp"),
                        core_ranges=crs(groups[g]),
                        compile_time_args=[
                            NPKT,
                            PB,
                            slots.buffer_address(),
                            cnt.buffer_address(),
                            int(flush),
                            dmask,
                            src.buffer_address(),
                            inc,
                        ]
                        + client_ct,
                        runtime_args=rt,
                        defines=FAB_DEFS,
                        config=dm(1),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("lane_fab_recv.cpp"),
                        core_ranges=crs(groups[g]),
                        compile_time_args=[NPKT // inc, cnt.buffer_address(), dmask],
                        runtime_args=rv,
                        config=dm(0),
                    )
                )
            mux_idx = len(kernels)
            mrt = ttnn.RuntimeArgs()
            for m in muxes:
                mrt[m[0]][m[1]] = []
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    core_ranges=crs(muxes),
                    compile_time_args=list(mcfg.kernel_compile_time_args()),
                    runtime_args=mrt,
                    defines=FAB_DEFS,
                    config=dm(0),
                    opt_level=fabric_mux.KernelBuildOptLevel.O3,
                )
            )
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
            for g in groups_on:
                link = ln[g][1]
                for d in range(2):
                    m = mux_core(g, d, G)
                    p.kernels[mux_idx].runtime_args[m[0]][m[1]].extend(
                        mcfg.kernel_runtime_args(
                            source_node_id=node(mesh, me_rc),
                            destination_node_id=node(mesh, nxt[g][d]),
                            link_index=link,
                            program_descriptor=p,
                            mux_logical_core=ttnn.CoreCoord(*m),
                        )
                    )
                for d in range(2):
                    m = mux_core(g, d, G)
                    mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*m))
                    master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*groups[g][0]))
                    term_sem = None
                    for j, (x, y) in enumerate(groups[g]):
                        args = fabric_mux.client_runtime_args(
                            connection_valid=True,
                            is_termination_master=j == 0,
                            channel_type=ctype,
                            mux_virtual_core=mux_vc,
                            client_index=j,
                            client_logical_core=ttnn.CoreCoord(x, y),
                            config=mcfg,
                            program_descriptor=p,
                            termination_master_virtual_core=master_vc,
                            termination_master_semaphore_id=term_sem,
                        )
                        if j == 0:
                            term_sem = args[10]
                        p.kernels[cl_idx[g]].runtime_args[x][y].extend(args)
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    ttnn.synchronize_device(mesh)
    ttnn.generic_op([src, slots, cnt, st], mpd)
    ttnn.synchronize_device(mesh)
    out = {}
    sts = ttnn.get_device_tensors(st)
    for X in range(R * Cn):
        s = (ttnn.to_torch(sts[X]).to(torch.int64) & 0xFFFFFFFF).reshape(len(C), 16)
        for gi, g in enumerate(groups_on):
            sg = s[gi * 20 : (gi + 1) * 20]
            for d in range(2):
                if dmask >> d & 1:
                    out[(X, g, d)] = int(sg[:, 5 + 2 * d].max() - sg[:, 4 + 2 * d].min())
        out[(X, "send")] = int((s[:, 1] - s[:, 0]).max())
        out[(X, "full")] = float(s[:, 8].double().mean())
    return out


CASES = {
    # (groups / lanes on, direction mask, flush)
    "lane": ((0, 1), 3, 0),  # the lane's pattern: 2 lanes, both directions, arr_nf
    "lane_flush": ((0, 1), 3, 1),
    "lane_dir0": ((0, 1), 1, 0),  # 2 lanes, direction 0 only (every link one way)
    "one_lane": ((0,), 3, 0),  # lane 0 only, both directions
    "one_lane_dir0": ((0,), 1, 0),
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_lane_fab_bench(mesh_device, case):
    groups_on, dmask, flush = CASES[case]
    NPKT = int(os.environ.get("WS_NPKT", "300"))
    clk = aiclk_mhz(0)
    runs = []
    for i in range(WARM + MEAS):
        o = launch(mesh_device, groups_on, dmask, NPKT, flush)
        if i >= WARM:
            runs.append(o)
    nb = 20 * NPKT * PB
    keys = [k for k in runs[0] if isinstance(k[1], int)]
    gbs = {k: statistics.median(nb / (r[k] / clk * 1e3) for r in runs) for k in keys}
    vals = sorted(gbs.values())
    N = mesh_device.get_num_devices()
    per_chip_in = [sum(v for k, v in gbs.items() if k[0] == X) for X in range(N)]
    send_us = statistics.median(max(r[(X, "send")] for X in range(N)) for r in runs) / clk
    full = statistics.median(statistics.mean(r[(X, "full")] for X in range(N)) for r in runs)
    logger.info(
        f"LFB {case}: {len(keys)} link directions, {NPKT} packets of {PB} B a core a direction, aiclk {clk}: GB/s a "
        f"link direction median {statistics.median(vals):.1f} (min {vals[0]:.1f}, max {vals[-1]:.1f}); into a chip "
        f"median {statistics.median(per_chip_in):.1f}; longest send loop {send_us:.0f} us; full-mux polls a core "
        f"{full:.0f}"
    )
    logger.info(
        f"LFB {case} per link direction (chip, group, dir): "
        + ", ".join(f"{k}: {v:.1f}" for k, v in sorted(gbs.items()))
    )


def fab(payload):
    from models.demos.ws_expert.tests.test_ws_fab_bench import router_cfg

    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": router_cfg(payload)}


# (router max payload, packet bytes, mux buffers a channel, atomic inc every INC packets); the lane pattern (2 lanes,
# both directions, all chips), the same bytes a core a direction in every case (300 x 13,824 B)
SWEEP = [
    (15232, 13824, 2, 1),
    (15232, 13824, 2, 15),
    (15232, 13824, 4, 1),
    (15232, 13824, 4, 15),
    (6912, 6912, 2, 1),
    (6912, 6912, 4, 1),
    (6912, 6912, 8, 1),
    (6912, 6912, 8, 30),
    (4608, 4608, 4, 1),
    (4608, 4608, 8, 1),
    (4608, 4608, 12, 1),
    (4608, 4608, 12, 45),
]


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params,pb,nbuf,inc",
    [(fab(p), b, n, i) for p, b, n, i in SWEEP],
    ids=[f"pay{p}_pb{b}_buf{n}_inc{i}" for p, b, n, i in SWEEP],
    indirect=["device_params"],
)
def test_lane_fab_sweep(mesh_device, pb, nbuf, inc):
    NPKT = 300 * 13824 // pb
    clk = aiclk_mhz(0)
    runs = []
    for i in range(WARM + MEAS):
        o = launch(mesh_device, (0, 1), 3, NPKT, 0, nbuf=nbuf, PB=pb, inc=inc)
        if i >= WARM:
            runs.append(o)
    nb = 20 * NPKT * pb
    keys = [k for k in runs[0] if isinstance(k[1], int)]
    gbs = {k: statistics.median(nb / (r[k] / clk * 1e3) for r in runs) for k in keys}
    vals = sorted(gbs.values())
    lane = {g: min(v for k, v in gbs.items() if k[1] == g) for g in (0, 1)}
    N = mesh_device.get_num_devices()
    send_us = statistics.median(max(r[(X, "send")] for X in range(N)) for r in runs) / clk
    logger.info(
        f"LFS payload {fab_payload(mesh_device)} pb {pb} nbuf {nbuf} inc every {inc}: GB/s a link direction median "
        f"{statistics.median(vals):.1f} (min {vals[0]:.1f}, max {vals[-1]:.1f}); slowest link a lane "
        f"{lane[0]:.1f} / {lane[1]:.1f}; longest send loop {send_us:.0f} us"
    )


def fab_payload(mesh):
    return fabric_mux.channel_buffer_size_bytes(mesh)
