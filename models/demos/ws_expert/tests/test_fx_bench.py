# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fabric link microbench (kernels fx_send.cpp / fx_recv.cpp): the smallest fabric program - one worker core a flow with
its own fabric connection (no mux), packets from one L1 buffer into one L1 slot of the same core on a neighbouring
chip, headers routed once, a fused atomic inc on the first and last packet only (the receiver times between them).
Flows: (source chip, destination chip, link, core). Sweeps: router max payload (device_params), packet size, the
connection's send path (MODE: 1 stateful, 2 posted, 4 an inc every packet), one / both directions, one / both links.
GB/s a flow = (NPKT - 1) x PB / the receiver's first -> last arrival; WS_WARM + WS_MEAS launches a case."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath
from models.demos.ws_expert.tests.test_ws_fab_bench import router_cfg
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node

WARM = int(os.environ.get("WS_WARM", "3"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
BYTES = int(os.environ.get("FX_BYTES", str(48 << 20)))  # bytes a flow a launch

A, B = (0, 0), (0, 1)  # an X neighbour pair
SCEN = {
    "1link_1dir": [(A, B, 0, (1, 0))],
    "1link_2dir": [(A, B, 0, (1, 0)), (B, A, 0, (2, 0))],
    "2link_1dir": [(A, B, 0, (1, 0)), (A, B, 1, (2, 0))],
    "2link_2dir": [(A, B, 0, (1, 0)), (A, B, 1, (2, 0)), (B, A, 0, (3, 0)), (B, A, 1, (4, 0))],
}


def dm(risc):
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )


def launch(mesh, flows, PB, NPKT, mode, risc):
    cores = sorted({f[3] for f in flows})
    words = lambda n: sharded_l1(mesh, cores, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    buf = words(PB // 16)  # source on the sender, landing slot on the receiver (same address)
    cnt = words(1)
    st = words(4)
    want = NPKT if mode & 4 else 2
    chips = sorted({f[0] for f in flows} | {f[1] for f in flows})
    mpd = ttnn.MeshProgramDescriptor()
    for ch in chips:
        sends = [f for f in flows if f[0] == ch]
        recvs = [f for f in flows if f[1] == ch]
        kernels = []
        if sends:
            rt = ttnn.RuntimeArgs()
            for src, dst, link, (x, y) in sends:
                xy = noc(mesh, x, y)
                rt[x][y] = [int(node(mesh, src).mesh_id), int(node(mesh, dst).chip_id), xy[0] | xy[1] << 8]
                rt[x][y].append(st.buffer_address())
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("fx_send.cpp"),
                    core_ranges=crs([f[3] for f in sends]),
                    compile_time_args=[
                        NPKT,
                        PB,
                        buf.buffer_address(),
                        buf.buffer_address(),
                        cnt.buffer_address(),
                        mode,
                    ],
                    runtime_args=rt,
                    defines=FAB_DEFS,
                    config=dm(risc),
                )
            )
        if recvs:
            rt = ttnn.RuntimeArgs()
            for *_, (x, y) in recvs:
                rt[x][y] = [st.buffer_address()]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("fx_recv.cpp"),
                    core_ranges=crs([f[3] for f in recvs]),
                    compile_time_args=[want, cnt.buffer_address()],
                    runtime_args=rt,
                    config=dm(0),
                )
            )
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        for src, dst, link, (x, y) in sends:
            p.kernels[0].runtime_args[x][y].extend(
                ttnn.setup_fabric_connection(
                    mesh_device=mesh,
                    src_fabric_node_id=node(mesh, src),
                    dst_fabric_node_id=node(mesh, dst),
                    link_idx=link,
                    program_descriptor=p,
                    worker_core=ttnn.CoreCoord(x, y),
                )
            )
        mc = ttnn.MeshCoordinate(*ch)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    ttnn.synchronize_device(mesh)
    ttnn.generic_op([buf, cnt, st], mpd)
    ttnn.synchronize_device(mesh)
    sts = ttnn.get_device_tensors(st)
    W = mesh.shape[1]
    out = []
    for src, dst, link, (x, y) in flows:
        i = cores.index((x, y))
        r = (ttnn.to_torch(sts[dst[0] * W + dst[1]]).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), 16)[i]
        s = (ttnn.to_torch(sts[src[0] * W + src[1]]).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), 16)[i]
        out.append((int(r[3] - r[2]), int(s[1] - s[0])))
    del buf, cnt, st, sts
    return out


def run(mesh, scen, PB, mode=1, risc=1, tag=""):
    flows = SCEN[scen]
    NPKT = max(64, BYTES // PB)
    clk = aiclk_mhz(0)
    rows = [launch(mesh, flows, PB, NPKT, mode, risc) for _ in range(WARM + MEAS)][WARM:]
    gbs = [[(NPKT - 1) * PB / (r[k][0] / clk * 1e3) for k in range(len(flows))] for r in rows]
    per_flow = [statistics.median(g[k] for g in gbs) for k in range(len(flows))]
    send_gbs = statistics.median(NPKT * PB / (r[0][1] / clk * 1e3) for r in rows)
    logger.info(
        f"FX {tag} {scen} PB {PB} mode {mode} risc {risc}: GB/s per flow {[round(v, 1) for v in per_flow]} "
        f"(min over launches {round(min(min(g) for g in gbs), 1)}), flow 0 send loop {send_gbs:.1f} GB/s, "
        f"{NPKT} packets a flow, aiclk {clk}"
    )
    return per_flow


def dev_params(payload):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": router_cfg(payload)}


PAYLOADS = [4352, 8704, 15232]


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in PAYLOADS], ids=[f"pay{p}" for p in PAYLOADS], indirect=True
)
def test_fx_packet_sweep(mesh_device):
    """One flow, one link, one direction: every packet size up to the router's payload, the stateful send path."""
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    for PB in [p for p in (1088, 2176, 4352, 6528, 8704, 10880, 13056, 15232) if p <= pay]:
        run(mesh_device, "1link_1dir", PB, tag=f"router {pay}")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(15232)], ids=["pay15232"], indirect=True)
def test_fx_modes(mesh_device):
    """15232 B packets on one flow: the send paths (plain / stateful / posted / an inc every packet), the RISC."""
    for mode, risc in [(0, 1), (1, 1), (3, 1), (5, 1), (1, 0)]:
        run(mesh_device, "1link_1dir", 15232, mode=mode, risc=risc, tag="modes")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(15232)], ids=["pay15232"], indirect=True)
def test_fx_directions_links(mesh_device):
    """15232 / 8704 / 4352 B packets: one / both directions on one / both links."""
    for PB in (15232, 8704, 4352):
        for scen in SCEN:
            run(mesh_device, scen, PB, tag="dirs")


SCEN["2link_1dir_far"] = [(A, B, 0, (1, 0)), (A, B, 1, (9, 9))]
SCEN["2link_2dir_far"] = [(A, B, 0, (1, 0)), (A, B, 1, (9, 9)), (B, A, 0, (4, 5)), (B, A, 1, (10, 2))]
SCEN["1link_1dir_far"] = [(A, B, 0, (9, 9))]


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in (8704, 15232)], ids=["pay8704", "pay15232"], indirect=True
)
def test_fx_placement(mesh_device):
    """Full-slot packets: every scenario, plus the flows' cores spread over the grid instead of side by side in row 0."""
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    for scen in SCEN:
        run(mesh_device, scen, pay, tag=f"router {pay}")


C = (1, 0)  # A's Y neighbour
NBR = {
    "y_1link_1dir": [(A, C, 0, (1, 0))],
    "2nbr_1dir": [(A, B, 0, (1, 0)), (A, C, 0, (2, 0))],
    "2nbr_2link_1dir": [(A, B, 0, (1, 0)), (A, B, 1, (2, 0)), (A, C, 0, (3, 0)), (A, C, 1, (4, 0))],
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_neighbours(mesh_device):
    """Full-slot 8704 B packets: a Y link alone, and one chip sending to two neighbours at once."""
    SCEN.update(NBR)
    for scen in NBR:
        run(mesh_device, scen, 8704, tag="nbr")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_noc(mesh_device):
    """Full-slot 8704 B packets from the BRISC (NOC0) instead of the NCRISC (NOC1): several flows out of one chip."""
    SCEN.update(NBR)
    for scen in ("1link_1dir", "2link_1dir", "2nbr_1dir", "2nbr_2link_1dir", "2link_2dir"):
        run(mesh_device, scen, 8704, risc=0, tag="noc0")
