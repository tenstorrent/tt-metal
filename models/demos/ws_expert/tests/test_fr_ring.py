# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fabric ring microbench in the weight-streaming shape (kernels/fr_lane.cpp): the tray's 8-chip snake as a ring, a
link core a link (BRISC: the lane to the next chip, NCRISC: the lane to the previous chip), every chip injecting E
packets a lane that travel N - 1 hops; ring slot credits piggy-backed on the opposite lane. GB/s a link direction =
(N - 1) E PB / (last send - first send) per lane; WS_WARM + WS_MEAS launches a case."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath
from models.demos.ws_expert.tests.test_fx_bench import dev_params
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_sf import snake

WARM = int(os.environ.get("WS_WARM", "2"))
MEAS = int(os.environ.get("WS_MEAS", "3"))
BYTES = int(os.environ.get("FR_BYTES", str(64 << 20)))  # bytes a link direction a launch
HS = 128


def dm(risc):
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )


def launch(mesh, PB, S, NO, FE, E, link_cores, links, mode=0, bmax=1):
    ring = snake(tuple(mesh.shape))
    N = len(ring)
    W_ = mesh.shape[1]
    lane_core = lambda l, D: link_cores[l][D] if isinstance(link_cores[l][0], tuple) else link_cores[l]
    cores = sorted({lane_core(l, D) for l in links for D in (0, 1)})
    lay, off = {}, 0
    for name, size in (
        ("r0", S * PB),
        ("r1", S * PB),
        ("o0", NO * PB),
        ("o1", NO * PB),
        ("h0", (S + 1) * HS),
        ("h1", (S + 1) * HS),
        ("w", 32 * 16),
    ):
        lay[name] = off
        off += size
    mem = sharded_l1(mesh, cores, [off // 16, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    dummy = sharded_l1(mesh, cores, [1, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    base = mem.buffer_address()
    A = {k: base + v for k, v in lay.items()}
    mpd = ttnn.MeshProgramDescriptor()
    for i, rc in enumerate(ring):
        nbr = (ring[(i + 1) % N], ring[(i - 1) % N])
        kernels = []
        for D in (0, 1):
            rt = ttnn.RuntimeArgs()
            for l in links:
                x, y = lane_core(l, D)
                xy = noc(mesh, x, y)
                po = lane_core(l, 1 - D)
                pxy = 0 if po == (x, y) else (lambda q: q[0] | q[1] << 8)(noc(mesh, *po))
                rt[x][y] = [int(node(mesh, rc).mesh_id), int(node(mesh, nbr[D]).chip_id), xy[0] | xy[1] << 8, i, pxy]
            ct = [D, S, PB, NO, E, N, FE, A["r0"], A["r1"], A[f"o{D}"], A[f"h{D}"], A["w"], HS, mode, bmax]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("fr_lane.cpp"),
                    core_ranges=crs([lane_core(l, D) for l in links]),
                    compile_time_args=ct,
                    runtime_args=rt,
                    defines=FAB_DEFS,
                    config=dm(D),
                )
            )
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        for D in (0, 1):
            for l in links:
                x, y = lane_core(l, D)
                p.kernels[D].runtime_args[x][y].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, rc),
                        dst_fabric_node_id=node(mesh, nbr[D]),
                        link_idx=l,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(x, y),
                    )
                )
        mc = ttnn.MeshCoordinate(*rc)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    ttnn.synchronize_device(mesh)
    ttnn.generic_op([dummy, mem], mpd)
    ttnn.synchronize_device(mesh)
    out = []  # (chip, link, lane, dt cycles, sends, arrivals)
    sts = ttnn.get_device_tensors(mem)
    wi = lay["w"] // 4
    for i, rc in enumerate(ring):
        t = (ttnn.to_torch(sts[rc[0] * W_ + rc[1]]).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), -1)
        for l in links:
            for D in (0, 1):
                ci = cores.index(lane_core(l, D))
                s = [int(t[ci, wi + 4 * (10 + 8 * D + q)]) for q in range(8)]
                out.append((i, l, D, (s[1] - s[0]) & 0xFFFFFFFF, s[2], s[3], s[4], s[5], s[6], s[7]))
    del mem, dummy, sts
    return out, N


def run(mesh, PB, S=32, NO=4, FE=0, links=(0, 1), link_cores=((1, 0), (2, 0)), tag="", mode=0, bmax=1):
    E = max(8, BYTES // (7 * PB))
    clk = aiclk_mhz(0)
    meas = []
    for it in range(WARM + MEAS):
        out, N = launch(mesh, PB, S, NO, FE, E, link_cores, links, mode, bmax)
        tot = (N - 1) * E
        assert all(o[4] == tot and o[5] == tot for o in out), f"sends / arrivals {[(o[4], o[5]) for o in out]}"
        if it >= WARM:
            meas.append([tot * PB / (o[3] / clk * 1e3) for o in out])
            last = out
    flat = [v for m in meas for v in m]
    worst = statistics.median(min(m) for m in meas)
    logger.info(
        f"FR {tag} PB {PB} S {S} NO {NO} FE {FE} mode {mode} bmax {bmax} links {links} cores {link_cores}: GB/s a link direction median "
        f"{statistics.median(flat):.1f} (min {min(flat):.1f}, max {max(flat):.1f}), slowest lane a launch {worst:.1f}; "
        f"{E} packets injected a lane, {(N - 1) * E * PB / 1e6:.0f} MB a link direction, aiclk {clk}; lane us (median "
        f"over lanes) router-space / downstream-credit / idle waits "
        f"{[round(statistics.median(o[q] for o in last) / clk) for q in (6, 7, 8)]} of "
        f"{round(statistics.median(o[3] for o in last) / clk)}, explicit credits {statistics.median(o[9] for o in last)}"
    )
    return worst


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in (8704, 15232)], ids=["pay8704", "pay15232"], indirect=True
)
def test_fr_ring(mesh_device):
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    run(mesh_device, pay, links=(0,), tag="one link")
    run(mesh_device, pay, tag="two links")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fr_sweep(mesh_device):
    """Ring slots S and sends between flushes FE (8704 B packets, one link)."""
    for S, NO, FE, links in [
        (32, 4, 2, (0,)),
        (32, 4, 0, (0,)),
        (32, 8, 0, (0,)),
        (48, 8, 0, (0,)),
        (32, 8, 0, (0, 1)),
    ]:
        run(mesh_device, 8704, S=S, NO=NO, FE=FE, links=links, tag="sweep")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in (8704, 15232)], ids=["pay8704", "pay15232"], indirect=True
)
def test_fr_opt(mesh_device):
    """Send path (stateful / posted), full-slot packets, link cores apart."""
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    S = 32 if pay <= 8704 else 20
    for mode in (0, 1, 3):
        run(mesh_device, pay, S=S, NO=4, links=(0,), mode=mode, tag="opt")
    run(mesh_device, pay, S=S, NO=4, links=(0, 1), mode=1, tag="opt")
    run(mesh_device, pay, S=S, NO=4, links=(0, 1), link_cores=((1, 0), (9, 9)), mode=1, tag="opt far")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fr_opt2(mesh_device):
    """Batched arrivals, stateful path; one link, then two links with the link cores at several places."""
    run(mesh_device, 8704, S=32, NO=4, links=(0,), mode=1, tag="opt2")
    run(mesh_device, 8704, S=32, NO=8, links=(0,), mode=1, tag="opt2")
    for lc in (((1, 0), (9, 9)), ((1, 9), (10, 0)), ((5, 4), (6, 5)), ((0, 0), (11, 9)), ((3, 2), (8, 7))):
        run(mesh_device, 8704, S=32, NO=4, links=(0, 1), link_cores=lc, mode=1, tag="opt2")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fr_bmax(mesh_device):
    for bmax in (1, 2, 4, 32):
        run(mesh_device, 8704, S=32, NO=4, links=(0,), mode=1, bmax=bmax, tag="bmax")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fr_split(mesh_device):
    """A core a lane (the partner's words written over the NoC), bmax 2 / 4."""
    for mode in (0, 1):
        run(mesh_device, 8704, S=32, NO=4, links=(0,), mode=mode, bmax=2, link_cores=(((1, 0), (2, 0)),), tag="split")
    for lc in (
        (((1, 0), (2, 0)), ((9, 9), (10, 9))),
        (((1, 0), (1, 9)), ((10, 0), (10, 9))),
        (((0, 0), (11, 9)), ((5, 4), (6, 5))),
    ):
        run(mesh_device, 8704, S=32, NO=4, links=(0, 1), mode=1, bmax=2, link_cores=lc, tag="split")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in (8704, 15232)], ids=["pay8704", "pay15232"], indirect=True
)
def test_fr_best(mesh_device):
    """Shared link core, stateful path: full-slot packets, arrivals 2 / 4 a pass, one link and both links (cores apart)."""
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    for bmax in (2, 4):
        run(mesh_device, pay, S=32, NO=4, links=(0,), mode=1, bmax=bmax, tag="best")
        run(mesh_device, pay, S=32, NO=4, links=(0, 1), link_cores=((1, 0), (9, 9)), mode=1, bmax=bmax, tag="best")
