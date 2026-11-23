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
    sc = lambda f: f[3][0] if isinstance(f[3][0], tuple) else f[3]  # (sender core, receiver core) or one core
    rc_ = lambda f: f[3][1] if isinstance(f[3][0], tuple) else f[3]
    cores = sorted({sc(f) for f in flows} | {rc_(f) for f in flows})
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
        for rk in (0, 1):
            grp = [f for f in sends if (f[4] if len(f) > 4 else risc) == rk]
            if grp:
                rt = ttnn.RuntimeArgs()
                for f in grp:
                    src, dst, link = f[:3]
                    x, y = sc(f)
                    xy = noc(mesh, *rc_(f))
                    rt[x][y] = [int(node(mesh, src).mesh_id), int(node(mesh, dst).chip_id), xy[0] | xy[1] << 8]
                    rt[x][y].append(st.buffer_address())
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("fx_send.cpp"),
                        core_ranges=crs([sc(f) for f in grp]),
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
                        config=dm(rk),
                    )
                )
        if recvs:
            rt = ttnn.RuntimeArgs()
            for f in recvs:
                x, y = rc_(f)
                rt[x][y] = [st.buffer_address()]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("fx_recv.cpp"),
                    core_ranges=crs([rc_(f) for f in recvs]),
                    compile_time_args=[want, cnt.buffer_address()],
                    runtime_args=rt,
                    config=dm(0),
                )
            )
        p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
        ki = {}
        for kk, rk in enumerate([rk for rk in (0, 1) if any((f[4] if len(f) > 4 else risc) == rk for f in sends)]):
            ki[rk] = kk
        for f in sends:
            src, dst, link = f[:3]
            x, y = sc(f)
            p.kernels[ki[f[4] if len(f) > 4 else risc]].runtime_args[x][y].extend(
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
    for f in flows:
        src, dst, link = f[:3]
        r = (ttnn.to_torch(sts[dst[0] * W + dst[1]]).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), 16)[
            cores.index(rc_(f))
        ]
        s = (ttnn.to_torch(sts[src[0] * W + src[1]]).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), 16)[
            cores.index(sc(f))
        ]
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


def all_link_flows(shape, links=(0, 1)):
    """Every mesh edge, both links, both directions: one flow each, a core of its own (the same logical core on both
    chips; no core used twice on a chip)."""
    R, C = shape
    edges = [((r, c), (r, c + 1)) for r in range(R) for c in range(C - 1)] + [((0, c), (1, c)) for c in range(C)]
    grid = [(x, y) for y in range(10) for x in range(12)]
    flows = []
    for a, b in edges:
        for l in links:
            for s, d in ((a, b), (b, a)):
                flows.append((s, d, l, grid[(len(flows) * 7) % len(grid)]))
    return flows


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [dev_params(p) for p in (8704, 15232)], ids=["pay8704", "pay15232"], indirect=True
)
def test_fx_all_links(mesh_device):
    """Every link of the mesh busy in both directions at once (middle chips: 6 flows out, 6 in), full-slot packets."""
    pay = ttnn.get_tt_fabric_max_payload_size_bytes()
    flows = all_link_flows(tuple(mesh_device.shape))
    SCEN["all"] = flows
    per = run(mesh_device, "all", pay, tag="all links")
    out, inn = {}, {}
    for (s, d, l, _), v in zip(flows, per):
        out[s] = out.get(s, 0) + v
        inn[d] = inn.get(d, 0) + v
    logger.info(
        f"FX all links router {pay}: GB/s out / in a chip "
        f"{ {c: (round(out[c]), round(inn[c])) for c in sorted(out)} }; per flow min {min(per):.1f} max {max(per):.1f}"
    )


def eth_column(x):
    """Logical Tensix column under (or nearest to) NoC column x on the 12 x 10 grid: x 1-7 -> 0-6, 10-14 -> 7-11."""
    if x <= 7:
        return max(0, x - 1)
    if x <= 9:
        return 6 if x == 8 else 7
    return min(11, x - 3)


def placed_flows(mesh, links=(0, 1), split_rows=False):
    """all_link_flows with each flow's sender core under its ethernet core on the sending chip and its receiver core
    under the receiving chip's ethernet core of the same link (row 0 / 1 / ... first free)."""
    from models.demos.ws_expert.ws_lane5 import eth_x

    fabric_mux = ttnn.experimental.fabric_mux
    mcfg = fabric_mux.Config(
        num_full_size_channels=1,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=2,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    used = {}

    def take(chip, col, bottom=False):
        for y in range(9, -1, -1) if bottom else range(10):
            if (col, y) not in used.setdefault(chip, set()):
                used[chip].add((col, y))
                return (col, y)
        raise AssertionError("column full")

    out = []
    for s_, d_, l, _ in all_link_flows(tuple(mesh.shape), links):
        sx = eth_column(eth_x(mesh, mcfg, s_, d_, l))
        rx = eth_column(eth_x(mesh, mcfg, d_, s_, l))
        out.append((s_, d_, l, (take(s_, sx), take(d_, rx, bottom=split_rows))))
    return out


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_all_links_placed(mesh_device):
    """Every link busy both ways, cores under their ethernet cores (sender on the sending chip, receiver on the
    receiving chip); then the scattered placement again for comparison."""
    for name, flows in (("placed", placed_flows(mesh_device)), ("scattered", all_link_flows(tuple(mesh_device.shape)))):
        SCEN[name] = flows
        per = run(mesh_device, name, 8704, tag="all links")
        out, inn = {}, {}
        for f, v in zip(flows, per):
            out[f[0]] = out.get(f[0], 0) + v
            inn[f[1]] = inn.get(f[1], 0) + v
        logger.info(
            f"FX all links {name}: GB/s out / in a chip {[(c, round(out[c]), round(inn[c])) for c in sorted(out)]}; "
            f"per flow min {min(per):.1f} median {statistics.median(per):.1f} max {max(per):.1f}"
        )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_all_links_rows(mesh_device):
    """Every link busy both ways: senders at the top row under their ethernet core (NOC1: one hop north into the
    ethernet row), receivers at the bottom row of their ethernet core's column (the router's NOC1 landing writes wrap
    north: two hops instead of the whole column)."""
    flows = placed_flows(mesh_device, split_rows=True)
    SCEN["rows"] = flows
    per = run(mesh_device, "rows", 8704, tag="all links")
    out, inn = {}, {}
    for f, v in zip(flows, per):
        out[f[0]] = out.get(f[0], 0) + v
        inn[f[1]] = inn.get(f[1], 0) + v
    logger.info(
        f"FX all links rows: GB/s out / in a chip {[(c, round(out[c]), round(inn[c])) for c in sorted(out)]}; "
        f"per flow min {min(per):.1f} median {statistics.median(per):.1f} max {max(per):.1f}; flows {flows}"
    )


CTRL = {
    "single_E": [((0, 0), (0, 1), 0, (1, 0))],
    "single_W": [((0, 1), (0, 0), 0, (1, 0))],
    "disjoint_2": [((0, 0), (0, 1), 0, (1, 0)), ((0, 2), (0, 3), 0, (2, 0))],
    "disjoint_3": [((0, 0), (0, 1), 0, (1, 0)), ((0, 2), (0, 3), 0, (2, 0)), ((1, 0), (1, 1), 0, (3, 0))],
    "two_in": [((0, 1), (0, 0), 0, (1, 0)), ((1, 0), (0, 0), 0, (2, 0))],
    "two_out": [((0, 0), (0, 1), 0, (1, 0)), ((0, 0), (1, 0), 0, (2, 0))],
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_controls(mesh_device):
    """Is the multi-flow drop per chip? Two / three flows sharing no chip, two flows into one chip, two out of one."""
    SCEN.update(CTRL)
    for scen in CTRL:
        run(mesh_device, scen, 8704, tag="ctrl")


CTRL2 = {
    "one_out_one_in": [((0, 0), (0, 1), 0, (1, 0)), ((1, 0), (0, 0), 0, (2, 0))],
    "two_out_l1": [((0, 0), (0, 1), 1, (1, 0)), ((0, 0), (1, 0), 1, (2, 0))],
    "two_out_mixed": [((0, 0), (0, 1), 0, (1, 0)), ((0, 0), (1, 0), 1, (2, 0))],
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_controls2(mesh_device):
    """One chip with one router sending and another receiving; two out on other link pairings."""
    SCEN.update(CTRL2)
    for scen in CTRL2:
        run(mesh_device, scen, 8704, tag="ctrl2")


CTRL3 = {
    "two_out_noc1_noc1": [((0, 0), (0, 1), 0, (1, 0), 1), ((0, 0), (1, 0), 0, (2, 0), 1)],
    "two_out_noc0_noc0": [((0, 0), (0, 1), 0, (1, 0), 0), ((0, 0), (1, 0), 0, (2, 0), 0)],
    "two_out_noc0_noc1": [((0, 0), (0, 1), 0, (1, 0), 0), ((0, 0), (1, 0), 0, (2, 0), 1)],
    "two_out_noc1_noc0": [((0, 0), (0, 1), 0, (1, 0), 1), ((0, 0), (1, 0), 0, (2, 0), 0)],
    "two_out_noc0_noc1_samecore": [((0, 0), (0, 1), 0, (1, 0), 0), ((0, 0), (1, 0), 0, (1, 0), 1)],
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_controls3(mesh_device):
    """Two senders on one chip: the same NoC (paths can share links) or one on each NoC (they cannot)."""
    SCEN.update(CTRL3)
    for scen in CTRL3:
        run(mesh_device, scen, 8704, tag="ctrl3")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_trace(mesh_device):
    """One launch of scenario FX_SCEN (FX_BYTES small: the NoC event trace holds ~2,900 events a RISC) for NoC traces
    (TT_METAL_DEVICE_PROFILER_NOC_EVENTS); prints each flow's chip ids and ethernet NoC x."""
    from models.demos.ws_expert.ws_lane5 import eth_x

    for d in (CTRL, CTRL2, CTRL3):
        SCEN.update(d)
    scen = os.environ.get("FX_SCEN", "two_out_noc1_noc1")
    fabric_mux = ttnn.experimental.fabric_mux
    mcfg = fabric_mux.Config(
        num_full_size_channels=1,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=2,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh_device),
        base_l1_address=ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1),
    )
    for f in SCEN[scen]:
        s_, d_, l = f[:3]
        logger.info(
            f"FXT flow {f}: chips {int(node(mesh_device, s_).chip_id)} -> {int(node(mesh_device, d_).chip_id)}, "
            f"sender eth x {eth_x(mesh_device, mcfg, s_, d_, l)}, receiver eth x {eth_x(mesh_device, mcfg, d_, s_, l)}, "
            f"sender core noc {noc(mesh_device, *(f[3][0] if isinstance(f[3][0], tuple) else f[3]))}"
        )
    global WARM, MEAS
    WARM, MEAS = 0, 1
    run(mesh_device, scen, 8704, mode=int(os.environ.get("FX_MODE", "1")), tag="trace")
    ttnn.ReadDeviceProfiler(mesh_device)


ETH_ALL_X = (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10)  # (physical ethernet channel -> NoC x)
P150_ETH_HARVEST = 0x120  # (channels 5 and 8 fused off on a P150; the fabric's channel ids skip them)


def eth_phys_x(mesh, src, dst, link, harvest=P150_ETH_HARVEST):
    """Physical NoC x of the ethernet core the fabric uses from src towards dst on link: the fabric's channel id counts
    the chip's unharvested channels (checked against the NoC trace's destinations)."""
    from models.demos.ws_expert.ws_lane5 import BH_ETH_X  # noqa: F401  (the unharvested order)

    fabric_mux = ttnn.experimental.fabric_mux
    mcfg = fabric_mux.Config(
        num_full_size_channels=1,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=2,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    a = list(
        mcfg.kernel_runtime_args(
            source_node_id=node(mesh, src),
            destination_node_id=node(mesh, dst),
            link_index=link,
            program_descriptor=ttnn.ProgramDescriptor(),
            mux_logical_core=ttnn.CoreCoord(0, 0),
        )
    )
    chan = a[1 + 2 * a[0]]
    live = [x for i, x in enumerate(ETH_ALL_X) if not harvest >> i & 1]
    return live[chan]


PHYS_TENSIX = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]  # (logical column -> physical x on bh-lb-09: x 10 harvested)


def to_logical(px, py):
    return (PHYS_TENSIX.index(px), py - 2)


def phys_placed_flows(mesh, links=(0, 1), only=None):
    """Senders right under their ethernet core on NOC1 (one hop north into it), receivers at the bottom row of their
    ethernet core's column (the routers' NOC1 landing writes: north 1 -> 0 -> 11, no row travel). Ethernet x 10 / 16
    (no Tensix column): sender at (7, 11) / (15, 11) on NOC0 (east along the bottom row, then south 11 -> 0 -> 1);
    receiver at x 7 / 15 on the bottom row."""
    out = []
    used = {}

    def free(chip, px, py, step):  # (a clash: the next row along the same column, still no row travel)
        while (px, py) in used.setdefault(chip, set()):
            py += step
        used[chip].add((px, py))
        return px, py

    flows = only if only is not None else all_link_flows(tuple(mesh.shape), links)
    for f in flows:
        s_, d_, l = f[:3]
        ex_s, ex_d = eth_phys_x(mesh, s_, d_, l), eth_phys_x(mesh, d_, s_, l)
        if ex_s in PHYS_TENSIX:
            snd, rk = to_logical(*free(s_, ex_s, 2, 1)), 1
        elif ex_s == 10:  # (every NOC1 path up column 11 shares (11, 2) -> (11, 1) with that column's own sender)
            snd, rk = to_logical(*free(s_, 7, 11, -1)), 0
        else:
            snd, rk = to_logical(*free(s_, 15, 11, -1)), 0
        rx = ex_d if ex_d in PHYS_TENSIX else {10: 7, 16: 15}[ex_d]
        out.append((s_, d_, l, (snd, to_logical(*free(d_, rx, 11, -1))), rk))
    return out


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dev_params(8704)], ids=["pay8704"], indirect=True)
def test_fx_phys_placed(mesh_device):
    """Two senders on one chip and every link busy, placed by physical NoC paths (one NoC for every sender)."""
    for d in (CTRL, CTRL2, CTRL3):
        SCEN.update(d)
    SCEN["two_out_phys"] = phys_placed_flows(mesh_device, only=CTRL["two_out"])
    SCEN["two_in_phys"] = phys_placed_flows(mesh_device, only=CTRL["two_in"])
    SCEN["2link_2dir_phys"] = phys_placed_flows(mesh_device, only=SCEN["2link_2dir"])
    SCEN["all_phys"] = phys_placed_flows(mesh_device)
    for scen in ():
        logger.info(f"FXP {scen}: {SCEN[scen]}")
        run(mesh_device, scen, 8704, tag="phys")
    flows = SCEN["all_phys"]
    per = run(mesh_device, "all_phys", 8704, tag="phys")
    out, inn = {}, {}
    for f, v in zip(flows, per):
        out[f[0]] = out.get(f[0], 0) + v
        inn[f[1]] = inn.get(f[1], 0) + v
    logger.info(
        f"FX all links phys: GB/s out / in a chip {[(c, round(out[c]), round(inn[c])) for c in sorted(out)]}; per flow "
        f"min {min(per):.1f} median {statistics.median(per):.1f} max {max(per):.1f}; flows {flows}"
    )


CTRL.update({"north_2link": [((1, 0), (0, 0), 0, (1, 0), 1), ((1, 0), (0, 0), 1, (2, 0), 1)]})
