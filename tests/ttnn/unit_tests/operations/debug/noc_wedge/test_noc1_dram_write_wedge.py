# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Blackhole NoC wedge: acknowledged (non-posted) DRAM writes on NOC1 while NOC1 DRAM reads are in flight, with
fabric traffic as the optional extra ingredient. Traffic only.

Pattern (all worker cores of every chip, logical coordinates):
  writers  row 0, NCRISC on NOC1: units of 32 x 640 B DRAM writes over the banks (the fused MoE head's dispatch-prep
           rows), non-posted, flushed (not acked) after each unit, acks collected at the kernel's end; 3 x 8 KB NOC1
           DRAM reads of their own in flight while each unit's writes go (the engine NCRISC's W stream).
           write_mode: acked (the hazard) | posted | posted_fence (the fix: posted + one non-posted 4 B write a bank
           + a write barrier per unit); shape "fin": 36 x 2048 B a unit (the reduce-scatter fin's y tiles)
  loaders  every other core, NCRISC on NOC1: 8 KB NOC1 DRAM reads over the banks, 4 in flight (the W re-reads)
  fabric   port cores (last column, rows 2, 3[, 4, 5]), BRISC on NOC1: stream packets over a fabric link to a neighbour
           chip - into its port core's L1 ("l1": as the head's gather / reduce-scatter ports, every 4th a fused write +
           atomic increment), into its DRAM banks ("dram": the receiving router's DRAM writes) or atomic increments
           only ("atomics"). fabric_dirs "x": both ports to the X neighbour on links 0 and 1 (the head's); "xy": X and
           Y neighbours, links 0 and 1 (4 ports)
Nothing waits on another core: the loaders read a fixed count, the fabric senders wait only for their router's slot
credits, packets land in scratch nobody reads. A dispatch timeout is a NoC (or DRAM / fabric) that stopped.

What a hang looked like in the op (K3 fused head, 2x4 LoudBox bh-lb-09, 2026-10-24): one chip's NOC1 wedged at a
random call (~1 in 9..200 calls with the prep in every chunk) - its prep-row cores and 2 port cores with NOC1 command
buffers busy; tt-triage EMPTY in 6 of 10 hangs (triage talks NOC1: an empty triage = NOC1 wedged; never retry triage
on another NoC). Fixed by posting the writes + a per-bank fence. A second instance (2026-10-30): the RS fin's ~35
acked NOC1 DRAM writes per core in flight wedged K3 ring 1280 (2 of 4 soaks).

The earlier traffic-only probe (moe_block test_noc1_traffic: same writers / loaders, 400k units a writer x 8 chips;
+ 2 port cores streaming 8 KB packets with fused atomics to an X neighbour's L1 or DRAM, 200 programs) did NOT
wedge, so the trigger is still open. Over that probe this one adds: no stop-word protocol (fixed counts), more fabric
(4 ports, X and Y links, atomics-only, DRAM landing), and the fin shape.
"""

import dataclasses

import pytest

import ttnn

from tests.ttnn.unit_tests.operations.debug.noc_wedge import noc_wedge_common as nw

SHAPES = {"prep": (32, 640), "fin": (36, 2048)}  # writes a unit, bytes a write


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    expect: str  # "hang" (the op), "clean" (control), "open" (not seen yet: the question this case asks)
    launches: int = 2000
    units: int = 2000  # writer units per launch
    shape: str = "prep"
    write_mode: str = "acked"
    write_barrier_each_unit: bool = False  # the op's "barrier per unit" variant still wedged
    own_reads: int = 3
    loader_reads: int = 1500  # 8 KB reads per loader per launch (about the writers' duration)
    fabric: str = "none"  # "none" | "l1" | "dram" | "atomics"
    fabric_dirs: str = "x"  # "x" | "xy"
    fabric_packets: int = 20000  # per port per launch
    fabric_packet_bytes: int = 4096

    @property
    def device_params(self):
        return {"fabric_config": ttnn.FabricConfig.FABRIC_2D} if self.fabric != "none" else {}


CASES = [
    # the plain NOC1 mix: clean in the earlier probe (400k units a writer x 8 chips) - the baseline
    Case("acked_reads", "open"),
    Case("fin_acked_reads", "open", shape="fin"),
    # + fabric, the suspected missing ingredient
    Case("acked_reads_fabric_l1", "open", fabric="l1"),
    Case("acked_reads_fabric_dram", "open", fabric="dram"),
    Case("acked_reads_fabric_atomics", "open", fabric="atomics"),
    Case("acked_reads_fabric_l1_xy", "open", fabric="l1", fabric_dirs="xy"),
    Case("fin_acked_reads_fabric_l1_xy", "open", shape="fin", fabric="l1", fabric_dirs="xy"),
    # controls (the fix): run only once an acked case hangs
    Case("posted_fence_reads_fabric_l1_xy", "clean", write_mode="posted_fence", fabric="l1", fabric_dirs="xy"),
    Case("posted_reads_fabric_l1_xy", "clean", write_mode="posted", fabric="l1", fabric_dirs="xy"),
    Case("noreads_fabric_l1_xy", "clean", own_reads=0, loader_reads=0, fabric="l1", fabric_dirs="xy"),
]


def _ports(grid, cfg, mesh_shape):
    """[(port core, (dr, dc) neighbour offset, link)] for one chip at (r, c)."""
    x = grid.x - 1
    if cfg.fabric_dirs == "x":
        return [((x, 2), "x", 0), ((x, 3), "x", 1)]
    out = [((x, 2), "x", 0), ((x, 3), "x", 1)]
    if mesh_shape[0] > 1:
        out += [((x, 4), "y", 0), ((x, 5), "y", 1)]
    return out


def _neighbour(rc, axis, shape):
    r, c = rc
    rows, cols = shape
    if axis == "x":
        return (r, c + 1) if c + 1 < cols else (r, c - 1)
    return (r + 1, c) if r + 1 < rows else (r - 1, c)


def build_traffic(mesh, s, cfg, port_cores):
    grid = mesh.compute_with_storage_grid_size()
    plan = nw.TrafficPlan(grid)
    writes, nbytes = SHAPES[cfg.shape]
    writers = [(x, 0) for x in range(grid.x)]
    per = []
    for i, c in enumerate(writers):
        ct, rt = nw.dram_write(
            s,
            cfg.units,
            writes,
            nbytes,
            i,
            mode=cfg.write_mode,
            barrier_each_unit=cfg.write_barrier_each_unit,
            own_reads=cfg.own_reads,
            index=i,
        )
        per.append((c, rt))
    plan.add("writers", nw.TRAFFIC_KERNEL, ct, per, 1)
    if cfg.loader_reads:
        loaders = [(x, y) for y in range(1, grid.y) for x in range(grid.x) if (x, y) not in port_cores]
        per = []
        for i, c in enumerate(loaders):
            ct, rt = nw.dram_read(s, cfg.loader_reads, 8192, i, in_flight=4)
            per.append((c, rt))
        plan.add("loaders", nw.TRAFFIC_KERNEL, ct, per, 1)
    return plan


def build(mesh, s, cfg):
    grid = mesh.compute_with_storage_grid_size()
    shape = tuple(mesh.shape)
    if cfg.fabric == "none":
        plan = build_traffic(mesh, s, cfg, set())
        return nw.replicated(mesh, plan.program()), plan
    kind = {"l1": 3, "dram": 1, "atomics": 2}[cfg.fabric]
    pkt = min(cfg.fabric_packet_bytes, ttnn.get_tt_fabric_max_payload_size_bytes())
    assert 4 * pkt <= nw.RD - nw.FAB
    ports = _ports(grid, cfg, shape)
    port_cores = {p for p, _, _ in ports}
    desc = ttnn.MeshProgramDescriptor()
    for r in range(shape[0]):
        for c in range(shape[1]):
            plan = build_traffic(mesh, s, cfg, port_cores)
            program = ttnn.ProgramDescriptor()
            here = mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            extra = []
            for core, axis, link in ports:
                dst = _neighbour((r, c), axis, shape)
                there = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*dst))
                cc = ttnn.CoreCoord(*core)
                v = mesh.worker_core_from_logical_core(cc)
                args = [
                    cfg.fabric_packets,
                    pkt,
                    s.l1_base + nw.SRC,
                    int(there.chip_id),
                    int(there.mesh_id),
                    v.x,
                    v.y,
                    s.l1_base + nw.FAB,
                    4,
                    s.l1_base + nw.WORDS,
                    s.dram_base + nw.DRAM_FAB,
                    nw.DRAM_FAB_REGION,
                    s.nbanks,
                ]
                args += list(ttnn.setup_fabric_connection(mesh, here, there, link, program, cc))
                rt = ttnn.RuntimeArgs()
                rt[cc.x][cc.y] = args
                extra.append(
                    (
                        ttnn.KernelDescriptor(
                            kernel_source=nw.FABRIC_KERNEL,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]),
                            compile_time_args=[kind],
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_0,
                                noc=ttnn.NOC.NOC_1,
                                noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC,
                            ),
                        ),
                        f"fabric|{kind}|{core}|{len(args)};",
                    )
                )
                # (the port core's NOC1 belongs to the fabric sender: no loader there)
                assert not plan.taken(core, 1)
            plan.program(extra_kernels=extra, program=program)
            desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return desc, plan


@pytest.mark.timeout(86400)
@pytest.mark.parametrize(
    "cfg, device_params", [(c, c.device_params) for c in CASES], ids=[c.name for c in CASES], indirect=["device_params"]
)
def test_noc1_dram_write_wedge(mesh_device, cfg, device_params, request):
    if cfg.fabric != "none" and mesh_device.get_num_devices() < 2:
        pytest.skip("the fabric cases need a multi-chip mesh")
    launches = nw.launches_option(request, cfg.launches)
    s = nw.alloc_scratch(mesh_device)
    desc, plan = build(mesh_device, s, cfg)
    nw.log_config(f"noc1_dram_write {cfg.name} (expect {cfg.expect})", cfg, plan)
    nw.run_launches(
        mesh_device,
        s.tensors,
        desc,
        launches,
        nw.sync_option(request),
        f"noc1_dram_write {cfg.name}",
        exit_on_hang=nw.exit_on_hang_option(request),
    )
