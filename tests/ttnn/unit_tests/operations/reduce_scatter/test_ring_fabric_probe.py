# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 (Ring topology) — FABRIC CONTRACT PROBE, run BEFORE implementation.

The verifier notes on Refinement 1 mandate probing whether the WRAP LINK
(device N-1 <-> device 0) is routable on the quietbox under FABRIC_1D before any
ring kernels are written:

  * the golden Ring cells run under the SAME device_params
    (fabric_config = FABRIC_1D) as Linear — the op must select behavior from the
    `topology` kwarg alone, so a ring algorithm only works if FABRIC_1D can
    service 1-hop wrap-link unicasts;
  * point_to_point verified its true-wraparound Ring path only under
    FABRIC_1D_RING (test_p2p_ring_confirm.py); its FABRIC_1D golden Ring cells
    use adjacent coords where the ring route degenerates to the linear one — so
    FABRIC_1D wrap-link routability on this box is genuinely UNKNOWN before this
    probe.

Three escalating probes (all 1-hop, per the verifier's "a 1-hop probe per
direction is enough"):

  1. HOST ROUTE MATH  — ccl_dm_route(.., Ring) picks the 1-hop wrap route for
     the (0, N-1) <-> (0, 0) pair (pure arithmetic + neighbor lookup; a failure
     here is a WRAP boundary-mode / fabric-node lookup gap).
  2. HOST CONNECTION  — setup_fabric_connection between the wrap-pair fabric
     node ids (control-plane link discovery; a TT_FATAL here means the fabric
     under FABRIC_1D has no router on the physical wrap link).
  3. DEVICE DATA      — a real 1-hop tile transfer across the wrap link in BOTH
     directions via the already-hardware-verified point_to_point op with
     topology=Ring (probe instrument only — reduce_scatter does not wrap it).

Run via the mandated CCL runner (never run_safe_pytest.sh):

  scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
      tests/ttnn/unit_tests/operations/reduce_scatter/test_ring_fabric_probe.py -v
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

MESH_SHAPE = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "1,4").split(","))

# The probes only exercise the FABRIC_1D contract the golden Ring cells pin.
FABRIC_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _wrap_pair(mesh_device):
    """The wrap-link coordinate pair on the 1-D line: (0, N-1) <-> (0, 0)."""
    n = prod(tuple(mesh_device.shape))
    if n < 3:
        pytest.skip("wraparound is indistinguishable from linear below 3 devices")
    return ttnn.MeshCoordinate(0, n - 1), ttnn.MeshCoordinate(0, 0), n


@pytest.mark.parametrize("device_params", [FABRIC_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_route_math(mesh_device):
    """Probe 1: ccl_dm_route(.., Ring) resolves the wrap pair as a 1-hop route."""
    last, first, n = _wrap_pair(mesh_device)

    # Diagnostics: what the route helper actually sees / returns across pairs.
    logger.info(
        f"DIAG mesh_device.shape={tuple(mesh_device.shape)} binding_doc={ttnn._ttnn.fabric.ccl_dm_route.__doc__!r}"
    )
    for s, r in [((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 0), (0, 3)), ((0, 3), (0, 0)), ((0, 1), (0, 3))]:
        for topo in (ttnn.Topology.Linear, ttnn.Topology.Ring):
            rt = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, ttnn.MeshCoordinate(*s), ttnn.MeshCoordinate(*r), topo)
            logger.info(
                f"DIAG route {s}->{r} {topo}: hops={rt.num_hops} is_forward={rt.is_forward} neighbor={rt.neighbor_id}"
            )

    # Forward around the wrap: (0, N-1) -> (0, 0). Line distance N-1; ring short
    # way = 1 hop across the wrap link.
    route_a = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, last, first, ttnn.Topology.Ring)
    logger.info(
        f"ring route {tuple(last)}->{tuple(first)}: num_hops={route_a.num_hops} is_forward={route_a.is_forward}"
    )
    assert route_a.num_hops == 1, f"expected 1-hop wrap route, got {route_a.num_hops}"

    # Backward around the wrap: (0, 0) -> (0, N-1).
    route_b = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, first, last, ttnn.Topology.Ring)
    logger.info(
        f"ring route {tuple(first)}->{tuple(last)}: num_hops={route_b.num_hops} is_forward={route_b.is_forward}"
    )
    assert route_b.num_hops == 1, f"expected 1-hop wrap route, got {route_b.num_hops}"

    # Sanity: under Linear topology the same pair is the full-line N-1 hops.
    route_lin = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, last, first, ttnn.Topology.Linear)
    assert route_lin.num_hops == n - 1


@pytest.mark.parametrize("device_params", [FABRIC_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_wrap_connection(mesh_device):
    """Probe 2: the fabric control plane can form a connection over the wrap link.

    setup_fabric_connection resolves the ethernet link between the two fabric
    nodes; under FABRIC_1D with no router on the physical wrap link this raises
    (loud host failure — the cheap signal, no device hang).
    """
    last, first, _ = _wrap_pair(mesh_device)
    fid_last = mesh_device.get_fabric_node_id(last)
    fid_first = mesh_device.get_fabric_node_id(first)
    core = ttnn.CoreCoord(0, 0)

    program_a = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
    args_a = ttnn.setup_fabric_connection(fid_last, fid_first, 0, program_a, core)
    logger.info(f"wrap connection {fid_last}->{fid_first}: {len(args_a)} rt args")
    assert len(args_a) > 0

    program_b = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
    args_b = ttnn.setup_fabric_connection(fid_first, fid_last, 0, program_b, core)
    logger.info(f"wrap connection {fid_first}->{fid_last}: {len(args_b)} rt args")
    assert len(args_b) > 0


def _round_up(v, m):
    return ((v + m - 1) // m) * m


@pytest.mark.parametrize("device_params", [FABRIC_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("direction", ["last_to_first", "first_to_last"])
def test_ring_wrap_data_1hop(mesh_device, direction):
    """Probe 3: real data crosses the wrap link (1 hop each direction).

    Direct minimal instrument (the point_to_point op does not compile against
    this lineage's ccl_routing_utils, so it cannot serve as the probe): one
    sender core fabric-writes ONE tile page from its local input shard into the
    wrap neighbour's output shard via FabricStreamSender + the (fixed)
    ccl_dm_route Ring wrap route, incs a counting semaphore; the receiver waits
    and re-arms. A pass proves the FABRIC_1D data plane services the wrap link
    end to end (routing + connection + packets + atomic inc).
    """
    KDIR = os.path.join(os.path.dirname(__file__), "probes", "kernels")
    CB_PAGE = 16

    last, first, n = _wrap_pair(mesh_device)
    sender, receiver = (last, first) if direction == "last_to_first" else (first, last)

    route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender, receiver, ttnn.Topology.Ring)
    assert route.num_hops == 1, f"wrap pair must be 1 hop under Ring (post-fix), got {route.num_hops}"

    torch.manual_seed(11)
    full = torch.randn((n, 1, 32, 32), dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    out = ttnn.from_torch(
        torch.zeros((n, 1, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)

    grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
    sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
    ttnn.synchronize_device(mesh_device)
    sem_addr = ttnn.get_global_semaphore_address(sem)

    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    target_noc = mesh_device.worker_core_from_logical_core(core)

    page_size = inp.buffer_page_size()
    l1_alignment = ttnn.get_l1_alignment()
    aligned_page_size = _round_up(page_size, l1_alignment)

    input_ta = list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    # --- sender program (at `sender` coord) ---
    cb_page = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_PAGE, data_format=inp.dtype, page_size=aligned_page_size)
        ],
    )
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[core.x][core.y] = [
        inp.buffer_address(),
        out.buffer_address(),
        page_size,
        route.num_hops,
        sem_addr,
        target_noc.x,
        target_noc.y,
    ]
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, "wrap_probe_sender.cpp"),
        core_ranges=core_set,
        compile_time_args=[CB_PAGE, l1_alignment] + input_ta + output_ta,
        runtime_args=sender_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    sender_program = ttnn.ProgramDescriptor(kernels=[sender_kernel], semaphores=[], cbs=[cb_page])
    # Fabric connection block appended AFTER ProgramDescriptor construction
    # (setup_fabric_connection mutates the program), has_forward/has_backward idiom.
    fid_sender = mesh_device.get_fabric_node_id(sender)
    fid_receiver = mesh_device.get_fabric_node_id(receiver)
    ref = sender_program.kernels[0].runtime_args[core.x][core.y]
    ref.append(int(route.is_forward))  # has_forward
    if route.is_forward:
        ref.extend(ttnn.setup_fabric_connection(fid_sender, fid_receiver, 0, sender_program, core))
    ref.append(int(not route.is_forward))  # has_backward
    if not route.is_forward:
        ref.extend(ttnn.setup_fabric_connection(fid_sender, fid_receiver, 0, sender_program, core))

    # --- receiver program (at `receiver` coord) ---
    receiver_rt = ttnn.RuntimeArgs()
    receiver_rt[core.x][core.y] = [sem_addr]
    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, "wrap_probe_receiver.cpp"),
        core_ranges=core_set,
        compile_time_args=[],
        runtime_args=receiver_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    receiver_program = ttnn.ProgramDescriptor(kernels=[receiver_kernel], semaphores=[], cbs=[])

    mpd = ttnn.MeshProgramDescriptor()
    mpd[ttnn.MeshCoordinateRange(sender, sender)] = sender_program
    mpd[ttnn.MeshCoordinateRange(receiver, receiver)] = receiver_program
    if hasattr(mpd, "semaphores"):
        mpd.semaphores = [sem]

    ttnn.generic_op([inp, out], mpd)
    ttnn.synchronize_device(mesh_device)

    shards_out = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    send_idx = sender[0] * mesh_device.shape[1] + sender[1]
    recv_idx = receiver[0] * mesh_device.shape[1] + receiver[1]
    assert_with_pcc(full[send_idx : send_idx + 1], shards_out[recv_idx], 0.999)
    logger.info(f"wrap-link data probe {tuple(sender)}->{tuple(receiver)} OK (1 hop, is_forward={route.is_forward})")
