# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf worker: production `dispatch` and `dispatch_fabric2d` on one set of inputs.

Both ops run in a single invocation, so Tracy captures DispatchDeviceOperation and
DispatchFabric2dDeviceOperation in one CSV and the ratio between them comes from one build, one
board state and one routing draw. Running them as separate invocations would fold board-to-board
and boot-to-boot variance into a number whose whole purpose is to be a ratio -- and on these
galaxies the realized fabric topology is not even stable across resets.

No PCC here: `test_prefill_dispatch_fabric2d.py` owns correctness, and a host-side comparison would
sit between the two ops in the capture.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_prefill_dispatch_fabric2d import (
    ROUTING_PROFILES,
    _draw_indices,
    _expert_dispatch_table,
    _mc_reach,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config, get_gate_outputs
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule

# Production prefill geometry: one 5120-token chunk over an 8-chip dispatch group.
CHUNK = 5 * 1024
DISPATCH_GROUP_SIZE = 8
DISPATCH_BUFFER_CAPACITY_FACTOR = 8
EMB_DIM = 7 * 1024
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8

# Enough launches that per-op mean is not dominated by the first, which pays program build.
ITERATIONS = 10


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len_per_chip", [CHUNK // DISPATCH_GROUP_SIZE, 64], ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("routing", list(ROUTING_PROFILES), ids=lambda r: r)
# Finite on purpose. A hang here does not fail the run, it wedges the board: the eth links do not
# retrain afterwards and recovering them needs privileges this account does not have.
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_perf_worker(mesh_device, device_params, num_links, seq_len_per_chip, routing):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"

    experts_per_chip, metadata_len, max_dispatch_buffer_token_size, _ = compute_constants(
        seq_len_per_chip,
        NUM_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        mesh_device.get_num_devices(),
        H,
        DISPATCH_BUFFER_CAPACITY_FACTOR,
        experts_per_chip_override=NUM_ROUTED_EXPERTS // G // H,
        emb_dim=EMB_DIM,
    )

    torch.manual_seed(42)
    table = _expert_dispatch_table(NUM_ROUTED_EXPERTS, H, G)
    in_group_share, hot_weight = ROUTING_PROFILES[routing]
    indices = _draw_indices(G, H, seq_len_per_chip, NUM_EXPERTS_PER_TOK, NUM_ROUTED_EXPERTS, in_group_share, hot_weight)
    realized = float((table[torch.arange(G).view(G, 1, 1, 1), indices] != -1).to(torch.float64).mean())

    offs = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    counts = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    region = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g],
            H,
            NUM_ROUTED_EXPERTS,
            experts_per_chip,
            seq_len_per_chip,
            NUM_EXPERTS_PER_TOK,
            expert_dispatch_table=table[g : g + 1],
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = torch.randn(H, G, seq_len_per_chip, EMB_DIM, dtype=torch.bfloat16)
    tt_x = shard(x, (0, 1), ttnn.bfloat16)
    # Both ops take UINT16 indices, so the A/B feeds them the identical tensor.
    tt_idx_u16 = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)
    # The two ops disagree about this tensor by design: fabric2d sizes runs it neither wrote nor
    # receives, so it needs every source chip's row replicated; production only ever needs its own.
    tt_offs_all = shard(offs, (None, 0), ttnn.int32)
    tt_offs_own = shard(offs.permute(1, 0, 2).reshape(H, G, NUM_ROUTED_EXPERTS), (0, 1), ttnn.int32)
    tt_counts = shard(counts[:, 0:1, :], (None, 0), ttnn.int32)
    tt_region = shard(region[:, 0:1, :], (None, 0), ttnn.int32)
    # Supplied by the test until masked_bincount emits it; see the note on routing setup.
    tt_reach = shard(
        _mc_reach(indices, table, offs, max_dispatch_buffer_token_size, G, H, seq_len_per_chip, NUM_EXPERTS_PER_TOK).to(
            torch.int32
        ),
        (None, 0),
        ttnn.int32,
    )

    logger.info(
        f"perf worker: mesh={tuple(mesh_device.shape)} seq={seq_len_per_chip} emb={EMB_DIM} "
        f"experts={NUM_ROUTED_EXPERTS} topk={NUM_EXPERTS_PER_TOK} epc={experts_per_chip} "
        f"capacity={max_dispatch_buffer_token_size} links={num_links} iters={ITERATIONS}"
    )
    logger.info(f"routing profile {routing}: picks landing in this dispatch group {100 * realized:.1f}%")
    # A drifting share is exactly how this measurement goes wrong without anyone noticing, in either
    # direction: too low reads as noise, too high overstates fan-out by roughly 3x.
    assert abs(realized - in_group_share) < 0.02, (
        f"routing profile {routing} asked for {100 * in_group_share:.1f}% of picks in this dispatch "
        f"group and drew {100 * realized:.1f}%"
    )

    production = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=H,
        experts_per_chip=experts_per_chip,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=EMB_DIM,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=ttnn.Topology.Ring,
    )
    tt_weights = shard(
        torch.zeros(H, G, seq_len_per_chip, NUM_EXPERTS_PER_TOK, dtype=torch.bfloat16), (0, 1), ttnn.bfloat16
    )

    def fabric2d(fanout):
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            tt_x,
            tt_idx_u16,
            tt_offs_all,
            tt_table,
            tt_counts,
            tt_region,
            fanout_reach=tt_reach if fanout else None,
            fanout=fanout,
            experts_per_chip=experts_per_chip,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # One untimed launch of each so the capture is not dominated by program build.
    #
    # The sync between the two ops is load-bearing, not hygiene. Program completion says nothing about
    # whether a fabric packet has reached its destination chip, and `dispatch` does not drain the
    # fabric before it retires; starting dispatch_fabric2d underneath that traffic hangs a relay
    # waiting on pages that never arrive. It takes a heavy routing draw to show up -- uniform and hot
    # pass, hottest deadlocks -- which is exactly the shape of a bug that survives a perf harness.
    production.forward(tt_x, tt_weights, tt_idx_u16, tt_offs_own, tt_table)
    ttnn.synchronize_device(mesh_device)
    fabric2d(False)
    fabric2d(True)
    ttnn.synchronize_device(mesh_device)

    signpost("dispatch_baseline")
    for _ in range(ITERATIONS):
        production.forward(tt_x, tt_weights, tt_idx_u16, tt_offs_own, tt_table)
    ttnn.synchronize_device(mesh_device)

    # Two transports, one routing draw, one board state. Store-and-forward moves the same bytes the
    # production op does, so it is expected at parity; multicast is where the link bytes come out.
    # Synchronising every iteration is not measurement hygiene, it is what keeps the op from
    # deadlocking. Back-to-back launches let a chip that finishes early start sending into a
    # neighbour that is still retiring the previous launch, and the arrival counters do not survive
    # that. Per-op device duration is unaffected by host pacing, so the numbers stay comparable.
    signpost("dispatch_fabric2d")
    for _ in range(ITERATIONS):
        fabric2d(False)
        ttnn.synchronize_device(mesh_device)
    ttnn.synchronize_device(mesh_device)

    signpost("dispatch_fabric2d_multicast")
    for _ in range(ITERATIONS):
        fabric2d(True)
        ttnn.synchronize_device(mesh_device)
    ttnn.synchronize_device(mesh_device)
    signpost("done")
