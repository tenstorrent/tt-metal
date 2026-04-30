# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone CCL chain that reproduces an MoE stage's traffic on a (4, 2)
per-rank submesh. Does NOT invoke the deepseek_v3_b1 MoeOp.

See FAKE_MOE_PLAN.md (Phase 2) for design.

Tier 1 (SINGLE_BH host): smoke-test the chain on a 2D mesh, single rank.
  Caveat: the local single_pod/conftest.py's bh_2d_mesh_device_context
  picks mesh shape from ttnn.get_num_devices(), so on SINGLE_BH (32
  devices) the test actually runs on (4, 8), not (4, 2). The chain logic
  doesn't depend on the exact shape, so this still validates that the
  primitives compose. Real (4, 2) validation is in Tier 3.

Tier 3 (single-pod 16-rank): each MPI rank has TT_VISIBLE_DEVICES = 8
  specific devices, bh_2d_mesh_device_context opens a real (4, 2) per
  rank, FABRIC_2D_TORUS_Y + fabric_router_config(15232) + worker_l1_size
  match the demo CLI. This is the actual configuration the FakeMoeStage
  will run under in Phase 4.

(Tier 2 — DUAL_BH 2-rank with custom (4,2)-per-rank rank-binding —
skipped; would require a hand-crafted rank-binding YAML and offers little
beyond Tier 3 since we already have the single-pod rank-binding.)
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox._a2a_moe_helpers import (
    run_all_to_all_combine_test,
    run_all_to_all_dispatch_test,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.exabox.single_pod._fake_moe_helpers import (
    FakeMoeShapes,
    _setup_sub_devices,
    _teardown_sub_devices,
    step_all_reduce,
    step_reduce_to_one_substitute,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (4, 2) is the per-rank submesh shape used by every MoE rank in the
# blitz_decode single-pod descriptor. The chain inside this file is
# topology-aware and assumes this shape.
PER_RANK_MESH_SHAPE = (4, 2)
N_DEVICES = PER_RANK_MESH_SHAPE[0] * PER_RANK_MESH_SHAPE[1]  # = 8

# PCC threshold: chain accumulates bf16 noise across multiple all_reduce
# steps; 0.998 matches what we landed on in test_all_reduce_exabox.py for
# 32-device sums. (4, 2) sums are smaller, so 0.999 is comfortable.
PCC_THRESHOLD = 0.999

# Single-pod stage layout (see models/demos/deepseek_v3_b1/demo/pipeline.py
# create_single_pod_pipeline_configuration): rank index = pipeline slot.
#   slot 0      → EmbeddingStage          (no MoE-shaped CCL)
#   slots 1..3  → DenseDecoderStage       (no MoE-shaped CCL)
#   slots 4..13 → MoEDecoderStage         ← these are the 10 MoE slots
#   slot 14     → LMHeadStage             (no MoE-shaped CCL)
#   slot 15     → PassthroughStage(TOKEN) (no MoE-shaped CCL)
SINGLE_POD_MOE_SLOTS = set(range(4, 14))


# ---------------------------------------------------------------------------
# Step 1 of chain — broadcast
# ---------------------------------------------------------------------------


def _bcast_step(mesh_device, sender_tensor, sender_coord, mem_config, num_links, topology, sub_device_id):
    """Broadcast sender_tensor from sender_coord to all 8 devices in (4, 2)
    along cluster_axis=1 (the LINE axis). Mirrors test_broadcast_exabox.py's
    pattern: place sender_tensor at sender_idx along cluster_axis, zeros at
    other positions, then broadcast.

    Returns the tt output tensor; every device in the submesh should now
    hold sender_tensor.
    """
    cluster_axis = 1
    mesh_shape = tuple(mesh_device.shape)
    sender_idx = sender_coord[cluster_axis]
    num_along_axis = mesh_shape[cluster_axis]

    shards = [sender_tensor if k == sender_idx else torch.zeros_like(sender_tensor) for k in range(num_along_axis)]
    torch_input = torch.cat(shards, dim=0)

    # 1D-style mapper (proven to work for ttnn.broadcast on multi-device meshes —
    # see test_broadcast_exabox.py and AGENTS.md §"Per-op multi-host quirks").
    placements = [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)]
    mapper_mesh_shape = ttnn.MeshShape(1, num_along_axis)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        device=mesh_device,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(placements, mapper_mesh_shape),
        ),
    )

    tt_out = ttnn.broadcast(
        tt_input,
        ttnn.MeshCoordinate(sender_coord),
        cluster_axis=cluster_axis,
        topology=topology,
        memory_config=mem_config,
        num_links=num_links,
        subdevice_id=sub_device_id,
    )
    return tt_out


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def _verify_all_devices_match(tt_tensor, expected_torch, mesh_device, pcc_threshold=PCC_THRESHOLD):
    """Every device in the submesh should hold a tensor close to expected_torch
    (within pcc_threshold). Mirrors test_broadcast_exabox.py's verification
    pattern.
    """
    output_tensor_torch = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # ConcatMeshToTensor on dim 0 stacks the per-device tensors. Each slice
    # of size expected_torch.shape[0] should equal expected_torch.
    slice_size = expected_torch.shape[0]
    num_slices = output_tensor_torch.shape[0] // slice_size
    for i in range(num_slices):
        received = output_tensor_torch[i * slice_size : (i + 1) * slice_size]
        eq, mess = comp_pcc(received, expected_torch, pcc_threshold)
        assert eq, f"slice {i} mismatch: {mess}"


# ---------------------------------------------------------------------------
# Tests — Tier 1: SINGLE_BH (4, 2) submesh, single rank
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["SINGLE_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_2d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "shapes",
    [
        # Smoke-test sized (matches FakeMoeShapes defaults). Real demo uses
        # batch=256, hidden=7168 — those are too heavy for an iterative
        # debug loop. We can scale up in a separate parametrize once the
        # smaller case is green.
        pytest.param(
            FakeMoeShapes(batch=32, seq_len=2, hidden=1024, select_experts_k=4, experts_per_device=4),
            id="b32_h1024",
        ),
    ],
)
def test_fake_moe_step_bcast_4x2(
    mesh_device,
    topology,
    num_links,
    shapes,
):
    """Phase 2 step A: validate ttnn.broadcast alone on a (4, 2) submesh.

    Smaller than the existing test_broadcast_8x4 (which uses 8x4); confirms
    that the same pattern works on the per-rank (4, 2) shape we care about.
    """
    torch.manual_seed(0)
    sender_tensor = torch.rand(shapes.activation_shape).bfloat16()
    sender_coord = (0, 0)  # row 0, col 0 — leftmost device

    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
    sub_device_id, _, sub_device_manager = _setup_sub_devices(mesh_device)
    try:
        tt_out = _bcast_step(
            mesh_device,
            sender_tensor,
            sender_coord,
            mem_config,
            num_links,
            topology,
            sub_device_id,
        )
        _verify_all_devices_match(tt_out, sender_tensor, mesh_device)
    finally:
        _teardown_sub_devices(mesh_device, sub_device_manager)


# ---------------------------------------------------------------------------
# Tier 1 chain — all_reduce(axis 0) → all_reduce(axis 1) (= reduce-to-one)
# ---------------------------------------------------------------------------
#
# Note: we don't chain directly from ttnn.broadcast's output. Broadcast
# (which we already validated in step A and in test_broadcast_exabox.py)
# requires a 1D-style mapper (MeshShape(1, N)) and its output tensor only
# carries the (1, N) topology, not the full (4, 2). Subsequent CCL ops
# fail their topology asserts (num_devices > 1) when given that view.
#
# In the real MoE the activation post-bcast is materialized by the model
# code with proper (4, 2) topology before reaching the next CCL primitive.
# Our standalone test simulates that "post-bcast state" by directly
# placing the tensor on every device of the (4, 2) mesh via Replicate
# placements, then exercising the rest of the chain.
#
# a2a_dispatch / a2a_combine deferred to a follow-up commit — those need
# expert-mapping/metadata setup similar to test_all_to_all_*_exabox.py.
#
# Expected per-device value after the chain (input = post-bcast = X
# replicated on every device):
#   X                                     (initial replicated state)
# → 4 * X                                 (after all_reduce axis=0: 4 rows summed)
# → 4 * 2 * X = 8 * X                     (after all_reduce axis=1: 2 cols summed)
# Every device ends with 8 * X, equivalent to "every device has the sum
# over all 8 devices in the submesh", which is the result we'd read from
# any single device for the reduce-to-one step.


@pytest.mark.requires_device(["SINGLE_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_2d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(
            FakeMoeShapes(batch=32, seq_len=2, hidden=1024, select_experts_k=4, experts_per_device=4),
            id="b32_h1024",
        ),
    ],
)
def test_fake_moe_chain_bcast_allreduce_reduce_to_one_4x2(
    mesh_device,
    topology,
    num_links,
    shapes,
):
    """Phase 2 chain: post-bcast → all_reduce(axis 0) → all_reduce(axis 1).

    Validates that two all_reduces on the (4, 2) submesh compose and that
    bf16 accumulation stays within PCC threshold. The end state is
    equivalent to a "reduce-to-one" since every device has the full
    8-way sum, which is what the reduce-to-one root device would have."""
    torch.manual_seed(0)
    activation = torch.rand(shapes.activation_shape).bfloat16()

    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
    sub_device_id, _, sub_device_manager = _setup_sub_devices(mesh_device)
    try:
        # Materialize the post-bcast state directly: same tensor replicated
        # on every device of the (4, 2) submesh via 2D-aware mapper. This
        # gives us a tensor with full (4, 2) topology that subsequent CCL
        # ops can operate on (vs. broadcast's (1, N) output, which carries
        # a degenerate topology).
        mesh_shape = tuple(mesh_device.shape)
        tt_input = ttnn.from_torch(
            activation,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                    ttnn.MeshShape(*mesh_shape),
                ),
            ),
        )

        # Step 1: all_reduce on axis 0 (4 devices). Every device on a
        # column ends up with sum over its 4 rows = 4 * activation.
        tt_after_axis0 = step_all_reduce(tt_input, mesh_device, 0, mem_config, num_links, topology)

        # Step 2: all_reduce on axis 1 (2 devices). Every device ends up
        # with the full 8-way sum = 8 * activation.
        tt_after_reduce = step_all_reduce(tt_after_axis0, mesh_device, 1, mem_config, num_links, topology)

        # Verify against fp32-computed golden (avoids bf16 multiply error).
        expected_after_chain = (activation.float() * 8.0).bfloat16()

        # PCC relaxed to 0.998 because of bf16 noise from two reductions;
        # matches the 32-device-sum threshold in test_all_reduce_exabox.py.
        _verify_all_devices_match(tt_after_reduce, expected_after_chain, mesh_device, pcc_threshold=0.998)
    finally:
        _teardown_sub_devices(mesh_device, sub_device_manager)


# ---------------------------------------------------------------------------
# Tier 3 — single-pod 16-rank, FABRIC_2D_TORUS_Y + router config + l1_size
# ---------------------------------------------------------------------------
#
# Same chain as the Tier 1 chain test, but:
#   - requires_device(["QUAD_BH"]) — single-pod runs on a 4-host (32x4=128
#     device) cluster
#   - device_params match the demo CLI exactly: FABRIC_2D_TORUS_Y,
#     fabric_router_config(15232), worker_l1_size=1431568
#   - When run under tt-run with the single-pod rank-binding (16 ranks ×
#     8 visible devices), each rank gets a real (4, 2) submesh
#
# Invocation (matches the test_single_pod_pipeline.py launch sequence):
#   See FAKE_MOE_PLAN.md §"Phase 2 tier 3" for the tt-run command.


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(15232),
                "worker_l1_size": 1431568,
            },
            ttnn.Topology.Linear,
            id="fabric_2d_torus_y",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(
            FakeMoeShapes(batch=32, seq_len=2, hidden=1024, select_experts_k=4, experts_per_device=4),
            id="b32_h1024",
        ),
    ],
)
def test_fake_moe_chain_4x2_single_pod(
    mesh_device,
    topology,
    num_links,
    shapes,
):
    """Phase 2 Tier 3: validate the chain on the actual (4, 2) per-rank
    submesh under FABRIC_2D_TORUS_Y, the configuration the FakeMoeStage
    will run under in Phase 4."""
    torch.manual_seed(0)
    activation = torch.rand(shapes.activation_shape).bfloat16()

    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
    sub_device_id, _, sub_device_manager = _setup_sub_devices(mesh_device)
    try:
        mesh_shape = tuple(mesh_device.shape)
        tt_input = ttnn.from_torch(
            activation,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                    ttnn.MeshShape(*mesh_shape),
                ),
            ),
        )
        tt_after_axis0 = step_all_reduce(tt_input, mesh_device, 0, mem_config, num_links, topology)
        tt_after_reduce = step_all_reduce(tt_after_axis0, mesh_device, 1, mem_config, num_links, topology)

        expected_after_chain = (activation.float() * 8.0).bfloat16()
        _verify_all_devices_match(tt_after_reduce, expected_after_chain, mesh_device, pcc_threshold=0.998)
    finally:
        _teardown_sub_devices(mesh_device, sub_device_manager)


# ---------------------------------------------------------------------------
# Tier 3 (multi-block) — N stacked MoE iterations on the same (4, 2) submesh
# ---------------------------------------------------------------------------
#
# The single-pod Blitz pipeline has 10 MoE decoder stages (slots 4..13). Each
# of those stages runs ONE MoE iteration per pipeline tick on its own rank's
# (4, 2) submesh. To approximate the per-rank cumulative CCL workload of
# walking through all 10 stages, this test loops the chain N times on a
# single rank's submesh:
#
#   for i in range(num_blocks):
#       activation := all_reduce(axis=0, activation)   # 4× on each device
#       activation := all_reduce(axis=1, activation)   # ×2 → 8× per iter
#
# After N iterations, every device should hold (input × 8^N), which we check
# against an fp32 torch reference. The bf16 mantissa loses precision as
# values grow, but PCC stays high because every element scales by the same
# factor. We start from small initial values (rand × 1e-2) to keep
# bf16-representable values reasonable even at N=10 (peak ≈ 1e7).


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(15232),
                "worker_l1_size": 1431568,
            },
            ttnn.Topology.Linear,
            id="fabric_2d_torus_y",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(
            FakeMoeShapes(batch=32, seq_len=2, hidden=1024, select_experts_k=4, experts_per_device=4),
            id="b32_h1024",
        ),
    ],
)
@pytest.mark.parametrize("num_blocks", [1, 5, 10], ids=["1block", "5blocks", "10blocks"])
def test_fake_moe_chain_multi_block_4x2_single_pod(
    mesh_device,
    topology,
    num_links,
    shapes,
    num_blocks,
):
    """Phase 2 Tier 3 (multi-block, slot-aware): mirrors the real single-pod
    pipeline rank layout. Only the 10 MoE slots (ranks 4..13) run the
    MoE-shaped CCL chain; non-MoE ranks (0..3 = Embedding/Dense, 14/15 =
    LMHead/token passthrough) skip — exactly matching what those ranks do
    in the real pipeline (no MoE-shaped CCL).

    On each MoE rank the chain runs `num_blocks` times in sequence, where
    each iteration is `all_reduce(axis=0) → all_reduce(axis=1)` on the
    rank's (4, 2) submesh. num_blocks=10 simulates 10 decoded tokens worth
    of MoE traffic on each MoE rank — i.e. 10 ranks × 10 iterations =
    100 MoE chains running concurrently across the cluster, mirroring the
    real pipeline at steady state for a 10-token decode loop.
    """
    import time

    from loguru import logger

    my_mesh_id = mesh_device.get_system_mesh_id()
    if my_mesh_id not in SINGLE_POD_MOE_SLOTS:
        # Non-MoE rank: in the real pipeline, this slot is Embedding /
        # Dense / LMHead / token-passthrough — none of which run MoE-shaped
        # CCL. Skip so this rank's wall-clock matches the real pipeline.
        logger.info(
            "[multi-block] rank {} is a non-MoE slot (single-pod layout: MoE slots = 4..13); skipping chain",
            my_mesh_id,
        )
        pytest.skip(f"rank {my_mesh_id} is not a MoE slot in the single-pod layout")

    torch.manual_seed(0)
    # Small initial values keep cumulative 8^N scaling within bf16 range.
    activation = (torch.rand(shapes.activation_shape) * 1e-2).bfloat16()

    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
    sub_device_id, _, sub_device_manager = _setup_sub_devices(mesh_device)
    try:
        mesh_shape = tuple(mesh_device.shape)
        tt_state = ttnn.from_torch(
            activation,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                    ttnn.MeshShape(*mesh_shape),
                ),
            ),
        )

        block_durations_s = []
        for i in range(num_blocks):
            t0 = time.perf_counter()
            tt_after_axis0 = step_all_reduce(tt_state, mesh_device, 0, mem_config, num_links, topology)
            tt_state = step_all_reduce(tt_after_axis0, mesh_device, 1, mem_config, num_links, topology)
            ttnn.synchronize_device(mesh_device)
            dt = time.perf_counter() - t0
            block_durations_s.append(dt)
            logger.info(
                "[multi-block] rank {} (MoE slot) block {}/{} on {} took {:.3f}s",
                my_mesh_id,
                i + 1,
                num_blocks,
                mesh_shape,
                dt,
            )

        # Expected accumulated reduction factor: 8^num_blocks (4× axis-0 × 2× axis-1 per iter).
        scale = 8.0**num_blocks
        expected = (activation.float() * scale).bfloat16()
        # bf16 noise compounds across iterations — relax PCC slightly per block.
        # 0.998 holds for 1 block; drop to 0.99 by 10 blocks. Linear-in-N is
        # conservative; in practice the floor is held by per-element scale
        # consistency, so PCC stays high even when individual values lose
        # mantissa precision.
        pcc_threshold = max(0.99, 0.998 - 0.001 * (num_blocks - 1))
        _verify_all_devices_match(tt_state, expected, mesh_device, pcc_threshold=pcc_threshold)

        logger.info(
            "[multi-block] rank {} (MoE slot): {} blocks complete; total {:.3f}s, mean per-block {:.3f}s",
            my_mesh_id,
            num_blocks,
            sum(block_durations_s),
            sum(block_durations_s) / len(block_durations_s),
        )
    finally:
        _teardown_sub_devices(mesh_device, sub_device_manager)


# ---------------------------------------------------------------------------
# Tier 3 — a2a_dispatch + a2a_combine on the (4, 2) per-rank submesh
# ---------------------------------------------------------------------------
#
# These tests exercise the two MoE-specific CCL primitives in isolation on
# the single-pod fabric (FABRIC_2D_TORUS_Y, fabric_router_config(15232),
# worker_l1_size=1431568) — the same configuration the FakeMoeDecoderStage
# (Phase 3) will run under. The shared a2a helper (run_all_to_all_*) builds
# its own torch goldens and verifies per-token dispatch/combine correctness,
# so each test is self-contained.
#
# The existing test_all_to_all_*_exabox.py covers full 16x4 / 32x4 meshes
# under FABRIC_1D; this complements that coverage with the (4, 2) per-rank
# slice under the single-pod 2D-torus fabric.
#
# Status (2026-04-30): both tests hang at ttnn.all_to_all_dispatch /
# all_to_all_combine when run under FABRIC_2D_TORUS_Y on the (4, 2) per-rank
# slice — pytest's --timeout=240 doesn't fire because the hang is below the
# python layer (no ops complete after the tracy "start" signpost). The op
# itself works under FABRIC_1D on the unified 16x4 / 32x4 meshes (covered by
# test_all_to_all_*_exabox.py); the 2D-torus path on a per-rank submesh is
# not yet validated. These tests are kept as documentation + reproducer for
# the issue. The FakeMoeDecoderStage (Phase 3) avoids a2a entirely, using
# only all_reduce primitives that we've verified do work under the 2D-torus.
# Marked skip until the underlying op is fixed.

_SINGLE_POD_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    "fabric_router_config": create_fabric_router_config(15232),
    "worker_l1_size": 1431568,
}


@pytest.mark.skip(
    reason="ttnn.all_to_all_dispatch hangs under FABRIC_2D_TORUS_Y on (4, 2) per-rank "
    "submesh; reproduced 2026-04-30 — see comment above. Kept as documented reproducer."
)
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [pytest.param(_SINGLE_POD_DEVICE_PARAMS, ttnn.Topology.Linear, id="fabric_2d_torus_y")],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("seq_len, num_iters, warmup_iters", [(2, 2, 1)], ids=["s2"])
def test_fake_moe_a2a_dispatch_4x2_single_pod(
    mesh_device,
    topology,
    num_links,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
):
    """Phase 2 Tier 3 (a2a-dispatch): exercise ttnn.all_to_all_dispatch on
    the (4, 2) per-rank submesh under the single-pod fabric. cluster_axis=1
    (the LINE axis) matches the real MoE dispatch direction."""
    mesh_shape = tuple(mesh_device.shape)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode=False,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        output_memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        cluster_axis=1,
    )


@pytest.mark.skip(
    reason="ttnn.all_to_all_combine path shares the dispatch hang on FABRIC_2D_TORUS_Y "
    "(see test_fake_moe_a2a_dispatch_4x2_single_pod). Kept as documented reproducer."
)
@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [pytest.param(_SINGLE_POD_DEVICE_PARAMS, ttnn.Topology.Linear, id="fabric_2d_torus_y")],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param(PER_RANK_MESH_SHAPE, id="4x2_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("batches_per_device", [2])
@pytest.mark.parametrize("experts_per_device", [2])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("seq_len, num_iters", [(2, 2)], ids=["s2"])
@pytest.mark.parametrize("local_reduce", [False])
def test_fake_moe_a2a_combine_4x2_single_pod(
    mesh_device,
    topology,
    num_links,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    local_reduce,
):
    """Phase 2 Tier 3 (a2a-combine): exercise ttnn.all_to_all_combine on
    the (4, 2) per-rank submesh under the single-pod fabric. axis=1 mirrors
    the dispatch direction so combine is its inverse."""
    mesh_shape = tuple(mesh_device.shape)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = batches_per_device * devices
    experts = experts_per_device * devices

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        axis=1,
        batch=batch,
        seq=seq_len,
        local_reduce=local_reduce,
        experts=experts,
        select_experts_k=select_experts_k,
        hidden_size=hidden_size,
        num_iters=num_iters,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        output_memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
