# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal smoke: ttnn.experimental.all_reduce_async on (8,4) Blackhole Galaxy,
cluster_axis=1 (TP, the 4-wide column axis), Topology.Linear, across two fabric
configs. This isolates the exact op the MoE gate runs in its matmul path
(tt_moe_gate_prefill.py:_device_matmul) — strips away the gate matmul, grouped-gate,
dispatch and combine, leaving just a TP-axis all-reduce.

Why this test exists
--------------------
The full prefill block hangs at the gate's `all_reduce_async` under FABRIC_2D_TORUS_Y.
That op runs on cluster_axis = TP_AXIS = 1 (the column axis), which stays LINEAR even
under TORUS_Y — only the SP/row axis (mesh dim 0) is closed into a ring. So the op
correctly uses Topology.Linear; the ring lives on the *other* axis. The
ring-aware dispatch/combine fix (commit 2d07b9c0fd7) does not touch all_reduce_async,
and the full block under TORUS_Y was never run.

This test runs the SAME op, on the SAME axis, with the SAME Topology.Linear, varying
ONLY the fabric config:
  - FABRIC_2D        : baseline. Matches the `fabric2d-mesh-8x4` block variant, which
                       exercises this all-reduce and passes. Expected: PASS.
  - FABRIC_2D_TORUS_Y: the suspect. Same linear TP-axis all-reduce, but over the
                       torus-routed fabric. If this hangs (timeout) while FABRIC_2D
                       passes, the problem is all_reduce_async over TORUS_Y on the
                       non-wrapped axis — a fabric/CCL-level issue, NOT the topology
                       argument and NOT something the dispatch/combine fix addressed.

Topology is Linear for BOTH configs: cluster_axis=1 (4-wide) is never the ring axis.
Asking for Ring here would be wrong — get_boundary_mode (ccl_common.cpp) would see the
column coords span 0..3 and naively return WRAP, retaining Ring and routing over a
column wrap link that does not physically exist in TORUS_Y.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

# Mirrors the gate's logits shape: (sp_dim_per_chip, n_routed_experts), TILE layout, L1.
# DeepSeek-V3 routed-expert count; ISL_TOTAL sharded across the 8-long SP axis.
N_ROUTED_EXPERTS = 256
ISL_TOTAL = 1024


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="FABRIC_2D",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="FABRIC_2D_TORUS_Y",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.timeout(180)
def test_minimal_all_reduce(mesh_device, device_params, num_links):
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]
    isl_per_chip = ISL_TOTAL // sp_factor

    # The gate always runs the TP-axis all-reduce as Linear (the ring is on the SP axis).
    # cluster_axis=1 (4-wide) stays a line under both FABRIC_2D and FABRIC_2D_TORUS_Y.
    topology = ttnn.Topology.Linear

    logger.info(
        f"sp_factor={sp_factor}, tp_factor={tp_factor}, isl_per_chip={isl_per_chip}, "
        f"n_routed_experts={N_ROUTED_EXPERTS}, num_links={num_links}, topology={topology}, "
        f"fabric={device_params['fabric_config']}"
    )

    # Logits-shaped tensor: (sp_dim_per_chip, n_routed_experts), matching _device_matmul output.
    # Sharded along ISL on the SP axis (dim 0 -> mesh axis 0); REPLICATED along the TP axis
    # (mesh axis 1 -> None). Each TP device in a column thus holds an identical partial, so the
    # summed all-reduce result is deterministic: tp_factor * the per-chip slice.
    logits_shape = (ISL_TOTAL, N_ROUTED_EXPERTS)
    shard_dims = [None, None]
    shard_dims[sp_axis] = 0  # shard ISL (tensor dim 0) along the SP mesh axis
    shard_dims[tp_axis] = None  # replicate across the TP mesh axis

    torch.manual_seed(0)
    logits_full = torch.randn(logits_shape, dtype=torch.bfloat16)

    logits_input = ttnn.from_torch(
        logits_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    logger.info(f"Calling all_reduce_async on cluster_axis={tp_axis} (TP) with topology={topology}...")
    # Identical call shape to TtMoEGatePrefill._device_matmul (tt_moe_gate_prefill.py).
    logits_out = ttnn.experimental.all_reduce_async(
        logits_input,
        cluster_axis=tp_axis,
        mesh_device=mesh_device,
        num_links=num_links,
        math_op=ttnn.ReduceType.Sum,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=topology,
    )
    logger.info(f"Returned from all_reduce; out.shape={logits_out.shape}")

    logger.info("Calling synchronize_device...")
    ttnn.synchronize_device(mesh_device)
    logger.success("all_reduce_async completed under this fabric config — no hang.")

    # Structural check: all-reduce sums across TP (no concat) and the tensor is SP-sharded,
    # so each device keeps its per-chip shard shape (isl_per_chip, n_experts).
    expected_device_shape = (isl_per_chip, N_ROUTED_EXPERTS)
    assert (
        tuple(logits_out.shape) == expected_device_shape
    ), f"all-reduce per-device shape {logits_out.shape} != {expected_device_shape}"

    # Numerical check: result is TP-replicated and SP-sharded. Compose SP shards back (concat
    # dim 0), collapse the TP replicas, and compare against tp_factor * the original input.
    # Catches a degenerate "ran but reduced nothing/wrong" without a heavyweight reference.
    out_host = ttnn.to_torch(
        logits_out,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device,
            config=ttnn.MeshComposerConfig(
                dims=(0, -1),
                mesh_shape_override=ttnn.MeshShape(sp_factor, 1),
            ),
        ),
    )
    expected = (logits_full.float() * tp_factor)[: out_host.shape[0]]
    pcc = torch.corrcoef(torch.stack([out_host.float().flatten(), expected.flatten()]))[0, 1].item()
    logger.info(f"all-reduce PCC vs tp_factor*input = {pcc:.6f}")
    assert pcc > 0.99, f"all-reduce result PCC {pcc:.6f} too low — reduced wrong/nothing"
