# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for full TtMoe forward using dispatch subgroups on Blackhole LoudBox.

The mesh is partitioned along axis 0 into `num_dispatch_subgroups` equal-sized subgroups,
each carrying a full replica of all routed experts. Token routing is scoped per subgroup
via the `num_dispatch_subgroups` parameter in dispatch/combine/offset_cumsum. Axis-1 CCLs
(gate all-reduce, shared-expert reduce-scatter, post-combine reduce-scatter,
pre-shared-expert all-gather) are orthogonal to the subgroup axis: no-ops on 8x1
(mesh_cols == 1), real work on 4x2 (TP=2 replication across expert dispatch groups).

Correctness strategy (same on 1D and 2D):
- Build a single-subgroup torch input of shape (dispatch_group_size, seq_len, emb_dim),
  run TorchMoe once as the reference.
- Tile that input `num_dispatch_subgroups` times along dim 0 to feed all subgroups.
- Shard to the mesh with ShardTensor2dMesh(dims=(0, -1)): row axis → subgroup chip slot,
  col axis → TP shard of emb_dim.
- Run TtMoe with num_dispatch_subgroups>1.
- Compose the output with get_tp_mesh_composer (dims=[0, -1]) and split into
  num_dispatch_subgroups row-slices; each must PCC-match the torch reference.
"""

import random

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor, gate_fallback_mode",
    [
        # Mirrors the smaller config from test_ttnn_moe.py so iteration stays fast.
        pytest.param(
            1600,
            DeepSeekV3Config.EMB_SIZE,
            DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
            64,
            8,
            2,
            GateComputeMode.HOST_ALL,
        ),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, num_dispatch_subgroups, dispatch_group_size",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            2,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="subgroups-2x4-linear-1link",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            2,
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="subgroups-2x2x2-mesh-4x2-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_moe_subgroups(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    gate_fallback_mode,
    num_dispatch_subgroups,
    dispatch_group_size,
):
    mesh_device.disable_and_clear_program_cache()

    profiler.clear()
    profiler.start("test_ttnn_moe_subgroups")

    random.seed(42)
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    n_sp_devices, n_tp_devices = mesh_device.shape
    assert n_sp_devices == dispatch_group_size * num_dispatch_subgroups, (
        f"mesh row axis ({n_sp_devices}) must equal "
        f"dispatch_group_size ({dispatch_group_size}) * num_dispatch_subgroups ({num_dispatch_subgroups})"
    )
    # num_dispatch_groups is the column-axis (EP) count; on 1D meshes it collapses to 1.
    mesh_config = extract_mesh_config(mesh_device)
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(
        f"TtMoe subgroups test: mesh={mesh_device.shape} "
        f"num_dispatch_subgroups={num_dispatch_subgroups} dispatch_group_size={dispatch_group_size} "
        f"num_dispatch_groups={num_dispatch_groups}"
    )

    signpost(
        f"TtMoe subgroups test - mesh {mesh_device.shape}, seq_len={seq_len_per_chip}, "
        f"emb_dim={emb_dim}, experts={num_routed_experts}"
    )

    # experts_per_chip is scoped to the subgroup. Each subgroup holds all N experts, distributed
    # across its `dispatch_group_size * num_dispatch_groups` chips (rows × EP-columns).
    subgroup_num_devices = dispatch_group_size * num_dispatch_groups
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        subgroup_num_devices,
        dispatch_group_size,
        capacity_factor,
    )
    logger.info(
        f"experts_per_chip={experts_per_chip} metadata_len={metadata_len} "
        f"max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}"
    )

    # One full expert set; TtRoutedExpert will tile it across subgroups internally.
    routed_expert_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim)
    shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)
    gate_weights = create_gate_weights(num_routed_experts, emb_dim)

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # One 4-chip torch reference, tiled for the 8-chip TTNN input.
    x_single = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    x_tiled = x_single.repeat(num_dispatch_subgroups, 1, 1)
    tt_x = ttnn.from_torch(
        x_tiled,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    # Torch reference on the 4-chip slice.
    torch_moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        gate_weights=gate_weights,
    )
    torch_output, _ = torch_moe(x_single, return_intermediates=True)
    logger.info(f"torch_output shape: {torch_output.shape}")

    tt_moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=num_links,
        topology=topology,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights=gate_weights,
        gate_fallback_mode=gate_fallback_mode,
        num_dispatch_subgroups=num_dispatch_subgroups,
    )
    ttnn.synchronize_device(mesh_device)

    # Debug gate: TT_MOE_SUBGROUPS_SKIP_EXPERTS=1 bypasses the routed-expert matmul to help
    # isolate whether a hang/mismatch lives in expert compute vs elsewhere. Default is False
    # (full pipeline, PCC required).
    import os

    skip_experts = os.environ.get("TT_MOE_SUBGROUPS_SKIP_EXPERTS", "0") == "1"
    if skip_experts:
        logger.warning("TT_MOE_SUBGROUPS_SKIP_EXPERTS=1 — routed-expert matmul bypassed; PCC assertion will be skipped")

    tt_output, _ = tt_moe(tt_x, return_intermediates=False, skip_experts=skip_experts)
    ttnn.synchronize_device(mesh_device)

    tt_output_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
    logger.info(f"tt_output_host shape: {tt_output_host.shape}")
    finite_ratio = torch.isfinite(tt_output_host.float()).float().mean().item()
    logger.info(f"finite_ratio={finite_ratio}")

    if skip_experts:
        logger.info("skip_experts=True — pipeline ran to completion; PCC check skipped.")
        return

    slices = torch.chunk(tt_output_host, num_dispatch_subgroups, dim=0)
    pcc_threshold = 0.96
    all_passed = True
    for sg_idx, tt_slice in enumerate(slices):
        logger.info(
            f"Validating subgroup {sg_idx} (shape {tt_slice.shape}) vs torch reference (shape {torch_output.shape})"
        )
        _, pcc = comp_pcc(torch_output.float(), tt_slice.float())
        if pcc >= pcc_threshold:
            logger.info(f"✅ subgroup {sg_idx} PCC={pcc:.6f} >= {pcc_threshold}")
        else:
            logger.error(f"❌ subgroup {sg_idx} PCC={pcc:.6f} < {pcc_threshold}")
            all_passed = False

    assert all_passed, "One or more subgroups failed PCC check. See logs."
    logger.info("✅ All subgroups matched the 4-chip torch reference")

    profiler.end("test_ttnn_moe_subgroups")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
