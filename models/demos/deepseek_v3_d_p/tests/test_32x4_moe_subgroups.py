# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
MoE subgroups tests on the 32x4 quad-BH-Galaxy exabox (128 chips, 4 hosts).

Two parametrizations share the same test body:

1D submesh (subgroups-4x8-32x1-submesh-linear-1link)
    A 32x1 submesh (column 0) is carved from the 32x4 parent, spanning all 4 hosts
    along the row dimension (32 chips).  The linear mesh is partitioned into four 8x1
    dispatch subgroups, one per host.  num_dispatch_groups=1.

2D full mesh (subgroups-4x8x4-mesh-32x4-linear-1link)
    The full 32x4 parent mesh (128 chips) is used directly.  It is partitioned into
    four 8x4 dispatch subgroups (8 SP rows x 4 TP columns = 32 chips each).
    num_dispatch_groups=4.

Correctness strategy mirrors test_ttnn_moe_subgroups.py:
- Build a single-subgroup torch reference on one SP slice (dispatch_group_size rows).
- Tile the input num_dispatch_subgroups times along the device axis.
- Run TtMoe with num_dispatch_subgroups=4 on the working mesh.
- Split the output into num_dispatch_subgroups row-slices and verify each against
  the torch reference (PCC >= 0.96).  Set TT_MOE_SUBGROUPS_SKIP_EXPERTS=1 to
  bypass the routed-expert matmul (dispatch->combine connectivity smoke
  check only); default is 0 (full PCC).

Run command (set HOSTS before invoking):
    HOSTS=bh-glx-c07u02,bh-glx-c07u08,bh-glx-c08u02,bh-glx-c08u08
    METAL_HOME=/data/ianastasijevic/workspaces/main/tt-metal
    tt-run \\
      --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \\
      --mpi-args "--host $HOSTS --rankfile $METAL_HOME/c78_32x4_quad_galaxy_rankfile --bind-to none --tag-output" \\
      bash -c "export TT_METAL_HOME=$METAL_HOME && export PYTHONPATH=$METAL_HOME && \\
               source $METAL_HOME/python_env/bin/activate && cd $METAL_HOME && \\
               pytest models/demos/deepseek_v3_d_p/tests/test_32x4_moe_subgroups.py -v"
"""

import os
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
        pytest.param(
            1600,
            DeepSeekV3Config.EMB_SIZE,
            DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
            256,
            8,
            2,
            GateComputeMode.DEVICE,
        ),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, num_dispatch_subgroups, dispatch_group_size, submesh_shape",
    [
        pytest.param(
            (32, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            4,
            8,
            (32, 1),
            id="subgroups-4x8-32x1-submesh-linear-1link",
        ),
        pytest.param(
            (32, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            4,
            8,
            None,
            id="subgroups-4x8x4-mesh-32x4-linear-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_moe_subgroups_32x4(
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
    submesh_shape,
):
    """MoE with four dispatch subgroups on the 32x4 exabox mesh (1D submesh or full 2D mesh)."""
    mesh_device.disable_and_clear_program_cache()

    # Carve a submesh if requested (1D case), otherwise use the full mesh.
    if submesh_shape is not None:
        working_mesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
        assert tuple(working_mesh.shape) == submesh_shape, f"Expected submesh {submesh_shape}, got {working_mesh.shape}"
    else:
        working_mesh = mesh_device

    profiler.clear()
    profiler.start("test_moe_subgroups_32x4")

    random.seed(42)
    torch.manual_seed(42)

    mesh_config = extract_mesh_config(working_mesh)
    num_dispatch_groups = mesh_config.num_dispatch_groups

    n_sp_devices = working_mesh.shape[0]
    assert n_sp_devices == dispatch_group_size * num_dispatch_subgroups, (
        f"working_mesh row axis ({n_sp_devices}) must equal "
        f"dispatch_group_size ({dispatch_group_size}) * num_dispatch_subgroups ({num_dispatch_subgroups})"
    )

    subgroup_num_devices = dispatch_group_size * num_dispatch_groups

    logger.info(
        f"MoE subgroups test: parent_mesh={mesh_device.shape} working_mesh={working_mesh.shape} "
        f"num_dispatch_subgroups={num_dispatch_subgroups} dispatch_group_size={dispatch_group_size} "
        f"num_dispatch_groups={num_dispatch_groups} subgroup_num_devices={subgroup_num_devices}"
    )
    signpost(
        f"MoE subgroups 32x4 — working_mesh {working_mesh.shape}, seq_len={seq_len_per_chip}, "
        f"emb_dim={emb_dim}, experts={num_routed_experts}"
    )

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

    routed_expert_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim)
    shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)
    gate_weights = create_gate_weights(num_routed_experts, emb_dim)

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Reference input: one SP slice.  Tile num_dispatch_subgroups times for the full mesh input.
    x_single = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
    x_tiled = x_single.repeat(num_dispatch_subgroups, 1, 1)

    tt_x = ttnn.from_torch(
        x_tiled,
        mesh_mapper=ttnn.ShardTensor2dMesh(working_mesh, mesh_shape=working_mesh.shape, dims=(0, -1)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=working_mesh,
        dtype=ttnn.bfloat16,
    )

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
        mesh_device=working_mesh,
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
        routed_expert_weights_dtype=ttnn.bfloat8_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights=gate_weights,
        gate_fallback_mode=gate_fallback_mode,
        num_dispatch_subgroups=num_dispatch_subgroups,
    )
    ttnn.synchronize_device(working_mesh)

    skip_experts = os.environ.get("TT_MOE_SUBGROUPS_SKIP_EXPERTS", "0") == "1"
    logger.info(f"skip_experts={skip_experts}")

    tt_output, _ = tt_moe(tt_x, return_intermediates=False, skip_experts=skip_experts)
    ttnn.synchronize_device(working_mesh)

    tt_output_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(working_mesh), dtype=torch.bfloat16)
    logger.info(f"tt_output_host shape: {tt_output_host.shape}")

    finite_ratio = torch.isfinite(tt_output_host.float()).float().mean().item()
    logger.info(f"finite_ratio={finite_ratio}")

    if skip_experts:
        logger.info("skip_experts=True — dispatch->combine ran to completion; PCC check skipped.")
        profiler.end("test_moe_subgroups_32x4")
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
            logger.info(f"subgroup {sg_idx} PCC={pcc:.6f} >= {pcc_threshold}")
        else:
            logger.error(f"subgroup {sg_idx} PCC={pcc:.6f} < {pcc_threshold}")
            all_passed = False

    assert all_passed, "One or more subgroups failed PCC check. See logs."
    logger.info(f"All subgroups matched the {dispatch_group_size}-row torch reference")

    profiler.end("test_moe_subgroups_32x4")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
