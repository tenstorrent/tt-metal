# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtSharedExpert module (TP=4).

Compares TorchExpert (reference) against TtSharedExpert (multi-chip TTNN)
to verify correctness of multi-chip sharding and CCL operations.
"""

from contextlib import contextmanager

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import ACTIVATION_SILU, ACTIVATION_SITU, TtSharedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.tt_transformers.tt.ccl import get_num_links
from tests.ttnn.utils_for_testing import assert_with_pcc


@contextmanager
def shared_expert_sub_device(mesh_device):
    """Split the Tensix grid the way TtMoe does: dispatch takes the first row, the expert the rest.

    The shared expert only ever runs confined like this -- tt_moe.py carves the grid so the two can
    overlap on chip -- so exercising it on the full grid would test matmul program configs that
    nothing outside this file ever builds.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    dispatch = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 0))})
    shared = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([dispatch]), ttnn.SubDevice([shared])], 0)
    mesh_device.load_sub_device_manager(manager)
    try:
        yield ttnn.SubDeviceId(1), shared
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, activation",
    [
        # Every case runs the one prefill ISL, so the matmul program configs come out shaped the
        # way prefill builds them.
        (PREFILL_CHUNK_TOKENS_PER_CHIP, KimiK26Config.EMB_SIZE, KimiK26Config.MOE_INTERMEDIATE_SIZE, ACTIVATION_SILU),
        # Kimi-K3: one shared-expert MLP at moe_intermediate_size * num_shared_experts = 3072 * 2.
        # Worth its own case because every prior model has num_shared_experts == 1, so hidden_dim and
        # the shared intermediate coincided and 6144 was never exercised here.
        (
            PREFILL_CHUNK_TOKENS_PER_CHIP,
            KimiK3Config.EMB_SIZE,
            KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
            ACTIVATION_SILU,
        ),
        # ...and the same shape on the activation the K3 checkpoint actually uses. The sub-device
        # below confines the expert the way TtMoe does, so this exercises ttnn.situ_glu in its
        # sub_core_grids form -- the only form the model ever builds.
        (
            PREFILL_CHUNK_TOKENS_PER_CHIP,
            KimiK3Config.EMB_SIZE,
            KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
            ACTIVATION_SITU,
        ),
    ],
    # Ids name what differs from the case above (the 6144 shared intermediate, the activation).
    ids=["isl_5k", "isl_5k-k3-6144", "isl_5k-k3-6144-situ"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (1, 4),
            torus_x_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="torus-x-1x4",
        ),
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        # BH Galaxy. A Blackhole box accepts no mesh smaller than all 32 devices, so this is the
        # only param where the SiTU cases can run at all -- SiTU needs ttnn.softcap, Blackhole-only.
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_shared_expert_pcc(
    mesh_device,
    device_params,
    seq_len_per_chip: int,
    emb_dim: int,
    hidden_dim: int,
    activation: str,
    num_links: int,
):
    """
    Test TtSharedExpert PCC against TorchExpert reference.

    This test verifies:
    1. Correct weight sharding (gate_proj/up_proj on -1, down_proj on -2)
    2. Proper all-gather before matmuls
    3. The GLU activation — SiLU fused into the gate matmul, or SiTU-GLU over both raw accumulators
    4. Proper reduce-scatter after final matmul
    5. Output matches torch reference with PCC > 0.97
    """
    if activation == ACTIVATION_SITU and not is_blackhole():
        pytest.skip("SiTU-GLU needs ttnn.softcap, which is Blackhole-only")

    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b
    topology = per_axis_topology(device_params["fabric_config"])[1]
    # Only read on the SiTU path; TorchExpert and TtSharedExpert must be given the same pair or the
    # comparison silently measures two different activations.
    situ_betas = dict(
        situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
    )

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.debug(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.debug(f"{seq_len_per_chip=} {emb_dim=} {hidden_dim=} {activation=}")

    # Add Tracy signpost for profiling
    signpost(f"SharedExpert PCC test - {mesh_shape=} {seq_len_per_chip=} {activation=} {num_links=} {topology=}")

    # Query available ethernet links
    actual_num_links = get_num_links(mesh_device, cluster_axis=1)  # Query along mesh columns
    logger.debug(f"Available ethernet links along mesh columns: {actual_num_links}")
    logger.debug(f"Using num_links={num_links}, topology={topology}")

    # ========================================
    # Step 1: Create PyTorch reference model
    # ========================================
    logger.debug("Creating TorchExpert reference")
    torch_model = TorchExpert(emb_dim, hidden_dim, activation=activation, **situ_betas)

    # Extract weights for TTNN model
    torch_weights = {
        "gate_proj": torch_model.gate_proj.data,
        "up_proj": torch_model.up_proj.data,
        "down_proj": torch_model.down_proj.data,
    }

    with shared_expert_sub_device(mesh_device) as (subdevice_id, subdevice_cores):
        # ========================================
        # Step 2: Create TTNN model with same weights
        # ========================================
        logger.debug("Creating TtSharedExpert with same weights")
        tt_model = TtSharedExpert(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=torch_weights,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            activation=activation,
            subdevice_id=subdevice_id,
            subdevice_cores=subdevice_cores,
            **situ_betas,
        )

        # ========================================
        # Step 3: Create input tensor
        # ========================================
        # 3D input matching test_ttnn_moe.py convention (post all-gather):
        #   shape = [dispatch_group_size, seq_len_per_chip, emb_dim]
        # Sharded along dim 0 across mesh rows (DP), replicated across mesh cols (TP).
        dispatch_group_size = mesh_shape[0]
        torch_input = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.float32)
        logger.debug(f"Created torch input: {torch_input.shape}")

        tt_input = ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=activations_dtype,
        )
        logger.debug(f"Created ttnn input (SP-sharded, TP-replicated): {tt_input.shape}")

        # ========================================
        # Step 4: Run forward passes
        # ========================================
        logger.debug("Running torch forward pass")
        torch_output = torch_model(torch_input)
        logger.debug(f"Torch output shape: {torch_output.shape}")

        logger.debug("Running ttnn forward pass")
        tt_output = tt_model(tt_input)
        logger.debug(f"TTNN output shape (sharded): {tt_output.shape}")

        # ========================================
        # Step 5: Convert TTNN output back to torch and compare
        # ========================================
        logger.debug("Converting TTNN output to torch for comparison")
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
        )
        logger.debug(f"TTNN output converted to torch: {tt_output_torch.shape}")

        # Compare with PCC
        logger.debug("Comparing outputs with PCC")
        pcc_passed, pcc_message = assert_with_pcc(
            torch_output.to(torch.float32),
            tt_output_torch.to(torch.float32),
            pcc=0.999,
        )

        logger.debug(f"PCC comparison: {pcc_message}")
        assert pcc_passed, f"PCC test failed: {pcc_message}"

        logger.debug("PCC test passed!")
