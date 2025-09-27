# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs


from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull

import os


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),
)
def test_qwen_layernorm_pcc(
    max_seq_len,
    batch_size,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    """
    Test PCC between TT and reference RMS norm for Qwen model.
    Loads attn_out and h tensors from TT decoder (.pt files) and
    reference tensors (x, at, fin) from reference model (.pth files).
    Passes TT tensors through ff_norm and compares with reference fin tensor.
    """
    dtype = ttnn.bfloat8_b

    # Check if the required .pth files exist
    attn_out_path = "attn_out.pt"
    h_path = "h.pt"
    x_path = "x.pth"
    at_path = "at.pth"
    fin_path = "fin.pth"

    if not os.path.exists(attn_out_path):
        pytest.skip(f"Required file {attn_out_path} not found. Run decoder test first to generate it.")
    if not os.path.exists(h_path):
        pytest.skip(f"Required file {h_path} not found. Run decoder test first to generate it.")
    if not os.path.exists(x_path):
        pytest.skip(f"Required file {x_path} not found. Run reference model first to generate it.")
    if not os.path.exists(at_path):
        pytest.skip(f"Required file {at_path} not found. Run reference model first to generate it.")
    if not os.path.exists(fin_path):
        pytest.skip(f"Required file {fin_path} not found. Run reference model first to generate it.")

    # Load the saved tensors
    logger.info("Loading saved tensors from .pth files")
    attn_out_torch = torch.load(attn_out_path)
    h_torch = torch.load(h_path)

    # Load reference tensors
    x_ref = torch.load(x_path)
    at_ref = torch.load(at_path)
    fin_ref = torch.load(fin_path)

    logger.info(f"Loaded attn_out shape: {attn_out_torch.shape}")
    logger.info(f"Loaded h shape: {h_torch.shape}")
    logger.info(f"Loaded reference x shape: {x_ref.shape}")
    logger.info(f"Loaded reference at shape: {at_ref.shape}")
    logger.info(f"Loaded reference fin shape: {fin_ref.shape}")

    # Initialize model args
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 2  # We need at least 2 layers to test layer 1

    state_dict = model_args.load_state_dict()

    # Extract CCL parameters directly without creating TT_CCL instance
    # From TT_CCL.__init__: self.num_cbs = 2, self.gather_idx = [0, 0]
    num_cbs = 2
    gather_idx = [0, 0]  # [cluster_axis_0, cluster_axis_1]

    # Create minimal semaphore handles (from TT_CCL.__init__ lines 57-72)
    sub_device_crs = model_args.sub_core_grids  # For decode mode
    gather_semaphore_handles = [[], []]
    for i in range(2):  # 2 cluster axes
        for _ in range(num_cbs):  # 2 circular buffers
            gather_semaphore_handles[i].append(ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0))

    # Extract RMSNorm parameters directly without creating TtRMSNorm instance
    layer_num = 1
    state_dict_prefix = model_args.get_state_dict_prefix("", layer_num)
    weight_key = "ffn_norm"

    # Extract epsilon (default from RMSNorm)
    epsilon = 1e-05

    # Extract weight directly from state_dict (following RMSNorm.__init__ logic)
    weight_name = f"{state_dict_prefix}{weight_key}.weight"
    SHARD_HEIGHT = 32  # From rmsnorm.py
    torch_weight = (
        state_dict[weight_name]
        .unsqueeze(0)
        .view(1, 1, model_args.dim)
        .reshape([1, 1, model_args.dim // SHARD_HEIGHT, SHARD_HEIGHT])
    )

    # Create distributed weight tensor (following RMSNorm distributed weight creation)
    weight_distributed = ttnn.as_tensor(
        torch_weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
    )

    # Output memory config
    output_mem_config = model_args.model_config["SHARDED_FF12_RING_MEMCFG"]

    # Setup memory configs for the sharded distributed call (from distributed_norm.py)
    core_grid_ln, grid_offset = (10, 2), ttnn.CoreCoord(1, 0)
    core_range = ttnn.CoreRange(
        grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
    )
    num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
    hidden_size_per_device_distributed_ln = model_args.dim // 4

    gather_in_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, hidden_size_per_device_distributed_ln // num_cores_ln),  # [1, 1, 32, 64]
        core_grid=ttnn.CoreRangeSet({core_range}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
        subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
        block_h=1,
        block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
        inplace=False,
    )

    # Setup reference model for layer 1 - just the ffn_norm
    layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", layer_num)
    ffn_norm_prefix = layer_prefix + "ffn_norm."
    partial_state_dict = {k[len(ffn_norm_prefix) :]: v for k, v in state_dict.items() if k.startswith(ffn_norm_prefix)}
    reference_ffn_norm = RefRMSNorm(dim=model_args.dim, eps=epsilon)  # Use the same epsilon
    reference_ffn_norm.load_state_dict(partial_state_dict)

    logger.info("Converting torch tensors to TT tensors")

    # Convert attn_out to TT tensor format (similar to how it's done in decoder)
    # Expand to full batch size and proper shape
    attn_out_expanded = attn_out_torch.expand(1, 1, batch_size, model_args.dim).contiguous()
    h_expanded = h_torch.expand(1, 1, batch_size, model_args.dim).contiguous()

    # Convert to TT tensors with the gather memory config for sharded distributed call
    # For the sharded distributed call, we need separate x and res tensors
    attn_out_tt = ttnn.from_torch(
        attn_out_expanded,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
        memory_config=gather_in_mem_cfg,
    )

    h_tt = ttnn.from_torch(
        h_expanded,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
        memory_config=gather_in_mem_cfg,
    )

    logger.info("Running TT fused RMSNorm forward pass directly")
    # Extract the core call from tt_sharded_distributed_rmsnorm (lines 1232-1253)
    # This is the actual ttnn.fused_rms_1_1_32_8192 call without the TT_CCL wrapper
    cluster_axis = 1
    semaphore = gather_semaphore_handles[cluster_axis][gather_idx[cluster_axis]]
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, 64),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    persistent_buffer = ttnn.from_torch(
        torch.zeros((1, 1, 32, 64)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=tt_stats_sharded_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # tt_ccl.all_gather_buffers.get("LAYERNORM", None) - set to None for minimal test

    tt_output = ttnn.fused_rms_1_1_32_8192(
        attn_out_tt,  # inp parameter
        ln_prg_cfg,  # ln_sharded_progcfg
        cluster_axis,
        mesh_device,  # tt_ccl.mesh_device
        semaphore,
        topology=model_args.model_config["CCL_TOPOLOGY"],  # ccl_topology
        residual_input_tensor=h_tt,  # res parameter
        num_links=1,
        epsilon=epsilon,
        weight=weight_distributed,  # gamma parameter
        stats=persistent_buffer,
        memory_config=output_mem_config,
        use_noc1_only=False,
    )

    # Convert TT output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )[:, :1, :, :]

    logger.info("Computing reference solution")
    # Use the reference fin tensor directly from the reference model
    # This is the output of ffn_norm(h) where h = x + at (from line 342 in model.py)
    reference_output = fin_ref
    reference_output = reference_output.view(1, 1, batch_size, model_args.dim)

    # Verify our understanding by computing it manually as well
    h_ref = x_ref + at_ref  # This should match the h computation in reference model
    manual_reference = reference_ffn_norm(h_ref).view(1, 1, batch_size, model_args.dim)

    logger.info("Verifying reference computation consistency")
    ref_consistency_pcc = comp_pcc(reference_output, manual_reference)
    logger.info(f"Reference consistency PCC: {ref_consistency_pcc[1]}")

    logger.info("Computing PCC between TT and reference results")
    # Compare the results
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Qwen LayerNorm PCC Test Passed!")
    else:
        logger.warning("Qwen LayerNorm PCC Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Got: {pcc_message}"
