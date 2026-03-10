# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Debug test for OLMo decode mode - trace Inf values through components.
"""

import torch
import ttnn
import pytest
from loguru import logger


def check_tensor(name, tt_tensor, mesh_device, cluster_shape):
    """Check tensor for NaN/Inf and print stats."""
    try:
        torch_tensor = ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
        )
        has_nan = torch.isnan(torch_tensor).any().item()
        has_inf = torch.isinf(torch_tensor).any().item()
        mean_val = torch_tensor.float().mean().item()
        std_val = torch_tensor.float().std().item()
        max_val = torch_tensor.float().abs().max().item()

        status = "OK" if not (has_nan or has_inf) else "BAD"
        logger.info(
            f"  [{status}] {name}: shape={list(torch_tensor.shape)}, mean={mean_val:.4e}, std={std_val:.4e}, max={max_val:.4e}, NaN={has_nan}, Inf={has_inf}"
        )
        return not (has_nan or has_inf)
    except Exception as e:
        logger.error(f"  [ERROR] {name}: {e}")
        return False


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", (32,))
def test_olmo_decode_debug(mesh_device, batch_size, device_params, reset_seeds):
    """Debug OLMo decode by checking intermediate values."""
    from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
    from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
    from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
    from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup

    logger.info("=== OLMo Decode Debug Test ===")

    # Initialize model config
    model_args = TtOlmoModelArgs(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=256,
    )
    logger.info(f"Model config: dim={model_args.dim}, n_layers={model_args.n_layers}")
    logger.info(f"USE_PREFETCHER: {model_args.model_config.get('USE_PREFETCHER', False)}")

    # Create sub_device_manager without prefetcher
    sub_device_manager = mesh_device.create_sub_device_manager(
        [ttnn.SubDevice([ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])])],
        0,
    )
    mesh_device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(0)

    # Load state dict
    state_dict = model_args.load_state_dict()
    logger.info(f"Loaded {len(state_dict)} weight tensors")

    # Initialize CCL
    tt_ccl = TT_CCL(
        mesh_device=mesh_device,
        model_args=model_args,
        worker_sub_device_id=worker_sub_device_id,
        mode="decode",
        allocate_prefill_buffers=False,
        is_olmo=True,
    )

    # Setup RoPE transformation matrices for YaRN
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Create decoder block for layer 0
    layer_num = 0
    num_layers = 1
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        layer_num=layer_num,
        n_layers=num_layers,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        transformation_mats=transformation_mats,
        paged_attention_config=None,
        prefetcher_setup=None,
        tt_ccl=tt_ccl,
    )
    logger.info(f"Created decoder block for layer {layer_num}")

    # Create input
    current_pos = torch.tensor([127 for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if batch_size > 1 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # Random input
    pt_input = (torch.rand(batch_size, 1, model_args.dim) * 2) - 1
    logger.info(f"Input: shape={list(pt_input.shape)}, mean={pt_input.mean():.4f}, std={pt_input.std():.4f}")

    # Prepare input tensor
    decode_input = model_args.prepare_residual_tensor_decode(
        pt_input.clone(),
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    # Get rotation matrices
    rot_mats = rope_setup.get_rm_rot_mats(current_pos)

    logger.info("=== Running decode forward ===")

    # Check input
    check_tensor("Input", decode_input, mesh_device, model_args.cluster_shape)

    # Run forward
    res = None
    tt_out, res = tt_model(
        decode_input,
        res,
        current_pos_tensor,
        rot_mats=rot_mats,
        mode="decode",
        page_table=None,
    )

    # Check output
    logger.info("=== Checking output ===")
    output_ok = check_tensor("Output", tt_out, mesh_device, model_args.cluster_shape)

    tt_ccl.close()

    assert output_ok, "Output contains NaN or Inf values"
    logger.info("=== Test PASSED ===")
