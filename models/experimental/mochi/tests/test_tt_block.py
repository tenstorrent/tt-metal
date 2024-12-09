import os
import pytest
import torch
import ttnn
import logging

from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricJointBlock
from models.experimental.mochi.block import TtAsymmetricJointBlock
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics
from models.demos.llama3.tt.llama_common import get_rot_transformation_mat
from models.experimental.mochi.tests.test_tt_attn import (
    load_model_weights,
    to_tt_tensor,
    to_torch_tensor,
    stack_cos_sin,
    PCC_REQUIRED,
    NUM_HEADS,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (512, 256),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "block_path",
    [
        "blocks.0",
    ],
)
def test_tt_block(mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, block_path):
    """Test TtAsymmetricJointBlock implementation by comparing with reference model."""
    state_dict, partial_state_dict = load_model_weights(block_path)

    block_kwargs = {
        "qk_norm": True,
        "qkv_bias": False,
        "out_bias": True,
        "attention_mode": "sdpa",
    }

    dim_x = 3072
    dim_y = 1536
    mlp_ratio_x = 4.0
    mlp_ratio_y = 4.0
    update_y = True
    multiple_of = 256
    ffn_dim_multiplier = None

    # Create reference model
    reference_model = AsymmetricJointBlock(
        hidden_size_x=dim_x,
        hidden_size_y=dim_y,
        num_heads=NUM_HEADS,
        mlp_ratio_x=mlp_ratio_x,
        mlp_ratio_y=mlp_ratio_y,
        update_y=update_y,
        device="cpu",
        **block_kwargs,
    )
    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    # Create TT model
    tt_model = TtAsymmetricJointBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=block_path,
        weight_cache_path=get_cache_path(os.environ.get("FAKE_DEVICE")),
        layer_num=0,
        dtype=ttnn.bfloat16,
        hidden_size_x=dim_x,
        hidden_size_y=dim_y,
        num_heads=NUM_HEADS,
        mlp_ratio_x=mlp_ratio_x,
        mlp_ratio_y=mlp_ratio_y,
        update_y=update_y,
        multiple_of=multiple_of,
        ffn_dim_multiplier=ffn_dim_multiplier,
        **block_kwargs,
    )

    # Create input tensors
    batch_size = 1
    x_input = torch.randn(batch_size, vision_seq_len, dim_x)
    y_input = torch.randn(batch_size, text_seq_len, dim_y)
    c_input = torch.randn(batch_size, dim_x)  # Conditioning tensor

    # Create RoPE tensors
    head_dim = dim_x // NUM_HEADS
    rope_cos = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)
    rope_sin = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)

    # Stack cos/sin for TT model
    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)

    # Create valid token indices
    total_seq_len = vision_seq_len + text_seq_len
    valid_token_indices = torch.arange(total_seq_len)
    max_seqlen_in_batch = total_seq_len

    # Convert inputs to TT tensors
    tt_x = to_tt_tensor(x_input.view(1, batch_size, vision_seq_len, dim_x), mesh_device)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, dim_y), mesh_device)
    tt_c = to_tt_tensor(c_input.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device)
    tt_trans_mat = to_tt_tensor(trans_mat, mesh_device)

    # Create packed indices
    packed_indices = {
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
        "cu_seqlens_kv": None,
    }

    logger.info("Run TtAsymmetricJointBlock forward")
    tt_x_out, tt_y_out = tt_model(
        tt_x,
        tt_c,
        tt_y,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
        packed_indices=packed_indices,
    )

    # Convert TT outputs to torch tensors
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device)
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device)

    # Get reference outputs
    ref_x, ref_y = reference_model(
        x_input, c_input, y_input, rope_cos=rope_cos, rope_sin=rope_sin, packed_indices=packed_indices
    )

    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        print(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all(pcc >= PCC_REQUIRED for _, pcc, _, _ in metrics)

    if passing:
        logger.info("TtAsymmetricJointBlock Passed!")
    else:
        logger.warning("TtAsymmetricJointBlock Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricJointBlock output does not meet PCC requirement {PCC_REQUIRED}"
