import torch
import pytest
from loguru import logger
import os
import ttnn
from models.utility_functions import skip_for_grayskull
from models.experimental.mochi.tt.dit.attention import AsymmetricAttention as TtAsymmetricAttention
from models.experimental.mochi.tt.common import (
    get_mochi_dir,
    get_cache_path,
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
    stack_cos_sin,
)

from models.experimental.mochi.tests.dit.common import (
    load_model_weights,
    NUM_HEADS,
    HEAD_DIM,
)
from models.demos.llama3.tt.llama_common import get_rot_transformation_mat

from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricAttention as RefAsymmetricAttention

# Common test configurations
PCC_REQUIRED = 0.99


def create_models(mesh_device, state_dict, partial_state_dict, attn_path, vision_seq_len, dim_x, dim_y, update_y=True):
    """Initialize both reference and TT models."""
    reference_model = RefAsymmetricAttention(
        dim_x=dim_x,
        dim_y=dim_y,
        num_heads=NUM_HEADS,
        qkv_bias=False,
        qk_norm=True,
        update_y=update_y,
        out_bias=True,
        attention_mode="sdpa",
        softmax_scale=None,
        device="cpu",
        qkv_proj_lora_rank=0,
        qkv_proj_lora_alpha=0,
        qkv_proj_lora_dropout=0.0,
        out_proj_lora_rank=0,
        out_proj_lora_alpha=0,
        out_proj_lora_dropout=0.0,
    )
    reference_model.load_state_dict(partial_state_dict)

    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtAsymmetricAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=attn_path,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        dtype=ttnn.bfloat16,
        vision_seq_len=vision_seq_len,
        dim_x=dim_x,
        dim_y=dim_y,
        num_heads=NUM_HEADS,
        qkv_bias=False,
        qk_norm=True,
        update_y=update_y,
        out_bias=True,
        attention_mode="sdpa",
        softmax_scale=None,
        qkv_proj_lora_rank=0,
        qkv_proj_lora_alpha=0,
        qkv_proj_lora_dropout=0.0,
        out_proj_lora_rank=0,
        out_proj_lora_alpha=0,
        out_proj_lora_dropout=0.0,
    )
    return reference_model, tt_model


def validate_outputs(tt_outputs, ref_outputs, test_name):
    """Validate and compare model outputs."""
    metrics = []
    for tt_out, ref_out, name in zip(tt_outputs, ref_outputs, ["Q", "K", "V"]):
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        logger.info(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all(pcc >= PCC_REQUIRED for _, pcc, _, _ in metrics)

    if passing:
        logger.info(f"{test_name} Passed!")
    else:
        logger.warning(f"{test_name} Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"{test_name} output does not meet PCC requirement {PCC_REQUIRED}"


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (256,),
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
    "attn_path, dim_x, dim_y",
    [
        ("blocks.0.attn", 3072, 1536),
    ],
)
def test_tt_attn_qkv_y(mesh_device, seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y):
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y)

    batch_size = 1
    # Create input tensor
    torch_input = torch.randn(batch_size, seq_len, dim_y)
    tt_input = to_tt_tensor(torch_input.view(1, batch_size, seq_len, dim_y), mesh_device)

    logger.info("Run TtAsymmetricAttention QKV_Y")
    tt_q, tt_k, tt_v = tt_model.run_qkv_y(tt_input)

    # Convert TT outputs to torch tensors
    tt_q_torch = to_torch_tensor(tt_q, mesh_device, dim=-3)
    tt_k_torch = to_torch_tensor(tt_k, mesh_device, dim=-3)
    tt_v_torch = to_torch_tensor(tt_v, mesh_device, dim=-3)

    # Get reference outputs
    with torch.no_grad():
        ref_q, ref_k, ref_v = reference_model.run_qkv_y(torch_input)

        # Reshape to [B, H, L, D]
        ref_q = ref_q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_k = ref_k.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_v = ref_v.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    validate_outputs([tt_q_torch, tt_k_torch, tt_v_torch], [ref_q, ref_k, ref_v], "TtAsymmetricAttention QKV_Y")


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (22 * 256 * 8, 256),  # Tests when X doesn't need padding
        # (44520, 118),  # Tests when X needs padding
        # (2048, 128)
        # (240, 118)
        # (44520, 118),
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
    "attn_path, dim_x, dim_y, update_y",
    [
        ("blocks.0.attn", 3072, 1536, True),
        # ("blocks.47.attn", 3072, 1536, False),
    ],
)
def test_tt_attn_prepare_qkv(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y, update_y
):
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, attn_path, vision_seq_len, dim_x, dim_y, update_y
    )

    # Create input tensors
    batch_size = 1
    x_input = torch.randn(batch_size, vision_seq_len, dim_x)
    y_input = torch.randn(batch_size, text_seq_len, dim_y)
    scale_x = torch.randn(batch_size, dim_x)
    scale_y = torch.randn(batch_size, dim_y)

    # Create RoPE tensors
    head_dim = dim_x // NUM_HEADS  # 24 heads
    rope_cos = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)
    rope_sin = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)

    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    # Create valid token indices
    total_seq_len = vision_seq_len + text_seq_len
    valid_token_indices = torch.arange(total_seq_len)
    max_seqlen_in_batch = total_seq_len

    tt_x = to_tt_tensor(x_input.view(1, batch_size, vision_seq_len, dim_x), mesh_device, shard_dim=-2)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, dim_y), mesh_device)
    tt_scale_x = to_tt_tensor(scale_x.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_scale_y = to_tt_tensor(scale_y.view(batch_size, 1, 1, dim_y), mesh_device)
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device, shard_dim=-2)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device, shard_dim=-2)

    trans_mat = get_rot_transformation_mat(None)
    trans_mat_tt = to_tt_tensor(trans_mat, mesh_device)

    logger.info("Run TtAsymmetricAttention prepare_qkv")
    tt_q_x, tt_k_x, tt_v_x, tt_q_y, tt_k_y, tt_v_y = tt_model.prepare_qkv(
        tt_x,
        tt_y,
        scale_x=tt_scale_x,
        scale_y=tt_scale_y,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=trans_mat_tt,
    )

    # Convert TT outputs to torch tensors
    tt_q_x_torch = to_torch_tensor(tt_q_x, mesh_device, dim=-3)
    tt_k_x_torch = to_torch_tensor(tt_k_x, mesh_device, dim=-3)
    tt_v_x_torch = to_torch_tensor(tt_v_x, mesh_device, dim=-3)
    tt_q_y_torch = to_torch_tensor(tt_q_y, mesh_device, dim=-3)
    tt_k_y_torch = to_torch_tensor(tt_k_y, mesh_device, dim=-3)
    tt_v_y_torch = to_torch_tensor(tt_v_y, mesh_device, dim=-3)

    # Concat for comparison
    tt_q_torch = torch.cat([tt_q_x_torch, tt_q_y_torch], dim=2)
    tt_k_torch = torch.cat([tt_k_x_torch, tt_k_y_torch], dim=2)
    tt_v_torch = torch.cat([tt_v_x_torch, tt_v_y_torch], dim=2)

    # Get reference outputs
    with torch.no_grad():
        ref_q, ref_k, ref_v = reference_model.prepare_qkv(
            x_input,
            y_input,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            valid_token_indices=valid_token_indices,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        # Reshape reference to be B, NH, S, D
        ref_q = ref_q.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_k = ref_k.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_v = ref_v.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    validate_outputs([tt_q_torch, tt_k_torch, tt_v_torch], [ref_q, ref_k, ref_v], "TtAsymmetricAttention prepare_qkv")


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (43 * 1024, 256),
        (44520, 118),  # Padding case
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
    "attn_path, dim_x, dim_y, update_y",
    [
        ("blocks.0.attn", 3072, 1536, True),
        ("blocks.47.attn", 3072, 1536, False),
    ],
)
def test_tt_attn_run_attention(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y, update_y
):
    """Test run_attention implementation by comparing with reference model."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y, update_y
    )

    batch_size = 1
    head_dim = dim_x // NUM_HEADS

    # Create input tensors
    seq_len = vision_seq_len + text_seq_len
    torch_q_x = torch.randn(batch_size, NUM_HEADS, vision_seq_len, head_dim)
    torch_k_x = torch.randn(batch_size, NUM_HEADS, vision_seq_len, head_dim)
    torch_v_x = torch.randn(batch_size, NUM_HEADS, vision_seq_len, head_dim)
    torch_q_y = torch.randn(batch_size, NUM_HEADS, text_seq_len, head_dim)
    torch_k_y = torch.randn(batch_size, NUM_HEADS, text_seq_len, head_dim)
    torch_v_y = torch.randn(batch_size, NUM_HEADS, text_seq_len, head_dim)

    # Convert to TT tensors
    tt_q_x = to_tt_tensor(torch_q_x, mesh_device, shard_dim=-3)
    tt_k_x = to_tt_tensor(torch_k_x, mesh_device, shard_dim=-3)
    tt_v_x = to_tt_tensor(torch_v_x, mesh_device, shard_dim=-3)
    tt_q_y = to_tt_tensor(torch_q_y, mesh_device, shard_dim=-3)
    tt_k_y = to_tt_tensor(torch_k_y, mesh_device, shard_dim=-3)
    tt_v_y = to_tt_tensor(torch_v_y, mesh_device, shard_dim=-3)

    logger.info("Run TtAsymmetricAttention run_attention")
    tt_out, tt_out_joint = tt_model.run_attention(
        tt_q_x,
        tt_k_x,
        tt_v_x,
        tt_q_y,
        tt_k_y,
        tt_v_y,
        B=batch_size,
    )
    # Convert TT output to torch tensor
    tt_out_torch = to_torch_tensor(tt_out, mesh_device, dim=-1)
    tt_out_joint_torch = to_torch_tensor(tt_out_joint, mesh_device, dim=-1)
    tt_out_joint_torch = to_torch_tensor(tt_out_joint, mesh_device, dim=-1)

    tt_out_torch = torch.cat([tt_out_torch, tt_out_joint_torch], dim=2)

    # Get reference output
    with torch.no_grad():
        # Reshape inputs for reference model
        torch_q = torch.cat([torch_q_x, torch_q_y], dim=2)
        torch_k = torch.cat([torch_k_x, torch_k_y], dim=2)
        torch_v = torch.cat([torch_v_x, torch_v_y], dim=2)
        ref_q = torch_q.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)
        ref_k = torch_k.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)
        ref_v = torch_v.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)

        ref_out = reference_model.run_attention(ref_q, ref_k, ref_v, B=batch_size)

        # Reshape to match TT output
        ref_out = ref_out.reshape(1, batch_size, seq_len, -1)

    # Validate outputs
    pcc, mse, mae = compute_metrics(ref_out, tt_out_torch)
    logger.info(f"Attention - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = pcc >= PCC_REQUIRED

    if passing:
        logger.info("TtAsymmetricAttention run_attention Passed!")
    else:
        logger.warning("TtAsymmetricAttention run_attention Failed!")
        logger.error(f"Attention failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricAttention run_attention output does not meet PCC requirement {PCC_REQUIRED}"


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (43 * 1024, 256),
        (44520, 118),
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
    "attn_path, dim_x, dim_y, update_y",
    [
        ("blocks.0.attn", 3072, 1536, True),
        ("blocks.47.attn", 3072, 1536, False),
    ],
)
def test_tt_attn_post_attention(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y, update_y
):
    """Test post_attention implementation by comparing with reference model."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y, update_y
    )

    batch_size = 1
    total_seq_len = vision_seq_len + text_seq_len

    # Create input tensor - simulating attention output
    torch_out = torch.randn(batch_size * total_seq_len, dim_x)

    # Convert to TT tensor
    torch_x = torch_out[:vision_seq_len].view(1, batch_size, vision_seq_len, dim_x)
    torch_y = torch_out[vision_seq_len:].view(1, batch_size, text_seq_len, dim_x)
    tt_x = to_tt_tensor(torch_x, mesh_device, shard_dim=-1)
    tt_y = to_tt_tensor(torch_y, mesh_device, shard_dim=-1)

    logger.info("Run TtAsymmetricAttention post_attention")
    tt_x, tt_y = tt_model.post_attention(
        tt_x,
        tt_y,
        B=batch_size,
        L=text_seq_len,
        dtype=ttnn.bfloat16,
    )

    # Convert TT outputs to torch tensors
    tt_x_torch = to_torch_tensor(tt_x, mesh_device, dim=-1)
    tt_y_torch = to_torch_tensor(tt_y, mesh_device, dim=-1)

    # Get reference outputs
    with torch.no_grad():
        ref_x, ref_y = reference_model.post_attention(
            torch_out,
            B=batch_size,
            M=vision_seq_len,
            L=text_seq_len,
            dtype=torch.bfloat16,
            valid_token_indices=None,
        )
        # Reshape to match TT output
        ref_x = ref_x.reshape(1, batch_size, vision_seq_len, -1)
        ref_y = ref_y.reshape(1, batch_size, text_seq_len, -1)

    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        logger.info(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all(pcc >= PCC_REQUIRED for _, pcc, _, _ in metrics)

    if passing:
        logger.info("TtAsymmetricAttention post_attention Passed!")
    else:
        logger.warning("TtAsymmetricAttention post_attention Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricAttention post_attention output does not meet PCC requirement {PCC_REQUIRED}"


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (43 * 1024, 256),
        (44520, 118),
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
    "attn_path, dim_x, dim_y, update_y",
    [
        ("blocks.0.attn", 3072, 1536, True),
        # ("blocks.47.attn", 3072, 1536, False),
    ],
)
def test_tt_attn_forward(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y, update_y
):
    min_pcc = 0.9975
    max_mse = 1.2e-4
    """Test complete forward pass of TtAsymmetricAttention."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y, update_y
    )

    # Create input tensors
    batch_size = 1
    x_input = torch.randn(batch_size, vision_seq_len, dim_x)
    y_input = torch.randn(batch_size, text_seq_len, dim_y)
    scale_x = torch.randn(batch_size, dim_x)
    scale_y = torch.randn(batch_size, dim_y)

    # Create RoPE tensors
    head_dim = dim_x // NUM_HEADS
    rope_cos = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)
    rope_sin = torch.randn(vision_seq_len, NUM_HEADS, head_dim // 2)

    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )

    # Create valid token indices
    total_seq_len = vision_seq_len + text_seq_len
    valid_token_indices = torch.arange(total_seq_len)
    max_seqlen_in_batch = total_seq_len

    tt_x = to_tt_tensor(x_input.unsqueeze(0), mesh_device)
    tt_y = to_tt_tensor(y_input.unsqueeze(0), mesh_device)
    tt_scale_x = to_tt_tensor(scale_x.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_scale_y = to_tt_tensor(scale_y.view(batch_size, 1, 1, dim_y), mesh_device)
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device, shard_dim=-3)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device, shard_dim=-3)

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)
    trans_mat_tt = to_tt_tensor(trans_mat, mesh_device)

    packed_indices = {
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
    }
    logger.info("Run TtAsymmetricAttention forward")
    tt_x_out, tt_y_out = tt_model(
        tt_x,
        tt_y,
        scale_x=tt_scale_x,
        scale_y=tt_scale_y,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=trans_mat_tt,
    )

    # Convert TT outputs to torch tensors
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device, dim=-1)
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device, dim=-1)

    # unpad TT
    tt_x_torch = tt_x_torch[:, :, :vision_seq_len, :]
    tt_y_torch = tt_y_torch[:, :, :text_seq_len, :]

    # Create packed_indices for reference model
    packed_indices = {
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
        "cu_seqlens_kv": None,
    }
    rope_rotation = {
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
    # Get reference outputs
    logger.info("Run reference model forward")
    with torch.no_grad():
        ref_x, ref_y = reference_model(
            x_input,
            y_input,
            scale_x=scale_x,
            scale_y=scale_y,
            packed_indices=packed_indices,
            **rope_rotation,
        )

    # unpad ref_y
    ref_y = ref_y[..., :text_seq_len, :]

    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        logger.info(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all((mse <= max_mse and pcc >= min_pcc) for _, pcc, mse, _ in metrics)

    if passing:
        logger.info("TtAsymmetricAttention forward Passed!")
    else:
        logger.warning("TtAsymmetricAttention forward Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricAttention forward output does not meet PCC requirement {PCC_REQUIRED}"
