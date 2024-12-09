import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.attn import TtAsymmetricAttention
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricAttention as RefAsymmetricAttention
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics
from models.demos.llama3.tt.llama_common import get_rot_transformation_mat
from models.experimental.mochi.tests.test_rope import stack_cos_sin

# Common test configurations
NUM_HEADS = 24
HEAD_DIM = 128
PCC_REQUIRED = 0.99


def load_model_weights(attn_path):
    """Load and prepare model weights."""
    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    return state_dict, {k[len(attn_path) + 1 :]: v for k, v in state_dict.items() if k.startswith(attn_path)}


def create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y, update_y=True):
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


def to_tt_tensor(tensor, mesh_device, dtype=ttnn.bfloat16):
    """Convert torch tensor to TT tensor."""
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def to_torch_tensor(tensor, mesh_device, dtype=ttnn.bfloat16):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1), dtype=dtype)


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
    tt_q_torch = to_torch_tensor(tt_q, mesh_device)
    tt_k_torch = to_torch_tensor(tt_k, mesh_device)
    tt_v_torch = to_torch_tensor(tt_v, mesh_device)

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
        (512, 256),
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
    "attn_path, dim_x, dim_y",
    [
        ("blocks.0.attn", 3072, 1536),
    ],
)
def test_tt_attn_prepare_qkv(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y
):
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y)

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

    # Convert inputs to TT tensors
    tt_x = to_tt_tensor(x_input.view(1, batch_size, vision_seq_len, dim_x), mesh_device)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, dim_y), mesh_device)
    tt_scale_x = to_tt_tensor(scale_x.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_scale_y = to_tt_tensor(scale_y.view(batch_size, 1, 1, dim_y), mesh_device)
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device)
    tt_valid_indices = to_tt_tensor(valid_token_indices, mesh_device)

    trans_mat = get_rot_transformation_mat(None)
    trans_mat_tt = to_tt_tensor(trans_mat, mesh_device)

    logger.info("Run TtAsymmetricAttention prepare_qkv")
    tt_q, tt_k, tt_v = tt_model.prepare_qkv(
        tt_x,
        tt_y,
        scale_x=tt_scale_x,
        scale_y=tt_scale_y,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        max_seqlen_in_batch=max_seqlen_in_batch,
        trans_mat=trans_mat_tt,
    )

    # Convert TT outputs to torch tensors
    tt_q_torch = to_torch_tensor(tt_q, mesh_device)
    tt_k_torch = to_torch_tensor(tt_k, mesh_device)
    tt_v_torch = to_torch_tensor(tt_v, mesh_device)

    print(f"tt_q_torch shape: {tt_q_torch.shape}")
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

        print(f"ref_q shape: {ref_q.shape}")
        # Reshape reference to be B, NH, S, D
        ref_q = ref_q.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_k = ref_k.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        ref_v = ref_v.reshape(batch_size, total_seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    validate_outputs([tt_q_torch, tt_k_torch, tt_v_torch], [ref_q, ref_k, ref_v], "TtAsymmetricAttention prepare_qkv")


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (512,),
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
def test_tt_attn_run_attention(mesh_device, seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y):
    """Test run_attention implementation by comparing with reference model."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y)

    batch_size = 1
    head_dim = dim_x // NUM_HEADS

    # Create input tensors
    torch_q = torch.randn(batch_size, NUM_HEADS, seq_len, head_dim)
    torch_k = torch.randn(batch_size, NUM_HEADS, seq_len, head_dim)
    torch_v = torch.randn(batch_size, NUM_HEADS, seq_len, head_dim)

    # Convert to TT tensors
    tt_q = to_tt_tensor(torch_q, mesh_device)
    tt_k = to_tt_tensor(torch_k, mesh_device)
    tt_v = to_tt_tensor(torch_v, mesh_device)

    logger.info("Run TtAsymmetricAttention run_attention")
    tt_out = tt_model.run_attention(
        tt_q,
        tt_k,
        tt_v,
        B=batch_size,
    )

    # Convert TT output to torch tensor
    tt_out_torch = to_torch_tensor(tt_out, mesh_device)

    # Get reference output
    with torch.no_grad():
        # Reshape inputs for reference model
        ref_q = torch_q.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)
        ref_k = torch_k.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)
        ref_v = torch_v.transpose(1, 2).view(-1, NUM_HEADS, head_dim)  # (B*L, H, D)

        ref_out = reference_model.run_attention(ref_q, ref_k, ref_v, B=batch_size)

        # Reshape to match TT output
        ref_out = ref_out.reshape(1, batch_size, seq_len, -1)

    # Validate outputs
    logger.info(f"ref_out shape: {ref_out.shape}")
    logger.info(f"tt_out_torch shape: {tt_out_torch.shape}")
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
    "attn_path, dim_x, dim_y",
    [
        ("blocks.0.attn", 3072, 1536),
    ],
)
def test_tt_attn_post_attention(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y
):
    """Test post_attention implementation by comparing with reference model."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y)

    batch_size = 1
    total_seq_len = vision_seq_len + text_seq_len
    local_dim = dim_x  # No head parallel support yet

    # Create input tensor - simulating attention output
    torch_out = torch.randn(batch_size * total_seq_len, local_dim)

    # Convert to TT tensor
    tt_out = to_tt_tensor(torch_out.reshape(1, batch_size, total_seq_len, local_dim), mesh_device)

    logger.info("Run TtAsymmetricAttention post_attention")
    tt_x, tt_y = tt_model.post_attention(
        tt_out,
        B=batch_size,
        M=vision_seq_len,
        L=text_seq_len,
        dtype=ttnn.bfloat16,
    )

    # Convert TT outputs to torch tensors
    tt_x_torch = to_torch_tensor(tt_x, mesh_device)
    tt_y_torch = to_torch_tensor(tt_y, mesh_device)

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

    print(f"tt_x_torch shape: {tt_x_torch.shape}")
    print(f"ref_x shape: {ref_x.shape}")
    print(f"tt_y_torch shape: {tt_y_torch.shape}")
    print(f"ref_y shape: {ref_y.shape}")

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
    "attn_path, dim_x, dim_y",
    [
        ("blocks.0.attn", 3072, 1536),
    ],
)
def test_tt_attn_forward(
    mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, attn_path, dim_x, dim_y
):
    """Test complete forward pass of TtAsymmetricAttention."""
    state_dict, partial_state_dict = load_model_weights(attn_path)
    reference_model, tt_model = create_models(mesh_device, state_dict, partial_state_dict, attn_path, dim_x, dim_y)

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

    # Convert inputs to TT tensors
    tt_x = to_tt_tensor(x_input.view(1, batch_size, vision_seq_len, dim_x), mesh_device)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, dim_y), mesh_device)
    tt_scale_x = to_tt_tensor(scale_x.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_scale_y = to_tt_tensor(scale_y.view(batch_size, 1, 1, dim_y), mesh_device)
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device)

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
        packed_indices=packed_indices,
    )

    # Convert TT outputs to torch tensors
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device)
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device)

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
    with torch.no_grad():
        ref_x, ref_y = reference_model(
            x_input,
            y_input,
            scale_x=scale_x,
            scale_y=scale_y,
            packed_indices=packed_indices,
            **rope_rotation,
        )

    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        logger.info(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all(pcc >= PCC_REQUIRED for _, pcc, _, _ in metrics)

    if passing:
        logger.info("TtAsymmetricAttention forward Passed!")
    else:
        logger.warning("TtAsymmetricAttention forward Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricAttention forward output does not meet PCC requirement {PCC_REQUIRED}"
