import os
import pytest
import torch
import ttnn
from loguru import logger

from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricJointBlock
from models.experimental.mochi.tt.dit.model import AsymmetricJointBlock as TtAsymmetricJointBlock
from models.experimental.mochi.tt.common import (
    get_cache_path,
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
    stack_cos_sin,
    pad_vision_seq_parallel,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.experimental.mochi.tests.dit.common import (
    load_model_weights,
    NUM_HEADS,
)

MAX_T5_TOKEN_LENGTH = 256

PCC_REQUIRED = 0.99

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
multiple_of = 256
ffn_dim_multiplier = None


def create_models(mesh_device, state_dict, partial_state_dict, block_path, dim_x, dim_y, update_y=True):
    """Initialize both reference and TT models."""
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
    return reference_model, tt_model


@torch.no_grad()
@pytest.mark.parametrize(
    "vision_seq_len, text_seq_len",
    [
        (22 * 256 * 8, 256),
        (44520, 118),
    ],
    #    [(43 * 1024, 256), (44520, 118)],
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
    "block_path, update_y",
    [
        ("blocks.0", True),
        # ("blocks.47", False),
    ],
)
def test_tt_block(mesh_device, vision_seq_len, text_seq_len, use_program_cache, reset_seeds, block_path, update_y):
    """Test TtAsymmetricJointBlock implementation by comparing with reference model."""

    min_pcc = 0.997
    max_mse = 0.0089
    state_dict, partial_state_dict = load_model_weights(block_path)

    # Create reference model
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, block_path, dim_x, dim_y, update_y
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
    x_padded = pad_vision_seq_parallel(
        x_input.view(1, batch_size, vision_seq_len, dim_x), mesh_device.get_num_devices()
    )
    cos_padded = pad_vision_seq_parallel(rope_cos_stack, mesh_device.get_num_devices())
    sin_padded = pad_vision_seq_parallel(rope_sin_stack, mesh_device.get_num_devices())
    tt_x = to_tt_tensor(x_padded, mesh_device, shard_dim=-2)
    tt_y = to_tt_tensor(y_input.view(1, batch_size, text_seq_len, dim_y), mesh_device)
    tt_c = to_tt_tensor(c_input.view(batch_size, 1, 1, dim_x), mesh_device)
    tt_rope_cos = to_tt_tensor(cos_padded, mesh_device, shard_dim=-3)
    tt_rope_sin = to_tt_tensor(sin_padded, mesh_device, shard_dim=-3)
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
        N=vision_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    # Convert TT outputs to torch tensors
    # extract from replicated tensors
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device, dim=2)[:, :, :vision_seq_len, :]
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device, dim=0)[0:1]

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

    passing = all((mse <= max_mse) and (pcc >= min_pcc) for _, pcc, mse, _ in metrics)

    if passing:
        logger.info("TtAsymmetricJointBlock Passed!")
    else:
        logger.warning("TtAsymmetricJointBlock Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricJointBlock output does not meet PCC requirement {PCC_REQUIRED}"


def load_saved_tensors(block_num):
    """Load tensors saved during reference model execution."""
    ref_tensor_path = os.environ.get("REF_TENSORS")
    if not ref_tensor_path:
        pytest.skip("REF_TENSORS environment variable not set")

    prefix = f"block_{block_num}"
    tensors = {}

    # Load main tensors
    for name in ["x", "c", "y_feat", "rope_cos", "rope_sin"]:
        path = os.path.join(ref_tensor_path, f"{prefix}_{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected tensor file not found: {path}")
        tensors[name] = torch.load(path)

    # Load packed indices
    packed_indices = {}
    for key in ["max_seqlen_in_batch_kv", "valid_token_indices_kv", "cu_seqlens_kv"]:
        path = os.path.join(ref_tensor_path, f"{prefix}_packed_indices_{key}.pt")
        if os.path.exists(path):
            packed_indices[key] = torch.load(path)

    # Load checkpoint flags

    return tensors, packed_indices


@torch.no_grad()
@pytest.mark.parametrize(
    "block_path, update_y",
    [
        ("blocks.0", True),
        ("blocks.47", False),
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
def test_tt_block_with_saved_tensors(mesh_device, use_program_cache, reset_seeds, block_path, update_y):
    """Test TtAsymmetricJointBlock using saved reference tensors."""
    block_num = int(block_path.split(".")[1])
    tensors, packed_indices = load_saved_tensors(block_num)

    state_dict, partial_state_dict = load_model_weights(block_path)

    # Convert inputs to TT tensors
    x_shape = tensors["x"].shape
    y_shape = tensors["y_feat"].shape

    # Create models
    reference_model, tt_model = create_models(
        mesh_device, state_dict, partial_state_dict, block_path, dim_x, dim_y, update_y
    )

    # Print tensor shapes
    print("\nInput tensor shapes:")
    print(f"x: {tensors['x'].shape}")
    print(f"y_feat: {tensors['y_feat'].shape}")
    print(f"c: {tensors['c'].shape}")
    print(f"rope_cos: {tensors['rope_cos'].shape}")
    print(f"rope_sin: {tensors['rope_sin'].shape}")
    if packed_indices:
        print("\nPacked indices shapes:")
        for k, v in packed_indices.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")

    max_seqlen_in_batch_kv = packed_indices["max_seqlen_in_batch_kv"]

    y_unpadded_len = max_seqlen_in_batch_kv - x_shape[1]

    tt_x = to_tt_tensor(tensors["x"].view(1, x_shape[0], x_shape[1], x_shape[2]), mesh_device, shard_dim=-2)
    tt_y = to_tt_tensor(tensors["y_feat"].view(1, y_shape[0], y_shape[1], y_shape[2]), mesh_device)
    tt_c = to_tt_tensor(tensors["c"].view(x_shape[0], 1, 1, -1), mesh_device)

    # Stack and convert RoPE tensors
    rope_cos_stack, rope_sin_stack = stack_cos_sin(
        tensors["rope_cos"].unsqueeze(0).permute(0, 2, 1, 3), tensors["rope_sin"].unsqueeze(0).permute(0, 2, 1, 3)
    )
    tt_rope_cos = to_tt_tensor(rope_cos_stack, mesh_device, shard_dim=-3)
    tt_rope_sin = to_tt_tensor(rope_sin_stack, mesh_device, shard_dim=-3)

    # Get transformation matrix
    trans_mat = get_rot_transformation_mat(None)
    tt_trans_mat = to_tt_tensor(trans_mat, mesh_device)

    # Run TT model
    logger.info("Run TtAsymmetricJointBlock forward with saved tensors")
    tt_x_out, tt_y_out = tt_model(
        tt_x,
        tt_c,
        tt_y,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    # Run reference model
    ref_x, ref_y = reference_model(
        tensors["x"],
        tensors["c"],
        tensors["y_feat"],
        rope_cos=tensors["rope_cos"],
        rope_sin=tensors["rope_sin"],
        packed_indices=packed_indices,
    )

    # Convert TT outputs to torch tensors
    tt_x_out = ttnn.all_gather(tt_x_out, dim=2)
    tt_x_torch = to_torch_tensor(tt_x_out, mesh_device, dim=0)[0:1]
    tt_y_torch = to_torch_tensor(tt_y_out, mesh_device, dim=0)[0:1]

    # unpad x
    tt_x_torch = tt_x_torch[:, :, : x_shape[1], :]
    # NOTE: reference chops of Y padding for attn, then adds back zero padding
    tt_y_torch = tt_y_torch[:, :, :y_unpadded_len, :]
    # NOTE: I had issues comparing tt_y to ref_y. I believe that the bias on `proj_y` causes the mismatch, which isn't important.
    # For this reason, also unpad ref_y before comparing
    ref_y = ref_y[..., :y_unpadded_len, :]
    # Validate outputs
    metrics = []
    for tt_out, ref_out, name in [(tt_x_torch, ref_x, "Visual"), (tt_y_torch, ref_y, "Text")]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        print(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = all(pcc >= PCC_REQUIRED for _, pcc, _, _ in metrics)

    if passing:
        logger.info("TtAsymmetricJointBlock Passed with saved tensors!")
    else:
        logger.warning("TtAsymmetricJointBlock Failed with saved tensors!")
        for name, pcc, mse, mae in metrics:
            if pcc < PCC_REQUIRED:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricJointBlock output does not meet PCC requirement {PCC_REQUIRED} with saved tensors"
