import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.attn import TtAsymmetricAttention
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmetricAttention as RefAsymmetricAttention
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics


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
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)
    from safetensors.torch import load_file

    # Load weights
    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    state_dict = load_file(weights_path)
    partial_state_dict = {k[len(attn_path) + 1 :]: v for k, v in state_dict.items() if k.startswith(attn_path)}
    print(partial_state_dict.keys())

    # Initialize reference model
    reference_model = RefAsymmetricAttention(
        dim_x=dim_x,
        dim_y=dim_y,
        num_heads=24,
        qkv_bias=False,
        qk_norm=True,
        update_y=True,
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

    # Initialize TT model
    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtAsymmetricAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=attn_path,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        dtype=dtype,
        dim_x=dim_x,
        dim_y=dim_y,
        num_heads=24,
        qkv_bias=False,
        qk_norm=True,
        update_y=False,
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

    # Create input tensor
    torch_input = torch.randn(1, seq_len, dim_y)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run TtAsymmetricAttention QKV_Y")
    tt_q, tt_k, tt_v = tt_model.run_qkv_y(tt_input)

    # Convert TT outputs to torch tensors
    tt_q_torch = ttnn.to_torch(tt_q, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_k_torch = ttnn.to_torch(tt_k, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_v_torch = ttnn.to_torch(tt_v, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Get reference outputs
    with torch.no_grad():
        ref_q, ref_k, ref_v = reference_model.run_qkv_y(torch_input)

    # Compute metrics for each output
    metrics = []
    for tt_out, ref_out, name in [
        (tt_q_torch, ref_q, "Q"),
        (tt_k_torch, ref_k, "K"),
        (tt_v_torch, ref_v, "V"),
    ]:
        pcc, mse, mae = compute_metrics(ref_out, tt_out)
        metrics.append((name, pcc, mse, mae))
        logger.info(f"{name} - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    # Check if model meets requirements
    pcc_required = 0.99
    passing = all(pcc >= pcc_required for _, pcc, _, _ in metrics)

    if passing:
        logger.info("TtAsymmetricAttention QKV_Y Passed!")
    else:
        logger.warning("TtAsymmetricAttention QKV_Y Failed!")
        for name, pcc, mse, mae in metrics:
            if pcc < pcc_required:
                logger.error(f"{name} failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"TtAsymmetricAttention QKV_Y output does not meet PCC requirement {pcc_required}"
