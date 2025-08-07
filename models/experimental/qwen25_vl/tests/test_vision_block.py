""""Test for Qwen 2.5 VL Vision Transformer Block"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

from models.experimental.qwen25_vl.tt.vision_block import TtQwen2_5_VLVisionBlock
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 4),),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_transformer_inference(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "visual.blocks.0."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim

    reference_model = model_args.reference_vision_block()
    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    vision_dim = model_args.vision_dim
    n_heads = model_args.vision_attn_n_heads
    head_dim = vision_dim // n_heads
    seq_len = model_args.vision_chunk_ntok - 1

    tt_model = TtQwen2_5_VLVisionBlock(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        model_args=model_args,
        dtype=dtype,
    )

    pt_attention_input = torch.randn(seq_len, vision_dim)  # no batch dim
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

    cos, sin = precompute_rope_cos_sin(seq_len, head_dim)

    reference_output = reference_model(
        pt_attention_input, cu_seqlens, rotary_pos_emb=None, position_embeddings=(cos, sin)
    )

    tt_attention_input = model_args.prepare_residual_tensor_prefill(
        pt_attention_input.unsqueeze(0), force_replicated=True
    )

    cos_tensor = ttnn.from_torch(cos, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tensor = ttnn.from_torch(sin, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_out = tt_model(tt_attention_input, cu_seqlens, position_embeddings=(cos_tensor, sin_tensor))

    tt_output_torch = ttnn.to_torch(tt_out, device=mesh_device).squeeze(0).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"


def precompute_rope_cos_sin(seq_len: int, dim: int, theta: float = 10000.0):
    """
    Precompute RoPE cos/sin tensors.
    Args:
        seq_len: sequence length (number of tokens)
        dim: hidden size (usually head_dim, not full hidden_size)
        theta: RoPE theta parameter (default 10000)
    Returns:
        cos, sin: [seq_len, dim] each
    """
    # Build the rope frequencies
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))

    # positions: [seq_len]
    positions = torch.arange(seq_len, dtype=torch.float32)

    # Outer product: [seq_len, half_dim]
    sinusoid_inp = torch.outer(positions, inv_freq)

    # Concatenate for complex dim
    sin = torch.sin(torch.cat([sinusoid_inp, sinusoid_inp], dim=-1))
    cos = torch.cos(torch.cat([sinusoid_inp, sinusoid_inp], dim=-1))

    return cos, sin
