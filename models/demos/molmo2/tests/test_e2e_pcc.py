# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC test comparing PyTorch reference vs TTNN implementation.

Builds PyTorch reference from weights directly (no HuggingFace model loading).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    if ref.shape != out.shape:
        return -1.0, f"Shape mismatch: {ref.shape} vs {out.shape}"
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, "zero std"
    pcc = numerator / denominator
    return pcc.item(), "ok"


class RefRMSNorm(nn.Module):
    """Reference RMSNorm implementation."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rotated * sin


def ref_block_forward(
    x,
    state_dict,
    layer_num,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    intermediate_dim=12288,
    hidden_dim=4096,
    rope_theta=1000000.0,
    rms_norm_eps=1e-5,
):
    """PyTorch reference for a single transformer block."""
    prefix = f"model.transformer.blocks.{layer_num}"
    seq_len = x.shape[1]

    # Load weights
    attn_norm_weight = state_dict[f"{prefix}.attn_norm.weight"]
    ff_norm_weight = state_dict[f"{prefix}.ff_norm.weight"]
    att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"]
    attn_out = state_dict[f"{prefix}.self_attn.attn_out.weight"]
    q_norm_weight = state_dict[f"{prefix}.self_attn.q_norm.weight"]
    k_norm_weight = state_dict[f"{prefix}.self_attn.k_norm.weight"]
    ff_proj = state_dict[f"{prefix}.mlp.ff_proj.weight"]
    ff_out = state_dict[f"{prefix}.mlp.ff_out.weight"]

    # Attention norm
    attn_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
    attn_norm.weight.data = attn_norm_weight
    normed = attn_norm(x)

    # Q, K, V projections
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    wq = att_proj[:q_dim, :]
    wk = att_proj[q_dim : q_dim + kv_dim, :]
    wv = att_proj[q_dim + kv_dim :, :]

    q = F.linear(normed, wq)
    k = F.linear(normed, wk)
    v = F.linear(normed, wv)

    # Reshape
    q = q.reshape(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # QK-norm
    q_norm = RefRMSNorm(head_dim, 1e-5)
    q_norm.weight.data = q_norm_weight
    k_norm = RefRMSNorm(head_dim, 1e-5)
    k_norm.weight.data = k_norm_weight
    q = q_norm(q)
    k = k_norm(k)

    # RoPE
    positions = torch.arange(seq_len)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # GQA expansion
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # Attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    attn_weights = attn_weights + causal_mask
    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    # Output projection
    attn_output = attn_output.transpose(1, 2).reshape(1, seq_len, hidden_dim)
    attn_output = F.linear(attn_output, attn_out)

    # Residual
    x = x + attn_output

    # MLP
    ff_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
    ff_norm.weight.data = ff_norm_weight
    ff_normed = ff_norm(x)

    # SwiGLU (HF order: UP, GATE)
    ff_proj_out = F.linear(ff_normed, ff_proj)
    up = ff_proj_out[..., :intermediate_dim]
    gate = ff_proj_out[..., intermediate_dim:]
    swiglu = F.silu(gate) * up
    mlp_out = F.linear(swiglu, ff_out)

    # Residual
    x = x + mlp_out

    return x


def test_embedding_pcc():
    """Test embedding PCC."""
    logger.info("=" * 60)
    logger.info("Testing Embedding PCC")
    logger.info("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    logger.info(f"Input: {prompt} (seq_len={input_ids.shape[1]})")

    # Reference embedding
    wte = state_dict["model.transformer.wte.embedding"]
    new_embedding = state_dict["model.transformer.wte.new_embedding"]
    full_embed = torch.cat([wte, new_embedding], dim=0)
    ref_hidden = F.embedding(input_ids, full_embed)
    logger.info(f"Ref embedding shape: {ref_hidden.shape}")

    # Open device
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))

    try:
        from models.demos.molmo2.tt.text_model import TextModel

        ttnn_model = TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            dtype=ttnn.bfloat8_b,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        ttnn_embed = ttnn_model.embed_tokens(input_ids_ttnn)
        ttnn_embed_torch = ttnn.to_torch(ttnn_embed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            0
        ].squeeze(0)

        # Squeeze ref_hidden to match ttnn_embed_torch shape
        ref_hidden_squeezed = ref_hidden.squeeze(0)
        pcc, status = calculate_pcc(ref_hidden_squeezed, ttnn_embed_torch)
        logger.info(f"Embedding PCC: {pcc:.6f} ({status})")
        assert pcc > 0.99, f"Embedding PCC {pcc} < 0.99"

    finally:
        ttnn.close_mesh_device(mesh_device)


def test_single_block_pcc():
    """Test single block PCC."""
    logger.info("=" * 60)
    logger.info("Testing Single Block PCC")
    logger.info("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]

    # Reference embedding
    wte = state_dict["model.transformer.wte.embedding"]
    new_embedding = state_dict["model.transformer.wte.new_embedding"]
    full_embed = torch.cat([wte, new_embedding], dim=0)
    ref_hidden = F.embedding(input_ids, full_embed)

    # Reference block 0
    ref_block_out = ref_block_forward(ref_hidden, state_dict, layer_num=0)
    logger.info(f"Ref block 0 output: min={ref_block_out.min():.4f}, max={ref_block_out.max():.4f}")

    # Open device
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))

    try:
        from models.demos.molmo2.tt.text_block import TextBlock
        from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

        rotary_setup = TextRotarySetup(
            mesh_device=mesh_device,
            head_dim=128,
            max_seq_len=8192,
            rope_theta=1000000.0,
        )

        ttnn_block = TextBlock(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_num=0,
            hidden_dim=4096,
            intermediate_dim=12288,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            rms_norm_eps=1e-5,
            dtype=ttnn.bfloat8_b,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        hidden_ttnn = ttnn.from_torch(
            ref_hidden.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        rot_mats = rotary_setup.get_rot_mats_prefill(seq_len, 0)
        transformation_mats = rotary_setup.get_transformation_mats()

        ttnn_out, _ = ttnn_block(hidden_ttnn, rot_mats, transformation_mats, None, 0, None)
        ttnn_out_torch = ttnn.to_torch(ttnn_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].squeeze(
            0
        )

        # Squeeze ref_block_out to match ttnn_out_torch shape (remove batch dim)
        ref_block_out_squeezed = ref_block_out.squeeze(0)
        pcc, status = calculate_pcc(ref_block_out_squeezed, ttnn_out_torch)
        logger.info(f"Block 0 PCC: {pcc:.6f} ({status})")

        # Check top-k match at last position for generation relevance
        ref_last = ref_block_out_squeezed[-1, :]
        ttnn_last = ttnn_out_torch[-1, :]
        last_pcc, _ = calculate_pcc(ref_last, ttnn_last)
        logger.info(f"Block 0 last position PCC: {last_pcc:.6f}")

        assert pcc > 0.95, f"Block 0 PCC {pcc} < 0.95"

    finally:
        ttnn.close_mesh_device(mesh_device)


def test_full_model_pcc():
    """Test full model PCC (all 36 layers)."""
    logger.info("=" * 60)
    logger.info("Testing Full Model PCC (36 layers)")
    logger.info("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Reference embedding
    wte = state_dict["model.transformer.wte.embedding"]
    new_embedding = state_dict["model.transformer.wte.new_embedding"]
    full_embed = torch.cat([wte, new_embedding], dim=0)
    ref_hidden = F.embedding(input_ids, full_embed)

    # Reference all 36 blocks
    logger.info("Running reference through all 36 layers...")
    for layer_num in range(36):
        ref_hidden = ref_block_forward(ref_hidden, state_dict, layer_num)
        if (layer_num + 1) % 6 == 0:
            logger.info(f"  Layer {layer_num + 1}/36 done")

    # Final norm
    ln_f_weight = state_dict["model.transformer.ln_f.weight"]
    ln_f = RefRMSNorm(4096, 1e-5)
    ln_f.weight.data = ln_f_weight
    ref_hidden = ln_f(ref_hidden)

    # LM head
    lm_head = state_dict["lm_head.weight"]
    ref_logits = F.linear(ref_hidden, lm_head)

    logger.info(f"Ref logits shape: {ref_logits.shape}")

    # Get reference top predictions
    ref_last_logits = ref_logits[0, -1, :]
    ref_top5 = torch.topk(ref_last_logits, 5)
    ref_top5_decoded = [tokenizer.decode([t]) for t in ref_top5.indices.tolist()]
    logger.info(f"Ref top-5 tokens: {ref_top5.indices.tolist()}")
    logger.info(f"Ref top-5 decoded: {ref_top5_decoded}")

    # TTNN model
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))

    try:
        from models.demos.molmo2.tt.molmo2_model import Molmo2Model

        ttnn_model = Molmo2Model(
            mesh_device=mesh_device,
            state_dict=state_dict,
            dtype=ttnn.bfloat8_b,
        )

        logger.info("Running TTNN forward...")
        ttnn_logits, _ = ttnn_model.forward(input_ids=input_ids)

        ttnn_logits_torch = (
            ttnn.to_torch(ttnn_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
            .squeeze(0)
            .squeeze(0)
        )

        logger.info(f"TTNN logits shape: {ttnn_logits_torch.shape}")

        # Compare last position
        ttnn_last_logits = ttnn_logits_torch[-1, :]
        pcc, status = calculate_pcc(ref_last_logits, ttnn_last_logits)
        logger.info(f"Last position logits PCC: {pcc:.6f} ({status})")

        ttnn_top5 = torch.topk(ttnn_last_logits, 5)
        ttnn_top5_decoded = [tokenizer.decode([t]) for t in ttnn_top5.indices.tolist()]
        logger.info(f"TTNN top-5 tokens: {ttnn_top5.indices.tolist()}")
        logger.info(f"TTNN top-5 decoded: {ttnn_top5_decoded}")

        # Check top-1 match
        top1_match = ref_top5.indices[0].item() == ttnn_top5.indices[0].item()
        logger.info(f"Top-1 match: {top1_match}")

        assert pcc > 0.90, f"Last position logits PCC {pcc} < 0.90"
        assert top1_match, "Top-1 prediction mismatch!"

    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "embedding":
        test_embedding_pcc()
    elif len(sys.argv) > 1 and sys.argv[1] == "block":
        test_single_block_pcc()
    else:
        test_full_model_pcc()
