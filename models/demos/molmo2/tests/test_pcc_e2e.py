# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC comparison between HuggingFace and TTNN Molmo2 implementations.

Compares outputs at key points:
1. Text embeddings
2. Vision backbone output (if multimodal)
3. Final logits after prefill
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()

    ref_centered = ref_flat - ref_mean
    test_centered = test_flat - test_mean

    numerator = (ref_centered * test_centered).sum()
    denominator = torch.sqrt((ref_centered**2).sum() * (test_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return (numerator / denominator).item()


@pytest.fixture
def mesh_device():
    """Create mesh device for tests."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    device = ttnn.open_mesh_device(mesh_shape)
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture
def hf_model():
    """Load HuggingFace Molmo2 model."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    """Load tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)


def test_text_embedding_pcc(mesh_device, hf_model, tokenizer):
    """Test that text embeddings match between HF and TTNN."""
    logger.info("Testing text embedding PCC...")

    # Load TTNN model
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    from models.demos.molmo2.tt.text_model import TextModel

    ttnn_model = TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
    )

    # Test input
    prompt = "What is the capital of France?"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    logger.info(f"Input shape: {input_ids.shape}")

    # HuggingFace embedding
    with torch.no_grad():
        # Access the embedding layer
        hf_embed = hf_model.model.transformer.wte
        hf_new_embed = hf_model.model.transformer.wte.new_embedding

        # Combine embeddings like HF does
        full_embed = torch.cat([hf_embed.embedding, hf_new_embed], dim=0)
        hf_embeddings = torch.nn.functional.embedding(input_ids, full_embed)

    logger.info(f"HF embeddings shape: {hf_embeddings.shape}")

    # TTNN embedding
    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    ttnn_embeddings = ttnn_model.embed_tokens(input_ids_ttnn)

    if is_mesh_device:
        ttnn_embeddings_torch = ttnn.to_torch(
            ttnn_embeddings, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )[0]
    else:
        ttnn_embeddings_torch = ttnn.to_torch(ttnn_embeddings)

    ttnn_embeddings_torch = ttnn_embeddings_torch.squeeze(0)  # [1, seq_len, hidden_dim]

    logger.info(f"TTNN embeddings shape: {ttnn_embeddings_torch.shape}")

    # Compute PCC
    pcc = compute_pcc(hf_embeddings, ttnn_embeddings_torch)
    logger.info(f"Text Embedding PCC: {pcc:.6f}")

    assert pcc > 0.99, f"Text embedding PCC {pcc} is too low (expected > 0.99)"


def test_prefill_logits_pcc(mesh_device, hf_model, tokenizer):
    """Test that prefill logits match between HF and TTNN."""
    logger.info("Testing prefill logits PCC...")

    # Load TTNN model
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    from models.demos.molmo2.tt.molmo2_model import Molmo2Model

    ttnn_model = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
    )

    # Test input
    prompt = "What is the capital of France?"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    logger.info(f"Input shape: {input_ids.shape}")

    # HuggingFace forward
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits  # [batch, seq_len, vocab_size]

    logger.info(f"HF logits shape: {hf_logits.shape}")

    # TTNN forward
    ttnn_logits, _ = ttnn_model.forward(input_ids=input_ids)

    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    if is_mesh_device:
        ttnn_logits_torch = ttnn.to_torch(ttnn_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    else:
        ttnn_logits_torch = ttnn.to_torch(ttnn_logits)

    ttnn_logits_torch = ttnn_logits_torch.squeeze(0)  # [1, seq_len, vocab_size]

    logger.info(f"TTNN logits shape: {ttnn_logits_torch.shape}")

    # Compare last position logits (most important for generation)
    hf_last_logits = hf_logits[0, -1, :]
    ttnn_last_logits = ttnn_logits_torch[0, -1, :]

    pcc = compute_pcc(hf_last_logits, ttnn_last_logits)
    logger.info(f"Last position logits PCC: {pcc:.6f}")

    # Also check top-k predictions match
    hf_top5 = torch.topk(hf_last_logits, 5).indices.tolist()
    ttnn_top5 = torch.topk(ttnn_last_logits, 5).indices.tolist()
    logger.info(f"HF top-5 tokens: {hf_top5}")
    logger.info(f"TTNN top-5 tokens: {ttnn_top5}")

    # Decode top predictions
    hf_decoded = [tokenizer.decode([t]) for t in hf_top5]
    ttnn_decoded = [tokenizer.decode([t]) for t in ttnn_top5]
    logger.info(f"HF top-5 decoded: {hf_decoded}")
    logger.info(f"TTNN top-5 decoded: {ttnn_decoded}")

    # Check if top-1 matches
    top1_match = hf_top5[0] == ttnn_top5[0]
    logger.info(f"Top-1 match: {top1_match}")

    # Full sequence PCC
    full_pcc = compute_pcc(hf_logits.squeeze(0), ttnn_logits_torch.squeeze(0))
    logger.info(f"Full sequence logits PCC: {full_pcc:.6f}")

    assert pcc > 0.95, f"Last position logits PCC {pcc} is too low (expected > 0.95)"
    assert top1_match, f"Top-1 prediction mismatch: HF={hf_top5[0]}, TTNN={ttnn_top5[0]}"


def test_single_layer_pcc(mesh_device, hf_model):
    """Test PCC for a single transformer layer."""
    logger.info("Testing single layer PCC...")

    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    from models.demos.molmo2.tt.text_block import TextBlock

    layer_idx = 0
    ttnn_block = TextBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        layer_num=layer_idx,
        hidden_dim=4096,
        intermediate_dim=12288,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rms_norm_eps=1e-5,
        dtype=ttnn.bfloat8_b,
    )

    # Create random input
    batch_size = 1
    seq_len = 32
    hidden_dim = 4096

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # HuggingFace layer forward
    hf_block = hf_model.model.transformer.blocks[layer_idx]

    with torch.no_grad():
        # Get attention mask and position ids
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # HF forward (simplified - may need adjustment based on actual HF API)
        hf_hidden = x.clone()
        hf_attn_norm = hf_block.attn_norm(hf_hidden)

        # We'd need to properly set up RoPE and attention here
        # For now, just test the norm layers
        hf_ff_norm = hf_block.ff_norm(hf_hidden)

    # TTNN norm test
    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

    x_ttnn = ttnn.from_torch(
        x.unsqueeze(0),  # [1, 1, seq_len, hidden_dim]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    ttnn_attn_norm = ttnn_block.attn_norm(x_ttnn)

    if is_mesh_device:
        ttnn_attn_norm_torch = ttnn.to_torch(ttnn_attn_norm, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            0
        ]
    else:
        ttnn_attn_norm_torch = ttnn.to_torch(ttnn_attn_norm)

    ttnn_attn_norm_torch = ttnn_attn_norm_torch.squeeze(0).squeeze(0)

    pcc = compute_pcc(hf_attn_norm, ttnn_attn_norm_torch)
    logger.info(f"Attention norm PCC: {pcc:.6f}")

    assert pcc > 0.99, f"Attention norm PCC {pcc} is too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
