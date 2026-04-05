# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Demo: Semantic similarity using all-MiniLM-L6-v2 on Wormhole.

Computes sentence embeddings via mean pooling, then shows a cosine
similarity matrix to demonstrate that semantically similar sentences
cluster together.
"""

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModel, AutoTokenizer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.minilm.tt.minilm_model import MiniLMModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Eight sentences: 4 about AI/tech, 4 about nature/weather
DEMO_SENTENCES = [
    "Machine learning models are transforming the tech industry.",
    "Deep neural networks can learn complex patterns from data.",
    "Artificial intelligence is revolutionizing healthcare and finance.",
    "Large language models understand and generate human text.",
    "The sunset over the ocean was breathtakingly beautiful.",
    "We hiked through the forest and enjoyed the fresh mountain air.",
    "Spring flowers are blooming in the garden after the rain.",
    "The beach was peaceful with gentle waves and warm sand.",
]


def mean_pool(hidden_states, attention_mask):
    """Mean pooling: average token embeddings weighted by attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states.float() * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def cosine_sim_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    norms = embeddings.norm(dim=1, keepdim=True)
    normalized = embeddings / norms.clamp(min=1e-9)
    return torch.mm(normalized, normalized.t())


def run_minilm_demo(device, sentences=None):
    """Run MiniLM sentence embedding demo on TT device."""
    sentences = sentences or DEMO_SENTENCES
    batch_size = len(sentences)

    logger.info(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModel.from_pretrained(MODEL_NAME).eval()
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Tokenize
    encoded = tokenizer(
        sentences,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"]

    # HF reference embeddings
    with torch.no_grad():
        hf_output = hf_model(**encoded)
    hf_embeddings = mean_pool(hf_output.last_hidden_state, attention_mask)

    # Prepare TT inputs
    seq_len = encoded["input_ids"].shape[1]
    position_ids = torch.arange(seq_len).expand(batch_size, -1)
    ext_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

    tt_input_ids = ttnn.from_torch(
        encoded["input_ids"],
        dtype=ttnn.uint32,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_token_type_ids = ttnn.from_torch(
        encoded["token_type_ids"],
        dtype=ttnn.uint32,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_position_ids = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_attn_mask = ttnn.from_torch(
        ext_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TT model
    logger.info("Running MiniLM on Tenstorrent device")
    parameters = preprocess_model_parameters(initialize_model=lambda: hf_model, device=device)
    tt_model = MiniLMModel(config=config, parameters=parameters)
    tt_hidden = tt_model(tt_input_ids, tt_token_type_ids, tt_position_ids, tt_attn_mask, device=device)

    # Convert back and mean pool on CPU
    tt_hidden_pt = ttnn.to_torch(tt_hidden)
    while tt_hidden_pt.dim() > 3:
        tt_hidden_pt = tt_hidden_pt.squeeze(1)
    tt_hidden_pt = tt_hidden_pt[:batch_size, :seq_len, : config.hidden_size]
    tt_embeddings = mean_pool(tt_hidden_pt, attention_mask)

    # Cosine similarity matrices
    tt_sim = cosine_sim_matrix(tt_embeddings)
    hf_sim = cosine_sim_matrix(hf_embeddings)

    # Display results
    logger.info("=" * 70)
    logger.info("MiniLM-L6-v2 Semantic Similarity Demo (Tenstorrent Wormhole)")
    logger.info("=" * 70)

    logger.info("\nSentences:")
    for i, s in enumerate(sentences):
        logger.info(f"  [{i}] {s}")

    logger.info("\nTT Device Cosine Similarity Matrix:")
    header = "     " + "".join(f"  [{i}]  " for i in range(batch_size))
    logger.info(header)
    for i in range(batch_size):
        row = f"[{i}]  " + "  ".join(f"{tt_sim[i][j]:.3f}" for j in range(batch_size))
        logger.info(row)

    # Accuracy vs HF
    sim_diff = (tt_sim - hf_sim).abs().max().item()
    logger.info(f"\nMax cosine similarity difference vs HuggingFace: {sim_diff:.4f}")

    # Check that AI sentences are more similar to each other than to nature sentences
    ai_ai_sim = tt_sim[:4, :4].mean().item()
    nature_nature_sim = tt_sim[4:, 4:].mean().item()
    cross_sim = tt_sim[:4, 4:].mean().item()

    logger.info(f"AI-AI avg similarity:         {ai_ai_sim:.4f}")
    logger.info(f"Nature-Nature avg similarity:  {nature_nature_sim:.4f}")
    logger.info(f"Cross-topic avg similarity:    {cross_sim:.4f}")

    assert sim_diff < 0.02, f"Similarity matrix differs by {sim_diff:.4f}, exceeds 0.02 tolerance"
    assert ai_ai_sim > cross_sim, "AI sentences should be more similar to each other"
    assert nature_nature_sim > cross_sim, "Nature sentences should be more similar to each other"

    logger.info("\nPASSED: Semantic clustering is correct!")
    return tt_embeddings


@pytest.mark.parametrize("batch_size", [8])
def test_demo_minilm(batch_size, device):
    """Demo test: semantic similarity with MiniLM on Wormhole."""
    run_minilm_demo(device, sentences=DEMO_SENTENCES[:batch_size])


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        run_minilm_demo(device)
    finally:
        ttnn.close_device(device)
