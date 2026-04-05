# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test optimized all-MiniLM-L6-v2 with sharded memory on Wormhole."""

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.minilm.tt.minilm_optimized import MiniLMOptimized, custom_preprocessor

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_minilm_optimized_hidden_states(batch_size, device):
    """Test optimized MiniLM encoder output PCC against HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModel.from_pretrained(MODEL_NAME).eval()
    config = AutoConfig.from_pretrained(MODEL_NAME)

    sentences = [
        "This is an example sentence",
        "Each sentence is converted",
        "The weather is nice today",
        "I love machine learning",
        "Tenstorrent builds AI hardware",
        "Sentence embeddings are useful",
        "Natural language processing",
        "Deep learning is powerful",
    ][:batch_size]

    encoded = tokenizer(sentences, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

    with torch.no_grad():
        hf_output = hf_model(**encoded)
    hf_hidden = hf_output.last_hidden_state

    seq_len = encoded["input_ids"].shape[1]
    position_ids = torch.arange(seq_len).expand(batch_size, -1)
    ext_mask = (1.0 - encoded["attention_mask"].unsqueeze(1).unsqueeze(2).float()) * -10000.0

    tt_input_ids = ttnn.from_torch(
        encoded["input_ids"], dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_token_type_ids = ttnn.from_torch(
        encoded["token_type_ids"], dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_position_ids = ttnn.from_torch(
        position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_attn_mask = ttnn.from_torch(
        ext_mask, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: hf_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    tt_model = MiniLMOptimized(config=config, parameters=parameters)

    tt_hidden = tt_model(tt_input_ids, tt_token_type_ids, tt_position_ids, tt_attn_mask, device=device)

    # Move to interleaved for to_torch
    if tt_hidden.is_sharded():
        tt_hidden = ttnn.sharded_to_interleaved(tt_hidden, ttnn.L1_MEMORY_CONFIG)

    tt_hidden_pt = ttnn.to_torch(tt_hidden)
    while tt_hidden_pt.dim() > 3:
        tt_hidden_pt = tt_hidden_pt.squeeze(1)
    tt_hidden_pt = tt_hidden_pt[:batch_size, :seq_len, : config.hidden_size]

    similarity = pcc(tt_hidden_pt, hf_hidden)
    max_diff = (tt_hidden_pt.float() - hf_hidden.float()).abs().max().item()

    print(f"\n{'=' * 60}")
    print(f"MiniLM-L6-v2 Optimized Hidden States Test")
    print(f"{'=' * 60}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Hidden: {config.hidden_size}")
    print(f"PCC: {similarity:.6f}")
    print(f"Max absolute diff: {max_diff:.6f}")

    # Slightly lower threshold for optimized (bfloat8_b weights)
    assert similarity > 0.96, f"PCC {similarity} below threshold 0.96"
    print(f"PASSED: PCC={similarity:.6f} > 0.96")


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_minilm_optimized_sentence_embedding(batch_size, device):
    """Test optimized MiniLM sentence embedding with mean pooling."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModel.from_pretrained(MODEL_NAME).eval()
    config = AutoConfig.from_pretrained(MODEL_NAME)

    sentences = [
        "This is an example sentence",
        "Each sentence is converted",
        "The weather is nice today",
        "I love machine learning",
        "Tenstorrent builds AI hardware",
        "Sentence embeddings are useful",
        "Natural language processing",
        "Deep learning is powerful",
    ][:batch_size]

    encoded = tokenizer(sentences, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        hf_output = hf_model(**encoded)
    hf_tok = hf_output.last_hidden_state
    mask_exp = attention_mask.unsqueeze(-1).expand(hf_tok.size()).float()
    hf_embeddings = torch.sum(hf_tok * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    seq_len = encoded["input_ids"].shape[1]
    position_ids = torch.arange(seq_len).expand(batch_size, -1)
    ext_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

    tt_input_ids = ttnn.from_torch(
        encoded["input_ids"], dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_token_type_ids = ttnn.from_torch(
        encoded["token_type_ids"], dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_position_ids = ttnn.from_torch(
        position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_attn_mask = ttnn.from_torch(
        ext_mask, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: hf_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    tt_model = MiniLMOptimized(config=config, parameters=parameters)

    tt_hidden = tt_model(tt_input_ids, tt_token_type_ids, tt_position_ids, tt_attn_mask, device=device)

    if tt_hidden.is_sharded():
        tt_hidden = ttnn.sharded_to_interleaved(tt_hidden, ttnn.L1_MEMORY_CONFIG)

    tt_hidden_pt = ttnn.to_torch(tt_hidden)
    while tt_hidden_pt.dim() > 3:
        tt_hidden_pt = tt_hidden_pt.squeeze(1)
    tt_hidden_pt = tt_hidden_pt[:batch_size, :seq_len, : config.hidden_size]

    # CPU mean pooling
    mask_exp = attention_mask.unsqueeze(-1).expand(tt_hidden_pt.size()).float()
    tt_embeddings = torch.sum(tt_hidden_pt.float() * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    similarity = pcc(tt_embeddings, hf_embeddings)
    print(f"\n{'=' * 60}")
    print(f"MiniLM-L6-v2 Optimized Sentence Embedding Test")
    print(f"{'=' * 60}")
    print(f"PCC: {similarity:.6f}")
    for i, s in enumerate(sentences):
        print(f"  [{i}] PCC={pcc(tt_embeddings[i], hf_embeddings[i]):.6f} | '{s[:40]}'")

    assert similarity > 0.96, f"PCC {similarity} below threshold 0.96"
    print(f"PASSED: PCC={similarity:.6f} > 0.96")
