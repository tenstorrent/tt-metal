# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for Molmo2-8B demo.

Runs a basic smoke test to verify the demo works correctly.
"""

import os

import numpy as np
import pytest
import torch

import ttnn


@pytest.fixture
def processor():
    """Load the Molmo2 processor."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )


@pytest.mark.parametrize("num_layers", [1])  # Use 1 layer for fast smoke test
def test_demo_smoke(device, processor, num_layers):
    """
    Smoke test for Molmo2 demo.

    Uses a single text layer for fast validation.
    """

    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
    from models.demos.molmo2.tt.text_model import TextModel

    model_id = "allenai/Molmo2-8B"
    hidden_dim = 4096
    seq_len = 16

    # Load minimal weights
    keys = [
        "model.transformer.wte.embedding",
        "model.transformer.wte.new_embedding",
        "model.transformer.ln_f.weight",
        "model.transformer.blocks.0.attn_norm.weight",
        "model.transformer.blocks.0.self_attn.q_norm.weight",
        "model.transformer.blocks.0.self_attn.k_norm.weight",
        "model.transformer.blocks.0.self_attn.att_proj.weight",
        "model.transformer.blocks.0.self_attn.attn_out.weight",
        "model.transformer.blocks.0.ff_norm.weight",
        "model.transformer.blocks.0.mlp.ff_proj.weight",
        "model.transformer.blocks.0.mlp.ff_out.weight",
        "lm_head.weight",
    ]
    state_dict = load_state_dict_from_safetensors(model_id, keys)

    # Create text model (simplified without full vision)
    text_model = TextModel(
        mesh_device=device,
        state_dict=state_dict,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Create random input
    torch.manual_seed(42)
    hidden_states = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)

    hidden_states_ttnn = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Forward pass
    logits, _ = text_model(hidden_states_ttnn)

    # Verify output
    logits_torch = ttnn.to_torch(logits).squeeze(0).squeeze(0)
    assert logits_torch.shape[0] == seq_len, f"Expected seq_len {seq_len}"
    assert not torch.isnan(logits_torch).any(), "Output contains NaN"

    print(f"Demo smoke test passed: output shape {logits_torch.shape}")


@pytest.mark.parametrize("prompt", ["What is in this image?", "Describe this dog."])
def test_processor_tokenization(processor, prompt):
    """
    Test that the processor can tokenize prompts correctly.
    """
    # Test text-only tokenization
    inputs = processor.tokenizer(
        prompt,
        return_tensors="pt",
    )

    assert inputs["input_ids"].shape[0] == 1
    assert inputs["input_ids"].shape[1] > 0

    # Decode back
    decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    assert prompt in decoded or decoded in prompt

    print(f"Tokenization test passed for: '{prompt}'")


def test_hf_video_processor_inputs(processor):
    """
    Smoke test: Molmo2 processor accepts synthetic video dict (no TTNN, no full generate).
    """
    from models.demos.molmo2.demo.demo import VIDEO_PROMPT

    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5], dtype=np.float64)
    videos = [{"frames": frames, "timestamps": timestamps, "sampled_fps": 2.0}]

    inputs = processor(
        text=[f"{VIDEO_PROMPT} Describe briefly."],
        videos=videos,
        return_tensors="pt",
    )

    assert "input_ids" in inputs
    assert inputs["input_ids"].dim() == 2
    assert inputs["input_ids"].shape[0] == 1
    assert inputs["input_ids"].shape[1] > 0
    video_keys = [k for k in inputs if "video" in k.lower()]
    assert len(video_keys) > 0, f"Expected video-related keys in processor output, got {list(inputs.keys())}"


if __name__ == "__main__":
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "allenai/Molmo2-8B",
            trust_remote_code=True,
        )

        test_demo_smoke(device, processor, num_layers=1)
        test_processor_tokenization(processor, "What is in this image?")
        print("\nAll demo tests passed!")
    finally:
        ttnn.close_device(device)
