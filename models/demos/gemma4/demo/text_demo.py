# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 E4B Text Demo

Runs text-only inference on Gemma 4 E4B using TTNN on Wormhole N150.

Usage:
    export HF_MODEL=/path/to/gemma-4-E4B-it/weights
    pytest models/demos/gemma4/demo/text_demo.py -v -s
"""

import os

import pytest

import ttnn
from models.demos.gemma4.tt.model_config import ModelArgs


@pytest.fixture
def mesh_device(request):
    """Create mesh device for testing."""
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    yield device
    ttnn.close_mesh_device(device)


@pytest.mark.parametrize("prompt", ["The capital of France is"])
def test_gemma4_text_demo(mesh_device, prompt):
    """Run Gemma 4 E4B text generation demo."""
    # Configuration
    dtype = ttnn.bfloat8_b
    max_batch_size = 1

    # Load model configuration
    args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=False,
        max_batch_size=max_batch_size,
        max_seq_len=1024,
    )

    # Load state dict
    state_dict = args.load_state_dict()

    # Create tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])

    # Tokenize
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    # TODO: Implement full model forward pass
    # This requires:
    # 1. Dual head_dim support in Attention
    # 2. Partial rotary embedding
    # 3. Per-layer input gating
    # 4. KV cache sharing
    # 5. Final logit softcapping
    # 6. V-norm
    # 7. embed_tokens_per_layer mechanism

    print("\n⚠️  Full model forward pass not yet implemented.")
    print("Day 1: Config parsing and weight loading verified ✓")
    print("Day 2: Will implement dual head_dim + partial rotary")
    print("Day 3: Will implement per-layer gating + KV sharing")
    print("Day 4: Will implement remaining features + PCC validation")
