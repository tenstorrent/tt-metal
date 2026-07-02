# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.cosy_voice.tt.cosyvoice_llm import CosyVoice3LM
from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig


@pytest.fixture(scope="module")
def cosyvoice_llm(mesh_device):
    """Initialize CosyVoice3LM once per module."""
    weights_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
    config = CosyVoiceModelConfig(mesh_device=mesh_device, max_batch_size=1, weights_dir=weights_dir)

    # We use bfloat16 for testing to match standard TTNN Transformer test precision
    llm = CosyVoice3LM(config, mesh_device, dtype=ttnn.bfloat16)
    return llm


def test_cosyvoice_llm_inference(cosyvoice_llm):
    """Test a small dummy inference run on CosyVoice3LM."""
    # Create dummy inputs
    text = torch.tensor([[100, 101, 102]], dtype=torch.int32)
    text_len = torch.tensor([3], dtype=torch.int32)

    # CosyVoice formatting usually puts the target text in prompt_text as well
    # For dummy testing, we don't need real embeddings, just need to see if it steps
    prompt_text = torch.tensor([[100, 151646, 101, 102]], dtype=torch.int32)
    prompt_text_len = torch.tensor([4], dtype=torch.int32)

    prompt_speech_token = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    prompt_speech_token_len = torch.tensor([3], dtype=torch.int32)

    # Embeddings (not used by our mock inference yet, but needed for signature)
    embedding = torch.zeros(1, 192)

    # Run a short inference
    generator = cosyvoice_llm.inference(
        text=text,
        text_len=text_len,
        prompt_text=prompt_text,
        prompt_text_len=prompt_text_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        embedding=embedding,
        sampling=25,
        max_token_text_ratio=2,  # Will limit max_len to 3 * 2 = 6 tokens
        min_token_text_ratio=1,
    )

    output_tokens = []
    for token in generator:
        output_tokens.append(token)
        if len(output_tokens) >= 5:
            break

    assert len(output_tokens) > 0, "No tokens generated"
    assert all(isinstance(t, int) for t in output_tokens), "Tokens should be ints"
    print(f"Generated tokens: {output_tokens}")
