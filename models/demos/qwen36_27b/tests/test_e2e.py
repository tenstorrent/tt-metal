# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
E2E tests for Qwen3.6-27B on TT device.

Tests:
  1. Smoke test with dummy weights (2 layers) — verifies the pipeline runs without errors
  2. DeltaNet layer decode correctness — verifies TT output matches PyTorch reference
  3. Full generate pipeline — prefill + decode loop
"""

import torch
import pytest
import ttnn

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import create_dummy_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator


@pytest.fixture
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture
def small_config():
    """2-layer model for fast testing."""
    return Qwen36ModelConfig(
        hidden_size=256,
        num_hidden_layers=4,  # 3 deltanet + 1 attention
        full_attention_interval=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,
        max_seq_len=128,
    )


def test_smoke_e2e(device, small_config):
    """Verify the full pipeline runs without errors on device."""
    state_dict = create_dummy_state_dict(small_config, num_layers=4)
    model = TtQwen36Model(device, state_dict, small_config)
    generator = Qwen36Generator(model, small_config)

    prompt = torch.tensor([[1, 42, 100, 7]])  # 4 tokens
    logits = generator.prefill(prompt)

    assert logits is not None
    logits_cpu = ttnn.to_torch(logits)
    assert logits_cpu.shape[-1] >= small_config.vocab_size

    token_in = torch.tensor([[42]])
    _, next_token = generator.decode_one_token(token_in)
    assert next_token.shape == (1, 1)
    assert next_token.item() < small_config.padded_vocab_size


def test_deltanet_decode_correctness(device, small_config):
    """Verify TT DeltaNet decode matches PyTorch reference numerically."""
    torch.manual_seed(42)
    from models.demos.qwen36_27b.reference.deltanet_reference import GatedDeltaNetLayer as RefLayer, Qwen36Config
    from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState

    ref_config = Qwen36Config(
        hidden_size=small_config.hidden_size,
        linear_num_key_heads=small_config.linear_num_key_heads,
        linear_num_value_heads=small_config.linear_num_value_heads,
        linear_key_head_dim=small_config.linear_key_head_dim,
        linear_value_head_dim=small_config.linear_value_head_dim,
        linear_conv_kernel_dim=small_config.linear_conv_kernel_dim,
        intermediate_size=small_config.intermediate_size,
    )

    ref_layer = RefLayer(ref_config, layer_idx=0)
    ref_layer.eval()

    ref_sd = ref_layer.state_dict()
    tt_sd = {}
    for k, v in ref_sd.items():
        tt_sd[f"model.layers.0.linear_attn.{k}"] = v

    tt_layer = TtGatedDeltaNet(device, tt_sd, layer_idx=0, config=small_config)
    state = TtDeltaNetState(1, ["linear_attention"], device, small_config)

    x = torch.randn(1, 1, small_config.hidden_size)
    x_tt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with torch.no_grad():
        ref_out, _, _ = ref_layer(x)

    tt_out = tt_layer(x_tt, state, mode="decode")
    tt_out_cpu = ttnn.to_torch(tt_out).reshape_as(ref_out)

    diff = (tt_out_cpu.float() - ref_out.float()).abs().max()
    print(f"DeltaNet decode max diff: {diff:.2e}")
    assert diff < 0.1, f"DeltaNet decode diff too large: {diff}"


def test_generate_pipeline(device, small_config):
    """Test full generate loop produces tokens."""
    state_dict = create_dummy_state_dict(small_config, num_layers=4)
    model = TtQwen36Model(device, state_dict, small_config)
    generator = Qwen36Generator(model, small_config)

    prompt = torch.tensor([[1, 2, 3]])
    generated = generator.generate(prompt, max_new_tokens=5)

    assert len(generated) == 5
    assert all(isinstance(t, int) for t in generated)
    assert all(0 <= t < small_config.padded_vocab_size for t in generated)
    print(f"Generated tokens: {generated}")
