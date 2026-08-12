# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the Gemma-4 encoder's config parsing, RoPE tables and
fused-QKV packing. No device: these cover the parts that are pure torch, so the
on-device parity test is left to explain only genuine numerical drift.
"""

import types

import pytest
import torch

from models.tt_dit.encoders.gemma4.model_gemma import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4Attention,
    Gemma4Config,
    Gemma4RotaryEmbedding,
)

# The text_config the packed LTX-2.5 encoder ships
# (gemma4-12b-with-proj-ltx-2.5-bf16.safetensors), trimmed to the keys the port reads.
LTX_TEXT_CONFIG = {
    "vocab_size": 262144,
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "rms_norm_eps": 1e-6,
    "sliding_window": 1024,
    "max_position_embeddings": 262144,
    "layer_types": [SLIDING_ATTENTION if (i + 1) % 6 else FULL_ATTENTION for i in range(48)],
    "rope_parameters": {
        FULL_ATTENTION: {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10000.0},
    },
}

HIDDEN = 3840


def test_config_matches_shipped_checkpoint():
    config = Gemma4Config.from_hf_text_config(LTX_TEXT_CONFIG)

    assert [i for i in range(48) if config.is_global(i)] == [5, 11, 17, 23, 29, 35, 41, 47]
    assert (config.attn_head_dim(False), config.attn_kv_heads(False)) == (256, 8)
    assert (config.attn_head_dim(True), config.attn_kv_heads(True)) == (512, 1)
    assert (config.rope_theta, config.global_rope_theta) == (10000.0, 1000000.0)
    assert config.partial_rotary_factor == 0.25
    assert config.attention_k_eq_v


def test_default_layer_pattern_matches_explicit_types():
    explicit = Gemma4Config.from_hf_text_config(LTX_TEXT_CONFIG)
    derived = Gemma4Config()
    assert derived.layer_types == explicit.layer_types


@pytest.mark.parametrize(
    "head_dim, factor, expected_rotated",
    [(512, 0.25, 64), (256, 1.0, 128), (512, 0.5, 128)],
)
def test_partial_rotary_leaves_tail_unrotated(head_dim, factor, expected_rotated):
    """Proportional RoPE keeps the table full-width and zeroes the trailing frequencies,
    so those dimensions pass through with cos=1, sin=0."""
    rope = Gemma4RotaryEmbedding(None, head_dim=head_dim, base=1e6, max_seq_len=64, partial_rotary_factor=factor)
    cos, sin = rope._cos_cached[0, 0], rope._sin_cached[0, 0]

    assert cos.shape == (64, head_dim // 2)
    # Positions beyond the first are only constant where the frequency is zero.
    rotated = (sin[1:] != 0).any(dim=0).sum().item()
    assert rotated == expected_rotated
    assert torch.all(cos[:, expected_rotated:] == 1.0)
    assert torch.all(sin[:, expected_rotated:] == 0.0)


def test_rope_matches_reference_frequencies():
    """Frequencies follow base**(-2i/head_dim) over the rotated prefix, in float32 —
    the reference builds these in float32 and float64 here would shift entries across a
    bfloat16 rounding boundary."""
    head_dim, base, factor = 512, 1000000.0, 0.25
    rope = Gemma4RotaryEmbedding(None, head_dim=head_dim, base=base, max_seq_len=8, partial_rotary_factor=factor)

    rope_angles = int(factor * head_dim // 2)
    inv_freq = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(torch.float32) / head_dim))
    inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - rope_angles)])
    expected = torch.outer(torch.arange(8, dtype=torch.float32), inv_freq)

    assert torch.equal(rope._cos_cached[0, 0], expected.cos())
    assert torch.equal(rope._sin_cached[0, 0], expected.sin())


def _attention_stub(*, tp, num_heads, num_kv_heads, head_dim, k_eq_v):
    kv_replicated = num_kv_heads < tp
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_local_heads=num_heads // tp,
        num_local_kv_heads=1 if kv_replicated else num_kv_heads // tp,
        kv_replicated=kv_replicated,
        k_eq_v=k_eq_v,
        parallel_config=types.SimpleNamespace(tensor_parallel=types.SimpleNamespace(factor=tp)),
    )


def _head_tagged(num, head_dim, tag):
    """Weight [num*head_dim, HIDDEN] whose every row of head h holds ``tag + h``."""
    weight = torch.zeros(num * head_dim, HIDDEN)
    for head in range(num):
        weight[head * head_dim : (head + 1) * head_dim] = tag + head
    return weight


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("is_global", [False, True])
def test_fused_qkv_gives_each_device_its_heads(tp, is_global):
    num_heads, head_dim = 16, (512 if is_global else 256)
    num_kv_heads = 1 if is_global else 8
    k_eq_v = is_global

    stub = _attention_stub(tp=tp, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, k_eq_v=k_eq_v)
    state = {
        "q_proj.weight": _head_tagged(num_heads, head_dim, 100.0),
        "k_proj.weight": _head_tagged(num_kv_heads, head_dim, 200.0),
    }
    if not k_eq_v:
        state["v_proj.weight"] = _head_tagged(num_kv_heads, head_dim, 300.0)

    Gemma4Attention._prepare_torch_state(stub, state)

    assert "q_proj.weight" not in state and "k_proj.weight" not in state
    fused = state["wqkv.weight"]
    local_q, local_kv = stub.num_local_heads, stub.num_local_kv_heads
    per_device = (local_q + 2 * local_kv) * head_dim
    assert fused.shape == (tp * per_device, HIDDEN)

    q_per_device = num_heads // tp
    for device in range(tp):
        chunk = fused[device * per_device : (device + 1) * per_device]
        q_part = chunk[: local_q * head_dim]
        k_part = chunk[local_q * head_dim : (local_q + local_kv) * head_dim]
        v_part = chunk[(local_q + local_kv) * head_dim :]

        for head in range(local_q):
            expected = 100.0 + device * q_per_device + head
            assert torch.all(q_part[head * head_dim : (head + 1) * head_dim] == expected)

        # Replicated KV hands each device the head its Q heads map to under GQA.
        kv_base = (device * q_per_device) * num_kv_heads // num_heads if stub.kv_replicated else device * local_kv
        for head in range(local_kv):
            expected_k = 200.0 + kv_base + head
            assert torch.all(k_part[head * head_dim : (head + 1) * head_dim] == expected_k)
            expected_v = expected_k if k_eq_v else 300.0 + kv_base + head
            assert torch.all(v_part[head * head_dim : (head + 1) * head_dim] == expected_v)


def test_layer_scalar_is_consumed_from_state():
    from models.tt_dit.encoders.gemma4.model_gemma import Gemma4EncoderLayer

    stub = types.SimpleNamespace(layer_scalar=1.0)
    state = {"layer_scalar": torch.tensor([2.5]), "input_layernorm.weight": torch.zeros(4)}
    Gemma4EncoderLayer._prepare_torch_state(stub, state)

    assert stub.layer_scalar == pytest.approx(2.5)
    assert "layer_scalar" not in state
    assert "self_attn.input_layernorm.weight" in state
