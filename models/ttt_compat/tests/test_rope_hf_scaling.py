# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""CPU-only check that get_rot_mats_hf matches transformers' rotary tables for every rope type."""
from unittest import mock

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GptOssConfig, LlamaConfig, Phi3Config, Qwen3Config  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

from models.ttt_compat.tt import rope  # noqa: E402
from models.ttt_compat.tt.common import rope_scaling_model_factory  # noqa: E402

SEQ = 4096


def _llama(rp):
    return (
        LlamaConfig(hidden_size=4096, num_attention_heads=32, max_position_embeddings=131072, rope_parameters=rp),
        128,
    )


def _qwen3_yarn():
    rp = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
    }
    cfg = Qwen3Config(
        hidden_size=4096, num_attention_heads=32, head_dim=128, max_position_embeddings=131072, rope_parameters=rp
    )
    return cfg, 128


def _gptoss_yarn():
    rp = {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
        "rope_theta": 150000,
    }
    return GptOssConfig(max_position_embeddings=131072, rope_parameters=rp), 64


def _longrope():
    dim = 96
    rp = {
        "rope_type": "longrope",
        "short_factor": [1.0 + 0.01 * i for i in range(dim // 2)],
        "long_factor": [2.0 + 0.05 * i for i in range(dim // 2)],
        "rope_theta": 10000.0,
    }
    cfg = Phi3Config(
        hidden_size=3072,
        num_attention_heads=32,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_parameters=rp,
    )
    return cfg, dim


CASES = {
    "llama3_default_bands": lambda: _llama(
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
        }
    ),
    "llama3_custom_bands": lambda: _llama(
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 2.0,
            "high_freq_factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
        }
    ),
    "linear": lambda: _llama({"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0}),
    "yarn_qwen3": _qwen3_yarn,
    "yarn_gptoss": _gptoss_yarn,
    "longrope": _longrope,
    "default": lambda: _llama({"rope_type": "default", "rope_theta": 10000.0}),
}


def _hf_tables(cfg, seq_len):
    rp = cfg.rope_parameters
    rope_type = rp.get("rope_type", "default")
    if rope_type == "default":
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        inv, att = 1.0 / (rp["rope_theta"] ** (torch.arange(0, head_dim, 2).float() / head_dim)), 1.0
    else:
        inv, att = ROPE_INIT_FUNCTIONS[rope_type](cfg, "cpu", seq_len=seq_len)
    freqs = torch.outer(torch.arange(seq_len).float(), inv.float())
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos() * att, emb.sin() * att


def _captured_tables(cfg, dim):
    rp = dict(cfg.rope_parameters)
    scaling = None if rp.get("rope_type", "default") == "default" else rope_scaling_model_factory(rp)
    captured = []

    def fake_from_torch(t, **kwargs):
        captured.append(t)
        return t

    with mock.patch.object(rope.ttnn, "from_torch", fake_from_torch), mock.patch.object(
        rope, "replicate_tensor_to_mesh_mapper", lambda device: None
    ):
        rope.get_rot_mats_hf(dim, None, SEQ, rp["rope_theta"], scaling)
    cos, sin = captured
    return cos[0, 0].float(), sin[0, 0].float()


@pytest.mark.parametrize("name", list(CASES))
def test_get_rot_mats_hf_matches_transformers(name):
    cfg, dim = CASES[name]()
    ref_cos, ref_sin = _hf_tables(cfg, SEQ)
    cos, sin = _captured_tables(cfg, dim)
    assert cos.shape == ref_cos.shape
    torch.testing.assert_close(cos, ref_cos, atol=5e-4, rtol=0)
    torch.testing.assert_close(sin, ref_sin, atol=5e-4, rtol=0)
