# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import types

import torch

from models.demos.hf_eager.hunyuanvideo_1_5.tt.byt5_encoder import analyze_byt5_support
from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import HunyuanVideo15Pipeline, TTTransformerAdapter
from models.demos.hf_eager.hunyuanvideo_1_5.tt.text_conditioning import encode_prompt_pair


def _hunyuan_byt5_config(**overrides):
    values = dict(
        architectures=["T5EncoderModel"],
        d_model=1472,
        d_ff=3584,
        d_kv=64,
        num_heads=6,
        num_layers=12,
        vocab_size=1510,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new",
        is_encoder_decoder=False,
        is_gated_act=True,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_hunyuan_byt5_accepts_dedicated_tp2_with_independent_attention_width():
    support = analyze_byt5_support(_hunyuan_byt5_config(), (1, 2))
    assert support.supported
    assert support.strategy == "TP2-axis1"
    assert "independent width 384" in support.reason


def test_hunyuan_byt5_fails_closed_on_full_dit_mesh():
    support = analyze_byt5_support(_hunyuan_byt5_config(), (8, 4))
    assert not support.supported
    assert "the 8x4 DiT mesh cannot express it" in support.reason


def test_hunyuan_byt5_fails_closed_on_checkpoint_variant():
    support = analyze_byt5_support(_hunyuan_byt5_config(d_kv=128), (1, 2))
    assert not support.supported
    assert "d_kv=128" in support.reason


def test_padding_trim_removes_encoder_specific_padding_values():
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    valid = torch.randn(1, 3, 8)
    host = torch.cat([valid, torch.randn(1, 2, 8)], dim=1)
    tt = torch.cat([valid, torch.full((1, 2, 8), 1.0e4)], dim=1)
    host_trimmed, _ = HunyuanVideo15Pipeline._trim_to_valid(host, mask)
    tt_trimmed, _ = HunyuanVideo15Pipeline._trim_to_valid(tt, mask)
    torch.testing.assert_close(host_trimmed, tt_trimmed)


class _FakeTransformer:
    config = types.SimpleNamespace()
    dtype = torch.bfloat16


class _FakeTTPipeline:
    def __init__(self):
        self.calls = []

    def run(self, inputs, granularity):
        mllm, _ = HunyuanVideo15Pipeline._trim_to_valid(
            inputs["encoder_hidden_states"], inputs["encoder_attention_mask"]
        )
        byt5, _ = HunyuanVideo15Pipeline._trim_to_valid(
            inputs["encoder_hidden_states_2"], inputs["encoder_attention_mask_2"]
        )
        self.calls.append((mllm.shape[1], byt5.shape[1]))
        return torch.zeros(inputs["hidden_states"].shape[0], 2, 1, 1, 1)


def _condition(mllm_length, byt5_length):
    return dict(
        hidden_states=torch.zeros(1, 1, 1, 1, 1),
        timestep=torch.zeros(1),
        encoder_hidden_states=torch.randn(1, 8, 4),
        encoder_attention_mask=torch.tensor([[1] * mllm_length + [0] * (8 - mllm_length)]),
        encoder_hidden_states_2=torch.randn(1, 6, 4),
        encoder_attention_mask_2=torch.tensor([[1] * byt5_length + [0] * (6 - byt5_length)]),
        image_embeds=torch.zeros(1, 1, 1),
    )


def test_mixed_length_cfg_runs_each_condition_at_its_valid_lengths(monkeypatch):
    monkeypatch.setenv("HY_CFG_PADDING_POLICY", "separate")
    ttpipe = _FakeTTPipeline()
    guider = types.SimpleNamespace(num_conditions=2)
    adapter = TTTransformerAdapter(_FakeTransformer(), ttpipe, guider)

    first = adapter(return_dict=False, **_condition(7, 1))[0]
    second = adapter(return_dict=False, **_condition(3, 4))[0]

    assert first.numel() > 0  # placeholder was backfilled after the second condition arrived
    assert second.shape == (1, 2, 1, 1, 1)
    assert ttpipe.calls == [(7, 1), (3, 4)]


def test_prompt_embedding_cache_skips_warm_encode(tmp_path):
    class FakePipe:
        text_encoder = types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="qwen-test"))
        text_encoder_2 = types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="byt5-test"))
        tokenizer = types.SimpleNamespace(name_or_path="qwen-tokenizer", vocab_size=151936)
        tokenizer_2 = types.SimpleNamespace(name_or_path="byt5-tokenizer", vocab_size=1510)
        tokenizer_max_length = 1000
        tokenizer_2_max_length = 256
        system_message = "system"
        prompt_template_encode_start_idx = 108
        transformer = _FakeTransformer()
        guider = types.SimpleNamespace(_enabled=True, num_conditions=2)
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def encode_prompt(self, prompt, **kwargs):
            self.calls += 1
            value = float(self.calls)
            return (
                torch.full((1, 4, 3), value),
                torch.ones(1, 4),
                torch.full((1, 2, 2), value),
                torch.ones(1, 2),
            )

    pipe = FakePipe()
    cold, cold_hit = encode_prompt_pair(pipe, "unique cache prompt", "", cache_dir=tmp_path)
    warm, warm_hit = encode_prompt_pair(pipe, "unique cache prompt", "", cache_dir=tmp_path)
    assert not cold_hit and warm_hit
    assert pipe.calls == 2
    assert cold.keys() == warm.keys()
    for key in cold:
        torch.testing.assert_close(cold[key], warm[key])
