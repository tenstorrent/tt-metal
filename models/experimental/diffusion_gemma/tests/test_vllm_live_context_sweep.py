# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch


def test_vllm_hybrid_kv_spec_keeps_one_full_attention_head_per_device():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt.generator_vllm import DiffusionGemmaForCausalLM

    text_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        num_key_value_heads=8,
        head_dim=256,
        sliding_window=1024,
        num_global_key_value_heads=2,
        global_head_dim=512,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=text_config), dtype=torch.bfloat16),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )

    specs = DiffusionGemmaForCausalLM.get_kv_cache_spec(vllm_config)

    assert specs["model.layers.0.self_attn"].num_kv_heads == 2
    assert specs["model.layers.1.self_attn"].num_kv_heads == 1


@pytest.mark.parametrize(("flag", "expected_upfront"), [("1", True), ("0", False)])
def test_vllm_session_selects_upfront_or_eager_from_sole_model_trace_flag(monkeypatch, flag, expected_upfront):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_UPFRONT_CAPTURE", flag)
    captured = {}

    def fake_session(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(generator_vllm, "BlockDiffusionServingSession", fake_session)
    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._upfront = generator_vllm.upfront_capture_enabled()
    model.model = [SimpleNamespace()]
    model._dg_state_dict = {}
    model._config = SimpleNamespace(canvas_length=256, max_denoise_steps=48)
    model._tokenizer = None
    model._gumbel_mode = "host"
    model.canvas_length = 256

    model._make_session()

    expected = generator_vllm.upfront_traced_denoise_block if expected_upfront else None
    assert captured["denoise_block_fn"] is expected


def test_vllm_prefill_failure_resets_unregistered_session(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []

    class _Session:
        def prefill(self, prompt):
            assert prompt == "prompt"
            return 32

        def decode_block(self):
            raise RuntimeError("injected block-0 failure")

        def reset(self):
            events.append("reset")

    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._sessions = {}
    model._make_session = lambda: _Session()
    model._prompt_tokens_for_row = lambda tokens, prompt_lens, row: "prompt"

    with expect_error(RuntimeError, match="injected block-0 failure"):
        model.prefill_forward(SimpleNamespace(shape=(1, 32)))

    assert events == ["reset"]
    assert model._sessions == {}


def test_vllm_decode_failure_releases_registered_session(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    class _Session:
        finished = False

        def decode_block(self):
            raise RuntimeError("injected replay failure")

    model = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    model.data_parallel = 1
    model._sessions = {3: _Session()}
    released = []

    def release(row):
        released.append(row)
        model._sessions.pop(row)

    model.release_request = release

    with expect_error(RuntimeError, match="injected replay failure"):
        model.decode_forward()

    assert released == [3]
    assert model._sessions == {}
