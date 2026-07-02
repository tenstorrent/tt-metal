# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.demo import text_demo
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import PromptPrefill


def test_prefill_prompt_tokenizes_and_writes_prompt_kv(monkeypatch):
    calls = {}
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)
    result = PromptPrefill(prompt_len=3, cache_len=32)

    def fake_tokenize_prompt(tokenizer, prompt):
        calls["tokenize"] = (tokenizer, prompt)
        return prompt_tokens

    def fake_prefill_prompt_tokens(tt_model, tokens):
        calls["prefill"] = (tt_model, tokens)
        return result

    monkeypatch.setattr(G, "tokenize_prompt", fake_tokenize_prompt)
    monkeypatch.setattr(G, "prefill_prompt_tokens", fake_prefill_prompt_tokens)

    checkpoint_model_inputs = SimpleNamespace(tokenizer="tokenizer", tt_model="tt-model")

    out = text_demo._prefill_prompt(checkpoint_model_inputs, "hello")

    assert out == result
    assert calls["tokenize"] == ("tokenizer", "hello")
    assert calls["prefill"] == ("tt-model", prompt_tokens)


def test_generation_success_summary_reports_blocks_and_text_chars():
    generation = SimpleNamespace(
        generation=SimpleNamespace(
            generated=torch.zeros((1, 512), dtype=torch.long),
            trajectories=[object(), object()],
            prompt_len=18,
            next_pos=544,
        ),
        sequences=torch.zeros((1, 530), dtype=torch.long),
        text=["", "ok"],
    )

    assert text_demo._generation_success_summary(generation) == (
        "DG_TEXT_DEMO_SUCCESS "
        "generated_tokens=512 "
        "blocks=2 "
        "prompt_len=18 "
        "next_pos=544 "
        "sequence_len=530 "
        "text_count=2 "
        "text_chars=2"
    )


def test_parse_success_summary_returns_integer_fields():
    fields = text_demo._parse_success_summary(
        "DG_TEXT_DEMO_SUCCESS "
        "generated_tokens=512 "
        "blocks=2 "
        "prompt_len=32 "
        "next_pos=544 "
        "sequence_len=530 "
        "text_count=1 "
        "text_chars=1409"
    )

    assert fields == {
        "generated_tokens": 512,
        "blocks": 2,
        "prompt_len": 32,
        "next_pos": 544,
        "sequence_len": 530,
        "text_count": 1,
        "text_chars": 1409,
    }


def test_parse_success_summary_rejects_malformed_summary():
    with pytest.raises(ValueError, match="must start"):
        text_demo._parse_success_summary("DG_TEXT_DEMO_FAILURE mode=generate")
    with pytest.raises(ValueError, match="malformed"):
        text_demo._parse_success_summary("DG_TEXT_DEMO_SUCCESS blocks")


def test_run_mode_reports_selected_smoke_mode():
    parser = text_demo.build_arg_parser()

    assert text_demo._run_mode(parser.parse_args([])) == "generate"
    assert text_demo._run_mode(parser.parse_args(["--build-only"])) == "build-only"
    assert text_demo._run_mode(parser.parse_args(["--prefill-only"])) == "prefill-only"
    assert text_demo._run_mode(parser.parse_args(["--adapter-only"])) == "adapter-only"


def test_failure_summary_is_single_greppable_line():
    parser = text_demo.build_arg_parser()
    args = parser.parse_args(["--build-only", "--mesh", "P150x4"])

    summary = text_demo._failure_summary(args, RuntimeError("boom"))

    assert summary == ("DG_TEXT_DEMO_FAILURE mode=build-only mesh=P150x4 error_type=RuntimeError")
    assert "\n" not in summary


def test_main_logs_failure_marker_and_reraises(monkeypatch):
    logged = {}

    def fake_run(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(text_demo, "_run", fake_run)
    monkeypatch.setattr(text_demo.logger, "error", lambda msg: logged.setdefault("error", msg))

    with pytest.raises(RuntimeError, match="kaboom"):
        text_demo.main(["--build-only", "--mesh", "P150x4"])

    assert logged["error"] == ("DG_TEXT_DEMO_FAILURE mode=build-only mesh=P150x4 error_type=RuntimeError")


def test_text_demo_rejects_conflicting_smoke_modes():
    parser = text_demo.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--build-only", "--prefill-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--prefill-only", "--adapter-only"])


def test_adapter_logits_once_prefills_and_calls_real_builder(monkeypatch):
    calls = {}
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)

    class _FakeDeviceCanvas:
        shape = (1, 1, 256, 1)

        def __init__(self):
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeLogits:
        shape = (1, 1, 256, 1024)

    class _FakeAdapter:
        def __init__(self):
            self.reset_called = False

        def __call__(self, canvas, step):
            calls["adapter_call"] = (canvas, step)
            return _FakeLogits()

        def reset(self):
            self.reset_called = True
            calls["adapter_reset"] = True

    fake_canvas = _FakeDeviceCanvas()
    fake_adapter = _FakeAdapter()

    def fake_tokenize_prompt(tokenizer, prompt):
        calls["tokenize"] = (tokenizer, prompt)
        return prompt_tokens

    def fake_prefill_prompt_tokens(tt_model, tokens):
        calls["prefill"] = (tt_model, tokens)
        return PromptPrefill(prompt_len=3, cache_len=32)

    def fake_builder_factory(state_dict, **kwargs):
        calls["builder_factory"] = (state_dict, kwargs)

        def fake_builder(tt_model, **builder_kwargs):
            calls["builder"] = (tt_model, builder_kwargs)
            return fake_adapter

        return fake_builder

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls["host_canvas"] = (mesh_device, canvas)
        return fake_canvas

    monkeypatch.setattr(G, "tokenize_prompt", fake_tokenize_prompt)
    monkeypatch.setattr(G, "prefill_prompt_tokens", fake_prefill_prompt_tokens)
    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)
    monkeypatch.setattr(DF, "make_generation_logits_fn_builder_from_checkpoint_state", fake_builder_factory)

    tt_model = SimpleNamespace(mesh_device="mesh", hf_config=SimpleNamespace(vocab_size=1024))
    checkpoint_model_inputs = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=1024),
        tt_model=tt_model,
        state_dict={"raw": "state"},
    )

    out = text_demo._adapter_logits_once(checkpoint_model_inputs, "hello", canvas_length=256, seed=123)

    assert out == (1, 1, 256, 1024)
    assert calls["tokenize"] == (checkpoint_model_inputs.tokenizer, "hello")
    assert calls["prefill"][0] is tt_model
    assert calls["prefill"][1] is prompt_tokens
    assert calls["builder_factory"] == ({"raw": "state"}, {"config": tt_model.hf_config})
    assert calls["builder"][0] is tt_model
    assert calls["builder"][1]["prompt_tokens"] is prompt_tokens
    assert calls["builder"][1]["prompt_len"] == 32
    assert calls["host_canvas"][0] == "mesh"
    assert calls["host_canvas"][1].shape == (1, 256)
    assert calls["adapter_call"] == (fake_canvas, 0)
    assert calls["adapter_reset"] is True
    assert fake_canvas.deallocated is True


def test_text_demo_disable_eos_stop_threads_generation_kwargs(monkeypatch):
    calls = {}

    class _FakeMesh:
        def get_num_devices(self):
            return 1

    fake_generation = SimpleNamespace(
        generation=SimpleNamespace(
            generated=torch.zeros((1, 64), dtype=torch.long),
            trajectories=[object(), object()],
            prompt_len=1024,
            next_pos=1088,
        ),
        sequences=torch.zeros((1, 1088), dtype=torch.long),
        text=["ok"],
    )

    def fake_generate(checkpoint_model_inputs, prompt, **kwargs):
        calls["generate"] = (checkpoint_model_inputs, prompt, kwargs)
        return fake_generation

    monkeypatch.setattr(text_demo, "load_checkpoint_inputs", lambda *args, **kwargs: "checkpoint-inputs")
    monkeypatch.setattr(text_demo, "_open_mesh_device", lambda mesh: _FakeMesh())
    monkeypatch.setattr(text_demo, "_close_mesh_device", lambda mesh: calls.setdefault("closed", mesh))
    monkeypatch.setattr(text_demo, "_log_mesh_dram", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_demo,
        "build_tt_model_from_checkpoint_inputs",
        lambda *args, **kwargs: SimpleNamespace(tokenizer="tok", tt_model="model", state_dict={}),
    )
    monkeypatch.setattr(text_demo, "generate_text_from_checkpoint_model_inputs", fake_generate)

    assert (
        text_demo.main(
            [
                "--checkpoint",
                "/tmp/ckpt",
                "--local-files-only",
                "--disable-eos-stop",
                "--num-blocks",
                "2",
                "--max-new-tokens",
                "64",
            ]
        )
        == 0
    )

    _, _, kwargs = calls["generate"]
    assert kwargs["eos_token_id"] is None
    assert kwargs["stop_token_ids"] == []
    assert kwargs["decode_kwargs"] == {"skip_special_tokens": True}
    assert kwargs["num_blocks"] == 2
    assert kwargs["max_new_tokens"] == 64
