# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DiffusionGemma text demo entry point, and the RUN-first regression target (#47464).

The host tests cover the CLI surface, the greppable success/failure summary lines and the
``main()`` wiring. The device-gated RUN smokes pin the canonical hardware runs that passed
on QB2 2026-07-02 as reproducible smoke tests so future changes cannot silently break
"prompt -> committed blocks -> decoded text" without crashing. Output *correctness* is
explicitly deferred (#48291); the RUN gate is a clean exit + the ``DG_TEXT_DEMO_SUCCESS``
marker, not text quality. The smokes disable EOS stopping so degenerate EOS-heavy output
still commits the requested blocks and exercises cross-block KV / position advancement.
``max_seq_len`` must cover ``prompt + num_blocks * canvas``; the two-block smoke uses 1024
so the second block's RoPE/cache slices do not overrun after the 32-token-aligned prompt
prefix.

Run the device smokes on QB2::

    source /home/zni/venvs/tt-diffusion-gemma/bin/activate
    export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn TT_METAL_HOME=/home/zni/tt-metal
    export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal MESH_DEVICE=P150x4 DG_RUN_DEVICE=1
    pytest models/experimental/diffusion_gemma/tests/test_demo.py -q

Override the checkpoint / mesh with ``DG_CKPT`` / ``MESH_DEVICE``; set
``DG_TEXT_DEMO_NUM_LAYERS`` for a cheaper reduced-depth short-prompt smoke.
The long-prompt smoke defaults to one layer; set
``DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS=full`` to run all layers.
The 256K-allocation smoke also defaults to one layer; set
``DG_TEXT_DEMO_256K_NUM_LAYERS=full`` to run all layers. The full-depth
256K variant uses ``--argmax-sampling`` because the default full-vocab device
Gumbel allocation is a known DRAM-fragmentation OOM at that size.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.demo import text_demo
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import PromptPrefill

requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)

_DEFAULT_CKPT = "/home/zni/dg_models/diffusiongemma-26B-A4B-it"


class _FakeMesh:
    def get_num_devices(self):
        return 1


def _checkpoint() -> str:
    return os.environ.get("DG_CKPT", _DEFAULT_CKPT)


def _require_local_checkpoint() -> str:
    checkpoint = _checkpoint()
    if not os.path.isdir(checkpoint):
        pytest.skip(f"checkpoint not found at {checkpoint!r}; set DG_CKPT")
    return checkpoint


def _patch_prompt_prefill(monkeypatch, calls, prompt_tokens, prefill):
    def fake_tokenize_prompt(tokenizer, prompt):
        calls["tokenize"] = (tokenizer, prompt)
        return prompt_tokens

    def fake_prefill_prompt_tokens(tt_model, tokens):
        calls["prefill"] = (tt_model, tokens)
        return prefill

    monkeypatch.setattr(G, "tokenize_prompt", fake_tokenize_prompt)
    monkeypatch.setattr(G, "prefill_prompt_tokens", fake_prefill_prompt_tokens)


def _fake_generation(generated_tokens, blocks, prompt_len, next_pos):
    return SimpleNamespace(
        generation=SimpleNamespace(
            generated=torch.zeros((1, generated_tokens), dtype=torch.long),
            trajectories=[object() for _ in range(blocks)],
            prompt_len=prompt_len,
            next_pos=next_pos,
        ),
        sequences=torch.zeros((1, next_pos), dtype=torch.long),
        text=["ok"],
    )


def _patch_main_pipeline(monkeypatch, calls, generation):
    def fake_generate(checkpoint_model_inputs, prompt, **kwargs):
        calls["generate"] = (checkpoint_model_inputs, prompt, kwargs)
        return generation

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


def _base_argv(checkpoint) -> list[str]:
    return [
        "--checkpoint",
        checkpoint,
        "--local-files-only",
        "--mesh",
        os.environ.get("MESH_DEVICE", "P150x4"),
    ]


def _demo_success_summary(monkeypatch, argv):
    info_lines: list[str] = []
    monkeypatch.setattr(text_demo.logger, "info", lambda message: info_lines.append(str(message)))

    assert text_demo.main(argv) == 0

    success_lines = [line for line in info_lines if line.startswith("DG_TEXT_DEMO_SUCCESS ")]
    assert len(success_lines) == 1
    return text_demo._parse_success_summary(success_lines[0])


# --- CLI arguments -------------------------------------------------------------------


def test_run_mode_reports_selected_smoke_mode():
    parser = text_demo.build_arg_parser()

    assert text_demo._run_mode(parser.parse_args([])) == "generate"
    assert text_demo._run_mode(parser.parse_args(["--build-only"])) == "build-only"
    assert text_demo._run_mode(parser.parse_args(["--prefill-only"])) == "prefill-only"
    assert text_demo._run_mode(parser.parse_args(["--adapter-only"])) == "adapter-only"


def test_text_demo_rejects_conflicting_smoke_modes(expect_error):
    parser = text_demo.build_arg_parser()

    with expect_error(SystemExit):
        parser.parse_args(["--build-only", "--prefill-only"])
    with expect_error(SystemExit):
        parser.parse_args(["--prefill-only", "--adapter-only"])


# --- success / failure summary lines -------------------------------------------------


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


def test_parse_success_summary_rejects_malformed_summary(expect_error):
    with expect_error(ValueError, match="must start"):
        text_demo._parse_success_summary("DG_TEXT_DEMO_FAILURE mode=generate")
    with expect_error(ValueError, match="malformed"):
        text_demo._parse_success_summary("DG_TEXT_DEMO_SUCCESS blocks")


# --- prompt prefill and adapter ------------------------------------------------------


def test_prefill_prompt_tokenizes_and_writes_prompt_kv(monkeypatch):
    calls = {}
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)
    result = PromptPrefill(prompt_len=3, cache_len=32)
    _patch_prompt_prefill(monkeypatch, calls, prompt_tokens, result)

    checkpoint_model_inputs = SimpleNamespace(tokenizer="tokenizer", tt_model="tt-model")

    out = text_demo._prefill_prompt(checkpoint_model_inputs, "hello")

    assert out == result
    assert calls["tokenize"] == ("tokenizer", "hello")
    assert calls["prefill"] == ("tt-model", prompt_tokens)


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

    def fake_builder_factory(state_dict, **kwargs):
        calls["builder_factory"] = (state_dict, kwargs)

        def fake_builder(tt_model, **builder_kwargs):
            calls["builder"] = (tt_model, builder_kwargs)
            return fake_adapter

        return fake_builder

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls["host_canvas"] = (mesh_device, canvas)
        return fake_canvas

    _patch_prompt_prefill(monkeypatch, calls, prompt_tokens, PromptPrefill(prompt_len=3, cache_len=32))
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


# --- main() wiring -------------------------------------------------------------------


def test_text_demo_disable_eos_stop_threads_generation_kwargs(monkeypatch):
    calls = {}
    _patch_main_pipeline(monkeypatch, calls, _fake_generation(64, 2, 1024, 1088))

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


def test_text_demo_argmax_sampling_threads_no_gumbel_hook(monkeypatch):
    calls = {}
    _patch_main_pipeline(monkeypatch, calls, _fake_generation(32, 1, 32, 64))

    assert (
        text_demo.main(
            [
                "--checkpoint",
                "/tmp/ckpt",
                "--local-files-only",
                "--argmax-sampling",
                "--num-blocks",
                "1",
                "--max-new-tokens",
                "32",
            ]
        )
        == 0
    )

    _, _, kwargs = calls["generate"]
    assert callable(kwargs["gumbel_noise_fn"])
    assert kwargs["gumbel_noise_fn"](0)(0) is None


def test_main_logs_failure_marker_and_reraises(monkeypatch, expect_error):
    logged = {}

    def fake_run(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(text_demo, "_run", fake_run)
    monkeypatch.setattr(text_demo.logger, "error", lambda msg: logged.setdefault("error", msg))

    with expect_error(RuntimeError, match="kaboom"):
        text_demo.main(["--build-only", "--mesh", "P150x4"])

    assert logged["error"] == ("DG_TEXT_DEMO_FAILURE mode=build-only mesh=P150x4 error_type=RuntimeError")


# --- device RUN smokes ---------------------------------------------------------------


@requires_device
def test_short_prompt_two_block_run_exits_clean(monkeypatch):
    """R-b: short prompt -> 2 committed 256-token blocks -> decoded text, no crash."""
    argv = _base_argv(_require_local_checkpoint()) + [
        "--max-seq-len",
        "1024",
        "--canvas-length",
        "256",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "512",
        "--num-blocks",
        "2",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_NUM_LAYERS")
    if num_layers is not None:
        argv += ["--num-layers", num_layers]

    summary = _demo_success_summary(monkeypatch, argv)
    assert summary["generated_tokens"] == 512
    assert summary["blocks"] == 2
    assert summary["prompt_len"] == 32
    assert summary["next_pos"] == 544


@requires_device
def test_short_prompt_256k_context_allocation_exits_clean(monkeypatch):
    """Short prompt with a 256K context allocation, without a huge input prompt."""
    argv = _base_argv(_require_local_checkpoint()) + [
        "--max-seq-len",
        "262144",
        "--canvas-length",
        "32",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "32",
        "--num-blocks",
        "1",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_256K_NUM_LAYERS", "1")
    if num_layers.lower() != "full":
        argv += ["--num-layers", num_layers]
    else:
        argv += ["--argmax-sampling"]

    summary = _demo_success_summary(monkeypatch, argv)
    assert summary["generated_tokens"] == 32
    assert summary["blocks"] == 1
    assert summary["prompt_len"] == 32
    assert summary["next_pos"] == 64


@requires_device
def test_long_prompt_two_block_maskless_run_exits_clean(monkeypatch):
    """R-a: prompt long enough to force maskless denoise -> 2 blocks, no crash."""
    argv = _base_argv(_require_local_checkpoint()) + [
        "--prompt",
        "hello " * 1000,
        "--max-seq-len",
        "1536",
        "--canvas-length",
        "32",
        "--max-denoising-steps",
        "1",
        "--max-new-tokens",
        "64",
        "--num-blocks",
        "2",
        "--seed",
        "0",
        "--disable-eos-stop",
    ]
    num_layers = os.environ.get("DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS", "1")
    if num_layers.lower() != "full":
        argv += ["--num-layers", num_layers]

    summary = _demo_success_summary(monkeypatch, argv)
    assert summary["generated_tokens"] == 64
    assert summary["blocks"] == 2
    assert summary["prompt_len"] == 1024
    assert summary["next_pos"] == 1088
