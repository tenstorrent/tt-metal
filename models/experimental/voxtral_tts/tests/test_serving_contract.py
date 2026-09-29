# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The serving contract tt-inference-server drives, the same shape as xtts_v2's model class.

`TtVoxtralPipeline()` with no arguments opens its own one-chip mesh device and finds the model
itself; `warmup()` once; `synthesize(text, voice, seed)` -> waveform per request; `close()`. The
host half checks the model-directory resolution order and the class surface; the device half runs it.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_serving_contract.py -m "not slow"   # host
    pytest -svv models/experimental/voxtral_tts/tests/test_serving_contract.py                 # + device
"""

import ast
import inspect
import os

import pytest

torch = pytest.importorskip("torch")

from models.experimental.voxtral_tts.reference import voxtral_paths as paths  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT  # noqa: E402

SAMPLES_PER_FRAME = 1920


@pytest.fixture
def hub(monkeypatch):
    """Record every huggingface_hub.snapshot_download call instead of touching the network."""
    import huggingface_hub

    calls = []

    def fake(repo_id, allow_patterns=None, local_files_only=False, **kw):
        calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns, "local_files_only": local_files_only})
        if local_files_only:
            raise FileNotFoundError("not cached")
        return "/hf/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    monkeypatch.delenv("VOXTRAL_CKPT", raising=False)
    return calls


def test_explicit_path_wins_over_the_environment(hub, monkeypatch):
    monkeypatch.setenv("VOXTRAL_CKPT", "/from/env")
    assert paths.resolve_model_dir("/explicit/dir") == "/explicit/dir"
    assert paths.resolve_model_dir("/explicit/dir/consolidated.safetensors") == "/explicit/dir"
    assert not hub, "an explicit path must not consult the hub"


def test_environment_is_next_and_accepts_the_checkpoint_file(hub, monkeypatch):
    monkeypatch.setenv("VOXTRAL_CKPT", "/from/env/consolidated.safetensors")
    assert paths.resolve_model_dir() == "/from/env"
    assert not hub


def test_download_is_last_and_fetches_every_file_the_model_reads(hub):
    assert paths.resolve_model_dir() == "/hf/cache/snapshot"
    assert [c["local_files_only"] for c in hub] == [True, False], "look in the cache, then download"
    got = hub[-1]
    assert got["repo_id"] == paths.HF_REPO == "mistralai/Voxtral-4B-TTS-2603"
    for need in ("consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*.pt"):
        assert need in got["allow_patterns"], f"the download omits {need}"


def test_the_local_lookup_never_downloads(hub):
    assert paths.local_model_dir() is None
    assert hub and all(c["local_files_only"] for c in hub)


def test_the_download_hint_covers_the_voice_presets():
    assert "voice_embedding" in paths.DOWNLOAD_HINT and "VOXTRAL_CKPT" in paths.DOWNLOAD_HINT


def test_the_class_has_the_serving_shape():
    """Constructor takes an optional mesh device and model path; the request-path methods exist."""
    pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

    sig = inspect.signature(TtVoxtralPipeline.__init__).parameters
    assert sig["mesh_device"].default is None and sig["ckpt_path"].default is None
    for name in ("warmup", "synthesize", "generate", "decode", "close"):
        assert callable(getattr(TtVoxtralPipeline, name)), name
    syn = inspect.signature(TtVoxtralPipeline.synthesize).parameters
    assert list(syn)[1:4] == ["text", "voice", "seed"]


def test_the_pipeline_module_logs_instead_of_printing():
    import models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline as mod

    tree = ast.parse(inspect.getsource(mod))
    prints = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
    ]
    assert not prints, f"print() at lines {prints}; use loguru"


def test_frame_budget_scales_with_text_and_has_a_floor():
    pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import frame_budget

    assert frame_budget("Hi.") == 320
    assert frame_budget("x" * 2000) > frame_budget("x" * 1000) > 320


def _deviceless_pipeline(monkeypatch, prompt_len, max_seq_len=2048):
    """synthesize()'s own logic, with the front end and generate/decode stubbed; records each frame cap."""
    pytest.importorskip("ttnn")
    from types import SimpleNamespace

    from models.experimental.voxtral_tts import frontend
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

    monkeypatch.setattr(frontend, "build_prompt_embeds", lambda *a, **kw: torch.zeros(1, prompt_len, 8))
    tts, caps = TtVoxtralPipeline.__new__(TtVoxtralPipeline), []
    tts.wb, tts.model_dir = None, "/model"
    tts.backbone = SimpleNamespace(max_seq_len=max_seq_len, reset=lambda: None)
    tts.generate = lambda embeds, max_frames, **kw: caps.append(max_frames) or (torch.zeros(1, 37), 0.0, 0.0)
    tts.decode = lambda frames: "wav"
    return tts, caps


def test_synthesize_caps_frames_by_the_text_then_the_cache(monkeypatch):
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import frame_budget

    tts, caps = _deviceless_pipeline(monkeypatch, prompt_len=100)
    assert tts.synthesize("Hi.", "neutral_male") == "wav"
    tts.synthesize("Hi.", "neutral_male", max_frames=50)
    tts.synthesize("x" * 5000, "neutral_male")
    assert caps == [frame_budget("Hi."), 50, 2048 - 100]


def test_a_non_positive_frame_limit_is_refused_before_any_work(monkeypatch, expect_error):
    tts, _ = _deviceless_pipeline(monkeypatch, prompt_len=100)
    del tts.generate
    tts.backbone.prefill_last = lambda *a, **kw: pytest.fail("generate() prefilled before checking max_frames")
    for n in (0, -5):
        with expect_error(ValueError, "max_frames must be positive"):
            tts.synthesize("Hi.", "neutral_male", max_frames=n)
        with expect_error(ValueError, "max_frames must be positive"):
            tts.generate(torch.zeros(1, 100, 8), max_frames=n)


def test_synthesize_refuses_a_prompt_that_fills_the_cache(monkeypatch, expect_error):
    tts, caps = _deviceless_pipeline(monkeypatch, prompt_len=2048)
    with expect_error(ValueError, "leaves no room for audio"):
        tts.synthesize("Hi.", "neutral_male")
    assert not caps, "generate() ran with no room in the cache"


# ------------------------------------------------------------------------------ on device

needs_ckpt = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


@pytest.mark.slow
@needs_ckpt
@pytest.mark.timeout(1800)
def test_owned_device_synthesize_matches_generate_and_decode():
    """No device given: the pipeline opens one, synthesizes, and releases it on close()."""
    ttnn = pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts import frontend
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

    tts = TtVoxtralPipeline()
    try:
        assert tts._owns_device and tts.mesh_device is not None
        tts.warmup()
        text, voice = "Hello from Tenstorrent.", "neutral_male"
        wav = tts.synthesize(text, voice, seed=0)
        n = tts.last_timings["frames"]
        assert wav.shape == (1, 1, n * SAMPLES_PER_FRAME), wav.shape
        assert torch.isfinite(wav).all() and wav.abs().max() > 0.01, "silent or non-finite audio"

        tts.backbone.reset()
        embeds = frontend.build_prompt_embeds(text, voice, tts.wb, model_dir=tts.model_dir)
        frames, _, _ = tts.generate(
            embeds, max_frames=tts.backbone.max_seq_len - embeds.shape[1], seed=0, verbose=False
        )
        ref = tts.decode(frames)
        print(f"\n  synthesize: {n} frames; generate+decode: {frames.shape[0]} frames")
        assert torch.equal(wav, ref), "synthesize differs from generate + decode on the same seed"
    finally:
        tts.close()
    assert tts.mesh_device is None, "close() kept a device the pipeline opened"
    tts.close()  # idempotent
    d = open_device()  # the chip is free again
    ttnn.close_device(d)


@pytest.mark.slow
@needs_ckpt
@pytest.mark.timeout(1800)
def test_close_leaves_a_passed_in_device_open():
    ttnn = pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

    d = open_device()
    try:
        tts = TtVoxtralPipeline(d)
        assert not tts._owns_device and tts.mesh_device is d
        tts.close()
        t = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
        assert float(ttnn.to_torch(t).sum()) == 1024.0, "the caller's device stopped working"
    finally:
        ttnn.close_device(d)
