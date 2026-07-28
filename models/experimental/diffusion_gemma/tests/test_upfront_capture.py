# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and device-gated coverage for model-startup denoise capture."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import serving
from models.experimental.diffusion_gemma.tt.denoise_forward import DenoiseLogitsAdapter, MutablePrefixKVReader
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    UPFRONT_DENOISE_STEPS,
    upfront_capture_enabled,
)

DEVICE_GATED = os.environ.get("DG_RUN_DEVICE", "0") == "1"
_DG_CKPT_INPUT = Path(
    os.path.expanduser(
        os.environ.get(
            "DG_CKPT",
            "~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it",
        )
    )
)
_DG_CKPT_REF = _DG_CKPT_INPUT / "refs" / "main"


def _checkpoint_has_weights(path: Path) -> bool:
    single = path / "model.safetensors"
    if single.is_file():
        return True
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return False
    filenames = set(json.loads(index.read_text())["weight_map"].values())
    return bool(filenames) and all((path / filename).is_file() for filename in filenames)


def _resolve_test_checkpoint(path: Path) -> Path:
    candidates = [path]
    if _DG_CKPT_REF.is_file():
        candidates.append(path / "snapshots" / _DG_CKPT_REF.read_text().strip())
    candidates.extend(sorted((path / "snapshots").glob("*")) if (path / "snapshots").is_dir() else [])
    return next((candidate for candidate in candidates if _checkpoint_has_weights(candidate)), path)


DG_CKPT = str(_resolve_test_checkpoint(_DG_CKPT_INPUT))


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True), ("true", True), ("off", False)])
def test_upfront_capture_flag_parses_truthy(monkeypatch, value, expected):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", value)
    assert upfront_capture_enabled() is expected


def test_upfront_capture_flag_defaults_on(monkeypatch):
    """Up-front capture is the shipped serving path; DG_UPFRONT_CAPTURE=0 is the opt-out."""
    monkeypatch.delenv("DG_UPFRONT_CAPTURE", raising=False)
    assert upfront_capture_enabled() is True


def test_mutable_prefix_reader_request_reset_can_shrink_only_with_fixed_span(expect_error):
    reader = object.__new__(MutablePrefixKVReader)
    reader.prompt_len = 288
    reader.read_span = 1024

    reader.reset_prompt_len(32)
    assert reader.prompt_len == 32
    reader.reset_prompt_len(992)
    assert reader.prompt_len == 992

    with expect_error(ValueError, match="exceeds reveal read span"):
        reader.reset_prompt_len(1056)
    reader.read_span = None
    with expect_error(RuntimeError, match="fixed reveal-mask read span"):
        reader.reset_prompt_len(32)


def test_adapter_rebind_refreshes_reader_mask_and_rope_in_place():
    events = []
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prompt_hidden_by_layer = SimpleNamespace(reset_prompt_len=lambda n: events.append(("reader", n)))
    adapter.prompt_len = 288
    adapter.q_rope_offset = 288
    adapter.use_reveal_mask = True
    adapter._reveal_p_max = 1024
    adapter._canvas_rope_len = 256
    adapter.update_reveal_mask_buffer = lambda n: events.append(("mask", n))
    adapter.update_canvas_rope_buffers = lambda n: events.append(("rope", n))

    adapter.rebind_prompt(64)

    assert adapter.prompt_len == 64
    assert adapter.q_rope_offset == 64
    assert events == [("reader", 64), ("mask", 64), ("rope", 64)]


def test_adapter_rebind_rejects_prompt_without_one_canvas_of_pmax_capacity(expect_error):
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prompt_hidden_by_layer = SimpleNamespace(
        reset_prompt_len=lambda n: pytest.fail("capacity must be checked before reader mutation")
    )
    adapter.use_reveal_mask = True
    adapter._reveal_p_max = 1024
    adapter._canvas_rope_len = 256

    with expect_error(ValueError, match=r"800 \+ 256 = 1056 > 1024"):
        adapter.rebind_prompt(800)


def test_adapter_rebind_rejects_adapter_without_persistent_reveal(expect_error):
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.use_reveal_mask = False
    with expect_error(RuntimeError, match="reveal"):
        adapter.rebind_prompt(32)


def test_session_reset_detaches_borrowed_persistent_adapter_without_releasing_it():
    events = []
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=SimpleNamespace(release=lambda: events.append("trace_release")),
        reset=lambda: events.append("adapter_reset"),
    )
    session = object.__new__(serving.BlockDiffusionServingSession)
    session._logits_fn = None
    session._persistent_adapter = None
    session.next_pos = None
    session.finished = False
    session.block_idx = 0

    session.attach_persistent_adapter(adapter)
    session._logits_fn = adapter
    session.next_pos = 288
    session.reset()

    assert events == []
    assert hasattr(adapter, "_upfront_traced_denoise_controller")
    assert session._logits_fn is None
    assert session._persistent_adapter is None


def test_session_prefill_rebinds_injected_adapter_instead_of_building(monkeypatch):
    rebound = []
    # rebind_prompt now also takes the request's TRUE prompt length, so the reveal mask can hide
    # that request's prefill pad slots instead of carrying the previous request's span.
    adapter = SimpleNamespace(rebind_prompt=lambda n, *, true_prompt_len=None: rebound.append(n))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session.tt_model = SimpleNamespace()
    session.page_table = None
    session.page_tables_per_layer = None
    session.prefill_reused = False
    session.prefill_time_s = 0.0
    session._persistent_adapter = adapter
    session._logits_fn = None
    session._logits_fn_builder = lambda *args, **kwargs: pytest.fail("persistent prefill must not rebuild adapter")
    session.prompt_len = None
    session.cache_len = None
    session.next_pos = None
    session.block_idx = 0
    session.finished = False
    monkeypatch.setattr(
        serving,
        "prefill_prompt_tokens",
        lambda *args, **kwargs: SimpleNamespace(prompt_len=3, cache_len=32),
    )

    assert session.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long)) == 32
    assert rebound == [32]
    assert session._logits_fn is adapter


def _set_valid_upfront_env(monkeypatch):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "1073741824")


def test_vllm_upfront_configuration_accepts_only_released_contract(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="host",
        )
        == 1024
    )

    with expect_error(RuntimeError, match="max_denoise_steps=48"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=47,
            gumbel_mode="host",
        )
    # The on-device permuted-vocab source is now an accepted up-front Gumbel mode.
    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
        )
        == 1024
    )
    # Non-materialized sources (chunked descriptor / argmax None) remain rejected loudly.
    for rejected_mode in ("argmax", "chunked"):
        with expect_error(RuntimeError, match=r"GUMBEL_MODE in \{host, device\}"):
            generator_vllm._validate_upfront_capture_configuration(
                canvas_length=256,
                max_denoise_steps=48,
                gumbel_mode=rejected_mode,
            )

    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "0")
    with expect_error(RuntimeError, match="DG_TRACE_REGION_SIZE > 0"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=48,
            gumbel_mode="host",
        )

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=48,
            gumbel_mode="host",
        )

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1000")
    with expect_error(RuntimeError, match="positive 32-token multiple"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=48,
            gumbel_mode="host",
        )


def test_vllm_upfront_pmax_derives_from_max_model_len(monkeypatch, expect_error):
    """With DG_DENOISE_REVEAL_PMAX unset the span comes from max_model_len, tile-rounded DOWN."""
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")

    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
            max_model_len=4096,
        )
        == 4096
    )
    # Non-tile-aligned served bounds round DOWN. The KV cache is allocated with seq dim ==
    # max_model_len verbatim, so rounding up (4090 -> 4096) would exceed the allocated span
    # and abort startup; 4064 is the largest tile multiple that still fits.
    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
            max_model_len=4090,
        )
        == 4064
    )
    # An explicit env value still wins over the derived one.
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
            max_model_len=4096,
        )
        == 1024
    )
    # A served bound too small for one prompt tile plus a canvas is still rejected loudly.
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")
    with expect_error(RuntimeError, match="cannot fit the startup prompt and one canvas"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
            max_model_len=128,
        )


def test_traced_denoise_reveal_pmax_default_registration(monkeypatch, expect_error):
    """The registered derived span satisfies the controller without DG_DENOISE_REVEAL_PMAX."""
    from models.experimental.diffusion_gemma.tt import traced_denoise as TD

    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX", raising=False)
    monkeypatch.setattr(TD, "_DEFAULT_REVEAL_PMAX", None, raising=False)
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        TD._resolve_reveal_pmax(object())

    TD.set_default_reveal_pmax(4096)
    try:
        assert TD._resolve_reveal_pmax(object()) == 4096
        # An explicit env value still wins.
        monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
        assert TD._resolve_reveal_pmax(object()) == 1024
        # Registered garbage is rejected by the same validation as the env value.
        monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")
        TD.set_default_reveal_pmax(1000)
        with expect_error(RuntimeError, match="positive 32-token multiple"):
            TD._resolve_reveal_pmax(object())
    finally:
        TD.set_default_reveal_pmax(None)


def test_vllm_warmup_captures_48_traces_and_detaches_persistent_adapter(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    controller = SimpleNamespace(
        captured=True,
        stats=lambda: {"capture_events": 1, "traces_captured": 48},
        release=lambda: None,
    )
    adapter = SimpleNamespace(
        use_reveal_mask=True,
        _upfront_traced_denoise_controller=controller,
        reset=lambda: None,
    )
    resets = []

    class _Session:
        def __init__(self):
            self._logits_fn = adapter

        def prefill(self, tokens):
            assert tokens.shape == (1, 1)
            return 32

        def decode_block(self):
            return SimpleNamespace(tokens=torch.zeros((1, 256), dtype=torch.long), next_pos=288)

        def trace_stats(self):
            return [controller.stats()]

        def reset(self):
            resets.append(self._logits_fn)

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = [
        SimpleNamespace(
            mesh_device=None,
            tt_kv_cache=[(SimpleNamespace(shape=(1, 1, 1024, 1)), None)],
        )
    ]
    wrapper.canvas_length = 256
    wrapper._tokenizer = SimpleNamespace(bos_token_id=2)
    wrapper._config = DiffusionConfig()
    wrapper._gumbel_mode = "host"
    wrapper._upfront = True
    wrapper._persistent_adapter = None
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._upfront_pmax = 1024
    wrapper._max_model_len = 1024
    wrapper._make_session = _Session
    monkeypatch.setattr(generator_vllm, "_dram_snapshot", lambda *args, **kwargs: {})
    metrics = []
    monkeypatch.setattr(generator_vllm, "_metric", lambda event, **fields: metrics.append((event, fields)))

    wrapper.warmup_model_prefill(None, True, True)

    assert wrapper._persistent_adapter is adapter
    assert resets == [None]
    assert metrics[0][0] == "upfront_capture"
    assert metrics[0][1]["trace_stats"] == [{"capture_events": 1, "traces_captured": 48}]


def test_vllm_upfront_warmup_defers_capture_until_trace_phase():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._upfront = True
    wrapper._persistent_adapter = None
    wrapper._make_session = lambda: pytest.fail("compile-only warmup must not build a capture session")

    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_decode()

    assert wrapper._upfront_compile_phase_seen is True
    assert wrapper._persistent_adapter is None


def test_vllm_upfront_compile_phase_warms_configured_prefill_lengths(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = [SimpleNamespace(mesh_device=None)]
    wrapper._upfront = True
    wrapper._upfront_pmax = 1024
    wrapper._persistent_adapter = None
    wrapper.canvas_length = 256
    monkeypatch.setenv("DG_UPFRONT_PREFILL_WARMUP_LENS", "192,160,192")
    warmed = []
    monkeypatch.setattr(
        generator_vllm,
        "prefill_prompt_tokens",
        lambda model, tokens: warmed.append(tuple(tokens.shape)),
    )
    monkeypatch.setattr(generator_vllm.ttnn, "synchronize_device", lambda mesh: None)

    wrapper.warmup_model_prefill(None, False, True)

    assert wrapper._upfront_compile_phase_seen is True
    # Duplicates collapse, and one tile is always present on top of whatever was configured -- the
    # capture phase compiles a 32-aligned prefill anyway (see
    # test_vllm_upfront_warmup_always_includes_one_tile).
    assert wrapper._upfront_prefill_warmup_lens == frozenset({32, 160, 192})
    assert warmed == [(1, 32), (1, 160), (1, 192)]


def test_vllm_upfront_trace_phase_rejects_missing_prefill_warmups(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._upfront = True
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset()

    with expect_error(RuntimeError, match="requires a compile-only warmup"):
        wrapper.warmup_model_prefill(None, True, True)


class _RejectSession:
    """Minimal session for the unwarmed-length rejection path."""

    stop_token_ids = [7]

    def __init__(self):
        self.reset_calls = 0
        self.finished = False

    def attach_persistent_adapter(self, adapter):
        assert adapter == "persistent"

    def reset(self):
        self.reset_calls += 1


def _reject_wrapper(generator_vllm, sessions_out):
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._sessions = {}
    wrapper._upfront = True
    wrapper._persistent_adapter = "persistent"
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper.canvas_length = 4

    def _make():
        session = _RejectSession()
        sessions_out.append(session)
        return session

    wrapper._make_session = _make
    return wrapper


def test_vllm_upfront_prefill_rejects_the_request_not_the_engine():
    """An unwarmed prefill length must cost ONE request, not the server.

    In vLLM V1 an exception out of ``execute_model`` is fatal to EngineCore, so the old ``raise``
    here meant a single out-of-band request emptied every request queued behind it while the eval
    still wrote a normal-looking score. Regression test for that: the call returns a stop-id block,
    frees the partially built session, and leaves no row registered.
    """
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    sessions = []
    wrapper = _reject_wrapper(generator_vllm, sessions)

    out = wrapper.prefill_forward(torch.zeros((1, 33), dtype=torch.long))

    assert out.shape == (1, 4), "a rejected row must still fill its slot in the emitted block"
    assert torch.equal(out, torch.full((1, 4), 7, dtype=torch.long)), "expected the session's stop id"
    # The row must stay REGISTERED and finished: decode_forward raises when _sessions is empty, and
    # that raise is engine-fatal too, so dropping the row would just move the crash one step later.
    assert wrapper._sessions == {0: sessions[0]}, "the rejected row must remain registered"
    assert sessions[0].finished is True, "so decode_forward takes its stop-id branch"


def test_vllm_decode_after_a_rejected_prefill_does_not_kill_the_engine():
    """The step AFTER a rejection must survive too.

    ``decode_forward`` raises when ``_sessions`` is empty, and that raise reaches EngineCore exactly
    like the prefill one did -- so rejecting by dropping the row would only move the crash. If vLLM
    asks for another block for the rejected request, it gets stop-id padding.
    """
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    sessions = []
    wrapper = _reject_wrapper(generator_vllm, sessions)
    wrapper.prefill_forward(torch.zeros((1, 33), dtype=torch.long))

    out = wrapper.decode_forward()

    assert out.shape == (1, 4)
    assert torch.equal(out, torch.full((1, 4), 7, dtype=torch.long))


def test_vllm_upfront_prefill_strict_mode_still_raises(expect_error, monkeypatch):
    """The engine-fatal behaviour stays reachable for bit-exactness gates, but only on request."""
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_UPFRONT_STRICT_PREFILL_LENS", "1")
    sessions = []
    wrapper = _reject_wrapper(generator_vllm, sessions)

    with expect_error(RuntimeError, match="unseen aligned prefill length 64"):
        wrapper.prefill_forward(torch.zeros((1, 33), dtype=torch.long))
    assert sessions[0].reset_calls == 1, "even the fatal path must free its session"


def test_vllm_upfront_warmup_always_includes_one_tile(monkeypatch):
    """32 is warmed whether or not it was listed.

    The capture phase prefills a single BOS token, so the 32-aligned program is compiled on every
    startup anyway. Omitting it from the whitelist is what let a 21-token request reach the
    rejection path at all.
    """
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    compiled = []
    monkeypatch.setattr(generator_vllm, "prefill_prompt_tokens", lambda model, toks: compiled.append(toks.shape[1]))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *_a, **_k: None)
    monkeypatch.setenv("DG_UPFRONT_PREFILL_WARMUP_LENS", "128,160")

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper._upfront = True
    wrapper.canvas_length = 256
    wrapper._upfront_pmax = 4096
    wrapper.model = [SimpleNamespace(mesh_device=object())]

    wrapper.warmup_model_prefill(None, enable_trace=False, can_sample_on_device=False)

    assert ttnn.TILE_SIZE in wrapper._upfront_prefill_warmup_lens
    assert wrapper._upfront_prefill_warmup_lens == frozenset({32, 128, 160})
    assert sorted(compiled) == [32, 128, 160], "every warmed length must actually be compiled"


def test_vllm_destructor_releases_persistent_controller_then_adapter_exactly_once():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=SimpleNamespace(release=lambda: events.append("trace_release")),
        reset=lambda: events.append("adapter_reset"),
    )
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._sessions = {}
    wrapper._persistent_adapter = adapter

    wrapper.__del__()
    wrapper.__del__()

    assert events == ["trace_release", "adapter_reset"]
    assert wrapper._persistent_adapter is None
    assert not hasattr(adapter, "_upfront_traced_denoise_controller")


@pytest.fixture(scope="module")
def upfront_device_bundle():
    if not DEVICE_GATED:
        pytest.skip("up-front capture device tests require DG_RUN_DEVICE=1")
    if not _checkpoint_has_weights(Path(DG_CKPT)):
        pytest.skip(f"complete checkpoint weights not available at {DG_CKPT}")
    raw_trace_region = os.environ.get("DG_TRACE_REGION_SIZE", "").strip()
    if not raw_trace_region or int(raw_trace_region) <= 0:
        pytest.skip("up-front capture device tests require an explicit DG_TRACE_REGION_SIZE > 0")
    pytest.importorskip("vllm")

    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device

    p_max = int(os.environ.get("DG_DENOISE_REVEAL_PMAX", "1024"))
    old_env = {
        name: os.environ.get(name)
        for name in (
            "DG_UPFRONT_CAPTURE",
            "DG_DENOISE_REVEAL_PMAX",
            "DG_TRACE_REGION_SIZE",
            "DG_VLLM_GUMBEL_MODE",
            "DG_UPFRONT_PREFILL_WARMUP_LENS",
        )
    }
    os.environ.update(
        {
            "DG_UPFRONT_CAPTURE": "1",
            "DG_DENOISE_REVEAL_PMAX": str(p_max),
            "DG_VLLM_GUMBEL_MODE": "host",
        }
    )

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    try:
        model_kwargs = {"max_seq_len": p_max, "create_kv_cache": True}
        num_layers = os.environ.get("DG_UPFRONT_NUM_LAYERS", "1")
        if num_layers.lower() != "full":
            model_kwargs["num_layers"] = int(num_layers)
        yield build_tt_model_from_checkpoint_dir(
            mesh,
            DG_CKPT,
            tokenizer_kwargs={"local_files_only": True},
            **model_kwargs,
        )
    finally:
        _close_mesh_device(mesh)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _tokenize(bundle, text: str) -> torch.Tensor:
    from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

    return tokenize_prompt(bundle.tokenizer, text)


def _aligned_prompt_lengths(prompts) -> list[int]:
    return sorted({((int(prompt.shape[1]) + 31) // 32) * 32 for prompt in prompts} | {32})


def _make_upfront_wrapper(bundle, prompts):
    from models.experimental.diffusion_gemma.tt import generator_vllm

    os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = ",".join(str(value) for value in _aligned_prompt_lengths(prompts))
    wrapper = generator_vllm.DiffusionGemmaForCausalLM(
        [bundle.tt_model],
        [bundle.model_args],
        bundle.tt_model.mesh_device,
        dg_state_dict=bundle.state_dict,
        tokenizer=bundle.tokenizer,
        config=DiffusionConfig(),
        gumbel_mode="host",
    )
    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_prefill(None, True, True)
    return wrapper


def _persistent_controller(wrapper):
    controller = wrapper._persistent_adapter._upfront_traced_denoise_controller
    stats = controller.stats()
    assert stats["capture_events"] == 1
    assert stats["traces_captured"] == UPFRONT_DENOISE_STEPS
    return controller


def _serve_once(wrapper, tokens: torch.Tensor, *, num_blocks: int = 2):
    controller = _persistent_controller(wrapper)
    outputs = []
    steps = []
    halted = []
    for block_idx in range(num_blocks):
        halted_before = controller.halted_blocks
        output = (
            wrapper.prefill_forward(tokens, prompt_lens=[int(tokens.shape[1])])
            if block_idx == 0
            else wrapper.decode_forward()
        )
        outputs.append(output)
        steps.append(len(controller.last_halt_trace))
        halted.append(controller.halted_blocks > halted_before)
    wrapper.release_request(0)
    return torch.cat(outputs, dim=1), steps, halted, controller.stats()


def _serve_eager(bundle, tokens: torch.Tensor, *, num_blocks: int = 2):
    session = serving.BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=DiffusionConfig(),
        tokenizer=bundle.tokenizer,
        gumbel_mode="host",
        seed=0,
        stop_token_ids=[],
    )
    try:
        session.prefill(tokens)
        emissions = [session.decode_block() for _ in range(num_blocks)]
        return (
            torch.cat([emission.tokens.reshape(1, -1) for emission in emissions], dim=1),
            [emission.num_denoise_steps for emission in emissions],
            [emission.halted for emission in emissions],
        )
    finally:
        session.reset()


def test_device_startup_capture_reuses_one_48_trace_set(upfront_device_bundle):
    prompts = [
        _tokenize(upfront_device_bundle, "Write one sentence about rain."),
        _tokenize(upfront_device_bundle, "Explain in detail why rainbows form. " * 8),
    ]
    wrapper = _make_upfront_wrapper(upfront_device_bundle, prompts)
    try:
        initial = _persistent_controller(wrapper).stats()
        first, _, _, after_first = _serve_once(wrapper, prompts[0])
        second, _, _, after_second = _serve_once(wrapper, prompts[1])
        first_again, _, _, after_third = _serve_once(wrapper, prompts[0])

        assert torch.equal(first, first_again)
        assert not torch.equal(first, second)
        for stats in (initial, after_first, after_second, after_third):
            assert stats["capture_events"] == 1
            assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()


def test_device_upfront_matches_eager_tokens_realized_k_and_halt(upfront_device_bundle):
    prompt = _tokenize(upfront_device_bundle, "Name the capital of France.")
    eager_tokens, eager_steps, eager_halted = _serve_eager(upfront_device_bundle, prompt)

    wrapper = _make_upfront_wrapper(upfront_device_bundle, [prompt])
    try:
        upfront_tokens, upfront_steps, upfront_halted, stats = _serve_once(wrapper, prompt)
        assert torch.equal(upfront_tokens, eager_tokens)
        assert upfront_steps == eager_steps
        assert upfront_halted == eager_halted
        assert stats["capture_events"] == 1
        assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()


def test_device_two_sequential_requests_match_eager_without_recapture(upfront_device_bundle):
    prompts = [
        _tokenize(upfront_device_bundle, "Give a friendly greeting."),
        _tokenize(upfront_device_bundle, "Describe a black hole in one sentence. " * 4),
    ]
    eager = [_serve_eager(upfront_device_bundle, prompt) for prompt in prompts]

    wrapper = _make_upfront_wrapper(upfront_device_bundle, prompts)
    try:
        for prompt, (expected_tokens, expected_steps, expected_halted) in zip(prompts, eager):
            tokens, steps, halted, stats = _serve_once(wrapper, prompt)
            assert torch.equal(tokens, expected_tokens)
            assert steps == expected_steps
            assert halted == expected_halted
            assert stats["capture_events"] == 1
            assert stats["traces_captured"] == 48
    finally:
        wrapper.release_persistent_capture()
