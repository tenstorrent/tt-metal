# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the up-front-only denoise trace architecture and its capture lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import serving
from models.experimental.diffusion_gemma.tt import traced_denoise as TD
from models.experimental.diffusion_gemma.tt.denoise_forward import DenoiseLogitsAdapter, MutablePrefixKVReader

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


# --- ttnn fakes for the CPU controller contracts -----------------------------


class _FakeTensor:
    def __init__(self, name, *, deallocate_error=None):
        self.name = name
        self.deallocated = False
        self.deallocate_attempted = False
        self.deallocate_error = deallocate_error

    def deallocate(self, force):
        assert force is True
        assert not self.deallocated, self.name
        self.deallocate_attempted = True
        if self.deallocate_error is not None:
            raise self.deallocate_error
        self.deallocated = True


class _FakeTtnn:
    TILE_SIZE = 32
    copies = []
    executions = []
    syncs = 0
    trace_events = []
    end_error = None
    release_errors = set()

    @classmethod
    def reset(cls):
        cls.copies = []
        cls.executions = []
        cls.syncs = 0
        cls.trace_events = []
        cls.end_error = None
        cls.release_errors = set()

    @staticmethod
    def clone(tensor):
        return _FakeTensor(f"clone({tensor.name})")

    @classmethod
    def copy(cls, source, destination):
        cls.copies.append((source.name, destination.name))

    @classmethod
    def execute_trace(cls, mesh, trace_id, blocking=False):
        assert mesh == "mesh"
        assert blocking is False
        cls.executions.append(trace_id)

    @classmethod
    def synchronize_device(cls, mesh):
        assert mesh == "mesh"
        cls.syncs += 1

    @classmethod
    def begin_trace_capture(cls, mesh, cq_id=0):
        cls.trace_events.append(("begin", mesh, cq_id))
        return "trace-id"

    @classmethod
    def end_trace_capture(cls, mesh, trace_id, cq_id=0):
        cls.trace_events.append(("end", mesh, trace_id, cq_id))
        if cls.end_error is not None:
            raise cls.end_error

    @classmethod
    def release_trace(cls, mesh, trace_id):
        cls.trace_events.append(("release", mesh, trace_id))
        if trace_id in cls.release_errors:
            raise RuntimeError(f"injected release failure {trace_id}")


@pytest.fixture
def fake_ttnn(monkeypatch):
    """Swap ttnn out under traced_denoise only for the tests that assert on its calls.

    Requested explicitly rather than autouse: the device tests at the bottom of this module
    drive the real controller, and a module-wide patch would hand them the fake instead.
    """
    _FakeTtnn.reset()
    monkeypatch.setattr(TD, "ttnn", _FakeTtnn)
    return _FakeTtnn


def _config(**overrides):
    values = {
        "canvas_length": 32,
        "max_denoise_steps": TD.UPFRONT_DENOISE_STEPS,
        "stable_steps_to_halt": 1,
    }
    values.update(overrides)
    return DiffusionConfig(**values)


def _controller():
    return TD.UpfrontTracedDenoiseController("mesh", _config())


# --- up-front capture flag ---------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True), ("true", True), ("off", False)])
def test_upfront_capture_flag_parses_truthy(monkeypatch, value, expected):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", value)
    assert TD.upfront_capture_enabled() is expected


def test_upfront_capture_flag_defaults_on(monkeypatch):
    """Up-front capture is the shipped serving path; DG_UPFRONT_CAPTURE=0 is the opt-out."""
    monkeypatch.delenv("DG_UPFRONT_CAPTURE", raising=False)
    assert TD.upfront_capture_enabled() is True


# --- controller lifecycle ----------------------------------------------------


def test_controller_accepts_only_released_48_step_schedule(fake_ttnn, expect_error):
    controller = _controller()
    assert controller.config.max_denoise_steps == 48

    with expect_error(ValueError, match="released 48-step schedule"):
        TD.UpfrontTracedDenoiseController("mesh", _config(max_denoise_steps=47))
    with expect_error(ValueError, match="stable_steps_to_halt=1"):
        TD.UpfrontTracedDenoiseController("mesh", _config(stable_steps_to_halt=2))


def test_controller_release_is_best_effort_idempotent_and_clears_state(fake_ttnn):
    controller = _controller()
    _FakeTtnn.release_errors = {"trace-0"}
    bad = _FakeTensor("bad-buffer", deallocate_error=RuntimeError("injected buffer failure"))
    good = _FakeTensor("good-buffer")
    controller.traces = ["trace-0", "trace-1"]
    controller.canvas_buf = bad
    controller.committed_buf = good
    controller.gumbel_buf = _FakeTensor("gumbel")
    controller.noise_buf = _FakeTensor("noise")
    controller.captured = True

    controller.release()
    controller.release()

    assert ("release", "mesh", "trace-0") in _FakeTtnn.trace_events
    assert ("release", "mesh", "trace-1") in _FakeTtnn.trace_events
    assert bad.deallocate_attempted
    assert good.deallocated
    assert controller.traces == []
    assert controller.canvas_buf is None
    assert controller.committed_buf is None
    assert controller.gumbel_buf is None
    assert controller.noise_buf is None
    assert controller.captured is False
    assert controller.released is True


def test_upfront_block_reuses_the_single_controller_attribute(fake_ttnn, monkeypatch):
    instances = []

    class _Controller:
        def __init__(self, mesh, config):
            self.calls = []
            instances.append(self)

        def denoise_block(self, logits_fn, init_canvas, config, **kwargs):
            self.calls.append((logits_fn, init_canvas, config, kwargs))
            return len(self.calls)

    monkeypatch.setattr(TD, "UpfrontTracedDenoiseController", _Controller)
    logits_fn = SimpleNamespace(
        tt_model=SimpleNamespace(mesh_device="mesh"),
        _upfront_capture_phase=True,
    )
    config = _config()

    assert (
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas-0",
            config,
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )
        == 1
    )
    assert (
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas-1",
            config,
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )
        == 2
    )

    assert len(instances) == 1
    assert logits_fn._upfront_traced_denoise_controller is instances[0]


def test_upfront_block_rejects_on_demand_capture_outside_startup(fake_ttnn, expect_error):
    logits_fn = SimpleNamespace(tt_model=SimpleNamespace(mesh_device="mesh"))
    with expect_error(RuntimeError, match="startup trace warmup"):
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas",
            _config(),
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )


# --- trace capture guard -----------------------------------------------------


def test_trace_capture_guard_ends_and_releases_aborted_trace(fake_ttnn, expect_error):
    with expect_error(RuntimeError, match="injected capture failure"):
        with TD._trace_capture_guard("mesh", cq_id=0):
            raise RuntimeError("injected capture failure")

    assert _FakeTtnn.trace_events == [
        ("begin", "mesh", 0),
        ("end", "mesh", "trace-id", 0),
        ("release", "mesh", "trace-id"),
    ]


def test_trace_capture_guard_releases_when_finalization_fails(fake_ttnn, expect_error):
    _FakeTtnn.end_error = RuntimeError("injected end failure")

    with expect_error(RuntimeError, match="injected end failure"):
        with TD._trace_capture_guard("mesh", cq_id=0):
            pass

    assert _FakeTtnn.trace_events[-1] == ("release", "mesh", "trace-id")


# --- materialized noise buffers ----------------------------------------------


def test_materialized_gumbel_uses_stable_buffer_and_consumes_sources(fake_ttnn):
    controller = _controller()
    first = _FakeTensor("gumbel-0")
    controller._initialize_gumbel(lambda step: first)

    assert controller.gumbel_buf.name == "clone(gumbel-0)"
    assert first.deallocated

    fresh = _FakeTensor("gumbel-7")
    assert controller._refresh_gumbel(lambda step: fresh, 7) is controller.gumbel_buf
    assert _FakeTtnn.copies == [("gumbel-7", "clone(gumbel-0)")]
    assert fresh.deallocated


@pytest.mark.parametrize("value", [None, object()])
def test_materialized_gumbel_rejects_non_tensor_descriptors(fake_ttnn, value, expect_error):
    controller = _controller()
    with expect_error(ValueError, match="requires materialized host noise"):
        controller._initialize_gumbel(lambda step: value)


def test_materialized_renoise_uses_one_stable_buffer_and_consumes_only_requested_steps(fake_ttnn):
    controller = _controller()
    first = _FakeTensor("noise-0")
    controller._initialize_noise(lambda step: first)
    assert controller.noise_buf.name == "clone(noise-0)"
    assert first.deallocated

    fresh = _FakeTensor("noise-7")
    assert controller._refresh_noise(lambda step: fresh, 7) is controller.noise_buf
    assert _FakeTtnn.copies == [("noise-7", "clone(noise-0)")]
    assert fresh.deallocated


# --- trace replay ------------------------------------------------------------


def test_replay_reuses_reveal_buffers_and_stops_on_materialized_halt(fake_ttnn, monkeypatch):
    controller = _controller()
    controller.captured = True
    controller.reveal_pmax = 1024
    controller._last_prompt_len = 32
    controller.traces = [f"trace-{step}" for step in range(48)]
    controller.canvas_buf = _FakeTensor("canvas-buf")
    controller.committed_buf = _FakeTensor("committed-buf")
    controller.gumbel_buf = _FakeTensor("gumbel-buf")
    controller.noise_buf = _FakeTensor("noise-buf")
    controller.halt_bufs = SimpleNamespace()

    events = []
    monkeypatch.setattr(
        controller,
        "_refresh_noise",
        lambda fn, step: events.append(("noise", step)),
    )
    monkeypatch.setattr(
        controller,
        "_refresh_gumbel",
        lambda fn, step: events.append(("gumbel", step)),
    )
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[19]], dtype=torch.long))
    halt_values = iter([(1.0, 1.0), (0.1, 0.0)])
    monkeypatch.setattr(TD, "read_halt_scalars", lambda buffers: next(halt_values))
    monkeypatch.setattr(
        TD,
        "eval_halt",
        lambda mean, mismatch, step, **kwargs: step == 1,
    )

    adapter = SimpleNamespace(
        prompt_len=64,
        q_rope_offset=64,
        update_canvas_rope_buffers=lambda start: events.append(("rope", start)),
        update_reveal_mask_buffer=lambda prompt: events.append(("reveal", prompt)),
        reset_signal_buffer=lambda: events.append("signal-reset"),
    )
    init_canvas = _FakeTensor("init-canvas")

    trajectory = controller.denoise_block(
        adapter,
        init_canvas,
        controller.config,
        gumbel_noise_fn=lambda step: _FakeTensor(f"unused-gumbel-{step}"),
        noise_tokens_fn=lambda step: _FakeTensor(f"unused-noise-{step}"),
    )

    assert trajectory.num_steps == 2
    assert trajectory.halted is True
    assert torch.equal(trajectory.committed, torch.tensor([[19]]))
    assert _FakeTtnn.executions == ["trace-0", "trace-1"]
    assert controller.capture_events == 0
    assert controller.traces_captured == 0
    assert controller.adapter_rebinds == 1
    assert ("reveal", 64) in events
    assert [event for event in events if isinstance(event, tuple) and event[0] == "gumbel"] == [
        ("gumbel", 0),
        ("gumbel", 1),
    ]
    assert [event for event in events if isinstance(event, tuple) and event[0] == "noise"] == [
        ("noise", 0),
        ("noise", 1),
    ]
    assert init_canvas.deallocated


# --- fixed reveal span -------------------------------------------------------


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


def test_traced_denoise_reveal_pmax_default_registration(monkeypatch, expect_error):
    """The registered derived span satisfies the controller without DG_DENOISE_REVEAL_PMAX."""
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


# --- adapter rebind ----------------------------------------------------------


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


# --- serving session persistent adapter --------------------------------------


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


# --- vLLM startup contract ---------------------------------------------------


def _set_valid_upfront_env(monkeypatch):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "1073741824")


def test_vllm_upfront_configuration_accepts_only_released_contract(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    # `device` (the on-device permuted-vocab RNG) is the ONLY materialized Gumbel source, so it
    # is the only mode the up-front controller can accept.
    assert (
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=TD.UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
        )
        == 1024
    )

    with expect_error(RuntimeError, match="max_denoise_steps=48"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=47,
            gumbel_mode="device",
        )
    # Everything else is rejected loudly: "chunked" is a descriptor, "argmax" is None, and "host"
    # (the per-step full-vocab torch Gumbel serving mode) was deleted 2026-07-28 -- it repaired 0
    # of the drifting prompts at 1.40x the cost. Passing it must now fail, not silently serve.
    for rejected_mode in ("argmax", "chunked", "host"):
        with expect_error(RuntimeError, match=r"GUMBEL_MODE='device'"):
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
            gumbel_mode="device",
        )

    _set_valid_upfront_env(monkeypatch)
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX")
    with expect_error(RuntimeError, match="explicit bounded DG_DENOISE_REVEAL_PMAX"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=48,
            gumbel_mode="device",
        )

    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1000")
    with expect_error(RuntimeError, match="positive 32-token multiple"):
        generator_vllm._validate_upfront_capture_configuration(
            canvas_length=256,
            max_denoise_steps=48,
            gumbel_mode="device",
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
            max_denoise_steps=TD.UPFRONT_DENOISE_STEPS,
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
            max_denoise_steps=TD.UPFRONT_DENOISE_STEPS,
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
            max_denoise_steps=TD.UPFRONT_DENOISE_STEPS,
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
            max_denoise_steps=TD.UPFRONT_DENOISE_STEPS,
            gumbel_mode="device",
            max_model_len=128,
        )


# --- vLLM warmup phases ------------------------------------------------------


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
    wrapper._gumbel_mode = "device"
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


# --- vLLM unwarmed-length rejection ------------------------------------------


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


def test_vllm_upfront_prefill_rejects_the_request_not_the_engine(monkeypatch):
    """An unwarmed prefill length must cost ONE request, not the server.

    In vLLM V1 an exception out of ``execute_model`` is fatal to EngineCore, so the old ``raise``
    here meant a single out-of-band request emptied every request queued behind it while the eval
    still wrote a normal-looking score. Regression test for that: the call returns a stop-id block,
    keeps the row registered as finished, and emits the ``prefill_rejected`` metric.

    The metric assertion and the ``reset_calls == 0`` assertion were folded in here when
    ``DG_UPFRONT_STRICT_PREFILL_LENS`` was deleted 2026-08-03. There is no engine-fatal arm left, so
    ``prefill_rejected`` is the only machine-readable evidence that a sample was lost, and it needs a
    test that pins its exact payload -- the scorers parse it.
    """
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    sessions = []
    wrapper = _reject_wrapper(generator_vllm, sessions)
    metrics = []
    monkeypatch.setattr(generator_vllm, "_metric", lambda event, **fields: metrics.append((event, fields)))

    out = wrapper.prefill_forward(torch.zeros((1, 33), dtype=torch.long))

    assert out.shape == (1, 4), "a rejected row must still fill its slot in the emitted block"
    assert torch.equal(out, torch.full((1, 4), 7, dtype=torch.long)), "expected the session's stop id"
    # The row must stay REGISTERED and finished: decode_forward raises when _sessions is empty, and
    # that raise is engine-fatal too, so dropping the row would just move the crash one step later.
    assert wrapper._sessions == {0: sessions[0]}, "the rejected row must remain registered"
    assert sessions[0].finished is True, "so decode_forward takes its stop-id branch"
    assert sessions[0].reset_calls == 0, "the rejected row keeps its session until release_request"
    # Pins that a 33-token prompt rounds UP to aligned 64 and is compared against the warmed set.
    assert ("prefill_rejected", {"row": 0, "cache_len": 64, "warmed": [32]}) in metrics


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


# --- vLLM teardown -----------------------------------------------------------


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


# --- device: startup capture replayed across requests ------------------------


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
            "DG_VLLM_GUMBEL_MODE": "device",
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
        gumbel_mode="device",
    )
    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_prefill(None, True, True)
    return wrapper


def _persistent_controller(wrapper):
    controller = wrapper._persistent_adapter._upfront_traced_denoise_controller
    stats = controller.stats()
    assert stats["capture_events"] == 1
    assert stats["traces_captured"] == TD.UPFRONT_DENOISE_STEPS
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
        gumbel_mode="device",
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
