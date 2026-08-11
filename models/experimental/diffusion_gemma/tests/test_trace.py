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


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True), ("true", True), ("off", False)])
def test_lazy_prefill_recapture_flag_is_explicit_opt_in(monkeypatch, value, expected):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_UPFRONT_LAZY_PREFILL_RECAPTURE", value)
    assert generator_vllm._lazy_prefill_recapture_enabled() is expected


def test_lazy_prefill_recapture_defaults_off(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.delenv("DG_UPFRONT_LAZY_PREFILL_RECAPTURE", raising=False)
    assert generator_vllm._lazy_prefill_recapture_enabled() is False


def test_coarse_prefill_buckets_cover_every_power_of_two_through_256k(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    expected = tuple(1 << exponent for exponent in range(7, 19))
    assert generator_vllm.PREFILL_BUCKETS == expected

    monkeypatch.setenv("DG_UPFRONT_COARSE_PREFILL_BUCKETS", "1")
    for bucket in expected:
        assert generator_vllm._resolve_prefill_execution_len(bucket, max_model_len=262144) == bucket
        if bucket > expected[0]:
            assert generator_vllm._resolve_prefill_execution_len(bucket // 2 + 1, max_model_len=262144) == bucket


def test_chunked_long_prefill_lengths_share_one_warmed_program():
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    assert generator_vllm._prefill_execution_len_is_warmed(131072, {65536})
    assert generator_vllm._prefill_execution_len_is_warmed(262144, {65536})
    assert not generator_vllm._prefill_execution_len_is_warmed(65536, {32768})
    assert not generator_vllm._prefill_execution_len_is_warmed(16384, {4096})


def test_fixed_chunk_prefill_uses_one_warmed_program_for_every_length(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_PREFILL_FIXED_CHUNKS", "1")
    for execution_len in generator_vllm.PREFILL_BUCKETS:
        assert generator_vllm._prefill_execution_len_is_warmed(execution_len, {32})


def test_coarse_prefill_buckets_are_opt_in_and_capacity_bounded(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.delenv("DG_UPFRONT_COARSE_PREFILL_BUCKETS", raising=False)
    assert generator_vllm._coarse_prefill_buckets_enabled() is False
    assert generator_vllm._resolve_prefill_execution_len(129, max_model_len=4096) == 160

    monkeypatch.setenv("DG_UPFRONT_COARSE_PREFILL_BUCKETS", "1")
    assert generator_vllm._resolve_prefill_execution_len(129, max_model_len=4096) == 256
    with expect_error(ValueError, match="no power-of-two prefill bucket"):
        generator_vllm._resolve_prefill_execution_len(4000, max_model_len=4090)


def test_model_owned_hybrid_kv_advertises_full_scheduler_capacity(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    assert (
        generator_vllm.DiffusionGemmaForCausalLM.get_max_tokens_all_users(
            max_model_len=262144,
            max_num_seqs=1,
        )
        == 262144
    )

    with expect_error(ValueError, match="max_num_seqs=1"):
        generator_vllm.DiffusionGemmaForCausalLM.get_max_tokens_all_users(
            max_model_len=262144,
            max_num_seqs=2,
        )


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


def test_traced_denoise_uses_hybrid_logical_capacity_not_physical_block_axis(monkeypatch):
    monkeypatch.delenv("DG_DENOISE_REVEAL_PMAX", raising=False)
    TD.set_default_reveal_pmax(131072)
    try:
        paged_cache = SimpleNamespace(shape=[16, 2, 64, 256])
        tt_model = SimpleNamespace(
            tt_kv_cache=[(paged_cache, paged_cache)],
            _dg_model_owned_hybrid_kv=True,
            _dg_hybrid_max_seq_len=131072,
        )
        adapter = SimpleNamespace(tt_model=tt_model)
        assert TD._resolve_reveal_pmax(adapter) == 131072
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
    prefill_calls = []
    # rebind_prompt now also takes the request's TRUE prompt length, so the reveal mask can hide
    # that request's prefill pad slots instead of carrying the previous request's span.
    adapter = SimpleNamespace(rebind_prompt=lambda n, *, true_prompt_len=None: rebound.append(n))
    session = object.__new__(serving.BlockDiffusionServingSession)
    session.tt_model = SimpleNamespace()
    session.page_table = None
    session.page_tables_per_layer = None
    session.prefill_execution_len = 128
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
        lambda *args, **kwargs: (prefill_calls.append(kwargs) or SimpleNamespace(prompt_len=3, cache_len=32)),
    )

    assert session.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long)) == 32
    assert rebound == [32]
    assert session._logits_fn is adapter
    assert prefill_calls == [{"page_table": None, "page_tables_per_layer": None, "execution_len": 128}]


# --- vLLM startup contract ---------------------------------------------------


def _set_valid_upfront_env(monkeypatch):
    monkeypatch.setenv("DG_UPFRONT_CAPTURE", "1")
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "1024")
    monkeypatch.setenv("DG_TRACE_REGION_SIZE", "1073741824")


def test_vllm_upfront_configuration_accepts_only_released_contract(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    _set_valid_upfront_env(monkeypatch)
    # `device` (the on-device seeded SFPU RNG) is the ONLY materialized Gumbel source, so it
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


def test_vllm_upfront_warmup_defers_capture_until_trace_phase(monkeypatch):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    compiled = []
    monkeypatch.setattr(generator_vllm, "prefill_prompt_tokens", lambda model, toks: compiled.append(toks.shape[1]))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *_a, **_k: None)
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = [SimpleNamespace(mesh_device=object())]
    wrapper._upfront = True
    wrapper._persistent_adapter = None
    wrapper._make_session = lambda: pytest.fail("compile-only warmup must not build a capture session")

    wrapper.warmup_model_prefill(None, False, True)
    wrapper.warmup_model_decode()

    assert wrapper._upfront_compile_phase_seen is True
    assert wrapper._persistent_adapter is None
    assert compiled == [32]


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


def test_vllm_upfront_warmup_deduplicates_lengths_by_coarse_bucket(monkeypatch):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    compiled = []
    monkeypatch.setattr(generator_vllm, "prefill_prompt_tokens", lambda model, toks: compiled.append(toks.shape[1]))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *_a, **_k: None)
    monkeypatch.setenv("DG_UPFRONT_COARSE_PREFILL_BUCKETS", "1")
    monkeypatch.setenv("DG_UPFRONT_PREFILL_WARMUP_LENS", "128,160,192,224,256,2432")

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper._upfront = True
    wrapper.canvas_length = 256
    wrapper._upfront_pmax = 4096
    wrapper._max_model_len = 4096
    wrapper.model = [SimpleNamespace(mesh_device=object())]

    wrapper.warmup_model_prefill(None, enable_trace=False, can_sample_on_device=False)

    assert wrapper._upfront_prefill_warmup_lens == frozenset({32, 128, 256, 4096})
    assert compiled == [32, 128, 256, 4096]


def test_vllm_upfront_trace_phase_rejects_missing_prefill_warmups(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.data_parallel = 1
    wrapper.model = []
    wrapper._upfront = True
    wrapper._upfront_compile_phase_seen = True
    wrapper._upfront_prefill_warmup_lens = frozenset()

    with expect_error(RuntimeError, match="requires its compile-only prefill warmup"):
        wrapper.warmup_model_prefill(None, True, True)


def test_vllm_cold_prefill_reserves_full_attention_concat_holes(monkeypatch):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    allocations = []
    releases = []
    metrics = []

    class _Reservation:
        def deallocate(self, force):
            releases.append(force)

    def fake_empty(shape, **kwargs):
        allocations.append((shape, kwargs))
        return _Reservation()

    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.canvas_length = 256
    wrapper._upfront_pmax = 262144
    wrapper.model = [
        SimpleNamespace(
            mesh_device="mesh",
            hf_config=SimpleNamespace(text_config=SimpleNamespace(global_head_dim=512)),
        )
    ]

    monkeypatch.setattr(ttnn, "empty", fake_empty)
    monkeypatch.setattr(generator_vllm, "_metric", lambda event, **fields: metrics.append((event, fields)))

    reservations = wrapper._reserve_cold_recapture_holes()

    assert len(reservations) == 4
    assert [entry[0] for entry in allocations] == [[1, 1, 262400, 512]] * 4
    assert all(entry[1]["dtype"] == ttnn.bfloat16 for entry in allocations)
    assert all(entry[1]["layout"] == ttnn.TILE_LAYOUT for entry in allocations)
    assert all(entry[1]["device"] == "mesh" for entry in allocations)
    assert metrics == [
        (
            "cold_prefill_recapture_holes_reserved",
            {
                "buffers": 4,
                "shape": [1, 1, 262400, 512],
                "bytes_per_buffer": 268697600,
            },
        )
    ]

    wrapper._release_cold_recapture_holes(reservations)
    assert releases == [True] * 4


def test_vllm_cold_prefill_rebuild_releases_trace_before_compile(monkeypatch):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []
    adapter = SimpleNamespace(use_reveal_mask=True)
    emission = SimpleNamespace(tokens=torch.zeros((1, 256), dtype=torch.long))

    class _Session:
        def __init__(self):
            self._logits_fn = None
            self._persistent_adapter = None

        def prefill(self, prompt_tokens):
            assert events == ["release", "sync"], "compile must start only after trace release and CQ drain"
            events.append("compile")
            self._logits_fn = adapter
            return 128

    session = _Session()
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.model = [SimpleNamespace(mesh_device="mesh")]
    wrapper._sessions = {}
    wrapper._persistent_adapter = SimpleNamespace()
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._upfront_rebuild_in_progress = False
    wrapper._upfront_rebuilds = 0

    def release_resident():
        events.append("release")
        wrapper._persistent_adapter = None

    def capture_prefilled(captured_session):
        assert captured_session is session
        events.append("capture")
        return emission, adapter, [{"traces_captured": 48}]

    wrapper.release_persistent_capture = release_resident
    wrapper._capture_prefilled_session = capture_prefilled
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append("sync"))
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator_vllm, "_dram_snapshot", lambda *args, **kwargs: {})

    cache_len, actual_emission = wrapper._rebuild_for_cold_prefill(
        session,
        torch.zeros((1, 97), dtype=torch.long),
        expected_cache_len=128,
    )

    assert events == ["release", "sync", "compile", "sync", "capture"]
    assert cache_len == 128
    assert actual_emission is emission
    assert wrapper._persistent_adapter is adapter
    assert session._persistent_adapter is adapter
    assert wrapper._upfront_prefill_warmup_lens == frozenset({32, 128})
    assert wrapper._upfront_rebuilds == 1
    assert wrapper._upfront_rebuild_in_progress is False


def test_vllm_failed_cold_prefill_capture_releases_unpublished_state(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    events = []
    controller = SimpleNamespace(release=lambda: events.append("controller_release"))
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=controller,
        reset=lambda: events.append("adapter_reset"),
    )

    class _Session:
        def __init__(self):
            self._logits_fn = None
            self._persistent_adapter = None

        def prefill(self, prompt_tokens):
            events.append("compile")
            self._logits_fn = adapter
            return 128

    session = _Session()
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.model = [SimpleNamespace(mesh_device="mesh")]
    wrapper._sessions = {}
    wrapper._persistent_adapter = SimpleNamespace()
    wrapper._upfront_prefill_warmup_lens = frozenset({32})
    wrapper._upfront_rebuild_in_progress = False
    wrapper._upfront_rebuilds = 0

    def release_resident():
        events.append("release")
        wrapper._persistent_adapter = None

    def fail_capture(captured_session):
        events.append("capture")
        raise RuntimeError("injected recapture failure")

    wrapper.release_persistent_capture = release_resident
    wrapper._capture_prefilled_session = fail_capture
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append("sync"))
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)

    with expect_error(RuntimeError, match="injected recapture failure"):
        wrapper._rebuild_for_cold_prefill(
            session,
            torch.zeros((1, 97), dtype=torch.long),
            expected_cache_len=128,
        )

    assert events == [
        "release",
        "sync",
        "compile",
        "sync",
        "capture",
        "controller_release",
        "adapter_reset",
        "sync",
    ]
    assert not hasattr(adapter, "_upfront_traced_denoise_controller")
    assert session._logits_fn is None
    assert session._persistent_adapter is None
    assert wrapper._persistent_adapter is None
    assert wrapper._upfront_rebuilds == 0
    assert wrapper._upfront_rebuild_in_progress is False


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


# --- reveal-span buckets (DG_DENOISE_REVEAL_BUCKETS) ----------------------------
# The up-front capture binds ONE fixed span; buckets re-capture at a per-request
# power-of-two span instead of the deployment worst case (measured 440 ms/step at
# 262144 vs 205 ms/step at 4096 for the same block, P150x4 2026-08-10). These are
# host-side policy and orchestration tests; the capture itself is exercised by the
# device suites above.


def test_reveal_bucket_resolution(expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm as GV

    assert GV._resolve_reveal_bucket(1, ceiling=262144) == 4096  # floor
    assert GV._resolve_reveal_bucket(4096, ceiling=262144) == 4096
    assert GV._resolve_reveal_bucket(4097, ceiling=262144) == 8192
    assert GV._resolve_reveal_bucket(70000, ceiling=262144) == 131072
    # A non-power-of-two ceiling still yields a servable bucket.
    assert GV._resolve_reveal_bucket(200000, ceiling=261888) == 261888
    with expect_error(ValueError, match="exceeds the deployment ceiling"):
        GV._resolve_reveal_bucket(262145, ceiling=262144)


def test_reveal_bucket_floor_env(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm as GV

    monkeypatch.setenv("DG_REVEAL_BUCKET_FLOOR", "8192")
    assert GV._resolve_reveal_bucket(1, ceiling=262144) == 8192
    monkeypatch.setenv("DG_REVEAL_BUCKET_FLOOR", "1000")  # not a tile multiple
    with expect_error(RuntimeError, match="DG_REVEAL_BUCKET_FLOOR"):
        GV._resolve_reveal_bucket(1, ceiling=262144)


def test_reveal_bucket_change_policy(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm as GV

    # Upshift is mandatory.
    assert GV._resolve_reveal_bucket_change(8192, 4096) is True
    # Downshift needs 4x hysteresis: adjacent buckets never ping-pong.
    assert GV._resolve_reveal_bucket_change(4096, 8192) is False
    assert GV._resolve_reveal_bucket_change(4096, 16384) is True
    assert GV._resolve_reveal_bucket_change(4096, 4096) is False
    monkeypatch.setenv("DG_REVEAL_BUCKET_DOWNSHIFT", "0")
    assert GV._resolve_reveal_bucket_change(4096, 16384) is False
    assert GV._resolve_reveal_bucket_change(32768, 16384) is True


def _reveal_upshift_wrapper(generator_vllm, *, session, adapter):
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.model = [SimpleNamespace(mesh_device="mesh")]
    wrapper.canvas_length = 256
    wrapper._upfront = True
    wrapper._upfront_pmax = 262144
    wrapper._upfront_reveal_bucket = 4096
    wrapper._upfront_rebuild_in_progress = False
    wrapper._upfront_rebuilds = 0
    wrapper._sessions = {0: session}
    wrapper._persistent_adapter = adapter
    return wrapper


def test_vllm_reveal_upshift_detection(monkeypatch):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    controller = SimpleNamespace(captured=True, reveal_pmax=4096)
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=controller,
        prompt_len=3968,  # 3968 + 256 = 4224 > 4096
        use_reveal_mask=True,
    )
    session = SimpleNamespace()
    wrapper = _reveal_upshift_wrapper(generator_vllm, session=session, adapter=adapter)

    assert wrapper._reveal_upshift_needed_span(session) == 4224
    adapter.prompt_len = 3840  # 3840 + 256 = 4096 fits exactly
    assert wrapper._reveal_upshift_needed_span(session) is None
    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "0")
    adapter.prompt_len = 3968
    assert wrapper._reveal_upshift_needed_span(session) is None


def test_vllm_reveal_upshift_recaptures_on_live_session(monkeypatch):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    events = []
    spans = []
    emission = SimpleNamespace(tokens=torch.zeros((1, 256), dtype=torch.long))
    controller = SimpleNamespace(captured=True, reveal_pmax=4096, release=lambda: events.append("controller_release"))
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=controller,
        prompt_len=3968,
        use_reveal_mask=True,
    )
    session = SimpleNamespace(_logits_fn=adapter, _persistent_adapter=None)
    wrapper = _reveal_upshift_wrapper(generator_vllm, session=session, adapter=adapter)

    def capture_prefilled(captured_session):
        assert captured_session is session
        # The resident controller must be gone before a fresh capture can bind.
        assert not hasattr(adapter, "_upfront_traced_denoise_controller")
        events.append("capture")
        return emission, adapter, [{"traces_captured": 48}]

    wrapper._capture_prefilled_session = capture_prefilled
    wrapper._reserve_cold_recapture_holes = lambda **kwargs: events.append(f"holes@{kwargs.get('span')}") or ["concat"]
    wrapper._release_cold_recapture_holes = lambda reservations: events.append(
        f"released:{reservations[0] if reservations else 'none'}"
    )
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append("sync"))
    monkeypatch.setattr(generator_vllm, "set_active_reveal_pmax", lambda span: spans.append(span))
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator_vllm, "_dram_snapshot", lambda *args, **kwargs: {})

    actual = wrapper._rebuild_for_reveal_upshift(session, needed_span=4224, row=0)

    assert actual is emission
    assert events == [
        "controller_release",
        "sync",
        "holes@8192",
        "capture",
        "released:concat",
    ]
    assert spans == [8192]
    assert wrapper._upfront_reveal_bucket == 8192
    assert wrapper._persistent_adapter is adapter
    assert session._persistent_adapter is adapter
    assert wrapper._upfront_rebuilds == 1
    assert wrapper._upfront_rebuild_in_progress is False
    # The live session was never detached.
    assert wrapper._sessions == {0: session}


def test_vllm_reveal_upshift_failure_restores_span_and_unpublishes(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    import ttnn

    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    spans = []
    released = []
    controller = SimpleNamespace(captured=True, reveal_pmax=4096, release=lambda: None)
    adapter = SimpleNamespace(
        _upfront_traced_denoise_controller=controller,
        prompt_len=3968,
        use_reveal_mask=True,
    )
    session = SimpleNamespace(_logits_fn=adapter, _persistent_adapter=None)
    wrapper = _reveal_upshift_wrapper(generator_vllm, session=session, adapter=adapter)

    def failing_capture(captured_session):
        raise RuntimeError("capture blew up")

    wrapper._capture_prefilled_session = failing_capture
    wrapper._reserve_cold_recapture_holes = lambda **kwargs: []
    wrapper._release_cold_recapture_holes = lambda reservations: None
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(generator_vllm, "set_active_reveal_pmax", lambda span: spans.append(span))
    monkeypatch.setattr(
        generator_vllm.DiffusionGemmaForCausalLM,
        "_release_unpublished_adapter",
        staticmethod(lambda a, *, label: released.append((a, label))),
    )
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)

    with expect_error(RuntimeError, match="capture blew up"):
        wrapper._rebuild_for_reveal_upshift(session, needed_span=4224, row=0)

    # Attempted bucket first, then restored to the still-recorded resident.
    assert spans == [8192, 4096]
    assert wrapper._upfront_reveal_bucket == 4096
    assert wrapper._persistent_adapter is None
    assert session._logits_fn is None
    assert session._persistent_adapter is None
    assert released and released[0][1] == "failed reveal-upshift"
    assert wrapper._upfront_rebuild_in_progress is False


def test_vllm_reveal_upshift_refuses_extra_sessions(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    controller = SimpleNamespace(captured=True, reveal_pmax=4096, release=lambda: None)
    adapter = SimpleNamespace(_upfront_traced_denoise_controller=controller, prompt_len=3968)
    session = SimpleNamespace(_logits_fn=adapter, _persistent_adapter=None)
    wrapper = _reveal_upshift_wrapper(generator_vllm, session=session, adapter=adapter)
    wrapper._sessions = {0: session, 1: SimpleNamespace()}

    with expect_error(RuntimeError, match="only active session"):
        wrapper._rebuild_for_reveal_upshift(session, needed_span=4224, row=0)


def test_active_reveal_pmax_overrides_env(monkeypatch):
    monkeypatch.setenv("DG_DENOISE_REVEAL_PMAX", "262144")
    adapter = SimpleNamespace(tt_model=None)
    try:
        TD.set_active_reveal_pmax(8192)
        assert TD._resolve_reveal_pmax(adapter) == 8192
        TD.set_active_reveal_pmax(None)
        assert TD._resolve_reveal_pmax(adapter) == 262144
    finally:
        TD.set_active_reveal_pmax(None)


def _serve_blocks_fresh_controller(wrapper, tokens: torch.Tensor, *, num_blocks: int):
    """Like _serve_once, but re-reads the persistent controller every block so a
    mid-request reveal upshift (which replaces the controller) keeps telemetry
    readable."""
    outputs = []
    reveal_pmax_per_block = []
    for block_idx in range(num_blocks):
        output = (
            wrapper.prefill_forward(tokens, prompt_lens=[int(tokens.shape[1])])
            if block_idx == 0
            else wrapper.decode_forward()
        )
        outputs.append(output)
        controller = wrapper._persistent_adapter._upfront_traced_denoise_controller
        reveal_pmax_per_block.append(int(controller.reveal_pmax))
    wrapper.release_request(0)
    return torch.cat(outputs, dim=1), reveal_pmax_per_block


def test_device_reveal_bucket_mid_request_upshift_matches_fixed_span(upfront_device_bundle, monkeypatch):
    """A generation crossing its bucket recaptures on the live session and stays bit-exact.

    Geometry: floor 512, ceiling 1024 (the bundle's max_seq_len). A short prompt
    admits at bucket 512; block 1's reveal (288 + 256 = 544) no longer fits, so
    decode upshifts to 1024 mid-request. Same seed as the fixed-span reference,
    so the reveal mask must make the span invisible to the committed tokens.
    """
    prompt = _tokenize(upfront_device_bundle, "Write one sentence about rain.")

    # Mechanism test, not a quality test: the bundle's short-context model can emit
    # canvases the degeneracy guard would end at block 0, and the request must keep
    # growing for the bucket to overflow. Correctness is the bit-exact token match.
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "off")
    # Budget 0 = pure growth-on-demand, which is what forces the mid-request upshift
    # this test exists to exercise (a real deployment provisions at admission).
    monkeypatch.setenv("DG_REVEAL_OUTPUT_BUDGET", "0")
    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "0")
    reference_wrapper = _make_upfront_wrapper(upfront_device_bundle, [prompt])
    try:
        reference_tokens, reference_spans = _serve_blocks_fresh_controller(reference_wrapper, prompt, num_blocks=3)
    finally:
        reference_wrapper.release_persistent_capture()
    assert set(reference_spans) == {1024}, "reference arm must run the fixed max_seq_len span"

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    monkeypatch.setenv("DG_REVEAL_BUCKET_FLOOR", "512")
    wrapper = _make_upfront_wrapper(upfront_device_bundle, [prompt])
    try:
        assert wrapper._upfront_reveal_bucket == 512
        tokens, spans = _serve_blocks_fresh_controller(wrapper, prompt, num_blocks=3)
        assert spans == [512, 1024, 1024], f"expected the block-1 upshift, got {spans}"
        assert wrapper._upfront_rebuilds == 1
        assert wrapper._upfront_reveal_bucket == 1024
        assert torch.equal(tokens, reference_tokens)
    finally:
        wrapper.release_persistent_capture()


def test_device_reveal_bucket_admission_change_matches_fixed_span(upfront_device_bundle, monkeypatch):
    """A bigger prompt re-buckets at admission; a later small prompt keeps the
    high bucket (4x hysteresis) without a rebuild."""
    small = _tokenize(upfront_device_bundle, "Write one sentence about rain.")
    large = _tokenize(upfront_device_bundle, "Explain in detail why rainbows form after a storm. " * 40)
    assert int(large.shape[1]) + 256 > 512, "large prompt must not fit the floor bucket"

    monkeypatch.setenv("DG_DEGENERACY_POLICY", "off")  # mechanism test; see the upshift test
    monkeypatch.setenv("DG_REVEAL_OUTPUT_BUDGET", "0")  # growth-on-demand geometry
    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "0")
    reference_wrapper = _make_upfront_wrapper(upfront_device_bundle, [small, large])
    try:
        small_reference, _ = _serve_blocks_fresh_controller(reference_wrapper, small, num_blocks=1)
        large_reference, _ = _serve_blocks_fresh_controller(reference_wrapper, large, num_blocks=1)
    finally:
        reference_wrapper.release_persistent_capture()

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    monkeypatch.setenv("DG_REVEAL_BUCKET_FLOOR", "512")
    wrapper = _make_upfront_wrapper(upfront_device_bundle, [small, large])
    try:
        assert wrapper._upfront_reveal_bucket == 512
        small_tokens, small_spans = _serve_blocks_fresh_controller(wrapper, small, num_blocks=1)
        assert small_spans == [512]
        assert wrapper._upfront_rebuilds == 0

        large_tokens, large_spans = _serve_blocks_fresh_controller(wrapper, large, num_blocks=1)
        assert large_spans == [1024], "the large prompt must re-bucket at admission"
        assert wrapper._upfront_rebuilds == 1

        small_again, again_spans = _serve_blocks_fresh_controller(wrapper, small, num_blocks=1)
        assert again_spans == [1024], "hysteresis must keep the high bucket for the small prompt"
        assert wrapper._upfront_rebuilds == 1, "no downshift rebuild within 4x hysteresis"

        assert torch.equal(small_tokens, small_reference)
        assert torch.equal(large_tokens, large_reference)
        assert torch.equal(small_again, small_reference)
    finally:
        wrapper.release_persistent_capture()


def test_reveal_output_budget_provisioning(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm as GV

    # Unset -> provision the ceiling: no mid-request growth can ever fire.
    monkeypatch.delenv("DG_REVEAL_OUTPUT_BUDGET", raising=False)
    assert GV._reveal_provisioned_span(160, canvas_length=256, ceiling=262144) == 262144
    # Explicit budget -> prompt + canvas + budget, clipped to the ceiling.
    monkeypatch.setenv("DG_REVEAL_OUTPUT_BUDGET", "13824")
    assert GV._reveal_provisioned_span(160, canvas_length=256, ceiling=262144) == 14240
    assert GV._resolve_reveal_bucket(14240, ceiling=262144) == 16384
    assert GV._reveal_provisioned_span(260000, canvas_length=256, ceiling=262144) == 262144
    # Budget 0 restores pure growth-on-demand (prompt + one canvas).
    monkeypatch.setenv("DG_REVEAL_OUTPUT_BUDGET", "0")
    assert GV._reveal_provisioned_span(160, canvas_length=256, ceiling=262144) == 416
    monkeypatch.setenv("DG_REVEAL_OUTPUT_BUDGET", "-5")
    with expect_error(RuntimeError, match="DG_REVEAL_OUTPUT_BUDGET"):
        GV._reveal_output_budget()


def test_vllm_failed_upshift_costs_one_request_not_the_engine(monkeypatch):
    """A failed growth recapture ends the request, restores a capture, and serving continues."""
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    events = []
    stop_block = torch.full((1, 256), 7, dtype=torch.long)
    session = SimpleNamespace(finished=False, _logits_fn=None, _persistent_adapter=None)
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.model = [SimpleNamespace(mesh_device="mesh")]
    wrapper.canvas_length = 256
    wrapper._upfront = True
    wrapper._sessions = {0: session}
    wrapper._reveal_upshift_needed_span = lambda s: 4224
    wrapper._stop_block = lambda s: events.append("stop_block") or stop_block

    def failing_upshift(s, *, needed_span, row):
        events.append("upshift")
        raise RuntimeError("DRAM says no")

    wrapper._rebuild_for_reveal_upshift = failing_upshift
    wrapper.release_request = lambda row: events.append(f"release:{row}")
    wrapper._restore_resident_capture = lambda: events.append("restore")
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)

    out = wrapper.decode_forward()

    assert events == ["upshift", "stop_block", "release:0", "restore"]
    assert torch.equal(out, stop_block)


def test_vllm_failed_upshift_and_failed_restore_is_fatal(monkeypatch, expect_error):
    pytest.importorskip("vllm")
    from models.experimental.diffusion_gemma.tt import generator_vllm

    monkeypatch.setenv("DG_DENOISE_REVEAL_BUCKETS", "1")
    session = SimpleNamespace(finished=False, _logits_fn=None, _persistent_adapter=None)
    wrapper = object.__new__(generator_vllm.DiffusionGemmaForCausalLM)
    wrapper.model = [SimpleNamespace(mesh_device="mesh")]
    wrapper.canvas_length = 256
    wrapper._upfront = True
    wrapper._sessions = {0: session}
    wrapper._reveal_upshift_needed_span = lambda s: 4224
    wrapper._stop_block = lambda s: torch.zeros((1, 256), dtype=torch.long)

    def failing_upshift(s, *, needed_span, row):
        raise RuntimeError("DRAM says no")

    def failing_restore():
        raise RuntimeError("restore also failed")

    wrapper._rebuild_for_reveal_upshift = failing_upshift
    wrapper.release_request = lambda row: None
    wrapper._restore_resident_capture = failing_restore
    monkeypatch.setattr(generator_vllm, "_metric", lambda *args, **kwargs: None)

    with expect_error(RuntimeError, match="restore also failed"):
        wrapper.decode_forward()
