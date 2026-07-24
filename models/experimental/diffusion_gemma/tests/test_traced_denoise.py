# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the accepted up-front-only denoise trace architecture."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import traced_denoise as TD


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


@pytest.fixture(autouse=True)
def _fake_ttnn(monkeypatch):
    _FakeTtnn.reset()
    monkeypatch.setattr(TD, "ttnn", _FakeTtnn)


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


def test_controller_accepts_only_released_48_step_schedule(expect_error):
    controller = _controller()
    assert controller.config.max_denoise_steps == 48

    with expect_error(ValueError, match="released 48-step schedule"):
        TD.UpfrontTracedDenoiseController("mesh", _config(max_denoise_steps=47))
    with expect_error(ValueError, match="stable_steps_to_halt=1"):
        TD.UpfrontTracedDenoiseController("mesh", _config(stable_steps_to_halt=2))


def test_trace_capture_guard_ends_and_releases_aborted_trace(expect_error):
    with expect_error(RuntimeError, match="injected capture failure"):
        with TD._trace_capture_guard("mesh", cq_id=0):
            raise RuntimeError("injected capture failure")

    assert _FakeTtnn.trace_events == [
        ("begin", "mesh", 0),
        ("end", "mesh", "trace-id", 0),
        ("release", "mesh", "trace-id"),
    ]


def test_trace_capture_guard_releases_when_finalization_fails(expect_error):
    _FakeTtnn.end_error = RuntimeError("injected end failure")

    with expect_error(RuntimeError, match="injected end failure"):
        with TD._trace_capture_guard("mesh", cq_id=0):
            pass

    assert _FakeTtnn.trace_events[-1] == ("release", "mesh", "trace-id")


def test_materialized_gumbel_uses_stable_buffer_and_consumes_sources():
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
def test_materialized_gumbel_rejects_non_tensor_descriptors(value, expect_error):
    controller = _controller()
    with expect_error(ValueError, match="requires materialized host noise"):
        controller._initialize_gumbel(lambda step: value)


def test_materialized_renoise_uses_one_stable_buffer_and_consumes_only_requested_steps():
    controller = _controller()
    first = _FakeTensor("noise-0")
    controller._initialize_noise(lambda step: first)
    assert controller.noise_buf.name == "clone(noise-0)"
    assert first.deallocated

    fresh = _FakeTensor("noise-7")
    assert controller._refresh_noise(lambda step: fresh, 7) is controller.noise_buf
    assert _FakeTtnn.copies == [("noise-7", "clone(noise-0)")]
    assert fresh.deallocated


def test_replay_reuses_reveal_buffers_and_stops_on_materialized_halt(monkeypatch):
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


def test_controller_release_is_best_effort_idempotent_and_clears_state():
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


def test_upfront_block_reuses_the_single_controller_attribute(monkeypatch):
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


def test_upfront_block_rejects_on_demand_capture_outside_startup(expect_error):
    logits_fn = SimpleNamespace(tt_model=SimpleNamespace(mesh_device="mesh"))
    with expect_error(RuntimeError, match="startup trace warmup"):
        TD.upfront_traced_denoise_block(
            logits_fn,
            "canvas",
            _config(),
            gumbel_noise_fn="gumbel",
            noise_tokens_fn="noise",
        )
