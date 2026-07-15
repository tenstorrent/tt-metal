# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for DiffusionGemma traced-denoise input lifetimes."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import sampling as TS
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
    uint32 = object()
    TILE_LAYOUT = object()
    DRAM_MEMORY_CONFIG = object()
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

    @staticmethod
    def from_torch(host, **kwargs):
        del kwargs
        return _FakeTensor(f"seed-{int(host[0, 0, 0, 0])}")

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


def _controller(steps=3):
    return TD.TracedDenoiseController("mesh", DiffusionConfig(canvas_length=32, max_denoise_steps=steps))


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

    assert _FakeTtnn.trace_events == [
        ("begin", "mesh", 0),
        ("end", "mesh", "trace-id", 0),
        ("release", "mesh", "trace-id"),
    ]


def test_materialized_gumbel_uses_stable_buffer_and_consumes_source():
    controller = _controller()
    fresh = _FakeTensor("gumbel-0")

    controller._initialize_gumbel_buffer_from(lambda step: fresh)

    assert controller.gumbel_mode == "materialized"
    assert controller.gumbel_buf.name == "clone(gumbel-0)"
    assert fresh.deallocated


def test_materialized_gumbel_refreshes_in_place_and_rejects_mode_change(expect_error):
    controller = _controller()
    controller._initialize_gumbel_buffer_from(lambda step: _FakeTensor("gumbel-0"))
    fresh = _FakeTensor("gumbel-2")

    assert controller._refresh_gumbel_buffer_from(lambda step: fresh, 2) is controller.gumbel_buf
    assert _FakeTtnn.copies == [("gumbel-2", "clone(gumbel-0)")]
    assert fresh.deallocated

    with expect_error(ValueError, match="changed mode"):
        controller._refresh_gumbel_buffer_from(lambda step: None, 3)


def test_chunked_gumbel_builds_trace_dynamic_seed_schedule():
    controller = _controller()

    controller._initialize_gumbel_buffer_from(lambda step: TS.ChunkedGumbelNoise(seed=17 + step, vocab_chunk_size=1024))

    assert controller.gumbel_mode == "chunked"
    assert controller.gumbel_chunked_seeds == (17, 18, 19)
    assert controller.gumbel_chunked_state.seed_tensor.name == "seed-17"
    assert [value.seed_offset for value in controller.gumbel_step_inputs] == [0, 0, 0]

    controller._refresh_chunked_gumbel_seed_from(
        lambda step: TS.ChunkedGumbelNoise(seed=101 + step, vocab_chunk_size=1024)
    )
    controller._refresh_chunked_gumbel_seed_for_step(0)
    assert _FakeTtnn.copies == [("seed-101", "seed-17")]


def test_chunked_gumbel_rejects_cross_block_signature_change(expect_error):
    controller = _controller()
    controller._initialize_gumbel_buffer_from(lambda step: TS.ChunkedGumbelNoise(seed=17 + step, vocab_chunk_size=1024))

    with expect_error(ValueError, match="signature changed"):
        controller._refresh_chunked_gumbel_seed_from(
            lambda step: TS.ChunkedGumbelNoise(seed=101 + step, vocab_chunk_size=512)
        )


def test_argmax_gumbel_rejects_mixed_step_and_cross_block_modes(expect_error):
    controller = _controller()
    mixed = _FakeTensor("mixed-step")

    with expect_error(ValueError, match="within the denoise step schedule"):
        controller._initialize_gumbel_buffer_from(lambda step: None if step == 0 else mixed)
    assert mixed.deallocated

    controller = _controller()
    controller._initialize_gumbel_buffer_from(lambda step: None)
    changed = _FakeTensor("changed-block")
    with expect_error(ValueError, match="changed from argmax"):
        controller._validate_argmax_gumbel_from(lambda step: changed if step == 1 else None)
    assert changed.deallocated


def test_grouped_trace_rejects_single_shared_materialized_gumbel_buffer(expect_error):
    controller = _controller()
    controller._initialize_gumbel_buffer_from(lambda step: _FakeTensor("gumbel-0"))
    buffer = controller.gumbel_buf

    with expect_error(NotImplementedError, match="one-step trace windows"):
        controller._reject_grouped_dynamic_gumbel(2)

    assert buffer.deallocated
    assert controller.gumbel_buf is None
    assert controller.gumbel_mode is None


def test_controller_release_is_best_effort_and_clears_state():
    controller = _controller()
    _FakeTtnn.release_errors = {"trace-0"}
    bad = _FakeTensor("bad-buffer", deallocate_error=RuntimeError("injected buffer failure"))
    good = _FakeTensor("good-buffer")
    chunk_release_attempted = []

    def fail_chunk_release():
        chunk_release_attempted.append(True)
        raise RuntimeError("injected chunk-state failure")

    controller.traces = ["trace-0", "trace-1"]
    controller.canvas_buf = bad
    controller.committed_buf = good
    controller.gumbel_chunked_state = SimpleNamespace(release=fail_chunk_release)
    controller.gumbel_mode = "chunked"
    controller.captured = True

    controller.release()

    assert ("release", "mesh", "trace-0") in _FakeTtnn.trace_events
    assert ("release", "mesh", "trace-1") in _FakeTtnn.trace_events
    assert bad.deallocate_attempted
    assert good.deallocated
    assert chunk_release_attempted == [True]
    assert controller.traces == []
    assert controller.canvas_buf is None
    assert controller.committed_buf is None
    assert controller.gumbel_chunked_state is None
    assert controller.gumbel_mode is None
    assert controller.captured is False


def test_steady_replay_refreshes_each_materialized_gumbel_before_trace(monkeypatch):
    controller = _controller(steps=3)
    controller.captured = True
    controller.traces = ["trace-0", "trace-1", "trace-2"]
    controller.canvas_buf = _FakeTensor("canvas-buf")
    controller.committed_buf = _FakeTensor("committed-buf")
    controller.gumbel_buf = _FakeTensor("gumbel-buf")
    controller.gumbel_mode = "materialized"
    controller.noise_bufs = [_FakeTensor(f"noise-{step}") for step in range(3)]
    monkeypatch.setattr(controller, "_refresh_noise_buffers_from", lambda fn: None)
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[7]], dtype=torch.long))

    adapter = SimpleNamespace(
        q_rope_offset=64,
        update_canvas_rope_buffers=lambda start_pos: None,
        reset_signal_buffer=lambda: None,
    )
    calls = []

    def gumbel_for_step(step):
        calls.append(step)
        return _FakeTensor(f"gumbel-{step}")

    init_canvas = _FakeTensor("init-canvas")
    trajectory = controller.denoise_block(
        adapter,
        init_canvas,
        controller.config,
        gumbel_noise_fn=gumbel_for_step,
        noise_tokens_fn=lambda step: _FakeTensor(f"unused-noise-{step}"),
    )

    assert calls == [0, 1, 2]
    assert _FakeTtnn.copies == [
        ("init-canvas", "canvas-buf"),
        ("gumbel-0", "gumbel-buf"),
        ("gumbel-1", "gumbel-buf"),
        ("gumbel-2", "gumbel-buf"),
    ]
    assert _FakeTtnn.executions == ["trace-0", "trace-1", "trace-2"]
    assert init_canvas.deallocated
    assert torch.equal(trajectory.committed, torch.tensor([[7]]))


def test_capture_block_reuses_prepared_step_zero_gumbel(monkeypatch):
    controller = _controller(steps=3)
    controller.canvas_buf = _FakeTensor("canvas-buf")
    controller.committed_buf = _FakeTensor("committed-buf")
    controller.noise_bufs = [_FakeTensor(f"noise-{step}") for step in range(3)]
    monkeypatch.setattr(controller, "_refresh_noise_buffers_from", lambda fn: None)
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[9]], dtype=torch.long))

    calls = []

    def gumbel_for_step(step):
        calls.append(step)
        return _FakeTensor(f"gumbel-{step}")

    def fake_capture(adapter, init_canvas, gumbel_noise_fn, noise_tokens_fn, start_pos):
        del adapter, init_canvas, noise_tokens_fn, start_pos
        controller._initialize_gumbel_buffer_from(gumbel_noise_fn)
        controller.traces = ["trace-0", "trace-1", "trace-2"]
        controller.captured = True

    monkeypatch.setattr(controller, "_capture", fake_capture)
    adapter = SimpleNamespace(
        q_rope_offset=32,
        update_canvas_rope_buffers=lambda start_pos: None,
        reset_signal_buffer=lambda: None,
    )

    controller.denoise_block(
        adapter,
        _FakeTensor("init-canvas"),
        controller.config,
        gumbel_noise_fn=gumbel_for_step,
        noise_tokens_fn=lambda step: _FakeTensor(f"unused-noise-{step}"),
    )

    # Step 0 initializes the stable buffer and is not regenerated before its first replay.
    assert calls == [0, 1, 2]


def test_prefix_growth_invalidates_and_recaptures_trace(monkeypatch):
    controller = _controller(steps=1)
    controller.captured = True
    controller.captured_prefix_len = 32
    controller.traces = ["old-trace"]
    controller.canvas_buf = _FakeTensor("old-canvas")
    controller.committed_buf = _FakeTensor("old-commit")
    events = []

    def fake_release():
        events.append("release-old")
        controller.captured = False
        controller.captured_prefix_len = None
        controller.traces = []
        controller.canvas_buf = None
        controller.committed_buf = None
        controller.gumbel_mode = None

    def fake_capture(adapter, init_canvas, gumbel_noise_fn, noise_tokens_fn, start_pos):
        del init_canvas, gumbel_noise_fn, noise_tokens_fn, start_pos
        events.append(("capture-new", adapter.prompt_len))
        controller.captured = True
        controller.captured_prefix_len = adapter.prompt_len
        controller.traces = ["new-trace"]
        controller.canvas_buf = _FakeTensor("new-canvas")
        controller.committed_buf = _FakeTensor("new-commit")
        controller.gumbel_mode = "argmax"

    monkeypatch.setattr(controller, "release", fake_release)
    monkeypatch.setattr(controller, "_capture", fake_capture)
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[11]], dtype=torch.long))
    adapter = SimpleNamespace(
        prompt_len=288,
        q_rope_offset=288,
        update_canvas_rope_buffers=lambda start_pos: None,
        reset_signal_buffer=lambda: None,
    )

    trajectory = controller.denoise_block(
        adapter,
        _FakeTensor("init-canvas"),
        controller.config,
        gumbel_noise_fn=lambda step: None,
        noise_tokens_fn=lambda step: _FakeTensor("noise"),
    )

    assert events == ["release-old", ("capture-new", 288)]
    assert _FakeTtnn.executions == ["new-trace"]
    assert controller.captured_prefix_len == 288
    assert torch.equal(trajectory.committed, torch.tensor([[11]]))


def test_frozen_prefix_reuses_block0_trace_without_recapture(monkeypatch):
    # DG_DENOISE_FROZEN_PREFIX: capture-once/replay-many — prefix growth must NOT recapture;
    # the block-0 trace is reused (restores the pre-recapture steady serving speed).
    monkeypatch.setenv("DG_DENOISE_FROZEN_PREFIX", "1")
    controller = _controller(steps=1)
    controller.captured = True
    controller.captured_prefix_len = 32
    controller.traces = ["block0-trace"]
    controller.canvas_buf = _FakeTensor("canvas-buf")
    controller.committed_buf = _FakeTensor("committed-buf")
    controller.gumbel_mode = "argmax"
    events = []
    monkeypatch.setattr(controller, "release", lambda: events.append("release"))
    monkeypatch.setattr(controller, "_capture", lambda *a, **k: events.append("capture"))
    monkeypatch.setattr(controller, "_refresh_noise_buffers_from", lambda fn: None)
    monkeypatch.setattr(controller, "_refresh_chunked_gumbel_seed_from", lambda fn: None)
    monkeypatch.setattr(controller, "_validate_argmax_gumbel_from", lambda fn: None)
    monkeypatch.setattr(TD, "_ids_to_torch", lambda tensor: torch.tensor([[13]], dtype=torch.long))
    adapter = SimpleNamespace(
        prompt_len=288,  # prefix grew 32 -> 288
        q_rope_offset=288,
        update_canvas_rope_buffers=lambda start_pos: None,
        reset_signal_buffer=lambda: None,
    )

    trajectory = controller.denoise_block(
        adapter,
        _FakeTensor("init-canvas"),
        controller.config,
        gumbel_noise_fn=lambda step: None,
        noise_tokens_fn=lambda step: _FakeTensor("noise"),
    )

    # No release, no recapture; the frozen block-0 trace replays and captured_prefix_len is unchanged.
    assert events == []
    assert controller.captured_prefix_len == 32
    assert _FakeTtnn.executions == ["block0-trace"]
    assert torch.equal(trajectory.committed, torch.tensor([[13]]))
