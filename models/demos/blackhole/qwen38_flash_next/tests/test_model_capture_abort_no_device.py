"""A failure inside a trace capture ends the capture and releases its trace before the error propagates (no device):
the failure path can then synchronize and close the mesh instead of hitting the event-sync TT_FATAL inside an open
capture and hanging the mesh close."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module


class FakeTTNN:
    def __init__(self, *, end_raises: bool = False):
        self.calls: list[tuple] = []
        self.end_raises = end_raises

    def begin_trace_capture(self, mesh, cq_id=0):
        self.calls.append(("begin", cq_id))
        return 7

    def end_trace_capture(self, mesh, trace_id, cq_id=0):
        self.calls.append(("end", trace_id, cq_id))
        if self.end_raises and sum(1 for c in self.calls if c[0] == "end") >= 2:
            raise RuntimeError("end refused")  # the aborted capture's end (the head's own end before it succeeds)

    def release_trace(self, mesh, trace_id):
        self.calls.append(("release", trace_id))

    @contextlib.contextmanager
    def corruptible_allocation_scope(self, mesh):
        yield


def _capture(monkeypatch, fake: FakeTTNN, failing_part: str):
    monkeypatch.setattr(model_module, "ttnn", fake)
    monkeypatch.setattr(model_module, "corruptible_allocation_scope", fake.corruptible_allocation_scope)
    owner = object.__new__(model_module.Qwen38TTNNTextModel)
    owner.mesh_device = SimpleNamespace()
    head = SimpleNamespace(active=True, residual=None)

    def forward_head(prepared, state):
        if failing_part == "head":
            raise RuntimeError("boom in the head body")
        return head

    def forward_tail(head_arg, prepared, state, *, release_head, retain_mtp_inputs):
        if failing_part == "tail":
            raise RuntimeError("boom in the tail body")
        return SimpleNamespace(logits=object())

    monkeypatch.setattr(owner, "forward_decode_generic_head", forward_head, raising=False)
    monkeypatch.setattr(owner, "forward_decode_generic_tail", forward_tail, raising=False)
    message = f"boom in the {failing_part} body"
    with pytest.raises(RuntimeError, match=message) as info:  # allow-pytest.raises: reads the exception
        owner.capture_decode_generic(
            SimpleNamespace(),
            SimpleNamespace(),
            residue=0,
            split=True,
            guard=lambda label: contextlib.nullcontext([]),
            epilogue=lambda output: ("epilogue", output),
            retain_mtp_inputs=True,
        )
    return info.value


@pytest.mark.parametrize("failing_part", ["head", "tail"])
def test_a_failure_inside_the_capture_ends_it_and_releases_the_trace_then_re_raises(monkeypatch, failing_part):
    fake = FakeTTNN()
    error = _capture(monkeypatch, fake, failing_part)
    assert str(error) == f"boom in the {failing_part} body"
    begins = [c for c in fake.calls if c[0] == "begin"]
    assert fake.calls[-2:] == [("end", 7, 0), ("release", 7)]  # the failing part's capture is ended and released
    assert len(begins) == (1 if failing_part == "head" else 2)  # a head failure never begins the tail capture


def test_the_abort_is_best_effort_and_keeps_the_original_error(monkeypatch):
    fake = FakeTTNN(end_raises=True)
    with pytest.warns(RuntimeWarning, match="capture end after a failure"):
        error = _capture(monkeypatch, fake, "tail")
    assert str(error) == "boom in the tail body"  # the cleanup's own failure is a warning, not the error
    assert ("release", 7) in fake.calls  # the release still runs after a refused end
