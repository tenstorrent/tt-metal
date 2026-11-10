# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""An Ideogram 4 pipeline built with trace=False captures no trace and asks for no trace region. Needs no device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn

from ...pipelines.ideogram4 import pipeline_ideogram4 as module
from ...pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline


class _Reached(Exception):
    pass


def _raise(*args: object, **kwargs: object) -> None:
    raise _Reached


@pytest.mark.parametrize("trace", [True, False])
def test_construction_checks_the_trace_region_only_when_tracing(monkeypatch: pytest.MonkeyPatch, trace: bool) -> None:
    monkeypatch.setattr(module, "_check_trace_region", _raise)
    config = SimpleNamespace(trace=trace, resolutions=((512, 512),))
    with pytest.raises(Exception) as raised:  # noqa: PT011
        Ideogram4Pipeline(device=SimpleNamespace(shape=(4, 8)), config=config)
    assert isinstance(raised.value, _Reached) is trace


@pytest.mark.parametrize("trace", [True, False])
def test_warmup_captures_only_when_tracing(monkeypatch: pytest.MonkeyPatch, trace: bool) -> None:
    checked: list[object] = []
    monkeypatch.setattr(module, "_check_trace_region", lambda *args: checked.append(args))
    monkeypatch.setattr(module, "Tracer", _raise)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda device: None)
    monkeypatch.setattr(Ideogram4Pipeline, "release_traces", lambda self: None)
    monkeypatch.setattr(Ideogram4Pipeline, "_encode", lambda self, prompt, warm_readback: (None, 0))
    monkeypatch.setattr(Ideogram4Pipeline, "_step_fn", lambda self, st, sb: None)

    pipeline = object.__new__(Ideogram4Pipeline)
    pipeline.trace = trace
    pipeline.mesh_device = SimpleNamespace(shape=(4, 8))
    pipeline._sizes = {(512, 512): SimpleNamespace(padded_len=1)}  # already warmed eagerly
    pipeline._seq_buffers = {1: None}

    if trace:
        with pytest.raises(_Reached):
            pipeline.warmup(sizes=[])
        assert len(checked) == 1
    else:
        pipeline.warmup(sizes=[])
        assert checked == []
