# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT_PERF_TRACE_REGION raises a run's trace region. It must never lower the one the MODEL declares.

optimize exports a region derived from the board's DRAM and both grow loops raise it when a capture
overflows: every writer of that variable wants MORE room. A generated perf test passed it straight
into the model's own device-open, so the ask behaved as an override -- and on Qwen-Image-Edit
(2026-10-23) that handed a 40 MiB region to an open the model had sized at 1 GiB. All five stages
died in mesh_trace.cpp's region assertion; the test still reported a pass, because an untimed stage
costs only its own row; the surviving three-stage snapshot was then ranked as the whole pipeline and
the attempts went to the two encoders while 75% of the runtime was invisible.

Measured on that box: 40 MiB -> 0 stage rows and TRACE_PER_TOKEN_MS=0.0000; 1 GiB -> 5 rows,
denoise 42346 ms of a 56605 ms pipeline. perf_adapter.resolve_trace_region is the one place that
decides, and these pin its rule and the skeletons' use of it.
"""

from __future__ import annotations

import inspect

from models.experimental.perf_automation.agent import perf_test_gen as G
from models.experimental.perf_automation.agent.perf_adapter import (
    DEFAULT_TRACE_REGION_BYTES,
    TRACE_REGION_ENV,
    resolve_trace_region,
)

_DECLARED = 1 << 30  # what a model states for its own largest trace
_ASKED = 40 * 1024 * 1024  # what the tool happened to ask for


def _skeletons():
    return [s for s in (G._SKELETON_REF, G._SELF_TRACED_SKELETON_REF, G._SKELETON_COMPONENT) if s]


def test_a_smaller_ask_never_shrinks_what_the_model_declares(monkeypatch):
    """THE REGRESSION THIS FILE EXISTS FOR."""
    monkeypatch.setenv(TRACE_REGION_ENV, str(_ASKED))
    assert resolve_trace_region(_DECLARED) == _DECLARED


def test_a_bigger_ask_still_raises_it(monkeypatch):
    """The grow loops must keep working: they only ever ask for more."""
    monkeypatch.setenv(TRACE_REGION_ENV, str(_DECLARED * 2))
    assert resolve_trace_region(_DECLARED) == _DECLARED * 2


def test_an_unset_ask_leaves_the_model_alone(monkeypatch):
    """A bare manual run behaves exactly as the demo does."""
    monkeypatch.delenv(TRACE_REGION_ENV, raising=False)
    assert resolve_trace_region(_DECLARED) == _DECLARED


def test_a_model_that_declares_nothing_falls_back_to_the_shared_default(monkeypatch):
    monkeypatch.delenv(TRACE_REGION_ENV, raising=False)
    assert resolve_trace_region(0) == DEFAULT_TRACE_REGION_BYTES
    assert resolve_trace_region() == DEFAULT_TRACE_REGION_BYTES


def test_an_unparseable_ask_is_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv(TRACE_REGION_ENV, "nonsense")
    assert resolve_trace_region(_DECLARED) == _DECLARED
    monkeypatch.setenv(TRACE_REGION_ENV, "")
    assert resolve_trace_region(_DECLARED) == _DECLARED


def test_the_generator_states_the_size_once():
    """Two copies of this number is how one of them came to disagree with the other."""
    assert G._DEFAULT_TRACE_REGION_BYTES is DEFAULT_TRACE_REGION_BYTES


def test_no_skeleton_reads_the_region_out_of_the_environment_itself():
    """Structural, not prose: the builder copies the skeleton far more reliably than it applies a rule."""
    for s in _skeletons():
        assert 'os.environ.get("%s"' % TRACE_REGION_ENV not in s, "skeleton reads the ask as an override"
        assert "resolve_trace_region(" in s, "skeleton does not resolve the region"


def test_the_builders_own_rule_names_the_resolver():
    """The qwen test was WRITTEN by the builder, not copied from a skeleton: the rule has to say it."""
    src = inspect.getsource(G.generate_perf_test)
    assert "resolve_trace_region" in src
