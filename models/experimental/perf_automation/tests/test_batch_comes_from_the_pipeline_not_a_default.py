# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline says how many users it serves, and every reader must ask the same way.

WHAT WAS PUBLISHED, 2026-08-15. Voxtral-Mini-3B declares DECODE_BATCH = 8 and the run genuinely
served 8 streams -- its own log says so:

    PERF_BATCH_STREAMS=8

The scorecard said otherwise:

    PERF_SCORECARD ... batch=1 ... TSU=11.42 TS=11.42

TSU == TS is the tell: aggregate throughput equals per-user throughput only at batch 1. The run's
total was reported as 11.42 tok/s against a true ~91.4 -- eightfold low.

THE CAUSE WAS TWO LISTS. The pipeline stores its batch as `self.B` and nothing else. The generated
perf test resolves that correctly, because its `_pipeline_batch` checks `B`. resolve_batch -- which
the ADAPTER asks, and which is therefore what the scorecard is built from -- stopped at "max_batch".
So one reader saw 8, the other saw nothing and fell back to 1.

Not the first time: resolve_batch's own docstring records the same bug from a hardcoded `batch=1`
("under-reported eightfold"). It came back because the property has two readers and only one was
fixed. This one is the authority; the test's copy is written per-model by an agent and cannot be
relied on to agree.

AND THE FALLBACK MUST BE AUDIBLE. `1` is not a missing value, it is a plausible measurement, so a
defaulted batch reads as a single-user run and nothing in the log contradicts it. A pipeline that
declares none of the known names now says so.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


class _OnlyB:
    """Voxtral's shape: DECODE_BATCH stored as `B`, and no other batch attribute."""

    def __init__(self, b=8):
        self.B = b


class _Silent:
    """Declares nothing at all."""


def test_a_pipeline_that_only_declares_B_is_read_correctly():
    """THE BUG. 8 users, reported as 1."""
    from agent.perf_adapter import resolve_batch

    assert resolve_batch(_OnlyB(8)) == 8


def test_every_name_the_other_reader_knows_is_known_here_too():
    """The two lists drifting is the whole defect; this one must be a superset."""
    src = (_PA / "agent" / "perf_adapter.py").read_text()
    i = src.index("def resolve_batch(")
    body = src[i : src.index("\nclass ", i)]
    for name in ("max_batch_size", "batch_size", "batch", "max_batch", "B"):
        assert '"%s"' % name in body, "resolve_batch does not know %r" % name


def test_an_explicit_request_still_wins():
    """A batch sweep must be possible without rebuilding the demo."""
    from agent.perf_adapter import resolve_batch

    assert resolve_batch(_OnlyB(8), 2) == 2


def test_a_pipeline_declaring_nothing_is_still_batch_one():
    """Unchanged behaviour -- the default is right, it just has to be audible."""
    from agent.perf_adapter import resolve_batch

    assert resolve_batch(_Silent()) == 1


def test_the_sentinel_means_ask_not_one():
    """0 is 'ask the pipeline'. Collapsing it to 1 before the pipeline exists is what hid this."""
    from agent.perf_adapter import resolve_batch

    assert resolve_batch(_OnlyB(8), 0) == 8


def test_an_unresolved_batch_announces_itself(capsys):
    """`1` looks like a measurement, not a missing value, so it must say which it is."""
    from agent.perf_adapter import PipelineStageAdapter

    a = PipelineStageAdapter(lambda _d: _Silent(), None, batch=0)
    try:
        a.setup(None)
    except Exception:  # noqa: BLE001 -- only the batch resolution matters here
        pass
    assert "PERF_BATCH_UNRESOLVED" in capsys.readouterr().out


def test_a_resolved_batch_stays_quiet(capsys):
    from agent.perf_adapter import PipelineStageAdapter

    a = PipelineStageAdapter(lambda _d: _OnlyB(8), None, batch=0)
    try:
        a.setup(None)
    except Exception:  # noqa: BLE001
        pass
    assert "PERF_BATCH_UNRESOLVED" not in capsys.readouterr().out


def test_the_adapter_resolves_after_the_build_not_before():
    """__init__ cannot know: the pipeline does not exist yet. Resolution belongs after the build."""
    src = (_PA / "agent" / "perf_adapter.py").read_text()
    # Scoped to PipelineStageAdapter: PipelineDecodeAdapter has its own build site earlier in the
    # file, and searching from the top finds that one instead.
    c = src.index("class PipelineStageAdapter")
    i = src.index("self._pipe = self._build(device)", c)
    assert "resolve_batch(" in src[i : i + 200]
    assert "self._requested_batch" in src[i : i + 200]
