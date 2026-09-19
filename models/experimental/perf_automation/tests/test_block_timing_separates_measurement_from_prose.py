"""Block-level timing shows what was MEASURED apart from what the agent wrote.

The section had one source: `stages_json`, a JSON list the AGENT passes to record_kernel_attempt.
Nothing validates it. On gemma-3-12b-it the agent wrote fourteen rows and put a phase prefix on ten
of them:

    decode MLP ff1+ff3    77.31    <- phase in the name
    LayerNorm             11.79    <- no phase; runs in BOTH
    LM head               10.85    <- no phase; decode-only

So a phase split derived from those strings would be guesswork, and guessing wrong matters here:
decode ms recurs on every token and sets tok/s/u, prefill ms happens once and sets TTFT. Putting
decode time in a prefill bucket invites optimizing the wrong phase -- the exact failure this model
already spent a run on.

Meanwhile the harness ALREADY measures the phases. trace_replay prints TRACE_STAGE_MS[<stage>] per
stage, derived from the PIPELINE_STAGES the MODEL declares (gemma3: ["prefill", "decode"]), and
perf_mcp threw those lines away -- it parsed only TRACE_PER_TOKEN_MS. They are now parsed, persisted
per (model, task), and rendered as their own block.

Two blocks, not one merged list, because they do not have the same standing: one is measurement, the
other is the agent's annotation. The annotation stays, because it is the more useful view for finding
hot spots -- it just stops being presentable as measurement.

The annotation block carries its own total and within-block percentages. The MEASURED block carries
neither: its rows are not in one currency -- prefill is per request, decode per token -- so summing
them describes nothing that happens. That sum read prefill as 71% of the work, when at OSL 128 a
request spends 128 decode steps and prefill is nearer 2%.

Also replaces the '#'/'.' bars with the block characters used by the utilization section, and pads
the label column -- the old rendering left the ms values unaligned because names longer than the
12-char field pushed the column out.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cc_optimize"))

import summary as S  # noqa: E402

AGENT = [
    {"name": "decode MLP ff1+ff3", "ms": 77.31, "dominant": True},
    {"name": "decode MLP ff2", "ms": 28.07},
    {"name": "prefill MLP ff1+ff3", "ms": 18.77},
    {"name": "LayerNorm", "ms": 11.79},
]


@pytest.fixture()
def measured(tmp_path, monkeypatch):
    """A persisted trace_replay stage file, as the gate now writes."""
    import json

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-under-test")
    # STAMPED, as the gate now writes it. An unstamped document is refused -- see
    # test_stage_ms_belongs_to_a_run -- because a measurement with no provenance is not one.
    (tmp_path / "perf_mcp_stage_ms_gemma3_main.json").write_text(
        json.dumps({"run": "run-under-test", "stages": {"prefill": 35.80, "decode": 138.49}})
    )
    return tmp_path


def _txt(stages=AGENT, model="gemma3", task="main"):
    return "\n".join(S._stage_table_lines(stages, model, task))


# ---------------------------------------------------------------- the two blocks are separate


def test_measured_and_annotation_are_distinct_blocks(measured):
    t = _txt()
    assert "measured by trace_replay" in t
    assert "agent breakdown (annotation, not measurement)" in t
    assert t.index("measured by trace_replay") < t.index("agent breakdown")


def test_the_measured_block_carries_the_declared_phases(measured):
    t = _txt()
    head = t[: t.index("agent breakdown")]
    assert "decode" in head and "prefill" in head


def test_the_annotation_block_is_labelled_as_such(measured):
    """The whole point: the agent's prose must not read as measurement."""
    t = _txt()
    assert "annotation, not measurement" in t


def test_the_measured_block_states_no_total(measured):
    """ITS ROWS ARE NOT IN ONE CURRENCY, so their sum describes nothing that happens.

    prefill is per REQUEST and decode per TOKEN. Summing them read prefill as 71% of the work; at
    OSL 128 a request spends 128 decode steps, so prefill is nearer 2%. A total nobody can act on,
    and percentages derived from it, are worse than no total -- each stage states its own ms."""
    t = _txt()
    assert "%.2f ms" % (35.80 + 138.49) not in t, "measured rows must not be summed"
    head = t.split("agent breakdown", 1)[0]
    assert "%" not in head, head  # no share, because there is no meaningful denominator
    assert "35.80 ms" in head and "138.49 ms" in head


def test_the_annotation_block_keeps_its_total(measured):
    """The agent's rows ARE one currency -- its own block timings -- so a share is meaningful."""
    assert "%.2f ms" % sum(x["ms"] for x in AGENT) in _txt(), "annotation total missing"


# ---------------------------------------------------------------- degradation


def test_no_measured_stages_leaves_only_the_annotation(tmp_path, monkeypatch):
    """A pipeline that declares no PIPELINE_STAGES: the report keeps working, minus the top block."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    t = _txt()
    assert "measured by trace_replay" not in t
    assert "decode MLP ff1+ff3" in t


def test_no_agent_stages_leaves_only_the_measurement(measured):
    t = _txt(stages=[])
    assert "measured by trace_replay" in t and "agent breakdown" not in t


def test_neither_renders_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    assert S._stage_table_lines([], "gemma3", "main") == []


def test_zero_and_malformed_rows_are_dropped(measured):
    t = _txt(stages=[{"name": "x", "ms": 0}, {"ms": 5.0}, "not a dict", {"name": "y", "ms": 3.0}])
    assert "y" in t and "\n    x " not in t


# ---------------------------------------------------------------- the bars


def test_bars_use_block_characters_not_hashes(measured):
    t = _txt()
    assert "█" in t and "#" not in t


def test_the_largest_row_fills_its_bar(measured):
    t = _txt()
    row = next(l for l in t.splitlines() if "decode MLP ff1+ff3" in l)
    assert row.count("█") == S._BAR_W


def test_percentages_are_within_the_block(measured):
    """A row's % is of ITS OWN block's total, not of the other block's."""
    t = _txt()
    own = 100.0 * 77.31 / sum(x["ms"] for x in AGENT)
    cross = 100.0 * 77.31 / (35.80 + 138.49)
    assert "%.1f%%" % own in t
    assert "%.1f%%" % cross not in t, "percentage computed against the wrong total"


def test_no_block_carries_a_hottest_marker(measured):
    """IT MARKED THE ONE TABLE THE REPORT SAYS NOT TO TRUST.

    The measured block was denied the marker and the annotation block got it, so the single word a
    reader acts on sat beside numbers labelled `not measurement` -- and 19% off the profiler. The
    bars already show which row is largest, and the optimizer ranks targets from the profile's
    gap_ms, never from this table."""
    assert "hottest" not in _txt()


# ---------------------------------------------------------------- alignment


def test_the_ms_column_lines_up(measured):
    """The old rendering pushed the column out for any name longer than 12 chars."""
    rows = [l for l in _txt().splitlines() if " ms  " in l and l.startswith("    ")]
    assert len({l.index(" ms  ") for l in rows}) == 1, [l for l in rows]


def test_a_long_name_cannot_break_the_column(measured):
    t = _txt(stages=[{"name": "x" * 60, "ms": 5.0}, {"name": "short", "ms": 3.0}])
    rows = [l for l in t.splitlines() if " ms  " in l and l.startswith("    ")]
    assert len({l.index(" ms  ") for l in rows}) == 1


# ---------------------------------------------------------------- no device


def test_rendering_touches_no_device(measured):
    src = Path(S.__file__).read_text()
    i = src.index("def _stage_table_lines")
    body = src[i : src.index("\ndef ", i + 1)]
    for forbidden in ("tt-smi", "ttnn", "subprocess", "MeshDevice"):
        assert forbidden not in body, forbidden
