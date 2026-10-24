# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence parallelism in the engine: the rule, the route, the export, the seam, the marker, the facts.

The planner (scripts/tt_hw_planner) and the engine's route decision share ONE rule for what the chips
TP leaves over do -- replicas for the requests the run states, token groups for the rest -- and it lives
here, in agent.tp. Everything downstream only READS a degree: the pipeline reads the export, the stage
walk reads the pipeline's seam, the marker names it, the facts parser and the ceilings divide by it. A
run that states no workload, or a pipeline that states no split, reads 1 everywhere and is unchanged.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

from agent import perf_adapter as PA
from agent import stage_marks, stage_seams
from agent.tp import TILE, decide_parallelism, seq_parallel_legal, split_spare_chips

_PA = Path(__file__).resolve().parent.parent
CAP = 100


# --------------------------------------------------------------------------------------------
# 1. the rule
# --------------------------------------------------------------------------------------------


def test_a_slice_is_legal_only_when_it_is_whole_tiles():
    assert seq_parallel_legal(1024, 4) and seq_parallel_legal(1024, 1) and seq_parallel_legal(4 * TILE, 4)
    assert not seq_parallel_legal(96, 4), "24 tokens a group is not a tile"
    assert not seq_parallel_legal(1024, 3), "does not divide"
    assert not seq_parallel_legal(0, 2) and not seq_parallel_legal(1024, 0)
    assert not seq_parallel_legal("x", 2) and not seq_parallel_legal(None, None)


def test_the_spare_chips_go_to_requests_first_then_to_tokens():
    assert split_spare_chips(4, 1, 1024) == (1, 4)
    assert split_spare_chips(4, 4, 1024) == (4, 1)
    assert split_spare_chips(8, 2, 1024) == (2, 4)
    assert split_spare_chips(8, 3, 1024) == (2, 4), "replicas are a divisor of the chips, never more than the requests"
    assert split_spare_chips(4, 1, 64) == (2, 2), "the largest legal split; what it leaves are replicas"
    assert split_spare_chips(4, 1, 96) == (4, 1), "no legal split -> the old answer"
    assert split_spare_chips(1, 1, 1024) == (1, 1)


def test_an_unstated_workload_keeps_every_spare_chip_a_replica():
    assert split_spare_chips(4) == (4, 1)
    assert split_spare_chips(4, None, 1024) == (4, 1)
    assert split_spare_chips(4, 1, None) == (4, 1)
    assert split_spare_chips(4, "x", "y") == (4, 1)
    assert split_spare_chips(0) == (1, 1)


# --------------------------------------------------------------------------------------------
# 2. the route
# --------------------------------------------------------------------------------------------


def test_a_planned_token_split_is_the_route():
    """The planner decided (split_spare_chips) and exported; the route reports that, not a second guess."""
    r = decide_parallelism(50, CAP, 4, 16, 2048, metric="fps", sp=4)
    assert r["route"] == "single-chip+seq-parallel"
    assert (r["tp"], r["dp"], r["sp"], r["tp_regime"], r["floor"]) == (1, 1, 4, False, 1)
    assert "SP=4" in r["reason"]
    r = decide_parallelism(50, CAP, 8, 16, 2048, metric="fps", sp=2)
    assert (r["dp"], r["sp"]) == (4, 2), "what the token groups leave are replicas"


def test_the_token_split_outranks_the_per_matmul_tp_sweep_for_the_same_chips():
    """Both want the spare chips; one decision is made about them, not two competing ones."""
    r = decide_parallelism(50, CAP, 4, 16, 2048, metric="device_ms", sp=4)
    assert r["route"] == "single-chip+seq-parallel" and r["tp_regime"] is False
    assert decide_parallelism(50, CAP, 4, 16, 2048, metric="device_ms")["route"] == "single-chip+tp-latency"


def test_every_old_route_reads_a_degree_of_one():
    for args in ((50, CAP, 4, 16, 2048, "fps"), (50, CAP, 4, 16, 2048, "device_ms"), (200, CAP, 4, 16, 2048)):
        assert decide_parallelism(*args).get("sp", 1) == 1
    assert decide_parallelism(50, CAP, 1, 16, 2048, sp=1)["route"] == "single-chip"
    assert decide_parallelism(640, CAP, 4, 16, 2048, sp=4)["route"] == "infeasible"


def test_a_degree_that_does_not_fit_the_mesh_is_ignored():
    """An export the chip count cannot honour (3 groups on 4 chips, or garbage) falls back to the old route."""
    assert decide_parallelism(50, CAP, 4, 16, 2048, metric="fps", sp=3)["route"] == "single-chip"
    assert decide_parallelism(50, CAP, 4, 16, 2048, metric="fps", sp="x")["route"] == "single-chip"
    assert decide_parallelism(50, CAP, 4, 16, 2048, metric="fps", sp=None)["sp"] == 1


def test_a_model_that_does_not_fit_is_not_split_by_tokens():
    """TP is the only way to make a too-big model fit; the token split is for the chips TP leaves over."""
    r = decide_parallelism(200, CAP, 4, 16, 2048, sp=4)
    assert r["route"] == "tensor-parallel" and r["tp"] == 4 and r["sp"] == 1


# --------------------------------------------------------------------------------------------
# 3. the export the pipeline reads
# --------------------------------------------------------------------------------------------


def test_the_pipeline_reads_the_planned_degree_never_the_mesh(monkeypatch, capsys):
    monkeypatch.delenv(PA.SEQ_PARALLEL_ENV, raising=False)
    assert PA.resolve_seq_parallel() == 1
    assert PA.resolve_seq_parallel(default_sp=2) == 2
    monkeypatch.setenv(PA.SEQ_PARALLEL_ENV, "4")
    assert PA.resolve_seq_parallel() == 4
    monkeypatch.setenv(PA.SEQ_PARALLEL_ENV, "garbage")
    assert PA.resolve_seq_parallel() == 1
    assert PA.SEQ_PARALLEL_ENV in capsys.readouterr().err, "an unparseable setting is said out loud"
    monkeypatch.setenv(PA.SEQ_PARALLEL_ENV, "0")
    assert PA.resolve_seq_parallel() == 1


def test_the_stated_workload_is_two_positive_counts_or_nothing(monkeypatch):
    monkeypatch.delenv(PA.BATCH_ENV, raising=False)
    monkeypatch.delenv(PA.SEQ_LEN_ENV, raising=False)
    assert PA.stated_workload() == (None, None)
    monkeypatch.setenv(PA.BATCH_ENV, "2")
    monkeypatch.setenv(PA.SEQ_LEN_ENV, "1024")
    assert PA.stated_workload() == (2, 1024)
    monkeypatch.setenv(PA.BATCH_ENV, "x")
    monkeypatch.setenv(PA.SEQ_LEN_ENV, "0")
    assert PA.stated_workload() == (None, None)


# --------------------------------------------------------------------------------------------
# 4. the seam, the pipeline's own degree, the stage
# --------------------------------------------------------------------------------------------


def test_the_seam_is_optional_and_the_stage_carries_it():
    assert stage_seams.SEQ_SPLIT in stage_seams.OPTIONAL and stage_seams.SEQ_SPLIT in stage_seams.ALL
    assert stage_seams.SEQ_SPLIT not in stage_seams.REQUIRED
    assert PA._Stage("alpha", None, seq_split=4).seq_split == 4
    assert PA._Stage("alpha", None).seq_split == 0, "unstated stays unstated"
    assert PA._Stage("alpha", None, split=2).split == 2, "the data-parallel split is untouched"


def test_the_pipeline_s_own_degrees_are_read_the_same_way():
    assert stage_marks.pipeline_sp(types.SimpleNamespace(sp=4)) == 4
    assert stage_marks.pipeline_sp(types.SimpleNamespace()) == 0
    assert stage_marks.pipeline_sp(types.SimpleNamespace(sp="x")) == 0
    assert stage_marks.pipeline_sp(types.SimpleNamespace(sp=0)) == 0
    assert stage_marks.pipeline_tp(types.SimpleNamespace(tp=8)) == 8 and stage_marks.pipeline_tp(object()) == 0


def test_the_stage_walk_binds_the_seam_from_the_pipeline():
    """The adapter reads <stage>_trace_seq_split off the pipeline exactly as it reads _trace_split."""
    src = (_PA / "agent" / "perf_adapter.py").read_text()
    assert "seq_split=_stated_count(p, name, _seams.SEQ_SPLIT)" in src


# --------------------------------------------------------------------------------------------
# 5. the marker, the scorecard, the ledger, the ceilings
# --------------------------------------------------------------------------------------------


def test_the_marker_names_the_degree_and_the_scorecard_pins_the_stage_split():
    replay = (_PA / "agent" / "trace_replay.py").read_text()
    assert "DP=%d TP=%d SP=%d shard_active=%s" in replay
    assert "TRACE_STAGE_SEQ_SPLIT[%s]=%d" in replay
    mcp = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert '("TRACE_STAGE_SEQ_SPLIT[", stage_seq_split)' in mcp
    assert "_ledger().KIND_STAGE_SEQ_SPLIT, stage_seq_split" in mcp
    from cc_optimize import measurements, summary

    assert summary._LED_SEQ_SPLIT == measurements.KIND_STAGE_SEQ_SPLIT


def test_the_facts_parser_reads_the_degree_and_defaults_it_to_one():
    spec = importlib.util.spec_from_file_location("cc_run_sp_under_test", _PA / "cc_optimize" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    facts = run._parse_facts("DP=1 TP=1 SP=4 shard_active=True", {"ttnn.matmul(x)"})
    assert (facts["dp"], facts["tp"], facts["sp"], facts["shard_active"], facts["parallelism_known"]) == (
        1,
        1,
        4,
        True,
        True,
    )
    old = run._parse_facts("[full-pipeline-gate] PERF_SCORECARD mesh=1x4 TP=4 DP=1 shard=True", set())
    assert (old["tp"], old["dp"], old["sp"]) == (4, 1, 1)
