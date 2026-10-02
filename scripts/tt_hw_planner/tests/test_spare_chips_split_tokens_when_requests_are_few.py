# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chips TP leaves over split one request's TOKENS when there are fewer requests than spare chips.

THE CASE. Voxtral-4B TTS on a 4-chip QB2 fits on one chip, so the planner picked TP=1 and made the
other three chips data-parallel replicas (TP=1 x DP=4). A replica serves a request of its own; with one
request there is nothing for three of them to do, and that one request runs no faster than on a single
chip. The tool knew two ways to use a mesh -- cut the WEIGHTS (TP) or copy the model (DP) -- and neither
helps a single long request. The third way is to cut the request's TOKENS: sequence parallelism, each
chip group running an equal, tile-aligned slice of the same request and exchanging only attention K/V.

WHAT THESE PIN. The planner splits the spare chips by the workload the run states -- replicas for the
requests there are, token groups for the rest; a run that states no workload keeps every spare chip a
replica, byte for byte as before. The degree rides on the mesh rows (rows = DP x SP), is exported
beside the mesh pair under its own name, is named in the manifest only when in play, and reaches the
emitted model through the chip-placement recipe.
"""

from __future__ import annotations

import os
from argparse import Namespace

import pytest

from models.experimental.perf_automation.agent.perf_adapter import BATCH_ENV, SEQ_LEN_ENV, SEQ_PARALLEL_ENV
from scripts.tt_hw_planner import parallelism as par
from scripts.tt_hw_planner.commands import emit_e2e as E
from scripts.tt_hw_planner.commands import optimize as O
from scripts.tt_hw_planner.parallelism import ParallelConfig, select_parallelism, split_label


class _KR:
    def __init__(self, grid, blocked=()):
        self.tp_grid = list(grid)
        self._blocked = set(blocked)

    def has_blockers(self, tp):
        return tp in self._blocked


def _args(**kw):
    kw.setdefault("mesh", None)
    kw.setdefault("devices", "")
    kw.setdefault("target", "some/model")
    return Namespace(**kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("TT_PERF_MESH_ROWS", "TT_PERF_MESH_COLS", SEQ_PARALLEL_ENV, BATCH_ENV, SEQ_LEN_ENV):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------------------------
# 1. the selector: what the spare chips do
# --------------------------------------------------------------------------------------------


def test_without_a_stated_workload_every_spare_chip_is_a_replica():
    """No workload, no change: the old answer, with the new degree at 1 and absent from the label."""
    pc = select_parallelism(4, _KR([1, 2, 4], blocked={2, 4}))
    assert (pc.tp, pc.dp, pc.sp, pc.chips) == (1, 4, 1, 4)
    assert pc.label == "TP=1,DP=4", "the config label is as it was"
    # half a workload is no workload
    assert select_parallelism(4, _KR([1, 2, 4], blocked={2, 4}), requests=1).sp == 1
    assert select_parallelism(4, _KR([1, 2, 4], blocked={2, 4}), seq_len=1024).sp == 1


def test_one_request_on_four_spare_chips_splits_its_tokens():
    pc = select_parallelism(4, _KR([1, 2, 4], blocked={2, 4}), requests=1, seq_len=1024)
    assert (pc.tp, pc.dp, pc.sp, pc.chips) == (1, 1, 4, 4)
    assert pc.label == "TP=1,SP=4" and split_label(pc.tp, pc.dp, pc.sp) == "TP=1 x DP=1 x SP=4"


def test_requests_claim_replicas_first_and_the_tokens_take_the_rest():
    kr = _KR([1], blocked=())
    assert (
        select_parallelism(8, kr, requests=2, seq_len=1024).dp,
        select_parallelism(8, kr, requests=2, seq_len=1024).sp,
    ) == (2, 4)
    # three requests: the largest replica count that divides the chips and is not more than the requests
    assert (
        select_parallelism(8, kr, requests=3, seq_len=1024).dp,
        select_parallelism(8, kr, requests=3, seq_len=1024).sp,
    ) == (2, 4)
    # enough requests for every chip: all replicas, nothing to split
    assert (
        select_parallelism(8, kr, requests=8, seq_len=1024).dp,
        select_parallelism(8, kr, requests=8, seq_len=1024).sp,
    ) == (8, 1)


def test_a_sequence_that_does_not_cut_into_tiles_stays_replicated():
    kr = _KR([1])
    # 96 tokens: 96/4 = 24 and 96/2 = 48 are not tile multiples -> no legal split, replicas as before
    assert (
        select_parallelism(4, kr, requests=1, seq_len=96).dp,
        select_parallelism(4, kr, requests=1, seq_len=96).sp,
    ) == (4, 1)
    # 64 tokens: 64/4 = 16 is not, 64/2 = 32 is -> the largest LEGAL split, the leftover chip pair replicas
    assert (
        select_parallelism(4, kr, requests=1, seq_len=64).dp,
        select_parallelism(4, kr, requests=1, seq_len=64).sp,
    ) == (2, 2)
    # 128 tokens: 128/4 = 32 -> all four
    assert (
        select_parallelism(4, kr, requests=1, seq_len=128).dp,
        select_parallelism(4, kr, requests=1, seq_len=128).sp,
    ) == (1, 4)


def test_tp_is_still_decided_first_and_the_tokens_split_only_what_it_leaves():
    pc = select_parallelism(8, _KR([1, 2, 4, 8], blocked={8}), requests=1, seq_len=1024)
    assert (pc.tp, pc.dp, pc.sp, pc.chips) == (4, 1, 2, 8)
    assert pc.label == "TP=4,SP=2" and split_label(pc.tp, pc.dp, pc.sp) == "TP=4 x DP=1 x SP=2"


def test_the_label_names_the_degree_only_when_it_is_in_play():
    assert split_label(2, 2) == "TP=2 x DP=2"
    assert split_label(2, 2, 1) == "TP=2 x DP=2"
    assert split_label(1, 1, 4) == "TP=1 x DP=1 x SP=4"


# --------------------------------------------------------------------------------------------
# 2. the manifest and the export
# --------------------------------------------------------------------------------------------


def test_the_manifest_names_the_degree_only_when_it_is_in_play(tmp_path):
    p = par.write_parallelism_manifest(tmp_path, chips=4, tp=1, dp=4)
    assert par.read_parallelism_manifest(tmp_path) == {"chips": 4, "tp": 1, "dp": 4, "mesh": [4, 1]}, "byte-identical"
    assert p.exists()
    par.write_parallelism_manifest(tmp_path, chips=4, tp=1, dp=2, sp=2)
    m = par.read_parallelism_manifest(tmp_path)
    assert (m["dp"], m["sp"], m["mesh"]) == (2, 2, [4, 1]), "the rows carry the replicas AND the token groups"


def test_an_unstated_workload_asks_the_planner_exactly_as_before(monkeypatch):
    """The planner's two-argument contract holds for every caller that never states a workload."""
    calls = []

    def _plan(mid, chips):  # the old signature, on purpose: a keyword would raise here
        calls.append((mid, chips))
        return ParallelConfig(tp=1, dp=4)

    monkeypatch.setattr("scripts.tt_hw_planner.parallelism.plan_parallelism", _plan)
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir=None)
    assert calls == [("some/model", 4)]
    assert (os.environ["TT_PERF_MESH_ROWS"], os.environ["TT_PERF_MESH_COLS"]) == ("4", "1")
    assert SEQ_PARALLEL_ENV not in os.environ


def test_a_stated_workload_reaches_the_planner_and_the_degree_rides_on_the_rows(monkeypatch):
    seen = {}

    def _plan(mid, chips, requests=None, seq_len=None):
        seen.update(requests=requests, seq_len=seq_len)
        return ParallelConfig(tp=1, dp=1, sp=4)

    monkeypatch.setattr("scripts.tt_hw_planner.parallelism.plan_parallelism", _plan)
    monkeypatch.setenv(BATCH_ENV, "1")
    monkeypatch.setenv(SEQ_LEN_ENV, "1024")
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir=None)
    assert seen == {"requests": 1, "seq_len": 1024}
    assert (os.environ["TT_PERF_MESH_ROWS"], os.environ["TT_PERF_MESH_COLS"]) == ("4", "1"), "rows = DP x SP"
    assert os.environ[SEQ_PARALLEL_ENV] == "4"


def test_a_stale_degree_is_cleared_when_the_plan_has_none(monkeypatch):
    monkeypatch.setenv(SEQ_PARALLEL_ENV, "4")
    monkeypatch.setattr(
        "scripts.tt_hw_planner.parallelism.plan_parallelism", lambda mid, chips: ParallelConfig(tp=2, dp=2)
    )
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir=None)
    assert SEQ_PARALLEL_ENV not in os.environ
    monkeypatch.setenv(SEQ_PARALLEL_ENV, "4")
    O._derive_topology_env(_args(devices="single"), model_dir=None)
    assert SEQ_PARALLEL_ENV not in os.environ, "a single chip has nothing to split"


def test_an_explicit_mesh_shape_states_no_degree(monkeypatch):
    """--mesh names a topology, not a token split; the operator's rows are replicas unless a model says otherwise."""
    monkeypatch.setenv(SEQ_PARALLEL_ENV, "2")
    O._derive_topology_env(_args(mesh="2,4", devices="all"), model_dir=None)
    assert (os.environ["TT_PERF_MESH_ROWS"], os.environ["TT_PERF_MESH_COLS"]) == ("2", "4")
    assert SEQ_PARALLEL_ENV not in os.environ


# --------------------------------------------------------------------------------------------
# 3. what the emitted model is told
# --------------------------------------------------------------------------------------------


def test_the_placement_recipe_puts_the_token_groups_on_the_rows():
    block = E._parallelism_prompt_block(ParallelConfig(tp=1, dp=1, sp=4))
    assert "TP=1 x DP=1 x SP=4" in block
    assert "open_mesh_device(ttnn.MeshShape(4, 1))" in block, "rows = DP x SP"
    assert "SEQUENCE-PARALLEL" in block and "SP=4" in block
    for must in (
        "resolve_seq_parallel",
        "ShardTensor2dMesh",
        "all_gather",
        "cluster_axis=0",
        "_trace_seq_split",
        "self.sp",
    ):
        assert must in block, must


def test_a_split_without_token_groups_reads_exactly_as_before():
    block = E._parallelism_prompt_block(ParallelConfig(tp=2, dp=2))
    assert "TP=2 x DP=2" in block and "open_mesh_device(ttnn.MeshShape(2, 2))" in block
    assert "SEQUENCE-PARALLEL" not in block and "SP=" not in block


def test_the_topology_guard_sees_the_token_split():
    graduated = {"chips": 8, "tp": 4, "dp": 1, "sp": 2, "mesh": [2, 4]}
    assert E._topology_mismatch(graduated, ParallelConfig(tp=4, dp=1, sp=2), 8) is None
    msg = E._topology_mismatch(graduated, ParallelConfig(tp=4, dp=2), 8)
    assert msg and "TP=4 x DP=1 x SP=2" in msg and "TP=4 x DP=2" in msg and "--mesh 2x4" in msg
    # a manifest written before the degree existed is read as SP=1, so an old graduation still matches
    assert E._topology_mismatch({"chips": 8, "tp": 4, "dp": 2, "mesh": [2, 4]}, ParallelConfig(tp=4, dp=2), 8) is None


def test_the_memory_model_prices_one_group_s_slice(monkeypatch):
    """SP cuts the KV cache and the activations a group holds; the weights stay whole on every group."""
    import scripts.tt_hw_planner.parallelism as P

    class _Arch:
        num_key_value_heads = 8

    class _Model:
        arch = _Arch()

        def weights_bytes(self, dtype):
            return 1000

        def kv_cache_bytes(self, batch, seq, kvb):
            return 800

        def activation_bytes(self, batch, seq, dtype="bf16"):
            return 400

    whole = P.shard(_Model(), "bf16", 1, 1024, 2.0, ParallelConfig(tp=1, dp=1))
    quarter = P.shard(_Model(), "bf16", 1, 1024, 2.0, ParallelConfig(tp=1, dp=1, sp=4))
    assert quarter.weights_bytes == whole.weights_bytes
    assert quarter.kv_cache_bytes == whole.kv_cache_bytes // 4
    assert quarter.activation_bytes == whole.activation_bytes // 4
