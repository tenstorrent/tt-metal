# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: GLM-5.4's four-stage placement (q171.2.8), without a device.

* The runner splits the 106 layer passes, and four ranks land on the approved stages G0 L1-26, G1 L27-54 pass 1,
  G2 L27-54 pass 2, G3 L55-78 (with and without the MTP tail); no rank starts inside a loop pass.
* Layer acks: every pass acks once per chunk; G2's ids (104-131) collide with no other rank's id and no MTP row, and
  G2 migrates no layer (its KVPE is a copy of G1's).
* G2 builds its loop layers 'shared' (no indexer, no index cache) and injects pass-1 layer L's own top-k set (A5).
* The glm_5_4 adapter reads GLM-5.3's TTNN cache directory and never creates one of its own.
* TtPrefillTransformer runs exactly its stage's steps, gates and ack ids.
"""

import pytest

from models.demos.common.prefill.runners.layer_completion_layout import layer_completion_layout
from models.demos.common.prefill.runners.pass_schedule import PassSchedule, RankSlice
from models.demos.common.prefill.runners.runner_utils import compute_layer_split
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import LoopStep
from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import pass_schedule, stage_indexer_types
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.two_bank import pass1_index_set
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_4 import GLM54Adapter

G0, G1, G2, G3 = (0, 26), (26, 28), (54, 28), (82, 24)
STAGES = [G0, G1, G2, G3]


@pytest.fixture
def schedule() -> PassSchedule:
    return pass_schedule(glm_5_4_hf_config())


def test_106_pass_schedule(schedule):
    assert len(schedule) == 106 and schedule.num_layers == 78
    assert schedule.passes[:26] == tuple((layer, 0) for layer in range(26))
    assert schedule.passes[26:54] == tuple((layer, 1) for layer in range(26, 54))
    assert schedule.passes[54:82] == tuple((layer, 2) for layer in range(26, 54))
    assert schedule.passes[82:] == tuple((layer, 0) for layer in range(54, 78))


def test_rank_starts_exclude_the_inside_of_a_loop_pass(schedule):
    starts = schedule.valid_starts()
    assert set(range(27, 54)).isdisjoint(starts) and set(range(55, 82)).isdisjoint(starts)
    assert {0, 25, 26, 54, 82, 100} <= starts


@pytest.mark.parametrize("mtp_levels", [0, 1], ids=["trunk", "mtp1"])
def test_four_ranks_land_on_the_approved_stages(schedule, mtp_levels):
    adapter = GLM54Adapter()
    split = compute_layer_split(106, 4, adapter.layer_split_boundaries(106), mtp_levels)
    assert split == STAGES
    assert [schedule.rank_slice(f, n) for f, n in split] == [
        RankSlice(0, 26, ()),
        RankSlice(26, 28, (1,)),
        RankSlice(26, 28, (2,)),
        RankSlice(54, 24, ()),
    ]


def test_one_rank_runs_the_whole_trunk(schedule):
    assert compute_layer_split(106, 1, GLM54Adapter().layer_split_boundaries(106)) == [(0, 106)]
    assert schedule.rank_slice(0, 106) == RankSlice(0, 78, (1, 2))


@pytest.mark.parametrize("first, n", [(20, 20), (30, 40), (60, 30)], ids=["into-pass1", "pass1-to-pass2", "into-pass2"])
def test_a_rank_cutting_a_loop_pass_is_rejected(schedule, first, n, expect_error):
    with expect_error(ValueError, "a rank holds whole loop passes"):
        schedule.rank_slice(first, n)


def test_a_cut_inside_a_loop_pass_is_rejected_by_the_split(schedule, monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "30,24,28,24")
    with expect_error(ValueError, "not a valid boundary"):
        compute_layer_split(106, 4, GLM54Adapter().layer_split_boundaries(106))


def test_mixed_stages_hold_prelude_or_coda_layers_around_a_pass(schedule):
    assert schedule.rank_slice(20, 34) == RankSlice(20, 34, (1,))  # L21-26 + pass 1
    assert schedule.rank_slice(54, 36) == RankSlice(26, 36, (2,))  # pass 2 + L55-62


def test_ack_ids_are_distinct_and_g2_never_acks_a_pass1_layer(schedule):
    ids = {name: schedule.ack_ids(*stage) for name, stage in zip("G0 G1 G2 G3".split(), STAGES)}
    assert ids["G0"] == list(range(0, 26))
    assert ids["G1"] == list(range(26, 54))
    assert ids["G2"] == list(range(104, 132))
    assert ids["G3"] == list(range(54, 78))
    flat = [i for stage in ids.values() for i in stage]
    assert len(flat) == len(set(flat)) == 106
    assert 78 not in flat  # MTP's row


@pytest.mark.parametrize("mtp_levels", [0, 1], ids=["trunk", "mtp1"])
def test_layer_completion_layout_per_stage(schedule, mtp_levels):
    layouts = [layer_completion_layout(STAGES, r, extra_layers=mtp_levels, pass_schedule=schedule) for r in range(4)]
    assert {lay.completions_per_chunk for lay in layouts} == {106 + mtp_levels}
    assert [lay.rank_first_record for lay in layouts] == [0, 26, 54, 82]
    assert [lay.rank_record_count for lay in layouts] == [26, 28, 28, 24 + mtp_levels]
    assert layouts[2].rank_layer_ids == list(range(104, 132))
    assert layouts[3].rank_layer_ids == list(range(54, 78)) + [78] * mtp_levels
    # Every record of a chunk has its own id.
    all_ids = [i for lay in layouts for i in lay.rank_layer_ids]
    assert len(all_ids) == len(set(all_ids)) == 106 + mtp_levels


def test_stage_alone_probe_acks_under_its_real_ids(schedule):
    """The runner shifts a PREFILL_PP_LAYER_OFFSET probe's one-rank split by the offset: G2 alone acks 104-131."""
    layout = layer_completion_layout([(54, 28)], 0, pass_schedule=schedule)
    assert layout.completions_per_chunk == 28 and layout.rank_layer_ids == list(range(104, 132))


def test_mtp_rows_must_not_reach_the_loop_ack_ids(schedule, expect_error):
    with expect_error(AssertionError, "collide with loop pass ack ids"):
        layer_completion_layout(STAGES, 3, extra_layers=27, pass_schedule=schedule)


def test_g2_migrates_no_layer(schedule):
    assert schedule.migrated_layers(*G1) == list(range(26, 54))
    assert schedule.migrated_layers(*G2) == []
    assert schedule.migrated_layers(0, 106) == list(range(78))


def test_g2_runs_no_indexer_and_injects_layer_ls_own_set():
    config = glm_5_4_hf_config()
    g2 = stage_indexer_types(config, (2,))
    assert all(t == "shared" for t in g2[26:54])
    assert g2[:26] == config.indexer_types[:26] and g2[54:] == config.indexer_types[54:]
    assert stage_indexer_types(config, (1,)) == config.indexer_types
    assert stage_indexer_types(config, (1, 2)) == config.indexer_types
    # Pass 1 makes 7 top-k sets, reuse groups of 4 from layer 26: layer L takes the set of its group's full layer.
    assert pass1_index_set(config, 53) + 1 == 7
    assert [pass1_index_set(config, layer) for layer in range(26, 54)] == [g for g in range(7) for _ in range(4)]


def test_adapter_reads_glm53_cache_dir_and_creates_none(tmp_path, monkeypatch):
    import models.demos.deepseek_v3_d_p.tt.runners.adapters.mla as mla_adapter

    glm53_dir = tmp_path / "glm_5_3_bh_32dev" / "8x4"
    glm53_dir.mkdir(parents=True)
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    monkeypatch.setattr(mla_adapter, "is_blackhole", lambda: True)
    monkeypatch.setattr(mla_adapter.ttnn, "get_num_devices", lambda: 32)
    adapter = GLM54Adapter()
    assert adapter.name == "glm_5_4"
    assert adapter.weight_cache_path((8, 4)) == glm53_dir
    assert sorted(p.name for p in tmp_path.iterdir()) == ["glm_5_3_bh_32dev"]
    assert not adapter.builds_missing_weight_cache  # an incomplete cache never rebuilds into GLM-5.3's
    assert adapter.ttnn_cache_default.startswith("/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache")


def test_adapter_gates_are_the_synthetic_seed_gates():
    from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import GLM54_SYNTHETIC_SEED, synthetic_gates

    config = glm_5_4_hf_config()
    gates = GLM54Adapter().loop_gate_weights(config)
    expected = synthetic_gates(config.hidden_size, 2, seed=GLM54_SYNTHETIC_SEED)
    assert len(gates) == 2
    for got, want in zip(gates, expected):
        assert all((got[k] == want[k]).all() for k in want)


# --- TtPrefillTransformer at each stage (host: the gates are stand-ins, no blocks are built) ---


class _Gate:
    width = 1

    def __init__(self, mesh_device, weights, **_):
        self.weights = weights


def _stage(monkeypatch, first, n, loop_passes, **kw):
    import models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer as tpt

    monkeypatch.setattr(tpt, "TtDampGate", _Gate)
    t = tpt.TtPrefillTransformer.__new__(tpt.TtPrefillTransformer)
    t.first_layer_idx, t.num_layers, t.kv_only_last_layer = first, n, kw.get("kv_only_last_layer", False)
    t._loop_passes = loop_passes
    t.mesh_device = None
    steps, gates = t._build_schedule(glm_5_4_hf_config(), {"loop_gates": ["gate1", "gate2"]}, 1, None, 1)
    return steps, {k: g.weights for k, g in gates.items()}


def test_transformer_runs_each_stage_and_its_gate(monkeypatch):
    steps, gates = _stage(monkeypatch, 26, 28, (1,))
    assert steps == [LoopStep("layer", 1, i) for i in range(26, 54)] + [LoopStep("gate", 1)]
    assert gates == {1: "gate1"}
    steps, gates = _stage(monkeypatch, 26, 28, (2,))
    assert steps == [LoopStep("layer", 2, i) for i in range(26, 54)] + [LoopStep("gate", 2)]
    assert gates == {2: "gate2"}
    steps, gates = _stage(monkeypatch, 0, 78, None)
    assert len(steps) == 108 and gates == {1: "gate1", 2: "gate2"}  # 106 passes + 2 gates
    steps, gates = _stage(monkeypatch, 26, 36, (2,))
    assert [s for s in steps if s.kind == "layer" and s.loop_pass == 0] == [
        LoopStep("layer", 0, i) for i in range(54, 62)
    ]


def test_transformer_handoff_roles():
    import models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer as tpt

    def roles(gates):
        t = tpt.TtPrefillTransformer.__new__(tpt.TtPrefillTransformer)
        t.loop_gates, t._loop_count = gates, 2
        return t.sends_loop_handoff, t.receives_loop_handoff

    assert roles({1: None}) == (True, False)  # G1
    assert roles({2: None}) == (False, True)  # G2
    assert roles({1: None, 2: None}) == (False, False)  # whole loop on one rank
    assert roles({}) == (False, False)  # G0, G3
