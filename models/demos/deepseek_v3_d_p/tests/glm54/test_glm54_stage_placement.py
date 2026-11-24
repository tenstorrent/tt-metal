# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: GLM-5.4's four-stage placement (q171.2.8), without a device.

* The runner splits the 106 layer passes, and four ranks land on the approved stages G0 L1-26, G1 L27-54 pass 1,
  G2 L27-54 pass 2, G3 L55-78 (with and without the MTP tail); a split cutting a loop pass is rejected. The
  schedule's generic contracts (rank slices, ack ids, the ack layout, no KV table) are
  common/prefill/tests/test_pass_schedule.py's.
* G2 builds its loop layers 'shared' (no indexer, no index cache) and injects pass-1 layer L's own top-k set (A5).
* The glm_5_4 adapter reads GLM-5.3's TTNN cache directory and never creates one of its own.
* TtPrefillTransformer runs exactly its stage's steps, gates and ack ids.
"""

import pytest

from models.demos.common.prefill.runners.pass_schedule import PassSchedule, RankSlice
from models.demos.common.prefill.runners.runner_utils import compute_layer_split
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import LoopStep
from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import (
    loop_count,
    loop_schedule,
    pass_schedule,
    runs_later_loop_pass,
    runs_only_later_loop_passes,
    stage_config,
    stage_indexer_types,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.two_bank import pass1_index_set
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_4 import GLM54Adapter

G0, G1, G2, G3 = (0, 26), (26, 28), (54, 28), (82, 24)
STAGES = [G0, G1, G2, G3]


@pytest.fixture
def schedule() -> PassSchedule:
    return pass_schedule(glm_5_4_hf_config())


def test_pass_schedule_projects_the_loop_schedule(schedule):
    """The runner's positions are the reference loop_schedule's layer steps (its owner, reference/glm_5_4/tests,
    pins the 106 passes); the projection adds the layer count and the pass tags."""
    config = glm_5_4_hf_config()
    steps = loop_schedule(config.num_hidden_layers, config.loop_first_layer, config.loop_last_layer, 2)
    assert schedule.num_layers == 78
    assert schedule.passes == tuple((s.layer, s.loop_pass) for s in steps if s.kind == "layer")


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


def test_a_cut_inside_a_loop_pass_is_rejected_by_the_split(schedule, monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "30,24,28,24")
    with expect_error(ValueError, "not a valid boundary"):
        compute_layer_split(106, 4, GLM54Adapter().layer_split_boundaries(106))


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

    def roles(loop_passes, gates):
        t = tpt.TtPrefillTransformer.__new__(tpt.TtPrefillTransformer)
        t._loop_passes, t.loop_gates, t._loop_count = loop_passes, gates, 2
        return t.sends_loop_handoff, t.receives_loop_handoff

    assert roles((1,), {1: None}) == (True, False)  # G1
    assert roles((2,), {2: None}) == (False, True)  # G2
    assert roles(None, {1: None, 2: None}) == (False, False)  # whole loop on one rank
    assert roles((), {}) == (False, False)  # G0, G3


@pytest.mark.parametrize(
    "stage, loop_passes, ring",
    [(G0, (), False), (G1, (1,), False), (G2, (2,), True), (G3, (), False), ((0, 106), (1, 2), True)],
    ids=["G0", "G1", "G2", "G3", "whole_trunk"],
)
def test_allocated_slot_rows_per_stage(schedule, stage, loop_passes, ring):
    """The KVPE slot the adapter allocates per stage (kv_cache_slot_rows, which derives it from the model's
    stage_kv_cache_seq_len): max_seq_len, bank-aligned for migration, except on a stage that runs loop pass 2 (G2,
    the whole trunk), whose slots also hold pass 2's 4-chunk window ring and are never migrated."""
    from types import SimpleNamespace

    max_seq, chunk = 1064960, 5120
    rank = schedule.rank_slice(*stage)
    assert rank.loop_passes == loop_passes
    params = SimpleNamespace(
        first_layer_idx=rank.first_layer,
        num_layers=rank.num_layers,
        loop_passes=rank.loop_passes,
        max_seq_len=max_seq,
        chunk_size=chunk,
    )
    expected = (max_seq + 4 * chunk, False) if ring else (max_seq, True)
    assert GLM54Adapter().kv_cache_slot_rows(glm_5_4_hf_config(), params) == expected


def test_glm53_stage_slot_length_is_max_seq_len():
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import stage_kv_cache_seq_len

    config = glm_5_3_hf_config()
    assert [stage_kv_cache_seq_len(config, first, n, None, 1064960, 5120) for first, n in [(0, 78), (54, 24)]] == [
        1064960
    ] * 2


def test_stage_loop_predicates():
    """One decision per fact, shared by the model and the cache allocator (None: every pass, the whole loop)."""
    assert [runs_later_loop_pass(p) for p in (None, (1, 2), (1,), (2,), ())] == [True, True, False, True, False]
    assert [runs_only_later_loop_passes(p) for p in (None, (1, 2), (1,), (2,), ())] == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_loop_count_and_stage_config():
    """One reading of the loop count (0 without a loop) and one stage config: the model's own except on G2, which
    gets a copy with its stage map and leaves the model's config untouched."""
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config

    config = glm_5_4_hf_config()
    assert loop_count(config) == 2 and loop_count(glm_5_3_hf_config()) == 0
    model_map = list(config.indexer_types)
    for passes in (None, (), (1,), (1, 2)):
        assert stage_config(config, passes) is config
    g2 = stage_config(config, (2,))
    assert g2 is not config and g2.indexer_types == stage_indexer_types(config, (2,))
    assert config.indexer_types == model_map
