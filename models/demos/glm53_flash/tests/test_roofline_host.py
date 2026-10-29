# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only sanity of the decode roofline model (``tt/roofline.py``; no device, no ttnn):

* bytes per expert match the ledger (26.7 MB per full bfp8 expert, 3.34 MB per hybrid chip slice);
* the per-layer and per-step floors are monotone in the users per row, and every step floor sits below every measured
  e2e step at its batch point (and below the 33 / 19 tok/s targets' budgets or says why);
* the CCL fits reproduce the report's Blackhole numbers; the classification lands where the profiles say;
* the ranked actions are consistent (non-negative savings, sorted, the knob-only item present on the no-knob path);
* the CLI runs and its JSON parses.

    pytest models/demos/glm53_flash/tests/test_roofline_host.py --noconftest -q
"""

from __future__ import annotations

import json

import pytest

from models.demos.glm53_flash.tt import roofline as rl
from models.demos.glm53_flash.tt.roofline import (
    BYTES,
    LAYER_COUNTS,
    MEASURED_STEPS_MS,
    TARGETS_MS,
    Layout,
    active_experts,
    ccl_us,
    expected_active_experts,
    layer_rows,
    layer_summary,
    layer_table,
    load_config,
    ranked_actions,
    step_table,
)

CFG = load_config()


def _row(layer: str, lay: Layout, key: str):
    rows = [r for r in layer_rows(CFG, lay, layer) if r.key == key]
    assert rows, key
    return rows[0]


# ----------------------------------------------------------------------------------------------- config
def test_config_schedule():
    assert CFG.num_hidden_layers == 45
    assert len(CFG.kda_layers) == rl.KDA_LAYERS and len(CFG.dsa_layers) == rl.DSA_LAYERS
    assert sum(LAYER_COUNTS[k] for k in ("KDA+MoE", "DSA+MoE", "KDA+dense")) == 45
    assert CFG.n_routed_experts == 288 and CFG.num_experts_per_tok == 8 and CFG.moe_intermediate_size == 2048


# ----------------------------------------------------------------------------------------------- bytes
def test_expert_bytes_match_ledger():
    h, i = CFG.hidden_size, CFG.moe_intermediate_size
    full = h * i * 3 * BYTES["bfp8"]  # gate | up | down of one full expert (EP32 layout)
    assert full == pytest.approx(26.7e6, rel=0.01)  # ledger: 26.7 MB bfp8 per expert
    slice_ = h * (i // 8) * 3 * BYTES["bfp8"]  # the hybrid chip slice (I / 8 rows)
    assert slice_ == pytest.approx(3.34e6, rel=0.01)  # expert_kernel_sweep.md: 3.34 MB per expert slice
    assert 72 * slice_ == pytest.approx(9 * full, rel=1e-9)  # the same 240 MB per layer per chip in both layouts
    r = _row("KDA+MoE", Layout(users_per_row=4), "moe.experts")
    assert r.dram_bytes == pytest.approx(28 * slice_ + 2 * 256 * h * 2, rel=1e-6)  # 28 measured active experts
    r = _row("KDA+MoE", Layout(users_per_row=32, moe="ep"), "ep.experts")
    assert r.dram_bytes == pytest.approx(8 * full + 2 * 32 * 8 * h * 2, rel=1e-6)


def test_weight_bytes_of_the_projections():
    lay = Layout(users_per_row=4)
    qkv = _row("KDA+MoE", lay, "kda.qkv")
    assert qkv.dram_bytes == pytest.approx(4096 * 6144 * BYTES["bfp8"] + 32 * (4096 + 6144) * 2, rel=1e-6)
    head = _row("head", lay, "head.matmul")
    assert head.dram_bytes / 1e6 == pytest.approx(4096 * 38720 * BYTES["bfp8"] / 1e6 + 32 * (4096 + 38720) * 2 / 1e6)
    state = _row("KDA+MoE", lay, "kda.decode")
    assert state.dram_bytes > 2 * 4 * 16 * 128 * 128 * 4  # the fp32 recurrent state read + written for 4 users
    pre = [r for r in layer_rows(CFG, lay, "KDA+MoE") if r.module == "MHC_PRE_ATTN"]
    assert sum(r.n_ops for r in pre) == 1  # the fused pre-block (the default since 3722af90ca0)
    composite = Layout(users_per_row=4, mhc_pre_block=False)
    pre = [r for r in layer_rows(CFG, composite, "KDA+MoE") if r.module == "MHC_PRE_ATTN"]
    assert sum(r.n_ops for r in pre) == 18  # 16 composite pre-block ops before A3 (2-op norm), 18 with its 4-op norm


# ----------------------------------------------------------------------------------------------- floors
@pytest.mark.parametrize("moe", ["hybrid", "ep"])
def test_floors_monotone_in_users(moe):
    prev_step, prev_layers = 0.0, {}
    for t in (1, 2, 4, 8, 16, 32):
        lay = Layout(users_per_row=t, moe=moe)
        step = step_table(CFG, lay)
        assert step["step_floor_ms"] >= prev_step - 1e-9, (t, step["step_floor_ms"], prev_step)
        assert step["step_floor_spec_ms"] <= step["step_floor_ms"]
        for name in LAYER_COUNTS:
            f = layer_summary(CFG, lay, name)["floor_us"]
            assert f >= prev_layers.get(name, 0.0) - 1e-9, (t, name)
            prev_layers[name] = f
        prev_step = step["step_floor_ms"]


@pytest.mark.parametrize("users_per_row,point", [(4, "32u"), (32, "256u")])
@pytest.mark.parametrize("moe", ["hybrid", "ep"])
def test_step_floor_below_every_measured_step(users_per_row, point, moe):
    step = step_table(CFG, Layout(users_per_row=users_per_row, moe=moe))
    assert step["layout"]["point"] == point
    for name, ms in MEASURED_STEPS_MS[point].items():
        assert step["step_floor_ms"] < ms, (name, ms, step["step_floor_ms"])
        assert step["step_floor_spec_ms"] < ms
    # the modelled sum of the measured / projected ops is within 10 % of the e2e steps of its own layout
    e2e = [v for k, v in MEASURED_STEPS_MS[point].items() if (("EP32" in k) == (moe == "ep")) and "same token" not in k]
    assert min(e2e) * 0.9 <= step["step_model_measured_ms"] <= max(e2e) * 1.1, (step["step_model_measured_ms"], e2e)


def test_targets_vs_floors():
    """The 33 tok/s target at 32 users sits above the floor at the achievable DRAM rate; the 19 tok/s target at 256
    users sits between the achievable-rate floor and the spec-peak floor (so it needs design changes, not tuning)."""
    s32 = step_table(CFG, Layout(users_per_row=4))
    assert s32["step_floor_ms"] > 0.9 * TARGETS_MS["32u"] and s32["step_floor_spec_ms"] < TARGETS_MS["32u"]
    s256 = step_table(CFG, Layout(users_per_row=32))
    assert s256["step_floor_spec_ms"] < TARGETS_MS["256u"] < s256["step_floor_ms"] * 1.15
    # bfp4 gate | up experts and the dense-regime gather removal bring the 256-user floor under the target
    s256b = step_table(CFG, Layout(users_per_row=32, expert_w13_dtype="bfp4"))
    assert s256b["step_floor_ms"] < s256["step_floor_ms"] - 5.0


def test_per_user_state_terms_scale_with_users():
    k4 = _row("KDA+MoE", Layout(users_per_row=4), "kda.decode")
    k32 = _row("KDA+MoE", Layout(users_per_row=32), "kda.decode")
    assert k32.dram_bytes > 7 * k4.dram_bytes  # 2 MB of fp32 state per user dominates
    assert k32.floor_us == pytest.approx(k32.dram_bytes / 400e3, rel=1e-6)
    g4 = _row("DSA+MoE", Layout(users_per_row=4), "dsa.pool_gather")
    g32 = _row("DSA+MoE", Layout(users_per_row=32), "dsa.pool_gather")
    assert g32.dram_bytes == pytest.approx(8 * g4.dram_bytes)
    assert g32.measured_src.startswith("proj")  # no DSA profile at 32 users/row: projections are labelled


# ----------------------------------------------------------------------------------------------- CCL + routing
def test_ccl_fits_reproduce_the_report():
    # CCL_bandwidth.md Blackhole line fits at the Galaxy's 2 x 25 GB/s links
    assert ccl_us("all_reduce", 4, 256 * 1024) == pytest.approx(4.4 + 4.4 * 3 + 1.5 * 262144 / 44e3, rel=1e-6)
    assert ccl_us("reduce_scatter", 8, 2 * 1024 * 1024) == pytest.approx(1.3 + 21 + 7 / 8 * 2097152 / 40e3, rel=1e-6)
    assert ccl_us("all_gather", 8, 32 * 1024) == pytest.approx(3.7 + 9.1 + 7 * 32768 / 48e3, rel=1e-6)
    # rated links halve the bandwidth term; one link doubles it; a ring is faster than a line on 8 devices
    line = ccl_us("reduce_scatter", 8, 2 * 1024 * 1024)
    assert ccl_us("reduce_scatter", 8, 2 * 1024 * 1024, link_gbs=50.0) < line
    assert ccl_us("reduce_scatter", 8, 2 * 1024 * 1024, links=1) > line
    assert ccl_us("reduce_scatter", 8, 2 * 1024 * 1024, topology="ring") < line
    r = _row("KDA+MoE", Layout(users_per_row=32), "moe.rs_rows")
    assert r.floors["ccl_link_bytes"] == pytest.approx(7 / 8 * 256 * 4096 * 2)


def test_active_expert_model():
    mean, mx = expected_active_experts(288, 8, 256, 72)
    assert 70.0 <= mean <= 72.0 and mx <= 72.0  # 256 tokens x 8 cover a column of 72
    mean32, mx32 = expected_active_experts(288, 8, 32, 72)
    assert 38.0 <= mean32 <= 47.0 and mean32 <= mx32 <= 55.0
    assert active_experts(CFG, Layout(users_per_row=4))[0] == 28.0  # measured real routing
    assert active_experts(CFG, Layout(users_per_row=32))[0] == 58.0
    a8 = active_experts(CFG, Layout(users_per_row=8))[0]
    assert 28.0 < a8 < 58.0
    assert active_experts(CFG, Layout(users_per_row=4, routing="uniform"))[0] == pytest.approx(mx32)
    assert active_experts(CFG, Layout(users_per_row=4, active_experts=10))[0] == 10.0


# ----------------------------------------------------------------------------------------------- classification
def test_bound_classification():
    lay = Layout(users_per_row=4)
    kda = {r.key: r for r in layer_rows(CFG, lay, "KDA+MoE")}
    assert kda["kda.qkv"].bound == "DRAM" and kda["kda.qkv"].ratio < 1.1  # 393 GB/s measured
    assert "mhc.pre_block_fused" in kda and "mhc.ksum" not in kda  # the fused pre-block is the default
    comp = {r.key: r for r in layer_rows(CFG, Layout(users_per_row=4, mhc_pre_block=False), "KDA+MoE")}
    assert comp["mhc.ksum"].bound == "data-movement"
    assert kda["moe.rs_rows"].bound == "ethernet/CCL" and kda["kda.all_reduce"].bound == "ethernet/CCL"
    assert comp["mhc.sinkhorn"].bound == "L1/NoC" and comp["mhc.sinkhorn"].cores == 1
    assert kda["moe.router_topk"].bound == "launch/gap"
    assert kda["layer.gaps"].bound == "launch/gap"
    assert kda["moe.experts"].bound in ("DRAM", "DRAM (under-utilized)")
    dsa = {r.key: r for r in layer_rows(CFG, lay, "DSA+MoE")}
    assert dsa["dsa.pool_gather"].bound == "L1/NoC"
    assert dsa["dsa.state_composite"].kind == "layout" and dsa["dsa.state_composite"].n_ops == 29
    for r in list(kda.values()) + list(dsa.values()):
        assert r.floor_us > 0 and r.floor_spec_us > 0 and r.floor_spec_us <= r.floor_us + 1e-9
        assert r.measured_us is not None and r.measured_src, r.key
        assert r.measured_us >= 0.5 * r.floor_us or r.key in ("head.argmax",), (r.key, r.measured_us, r.floor_us)


def test_knob_variants_change_the_rows():
    base = {r.key for r in layer_rows(CFG, Layout(users_per_row=4), "DSA+MoE")}
    fused = {r.key for r in layer_rows(CFG, Layout(users_per_row=4, dsa_state_fused=True), "DSA+MoE")}
    assert "dsa.state_composite" in base and "dsa.state_fused" in fused and "dsa.absorb_q" not in fused
    a1 = {r.key for r in layer_rows(CFG, Layout(users_per_row=4, mhc_pre_block=True), "KDA+MoE")}
    assert "mhc.pre_block_fused" in a1 and "mhc.sinkhorn" not in a1
    s_a1 = layer_summary(CFG, Layout(users_per_row=4, mhc_pre_block=True), "KDA+MoE")["measured_us"]
    s_base = layer_summary(CFG, Layout(users_per_row=4, mhc_pre_block=False), "KDA+MoE")["measured_us"]
    assert s_a1 < s_base - 100  # the fused pre-block measured 101 us per wrap against ~168 (post-A3)
    assert layer_summary(CFG, Layout(users_per_row=4), "KDA+MoE")["measured_us"] == s_a1  # the default is fused


# ----------------------------------------------------------------------------------------------- actions + tables
def test_ranked_actions():
    acts = ranked_actions(CFG, Layout(users_per_row=4))
    ms = [a["ms"] for a in acts]
    assert ms == sorted(ms, reverse=True) and all(m >= 0 for m in ms)
    ids = [a["action_id"] for a in acts]
    assert "E0" not in ids and "H1" in ids and "H2" in ids  # E0 (the stream op) is the default path now
    assert "E0" not in [a["action_id"] for a in ranked_actions(CFG, Layout(users_per_row=4, expert_path="stream"))]
    assert sum(ms) < step_table(CFG, Layout(users_per_row=4))["step_model_measured_ms"]
    acts256 = ranked_actions(CFG, Layout(users_per_row=32))
    assert acts256[0]["action_id"] in ("E1", "B8")


def test_layer_table_columns_and_counts():
    lay = Layout(users_per_row=4)
    composite = Layout(users_per_row=4, mhc_pre_block=False)
    # the MoE glue counted at its measured op counts (today_8x4_dsa3 / kda4: 11-op gathers, a 32-op reduce-scatter)
    for name, n, n_comp in (("KDA+MoE", 79, 113), ("DSA+MoE", 122, 156), ("KDA+dense", 18, 52), ("head", 6, 6)):
        s = layer_summary(CFG, lay, name)
        assert s["ops"] == n, (name, s["ops"])  # the fused pre-block: 2 mHC rows per wrap instead of 9
        assert layer_summary(CFG, composite, name)["ops"] == n_comp, (name, "composite")
        tab = layer_table(CFG, lay, name)
        assert {"key", "floor_us", "measured_us", "ratio", "bound", "action", "after_us", "measured_src"} <= set(tab[0])
        assert sum(r["floor_us"] for r in tab) == pytest.approx(s["floor_us"], abs=0.5)


def test_cli_json(capsys):
    rl.main(["--users-per-row", "4", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["step"]["step_floor_ms"] > 0 and out["actions"]
    rl.main(["--users-per-row", "32", "--layer", "KDA+MoE", "--markdown"])
    text = capsys.readouterr().out
    assert "| key |" in text and "step floor" in text


def test_chip_numbers_have_sources():
    for k, (v, src) in rl.CHIP.items():
        assert isinstance(src, str) and len(src) > 20, k
    assert rl.chip("tflops_per_core")["HiFi2"] == pytest.approx(2.7)
    assert rl.chip("worker_cores") == 120 and rl.chip("links_per_axis") == 2


# ----------------------------------------------------------------------------------------------- speculative decode
def test_spec_rows_and_plain_reduction():
    """R = B (K + 1) <= 32 is enforced (one tile row: spec_decode.candidate_rows refuses R > 32, so B = 4, K = 7 = the
    policy's K is the last allowed point); K = 0 is the plain step; the verify's KDA row is K + 1 launches at B users
    plus a backup copy, the dense-range DSA rows carry no index work (the verify never scores / selects / gathers: the
    index maintenance is the commit's), the MoE routes 8 R tokens."""
    try:  # (try/except: the expect_error fixture lives in conftest, which this host test runs without)
        Layout(users_per_row=4, spec_k=8).verify_rows  # B (K + 1) = 36 > 32
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert Layout(users_per_row=4, spec_k=7).verify_rows == 32  # = the policy's K at B = 4
    assert Layout(users_per_row=4, spec_k=3).verify_rows == 16 and Layout(users_per_row=1, spec_k=5).verify_rows == 6
    assert [rl.spec_policy_k(b, 3) for b in (1, 4, 8, 16, 32)] == [3, 3, 3, 1, 0]
    assert rl.spec_policy_k(4, 10) == 7 and rl.spec_policy_k(1, 10) == 10
    lay = Layout(users_per_row=4, spec_k=3)
    keys = {r.key: r for r in layer_rows(CFG, lay, "KDA+MoE")}
    assert keys["kda.decode"].n_ops == 4 and keys["kda.decode"].measured_us == pytest.approx(4 * (48.6 + 6.6 * 4))
    assert "kda.spec_backup" in keys and keys["kda.spec_backup"].dram_bytes == pytest.approx(
        2 * 4 * (16 * 128 * 128 * 4 + 3 * 6144 * 2)
    )
    assert "for 128 gathered tokens" in keys["moe.experts"].op
    assert active_experts(CFG, lay)[0] == pytest.approx(48.0)  # 128 tokens: log-linear between 28 (32) and 58 (256)
    dsa = {r.key for r in layer_rows(CFG, lay, "DSA+MoE")}
    assert "dsa.spec_append" in dsa and not dsa & {
        "dsa.pool_gather",
        "dsa.index_score",
        "dsa.idx_query",
        "dsa.topk_rows",
    }
    assert "dsa.state_composite" not in dsa  # the index maintenance is not double counted (commit: spec.index_commit)
    plain_keys = {r.key for r in layer_rows(CFG, Layout(users_per_row=4), "KDA+MoE")}
    assert "kda.spec_backup" not in plain_keys and rl.step_table(CFG, Layout(users_per_row=4))[
        "step_floor_ms"
    ] == pytest.approx(
        28.97, abs=0.05
    )  # the fused pre-block default; 32.91 with the composite pre-block (the measured MoE glue op counts)
    assert rl.step_table(CFG, Layout(users_per_row=4, mhc_pre_block=False))["step_floor_ms"] == pytest.approx(
        32.91, abs=0.05
    )
    # the stream-op cost model reproduces the measured stream row at the plain point (28 active: 341 + 8 + 2 us)
    plain_stream = {r.key: r for r in layer_rows(CFG, Layout(users_per_row=4, expert_path="stream"), "KDA+MoE")}
    model = rl.EXPERT_FIXED_US + 28.0 * 3.34 * 1e6 / (rl.EXPERT_GBS * 1e3) + 10.0
    assert plain_stream["moe.experts"].measured_us == pytest.approx(264.4, abs=1.0)
    assert model == pytest.approx(351.0, abs=2.0)


def test_spec_iteration_monotone_and_above_plain():
    prev = None
    for k in (1, 2, 3, 5):
        it = rl.spec_iteration(CFG, Layout(users_per_row=4, spec_k=k))
        assert it["R"] == 4 * (k + 1)
        v = it["phases_ms"]["V main model (R rows)"]
        assert (
            v[0] > it["plain_step_floor_ms"] and v[1] > it["plain_step_model_ms"]
        )  # the verify costs more than a plain step
        assert it["iteration_floor_ms"] <= it["iteration_model_ms"]
        # the design floor (SP1: no replay / blend rows) sits below the current graph's floor; the after-actions time
        # is not bounded by either (A1 / B8a / HD1 / C12 change the op graph, as in the plain step: floor 32.3 ms,
        # after 35.1 at 4 users/row but 30.7 before the fused pre-block's cost was carried), only by the model
        assert it["design_floor_ms"] <= it["iteration_floor_ms"]
        assert it["iteration_after_ms"] < it["iteration_model_ms"]
        assert it["verify_ms"][1] == pytest.approx(v[1] + it["phases_ms"]["V MTP level 1 (R rows)"][1]) and it[
            "verify_ratio_model"
        ] == pytest.approx(it["verify_ms"][1] / it["plain_step_model_ms"])
        assert 1.0 < it["verify_ratio_model"] < 2.0  # R rows cost more than B rows but far less than R / B times
        for tgt, (model, floor, after) in it["tokens_needed"].items():
            assert model == pytest.approx(tgt * it["iteration_model_ms"] / 1e3) and floor < model and after < model
        c = it["phases_ms"]["C commit (KDA replay + blends, index)"][1]
        if prev is not None:
            assert (
                it["iteration_model_ms"] > prev["iteration_model_ms"]
                and c > prev["phases_ms"]["C commit (KDA replay + blends, index)"][1]
            )
            assert it["iteration_floor_ms"] > prev["iteration_floor_ms"]
        prev = it
    sweep = rl.spec_sweep(CFG, Layout(users_per_row=4))
    assert [s["K"] for s in sweep] == [1, 2, 3, 5]
    assert [s["K"] for s in rl.spec_sweep(CFG, Layout(users_per_row=16))] == [1]  # 16 (K + 1) <= 32


def test_spec_commit_cost_structure():
    it = rl.spec_iteration(CFG, Layout(users_per_row=4, spec_k=3))
    rows = {r.key: r for r in it["commit_rows"]}
    assert rows["spec.kda_replay"].n_ops == 34 * 4 and rows["spec.kda_blend"].n_ops == 6 * 34 * 4
    assert rows["spec.kda_replay"].measured_us == pytest.approx(136 * (48.6 + 6.6 * 4))
    assert rows["spec.kda_blend"].dram_bytes == pytest.approx(136 * 7 * 4 * (16 * 128 * 128 * 4 + 3 * 6144 * 2))
    assert rows["spec.kda_blend"].after_us == 0.0 and rows["spec.kda_replay"].after_us == 0.0  # SP1 removes both
    assert rows["spec.index_commit"].measured_us == pytest.approx(11 * 4 * 100.3)
    fused = {
        r.key: r for r in rl.spec_iteration(CFG, Layout(users_per_row=4, spec_k=3, dsa_state_fused=True))["commit_rows"]
    }
    assert fused["spec.index_commit"].measured_us < rows["spec.index_commit"].measured_us
    # tokens per iteration and tok/s per user are consistent with the iteration time
    assert it["tok_s_per_user_model"] == pytest.approx(it["accepted"] * 1e3 / it["iteration_model_ms"])
    assert it["tokens_needed_for_33"] == pytest.approx(33 * it["iteration_model_ms"] / 1e3)


def test_spec_scenarios(capsys):
    sc = rl.spec_scenarios(CFG, Layout(users_per_row=4, spec_k=3))
    t = sc["table"]
    assert set(t) == {80.0, 45.0, 33.0} and set(t[45.0]) == {1.5, 2.0, 2.5}
    for p in t:
        assert t[p][1.5] < t[p][2.0] < t[p][2.5]
    assert t[80.0][2.0] < t[45.0][2.0] < t[33.0][2.0]
    e = sc["elasticities"]
    assert e["accepted tokens"] == pytest.approx(1.0, abs=1e-6) and -1.0 < e["plain step"] < -0.5
    assert abs(e["host overhead"]) < 0.1 and -0.5 < e["commit + chain cost"] < 0
    assert sc["dominant"] in ("accepted tokens", "plain step")
    # tokens needed = target x iteration; the SP1 + B5 design needs fewer at every plain step
    for tgt in (30.0, 50.0):
        for p in (80.0, 45.0, 33.0):
            assert sc["tokens_needed"][tgt][p] == pytest.approx(tgt * sc["iteration_at"][p] / 1e3)
            assert sc["tokens_needed_design"][tgt][p] < sc["tokens_needed"][tgt][p]
            assert sc["table_design"][p][2.0] > sc["table"][p][2.0]
    assert sc["tokens_needed"][50.0][80.0] > sc["tokens_needed"][30.0][80.0] > sc["tokens_needed"][30.0][33.0]
    rl.main(["--users-per-row", "4", "--spec-k", "3", "--markdown"])
    out = capsys.readouterr().out
    assert "speculative decode" in out and "| K |" in out and "elasticities" in out
    assert "tokens for 30 tok/s" in out and "SP1 + B5 design" in out
    rl.main(["--users-per-row", "1", "--spec-k", "2", "--json"])
    j = json.loads(capsys.readouterr().out)
    assert j["spec"]["R"] == 3 and len(j["sweep"]) == 4 and "table" in j["scenarios"]
