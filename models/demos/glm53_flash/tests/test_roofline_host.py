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
    assert sum(r.n_ops for r in pre) == 18  # 16 pre-block ops before A3 (2-op norm), 18 with its 4-op norm


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
    assert kda["mhc.ksum"].bound == "data-movement"
    assert kda["moe.rs_rows"].bound == "ethernet/CCL" and kda["kda.all_reduce"].bound == "ethernet/CCL"
    assert kda["mhc.sinkhorn"].bound == "L1/NoC" and kda["mhc.sinkhorn"].cores == 1
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
    s_base = layer_summary(CFG, Layout(users_per_row=4), "KDA+MoE")["measured_us"]
    assert s_a1 < s_base - 100  # the fused pre-block measured 101 us per wrap against ~168 (post-A3)


# ----------------------------------------------------------------------------------------------- actions + tables
def test_ranked_actions():
    acts = ranked_actions(CFG, Layout(users_per_row=4))
    ms = [a["ms"] for a in acts]
    assert ms == sorted(ms, reverse=True) and all(m >= 0 for m in ms)
    ids = [a["action_id"] for a in acts]
    assert "E0" in ids and "A1" in ids and "H2" in ids
    assert "E0" not in [a["action_id"] for a in ranked_actions(CFG, Layout(users_per_row=4, expert_path="stream"))]
    assert sum(ms) < step_table(CFG, Layout(users_per_row=4))["step_model_measured_ms"]
    acts256 = ranked_actions(CFG, Layout(users_per_row=32))
    assert acts256[0]["action_id"] in ("E1", "B8")


def test_layer_table_columns_and_counts():
    lay = Layout(users_per_row=4)
    for name, n in (("KDA+MoE", 88), ("DSA+MoE", 131), ("KDA+dense", 52), ("head", 6)):
        s = layer_summary(CFG, lay, name)
        assert s["ops"] == n, (name, s["ops"])
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
