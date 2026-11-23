# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host tests of the GLM-5.4 prefill performance model: hand-computed operation bounds and the stage composition."""

import dataclasses

import pytest

from models.demos.deepseek_v3_d_p.utils import glm54_perf_model as m

ZERO_LAUNCH = dataclasses.replace(m.OPTIMISTIC, name="zero-launch", launch_ns=0.0)


def _anchor(g0=100.0, g1=180.0, g3=140.0, fe=2.0, slowest=120.0, ci=None):
    probes = list(m.PROBES)
    flat = lambda v: {p: v for p in probes}
    return m.Glm53Anchor(
        stage_ms={"G0": flat(g0), "G1": flat(g1), "G3": flat(g3)},
        full_extra_ms=flat(fe),
        glm53_slowest_ms=flat(slowest),
        sc4_ci_tok_s={} if ci is None else flat(ci),
    )


def test_payload_bytes():
    # 5120 x 6144 bf16; 28 layers x 5120 rows x 704 B pages; 7 sets x 5120 x 2048 u32
    assert m.KVPE_PAGE_BYTES == 704
    assert m.d2d_payload_bytes("G0->G1") == 62_914_560
    assert m.d2d_payload_bytes("G1->G2") == 62_914_560 + 100_925_440 + 293_601_280


def test_d2d_transfer_seam_bound():
    op = m.d2d_transfer("G1->G2", seam_gb_s=400.0)
    assert op.fabric_ns == pytest.approx(457_441_280 / 400.0)
    assert op.dram_bytes == pytest.approx(457_441_280 / 32)
    assert op.time_ns == op.fabric_ns  # the seam dominates the per-chip DRAM share


def test_ccl_fit_all_gather_ring():
    # 8-device ring, 640 pages of 704 B per device: (3.7 + 1.6 * 4) us + 7 * 450,560 B / 95 GB/s
    assert m.GALAXY.ccl_ns("all_gather", "ring", 8, 640 * 704) == pytest.approx(10_100 + 7 * 450_560 / 95)


def test_small_copies():
    copy = m.bank_a_page_copy(ZERO_LAUNCH)
    assert copy.dram_bytes == 2 * 160 * 704 and copy.time_ns == pytest.approx(225_280 / 512)
    carry = m.window_carry(m.CONSERVATIVE)
    assert carry.dram_bytes == 2 * 127 * 704
    assert carry.time_ns == pytest.approx(178_816 / 512 + 6_700)


def test_damp_gate_scalar():
    g = m.damp_gate(ZERO_LAUNCH, "scalar")
    # 640 x 1536 bf16 per chip: dot product 2 * 983,040 FLOP; 7 eltwise per element; read x, h, write x_k, W shard
    assert g.matmul_flop == 2 * 983_040 and g.eltwise == 7 * 983_040
    assert g.compute_ns == pytest.approx((2 * 983_040 * 2 / 4096 + 7 * 983_040 / 128) / (120 * 1.35))
    assert g.dram_ns == pytest.approx((3 * 1_966_080 + 1536 * 2) / 512)
    # two TP all-reduces (line, 4 chips) of 640 fp32 partials
    assert g.fixed_ns == pytest.approx(2 * (17_600 + 1.5 * 2560 / 44))
    assert g.sfpu == {"rsqrt": 640, "sigmoid": 640}


def test_damp_gate_channel_reads_its_weight_shard(expect_error):
    g = m.damp_gate(ZERO_LAUNCH, "channel")
    assert g.matmul_flop == 2 * 640 * 6144 * 1536
    assert g.dram_bytes == 3 * 1_966_080 + 6144 * 1536 * 2
    with expect_error(AssertionError, "gate width"):
        m.damp_gate(ZERO_LAUNCH, "head")


def test_two_bank_attention_delta():
    d = m.two_bank_attention_delta(m.CONSERVATIVE)
    # 160 queries x 128 window keys x 64 heads, q.k over 576 and p.v over 512
    assert d.matmul_flop == 2 * 160 * 128 * 64 * 1088
    assert d.sfpu == {"exp": 160 * 128 * 64}
    # table (287 pages) read + write, index lists, window reads once per query (no reuse) and their indices
    table, lists = 2 * 287 * 704, 160 * (2048 + 2176) * 4
    assert d.dram_bytes == table + lists + 160 * 128 * 704 + 160 * 128 * 4
    reuse = m.two_bank_attention_delta(m.OPTIMISTIC)
    assert reuse.dram_bytes == table + lists + 287 * 704 + 160 * 128 * 4


def test_sp_halo_variants():
    p2p = m.sp_halo(ZERO_LAUNCH)
    assert p2p.dram_bytes == 2 * 127 * 704
    gather = m.sp_halo(m.CONSERVATIVE)
    assert gather.dram_bytes == 2 * 7 * 640 * 704
    assert gather.time_ns > p2p.time_ns


def test_stage_composition():
    a = _anchor()
    st = m.stage_times("50k", a, ZERO_LAUNCH)
    assert st["G0"].total_ms == 100.0 and st["G0"].nodes == {}
    assert st["G3"].total_ms == 140.0
    gate = m.damp_gate(ZERO_LAUNCH).time_ns / 1e6
    keep = 2 * 1_966_080 / 512 / 1e6
    assert st["G1"].nodes == pytest.approx(
        {"keep_pass_input": keep, "damp_gate": gate, "g1_collect": m.g1_collect(ZERO_LAUNCH).time_ns / 1e6}
    )
    # G2: G1's measured time without its 7 indexers, plus 28 x the per-layer nodes and gate 2
    assert st["G2"].base_ms == pytest.approx(180.0 - 7 * 2.0)
    per_layer = sum(
        f(ZERO_LAUNCH).time_ns for f in (m.two_bank_attention_delta, m.sp_halo, m.window_carry, m.bank_a_page_copy)
    )
    assert st["G2"].total_ms == pytest.approx(166.0 + 28 * per_layer / 1e6 + keep + gate)
    lo, hi = st["G2"].band_ms
    assert (lo, hi) == pytest.approx((0.85 * st["G2"].total_ms, 1.15 * st["G2"].total_ms))


def test_pipeline_rates_and_ratio():
    p = m.pipeline("1020k", _anchor(ci=5120 / 0.1), m.OPTIMISTIC)  # GLM-5.3 SC4 period 100 ms vs 120 ms b06u02
    assert p.slowest_stage == "G1"
    assert p.tok_s == pytest.approx(5120 / (p.stages["G1"].total_ms / 1e3))
    assert p.ratio_vs_glm53 == pytest.approx(p.stages["G1"].total_ms / 120.0)
    assert p.sc4_pacing == "G1" and p.sc4_period_ms == p.slowest_stage_ms
    assert p.sc4_tok_s == pytest.approx(p.tok_s * 120.0 / 100.0)


def test_pipeline_d2d_scenarios():
    a = _anchor(fe=0.0)  # G2 ~ G1: a serialized D2D extra lands on both
    ser = m.pipeline("5k", a, m.CONSERVATIVE, seam_gb_s=4.0)
    extra = ser.d2d_ms["G1->G2"] - ser.d2d_ms["G0->G1"]
    assert extra == pytest.approx((457_441_280 - 62_914_560) / 4.0 / 1e6)
    assert ser.sc4_period_ms == pytest.approx(max(ser.stages[s].total_ms for s in ("G1", "G2")) + extra)
    over = m.pipeline("5k", a, m.OPTIMISTIC, seam_gb_s=1.0)  # a 457 ms transfer paces an overlapped pipeline
    assert over.sc4_pacing == "G1->G2" and over.sc4_period_ms == pytest.approx(457.44128)
    assert over.sc4_tok_s is None  # no SC4 CI anchor given
    assert over.tok_s == pytest.approx(5120 / (over.slowest_stage_ms / 1e3))  # stage-only rate ignores D2D


def test_invalid_inputs(expect_error):
    with expect_error(AssertionError, "unknown probe"):
        m.pipeline("10k", _anchor(), m.OPTIMISTIC)
    with expect_error(AssertionError, "anchor stages"):
        m.Glm53Anchor(stage_ms={"G0": {}, "G1": {}}, full_extra_ms={}, glm53_slowest_ms={})
