# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the K3.1 prefill performance model (``utils/kimi_k31_perf_model.py``)."""

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config as C
from models.demos.deepseek_v3_d_p.utils import kimi_k31_perf_model as m


@pytest.mark.parametrize(
    "prefix, queries, cap, expected",
    [
        (0, 4, None, 1 + 2 + 3 + 4),
        (0, 4, 2, 1 + 2 + 2 + 2),
        (10, 2, None, 11 + 12),
        (10, 2, 5, 5 + 5),
        (3, 4, 5, 4 + 5 + 5 + 5),
    ],
)
def test_causal_pairs(prefix, queries, cap, expected):
    assert m.causal_pairs(prefix, queries, cap) == expected


def test_gather_fabric_hand_values():
    # 3 chips, each receives 200 B (100 B from each other chip). Ingress 25 * 2 links * 4 ports = 200 B/ns -> 1 ns.
    # Ring edge: (3-1)/2 shards of 100 B = 100 B over 22 B/ns * 2 links, plus the 14 us latency.
    opt, cal = m._gather_fabric(200.0, 3, m.GALAXY)
    assert opt == pytest.approx(1.0)
    assert cal == pytest.approx(14_000.0 + 100.0 / 44.0)


def test_sparse_attention_saturates_at_topk():
    # Every query on the busiest chip sees more than top-k keys: q * topk selected pairs.
    op = next(o for o in m.dsa_ops(prefix=100_000) if o.name == "sparse_attention")
    q = m.CHUNK // 32
    assert op.matmul_flop == 2 * q * C.INDEX_TOPK * C.NUM_ATTENTION_HEADS * (2 * 512 + 64)
    # Conservative reads every selected row per query; optimistic each distinct row once.
    assert op.dram_bytes_cons - op.dram_bytes == pytest.approx((q * 2048 - min(105_120, q * 2048)) * 656)


def test_indexer_scores_the_busiest_chip_causally():
    op = next(o for o in m.dsa_ops(prefix=0) if o.name == "index_score")
    # The last 160 queries of the first chunk see 4961..5120 keys.
    assert op.matmul_flop == 2 * sum(range(4961, 5121)) * 32 * 128


def test_finish_hand_values():
    # 2 cores at 1 GHz: 2 core-cycles per ns.
    hw = m.Galaxy8x4(cores=2, clock_ghz=1.0)
    op = m._finish(
        m.OpCost(
            "hand",
            matmul_flop=3 * 4096,  # 3 LoFi cycles, x2 at HiFi2 -> 6 core-cycles
            fidelity="HiFi2",
            eltwise=5 * 128,  # 5 core-cycles
            topk_ns=7.0,  # measured time, added to compute as is
            dram_bytes=1024,
            dram_bytes_cons=4096,
        ),
        hw,
    )
    assert op.fpu_ns == pytest.approx((6 + 5) / 2)
    assert op.compute_ns == pytest.approx((6 + 5) / 2 + 7.0)
    assert op.dram_ns == pytest.approx(2.0)
    assert op.dram_cons_ns == pytest.approx(8.0)


def test_index_k_gather_hand_values():
    op = next(o for o in m.dsa_ops(prefix=0) if o.name == "index_k_gather")
    received = 5120 * 128 * 1088 / 1024 * 31 / 32  # visible bfp8 keys this chip does not own
    assert op.dram_bytes == pytest.approx(received)
    assert op.fabric_ns == pytest.approx(received / (25 * 2 * 4))


def test_dsa_core_target_doubles_only_modelled_work():
    e = m.dsa_core(prefix=126_976)
    fpu = sum(o.fpu_ns for o in e.ops)
    dram = sum(o.dram_ns for o in e.ops)
    fabric = sum(o.fabric_ns for o in e.ops)
    measured = sum(o.topk_ns + o.fixed_ns for o in e.ops)
    assert e.measured_ns == measured > 0
    assert e.target_ns == max(2 * max(fpu, dram, fabric) + measured, e.conservative_ns)


def test_indexer_score_fixed_cost_reproduces_its_floor():
    # The fused indexer score program (index-K ring gather + score): every chip took >= 197.7 us at prefix 0 at HiFi2
    # (ao3i.2.7); the conservative model of the fused pair with the fixed cost reproduces that floor.
    ops = {o.name: o for o in m.dsa_ops(prefix=0)}
    fused = sum(ops[n].roofline_cons_ns + ops[n].fabric_cons_ns for n in ("index_k_gather", "index_score"))
    assert fused == pytest.approx(197_700, abs=100)


@pytest.mark.parametrize(
    "prefix, topk_us, gather_us",
    [
        (0, 128.4, 57.3),  # the top-k outlasts the gather
        # (160 x 1,064,960 bf16 scores + 160 x 2,048 int32 indices) / 512 GB/s = 668.2 us of DRAM; the gather hides it
        (1_059_840, 668.2, 9027.0),
    ],
)
def test_topk_and_kvpe_gather_count_once(prefix, topk_us, gather_us):
    # They run concurrently on separate sub-devices: the conservative core holds the longer of the two, not both.
    e = m.dsa_core(prefix)
    each = {o.name: (o.roofline_cons_ns + o.fabric_cons_ns) / 1e3 for o in e.ops}
    assert each["index_topk"] == pytest.approx(topk_us, abs=1.0)
    assert each["kvpe_gather"] == pytest.approx(gather_us, abs=1.0)
    assert e.conservative_ns / 1e3 == pytest.approx(sum(each.values()) - min(topk_us, gather_us), abs=2.0)


def _hand_anchor():
    # Period 1 s at chunk 0 and 2 s at chunk 2; the first chunk's TTFT 1.5 s -> fill 0.5 s.
    return m.K3Measured(throughput_tok_s={0: 5120.0, 2: 2560.0}, chunk_time_ms={0: 1500.0}, ttft_s={})


def test_ttft_hand_anchor():
    k3 = _hand_anchor()
    assert k3.fill_ns() == pytest.approx(0.5e9)
    # The dense core grows by the measured slope: periods 1, 1.5, 2 s, plus the fill.
    r = m.ttft(3 * m.CHUNK, "k3", k3=k3)
    assert (r.chunks, r.first_period_ms, r.last_period_ms) == (3, pytest.approx(1000.0), pytest.approx(2000.0))
    assert r.ttft_s == pytest.approx(1.0 + 1.5 + 2.0 + 0.5)
    # Per layer: slope = 1 s / (2 chunks * 5120 tokens) / 6 layers; the first chunk's core has 5120*5121/2 pairs.
    assert m.dense_mla_core_ns(0, k3) == pytest.approx(1e9 / (2 * 5120 * 6) * 5121 / 2)


def test_ttft_swaps_the_dense_cores_for_dsa_cores():
    k3 = _hand_anchor()
    dense0 = m.dense_mla_core_ns(0, k3)
    dsa0 = m.dsa_core(0).optimistic_ns
    assert m.ttft(m.CHUNK, "optimistic", k3=k3).ttft_s == pytest.approx((1e9 - 6 * dense0 + 6 * dsa0 + 0.5e9) / 1e9)


def test_k3_probes_and_stages():
    k3 = m.K3Measured.load()
    assert sorted(k3.throughput_tok_s) == [0, 9, 19, 50, 101, 204]  # @5k @50k @100k @255k @510k @1M
    assert m.stage_attention_layers() == [6, 6, 6, 6]


def test_k3_self_check_reproduces_measured_ttft():
    k3 = m.K3Measured.load()
    # Chunk 0's chunk_time is its TTFT (summarize_ci_run.py): reproduced exactly.
    assert m.ttft(m.CHUNK, "k3").ttft_s == pytest.approx(k3.chunk_time_ms[0] / 1e3)
    for d, measured in k3.ttft_s.items():
        assert m.ttft((d + 1) * m.CHUNK, "k3").ttft_s == pytest.approx(measured, rel=0.05), d


def test_dsa_beats_dense_at_long_context():
    assert m.ttft(131_072, "target").ttft_s < m.ttft(131_072, "k3").ttft_s
