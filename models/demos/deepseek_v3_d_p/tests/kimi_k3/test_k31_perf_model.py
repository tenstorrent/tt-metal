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


def test_dsa_core_composition():
    e = m.dsa_core(prefix=123_904)
    ops = e.ops
    assert e.optimistic_ns == max(
        sum(o.compute_ns for o in ops), sum(o.dram_ns for o in ops), sum(o.fabric_ns for o in ops)
    )
    assert e.conservative_ns == pytest.approx(sum(o.roofline_cons_ns for o in ops) + sum(o.fabric_cons_ns for o in ops))
    assert e.target_ns == max(2 * e.optimistic_ns, e.conservative_ns)


def test_k3_probes_and_stages():
    k3 = m.K3Measured.load()
    assert sorted(k3.throughput_tok_s) == [0, 9, 19, 50, 101, 204]  # @5k @50k @100k @255k @510k @1M
    assert m.stage_attention_layers() == [6, 6, 6, 6]


def test_dense_core_reproduces_the_measured_period_slope():
    k3 = m.K3Measured.load()
    growth = 6 * (m.dense_mla_core_ns(204 * m.CHUNK, k3) - m.dense_mla_core_ns(0, k3))
    assert growth == pytest.approx(k3.period_ns(204) - k3.period_ns(0))


def test_k3_self_check_reproduces_measured_ttft():
    k3 = m.K3Measured.load()
    for d, measured in k3.ttft_s.items():
        assert m.ttft((d + 1) * m.CHUNK, "k3").ttft_s == pytest.approx(measured, rel=0.05), d


def test_dsa_beats_dense_at_long_context_and_scenarios_are_ordered():
    for tokens in (10_240, 131_072):
        r = {s: m.ttft(tokens, s).ttft_s for s in m.SCENARIOS}
        assert r["optimistic"] <= r["target"] and r["conservative"] <= r["target"]
    assert m.ttft(131_072, "target").ttft_s < m.ttft(131_072, "k3").ttft_s
