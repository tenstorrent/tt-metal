# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hand-verifiable goldens for the V4.1 theoretical performance model (no device)."""

import math

import pytest

from models.demos.deepseek_v3_d_p.utils import v41_perf_model as m

# hand values: all-gather 25 B/ns per link + 600 ns per op, reduce-scatter 10 B/ns + 1000 ns, all-to-all 15 B/ns + 500 ns
CCL = {
    "all_gather": m.CclRate(link_bytes_per_ns=25.0, latency_ns=600.0),
    "reduce_scatter": m.CclRate(link_bytes_per_ns=10.0, latency_ns=1000.0),
    "all_to_all": m.CclRate(link_bytes_per_ns=15.0, latency_ns=500.0),
}
HW = m.Hardware(cores=100, clock_mhz=1000.0, ccl=CCL)


def test_linear_all_gather_bottleneck_edge():
    # 4 chips in a line: the end edge forwards 3 shards of 1000 B at 25 B/ns, plus the 600 ns op latency.
    coll = m.Collective("all_gather", "tp", 1000)
    assert m.collective_ns(coll, m.Layout(sp=1, tp=4), HW) == 3 * 1000 / 25 + 600


def test_ring_all_reduce_two_links():
    # 8-chip ring, 2 links: reduce-scatter then all-gather, each 3.5 shards over the bidirectional ring:
    # RS 3500 B / (10 x 2) + 1000 = 1175 ns, AG 3500 B / (25 x 2) + 600 = 670 ns.
    coll = m.Collective("all_reduce", "sp", 1000)
    layout = m.Layout(sp=8, tp=1, links=2, sp_ring=True)
    assert math.isclose(m.collective_ns(coll, layout, HW), 1175 + 670)


def test_axis_all_to_all_injection():
    # 3000 B egress per chip. 4 chips in a line (interior chips inject on 2 ports): 1500 B at 15 B/ns + 500 ns
    # = 600 ns; a ring is the same. A 2-chip line has 1 port: 3000 / 15 + 500 = 700 ns.
    coll = m.Collective("all_to_all", "tp", 3000)
    assert math.isclose(m.collective_ns(coll, m.Layout(sp=1, tp=4), HW), 600)
    assert math.isclose(m.collective_ns(coll, m.Layout(sp=1, tp=4, tp_ring=True), HW), 600)
    assert math.isclose(m.collective_ns(coll, m.Layout(sp=1, tp=2), HW), 700)


def test_loudbox_calibrated_all_gather():
    # LoudBox 2x4 at the fabric's 2 links: TP all-gather of a 1 MiB shard = 3 x 1,048,576 B / (22 B/ns x 2)
    # + 14 us = 71,494 + 14,000 ns (calibrated: 90 us measured standalone, evidence/G2/ccl).
    assert m.LOUDBOX_2X4.links == 2
    t = m.collective_ns(m.Collective("all_gather", "tp", 2**20), m.LOUDBOX_2X4, m.BLACKHOLE_P150B)
    assert math.isclose(t, 3 * 2**20 / 44 + 14_000)


def test_single_chip_axis_has_no_collective_cost():
    assert m.collective_ns(m.Collective("all_reduce", "tp", 1e6), m.Layout(sp=2, tp=1), HW) == 0


def test_wq_a_work_and_time():
    # chunk 5120 on 2x4: 2560 tokens/chip; [2560, 5120] @ [5120, 1280] split over TP=4.
    ops = {o.name: o for o in m.block_ops(0, m.Workload(), m.LOUDBOX_2X4, HW)}
    wq_a = ops["wq_a"]
    assert wq_a.matmul_flop == 2 * 2560 * 5120 * 1280 / 4
    # bfp8 weights -> HiFi2 (2 phases); 100 cores x 4096 FLOP/cycle x 1 cycle/ns = 409,600 FLOP/ns -> 40,960 ns.
    assert math.isclose(wq_a.compute_ns, 40960.0)


def test_kv_cache_bytes():
    # replicated state for max_seq_len 4, chunk 64. Layer 2 (ratio-2 KV source): KV tensor of 128 slot + 64 chunk +
    # 2 compressed rows x 512 bf16, 2 index-K rows x 128 bf16, and its 128-row window carry.
    w = m.Workload(chunk=64)
    assert m.kv_cache_bytes(4, w, [2]) == (128 + 64 + 2) * 1024 + 2 * 256 + 128 * 1024
    # SCALED_FP8 rows are 528 B (512 FP8 + 4 fp32 scales); index-K stays bf16.
    fp8 = m.Workload(chunk=64, kv_dtype="scaled_fp8")
    assert m.kv_cache_bytes(4, fp8, [2]) == (128 + 64 + 2) * 528 + 2 * 256 + 128 * 528
    # layer 3 is a consumer: its window carry only; layer 0 (ratio 0) adds the bf16 SWA scratch (slot + chunk).
    assert m.kv_cache_bytes(4, w, [3]) == 128 * 1024
    assert m.kv_cache_bytes(4, fp8, [0]) == 128 * 1024 + (128 + 64) * 1024


def test_routed_expert_bytes_bfp8_vs_bfp4():
    b8 = m.weight_bytes_per_layer(3, m.Workload(expert_dtype="bfp8"))["routed_experts"]
    b4 = m.weight_bytes_per_layer(3, m.Workload(expert_dtype="bfp4"))["routed_experts"]
    elems = 384 * 3 * 5120 * 2304
    assert b8 == elems * 1088 / 1024 and b4 == elems * 576 / 1024


def test_lm_head_runs_on_last_token_only():
    # vocab split over TP (column-parallel head, replicated over SP)
    ops = {o.name: o for o in m.final_ops(m.Workload(), m.LOUDBOX_2X4, HW)}
    assert ops["lm_head_last_token"].matmul_flop == 2 * 5120 * 129280 / 4


def test_dspark_uses_last_128_rows():
    ops = {o.name: o for o in m.dspark_prefill_ops(m.Workload(chunk=5120), m.LOUDBOX_2X4, HW)}
    # main_proj: [128, 15360] @ [15360, 5120] split over TP=4
    assert ops["main_proj"].matmul_flop == 2 * 128 * 15360 * 5120 / 4
    short = {o.name: o for o in m.dspark_prefill_ops(m.Workload(chunk=64), m.LOUDBOX_2X4, HW)}
    assert short["main_proj"].matmul_flop == 2 * 64 * 15360 * 5120 / 4


def test_vision_patch_embed_work():
    # 1 aligner token = 9 patches; patch_embed [9, 588] @ [588, 1024] split over 8 chips
    ops = {o.name: o for o in m.vision_ops(1, m.LOUDBOX_2X4, HW)}
    assert ops["patch_embed"].matmul_flop == 2 * 9 * 588 * 1024 / 8


@pytest.mark.parametrize("layout", [m.GALAXY_8X4, m.GALAXY_4X8], ids=["8x4", "4x8"])
def test_galaxy_layouts_per_chip_work(layout):
    # chunk 1024 = the smallest chunk on 32 chips (32 * sp * tp): 1024 / sp tokens per chip
    assert layout.chips == 32 and layout.links == 2
    w = m.Workload(chunk=1024)
    s = 1024 // layout.sp
    ops = {o.name: o for o in m.block_ops(3, w, layout, HW)}
    assert ops["wq_a"].matmul_flop == 2 * s * 5120 * 1280 / layout.tp
    # head->sequence all-to-all: this chip's 64/tp heads of s tokens, all but its own 1/tp leave
    (a2a,) = ops["q_head_to_seq"].collectives
    assert (a2a.kind, a2a.axis) == ("all_to_all", "tp")
    assert a2a.shard_bytes == s * (64 // layout.tp) * 512 * 2 * (1 - 1 / layout.tp)
    # dispatch buffer: the column's 1024 tokens x top-k 6, plus one tile per further local expert (12 per chip)
    assert m.moe_dispatch_rows(w, layout) == 1024 * 6 + 32 * 11


@pytest.mark.parametrize("layout", [m.GALAXY_8X4, m.GALAXY_4X8], ids=["8x4", "4x8"])
def test_galaxy_capacity_components(layout):
    cap = m.capacity_per_chip([1], m.Workload(engram_tables="device"), layout, context_tokens=0, dspark=False)
    # 384 / 32 = 12 experts per chip, 3 bfp8 matrices of 5120 x 2304
    assert cap["routed_experts"] == 12 * 3 * 5120 * 2304 * 1088 / 1024
    # bf16 embedding split over TP, replicated over SP
    assert cap["embedding"] == 129280 * 5120 * 2 / layout.tp
    # layer 1's Engram table: ceil(384,006,168 / 32) = 12,000,193 packed 320 B rows + one zero row
    assert cap["engram_tables"] == 12_000_194 * 320
    host = m.capacity_per_chip([1], m.Workload(), layout, context_tokens=0, dspark=False)
    assert host["engram_tables"] == 0


# --- scenarios (bead 8y7.9.13) ---------------------------------------------------------------------------------


def test_topk_time_at_measured_rate_with_row_imbalance():
    # 220 rows on 100 cores: 3 rounds of rows, so as if 300 rows ran; 300 x 1000 elements at 24.7 /ns (k <= 512)
    assert math.isclose(m.topk_ns(220, 1000, 512, HW), 300 * 1000 / 24.7)
    # k = 2048 at 7.1 /ns; 200 rows on 100 cores are balanced
    assert math.isclose(m.topk_ns(200, 1000, 2048, HW), 200 * 1000 / 7.1)


@pytest.mark.parametrize(
    "layer, t, candidate_calls, row_width",
    [
        (20, 5120, [], 5120),  # 640 blocks of 8 <= 2048: every block kept, no ranking; top-512 over the row
        (20, 16384, [], 16384),  # exactly 2048 blocks: still no candidates
        (20, 16416, [(640, 2052, 2048)], 16384),  # 2052 blocks: top-2048 over them; row top-k in 2048 x 8 columns
        (20, 517120, [(640, 64640, 2048)], 16384),  # all 64640 block maxima ranked (no superblock level)
        (24, 56320, [], 16384),  # candidate index source: -inf outside its 2048 x 8 candidate columns
        (24, 517120, [], 16384),
        (24, 5120, [], 5120),  # no candidates yet: the whole row
        (2, 258560, [], 258560),  # ratio-2 index source: always the whole row
    ],
)
def test_selection_topk_calls_follow_semantics(layer, t, candidate_calls, row_width):
    cand, row = m.selection_topk_calls(m.C.block_type(layer), 640, t)
    assert cand == candidate_calls and row == [(640, row_width, 512)]


def test_candidate_selection_work_and_traffic():
    # LoudBox 2x4 at start 51200: 640 queries per chip; L20 / L24 (ratio 1) see 56,320 columns = 7040 blocks of 8.
    ops = {o.name: o for o in m.block_ops(20, m.Workload(start=51200), m.LOUDBOX_2X4, HW)}
    sel = ops["candidate_select"]
    # 7 max per block of 8 over 7040 blocks + 1 pin per query: 640 x (56320 - 7040 + 1) = 31,539,840
    assert sel.eltwise == 31_539_840
    # score read 640 x 56320 x 2 = 72,089,600 B; block maxima written + read 640 x 7040 x 2 x 2 = 18,022,400 B;
    # 2048 int32 block ids per query written 5,242,880 B
    assert sel.dram_bytes == 72_089_600 + 18_022_400 + 5_242_880
    assert sel.topk_calls == [(640, 7040, 2048)]
    # row top-k: the 16384 candidate columns (20,971,520 B) + block ids (5,242,880 B) read, 512 int32 ids written
    # (1,310,720 B); a candidate index source does the same and has no selection op of its own
    for layer in (20, 24):
        by_name = {o.name: o for o in m.block_ops(layer, m.Workload(start=51200), m.LOUDBOX_2X4, HW)}
        assert by_name["topk"].dram_bytes == 20_971_520 + 5_242_880 + 1_310_720
        assert by_name["topk"].topk_calls == [(640, 16384, 512)]
    assert "candidate_select" not in {o.name for o in m.block_ops(24, m.Workload(start=51200), m.LOUDBOX_2X4, HW)}
    # the score is written once by the scoring op (visibility mask fused): 72,089,600 B of its DRAM bytes
    scores = ops["index_scores"]
    assert scores.dram_bytes == 640 * 32 * 128 * 2 + 56320 * 128 * 2 * 3 + 72_089_600


def test_indexer_topk_is_timed_in_compute():
    # LoudBox 2x4 at start 512000, L2 (ratio 2): 640 query rows x 258,560 visible columns, one direct top-512
    ops = {o.name: o for o in m.block_ops(2, m.Workload(start=512000), m.LOUDBOX_2X4, HW)}
    topk = ops["topk"]
    assert topk.sfpu["topk"] == 640 * 258560
    assert math.isclose(topk.sfpu_ns, 700 * 258560 / 24.7)  # 640 rows on 100 cores run as 700
    assert topk.compute_ns == topk.fpu_ns + topk.sfpu_ns


def test_sparse_attention_dram_bounds():
    # LoudBox 2x4, 640 queries per chip, BF16 KV rows of 512 x 2 B. At start 512000 layer 21 (ratio 1) sees 517,120
    # compressed rows; every query selects 128 window + 512 top-k rows.
    w = m.Workload(start=512000)
    op = {o.name: o for o in m.block_ops(21, w, m.LOUDBOX_2X4, HW)}["sparse_attention"]
    union = 640 + 127 + min(517120, 640 * 512)  # window span + top-k union (capped by the query picks)
    assert op.dram_bytes_cons - op.dram_bytes == (640 * 640 - union) * 1024
    # layer 0 (ratio 0) selects its 128 window rows only: union = the chunk span, no reuse = 640 x 128
    op0 = {o.name: o for o in m.block_ops(0, w, m.LOUDBOX_2X4, HW)}["sparse_attention"]
    assert op0.dram_bytes_cons - op0.dram_bytes == (640 * 128 - (640 + 127)) * 1024


def test_block_conservative_uses_conservative_dram_and_target():
    ops = [m.OpCost("X", "a", dram_bytes=512e3, dram_bytes_cons=1024e3), m.OpCost("X", "b", matmul_flop=0)]
    ops = [m._finish(o, m.LOUDBOX_2X4, HW) for o in ops]
    e = m.compose(ops)
    # 512 kB at 512 B/ns = 1000 ns optimistic; 2000 ns conservative; target = max(2 x 1000, 2000)
    assert e.optimistic_ns == 1000 and e.conservative_ns == 2000 and m.g2_target_ns(e) == 2000


def test_galaxy_slice_scales_chunk_and_start():
    assert [
        (w.chunk, w.start)
        for w in (m.scenario_workload(s, m.LOUDBOX_2X4, galaxy_slice=True) for s in ("S0", "S1", "S2"))
    ] == [(1280, 0), (1280, 12800), (1280, 128000)]
    assert m.scenario_workload("S2", m.GALAXY_8X4, galaxy_slice=True).chunk == 5120
    assert m.scenario_workload("S1", m.LOUDBOX_2X4).start == 51200


def test_full_model_layer_types_and_totals():
    e = m.chunk_estimate(m.scenario_workload("S0", m.LOUDBOX_2X4), m.LOUDBOX_2X4)
    counts = {k: len(v["layers"]) for k, v in e["by_type"].items()}
    assert counts == {
        "swa_only": 2,
        "kv_index_source": 3,
        "consumer_ratio2": 15,
        "candidate_source": 1,
        "consumer_ratio1": 15,
        "candidate_index_source": 4,
    }
    assert math.isclose(
        e["total"]["optimistic_ns"], sum(b.optimistic_ns for b in e["blocks"]) + e["tail"].optimistic_ns
    )
    assert math.isclose(sum(g["conservative_ns"] for g in e["groups"].values()), e["total"]["conservative_ns"])


def test_cache_length_changes_only_attention_side():
    # the chunk start moves only the attention's selection and the indexer; MoE, mHC and projections are unchanged
    g0 = m.chunk_estimate(m.scenario_workload("S0", m.LOUDBOX_2X4), m.LOUDBOX_2X4)["groups"]
    g2 = m.chunk_estimate(m.scenario_workload("S2", m.LOUDBOX_2X4), m.LOUDBOX_2X4)["groups"]
    for group in ("mhc", "attn.proj", "moe.routed", "moe.dispatch_combine", "moe.shared"):
        assert g0[group] == g2[group]
    assert g2["attn.indexer"]["conservative_ns"] > 10 * g0["attn.indexer"]["conservative_ns"]
