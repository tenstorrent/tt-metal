# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hand-verifiable goldens for the V4.1 theoretical performance model (no device)."""

import math

import pytest

from models.demos.deepseek_v3_d_p.utils import v41_perf_model as m

HW = m.Hardware(cores=100, clock_mhz=1000.0, link_bytes_per_ns=25.0, hop_latency_ns=600.0)


def test_linear_all_gather_bottleneck_edge():
    # 4 chips in a line: the end edge forwards 3 shards of 1000 B at 25 B/ns, plus 3 hops of 600 ns.
    coll = m.Collective("all_gather", "tp", 1000)
    assert m.collective_ns(coll, m.Layout(sp=1, tp=4), HW) == 3 * 1000 / 25 + 3 * 600


def test_ring_all_reduce_two_links():
    # 8-chip ring, 2 links: each pass carries 3.5 shards over 50 B/ns and 4 hops; all-reduce = 2 passes.
    coll = m.Collective("all_reduce", "sp", 1000)
    layout = m.Layout(sp=8, tp=1, links=2, sp_ring=True)
    assert math.isclose(m.collective_ns(coll, layout, HW), 2 * (3.5 * 1000 / 50 + 4 * 600))


def test_axis_all_to_all_bisection():
    # 4 chips in a line, 3000 B egress per chip (1000 B to each peer): 2 x 2 pairs cross the middle edge,
    # 4000 B at 25 B/ns, plus 3 hops of 600 ns. A ring has 2 cut edges and 2 hops.
    coll = m.Collective("all_to_all", "tp", 3000)
    assert math.isclose(m.collective_ns(coll, m.Layout(sp=1, tp=4), HW), 4000 / 25 + 3 * 600)
    assert math.isclose(m.collective_ns(coll, m.Layout(sp=1, tp=4, tp_ring=True), HW), 4000 / 50 + 2 * 600)


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
