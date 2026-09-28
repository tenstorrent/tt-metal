# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hand-verifiable goldens for the V4.1 theoretical performance model (no device)."""

import math

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
    # layer 2 (ratio 2, KV source): 4 tokens -> 2 rows x (512 + 128) bf16, plus a 128 x 512 bf16 window ring.
    assert m.kv_cache_bytes(4, m.Workload(), [2]) == 2 * 640 * 2 + 128 * 512 * 2
    # layer 3 is a consumer: window ring only.
    assert m.kv_cache_bytes(4, m.Workload(), [3]) == 128 * 512 * 2


def test_routed_expert_bytes_bfp8_vs_bfp4():
    b8 = m.weight_bytes_per_layer(3, m.Workload(expert_dtype="bfp8"))["routed_experts"]
    b4 = m.weight_bytes_per_layer(3, m.Workload(expert_dtype="bfp4"))["routed_experts"]
    elems = 384 * 3 * 5120 * 2304
    assert b8 == elems * 1088 / 1024 and b4 == elems * 576 / 1024


def test_lm_head_runs_on_last_token_only():
    ops = {o.name: o for o in m.final_ops(m.Workload(), m.LOUDBOX_2X4, HW)}
    assert ops["lm_head_last_token"].matmul_flop == 2 * 5120 * 129280 / 8


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
