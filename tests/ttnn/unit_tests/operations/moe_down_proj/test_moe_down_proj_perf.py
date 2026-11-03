# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf rows for moe_down_proj (production placement: x and output in L1, weight and counts in DRAM).

Run under run_safe_pytest.sh --profile and read DEVICE KERNEL DURATION [ns] from the profiler CSV, one row per
case in PERF_CASES order. Correctness is checked, perf is not asserted. Stock / target us are from
eval/prompts/moe_down_proj.txt.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import check, make_operands, run

LAYERS = (
    (2421, 536, 1535, 268, 299, 327, 1246, 1560),
    (2044, 144, 1080, 1314, 377, 1269, 583, 1381),
    (1646, 207, 557, 515, 1491, 613, 251, 2912),
    (1791, 463, 352, 72, 3551, 160, 1653, 150),
    (2360, 1483, 479, 248, 2841, 24, 189, 568),
    (228, 481, 3027, 332, 1944, 427, 1339, 414),
)
SCALED = {
    3072: (1816, 402, 1151, 201, 224, 245, 935, 1170),
    2048: (1211, 268, 768, 134, 149, 163, 623, 780),
    1024: (605, 134, 384, 67, 75, 82, 311, 390),
}

# (id, tokens, counts, stock us, target us)
PERF_CASES = [
    *(
        (f"t4096_layer{idx}", 4096, counts, stock, 90.8)
        for idx, (counts, stock) in enumerate(zip(LAYERS, (179.0, 174.8, 191.9, 187.3, 188.4, 180.2)))
    ),
    *(
        (f"t{tokens}_scaled", tokens, SCALED[tokens], stock, 74.6)
        for tokens, stock in ((3072, 158.9), (2048, 136.5), (1024, 127.7))
    ),
]


@pytest.mark.parametrize("name, tokens, counts, stock_us, target_us", PERF_CASES, ids=[c[0] for c in PERF_CASES])
def test_moe_down_proj_perf(device, name, tokens, counts, stock_us, target_us):
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts)
    out = run(x_dev, w_dev, counts_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    check(out, xq, wq, counts, ttnn.L1_MEMORY_CONFIG)


PLACEMENTS = {
    "l1_l1": (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    "dram_x": (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    "dram_out": (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    "dram_both": (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
}
PLACEMENT_CASES = [("t4096_layer0", 4096, LAYERS[0]), ("t1024_scaled", 1024, SCALED[1024])]


@pytest.mark.parametrize("placement", list(PLACEMENTS), ids=list(PLACEMENTS))
@pytest.mark.parametrize("name, tokens, counts", PLACEMENT_CASES, ids=[c[0] for c in PLACEMENT_CASES])
def test_moe_down_proj_perf_placement(device, name, tokens, counts, placement):
    x_memory_config, out_memory_config = PLACEMENTS[placement]
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts, x_memory_config)
    out = run(x_dev, w_dev, counts_dev, memory_config=out_memory_config)
    check(out, xq, wq, counts, out_memory_config)
