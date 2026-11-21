# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug tests of paged_scaled_dot_product_attention_decode: chosen cells at a forced thread plan and cluster cap,
checked exactly as the golden suite checks them (eval.golden_runner.run_cell). DO NOT DELETE: they document the
bring-up (the plan at one thread of each kind first, then the design's plan, then several clusters)."""

import pytest
import ttnn

from eval import golden_runner
from eval.specs.paged_scaled_dot_product_attention_decode import spec as S
from ttnn.operations.paged_scaled_dot_product_attention_decode import (
    paged_scaled_dot_product_attention_decode_program_artifacts as artifacts,
)

OP = "paged_scaled_dot_product_attention_decode"
CELL = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": "L1", "sharding": "height",
        "output_memory": "DRAM"}


@pytest.fixture
def plan_override():
    saved = (artifacts._THREAD_PLAN_OVERRIDE, artifacts._MAX_CLUSTERS_PER_USER)
    yield
    artifacts._THREAD_PLAN_OVERRIDE, artifacts._MAX_CLUSTERS_PER_USER = saved


CASES = [
    # id, inputs, threads, max clusters, fp32, positions
    ("one_tile_wide_111", S.ONE_TILE_WIDE, (1, 1, 1), 1, True, [5]),
    ("one_tile_wide_111_last", S.ONE_TILE_WIDE, (1, 1, 1), 1, True, [40]),
    ("llama_split_111", S.LLAMA_SPLIT, (1, 1, 1), 1, True, [200]),
    ("qwen_two_111", S.QWEN_TWO_KV_HEADS, (1, 1, 1), 1, True, [100]),
    ("blocks_of_two_111", S.BLOCKS_OF_TWO_TILES, (1, 1, 1), 1, True, [300]),
    ("one_tile_wide_442", S.ONE_TILE_WIDE, (4, 4, 2), 1, True, [40]),
    ("llama_split_442", S.LLAMA_SPLIT, (4, 4, 2), 1, True, [200]),
    ("qwen_two_442", S.QWEN_TWO_KV_HEADS, (4, 4, 2), 1, True, [100]),
    ("pairs_442_multi", S.PAIRS_OF_HEADS, (4, 4, 2), None, True, [100]),
    ("blocks_of_two_442_multi", S.BLOCKS_OF_TWO_TILES, (4, 4, 2), None, True, [300]),
    ("llama_split_442_bf16", S.LLAMA_SPLIT, (4, 4, 2), None, False, [200]),
    # bisecting the single-cluster C=4 hang
    ("llama_split_442_p255", S.LLAMA_SPLIT, (4, 4, 2), 1, True, [255]),
    ("llama_split_442_p7", S.LLAMA_SPLIT, (4, 4, 2), 1, True, [7]),
    ("llama_split_222", S.LLAMA_SPLIT, (2, 2, 2), 1, True, [200]),
    ("llama_split_221", S.LLAMA_SPLIT, (2, 2, 1), 1, True, [200]),
    ("qwen_one_442", S.QWEN_ONE_KV_HEAD, (4, 4, 2), 1, True, [200]),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_debug_cell(device, plan_override, case):
    _, inputs, threads, max_clusters, fp32, positions = case
    artifacts._THREAD_PLAN_OVERRIDE = threads
    artifacts._MAX_CLUSTERS_PER_USER = max_clusters
    golden_runner.run_cell(OP, inputs, {**CELL, "fp32_dest_acc_en": fp32}, device,
                           extras={"positions": positions}, seed=42)
