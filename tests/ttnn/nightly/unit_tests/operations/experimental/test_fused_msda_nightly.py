# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Nightly subset of the fused_msda and untilize_and_split_heads unit tests.

The full files under tests/ttnn/unit_tests/operations/experimental/ take about 25 minutes
from a cold kernel cache on Blackhole, too long for the shared experimental sweep; this
subset takes about 2. Each test here delegates to the unit-test implementation so the
assertions can never drift apart; only the parameter sets are narrowed. The head-major
cases keep the shapes whose 32-key pages split a horizontal corner pair (HEAD_MAJOR_SHAPES)
and the writer's split-gather tail, which only Blackhole takes.
"""

import pytest

from tests.ttnn.unit_tests.operations.experimental.test_fused_msda import (
    LARGE_COUNTS,
    SMALL_COUNTS,
    SPLIT_ROW,
    test_fused_msda_l1_check_takes_tiled_head_major_value as _l1_check_takes_tiled_head_major_value,
    test_fused_msda_untilize_and_split_heads_feeds_fused_msda as _untilize_and_split_heads_feeds_fused_msda,
    test_fused_msda_untilize_and_split_heads_repeated_calls_hit_program_cache as _head_major_program_cache,
    test_fused_msda_v1_head_major_value_equivalence as _v1_head_major_value_equivalence,
    test_fused_msda_v1_head_major_value_scatter_path as _v1_head_major_value_scatter_path,
    test_fused_msda_v1_realistic_shapes as _v1_realistic_shapes,
    test_fused_msda_v1_repeated_calls_hit_program_cache as _v1_program_cache,
    test_fused_msda_v2_bevformer_pillar_shape as _v2_bevformer_pillar_shape,
    test_fused_msda_v2_head_major_value_equivalence as _v2_head_major_value_equivalence,
    test_fused_msda_v2_head_major_value_with_num_valid_queries_tensor as _v2_head_major_num_valid_queries_tensor,
    test_fused_msda_v2_num_valid_queries_tensor_follows_counts_in_trace as _v2_num_valid_queries_tensor_in_trace,
    test_fused_msda_v2_repeated_calls_hit_program_cache as _v2_program_cache,
    test_reference_matches_bevformer_reference as _reference_matches_bevformer_reference,
)
from tests.ttnn.unit_tests.operations.experimental.test_untilize_and_split_heads import (
    test_untilize_and_split_heads as _untilize_and_split_heads,
    test_untilize_and_split_heads_default_memory_config as _untilize_default_memory_config,
    test_untilize_and_split_heads_program_cache as _untilize_program_cache,
    test_untilize_and_split_heads_rejects_extra_tile_padding as _untilize_rejects_extra_tile_padding,
    test_untilize_and_split_heads_rejects_head_count_that_overflows_uint32 as _untilize_rejects_overflowing_head_count,
    test_untilize_and_split_heads_rejects_heads_narrower_than_a_tile as _untilize_rejects_narrow_heads,
    test_untilize_and_split_heads_rejects_transposed_tiles as _untilize_rejects_transposed_tiles,
    test_untilize_and_split_heads_wide_heads as _untilize_wide_heads,
)


def test_fused_msda_reference_matches_bevformer_reference():
    _reference_matches_bevformer_reference()


def test_fused_msda_v1_realistic_bevformer_sca_shape(device):
    _v1_realistic_shapes(device, "bevformer_sca", 1, 256, 8, 4, 4, 32, [(116, 200), (58, 100), (29, 50), (15, 25)])


def test_fused_msda_v2_bevformer_pillar_shape(device):
    _v2_bevformer_pillar_shape(device)


@pytest.mark.timeout(300)
def test_fused_msda_v1_repeated_calls_hit_program_cache(device):
    _v1_program_cache(device, "l1")


@pytest.mark.timeout(300)
def test_fused_msda_v2_repeated_calls_hit_program_cache(device):
    _v2_program_cache(device, 4)


# Q = 32 + SPLIT_ROW + 1 leaves the writer exactly one row of the tail block to gather.
@pytest.mark.parametrize(
    "memory, B, Q, H",
    [(m, B, Q, H) for m, s in SPLIT_ROW.items() for B, Q, H in [(2, 129, 8), (1, 32 + s + 1, 2)]],
)
def test_fused_msda_v1_head_major_value_equivalence(device, memory, B, Q, H):
    _v1_head_major_value_equivalence(device, B, Q, H, 32, memory)


@pytest.mark.parametrize("D", [16, 48])
def test_fused_msda_v1_head_major_value_scatter_path(device, D):
    _v1_head_major_value_scatter_path(device, D, "l1")


@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_fused_msda_v2_head_major_value_equivalence(device, memory):
    _v2_head_major_value_equivalence(device, memory)


@pytest.mark.parametrize(
    "B,Q,H,counts,packed,memory",
    [(*SMALL_COUNTS, True, "l1"), (*LARGE_COUNTS, True, "l1"), (*LARGE_COUNTS, False, "dram")],
    ids=["small-packed-l1", "large-packed-l1", "large-unpacked-dram"],
)
def test_fused_msda_v2_head_major_value_with_num_valid_queries_tensor(device, B, Q, H, counts, packed, memory):
    _v2_head_major_num_valid_queries_tensor(device, B, Q, H, counts, packed, memory)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_fused_msda_v2_head_major_num_valid_queries_tensor_follows_counts_in_trace(device):
    _v2_num_valid_queries_tensor_in_trace(device, True, "head_major")


def test_fused_msda_untilize_and_split_heads_feeds_fused_msda(device):
    _untilize_and_split_heads_feeds_fused_msda(device)


@pytest.mark.timeout(300)
def test_fused_msda_untilize_and_split_heads_repeated_calls_hit_program_cache(device):
    _head_major_program_cache(device, "l1")


def test_fused_msda_l1_check_takes_tiled_head_major_value(device):
    _l1_check_takes_tiled_head_major_value(device)


@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize("H", [1, 8])
@pytest.mark.parametrize("B,S", [(2, 33), (2, 332)])
def test_untilize_and_split_heads(device, B, S, H, D):
    _untilize_and_split_heads(device, B, S, H, D, "l1", "dram")


@pytest.mark.parametrize("D", [96, 288])
def test_untilize_and_split_heads_wide_heads(device, D):
    _untilize_wide_heads(device, 2, D, "l1")


def test_untilize_and_split_heads_program_cache(device):
    _untilize_program_cache(device)


def test_untilize_and_split_heads_default_memory_config(device):
    _untilize_default_memory_config(device)


def test_untilize_and_split_heads_rejects_transposed_tiles(device, expect_error):
    _untilize_rejects_transposed_tiles(device, expect_error)


def test_untilize_and_split_heads_rejects_head_count_that_overflows_uint32(device, expect_error):
    _untilize_rejects_overflowing_head_count(device, expect_error)


@pytest.mark.parametrize("padded_shape", [[2, 96, 64], [2, 64, 96]], ids=["rows", "channels"])
def test_untilize_and_split_heads_rejects_extra_tile_padding(device, expect_error, padded_shape):
    _untilize_rejects_extra_tile_padding(device, expect_error, padded_shape)


def test_untilize_and_split_heads_rejects_heads_narrower_than_a_tile(device, expect_error):
    _untilize_rejects_narrow_heads(device, expect_error)
