# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.extract.

Verifies constraint rejection as well as the functional behavior: for a given
global_expert_id, the op copies global_tensor[start : start + ceil_tile(counts), :]
into the first rows of an output tensor shaped
[max_dispatched_tokens_per_expert, hidden_dim].
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


GLOBAL_ROWS = 64
HIDDEN_DIM = 128
NUM_EXPERTS = 8
GLOBAL_EXPERT_ID = 0
MAX_TOKENS = 32
TILE = 32


def _ceil_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def _make_global(
    device, *, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, shape=(GLOBAL_ROWS, HIDDEN_DIM), memory_config=None
):
    torch_tensor = torch.zeros(shape, dtype=torch.float32)
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_index(device, *, dtype=ttnn.uint32, shape=(NUM_EXPERTS,), memory_config=None):
    torch_tensor = torch.zeros(shape, dtype=torch.int32)
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_identity_idx_table(device, length=NUM_EXPERTS):
    """Build a 1D UINT32 DRAM-interleaved tensor `[0, 1, ..., length-1]`.

    Using an identity table preserves the old `global_expert_id` test semantics under
    the new API: local_expert_id=i -> table[i] = i (= old global_expert_id).
    """
    torch_tensor = torch.arange(length, dtype=torch.int32)
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run(global_tensor, start, counts, global_expert_id=GLOBAL_EXPERT_ID, max_tokens=MAX_TOKENS, idx_table=None):
    device = global_tensor.device()
    if idx_table is None:
        idx_table = _make_identity_idx_table(device)
    return ttnn.experimental.deepseek_prefill.extract(
        global_tensor,
        start,
        counts,
        idx_table,
        local_expert_id=global_expert_id,
        max_dispatched_tokens_per_expert=max_tokens,
    )


# ---------------------------------------------------------------------------
# Valid inputs should succeed.
# ---------------------------------------------------------------------------


def test_valid_inputs_1d_index(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    out = _run(g, s, c)
    assert out is not None


def test_valid_inputs_2d_index_first_dim_one(device):
    g = _make_global(device)
    s = _make_index(device, shape=(1, NUM_EXPERTS))
    c = _make_index(device, shape=(1, NUM_EXPERTS))
    out = _run(g, s, c)
    assert out is not None


@pytest.mark.parametrize(
    "global_shape",
    [
        (1, GLOBAL_ROWS, HIDDEN_DIM),
        (1, 1, GLOBAL_ROWS, HIDDEN_DIM),
    ],
)
def test_valid_global_tensor_leading_singleton_dims(device, global_shape):
    """global_tensor may carry leading singleton dims; it is treated as effectively
    2D (rows, hidden) using the last two dims, so no squeeze is required from callers."""
    g = _make_global(device, shape=global_shape)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    out = _run(g, s, c)
    assert out is not None


# ---------------------------------------------------------------------------
# Global tensor constraints.
# ---------------------------------------------------------------------------


def test_global_tensor_wrong_dtype(device, expect_error):
    g = _make_global(device, dtype=ttnn.float32)
    s = _make_index(device)
    c = _make_index(device)
    with expect_error(RuntimeError, "global_tensor must be BFLOAT8_B or BFLOAT16"):
        _run(g, s, c)


def test_global_tensor_leading_dim_not_one_rejected(device, expect_error):
    # rank >= 2 is allowed, but every dim beyond the last two must be 1. A leading
    # dim != 1 means the buffer is not effectively 2D and must be rejected.
    g = _make_global(device, shape=(2, GLOBAL_ROWS, HIDDEN_DIM))
    s = _make_index(device)
    c = _make_index(device)
    with expect_error(RuntimeError, "global_tensor leading dim 0 must be 1"):
        _run(g, s, c)


def test_global_tensor_must_be_dram_interleaved(device, expect_error):
    g = _make_global(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    s = _make_index(device)
    c = _make_index(device)
    with expect_error(RuntimeError, "global_tensor must be DRAM interleaved"):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Start tensor constraints.
# ---------------------------------------------------------------------------


def test_start_wrong_dtype(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, dtype=ttnn.uint16)
    c = _make_index(device)
    with expect_error(RuntimeError, "start must be UINT32"):
        _run(g, s, c)


def test_start_must_be_dram_interleaved(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    c = _make_index(device)
    with expect_error(RuntimeError, "start must be DRAM interleaved"):
        _run(g, s, c)


def test_start_2d_first_dim_not_one(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(2, NUM_EXPERTS))
    c = _make_index(device)
    with expect_error(RuntimeError, "start must be 1D or 2D with first dimension == 1"):
        _run(g, s, c)


def test_start_3d_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    c = _make_index(device)
    with expect_error(RuntimeError, "start must be 1D or 2D with first dimension == 1"):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Counts tensor constraints (same as start tensor).
# ---------------------------------------------------------------------------


def test_counts_wrong_dtype(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, dtype=ttnn.uint16)
    with expect_error(RuntimeError, "counts must be UINT32"):
        _run(g, s, c)


def test_counts_must_be_dram_interleaved(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error(RuntimeError, "counts must be DRAM interleaved"):
        _run(g, s, c)


def test_counts_2d_first_dim_not_one(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, shape=(2, NUM_EXPERTS))
    with expect_error(RuntimeError, "counts must be 1D or 2D with first dimension == 1"):
        _run(g, s, c)


def test_counts_3d_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    with expect_error(RuntimeError, "counts must be 1D or 2D with first dimension == 1"):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Cross-tensor and scalar constraints.
# ---------------------------------------------------------------------------


def test_start_and_counts_last_dim_mismatch_1d(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS + 1,))
    with expect_error(RuntimeError, "start and counts must have the same last dimension"):
        _run(g, s, c)


def test_start_and_counts_last_dim_mismatch_2d(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(1, NUM_EXPERTS))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with expect_error(RuntimeError, "start and counts must have the same last dimension"):
        _run(g, s, c)


def test_start_and_counts_last_dim_mismatch_mixed_rank(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with expect_error(RuntimeError, "start and counts must have the same last dimension"):
        _run(g, s, c)


def test_global_expert_id_equal_to_last_dim_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with expect_error(RuntimeError, "local_expert_id .* must be in the range"):
        _run(g, s, c, global_expert_id=NUM_EXPERTS)


def test_global_expert_id_beyond_last_dim_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with expect_error(RuntimeError, "local_expert_id .* must be in the range"):
        _run(g, s, c, global_expert_id=NUM_EXPERTS + 5)


def test_global_expert_id_max_valid_accepted(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    out = _run(g, s, c, global_expert_id=NUM_EXPERTS - 1)
    assert out is not None


# ---------------------------------------------------------------------------
# max_dispatched_tokens_per_expert constraints.
# ---------------------------------------------------------------------------


def test_max_tokens_zero_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device)
    with expect_error(RuntimeError, "max_dispatched_tokens_per_expert must be > 0"):
        _run(g, s, c, max_tokens=0)


def test_max_tokens_not_tile_aligned_rejected(device, expect_error):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device)
    with expect_error(RuntimeError, "max_dispatched_tokens_per_expert .* must be a multiple of TILE_HEIGHT"):
        _run(g, s, c, max_tokens=17)


# ---------------------------------------------------------------------------
# Functional: output[:ceil_tile(counts), :] == global[start:start+ceil_tile(counts), :].
# ---------------------------------------------------------------------------


def _make_index_from_values(device, values):
    torch_tensor = torch.tensor(values, dtype=torch.int32)
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_global_from_torch(device, torch_tensor):
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "starts, counts, expert_id, max_tokens",
    [
        ([0, 32, 64, 96], [32, 32, 32, 32], 0, 32),
        ([0, 32, 64, 96], [32, 32, 32, 32], 2, 64),  # output is larger than slice
        ([0, 64, 96, 128], [32, 17, 32, 5], 1, 32),  # counts rounds up 17 -> 32 tile-rows
        ([0, 64, 96, 128], [32, 17, 32, 5], 3, 64),  # counts rounds up 5 -> 32 tile-rows, output padded
        ([0, 32, 64, 96], [96, 32, 32, 32], 0, 128),  # full slice, output padded
    ],
)
def test_extract_matches_torch_slice(device, starts, counts, expert_id, max_tokens):
    hidden_dim = 128
    global_rows = _ceil_to_tile(max(s + c for s, c in zip(starts, counts)) + TILE)

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, s, c, global_expert_id=expert_id, max_tokens=max_tokens)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (
        max_tokens,
        hidden_dim,
    ), f"expected shape ({max_tokens}, {hidden_dim}), got {out_torch.shape}"
    rows = _ceil_to_tile(counts[expert_id])
    # Compare against bfp8_b-quantized source (tile-level kernel copy is bit-exact).
    global_quantized = ttnn.to_torch(g)
    expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_torch[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    # PCC against the original (pre-quantization) bfloat16 source: bfp8_b quantization
    # should preserve correlation well above 0.999.
    original = global_torch[starts[expert_id] : starts[expert_id] + rows, :]
    assert_with_pcc(original.float(), out_torch[:rows, :].float(), pcc=0.9999)


def test_extract_2d_indices_matches_torch_slice(device):
    hidden_dim = 64
    starts = [0, 32, 96]
    counts = [32, 48, 32]
    expert_id = 1
    max_tokens = 64
    global_rows = 160

    torch.manual_seed(1)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)

    s_torch = torch.tensor([starts], dtype=torch.int32)
    c_torch = torch.tensor([counts], dtype=torch.int32)
    s = ttnn.from_torch(
        s_torch, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    c = ttnn.from_torch(
        c_torch, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = _run(g, s, c, global_expert_id=expert_id, max_tokens=max_tokens)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (max_tokens, hidden_dim)
    rows = _ceil_to_tile(counts[expert_id])
    # Compare against bfp8_b-quantized source (tile-level kernel copy is bit-exact).
    global_quantized = ttnn.to_torch(g)
    expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_torch[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    # PCC against the original (pre-quantization) bfloat16 source.
    original = global_torch[starts[expert_id] : starts[expert_id] + rows, :]
    assert_with_pcc(original.float(), out_torch[:rows, :].float(), pcc=0.9999)


@pytest.mark.parametrize(
    "leading_dims",
    [
        (1,),
        (1, 1),
    ],
)
def test_extract_leading_singleton_dims_matches_2d(device, leading_dims):
    """A global_tensor presented with leading singleton dims must extract the same
    slice as the equivalent flat 2D buffer (the op treats it as effectively 2D).
    The output mirrors the input's rank: those leading singleton dims are carried
    through, so a (1, 1, rows, hidden) input yields a (1, 1, max_tokens, hidden) output."""
    hidden_dim = 128
    starts = [0, 32, 64, 96]
    counts = [32, 17, 32, 32]
    expert_id = 1
    max_tokens = 32
    global_rows = 128

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    # Same data, but reshaped to carry leading singleton dims, e.g. (1, 1, rows, hidden).
    g = _make_global_from_torch(device, global_torch.reshape(*leading_dims, global_rows, hidden_dim))
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, s, c, global_expert_id=expert_id, max_tokens=max_tokens)
    out_torch = ttnn.to_torch(out)

    # Output carries the same leading singleton dims as the input, with the last two
    # dims being [max_tokens, hidden].
    assert out_torch.shape == (*leading_dims, max_tokens, hidden_dim)
    # Flatten the leading singleton dims for the row-slice comparison.
    out_flat = out_torch.reshape(max_tokens, hidden_dim)
    rows = _ceil_to_tile(counts[expert_id])
    global_quantized = ttnn.to_torch(g).reshape(global_rows, hidden_dim)
    expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_flat[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    original = global_torch[starts[expert_id] : starts[expert_id] + rows, :]
    assert_with_pcc(original.float(), out_flat[:rows, :].float(), pcc=0.9999)


# ---------------------------------------------------------------------------
# Stress test: DRAM utilization with a large global tensor.
#
# Global tensor is shaped (2 * 25k, 7k) with 1k = 1024. max_dispatched_tokens
# per expert is 25k. Scenarios cover uniform extraction (every expert gets the
# same slice) and non-uniform extraction (one expert dominates, tail-heavy,
# irregular tile-unaligned counts).
# ---------------------------------------------------------------------------


K = 1024
STRESS_GLOBAL_ROWS = 2 * 25 * K  # 51200
STRESS_HIDDEN_DIM = 7 * K  # 7168
STRESS_MAX_TOKENS = 25 * K  # 25600


@pytest.mark.parametrize(
    "starts, counts, expert_id",
    [
        # Uniform: each of 8 experts gets an equal 3k-row slice.
        ([0, 3 * K, 6 * K, 9 * K, 12 * K, 15 * K, 18 * K, 21 * K], [3 * K] * 8, 0),
        ([0, 3 * K, 6 * K, 9 * K, 12 * K, 15 * K, 18 * K, 21 * K], [3 * K] * 8, 5),
        # One expert takes the full max_tokens, the rest take a single tile row each.
        (
            [0, 25 * K, 25 * K + 32, 25 * K + 64, 25 * K + 96, 25 * K + 128, 25 * K + 160, 25 * K + 192],
            [25 * K, 32, 32, 32, 32, 32, 32, 32],
            0,
        ),
        (
            [0, 25 * K, 25 * K + 32, 25 * K + 64, 25 * K + 96, 25 * K + 128, 25 * K + 160, 25 * K + 192],
            [25 * K, 32, 32, 32, 32, 32, 32, 32],
            3,
        ),
        # Tail-heavy: last expert takes the full max_tokens, others are tiny.
        (
            [0, 512, 1024, 1536, 2048, 2560, 3072, 4 * K],
            [512, 512, 512, 512, 512, 512, 512, 25 * K],
            7,
        ),
        # Irregular: mix of tile-aligned and non-tile-aligned counts.
        (
            [0, 1024, 4032, 4064, 12256, 12288, 12416, 25216],
            [1024, 3000, 17, 8192, 5, 100, 12800, 513],
            6,
        ),
        (
            [0, 1024, 4032, 4064, 12256, 12288, 12416, 25216],
            [1024, 3000, 17, 8192, 5, 100, 12800, 513],
            1,
        ),
    ],
)
def test_extract_stress_dram_utilization(device, starts, counts, expert_id):
    assert len(starts) == NUM_EXPERTS
    assert len(counts) == NUM_EXPERTS

    torch.manual_seed(0)
    global_torch = torch.empty(STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, s, c, global_expert_id=expert_id, max_tokens=STRESS_MAX_TOKENS)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (STRESS_MAX_TOKENS, STRESS_HIDDEN_DIM)
    rows = _ceil_to_tile(counts[expert_id])
    expected = ttnn.to_torch(g)[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_torch[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    original_slice = global_torch[starts[expert_id] : starts[expert_id] + rows, :].float()
    assert_with_pcc(original_slice, out_torch[:rows, :].float(), pcc=0.9999)


@pytest.mark.parametrize("count", [25 * K, 16 * K, 8 * K, 4 * K, 2 * K, 1 * K])
# @pytest.mark.parametrize("count", [1 * K])
def test_extract_stress_dram_utilization_single_expert(device, count):
    starts = [0]
    counts = [count]
    expert_id = 0

    torch.manual_seed(0)
    global_torch = torch.empty(STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, s, c, global_expert_id=expert_id, max_tokens=STRESS_MAX_TOKENS)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (STRESS_MAX_TOKENS, STRESS_HIDDEN_DIM)
    rows = _ceil_to_tile(counts[expert_id])
    expected = ttnn.to_torch(g)[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_torch[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    original_slice = global_torch[starts[expert_id] : starts[expert_id] + rows, :].float()
    assert_with_pcc(original_slice, out_torch[:rows, :].float(), pcc=0.9999)


# ---------------------------------------------------------------------------
# Sub-device confinement (proof of concept).
#
# extract splits work purely by flat core_id / num_cores, so running it on a
# sub-device's worker cores instead of the full grid must produce identical
# output. This exercises the `subdevice_id` plumbing end-to-end.
# ---------------------------------------------------------------------------


def _run_extract_on_subdevice(global_tensor, start, counts, sub_core_range_set, *, expert_id, max_tokens):
    """Run extract confined to `sub_core_range_set`, returning the host output tensor.

    Creates + loads a single-sub-device manager so SubDeviceId(0) resolves to the
    provided cores, runs the op with subdevice_id=SubDeviceId(0), then restores the
    default sub-device manager.
    """
    device = global_tensor.device()
    idx_table = _make_identity_idx_table(device)
    sub_device = ttnn.SubDevice([sub_core_range_set])
    manager_id = device.create_sub_device_manager([sub_device], 0)
    try:
        device.load_sub_device_manager(manager_id)
        out = ttnn.experimental.deepseek_prefill.extract(
            global_tensor,
            start,
            counts,
            idx_table,
            local_expert_id=expert_id,
            max_dispatched_tokens_per_expert=max_tokens,
            subdevice_id=ttnn.SubDeviceId(0),
        )
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        out_host = ttnn.to_torch(out)
    finally:
        device.remove_sub_device_manager(manager_id)
    return out_host


def test_extract_subdevice_matches_full_grid(device):
    starts, counts, expert_id, max_tokens = [0, 32, 64, 96], [32, 17, 32, 5], 1, 64
    hidden_dim = 128
    global_rows = _ceil_to_tile(max(s + c for s, c in zip(starts, counts)) + TILE)

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    # Full-grid reference (subdevice_id defaults to None).
    out_full = ttnn.to_torch(_run(g, s, c, global_expert_id=expert_id, max_tokens=max_tokens))

    # Confine to rows [1, grid_y) — a strict subset mirroring the MoE "compute" strip.
    grid = device.compute_with_storage_grid_size()
    assert grid.y > 1, "test requires a compute grid with at least 2 rows"
    sub_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    out_sd = _run_extract_on_subdevice(g, s, c, sub_cores, expert_id=expert_id, max_tokens=max_tokens)

    assert out_sd.shape == out_full.shape, f"{out_sd.shape} vs {out_full.shape}"
    # Sub-device-confined output must be bit-identical to the full-grid output.
    torch.testing.assert_close(out_sd.float(), out_full.float(), atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Preallocated output (`optional_output_tensor`).
#
# When the caller passes a preallocated DRAM buffer, extract writes into it
# instead of allocating a fresh output. The returned tensor must alias that
# buffer (same DRAM address), and reusing one buffer across calls must keep the
# address stable. This is what lets a caller hold the extract-output allocation
# alive across calls (no per-call free + realloc of the tokens buffer).
# ---------------------------------------------------------------------------


def _make_output_buffer(device, max_tokens, hidden_dim, leading_dims=()):
    """Preallocated extract-output buffer matching the op's output spec:
    [*leading_dims, max_tokens, hidden_dim], BFLOAT8_B, TILE, DRAM interleaved."""
    shape = (*leading_dims, max_tokens, hidden_dim)
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "starts, counts, expert_id, max_tokens",
    [
        ([0, 32, 64, 96], [32, 32, 32, 32], 0, 32),
        ([0, 64, 96, 128], [32, 17, 32, 5], 1, 32),  # counts rounds up 17 -> 32 tile-rows
        ([0, 32, 64, 96], [96, 32, 32, 32], 0, 128),  # full slice, output padded
    ],
)
def test_extract_into_preallocated_output_matches_torch_slice(device, starts, counts, expert_id, max_tokens):
    """Passing optional_output_tensor produces the same slice as the allocating path,
    and the op writes into the caller's buffer (returned tensor aliases it)."""
    hidden_dim = 128
    global_rows = _ceil_to_tile(max(s + c for s, c in zip(starts, counts)) + TILE)

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)
    idx_table = _make_identity_idx_table(device)

    out_buf = _make_output_buffer(device, max_tokens, hidden_dim)
    out = ttnn.experimental.deepseek_prefill.extract(
        g,
        s,
        c,
        idx_table,
        local_expert_id=expert_id,
        max_dispatched_tokens_per_expert=max_tokens,
        optional_output_tensor=out_buf,
    )

    # Returned tensor must alias the caller-provided buffer (no fresh allocation).
    assert out.buffer_address() == out_buf.buffer_address()

    rows = _ceil_to_tile(counts[expert_id])
    global_quantized = ttnn.to_torch(g)
    expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
    # Reading back the PREALLOCATED handle (not the return value) proves the op wrote
    # into the caller's buffer in place.
    buf_host = ttnn.to_torch(out_buf)
    assert buf_host.shape == (max_tokens, hidden_dim)
    torch.testing.assert_close(buf_host[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    # And the returned handle reads back identically.
    out_host = ttnn.to_torch(out)
    torch.testing.assert_close(out_host[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)


def test_extract_preallocated_output_matches_allocating_path(device):
    """The preallocated-output path produces the same VALID rows as the default allocating
    path. Only rows [0, ceil_tile(counts)) are compared — the op leaves the padding rows
    beyond that uninitialized, so they legitimately differ between a zero-filled preallocated
    buffer and a freshly-allocated one."""
    starts, counts, expert_id, max_tokens = [0, 32, 64, 96], [32, 17, 32, 5], 1, 64
    hidden_dim = 128
    global_rows = _ceil_to_tile(max(s + c for s, c in zip(starts, counts)) + TILE)

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out_alloc = ttnn.to_torch(_run(g, s, c, global_expert_id=expert_id, max_tokens=max_tokens))

    out_buf = _make_output_buffer(device, max_tokens, hidden_dim)
    out_pre = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.extract(
            g,
            s,
            c,
            _make_identity_idx_table(device),
            local_expert_id=expert_id,
            max_dispatched_tokens_per_expert=max_tokens,
            optional_output_tensor=out_buf,
        )
    )
    assert out_pre.shape == out_alloc.shape
    rows = _ceil_to_tile(counts[expert_id])
    torch.testing.assert_close(out_pre[:rows, :].float(), out_alloc[:rows, :].float(), atol=0.0, rtol=0.0)


def test_extract_preallocated_output_reused_across_experts(device):
    """One preallocated buffer reused across calls for different experts: each call
    overwrites it with the correct slice, and its DRAM address stays stable (the
    allocation is never freed/reallocated between calls)."""
    starts = [0, 32, 64, 96]
    counts = [32, 17, 32, 5]
    max_tokens = 64
    hidden_dim = 128
    global_rows = _ceil_to_tile(max(s + c for s, c in zip(starts, counts)) + TILE)

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)
    idx_table = _make_identity_idx_table(device)
    global_quantized = ttnn.to_torch(g)

    out_buf = _make_output_buffer(device, max_tokens, hidden_dim)
    base_addr = out_buf.buffer_address()

    for expert_id in [0, 1, 2, 3, 1, 0]:  # revisit experts to exercise repeated overwrite
        out = ttnn.experimental.deepseek_prefill.extract(
            g,
            s,
            c,
            idx_table,
            local_expert_id=expert_id,
            max_dispatched_tokens_per_expert=max_tokens,
            optional_output_tensor=out_buf,
        )
        # Same buffer every call — address never changes (no realloc).
        assert out.buffer_address() == base_addr
        assert out_buf.buffer_address() == base_addr

        rows = _ceil_to_tile(counts[expert_id])
        expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
        out_host = ttnn.to_torch(out)
        torch.testing.assert_close(out_host[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("leading_dims", [(1,), (1, 1)])
def test_extract_preallocated_output_leading_singleton_dims(device, leading_dims):
    """Preallocated output carrying leading singleton dims (mirroring the global tensor's
    rank, e.g. (1, 1, max_tokens, hidden)) extracts the correct slice."""
    hidden_dim = 128
    starts = [0, 32, 64, 96]
    counts = [32, 17, 32, 32]
    expert_id = 1
    max_tokens = 32
    global_rows = 128

    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    g = _make_global_from_torch(device, global_torch.reshape(*leading_dims, global_rows, hidden_dim))
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out_buf = _make_output_buffer(device, max_tokens, hidden_dim, leading_dims=leading_dims)
    out = ttnn.experimental.deepseek_prefill.extract(
        g,
        s,
        c,
        _make_identity_idx_table(device),
        local_expert_id=expert_id,
        max_dispatched_tokens_per_expert=max_tokens,
        optional_output_tensor=out_buf,
    )
    assert out.buffer_address() == out_buf.buffer_address()

    out_torch = ttnn.to_torch(out)
    assert out_torch.shape == (*leading_dims, max_tokens, hidden_dim)
    out_flat = out_torch.reshape(max_tokens, hidden_dim)
    rows = _ceil_to_tile(counts[expert_id])
    global_quantized = ttnn.to_torch(g).reshape(global_rows, hidden_dim)
    expected = global_quantized[starts[expert_id] : starts[expert_id] + rows, :]
    torch.testing.assert_close(out_flat[:rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
