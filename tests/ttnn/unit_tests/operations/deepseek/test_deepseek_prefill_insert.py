# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.insert.

Verifies constraint rejection as well as the functional behavior: for a given
global_expert_id, the op copies local_tensor[:ceil_tile(counts), :] into
global_tensor[start : start + ceil_tile(counts), :] in place and returns a
handle to global_tensor.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


GLOBAL_ROWS = 128
HIDDEN_DIM = 64
LOCAL_ROWS = 64
NUM_EXPERTS = 8
GLOBAL_EXPERT_ID = 0
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


def _make_local(
    device, *, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, shape=(LOCAL_ROWS, HIDDEN_DIM), memory_config=None
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


def _run(global_tensor, local_tensor, start, counts, global_expert_id=GLOBAL_EXPERT_ID, idx_table=None):
    device = global_tensor.device()
    if idx_table is None:
        idx_table = _make_identity_idx_table(device)
    return ttnn.experimental.deepseek_prefill.insert(
        global_tensor,
        local_tensor,
        start,
        counts,
        idx_table,
        local_expert_id=global_expert_id,
    )


# ---------------------------------------------------------------------------
# Valid inputs should succeed.
# ---------------------------------------------------------------------------


def test_valid_inputs_1d_index(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    out = _run(g, l, s, c)
    assert out is not None


def test_valid_inputs_2d_index_first_dim_one(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(1, NUM_EXPERTS))
    c = _make_index(device, shape=(1, NUM_EXPERTS))
    out = _run(g, l, s, c)
    assert out is not None


# ---------------------------------------------------------------------------
# Global tensor constraints.
# ---------------------------------------------------------------------------


def test_global_tensor_wrong_dtype(device):
    g = _make_global(device, dtype=ttnn.bfloat16)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_global_tensor_must_be_2d(device):
    g = _make_global(device, shape=(1, GLOBAL_ROWS, HIDDEN_DIM))
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_global_tensor_must_be_dram_interleaved(device):
    g = _make_global(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


# ---------------------------------------------------------------------------
# Local tensor constraints (same static invariants as global).
# ---------------------------------------------------------------------------


def test_local_tensor_wrong_dtype(device):
    g = _make_global(device)
    l = _make_local(device, dtype=ttnn.bfloat16)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_local_tensor_must_be_2d(device):
    g = _make_global(device)
    l = _make_local(device, shape=(1, LOCAL_ROWS, HIDDEN_DIM))
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_local_tensor_must_be_dram_interleaved(device):
    g = _make_global(device)
    l = _make_local(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_local_tensor_hidden_dim_must_match_global(device):
    g = _make_global(device, shape=(GLOBAL_ROWS, HIDDEN_DIM))
    l = _make_local(device, shape=(LOCAL_ROWS, HIDDEN_DIM * 2))
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


# ---------------------------------------------------------------------------
# Start tensor constraints.
# ---------------------------------------------------------------------------


def test_start_wrong_dtype(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, dtype=ttnn.uint16)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_start_must_be_dram_interleaved(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_start_2d_first_dim_not_one(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(2, NUM_EXPERTS))
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_start_3d_rejected(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


# ---------------------------------------------------------------------------
# Counts tensor constraints (same as start tensor).
# ---------------------------------------------------------------------------


def test_counts_wrong_dtype(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device, dtype=ttnn.uint16)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_counts_must_be_dram_interleaved(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_counts_2d_first_dim_not_one(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device, shape=(2, NUM_EXPERTS))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_counts_3d_rejected(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device)
    c = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


# ---------------------------------------------------------------------------
# Cross-tensor and scalar constraints.
# ---------------------------------------------------------------------------


def test_start_and_counts_last_dim_mismatch_1d(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS + 1,))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_start_and_counts_last_dim_mismatch_2d(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(1, NUM_EXPERTS))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_start_and_counts_last_dim_mismatch_mixed_rank(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c)


def test_global_expert_id_equal_to_last_dim_rejected(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c, global_expert_id=NUM_EXPERTS)


def test_global_expert_id_beyond_last_dim_rejected(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with pytest.raises(RuntimeError):
        _run(g, l, s, c, global_expert_id=NUM_EXPERTS + 5)


def test_global_expert_id_max_valid_accepted(device):
    g = _make_global(device)
    l = _make_local(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    out = _run(g, l, s, c, global_expert_id=NUM_EXPERTS - 1)
    assert out is not None


# ---------------------------------------------------------------------------
# Functional: global[start:start+ceil_tile(counts), :] == local[:ceil_tile(counts), :].
# Rest of global_tensor is left untouched.
# ---------------------------------------------------------------------------


def _make_index_from_values(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_tile_bfp8(device, torch_tensor):
    return ttnn.from_torch(
        torch_tensor,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "starts, counts, expert_id, global_rows, local_rows, hidden_dim",
    [
        ([0, 32, 64, 96], [32, 32, 32, 32], 0, 128, 32, 64),
        ([0, 32, 64, 96], [32, 32, 32, 32], 2, 128, 64, 64),
        ([0, 64, 96, 128], [32, 17, 32, 5], 1, 256, 32, 64),  # counts rounds 17 → 32
        ([0, 64, 96, 128], [32, 40, 32, 32], 1, 256, 64, 64),  # counts rounds 40 → 64
        ([0, 32, 64, 96], [96, 32, 32, 32], 0, 256, 128, 64),  # counts rounds 96 → 96
    ],
)
def test_insert_copies_slice(device, starts, counts, expert_id, global_rows, local_rows, hidden_dim):
    torch.manual_seed(0)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_torch = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)

    g = _to_tile_bfp8(device, global_torch)
    l = _to_tile_bfp8(device, local_torch)
    # Snapshot bfp8_b-quantized inputs for comparison (kernel copy is bit-exact at tile level).
    global_q = ttnn.to_torch(g).clone()
    local_q = ttnn.to_torch(l)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, l, s, c, global_expert_id=expert_id)
    out_torch = ttnn.to_torch(out)

    # Build reference from the quantized inputs and overwrite the slice.
    rows = _ceil_to_tile(counts[expert_id])
    start = starts[expert_id]
    expected = global_q.clone()
    expected[start : start + rows, :] = local_q[:rows, :]

    # Shape preserved (in-place).
    assert out_torch.shape == global_torch.shape, f"shape changed: {out_torch.shape} vs {global_torch.shape}"
    # Full tensor must match the reference everywhere (both inside and outside the slice).
    torch.testing.assert_close(out_torch.float(), expected.float(), atol=0.0, rtol=0.0)
    # PCC against the original (pre-quantization) bfloat16 reference.
    original = global_torch.clone()
    original[start : start + rows, :] = local_torch[:rows, :]
    assert_with_pcc(original.float(), out_torch.float(), pcc=0.9999)


def test_insert_is_in_place(device):
    """Inserting one expert's slice must leave other rows untouched, and the
    returned tensor must reflect all accumulated inserts (in-place contract)."""
    torch.manual_seed(0)
    global_rows, local_rows, hidden_dim = 128, 64, 64
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_a = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_b = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)

    g = _to_tile_bfp8(device, global_torch)
    # Snapshot bfp8_b-quantized global before any insert.
    initial_g = ttnn.to_torch(g).clone()
    l_a = _to_tile_bfp8(device, local_a)
    l_b = _to_tile_bfp8(device, local_b)
    local_a_q = ttnn.to_torch(l_a)
    local_b_q = ttnn.to_torch(l_b)
    s = _make_index_from_values(device, [0, 32, 64, 96])
    c = _make_index_from_values(device, [32, 32, 32, 32])

    # Insert two disjoint slices back-to-back.
    _run(g, l_a, s, c, global_expert_id=0)
    out = _run(g, l_b, s, c, global_expert_id=2)
    out_torch = ttnn.to_torch(out)

    expected = initial_g.clone()
    expected[0:32, :] = local_a_q[:32, :]
    expected[64:96, :] = local_b_q[:32, :]

    assert out_torch.shape == global_torch.shape
    torch.testing.assert_close(out_torch.float(), expected.float(), atol=0.0, rtol=0.0)
    # PCC against the original (pre-quantization) bfloat16 reference.
    original = global_torch.clone()
    original[0:32, :] = local_a[:32, :]
    original[64:96, :] = local_b[:32, :]
    assert_with_pcc(original.float(), out_torch.float(), pcc=0.9999)


def test_insert_2d_indices_matches_torch_slice(device):
    global_rows = 160
    local_rows = 64
    hidden_dim = 64
    starts = [0, 32, 96]
    counts = [32, 48, 32]
    expert_id = 1

    torch.manual_seed(1)
    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_torch = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)

    g = _to_tile_bfp8(device, global_torch)
    l = _to_tile_bfp8(device, local_torch)
    global_q = ttnn.to_torch(g).clone()
    local_q = ttnn.to_torch(l)
    s = ttnn.from_torch(
        torch.tensor([starts], dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = ttnn.from_torch(
        torch.tensor([counts], dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = _run(g, l, s, c, global_expert_id=expert_id)
    out_torch = ttnn.to_torch(out)

    rows = _ceil_to_tile(counts[expert_id])
    expected = global_q.clone()
    expected[starts[expert_id] : starts[expert_id] + rows, :] = local_q[:rows, :]

    assert out_torch.shape == global_torch.shape
    torch.testing.assert_close(out_torch.float(), expected.float(), atol=0.0, rtol=0.0)
    # PCC against the original (pre-quantization) bfloat16 reference.
    original = global_torch.clone()
    original[starts[expert_id] : starts[expert_id] + rows, :] = local_torch[:rows, :]
    assert_with_pcc(original.float(), out_torch.float(), pcc=0.9999)


# ---------------------------------------------------------------------------
# Stress test: DRAM utilization with a large global tensor.
#
# Global tensor is shaped (2 * 25k, 7k) with 1k = 1024. Local tensor holds 25k
# rows. Scenarios cover uniform insertion (every expert gets the same slice)
# and non-uniform insertion (one expert dominates, tail-heavy, irregular
# tile-unaligned counts).
# ---------------------------------------------------------------------------


K = 1024
STRESS_GLOBAL_ROWS = 2 * 25 * K  # 51200
STRESS_HIDDEN_DIM = 7 * K  # 7168
STRESS_LOCAL_ROWS = 25 * K  # 25600


@pytest.mark.parametrize(
    "starts, counts, expert_id",
    [
        # Uniform: each of 8 experts inserts an equal 3k-row slice.
        ([0, 3 * K, 6 * K, 9 * K, 12 * K, 15 * K, 18 * K, 21 * K], [3 * K] * 8, 0),
        ([0, 3 * K, 6 * K, 9 * K, 12 * K, 15 * K, 18 * K, 21 * K], [3 * K] * 8, 5),
        # One expert inserts the full local rows, the rest insert a single tile row each.
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
        # Tail-heavy: last expert inserts the full local rows, others are tiny.
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
def test_insert_stress_dram_utilization(device, starts, counts, expert_id):
    assert len(starts) == NUM_EXPERTS
    assert len(counts) == NUM_EXPERTS

    torch.manual_seed(0)
    global_torch = torch.empty(STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()
    local_torch = torch.empty(STRESS_LOCAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()

    g = _to_tile_bfp8(device, global_torch)
    l = _to_tile_bfp8(device, local_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, l, s, c, global_expert_id=expert_id)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM)
    rows = _ceil_to_tile(counts[expert_id])
    start = starts[expert_id]
    expected = ttnn.to_torch(l)[:rows, :]
    torch.testing.assert_close(out_torch[start : start + rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    local_slice = local_torch[:rows, :].float()
    assert_with_pcc(local_slice, out_torch[start : start + rows, :].float(), pcc=0.9999)


@pytest.mark.parametrize("count", [25 * K, 16 * K, 8 * K, 4 * K, 2 * K, 1 * K])
def test_insert_stress_dram_utilization_single_expert(device, count):
    starts = [0]
    counts = [count]
    expert_id = 0

    torch.manual_seed(0)
    global_torch = torch.empty(STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()
    local_torch = torch.empty(STRESS_LOCAL_ROWS, STRESS_HIDDEN_DIM, dtype=torch.bfloat16).normal_()

    g = _to_tile_bfp8(device, global_torch)
    l = _to_tile_bfp8(device, local_torch)
    s = _make_index_from_values(device, starts)
    c = _make_index_from_values(device, counts)

    out = _run(g, l, s, c, global_expert_id=expert_id)
    out_torch = ttnn.to_torch(out)

    assert out_torch.shape == (STRESS_GLOBAL_ROWS, STRESS_HIDDEN_DIM)
    rows = _ceil_to_tile(counts[expert_id])
    start = starts[expert_id]
    expected = ttnn.to_torch(l)[:rows, :]
    torch.testing.assert_close(out_torch[start : start + rows, :].float(), expected.float(), atol=0.0, rtol=0.0)
    local_slice = local_torch[:rows, :].float()
    assert_with_pcc(local_slice, out_torch[start : start + rows, :].float(), pcc=0.9999)
