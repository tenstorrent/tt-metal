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


def _run(global_tensor, start, counts, global_expert_id=GLOBAL_EXPERT_ID, max_tokens=MAX_TOKENS):
    return ttnn.experimental.deepseek_prefill.extract(
        global_tensor,
        start,
        counts,
        global_expert_id=global_expert_id,
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


# ---------------------------------------------------------------------------
# Global tensor constraints.
# ---------------------------------------------------------------------------


def test_global_tensor_wrong_dtype(device):
    g = _make_global(device, dtype=ttnn.bfloat16)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_global_tensor_must_be_2d(device):
    g = _make_global(device, shape=(1, GLOBAL_ROWS, HIDDEN_DIM))
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_global_tensor_must_be_dram_interleaved(device):
    g = _make_global(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Start tensor constraints.
# ---------------------------------------------------------------------------


def test_start_wrong_dtype(device):
    g = _make_global(device)
    s = _make_index(device, dtype=ttnn.uint16)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_start_must_be_dram_interleaved(device):
    g = _make_global(device)
    s = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_start_2d_first_dim_not_one(device):
    g = _make_global(device)
    s = _make_index(device, shape=(2, NUM_EXPERTS))
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_start_3d_rejected(device):
    g = _make_global(device)
    s = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Counts tensor constraints (same as start tensor).
# ---------------------------------------------------------------------------


def test_counts_wrong_dtype(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, dtype=ttnn.uint16)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_counts_must_be_dram_interleaved(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_counts_2d_first_dim_not_one(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, shape=(2, NUM_EXPERTS))
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_counts_3d_rejected(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device, shape=(1, 1, NUM_EXPERTS))
    with pytest.raises(RuntimeError):
        _run(g, s, c)


# ---------------------------------------------------------------------------
# Cross-tensor and scalar constraints.
# ---------------------------------------------------------------------------


def test_start_and_counts_last_dim_mismatch_1d(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS + 1,))
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_start_and_counts_last_dim_mismatch_2d(device):
    g = _make_global(device)
    s = _make_index(device, shape=(1, NUM_EXPERTS))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_start_and_counts_last_dim_mismatch_mixed_rank(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(1, NUM_EXPERTS + 1))
    with pytest.raises(RuntimeError):
        _run(g, s, c)


def test_global_expert_id_equal_to_last_dim_rejected(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with pytest.raises(RuntimeError):
        _run(g, s, c, global_expert_id=NUM_EXPERTS)


def test_global_expert_id_beyond_last_dim_rejected(device):
    g = _make_global(device)
    s = _make_index(device, shape=(NUM_EXPERTS,))
    c = _make_index(device, shape=(NUM_EXPERTS,))
    with pytest.raises(RuntimeError):
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


def test_max_tokens_zero_rejected(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
        _run(g, s, c, max_tokens=0)


def test_max_tokens_not_tile_aligned_rejected(device):
    g = _make_global(device)
    s = _make_index(device)
    c = _make_index(device)
    with pytest.raises(RuntimeError):
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
    assert_with_pcc(original.float(), out_torch[:rows, :].float(), pcc=0.999)


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
    assert_with_pcc(original.float(), out_torch[:rows, :].float(), pcc=0.999)
