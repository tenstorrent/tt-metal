# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Regression tests for PR #63020: composite data-movement ops must write into
# the caller's preallocated `out=` buffer, not into a replacement tensor the
# host wrapper allocated. For each site, we grab the preallocated buffer
# address before the call and assert it is unchanged after. If the op returns
# a different buffer than the caller provided, this is the exact bug the PR
# fixes.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


# ---------- accumulation (cumsum / cumprod) ----------


@pytest.mark.parametrize("op_name,torch_op", [("cumsum", torch.cumsum), ("cumprod", torch.cumprod)])
def test_accumulation_direct_write_preserves_buffer_on_noop_preprocess(op_name, torch_op, device):
    # 4D bf16 scan along a tile axis (dim=-1): preprocess is a no-op, so the
    # primitive should write straight into out=. Buffer address must match.
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    addr_before = tt_out.buffer_address()

    tt_result = getattr(ttnn, op_name)(tt_input, dim=-1, out=tt_out)

    assert tt_result.buffer_address() == addr_before, (
        f"{op_name} with no-op preprocess replaced caller buffer: "
        f"before={addr_before}, after={tt_result.buffer_address()}"
    )
    assert_equal(ttnn.to_torch(tt_result), torch_op(torch_input, dim=-1).to(torch.bfloat16))


@pytest.mark.parametrize("op_name,torch_op", [("cumsum", torch.cumsum), ("cumprod", torch.cumprod)])
def test_accumulation_fallback_copies_into_preallocated_buffer(op_name, torch_op, device):
    # dim=0 on a 4D tensor forces preprocess to permute cum_axis to dim 0, so the
    # direct-write path cannot fire. The fallback must copy into out= after the
    # primitive and postprocess_output_tensor run; out= buffer address must be
    # preserved.
    torch.manual_seed(1)
    shape = (8, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    addr_before = tt_out.buffer_address()

    tt_result = getattr(ttnn, op_name)(tt_input, dim=0, out=tt_out)

    assert tt_result.buffer_address() == addr_before, (
        f"{op_name} fallback path replaced caller buffer: "
        f"before={addr_before}, after={tt_result.buffer_address()}"
    )
    assert_equal(ttnn.to_torch(tt_result), torch_op(torch_input, dim=0).to(torch.bfloat16))


def test_accumulation_rejects_dtype_mismatched_out(device):
    # A preallocated out= with a dtype that doesn't match the computed output
    # dtype used to be silently rewritten by the trailing ttnn::copy; it now
    # fails fast in validate_output_tensor.
    torch.manual_seed(2)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out_wrong_dtype = ttnn.zeros(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="dtype"):
        ttnn.cumsum(tt_input, dim=-1, out=tt_out_wrong_dtype)


# ---------- padded_slice ----------


def test_padded_slice_noop_fastpath_preserves_preallocated_buffer(device):
    # Slice covers the full input shape with step=1, ROW_MAJOR input: the no-op
    # fast path returns finalize_into_preallocated(input). If out= is sharded
    # with matching layout, its buffer must be the one that comes back.
    torch.manual_seed(3)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    shard_spec = ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    tt_out = ttnn.zeros(
        shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=sharded_mem_config
    )
    addr_before = tt_out.buffer_address()

    tt_result = ttnn.experimental.padded_slice(
        tt_input,
        begins=[0, 0, 0, 0],
        ends=list(shape),
        step=[1, 1, 1, 1],
        memory_config=sharded_mem_config,
        optional_output_tensor=tt_out,
    )

    assert tt_result.buffer_address() == addr_before, (
        f"padded_slice no-op fast path returned a different buffer: "
        f"before={addr_before}, after={tt_result.buffer_address()}"
    )


# ---------- embedding ----------


def test_embedding_forwards_out_when_spec_matches(device):
    # Weight is TILE_LAYOUT and bfloat16. Request layout=TILE and no dtype
    # override: final spec is (batch, sentence, hidden), TILE_LAYOUT, bfloat16.
    # If out= is preallocated with that spec, the host wrapper must forward it
    # to the primitive, and no post-primitive adjustment runs. Buffer must be
    # preserved.
    torch.manual_seed(4)
    batch_size, sentence_size, vocab, hidden = 1, 32, 128, 64
    torch_indices = torch.randint(0, vocab, (batch_size, sentence_size), dtype=torch.int32)
    torch_weight = torch.randn((vocab, hidden), dtype=torch.bfloat16)

    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.zeros(
        (batch_size, sentence_size, hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    addr_before = tt_out.buffer_address()

    tt_result = ttnn.embedding(
        tt_indices,
        tt_weight,
        layout=ttnn.TILE_LAYOUT,
        output_tensor=tt_out,
    )

    assert tt_result.buffer_address() == addr_before, (
        f"embedding matching-spec path returned a different buffer: "
        f"before={addr_before}, after={tt_result.buffer_address()}"
    )


def test_embedding_layout_adjust_copies_into_out(device):
    # Weight is ROW_MAJOR but the caller wants TILE output: to_layout runs on
    # the primitive output, producing a new buffer. The host wrapper must then
    # copy that into the caller's out= instead of returning the adjusted buffer.
    # Buffer address of out= must be preserved.
    torch.manual_seed(5)
    batch_size, sentence_size, vocab, hidden = 1, 32, 128, 64
    torch_indices = torch.randint(0, vocab, (batch_size, sentence_size), dtype=torch.int32)
    torch_weight = torch.randn((vocab, hidden), dtype=torch.bfloat16)

    tt_indices = ttnn.from_torch(torch_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = ttnn.zeros(
        (batch_size, sentence_size, hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    addr_before = tt_out.buffer_address()

    tt_result = ttnn.embedding(
        tt_indices,
        tt_weight,
        layout=ttnn.TILE_LAYOUT,
        output_tensor=tt_out,
    )

    assert tt_result.buffer_address() == addr_before, (
        f"embedding layout-adjust path replaced caller buffer: "
        f"before={addr_before}, after={tt_result.buffer_address()}"
    )
