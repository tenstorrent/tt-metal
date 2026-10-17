# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unmarked reduction guards (N3 / N5 / N7 / output-spec).

Each case targets one regression the merge-gate selection still misses.
Marks are added only after Sanity history and the promotion checklist.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.reduction import test_topk_stable_adversarial as _topk_adv
from tests.ttnn.utils_for_testing import assert_equal, assert_numeric_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _captured_program_factories(fn):
    """Program-factory type strings recorded while running `fn` with no device dispatch."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    try:
        fn()
    finally:
        captured = ttnn.graph.end_graph_capture()
    return [
        node["params"]["program_factory_type"]
        for node in captured
        if node.get("node_type") == "function_start" and "program_factory_type" in node.get("params", {})
    ]


def _factory_names_contain(factories, token):
    return any(token in name for name in factories)


# ---------------------------------------------------------------------------
# N3 — LLK numerics state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["max", "min"], ids=["max", "min"])
@pytest.mark.parametrize("dim", [0, 1], ids=["dim0", "dim1"])
def test_int32_max_min_batch_dims_with_negatives(device, op_name, dim):
    """INT32 max/min on dims 0 and 1 with negatives. SFPU reduce used to drop bit 11 on these dims."""
    torch.manual_seed(0)
    torch_input = torch.randint(-50_000, 50_000, (4, 3, 32, 32), dtype=torch.int32)
    torch_input[0, 0, 0, 0] = -50_000
    torch_input[3, 2, 16, 16] = -1
    torch_input[1, 1, 31, 31] = torch.iinfo(torch.int32).min + 1

    torch_op = torch.amax if op_name == "max" else torch.amin
    ttnn_op = ttnn.max if op_name == "max" else ttnn.min
    expected = torch_op(torch_input, dim=dim)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn_op(tt_input, dim=dim))
    assert_equal(expected, got)


def test_bf16_uint16_reconfig_roundtrip_bit_exact(device):
    """bf16 → uint16 → bf16 reconfig. A stuck UInt16 low-byte-zero flag turns 1→0.

    0..255 are exact in bf16, so a numeric typecast must come back as the same integers.
    256 and above are not a useful canary here: bf16 cannot represent 257, and 256's
    uint16 encoding already has a zero low byte.
    """
    values = torch.arange(256, dtype=torch.bfloat16).reshape(1, 1, 8, 32)
    tt_bf16 = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_u16 = ttnn.typecast(tt_bf16, ttnn.uint16)
    tt_back = ttnn.typecast(tt_u16, ttnn.bfloat16)

    got_u16 = ttnn.to_torch(tt_u16).reshape(-1)[:256].to(torch.int64)
    got_bf16 = ttnn.to_torch(tt_back).reshape(-1)[:256]
    assert_equal(torch.arange(256, dtype=torch.int64), got_u16)
    assert_equal(values.reshape(-1), got_bf16)


@pytest.mark.parametrize("op", [ttnn.sum, ttnn.mean], ids=["sum", "mean"])
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1, 1, 41, 31), -1),  # both H and W pad to the next tile
        ((1, 1, 785, 144), -2),  # tall padded H — the H-axis-split path
    ],
    ids=["unaligned_hw", "tall_padded_h"],
)
def test_row_major_mean_sum_tile_padded(device, op, shape, dim):
    """ROW_MAJOR mean/sum on an input whose H/W are not tile-aligned, so the reader pads to 32."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_op = torch.sum if op == ttnn.sum else torch.mean
    expected = torch_op(torch_input, dim=dim, keepdim=True)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(op(tt_input, dim=dim, keepdim=True))
    # Small unaligned shapes can take the FPU RM path; tall H takes the accurate split.
    # Bounds match test_row_major_reduce.py mean/sum.
    if op == ttnn.sum:
        rtol, atol = 0.510, 8.16
    else:
        rtol, atol = 0.008, 0.004
    assert_numeric_metrics(
        expected,
        got,
        pcc_threshold=0.999,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=0.003,
    )


_INTEGER_REDUCE_OPS = (
    ("sum", ttnn.sum, torch.sum),
    ("max", ttnn.max, torch.amax),
    ("min", ttnn.min, torch.amin),
    ("mean", ttnn.mean, torch.mean),
    ("std", ttnn.std, torch.std),
    ("var", ttnn.var, torch.var),
)


@pytest.mark.parametrize("op_name, ttnn_op, torch_op", _INTEGER_REDUCE_OPS, ids=[n for n, _, _ in _INTEGER_REDUCE_OPS])
@pytest.mark.parametrize(
    "dtype, torch_dtype",
    [(ttnn.int32, torch.int32), (ttnn.uint32, torch.int64)],
    ids=["int32", "uint32"],
)
def test_integer_reduce_returns_result_or_raises(device, op_name, ttnn_op, torch_op, dtype, torch_dtype):
    """INT32/UINT32 reduce must match torch or raise. A rejected dtype must not come back as zeros."""
    torch.manual_seed(0)
    # Positive values so UINT32 and INT32 share the same host tensor.
    host = torch.randint(1, 1000, (1, 1, 32, 32), dtype=torch.int64)
    torch_input = host.to(torch_dtype)
    tt_input = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    try:
        tt_out = ttnn_op(tt_input, dim=-1)
    except RuntimeError:
        return

    got = ttnn.to_torch(tt_out).to(torch.int64)
    expected = torch_op(torch_input, dim=-1).to(torch.int64)
    assert not (
        torch.all(got == 0) and not torch.all(expected == 0)
    ), f"{op_name} on {dtype} returned silent zeros; expected {expected.flatten()[:8].tolist()}"
    # UINT32 sum/max/min are accepted by validation but run the FPU path, which reinterprets
    # the bits as float and is not integer-exact. The contract for that path is "not silent
    # zeros"; INT32 sum/max/min and every op that returns a result must match torch.
    if dtype == ttnn.uint32 and op_name in ("sum", "max", "min"):
        return
    assert_equal(expected, got)


# ---------------------------------------------------------------------------
# N5 — dispatch shape (graph capture, no timing)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "TopK multi-core eligibility still requires a power-of-two width; "
        "W=4224 selects TopKSingleCoreProgramFactory. Remove this xfail when "
        "non-pow2 widths use TopKMultiCoreProgramFactory."
    ),
    strict=True,
)
def test_topk_non_power_of_two_width_uses_multicore_factory(device):
    """Non-power-of-two width must pick the multi-core factory, not a single-core fallback."""
    # 4224 = 33 tiles: tile-aligned, not a power of two, k=32 stays in the multi-core k<=64 band.
    width = 4224
    torch_input = torch.randn(1, 1, 32, width, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    factories = _captured_program_factories(lambda: ttnn.topk(tt_input, 32, dim=-1, largest=True, sorted=True))
    assert factories, f"no program factory recorded for topk W={width}"
    assert _factory_names_contain(
        factories, "TopKMultiCore"
    ), f"expected TopKMultiCore for non-pow2 W={width}, got {factories}"


def test_row_major_mean_uses_multicore_factory(device):
    """Tall ROW_MAJOR mean must run the multi-core H factory, not a single-core path."""
    # 3136 rows is the EfficientNetB0 global-pool height; enough H-tiles to split across cores.
    torch_input = torch.rand((1, 1, 3136, 64), dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    factories = _captured_program_factories(lambda: ttnn.mean(tt_input, dim=-2, keepdim=False))
    assert factories, "no program factory recorded for ROW_MAJOR mean"
    assert _factory_names_contain(
        factories, "MultiCore"
    ), f"expected a multi-core reduce factory for tall ROW_MAJOR mean, got {factories}"


# ---------------------------------------------------------------------------
# N7 — index ops, exact
# ---------------------------------------------------------------------------


def test_sort_seeded_unique_values_match_torch(device):
    """Seeded unique values have one correct argsort; values and indices must match torch bit-exactly."""
    torch.manual_seed(2026)
    shape = (1, 1, 32, 64)
    # A permutation of 0..2047 is unique and exact in float32, so values and indices have one answer.
    values = torch.randperm(32 * 64, dtype=torch.int64).to(torch.float32).reshape(shape)
    expected_values, expected_indices = torch.sort(values, dim=-1, descending=False)

    tt_input = ttnn.from_torch(values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_values, tt_indices = ttnn.sort(tt_input, dim=-1, descending=False)
    assert_equal(expected_values, ttnn.to_torch(tt_values))
    assert_equal(expected_indices, ttnn.to_torch(tt_indices).to(torch.int64))


@pytest.mark.parametrize(
    "runner",
    [
        lambda device: _topk_adv.test_a1_all_negative_ties(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a1_all_negative_ties(_topk_adv.W_COMPARATOR, False, device),
        lambda device: _topk_adv.test_a2_mixed_sign_ties(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a3_signed_zero(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a4_infinities(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a5_nan_payloads(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a6_tie_straddling_k(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a7_index_extremes(_topk_adv.W_COMPARATOR, True, device),
        lambda device: _topk_adv.test_a7_index_extremes(_topk_adv.W_COMPARATOR, False, device),
        lambda device: _topk_adv.test_a7_zero_at_index_zero_smallest(_topk_adv.W_COMPARATOR, device),
    ],
    ids=[
        "a1_largest",
        "a1_smallest",
        "a2_largest",
        "a3_largest",
        "a4_largest",
        "a5_largest",
        "a6_largest",
        "a7_largest",
        "a7_smallest",
        "a7_zero_at_index_zero",
    ],
)
def test_topk_stable_adversarial_unit_subset(device, runner):
    """W=64 (single-tile comparator) subset of the nightly A1–A7 stable top-k classes."""
    runner(device)


def test_argmax_width_requires_uint32_index(device):
    """W > 65535: the winning index is 65567, which does not fit in 16 bits."""
    # Smallest tile-aligned width whose last index (65567) needs 32 bits. Argmax output is UINT32;
    # a 16-bit store would wrap 65567 to 31.
    width = 65536 + 32
    torch_input = torch.arange(width, dtype=torch.float32).reshape(1, width)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = ttnn.argmax(tt_input, dim=-1, keepdim=False)
    assert tt_out.dtype == ttnn.uint32, f"expected UINT32 indices at W={width}, got {tt_out.dtype}"
    got = ttnn.to_torch(tt_out).to(torch.int64)
    assert got.item() == width - 1, f"expected argmax index {width - 1}, got {got.item()}"


# ---------------------------------------------------------------------------
# Output-spec: runtime spec must keep the requested shard spec / declared layout
# ---------------------------------------------------------------------------


def _height_sharded_w_reduce_configs(device):
    """Height-sharded input and output for a W-reduce that keeps one tile of width."""
    grid = device.compute_with_storage_grid_size()
    num_cores = min(4, grid.x * grid.y)
    assert num_cores >= 2, "need at least two worker cores to pin a sharded output spec"
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    input_spec = ttnn.ShardSpec(shard_grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    output_spec = ttnn.ShardSpec(shard_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_spec)
    output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_spec)
    shape = (1, 1, 32 * num_cores, 64)
    return shape, input_mem, output_mem, output_spec


@pytest.mark.parametrize("op", [ttnn.sum, ttnn.mean], ids=["sum", "mean"])
def test_reduce_sharded_output_keeps_requested_shard_spec(device, op):
    """A sharded output memory_config must keep its shard spec, not just the buffer type."""
    torch.manual_seed(0)
    shape, input_mem, output_mem, output_spec = _height_sharded_w_reduce_configs(device)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem
    )
    tt_out = op(tt_input, dim=-1, keepdim=True, memory_config=output_mem)

    runtime = tt_out.memory_config()
    assert runtime.is_sharded(), "output dropped sharding"
    assert runtime.shard_spec is not None, "output kept a sharded layout but dropped its shard spec"
    assert (
        runtime.shard_spec.grid == output_spec.grid
    ), f"output shard grid {runtime.shard_spec.grid} != requested {output_spec.grid}"
    assert list(runtime.shard_spec.shape) == list(
        output_spec.shape
    ), f"output shard shape {list(runtime.shard_spec.shape)} != requested {list(output_spec.shape)}"
    assert tt_out.spec.layout == ttnn.TILE_LAYOUT
    assert tt_out.spec.dtype == ttnn.bfloat16


def test_argmax_runtime_spec_matches_declared_row_major_uint32(device):
    """Argmax declares UINT32 ROW_MAJOR interleaved; the runtime tensor must match that spec."""
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.argmax(tt_input, dim=-1, keepdim=True)
    assert tt_out.dtype == ttnn.uint32, f"declared output dtype is UINT32, got {tt_out.dtype}"
    assert tt_out.layout == ttnn.ROW_MAJOR_LAYOUT, f"declared output layout is ROW_MAJOR, got {tt_out.layout}"
    assert tt_out.spec.dtype == ttnn.uint32
    assert tt_out.spec.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not tt_out.memory_config().is_sharded(), "declared output memory config is interleaved"
    assert list(tt_out.shape) == [1, 1, 32, 1], f"keepdim argmax shape {list(tt_out.shape)} != [1, 1, 32, 1]"
