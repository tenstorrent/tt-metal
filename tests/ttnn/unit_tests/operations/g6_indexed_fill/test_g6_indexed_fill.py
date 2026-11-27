# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_indexed_fill: the immutable spec of the op.

y equals input_tensor except at the batches batch_ids names (read on the device, unsigned), which
take value_tensor's batches in order; an entry not below B writes nothing. Values are moved, never
computed, so every element is graded by bit equality over the whole uint32 range.

Scope follows the campaign rule BB-22: the production case and the cases the spec's rules require.
The regime-pinned plans of op_design.md reuse the rule-9 shape through PLAN_CAPS instead of adding
shapes.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_indexed_fill import PLAN_CAPS, describe_plan, g6_indexed_fill

# The module-scoped device takes its parameters from the module's own marker; rule 1 is shown by a
# trace replay, which needs a trace region.
pytestmark = pytest.mark.use_module_device({"trace_region_size": 1 << 20})

L1_BOUND = 1347776
UINT32_MAX = 2**32 - 1
NOT_FLOAT32 = 2**24 + 1  # the smallest positive integer float32 cannot hold

# (ids_shape, input_shape, value_shape)
PRODUCTION = ((1,), (7, 1, 1, 32), (1, 1, 1, 32))
OTHER_SHAPE = ((2,), (5, 2, 1, 64), (2, 2, 1, 64))


# --- Reference and tensors ---------------------------------------------------------------------


def reference(ids, x, s, *, last_wins=True):
    """x with batch ids[i] replaced by s[i] for every ids[i] < B (int64 values, ids a list of ints)."""
    out = x.clone()
    order = list(enumerate(ids))
    if not last_wins:
        order.reverse()
    for i, b in order:
        if b < x.shape[0]:
            out[b] = s[i]
    return out


def make_inputs(input_shape, value_shape):
    """Full-range uint32 values held as int64.

    Every batch carries UINT32_MAX and NOT_FLOAT32, and no element of s equals the element of x at
    the same position of any batch, so a missed or misplaced write always shows.
    """
    x = torch.randint(0, 2**32, input_shape, dtype=torch.int64)
    s = torch.randint(0, 2**32, value_shape, dtype=torch.int64)
    xr, sr = x.view(x.shape[0], -1), s.view(s.shape[0], -1)
    xr[:, 0], xr[:, -1] = UINT32_MAX, NOT_FLOAT32
    sr[:, 0], sr[:, -1] = NOT_FLOAT32, UINT32_MAX
    while True:
        clash = (sr.unsqueeze(1) == xr.unsqueeze(0)).any(dim=1)
        if not clash.any():
            return x, s
        sr[clash] = torch.randint(0, 2**32, (int(clash.sum()),), dtype=torch.int64)


def to_device(
    values,
    device,
    *,
    dtype=ttnn.uint32,
    torch_dtype=torch.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        values.to(torch_dtype), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )


def ids_on_device(ids, device):
    return to_device(torch.tensor(ids, dtype=torch.int64), device)


def ids_on_host(ids):
    return ttnn.from_torch(
        torch.tensor(ids, dtype=torch.int64).to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def setup(device, shapes):
    _, input_shape, value_shape = shapes
    x, s = make_inputs(input_shape, value_shape)
    return x, s, to_device(x, device), to_device(s, device)


def read_back(tensor):
    t = ttnn.to_torch(tensor)
    if t.dtype == torch.uint32:
        return t.to(torch.int64)
    assert t.dtype == torch.int32, f"unexpected readback dtype {t.dtype}"
    return t.to(torch.int64) & 0xFFFFFFFF


def check(output, expected, label):
    assert list(output.shape) == list(expected.shape), f"[{label}] shape {list(output.shape)}"
    assert output.dtype == ttnn.uint32, f"[{label}] dtype {output.dtype}"
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT, f"[{label}] layout {output.layout}"
    memory_config = output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM, f"[{label}] {memory_config}"
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"[{label}] {memory_config}"
    got = read_back(output)
    if not torch.equal(got, expected):
        where = (got != expected).nonzero()
        first = tuple(where[0].tolist())
        batches = sorted({int(b) for b in where[:, 0]})
        pytest.fail(
            f"[{label}] {len(where)} elements differ, batches {batches}, first at {list(first)}: "
            f"got {int(got[first])} expected {int(expected[first])}"
        )


def check_unchanged(tensor, host, name, label):
    assert torch.equal(read_back(tensor), host), f"[{label}] {name} was changed by the call"


# --- Production case ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(7))
def test_production_every_index(device, index):
    """The model's call at every position: a kernel with a fixed position passes one index only."""
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, PRODUCTION)
    output = g6_indexed_fill(ids_on_device([index], device), x_dev, s_dev, dim=0)
    check(output, reference([index], x, s), f"batch_ids=[{index}]")


@pytest.mark.parametrize("index", [7, UINT32_MAX], ids=["B", "uint32_max"])
def test_index_out_of_range_writes_nothing(device, index):
    """Rule 3: an entry not below B, read unsigned, writes nothing and no byte outside the result."""
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, PRODUCTION)
    ids_dev = ids_on_device([index], device)
    shape = PRODUCTION[1]
    low_host = torch.randint(0, 2**32, shape, dtype=torch.int64)
    high_host = torch.randint(0, 2**32, shape, dtype=torch.int64)
    low = to_device(low_host, device)
    hole = to_device(torch.zeros(shape, dtype=torch.int64), device)
    high = to_device(high_host, device)
    # The result's allocation lands in the freed hole between the guards, so a write past either
    # end of the result lands in a guard.
    ttnn.deallocate(hole)
    label = f"batch_ids=[{index}]"
    output = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
    check(output, x, label)
    check_unchanged(low, low_host, "guard_before", label)
    check_unchanged(high, high_host, "guard_after", label)
    check_unchanged(x_dev, x, "input_tensor", label)
    check_unchanged(s_dev, s, "value_tensor", label)


def test_new_result_inputs_unchanged(device):
    """Rule 4 and the call without dim: a new result; input_tensor and value_tensor unchanged."""
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, PRODUCTION)
    label = "batch_ids=[3], dim by default"
    output = g6_indexed_fill(ids_on_device([3], device), x_dev, s_dev)
    check(output, reference([3], x, s), label)
    assert output.buffer_address() != x_dev.buffer_address(), f"[{label}] the result must be a new tensor"
    check_unchanged(x_dev, x, "input_tensor", label)
    check_unchanged(s_dev, s, "value_tensor", label)


def test_index_rewritten_on_device_and_trace_replay(device):
    """Rule 1: batch_ids is read on the device in every call.

    One batch_ids buffer is rewritten in place between cached calls and between replays of one
    captured trace; a position read on the host or frozen at the first call fails. A new index
    value must not add a program to the cache.
    """
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, PRODUCTION)
    ids_dev = ids_on_device([2], device)
    # Every result stays allocated: a freed buffer handed back still holding a right answer would
    # let a call that wrote nothing pass.
    results = []
    cache_entries = None
    for index in [2, 5, 5, 0]:
        ttnn.copy_host_to_device_tensor(ids_on_host([index]), ids_dev)
        output = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
        results.append(output)
        check(output, reference([index], x, s), f"eager batch_ids=[{index}]")
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == cache_entries, f"batch_ids=[{index}] built a new program"

    ttnn.copy_host_to_device_tensor(ids_on_host([3]), ids_dev)
    trace_id = ttnn.begin_trace_capture(device)
    try:
        try:
            traced = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
        finally:
            ttnn.end_trace_capture(device, trace_id)
        for index in [3, 6]:
            ttnn.copy_host_to_device_tensor(ids_on_host([index]), ids_dev)
            ttnn.execute_trace(device, trace_id, blocking=True)
            check(traced, reference([index], x, s), f"trace replay batch_ids=[{index}]")
    finally:
        ttnn.release_trace(device, trace_id)


def test_repeat_runs_bit_identical(device):
    """BB-06: the same call repeated returns the same bits on every run."""
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, PRODUCTION)
    ids_dev = ids_on_device([4], device)
    expected = reference([4], x, s)
    results = []
    for run in range(20):
        output = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
        results.append(output)
        check(output, expected, f"batch_ids=[4] run {run}")


def _refused_call(case, device, ids_dev, x_dev, s_dev):
    zeros = lambda shape: torch.zeros(shape, dtype=torch.int64)
    if case == "dim":
        return lambda: g6_indexed_fill(ids_dev, x_dev, s_dev, dim=1)
    if case == "dim_negative":
        return lambda: g6_indexed_fill(ids_dev, x_dev, s_dev, dim=-4)
    if case == "value_batches":
        value = to_device(zeros((2, 1, 1, 32)), device)
        return lambda: g6_indexed_fill(ids_dev, x_dev, value, dim=0)
    if case == "value_row_width":
        value = to_device(zeros((1, 1, 1, 64)), device)
        return lambda: g6_indexed_fill(ids_dev, x_dev, value, dim=0)
    if case == "ids_rank":
        ids = to_device(zeros((1, 1)), device)
        return lambda: g6_indexed_fill(ids, x_dev, s_dev, dim=0)
    if case == "input_dtype":
        x = to_device(zeros(PRODUCTION[1]), device, dtype=ttnn.int32, torch_dtype=torch.int32)
        return lambda: g6_indexed_fill(ids_dev, x, s_dev, dim=0)
    if case == "input_layout":
        x = to_device(zeros(PRODUCTION[1]), device, layout=ttnn.TILE_LAYOUT)
        return lambda: g6_indexed_fill(ids_dev, x, s_dev, dim=0)
    if case == "input_l1":
        x = to_device(zeros(PRODUCTION[1]), device, memory_config=ttnn.L1_MEMORY_CONFIG)
        return lambda: g6_indexed_fill(ids_dev, x, s_dev, dim=0)
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    ["dim", "dim_negative", "value_batches", "value_row_width", "ids_rank", "input_dtype", "input_layout", "input_l1"],
)
def test_refusals(device, expect_error, case):
    """Rule 7: dim other than 0, a tensor that is not uint32 / ROW_MAJOR / DRAM interleaved, and
    shapes that do not fit together are refused on the host."""
    torch.manual_seed(42)
    _, _, x_dev, s_dev = setup(device, PRODUCTION)
    call = _refused_call(case, device, ids_on_device([3], device), x_dev, s_dev)
    # The contract fixes the exception types, not the message text.
    with expect_error((ValueError, RuntimeError), ".*"):
        call()


# --- Rule 9 shape ------------------------------------------------------------------------------


def test_other_shape(device):
    """Rule 9: K = 2, two rows per batch, 256-B pages; a kernel fixed to the production extents fails."""
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, OTHER_SHAPE)
    output = g6_indexed_fill(ids_on_device([4, 1], device), x_dev, s_dev, dim=0)
    check(output, reference([4, 1], x, s), "batch_ids=[4, 1]")


# --- Plans (BB-05, BB-08, BB-17, BB-21) and regime-pinned runs -----------------------------------


def test_production_plan(device):
    """The production call reports its plan: the resident-index regime, one rectangle of cores, one
    block per core, the L1 bound, and a role for every core that covers every unit once."""
    torch.manual_seed(42)
    _, _, x_dev, s_dev = setup(device, PRODUCTION)
    plan = describe_plan(ids_on_device([3], device), x_dev, s_dev, dim=0)
    assert plan["regime"] == "ids_resident"
    assert plan["core_ranges"] == 1
    assert plan["max_blocks_per_core"] == 1
    assert plan["l1_bytes_per_core"] <= L1_BOUND
    assert len(plan["roles"]) == plan["num_cores"]
    assert sum(role["unit_count"] for role in plan["roles"]) == plan["units"]


# name -> (PLAN_CAPS overrides, plan fields they must produce on OTHER_SHAPE)
FORCED_PLANS = {
    "ids_streamed": ({"ids_chunk_cap_entries": 1}, {"regime": "ids_streamed", "ids_chunks": 2}),
    "multi_block_depth2": ({"max_work_cores": 2, "staging_budget_bytes": 512}, {"staging_depth": 2}),
    "multi_block_depth1": (
        {"max_work_cores": 2, "staging_budget_bytes": 768, "max_staging_depth": 1},
        {"staging_depth": 1},
    ),
    "row_chunks": ({"max_row_chunk_bytes": 64}, {"row_chunks_per_page": 4}),
    "combined": (
        {"ids_chunk_cap_entries": 1, "max_row_chunk_bytes": 64, "max_work_cores": 2, "staging_budget_bytes": 512},
        {"regime": "ids_streamed", "staging_depth": 2, "row_chunks_per_page": 4},
    ),
}
MULTI_BLOCK_PLANS = {"multi_block_depth2", "multi_block_depth1", "combined"}


def _force_plan(monkeypatch, name):
    caps, _ = FORCED_PLANS[name]
    for key, value in caps.items():
        monkeypatch.setitem(PLAN_CAPS, key, value)


@pytest.mark.parametrize("ids", [[4, 1], [UINT32_MAX, 2]], ids=["valid", "one_out_of_range"])
@pytest.mark.parametrize("plan_name", sorted(FORCED_PLANS))
def test_forced_plan(device, monkeypatch, plan_name, ids):
    """Each regime and knob path of op_design.md, pinned on the rule-9 shape."""
    _force_plan(monkeypatch, plan_name)
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, OTHER_SHAPE)
    ids_dev = ids_on_device(ids, device)
    plan = describe_plan(ids_dev, x_dev, s_dev, dim=0)
    for key, value in FORCED_PLANS[plan_name][1].items():
        assert plan[key] == value, f"[{plan_name}] plan[{key!r}] = {plan[key]!r}, expected {value!r}"
    if plan_name in MULTI_BLOCK_PLANS:
        assert plan["max_blocks_per_core"] > 1, f"[{plan_name}] {plan}"
    assert plan["l1_bytes_per_core"] <= L1_BOUND
    output = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
    check(output, reference(ids, x, s), f"{plan_name} batch_ids={ids}")


@pytest.mark.parametrize("plan_name", [None, "combined"], ids=["default_plan", "combined_plan"])
def test_repeated_entry_is_deterministic(device, monkeypatch, plan_name):
    """Two equal entries: the batch takes the row of one of them, the same one on every run."""
    if plan_name is not None:
        _force_plan(monkeypatch, plan_name)
    torch.manual_seed(42)
    x, s, x_dev, s_dev = setup(device, OTHER_SHAPE)
    ids_dev = ids_on_device([3, 3], device)
    first = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
    got = read_back(first)
    candidates = [reference([3, 3], x, s, last_wins=True), reference([3, 3], x, s, last_wins=False)]
    assert any(torch.equal(got, c) for c in candidates), "batch 3 holds neither entry's rows"
    results = [first]
    for run in range(5):
        again = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
        results.append(again)
        assert torch.equal(read_back(again), got), f"run {run} differs from the first run"
