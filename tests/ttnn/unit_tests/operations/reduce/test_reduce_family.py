# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One table-driven suite for the arithmetic reduction op class.

Replaces the four per-op cartesian grids (test_sum.py::test_sum/_global/_4d,
test_max.py::test_max/_2d/_4d/_global/_dim, test_reduction_min.py::test_min/
_global, test_reduction_mean.py::test_mean/_scaling/_scaling_factor) and the
std/var grids in test_reduction.py, which together collected ~1,400 cases of
the same tile-interleaved happy path differing only in the op.

Structure: CELLS is a curated list of named (shape, dim, keepdim) rows — each
row exists because it selects a distinct program-factory branch or sits on a
boundary that produced an escape (issue numbers inline). The op is a genuine
axis (same reduce factories, different pool math), so crossing op x cell x
dtype is meaningful, unlike the old per-op grids. No xfail/skip decorators:
a failure here is a real gap.

Also carries the escape-ledger boundary cells for topk (multi-core row counts,
index-dtype selection — issues #53453/#53466) and the model-config pins for
the load-bearing configurations real models run (greedy-decode argmax at vocab
scale, MoE-combine sum(dim=0), sampling with a preallocated output, topk
stable= as a tie-order contract wired to #33492).

Kept in their own files (unique axes, not duplicates): ND/BLOCK/HEIGHT
sharding (test_sum_nd_shard, test_mean_shard, test_reduction_on_batch),
sub_core_grids (test_sum_subcores), fp32 fast_and_approximate_mode
(test_max.py), RM-layout suites (test_row_major_reduce.py), regression pins
(test_min_row_major #32829, test_min_multi_dim #40854), program-cache suites,
and the topk functional grid (test_topk.py).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_equal, assert_numeric_metrics, assert_with_ulp

# Poison for implicit tile padding: per-op value that WOULD corrupt the result
# if the op's own identity-fill of the padding ever regressed.
POISON = {"sum": 42.0, "mean": 42.0, "max": 142.0, "min": -142.0, "std": 42.0, "var": 42.0}

TTNN_OP = {"sum": ttnn.sum, "mean": ttnn.mean, "max": ttnn.max, "min": ttnn.min, "std": ttnn.std, "var": ttnn.var}


def _torch_reduce(op, x, dim, keepdim, correction=True):
    if op == "sum":
        return torch.sum(x, dim=dim, keepdim=keepdim) if dim is not None else torch.sum(x)
    if op == "mean":
        return torch.mean(x, dim=dim, keepdim=keepdim) if dim is not None else torch.mean(x)
    if op == "max":
        return torch.amax(x, dim=dim, keepdim=keepdim) if dim is not None else torch.max(x)
    if op == "min":
        return torch.amin(x, dim=dim, keepdim=keepdim) if dim is not None else torch.min(x)
    if op == "std":
        return torch.std(x, dim=dim, keepdim=keepdim, correction=correction)
    assert op == "var"
    return torch.var(x, dim=dim, keepdim=keepdim, correction=correction)


# Each cell names the dispatch branch / boundary it covers. Shapes use
# non-tile-aligned extents (41/37/63/31, sub-tile 6/7) wherever the branch
# allows, so padding handling rides along for free.
#   id                     shape                dim         keepdim
CELLS = [
    ("w_multicore", (1, 64, 64), -1, True),  # ReduceW factory, aligned
    ("w_unaligned_batch", (16, 41, 63), -1, False),  # ReduceW, padded + batch
    ("h_multicore", (1, 64, 64), -2, False),  # ReduceH factory
    ("h_unaligned_batch", (16, 37, 31), -2, True),  # ReduceH, padded + batch
    ("hw_pair", (16, 41, 63), (-2, -1), True),  # HW two-step (multi-tile)
    ("rank2_w", (37, 63), -1, False),  # rank-2 squeeze/unsqueeze
    ("subtile_w", (2, 6, 7), -1, False),  # single sub-tile shape
    ("rank3_dim0", (32, 6, 7), 0, True),  # NC reduce on rank 3
    ("rank4_nc_dim0", (9, 2, 37, 63), 0, False),  # fast_reduce_nc (sum-bf16) / per-axis loop
    ("rank4_nc_dim1", (2, 9, 37, 63), 1, False),  # same, inner batch dim
    ("rank4_dims01", (2, 4, 37, 63), (0, 1), False),  # multi-axis, both NC
    ("rank4_dims012", (2, 4, 37, 63), (0, 1, 2), False),  # multi-axis, NC + H
    ("rank4_dims023", (2, 4, 37, 63), (0, 2, 3), True),  # escape #23876 (mean wrong on (0,2,3))
    ("rank4_noncontig", (2, 4, 32, 63), (0, 3), False),  # non-contiguous dim set
    ("rank5_w", (2, 2, 3, 37, 63), -1, False),  # rank-5 squeeze path
    ("global_all", (16, 41, 63), None, False),  # dim=None; escape #32274 (fp32)
]

_CELL_IDS = [c[0] for c in CELLS]

# Tolerances inherited from the grids this file replaces (see module docstring);
# frobenius + pcc are the scale-invariant anchors, rtol/atol are backstops.
_SUM_TOL = {
    ttnn.bfloat16: dict(pcc_threshold=0.999, rtol=2.5, atol=70.0, frobenius_threshold=0.013),
    ttnn.float32: dict(pcc_threshold=0.999, rtol=0.02, atol=33.0, frobenius_threshold=0.02),
    ttnn.bfloat8_b: dict(pcc_threshold=0.999, rtol=0.1, atol=230.0, frobenius_threshold=0.068),
}
_MEAN_TOL = {
    ttnn.bfloat16: dict(pcc_threshold=0.999, rtol=0.118, atol=0.002, frobenius_threshold=0.005),
    ttnn.float32: dict(pcc_threshold=0.9999, rtol=0.01, atol=1e-4, frobenius_threshold=0.001),
}
# fp32 mean on batch/channel dims takes the NC path with a bf16-rounded 1/N
# scaler (~1e-3 error), so those cells get the bf16-grade tolerances.
_MEAN_TOL_FP32_NC = dict(pcc_threshold=0.999, rtol=0.01, atol=0.002, frobenius_threshold=0.005)


def _has_nc_dim(shape, dim):
    rank = len(shape)
    if dim is None:
        return rank > 2
    axes = [dim] if isinstance(dim, int) else list(dim)
    return any((a % rank) < rank - 2 for a in axes)


@pytest.mark.parametrize("op", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("cell", CELLS, ids=_CELL_IDS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_reduce_family(device, op, cell, dtype):
    _, shape, dim, keepdim = cell
    torch.manual_seed(0)

    if op == "mean":
        torch_input = torch_random(shape, -1, 1, dtype=torch.bfloat16)
    else:
        torch_input = torch_random(shape, -100, 100, dtype=torch.bfloat16)
    torch_output = _torch_reduce(op, torch_input, dim, keepdim)

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, POISON[op])

    output_tensor = TTNN_OP[op](input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))

    if op in ("max", "min"):
        # Selections are exact: inputs are bf16-representable in both dtypes.
        assert_equal(torch_output, output_tensor)
    elif op == "sum":
        assert_numeric_metrics(torch_output, output_tensor, **_SUM_TOL[dtype])
    elif torch_output.numel() <= 1:
        # Global mean of zero-centered data lands near 1e-4, where the relative
        # metrics (rtol, Frobenius) are undefined; PCC is auto-skipped for
        # scalars. Absolute error is the meaningful bound here.
        assert torch.allclose(
            torch_output.float(), output_tensor.reshape(torch_output.shape).float(), atol=0.004, rtol=0.0
        )
    else:
        tol = _MEAN_TOL_FP32_NC if (dtype == ttnn.float32 and _has_nc_dim(shape, dim)) else _MEAN_TOL[dtype]
        assert_numeric_metrics(torch_output, output_tensor, **tol)


@pytest.mark.parametrize(
    "cell",
    [c for c in CELLS if c[0] in ("w_unaligned_batch", "global_all")],
    ids=["w_unaligned_batch", "global_all"],
)
def test_reduce_family_bfp8(device, cell):
    """Block-float sum on the W and global branches (from test_sum_global's bfp8 axis)."""
    _, shape, dim, keepdim = cell
    torch.manual_seed(0)
    torch_input = torch_random(shape, -100, 100, dtype=torch.bfloat16)
    # Accumulate the golden in FP32 so it is stable across host thread counts.
    torch_output = (
        torch.sum(torch_input, dim=dim, keepdim=keepdim)
        if dim is not None
        else torch.sum(torch_input, dtype=torch.float32)
    )

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, POISON["sum"])
    output_tensor = ttnn.to_torch(ttnn.from_device(ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)))
    assert_numeric_metrics(torch_output, output_tensor, **_SUM_TOL[ttnn.bfloat8_b])


_INT32_CELLS = [
    ("w", (16, 41, 63), -1),
    ("h", (16, 37, 31), -2),
    ("global", (16, 41, 63), None),
]


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("cell", _INT32_CELLS, ids=[c[0] for c in _INT32_CELLS])
def test_reduce_family_int32(device, op, cell):
    """INT32 SFPU reduce paths, exact. min/max with negatives also exercise the
    INT32 pad sentinels (escape #19224: integer reduce cells were untested)."""
    _, shape, dim = cell
    torch.manual_seed(0)
    torch_input = torch.randint(-100, 100, shape, dtype=torch.int32)
    torch_output = (
        _torch_reduce(op, torch_input, dim, keepdim=True)
        if dim is not None
        else _torch_reduce(op, torch_input, None, False)
    )

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = TTNN_OP[op](input_tensor, dim=dim, keepdim=True) if dim is not None else TTNN_OP[op](input_tensor)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))
    assert_equal(torch_output, output_tensor)


@pytest.mark.parametrize(
    "op, scalar, dim",
    [
        ("mean", 2.0, (2, 3)),  # scalar post-scale on the AVG path (was test_mean_scaling_factor)
        ("sum", 0.5, -1),  # scalar on the SUM path
        ("max", -2.5, -1),  # negative scalar flips max -> min (front-end flip branch)
    ],
    ids=["mean_scale2", "sum_scale05", "max_negative_scale"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_reduce_family_scalar(device, op, scalar, dim, dtype):
    """scalar= on ones input: results are exactly representable, checked to 1 ULP."""
    torch.manual_seed(0)
    shape = (2, 4, 37, 63)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.ones(shape, dtype=torch_dtype)
    torch_output = _torch_reduce(op, scalar * torch_input, dim, keepdim=False)

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, POISON[op])
    output_tensor = TTNN_OP[op](input_tensor, dim=dim, keepdim=False, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output.shape)
    assert_with_ulp(torch_output, output_tensor, ulp_threshold=1)


# std/var: welford kernels. Cells cover the W/H/HW kernel variants, the
# permute-to-H path for a batch dim, the multi-dim fold, dims=[1,2,3]
# (the SDXL CFG-rescale configuration, previously absent from any grid),
# and the reduce-to-scalar path.
_STD_VAR_CELLS = [
    ("w_unaligned_batch", (16, 41, 63), -1, False),
    ("h_multicore", (1, 64, 64), -2, True),
    ("hw_pair", (16, 41, 63), (-2, -1), True),
    ("dim0_permute", (9, 2, 37, 63), 0, False),
    ("dims123_sdxl", (2, 4, 37, 63), (1, 2, 3), True),
    ("global_all", (16, 41, 63), None, False),
    ("empty_dim_list", (16, 41, 63), [], True),  # dim=[] edge (was in test_var's grid)
]


@pytest.mark.parametrize("op", ["std", "var"])
@pytest.mark.parametrize("cell", _STD_VAR_CELLS, ids=[c[0] for c in _STD_VAR_CELLS])
@pytest.mark.parametrize("correction", [True, False])
def test_std_var_family(device, op, cell, correction):
    _, shape, dim, keepdim = cell
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = _torch_reduce(op, torch_input, dim, keepdim, correction=correction)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, POISON[op])
    output_tensor = TTNN_OP[op](input_tensor, dim=dim, keepdim=keepdim, correction=correction)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))

    # randn outputs cluster near 1, so 1-ULP bf16 rounding dents PCC; the
    # rtol/atol/frobenius bounds carry the correctness check (as in the
    # replaced test_reduction.py::test_std).
    assert_numeric_metrics(
        torch_output, output_tensor, pcc_threshold=0.98, rtol=0.01, atol=0.01, frobenius_threshold=0.005
    )


# ---------------------------------------------------------------------------
# topk boundary cells (escape ledger)
# The multi-core factory needs W >= 8192, power-of-2 width, k <= 64; index
# dtype is UINT16 unless the input is FP32. The last two wrong-result escapes
# lived exactly on these boundaries: rows > 32 on the multi-core path (#53453)
# and 32-bit index dtype crossed with non-tile-multiple k (#53466).
# ---------------------------------------------------------------------------

_TOPK_CELLS = [
    # id                          shape               k    dtype           expected index dtype
    ("multicore_rows64_k50", (1, 2, 32, 8192), 50, ttnn.bfloat16, ttnn.uint16),
    ("multicore_rows32_k64_fp32", (1, 1, 32, 8192), 64, ttnn.float32, ttnn.uint32),
    ("singlecore_w8128_k50", (1, 1, 32, 8128), 50, ttnn.bfloat16, ttnn.uint16),
    ("singlecore_small_w", (1, 1, 64, 64), 32, ttnn.bfloat16, ttnn.uint16),
]


@pytest.mark.parametrize("cell", _TOPK_CELLS, ids=[c[0] for c in _TOPK_CELLS])
def test_topk_boundary_cells(device, cell):
    _, shape, k, dtype, index_dtype = cell
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)
    torch_values, _ = torch.topk(torch_input, k, dim=-1, largest=True, sorted=True)

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    values, indices = ttnn.topk(input_tensor, k, -1, True, True)

    assert values.shape[-1] == k
    assert indices.dtype == index_dtype

    tt_values = ttnn.to_torch(ttnn.from_device(values))
    tt_indices = ttnn.to_torch(ttnn.from_device(indices)).to(torch.int64)

    # Values are a selection: exact match against torch. Indices are checked by
    # gathering — ties make raw index comparison ill-defined, but every returned
    # index must point at an element equal to the returned value.
    assert_equal(torch_values, tt_values)
    gathered = torch.gather(torch_input, -1, tt_indices)
    assert_equal(gathered, tt_values)


@pytest.mark.xfail(
    reason="topk stable=True is documented best-effort until tt-llk#33492/#33473; an XPASS means the fix landed",
    strict=False,
)
def test_topk_stable_tie_order(device):
    """stable=True contract: tied values keep their original (ascending index) order."""
    shape, k = (1, 1, 32, 64), 32
    torch_input = torch.ones(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _, indices = ttnn.topk(input_tensor, k, -1, True, True, stable=True)
    tt_indices = ttnn.to_torch(ttnn.from_device(indices)).to(torch.int64)
    expected = torch.arange(k).expand(1, 1, 32, k)
    assert_equal(expected, tt_indices)


# ---------------------------------------------------------------------------
# Model-config pins: the configurations real in-tree models run every day.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_sub_core_grids", [False, True], ids=["full_grid", "sub_core_grids"])
def test_argmax_greedy_decode_vocab_scale(device, use_sub_core_grids):
    """Greedy LLM decode: ttnn.argmax over a vocab-scale row-major tensor with a
    preallocated output (models/common/modules/sampling/sampling_1d.py). Open
    issue #21120 reports wrong greedy-decode argmax results in this shape class;
    previously no test exceeded width 8192."""
    batch, vocab = 32, 128256
    torch.manual_seed(0)
    torch_input = torch_random((1, 1, batch, vocab), -100, 100, dtype=torch.bfloat16)
    # Plant a unique, exactly-representable peak per user so ties cannot occur.
    peaks = [(7919 * (r + 1)) % vocab for r in range(batch)]
    for r, p in enumerate(peaks):
        torch_input[0, 0, r, p] = 200.0 + r

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, batch, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    sub_core_grids = None
    if use_sub_core_grids:
        sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(6, 6)),
            ]
        )

    result = ttnn.argmax(
        input_tensor, dim=-1, keepdim=False, output_tensor=output_tensor, sub_core_grids=sub_core_grids
    )

    expected = torch.tensor(peaks, dtype=torch.int64).reshape(1, 1, batch)
    assert_equal(expected, ttnn.to_torch(ttnn.from_device(result)).to(torch.int64))
    # The preallocated buffer itself must hold the result (models reuse it across steps).
    assert_equal(expected, ttnn.to_torch(ttnn.from_device(output_tensor)).to(torch.int64))


def test_moe_combine_sum_dim0(device):
    """DeepSeek MoE expert combine: ttnn.sum(dim=0, keepdim=True) at the real
    [experts, 1, users, hidden] shape with an explicit memory_config
    (models/demos/deepseek_v3/tt/moe.py). Previously only toy dim-0 extents ran."""
    shape = (8, 1, 32, 7168)
    torch.manual_seed(0)
    torch_input = torch_random(shape, -1, 1, dtype=torch.bfloat16)
    torch_output = torch.sum(torch_input, dim=0, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.sum(input_tensor, dim=0, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))
    # 8-element bf16 sums of [-1, 1] data.
    assert_numeric_metrics(
        torch_output, output_tensor, pcc_threshold=0.999, rtol=0.05, atol=0.06, frobenius_threshold=0.01
    )


def test_sampling_preallocated_output(device):
    """The trace-capture sampling path: ttnn.sampling writes into a persistent
    preallocated output buffer (models/common/models/executor.py). Greedy
    per-user k=1/p=0 makes the expected pick deterministic."""
    users, w = 32, 64
    torch.manual_seed(0)
    values = torch.randn(1, 1, users, w, dtype=torch.bfloat16)
    # Make column 0 the strict per-row maximum.
    values[..., 0] = 200.0
    indices = torch.arange(w, dtype=torch.int32).expand(1, 1, users, w).contiguous()
    for r in range(users):
        indices[0, 0, r] += r * 1000

    values_tensor = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    indices_tensor = ttnn.from_torch(indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    k = ttnn.from_torch(
        torch.full((users,), 1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    p = ttnn.from_torch(
        torch.zeros(users, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    temp = ttnn.from_torch(
        torch.ones(users, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    prealloc = ttnn.from_torch(
        torch.zeros(1, 1, 1, users, dtype=torch.int32),
        dtype=ttnn.int32 if device.arch() == ttnn.device.Arch.QUASAR else ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    result = ttnn.sampling(values_tensor, indices_tensor, k, p, temp, seed=42, output_tensor=prealloc)

    expected = torch.tensor([r * 1000 for r in range(users)], dtype=torch.int64).reshape(1, 1, 1, users)
    assert_equal(expected, ttnn.to_torch(ttnn.from_device(result)).to(torch.int64))
    assert_equal(expected, ttnn.to_torch(ttnn.from_device(prealloc)).to(torch.int64))
