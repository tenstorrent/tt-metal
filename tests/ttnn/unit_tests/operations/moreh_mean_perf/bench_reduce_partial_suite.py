# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perf bench across every production op touched by the reduce-partial-scaler branch.

    base = merge-base(HEAD, main) = fda1e45f96f   (mask-tile path)
    head = 9751f4fd9f5                            (partial-scaler path)

Run under the device profiler and diff the two commits:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/moreh_mean_perf/bench_reduce_partial_suite.py

BENCH_CHECK=1 additionally verifies numerics (adds to_torch readbacks; leave off
when measuring). Correctness here is coarse on purpose -- these ops have their own
shipped test suites; this file exists to produce comparable device timings.

Naming: <family>_<variant>_<ragged|aligned|control>.

  ragged  -> reduce dim % 32 != 0, so the partial-scaler path is active. This is
             where a win is expected.
  aligned -> reduce dim % 32 == 0. NOT a null control: on base several of these
             kernels still split into two reduce calls plus an accumulator reload,
             while head emits one.
  control -> a sibling code path this branch does NOT touch, measured to detect
             build/clock skew between the two commits. Where a per-family control
             is free it is included:
               moreh_softmax          -> LARGE_C (softmax_c_large untouched)
               moreh_softmax_backward -> LARGE_H (only the _small factories changed)
               moreh_sum              -> reduce along W (only _h changed)
               normalization/softmax  -> general C large (untouched)
             layernorm and bias_backward have no cheap untouched sibling, so they
             rely on the session-level controls above -- build/clock skew is a
             global effect, so controls do not need to be per-op to catch it.

topk_router_gpt is NOT here: it needs a different device fixture (dispatch_core_axis)
and is a pure refactor onto the helper (single-tile block, no partial scaler at all),
so it is a perf-NEUTRALITY check rather than a win candidate. See its own bench file.
"""

import os

import pytest
import torch

import ttnn

TILE = 32
CHECK = os.environ.get("BENCH_CHECK", "0") == "1"

STRAT = ttnn.operations.moreh.SoftmaxOpParallelizationStrategy
# softmax_backward has its OWN strategy enum -- passing the forward one is a TypeError.
BSTRAT = ttnn.operations.moreh.SoftmaxBackwardOpParallelizationStrategy

MOREH_K = "ttnn/cpp/ttnn/operations/moreh"
LN_K = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute"


def ragged(nt):
    """Reduce-dim length with nt tiles and a 17-element partial last tile."""
    return (nt - 1) * TILE + 17


def aligned(nt):
    return nt * TILE


def mk(shape, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.float32),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan"),
    )


# ---------------------------------------------------------------- runners


def run_moreh_sum(device, h, w, dim):
    x = mk([1, 7, h, w], device)
    return ttnn.operations.moreh.sum(x, dim, keepdim=True)


def run_moreh_softmax(device, h, w, dim, strategy):
    x = mk([1, 7, h, w], device)
    return ttnn.operations.moreh.softmax(x, dim, strategy=strategy)


def run_moreh_softmax_backward(device, h, w, dim, strategy):
    y = mk([1, 7, h, w], device)
    dy = mk([1, 7, h, w], device)
    return ttnn.operations.moreh.softmax_backward(y, dy, dim, strategy=strategy)


def run_ttnn_softmax(device, shape, dim):
    x = mk(shape, device)
    return ttnn.softmax(x, dim)


def run_layernorm(device, h, w):
    x = mk([1, 1, h, w], device)
    # gamma/beta present: exercises the with-weights L1 budget, and the rm_gb reader.
    g = mk([1, 1, TILE, w], device)
    b = mk([1, 1, TILE, w], device)
    return ttnn.layer_norm(x, epsilon=1e-5, weight=g, bias=b)


def run_bias_backward(device, m):
    """Isolate the bias-grad reduce: only bias_grad is requested.

    bias_grad must be pre-allocated -- moreh_linear_backward.cpp TT_FATALs on
    bias_grad.has_value() rather than allocating it for you.
    """
    k, n = 1024, 2048
    inp = mk([2, 3, m, k], device)
    weight = mk([n, k], device)
    bias = mk([1, n], device)
    out_grad = mk([2, 3, m, n], device)
    bias_grad = mk([1, n], device)
    _, _, bg = ttnn.operations.moreh.linear_backward(
        out_grad,
        inp,
        weight,
        are_required_outputs=(False, False, True),
        bias=bias,
        bias_grad=bias_grad,
    )
    return bg


# ---------------------------------------------------------------- cases
# axis: which INPUT_0 dim carries the reduce ("Y" = H, "X" = W), and its expected
# LOGICAL length -- used by the extractor to verify the call-order mapping.
# axis=None means do not assert the shape (op has a non-obvious INPUT_0 mapping).

CASES = []


def case(cid, fn, op, kernel, axis, logical):
    CASES.append({"id": cid, "fn": fn, "op": op, "kernel": kernel, "axis": axis, "logical": logical})


# --- moreh_sum (H) : mirrors moreh_mean, 448 columns -------------------------
for tag, h in (("ragged", ragged(4)), ("aligned", aligned(4))):
    case(
        f"sum_h_{tag}",
        lambda d, h=h: run_moreh_sum(d, h, 2048, 2),
        "MorehSum",
        f"{MOREH_K}/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp",
        "Y",
        h,
    )
case(
    "sum_w_control",
    lambda d: run_moreh_sum(d, 2048, aligned(4), 3),
    "MorehSum",
    "moreh_sum_w.cpp",
    "X",
    aligned(4),
)

# --- moreh_softmax : all four changed kernels, forced strategy ---------------
for tag, n in (("ragged", ragged(4)), ("aligned", aligned(4))):
    case(
        f"msoftmax_small_h_{tag}",
        lambda d, n=n: run_moreh_softmax(d, n, 2048, 2, STRAT.SMALL_H),
        "MorehSoftmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_h.cpp",
        "Y",
        n,
    )
    case(
        f"msoftmax_small_w_{tag}",
        lambda d, n=n: run_moreh_softmax(d, 2048, n, 3, STRAT.SMALL_W),
        "MorehSoftmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_w.cpp",
        "X",
        n,
    )
for tag, n in (("ragged", ragged(32)), ("aligned", aligned(32))):
    case(
        f"msoftmax_large_h_{tag}",
        lambda d, n=n: run_moreh_softmax(d, n, 2048, 2, STRAT.LARGE_H),
        "MorehSoftmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp",
        "Y",
        n,
    )
    case(
        f"msoftmax_large_w_{tag}",
        lambda d, n=n: run_moreh_softmax(d, 2048, n, 3, STRAT.LARGE_W),
        "MorehSoftmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp",
        "X",
        n,
    )
case(
    "msoftmax_large_c_control",
    lambda d: run_moreh_softmax(d, aligned(4), 2048, 1, STRAT.LARGE_C),
    "MorehSoftmax",
    "moreh_softmax_c_large.cpp",
    "Y",
    aligned(4),
)

# --- moreh_softmax_backward : only the _small factories changed -------------
for tag, n in (("ragged", ragged(4)), ("aligned", aligned(4))):
    case(
        f"msoftmax_bw_small_h_{tag}",
        lambda d, n=n: run_moreh_softmax_backward(d, n, 2048, 2, BSTRAT.SMALL_H),
        "MorehSoftmaxBackward",
        "moreh_softmax_backward_h.cpp",
        "Y",
        n,
    )
    case(
        f"msoftmax_bw_small_w_{tag}",
        lambda d, n=n: run_moreh_softmax_backward(d, 2048, n, 3, BSTRAT.SMALL_W),
        "MorehSoftmaxBackward",
        "moreh_softmax_backward_w.cpp",
        "X",
        n,
    )
case(
    "msoftmax_bw_large_h_control",
    lambda d: run_moreh_softmax_backward(d, aligned(4), 2048, 2, BSTRAT.LARGE_H),
    "MorehSoftmaxBackward",
    "moreh_softmax_backward_h_large.cpp",
    "Y",
    aligned(4),
)

# --- normalization/softmax (ttnn.softmax) : the general factories -----------
# H: rank 4, dim=2. W: MUST be rank 3 -- rank 4 + dim==rank-1 routes to the
# attention-optimized factory instead of general_w.
for tag, n in (("ragged", ragged(4)), ("aligned", aligned(4))):
    case(
        f"nsoftmax_small_h_{tag}",
        lambda d, n=n: run_ttnn_softmax(d, [1, 7, n, 2048], 2),
        "Softmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_h.cpp",
        "Y",
        n,
    )
    case(
        f"nsoftmax_small_w_{tag}",
        lambda d, n=n: run_ttnn_softmax(d, [2, 2, 4, 256, n], 4),
        "Softmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_w.cpp",
        "X",
        n,
    )
for tag, n in (("ragged", ragged(64)), ("aligned", aligned(64))):
    case(
        f"nsoftmax_large_h_{tag}",
        lambda d, n=n: run_ttnn_softmax(d, [1, 7, n, 512], 2),
        "Softmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_h_large.cpp",
        "Y",
        n,
    )
    case(
        f"nsoftmax_large_w_{tag}",
        lambda d, n=n: run_ttnn_softmax(d, [2, 2, 4, 256, n], 4),
        "Softmax",
        f"{MOREH_K}/moreh_softmax/device/kernels/moreh_softmax_w_large.cpp",
        "X",
        n,
    )
case(
    "nsoftmax_large_c_control",
    lambda d: run_ttnn_softmax(d, [1, 7, aligned(4), 2048], 1),
    "Softmax",
    "moreh_softmax_c_large.cpp",
    "Y",
    aligned(4),
)

# --- layernorm : reduces along W, so W is the ragged axis -------------------
# Wt=32 fits L1 -> layernorm.cpp; Wt=256 overflows -> layernorm_large_tensor.cpp
for tag, n in (("ragged", ragged(32)), ("aligned", aligned(32))):
    case(
        f"layernorm_{tag}",
        lambda d, n=n: run_layernorm(d, 1024, n),
        "LayerNorm",
        f"{LN_K}/layernorm.cpp",
        "X",
        n,
    )
for tag, n in (("ragged", ragged(256)), ("aligned", aligned(256))):
    case(
        f"layernorm_large_{tag}",
        lambda d, n=n: run_layernorm(d, 256, n),
        "LayerNorm",
        f"{LN_K}/layernorm_large_tensor.cpp",
        "X",
        n,
    )

# --- moreh bias_backward : bias grad reduces output_grad over H ------------
# INPUT_0 mapping for linear_backward is not obviously output_grad, so the shape
# assertion is skipped here; the op code + kernel file still pin the variant.
for tag, m in (("ragged", ragged(4)), ("aligned", aligned(4))):
    case(
        f"bias_bw_{tag}",
        lambda d, m=m: run_bias_backward(d, m),
        "MorehBiasAddBackward",
        "moreh_bias_backward_multi_core_h.cpp",
        None,
        m,
    )


@pytest.mark.parametrize("c", CASES, ids=[c["id"] for c in CASES])
def test_reduce_partial_bench(c, device):
    torch.manual_seed(2024)
    out = c["fn"](device)
    if CHECK:
        t = ttnn.to_torch(out)
        assert torch.isfinite(t).all(), f"{c['id']}: non-finite output"
