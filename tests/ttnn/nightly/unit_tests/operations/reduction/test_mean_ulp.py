# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.mean (issue #33740).

Relationship to existing mean tests (do not duplicate; different goal):

- ``test_reduction.py`` (e.g. ``test_mean_4d_tensor_dims``, ``test_mean_3d_tensor_dims``,
  ``test_mean_2d_tensor_dims``): equivalence vs torch via PCC / rtol / atol on modest
  shapes and many ``dim`` / ``keepdim`` combinations.
- ``test_reduction_mean.py``: batched 3D means (``dim`` -1 / -2) with tile padding
  stress, ``scalar=`` scaling, and sharded DRAM/block configs—still PCC-style metrics.
- ``tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py``:
  reduction corner cases (empty tensors, preallocated outputs, etc.) with ``op`` including
  ``mean``—not large numeric sweeps.

This file is only for **bounded max-ULP** characterization (plus near-zero absolute
tolerance) and **wide shape sweeps** to track regression in reduction length; it does
not replace functional, sharding, or corner-case coverage above.

BF16 golden: torch.mean(bf16_input.float(), dim=...).to(torch.bfloat16)
  - Accumulates in FP32 on the host, then casts the result to BF16.
  - This matches the "best practice" (FP32 accumulation) that the device
    path should also follow.

FP32 golden: torch.mean(fp32_input, dim=...)
  - PyTorch uses true IEEE 754 float32 accumulation.
  - Unless the API uses SFPU-based true float32 accumulation kernels, the
    tile engine accumulates in TF32, so accuracy may be lower / ULP higher.

ULP is measured in the output dtype (BF16 or FP32).  Elements where
|golden| is very small relative to the tensor's dynamic range are excluded
from ULP (where the metric breaks down due to division by a tiny ULP
quantum) and validated with a scaled absolute-tolerance check instead.

Metrics are logged with loguru at INFO for every parametrized case (pass or fail),
consistent with other tests under ``tests/ttnn`` (default sink: stderr).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mean_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """Compute kernel config from device arch."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _golden_mean_bf16(input_bf16: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """FP32-accumulated mean, result cast back to BF16."""
    return torch.mean(input_bf16.float(), dim=dim, keepdim=keepdim).to(torch.bfloat16)


def _golden_mean_fp32(input_fp32: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch IEEE 754 float32 mean; tile engine accumulates in TF32 unless SFPU kernels are used."""
    return torch.mean(input_fp32, dim=dim, keepdim=keepdim)


def _run_ttnn_mean(
    input_torch: torch.Tensor,
    ttnn_dtype,
    device,
    dim,
    keepdim: bool,
    compute_kernel_config=None,
) -> torch.Tensor:
    """Send tensor to device, run ttnn.mean, return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn_mean(tt_input, dim=dim, keepdim=keepdim, compute_kernel_config=compute_kernel_config)
    return ttnn.to_torch(tt_out)


# ---------------------------------------------------------------------------
# Test parameters — 2×2×3×3 grid: small/large for N,C; small/medium/large for H,W
# ---------------------------------------------------------------------------


def _build_mean_shapes_and_dims():
    """Build (shape, dim, id) cases: N×C grid for W and H reduction, plus HW and odd."""
    out = []

    # 2×2×3×3 grid: small/large for N,C; small/medium/large for H,W
    _N_SIZES = [1, 8]  # small, large batch
    _C_SIZES = [1, 4]  # small, large channel
    _H_SIZES = [32, 128, 512]  # small, medium, large H
    _W_SIZES = [64, 512, 2048]  # small, medium, large W
    _H_FIXED = 32  # non-reduction H for W-reduction shapes
    _W_FIXED = 64  # non-reduction W for H-reduction shapes

    # W-reduction: vary N, C, W; fix H
    for n in _N_SIZES:
        for c in _C_SIZES:
            for w in _W_SIZES:
                out.append(((n, c, _H_FIXED, w), -1, f"W-{n}x{c}x{_H_FIXED}x{w}"))

    # H-reduction: vary N, C, H; fix W
    for n in _N_SIZES:
        for c in _C_SIZES:
            for h in _H_SIZES:
                out.append(((n, c, h, _W_FIXED), -2, f"H-{n}x{c}x{h}x{_W_FIXED}"))

    # 2D HW-reduction: one small and one large representative shape
    out.append(((1, 1, 64, 128), [-2, -1], "HW-small"))
    out.append(((4, 4, 256, 512), [-2, -1], "HW-large"))

    # One non-tile-aligned shape to catch padding edge cases
    out.append(((1, 1, 37, 41), -1, "W-odd"))

    return out


_SHAPES_AND_DIMS = _build_mean_shapes_and_dims()


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs FP32-accumulated torch golden (see measure_ulp_with_near_zero_atol).
# fp32_dest_acc_en=True (BF16 inputs → FP32 accumulation → BF16 out): peak ~8 ULP on this grid.
# fp32_dest_acc_en=False (BF16 accumulation throughout): much wider; long reductions over
# near-zero-mean tensors see ~77x more ULP error than the FP32-acc path.
# This demonstrates that fp32 accumulation is essential for BF16 mean accuracy.
_BF16_ULP_THRESHOLD_FP32_DEST = 30
_BF16_ULP_THRESHOLD_BF16_DEST = 2500  # ~4x headroom over observed peak 618 ULP
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002  # tight; fp32 rounding error is tiny
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.40  # loose; BF16 accum can be ~35% of range on near-zero mean


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_mean_ulp_bf16(device, shape, dim, desc, distribution, keepdim, fp32_dest_acc_en):
    """Characterize BF16 mean ULP vs FP32-accumulated Torch golden.

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.
    """
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=keepdim)
    ckc = _make_mean_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=keepdim, compute_kernel_config=ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.mean ULP (BF16) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[BF16 {desc} {distribution} keepdim={keepdim} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# ttnn.mean defaults to the accurate SFPU path, so this measures SFPU vs torch fp32 golden; the two fp32 orderings differ by at most a few hundred ULP here
_FP32_ULP_THRESHOLD = 512
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.00125


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
def test_mean_ulp_fp32(device, shape, dim, desc, distribution, keepdim):
    """Characterize FP32 mean ULP vs Torch FP32 golden; fp32_dest_acc_en=True only."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=keepdim)
    ckc = _make_mean_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=keepdim, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} {distribution} shape={shape} dim={dim} keepdim={keepdim}"
    logger.info(
        f"ttnn.mean ULP (FP32, fp32_dest_acc_en=True) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {desc} {distribution} keepdim={keepdim}] {msg}"


# ---------------------------------------------------------------------------
# FP32 accurate (SFPU) vs fast (FPU) mode
# ---------------------------------------------------------------------------

# fast_and_approximate_mode: False (default) = accurate SFPU (full fp32); True = fast FPU (tf32). Accuracy = max fp32 ULP vs an fp64 golden.
_SFPU_ULP_MAX = 8  # accurate-path cap; observed peak is 2 ULP on this grid (4x headroom)


def _fp32_input(distribution: str, shape, seed: int) -> torch.Tensor:
    """FP32 test input. Distributions stress different accumulation regimes."""
    torch.manual_seed(seed)
    if distribution == "unit":
        return torch.empty(shape, dtype=torch.float32).uniform_(1.0, 2.0)
    if distribution == "large_offset":  # large shared exponent → catastrophic tf32 cancellation
        return 1.0e4 + torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0)
    if distribution == "wide_range":  # many exponents in one reduction
        return (
            torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0)
            * torch.pow(10.0, torch.empty(shape, dtype=torch.float32).uniform_(0.0, 5.0))
            + 5.0e3
        )
    if distribution == "uniform_01":  # uniform [0, 1)
        return torch.rand(shape, dtype=torch.float32)
    raise ValueError(f"unknown distribution {distribution}")


def _fp32_ulp_max(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Max signed-magnitude fp32 ULP distance between two same-shape fp32 tensors."""
    a = actual.reshape(reference.shape).to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
    r = reference.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
    same = (a & 0x80000000) == (r & 0x80000000)
    av, rv = a & 0x7FFFFFFF, r & 0x7FFFFFFF
    ulp = torch.where(same, (av - rv).abs(), av.abs() + rv.abs())
    return ulp.max().item()


def _mean_pair(device, x, dim, keepdim, fp32_dest_acc_en=True, input_memory_config=None, output_memory_config=None):
    """Run ttnn.mean twice (fast FPU, accurate SFPU) on the same input; return (fpu, sfpu) as torch."""
    ckc = _make_mean_compute_kernel_config(device, fp32_dest_acc_en)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    kw = dict(dim=dim, keepdim=keepdim, compute_kernel_config=ckc, memory_config=output_memory_config)
    fpu = ttnn.to_torch(ttnn.mean(tt_in, fast_and_approximate_mode=True, **kw))
    sfpu = ttnn.to_torch(ttnn.mean(tt_in, fast_and_approximate_mode=False, **kw))
    return fpu, sfpu


_ACCURATE_SHAPES = [
    ((1, 22, 1536), -1, "W1536"),  # long W reduce
    ((1, 1, 32, 2048), -1, "W2048"),
    ((2, 4, 512, 64), -2, "H512"),  # H reduce
    ((2, 3, 256, 256), [-2, -1], "HW256"),  # HW reduce
    ((1, 1, 30, 1000), -1, "pad30x1000"),  # non-tile-aligned H and W (padding)
]


# Broad sweep — this test carries the distribution/keepdim/seed variation
@pytest.mark.parametrize("distribution", ["unit", "large_offset", "wide_range"])
@pytest.mark.parametrize("shape, dim, desc", _ACCURATE_SHAPES, ids=[c[2] for c in _ACCURATE_SHAPES])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
@pytest.mark.parametrize("seed", [0, 1234])
def test_mean_fp32_accurate_vs_fast(device, distribution, shape, dim, desc, keepdim, seed):
    """Accurate SFPU fp32 mean must be within the tight ULP cap of an fp64 golden; the FPU gap is logged."""
    x = _fp32_input(distribution, shape, seed)
    ref = torch.mean(x.to(torch.float64), dim=dim, keepdim=keepdim).to(torch.float32)
    fpu, sfpu = _mean_pair(device, x, dim, keepdim)
    fpu_ulp = _fp32_ulp_max(fpu, ref)
    sfpu_ulp = _fp32_ulp_max(sfpu, ref)
    logger.info(
        f"ttnn.mean fp32 accurate-vs-fast | {desc} {distribution} seed={seed} keepdim={keepdim} | "
        f"FPU ulp={fpu_ulp:.0f} | SFPU ulp={sfpu_ulp:.0f}/{_SFPU_ULP_MAX}"
    )
    assert sfpu_ulp <= _SFPU_ULP_MAX, f"SFPU ulp {sfpu_ulp:.0f} exceeds tight cap {_SFPU_ULP_MAX} (FPU={fpu_ulp:.0f})"


def _mean_input_memory_config(mem_layout: str, shape):
    """Input memory config for a 4D (1,1,H,W) shape whose H and W are divisible by 8 tiles."""
    _, _, h, w = shape
    if mem_layout == "dram_interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    if mem_layout == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    if mem_layout == "l1_height_sharded":
        return ttnn.create_sharded_memory_config(
            shape=(h // 8, w),
            core_grid=ttnn.CoreGrid(x=1, y=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    if mem_layout == "l1_width_sharded":
        return ttnn.create_sharded_memory_config(
            shape=(h, w // 8),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
    raise ValueError(f"unknown mem_layout {mem_layout}")


@pytest.mark.parametrize("mem_layout", ["dram_interleaved", "l1_interleaved", "l1_width_sharded", "l1_height_sharded"])
@pytest.mark.parametrize("distribution", ["unit", "wide_range"])
def test_mean_fp32_accurate_memory_layouts(device, mem_layout, distribution):
    """Accurate SFPU path must hold the tight ULP cap across DRAM/L1 interleaved and L1 width/height sharded inputs."""
    shape, dim = (1, 1, 256, 1024), -1
    x = _fp32_input(distribution, shape, seed=0)
    ref = torch.mean(x.to(torch.float64), dim=dim, keepdim=True).to(torch.float32)
    _, sfpu = _mean_pair(
        device,
        x,
        dim,
        keepdim=True,
        input_memory_config=_mean_input_memory_config(mem_layout, shape),
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sfpu_ulp = _fp32_ulp_max(sfpu, ref)
    logger.info(
        f"ttnn.mean fp32 accurate | mem_layout={mem_layout} {distribution} | SFPU ulp={sfpu_ulp:.0f}/{_SFPU_ULP_MAX}"
    )
    assert sfpu_ulp <= _SFPU_ULP_MAX, f"SFPU ulp {sfpu_ulp:.0f} exceeds tight cap {_SFPU_ULP_MAX} for {mem_layout}"


@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim_false", "keepdim_true"])
def test_mean_fp32_long_reduction_regression(device, keepdim):
    """Regression: fp32 mean over a long W (uniform [0,1)) is thousands of ULP off on the FPU/tf32 path; accurate SFPU must be within a few ULP."""
    shape, dim = (1, 22, 1536), -1
    x = _fp32_input("uniform_01", shape, seed=404)
    ref = torch.mean(x.to(torch.float64), dim=dim, keepdim=keepdim).to(torch.float32)
    fpu, sfpu = _mean_pair(device, x, dim, keepdim)
    fpu_ulp = _fp32_ulp_max(fpu, ref)
    sfpu_ulp = _fp32_ulp_max(sfpu, ref)
    logger.info(
        f"fp32 mean long-W uniform keepdim={keepdim} | FPU ulp={fpu_ulp:.0f} | SFPU ulp={sfpu_ulp:.0f}/{_SFPU_ULP_MAX}"
    )
    assert sfpu_ulp <= _SFPU_ULP_MAX, f"SFPU ulp {sfpu_ulp:.0f} exceeds tight cap {_SFPU_ULP_MAX}"


@pytest.mark.parametrize(
    "shape, dim, desc",
    [((1, 1, 4096, 8192), -1, "W8192_deep"), ((1, 1, 8192, 64), -2, "H8192_deep")],
    ids=["W8192_deep", "H8192_deep"],
)
def test_mean_fp32_accurate_large_depth(device, shape, dim, desc):
    """High accumulation depth (thousands of elements) is where the fast/tf32 path loses the most precision; accurate SFPU must still hold the tight cap."""
    x = _fp32_input("large_offset", shape, seed=0)
    ref = torch.mean(x.to(torch.float64), dim=dim, keepdim=True).to(torch.float32)
    fpu, sfpu = _mean_pair(device, x, dim, keepdim=True)
    fpu_ulp = _fp32_ulp_max(fpu, ref)
    sfpu_ulp = _fp32_ulp_max(sfpu, ref)
    logger.info(
        f"ttnn.mean fp32 accurate large-depth | {desc} | FPU ulp={fpu_ulp:.0f} | SFPU ulp={sfpu_ulp:.0f}/{_SFPU_ULP_MAX}"
    )
    assert sfpu_ulp <= _SFPU_ULP_MAX, f"SFPU ulp {sfpu_ulp:.0f} exceeds tight cap {_SFPU_ULP_MAX}"


@pytest.mark.parametrize("shape, dim", [((1, 1, 32, 2048), -1)])
def test_mean_fp32_accurate_falls_back_without_fp32_dest_acc(device, shape, dim):
    """Without fp32_dest_acc_en, accurate mode must fall back to the FPU (SFPU can't preserve fp32), giving results bit-identical to fast mode."""
    torch.manual_seed(0)
    x = 1.0e4 + torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0)
    fpu, sfpu = _mean_pair(device, x, dim, keepdim=True, fp32_dest_acc_en=False)
    assert torch.equal(fpu, sfpu), "accurate path did not fall back to FPU when fp32_dest_acc_en=False"
