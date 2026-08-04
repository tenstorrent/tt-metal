# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.layer_norm (issue #33749).

BF16 golden: torch.nn.functional.layer_norm on BF16 input (same dtype as device).
  PyTorch promotes BF16 internally for the reduction, so this already reflects
  FP32-accumulated arithmetic—matching the best-practice reference.

FP32 golden: torch.nn.functional.layer_norm on FP32 input.
  Unless the API uses SFPU-based true float32 accumulation kernels, the
  tile engine accumulates in TF32, so accuracy may be lower / ULP higher.

BF16 tests parametrize fp32_dest_acc_en=[False, True] to document the accuracy
cost of BF16-only accumulation and verify the recommended FP32-acc path.
Separate ULP and ATOL thresholds apply to each mode (tighter for fp32_acc=True).

FP32 tests use fp32_dest_acc_en=True only (device enforces this for FP32 inputs).

ULP is measured in the output dtype.  Near-zero golden elements use a scaled
absolute tolerance (same pattern as test_mean_ulp).

Metrics logged with loguru at INFO per parametrized case.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import math

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import ttnn_layer_norm
from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
    ttnn_layer_norm_sharded,
    ttnn_rms_norm_sharded,
    make_sharded_norm_mem_config,
    to_poisoned_sharded,
)

# Poison value to ensure Welford's algorithm ignores padded elements (#31982)
PAD_VALUE = -42


def create_recip_tensor(device, w, use_welford):
    """Helper to create reciprocal tensor for non-sharded welford tests."""
    if not use_welford:
        return None
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(device, core_range_set, w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ln_compute_kernel_config(device, fp32_dest_acc_en: bool):
    """Compute kernel config from device arch."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _make_ln_input(h: int, w: int, dtype: torch.dtype, distribution: str) -> torch.Tensor:
    """Generate a (h, w) layer-norm input for the requested distribution."""
    if distribution == "normal":
        return torch.randn((h, w), dtype=torch.float32).to(dtype)
    if distribution == "wide_uniform":
        return torch.empty((h, w), dtype=torch.float32).uniform_(-1e3, 1e3).to(dtype)
    if distribution == "centered_uniform":
        return torch.empty((h, w), dtype=torch.float32).uniform_(-0.5, 0.5).to(dtype)
    # uniform_01 (default, matches original torch.rand behaviour)
    return torch.rand((h, w), dtype=torch.float32).to(dtype)


def _run_ttnn_layer_norm(
    torch_input_tensor: torch.Tensor,
    device,
    use_welford: bool,
    torch_weight: torch.Tensor | None = None,
    torch_bias: torch.Tensor | None = None,
    compute_kernel_config=None,
) -> torch.Tensor:
    """Same path as test_layer_norm: TILE, fill_implicit_tile_padding, optional weight/bias.

    Supports all four wb combinations: no_wb, weight_only, bias_only, and wb.
    Each non-None tensor is converted individually; None parameters are omitted
    from the ttnn.layer_norm call (not passed as explicit None) so the C++ binding
    uses its own default, matching the no-wb code path for missing parameters.
    """
    h, w = torch_input_tensor.shape[-2], torch_input_tensor.shape[-1]
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    ln_kwargs: dict = dict(program_config=program_config, recip_tensor=recip_tensor)
    if compute_kernel_config is not None:
        ln_kwargs["compute_kernel_config"] = compute_kernel_config
    if torch_weight is not None:
        ln_kwargs["weight"] = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    if torch_bias is not None:
        ln_kwargs["bias"] = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_layer_norm(input_tensor, **ln_kwargs)
    output_tensor = ttnn.from_device(output_tensor)
    return ttnn.to_torch(output_tensor)


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 max-ULP cap vs torch golden.
# fp32_dest_acc_en=True (BF16 inputs → FP32 accumulation → BF16 out): peak ~25 ULP (W up to 4096).
# fp32_dest_acc_en=False (BF16 accumulation throughout): much wider tail; near-zero normalized
# outputs see substantially more absolute error due to BF16 variance/mean rounding.
# This demonstrates that fp32 accumulation is essential for BF16 layer_norm accuracy.
_BF16_ULP_THRESHOLD_FP32_DEST = 32
_BF16_ULP_THRESHOLD_BF16_DEST = 20000  # W=14336 fp32_acc=False peaks at ~6679 ULP; 3x headroom
_BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST = 0.002  # tight; fp32 rounding error is tiny
_BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST = 0.06  # loose; BF16 accum can be ~2% of range on near-zero output


def _build_layer_norm_shapes():
    seen = set()
    rows = []

    # small / medium / large W (normalization dim), H (batch rows), and squares
    _LN_W_SIZES = [64, 512, 4096]
    _LN_H_SIZES = [32, 256, 2048]
    _LN_SQUARES = [64, 256]
    _LN_H_FIXED = 32
    _LN_W_FIXED = 64

    def add(h, w, tag):
        key = (h, w)
        if key in seen:
            return
        seen.add(key)
        rows.append((h, w, f"{h}x{w}-{tag}"))

    for w in _LN_W_SIZES:
        add(_LN_H_FIXED, w, "Wsweep")
    for h in _LN_H_SIZES:
        add(h, _LN_W_FIXED, "Hsweep")
    for s in _LN_SQUARES:
        add(s, s, "square")
    # Non-tile-aligned shape
    add(37, 41, "odd")
    # Single-row vector (stress for very short H)
    add(1, 512, "vector")
    return rows


_SHAPES = _build_layer_norm_shapes()


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_layer_norm_ulp_bf16_no_weight_bias(device, h, w, desc, use_welford, distribution, fp32_dest_acc_en):
    """BF16 layer_norm ULP vs torch.nn.functional.layer_norm (no weight/bias).

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.
    """
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.bfloat16, distribution)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford, compute_kernel_config=ckc)

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}"
    logger.info(
        f"ttnn.layer_norm ULP (BF16, no wb) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert (
        passed
    ), f"[BF16 no_wb {desc} use_welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}] {msg}"


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["uniform_01", "normal", "wide_uniform"])
@pytest.mark.parametrize("wb_mode", ["wb", "weight_only", "bias_only"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32_acc_off", "fp32_acc_on"])
def test_layer_norm_ulp_bf16_with_weight_bias(device, h, w, desc, use_welford, distribution, wb_mode, fp32_dest_acc_en):
    """BF16 layer_norm ULP with weight/bias variants vs torch reference.

    fp32_dest_acc_en=True reflects the recommended path (BF16 inputs accumulated in FP32).
    fp32_dest_acc_en=False documents the accuracy cost of BF16-only accumulation.

    wb_mode controls which affine parameters are active:
      "wb"          – both weight and bias (original coverage)
      "weight_only" – weight only, bias=None
      "bias_only"   – bias only, weight=None
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    torch_input_tensor = _make_ln_input(h, w, dtype, distribution)
    torch_weight = torch.rand((w,), dtype=dtype) if wb_mode in ("wb", "weight_only") else None
    torch_bias = torch.rand((w,), dtype=dtype) if wb_mode in ("wb", "bias_only") else None
    golden = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en)
    actual = _run_ttnn_layer_norm(
        torch_input_tensor,
        device,
        use_welford,
        torch_weight=torch_weight,
        torch_bias=torch_bias,
        compute_kernel_config=ckc,
    )

    ulp_threshold = _BF16_ULP_THRESHOLD_FP32_DEST if fp32_dest_acc_en else _BF16_ULP_THRESHOLD_BF16_DEST
    atol_fraction = (
        _BF16_NEAR_ZERO_ATOL_FRACTION_FP32_DEST if fp32_dest_acc_en else _BF16_NEAR_ZERO_ATOL_FRACTION_BF16_DEST
    )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    spec = (
        f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} wb={wb_mode} fp32_acc={fp32_dest_acc_en}"
    )
    logger.info(
        f"ttnn.layer_norm ULP (BF16, {wb_mode}) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert (
        passed
    ), f"[BF16 {wb_mode} {desc} use_welford={use_welford} dist={distribution} fp32_acc={fp32_dest_acc_en}] {msg}"


# ---------------------------------------------------------------------------
# FP32 thresholds
# ---------------------------------------------------------------------------
_FP32_ULP_THRESHOLD = 2_500_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.004


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform", "centered_uniform"])
def test_layer_norm_ulp_fp32_no_weight_bias(device, h, w, desc, use_welford, distribution):
    """FP32 layer_norm ULP vs torch float32 golden (no weight/bias); fp32_dest_acc_en=True only."""
    if is_blackhole() and use_welford and w == 4096:
        # The large-tensor Welford layer_norm kernels produce nondeterministic FP32 output on
        # Blackhole (back-to-back runs differ; failure disappears under watcher), so the
        # determinism check in ttnn_layer_norm trips. Tracked in issue #46523.
        pytest.skip("Blackhole large-tensor Welford layer_norm is nondeterministic (FP32); see #46523")
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.float32, distribution)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_layer_norm(torch_input_tensor, device, use_welford, compute_kernel_config=ckc)

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution}"
    logger.info(
        f"ttnn.layer_norm ULP (FP32, no wb) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 no_wb {desc} use_welford={use_welford} dist={distribution}] {msg}"


@pytest.mark.parametrize("h, w, desc", _SHAPES, ids=[c[2] for c in _SHAPES])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform", "centered_uniform"])
@pytest.mark.parametrize("wb_mode", ["wb", "weight_only", "bias_only"])
def test_layer_norm_ulp_fp32_with_weight_bias(device, h, w, desc, use_welford, distribution, wb_mode):
    """FP32 layer_norm ULP with weight/bias variants vs torch float32 golden; fp32_dest_acc_en=True only.

    wb_mode controls which affine parameters are active:
      "wb"          – both weight and bias
      "weight_only" – weight only, bias=None
      "bias_only"   – bias only, weight=None
    """
    if is_blackhole() and use_welford and w == 4096:
        # The large-tensor Welford layer_norm kernels produce nondeterministic FP32 output on
        # Blackhole (back-to-back runs differ; failure disappears under watcher), so the
        # determinism check in ttnn_layer_norm trips. Tracked in issue #46523.
        pytest.skip("Blackhole large-tensor Welford layer_norm is nondeterministic (FP32); see #46523")
    torch.manual_seed(0)
    torch_input_tensor = _make_ln_input(h, w, torch.float32, distribution)
    torch_weight = torch.rand((w,), dtype=torch.float32) if wb_mode in ("wb", "weight_only") else None
    torch_bias = torch.rand((w,), dtype=torch.float32) if wb_mode in ("wb", "bias_only") else None
    golden = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )
    ckc = _make_ln_compute_kernel_config(device, fp32_dest_acc_en=True)
    actual = _run_ttnn_layer_norm(
        torch_input_tensor,
        device,
        use_welford,
        torch_weight=torch_weight,
        torch_bias=torch_bias,
        compute_kernel_config=ckc,
    )

    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    spec = f"{desc} shape_hw=({h},{w}) welford={use_welford} dist={distribution} wb={wb_mode}"
    logger.info(
        f"ttnn.layer_norm ULP (FP32, {wb_mode}) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{_FP32_ULP_THRESHOLD} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[FP32 {wb_mode} {desc} use_welford={use_welford} dist={distribution}] {msg}"


# BF16 caps for the sharded non-tile-aligned tests. These are far tighter than the generic
# BF16-accumulation cap because normalizing over the padded width leaks only a moderate error
# (a few percent), which the generic cap would absorb. Correctly masked output is within a few
# ULP and a tiny absolute error of the golden, so a tight cap turns padding contamination into a
# test failure rather than a silent pass.
_SHARDED_NONALIGNED_BF16_ULP_THRESHOLD = 64
_SHARDED_NONALIGNED_BF16_NEAR_ZERO_ATOL_FRACTION = 0.004


def _assert_sharded_norm_ulp(golden, actual, dtype, log_prefix: str, spec: str, fail_prefix: str):
    """Pick dtype-appropriate ULP/atol caps, measure ULP vs golden, log the per-case metrics, and assert.

    The caps are intentionally tight so padding contamination is caught rather than absorbed. ULP is
    measured in the output dtype, so FP32 inputs use the FP32 cap. That cap is loose in FP32-ULP terms
    because the FPU mask multiply that zeroes the padding columns also rounds the final tile's valid
    columns to TF32; the resulting TF32-precision error lands in the FP32 output and is large when
    counted in FP32 ULP, so the cap is sized to absorb it. BF16 uses the dedicated tight
    sharded-non-aligned cap rather than the loose BF16-accumulation cap.

    The per-case metrics are logged at INFO regardless of pass/fail (accuracy characterization); the
    failure detail is logged only on failure.
    """
    if dtype == torch.float32:
        ulp_threshold, atol_fraction = _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    else:
        ulp_threshold, atol_fraction = (
            _SHARDED_NONALIGNED_BF16_ULP_THRESHOLD,
            _SHARDED_NONALIGNED_BF16_NEAR_ZERO_ATOL_FRACTION,
        )
    passed, max_ulp, max_atol_err, atol_tol, msg, ulp_stats = measure_ulp_with_near_zero_atol(
        golden, actual, ulp_threshold, atol_fraction
    )
    logger.info(
        f"{log_prefix} | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"{fail_prefix} {msg}"


def _run_sharded_norm_ulp(device, norm, w, num_cores_w, distribution, dtype, use_welford=False, residual=False):
    """Run a sharded layer_norm / rms_norm over width w split across num_cores_w cores and compare ULP
    vs the torch golden. The shard width is the per-core logical share rounded up to a whole tile, so a
    non-tile-aligned width (or one that does not divide evenly across cores) leaves padding on the final
    shard. That implicit tile padding is poisoned, so any path that folds the padded columns into its
    statistics is observably wrong. residual=True fuses a second poisoned input (the norm is applied to
    a + b). use_welford applies to layer_norm only.
    """
    torch.manual_seed(0)
    h = 32
    eps = 1e-12
    shard_w = math.ceil(w / num_cores_w / 32) * 32
    block_wt = shard_w // 32

    torch_a = _make_ln_input(h, w, dtype, distribution)
    torch_b = _make_ln_input(h, w, dtype, distribution) if residual else None
    src = torch_a + torch_b if residual else torch_a
    if norm == "layernorm":
        golden = torch.nn.functional.layer_norm(src, normalized_shape=[w])
    else:
        ms = src.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        golden = (src.to(torch.float32) * torch.rsqrt(ms + eps)).to(dtype)

    sharded_mem_config = make_sharded_norm_mem_config(num_cores_w=num_cores_w, h=h, shard_w=shard_w)
    tt_a = to_poisoned_sharded(device, torch_a, sharded_mem_config, PAD_VALUE)
    tt_b = to_poisoned_sharded(device, torch_b, sharded_mem_config, PAD_VALUE) if residual else None

    if norm == "layernorm":
        actual = ttnn_layer_norm_sharded(
            device, tt_a, use_welford=use_welford, block_ht=h // 32, block_wt=block_wt, subblock_w=1, residual=tt_b
        )
    else:
        actual = ttnn_rms_norm_sharded(device, tt_a, block_ht=h // 32, block_wt=block_wt, subblock_w=1, residual=tt_b)
    # Discard padding before comparison.
    actual = actual[..., :w]

    spec = (
        f"sharded {norm} shape_hw=({h},{w}) cores={num_cores_w} shard_w={shard_w} "
        f"welford={use_welford} residual={residual} dist={distribution} dtype={dtype}"
    )
    _assert_sharded_norm_ulp(golden, actual, dtype, f"ttnn.{norm} ULP (sharded)", spec, f"[{spec}]")


# Widths that are not multiples of the tile width (32), each on a single core so the
# whole logical row plus its tile padding lives in one shard.
@pytest.mark.parametrize("w", [40, 72, 200])
# Restricted to small-range distributions. The near-zero atol tolerance scales with the
# golden's value range, so a wide-range distribution (e.g. wide_uniform, ±1e3) inflates the
# tolerance enough to absorb the error from normalizing over padded columns. normal and
# centered_uniform keep the range near unity, so PAD_VALUE-contaminated statistics are caught.
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
# Both reduction paths must normalize over the logical width: the Welford path (reciprocal LUT)
# and the legacy reduce path (1/N reduction scaler).
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_ulp_sharded_non_tile_aligned_width(device, w, distribution, use_welford, dtype):
    """Sharded layer_norm over a non-tile-aligned width vs torch golden, for BF16 and FP32 inputs.

    LayerNorm normalizes over the logical width and must ignore the tile padding. This test exercises
    the single-core sharded path; the interleaved path is covered by the 37x41 shape above, and the
    multi-core case (a non-tile-aligned width split across cores, on the legacy and Welford reduce
    paths) is covered by test_layer_norm_sharded_uneven_multicore_logical_width in
    test_layer_norm_sharded.py. Poisoning the implicit tile padding with a large out-of-distribution
    value makes any read of the padded columns observable: a kernel that normalizes over the logical
    width is unaffected, while one that folds the padded columns into the mean/variance produces a
    grossly wrong result.
    """
    _run_sharded_norm_ulp(
        device, "layernorm", w, num_cores_w=1, distribution=distribution, dtype=dtype, use_welford=use_welford
    )


# Tile-aligned logical width split across two cores, where the shard rounding leaves a whole padding
# tile on the final core, because all cores own the same number of tiles.
# For example, w=96 results in 3 tiles, which when sharded on two cores results in two real tiles on
# the first core, and one real tile + one padding tile on the second core (since sharding rules dictate
# that all cores must get the same number of tiles).
# Unlike the non-tile-aligned cases, no width tile is partially valid here: the whole final tile is
# padding, and the op must still normalize over the logical width rather than the padded shard width.
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_sharded_tile_aligned_width_split_across_cores(device, use_welford):
    """Sharded layer_norm over a tile-aligned width (w=96) split across two cores vs torch golden."""
    _run_sharded_norm_ulp(
        device, "layernorm", w=96, num_cores_w=2, distribution="normal", dtype=torch.bfloat16, use_welford=use_welford
    )


# Widths that are not multiples of the tile width (32), each on a single core so the
# whole logical row plus its tile padding lives in one shard.
@pytest.mark.parametrize("w", [40, 72, 200])
# Restricted to small-range distributions for the same reason as the layer_norm sharded test: a
# wide-range distribution inflates the near-zero atol tolerance enough to hide padding contamination.
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_ulp_sharded_non_tile_aligned_width(device, w, distribution, dtype):
    """Sharded rms_norm over a non-tile-aligned width vs torch golden, for BF16 and FP32 inputs.

    RMSNorm normalizes over the mean of squares of the logical width and must ignore the tile
    padding. Poisoning the implicit tile padding with a large out-of-distribution value makes any
    read of the padded columns observable: a kernel that normalizes over the logical width is
    unaffected, while one that folds the padded columns into the mean of squares produces a grossly
    wrong result.
    """
    _run_sharded_norm_ulp(device, "rmsnorm", w, num_cores_w=1, distribution=distribution, dtype=dtype)


# Block sharding requires every core to be assigned the same shard size,
# and each shard must be tile-aligned. Therefore, each core's width is its share of the logical width
# rounded up to a tile: shard_w = ceil(w / cores / 32) * 32, and the padded width is cores * shard_w.
# w=40/72/200 are non-tile-aligned. w=96 is tile-aligned but still does not tile evenly when sharded
# across two cores (96 / 2 = 48, which is not divisible by 32). Because the shard width is rounded up,
# the logical columns can run out partway through the second core's block, leaving it with
# a mix of tile kinds: fully-valid tiles, at most one partially-valid tile, and fully-padding tiles.
# For example, for w=200 and 2 cores: shard_w = ceil(200/2/32)*32 = 128 (4 tiles per core), so the padded width is
# 256. The first core's tiles (columns 0-127) are all fully valid. The second core's tiles (columns 128-255) are
# valid only through column 199, so the four tiles are:
# - Tiles 1 and 2 (columns 128-159 and 160-191) fully valid.
# - Tile 3 (columns 192-223) partially valid (only 192-199 valid).
# - Tile 4 (columns 224-255) fully padding.
@pytest.mark.parametrize("w", [40, 72, 96, 200])
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_ulp_sharded_unevenly_split_width_across_cores(device, w, distribution, dtype):
    """Sharded rms_norm over a width split across two cores vs torch golden.

    Splitting the logical width across two shards places the padding in whichever shard holds the
    final columns; the op must still normalize over the logical width and ignore that padding.
    Poisoning the padding makes any read of the padded columns observable.
    """
    _run_sharded_norm_ulp(device, "rmsnorm", w, num_cores_w=2, distribution=distribution, dtype=dtype)


@pytest.mark.parametrize("w", [40, 72, 200])
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("num_cores_w", [1, 2])
def test_norm_ulp_sharded_non_tile_aligned_residual(device, w, distribution, dtype, norm, num_cores_w):
    """Sharded layer_norm / rms_norm with a fused residual add: the norm is applied to a + b (a = the
    input tensor, b = the residual tensor), over a non-tile-aligned width.

    Runs the legacy (non-Welford) reduce. That is the path where the fused residual interacts with
    padding handling: it reads full width tiles, including the padding columns, and zeroes them with a
    column mask, so the sum a + b (whose padding columns are themselves compute-produced) must still be
    masked out of the statistics. Welford reduces only each core's logical column count and ignores the
    padding, so a fused residual adds no new padding risk there.
    Both a and b have their tile padding poisoned so any read of the padded columns is observable.
    """
    _run_sharded_norm_ulp(
        device, norm, w, num_cores_w=num_cores_w, distribution=distribution, dtype=dtype, residual=True
    )
