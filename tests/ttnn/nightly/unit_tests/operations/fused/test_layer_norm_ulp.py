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
from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import ttnn_layer_norm_sharded, ttnn_rms_norm_sharded

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
# FP32 tests
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

    Welford layer_norm must normalize over the logical width and ignore the tile
    padding. The interleaved path is covered by the 37x41 shape above.
    This test exercises the sharded path, where the normalization width and
    the per-core reciprocal-LUT size must track the logical width rather than the
    padded shard width.
    Poisoning the implicit tile padding with a large out-of-distribution value makes
    any read of the padded columns observable: a kernel that normalizes over the
    logical width is unaffected, while one that folds the padded columns into the
    mean/variance produces a grossly wrong result.

    For the Welford path the reciprocal LUT is created exactly as production callers do (sized to
    the per-core shard width via ttnn_layer_norm_sharded); the legacy path uses the 1/N reduction
    scaler with a column mask that zeroes the final tile's padding columns. The implicit tile
    padding is poisoned with PAD_VALUE so that normalizing over the padded width corrupts the
    output.
    """
    torch.manual_seed(0)
    h = 32
    padded_w = math.ceil(w / 32) * 32

    torch_input_tensor = _make_ln_input(h, w, dtype, distribution)
    golden = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    # Single-core block-sharded config; the shard width is the tile-padded width.
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [h, padded_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    tt_input_tensor = ttnn.fill_implicit_tile_padding(tt_input_tensor, PAD_VALUE)

    actual = ttnn_layer_norm_sharded(
        device,
        tt_input_tensor,
        use_welford=use_welford,
        block_ht=h // 32,
        block_wt=padded_w // 32,
        subblock_w=1,
    )
    # Discard padding before comparison.
    actual = actual[..., :w]

    # Tight caps so padding contamination is caught rather than absorbed. FP32 uses the FP32 cap
    # (the mask multiply truncates the final tile's valid columns to TF32 on the FPU); BF16 uses the
    # dedicated tight sharded-non-aligned cap rather than the loose BF16-accumulation cap.
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
    spec = f"sharded shape_hw=({h},{w}) padded_w={padded_w} welford={use_welford} dist={distribution} dtype={dtype}"
    logger.info(
        f"ttnn.layer_norm ULP (sharded) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[sharded non-aligned dtype={dtype} w={w} welford={use_welford} dist={distribution}] {msg}"


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
    padding. The legacy (non-Welford) reduce is the only sharded RMSNorm path (Welford does not
    support RMSNorm), so it must mask the padding columns of the final width tile before the mean
    of squares. Poisoning the implicit tile padding with a large out-of-distribution value makes any
    read of the padded columns observable: a kernel that normalizes over the logical width is
    unaffected, while one that folds the padded columns into the mean of squares produces a grossly
    wrong result.
    """
    torch.manual_seed(0)
    h = 32
    padded_w = math.ceil(w / 32) * 32
    eps = 1e-12

    torch_input_tensor = _make_ln_input(h, w, dtype, distribution)
    ms = torch_input_tensor.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    golden = (torch_input_tensor.to(torch.float32) * torch.rsqrt(ms + eps)).to(dtype)

    # Single-core block-sharded config; the shard width is the tile-padded width.
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [h, padded_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    tt_input_tensor = ttnn.fill_implicit_tile_padding(tt_input_tensor, PAD_VALUE)

    actual = ttnn_rms_norm_sharded(
        device,
        tt_input_tensor,
        block_ht=h // 32,
        block_wt=padded_w // 32,
        subblock_w=1,
    )
    # Discard padding before comparison.
    actual = actual[..., :w]

    # Tight caps so padding contamination is caught rather than absorbed, matching the layer_norm
    # sharded test in this file. FP32 uses the FP32 cap; BF16 uses the dedicated tight cap.
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
    spec = f"sharded rms shape_hw=({h},{w}) padded_w={padded_w} dist={distribution} dtype={dtype}"
    logger.info(
        f"ttnn.rms_norm ULP (sharded) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[sharded rms non-aligned dtype={dtype} w={w} dist={distribution}] {msg}"


# Non-tile-aligned widths split across two width shards. RMSNorm masks the compute-produced squares
# with the per-shard host mask, which carries each shard's own validity, so the logical width can end
# mid-shard (the last shard then holds full, partial, and all-padding tiles, e.g. w=200 -> 2x128).
@pytest.mark.parametrize("w", [40, 72, 200])
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_ulp_sharded_non_tile_aligned_multi_width_shard(device, w, distribution, dtype):
    """Sharded rms_norm over a non-tile-aligned width split across two width shards vs torch golden.

    Each width shard owns block_w columns of the tile-padded width; the cross-core reduce combines
    them. The per-shard host mask zeroes the padding columns of whichever shard they fall in, and the
    reduction scaler lands the divide on the logical width. Poisoning the padding makes any read of the
    padded columns observable.
    """
    torch.manual_seed(0)
    h = 32
    num_cores_w = 2
    shard_w = math.ceil(w / num_cores_w / 32) * 32
    padded_w = shard_w * num_cores_w
    eps = 1e-12

    torch_input_tensor = _make_ln_input(h, w, dtype, distribution)
    ms = torch_input_tensor.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    golden = (torch_input_tensor.to(torch.float32) * torch.rsqrt(ms + eps)).to(dtype)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, 0))}),
        [h, shard_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    tt_input_tensor = ttnn.fill_implicit_tile_padding(tt_input_tensor, PAD_VALUE)

    actual = ttnn_rms_norm_sharded(
        device,
        tt_input_tensor,
        block_ht=h // 32,
        block_wt=shard_w // 32,
        subblock_w=1,
    )
    actual = actual[..., :w]

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
    spec = f"sharded rms multi-shard shape_hw=({h},{w}) shard_w={shard_w} dist={distribution} dtype={dtype}"
    logger.info(
        f"ttnn.rms_norm ULP (sharded multi-shard) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[sharded rms multi-shard dtype={dtype} w={w} dist={distribution}] {msg}"


# Fused residual add (a + b computed on-device) over a non-tile-aligned width, on the legacy
# (non-Welford) path. The normalized input is compute-produced, so the column mask must still exclude
# the padding columns from the statistics. num_cores_w=2 splits the width across shards (RMSNorm only;
# LayerNorm rejects a non-tile-aligned width across multiple width shards).
@pytest.mark.parametrize("w", [40, 72, 200])
@pytest.mark.parametrize("distribution", ["normal", "centered_uniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("num_cores_w", [1, 2])
def test_norm_ulp_sharded_non_tile_aligned_residual(device, w, distribution, dtype, norm, num_cores_w):
    """Sharded layer_norm / rms_norm with a fused residual add over a non-tile-aligned width.

    The residual sum a + b is computed on-device, so the normalized input is compute-produced rather
    than host-tilized; the column mask must still exclude the padding columns from the statistics.
    Both the input and residual tile padding are poisoned so any read of the padded columns is
    observable.
    """
    if norm == "layernorm" and num_cores_w > 1:
        pytest.skip("LayerNorm does not support a non-tile-aligned width split across multiple width shards")
    torch.manual_seed(0)
    h = 32
    shard_w = math.ceil(w / num_cores_w / 32) * 32
    padded_w = shard_w * num_cores_w
    eps = 1e-12

    torch_a = _make_ln_input(h, w, dtype, distribution)
    torch_b = _make_ln_input(h, w, dtype, distribution)
    torch_sum = torch_a + torch_b
    if norm == "layernorm":
        golden = torch.nn.functional.layer_norm(torch_sum, normalized_shape=[w])
    else:
        ms = torch_sum.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        golden = (torch_sum.to(torch.float32) * torch.rsqrt(ms + eps)).to(dtype)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, 0))}),
        [h, shard_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    def to_sharded_poisoned(t):
        tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config)
        return ttnn.fill_implicit_tile_padding(tt, PAD_VALUE)

    tt_a = to_sharded_poisoned(torch_a)
    tt_b = to_sharded_poisoned(torch_b)

    if norm == "layernorm":
        actual = ttnn_layer_norm_sharded(
            device, tt_a, use_welford=False, block_ht=h // 32, block_wt=shard_w // 32, subblock_w=1, residual=tt_b
        )
    else:
        actual = ttnn_rms_norm_sharded(
            device, tt_a, block_ht=h // 32, block_wt=shard_w // 32, subblock_w=1, residual=tt_b
        )
    actual = actual[..., :w]

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
    spec = f"sharded {norm} residual shape_hw=({h},{w}) padded_w={padded_w} dist={distribution} dtype={dtype}"
    logger.info(
        f"ttnn {norm} ULP (sharded residual) | {spec} | ulp mean={ulp_stats['mean']:.3g} p95={ulp_stats['p95']:.3g} p99={ulp_stats['p99']:.3g} max={max_ulp:.4g}/{ulp_threshold} atol {max_atol_err:.4g}/{atol_tol:.4g} | {'ok' if passed else 'FAIL'}"
    )
    if ulp_stats["worst"]:
        logger.info(f"  worst: {ulp_stats['worst']}")
    if not passed:
        logger.info(f"  {msg}")
    assert passed, f"[sharded {norm} residual dtype={dtype} w={w} dist={distribution}] {msg}"


@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_ulp_sharded_non_tile_aligned_multi_width_shard_rejected(device, use_welford):
    """LayerNorm (non-RMS) over a non-tile-aligned width split across multiple width shards is
    rejected, not silently wrong.

    LayerNorm masks the host-tilized input at the E[x] site with one partial tile at the end, which is
    only correct when a single width shard owns the whole row; the Welford partial-tile handling has
    the same single-shard assumption in its cross-core combine. The op must throw rather than normalize
    the full shards over their padding columns. (RMSNorm, by contrast, masks the per-shard squares and
    does support this split.)
    """
    torch.manual_seed(0)
    h, w = 32, 200  # 200 -> padded 256; split across 2 width shards of 128 (4 tiles) each
    num_cores_w = 2
    shard_w = math.ceil(w / num_cores_w / 32) * 32

    torch_input_tensor = _make_ln_input(h, w, torch.bfloat16, "normal")
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, 0))}),
        [h, shard_w],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    with pytest.raises(RuntimeError, match="non-tile-aligned width"):
        ttnn_layer_norm_sharded(
            device,
            tt_input_tensor,
            use_welford=use_welford,
            block_ht=h // 32,
            block_wt=shard_w // 32,
            subblock_w=1,
        )
