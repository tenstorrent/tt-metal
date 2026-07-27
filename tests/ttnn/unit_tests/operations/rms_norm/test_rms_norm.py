# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for ttnn.operations.rms_norm.

IMMUTABLE — this file is the specification. The implementer must NOT modify it.

Coverage mirrors `ttnn/ttnn/operations/rms_norm/op_design.md`:
  * shapes: single-tile, multi-tile, non-square, multi-batch, wide-hidden
  * layouts: TILE and ROW_MAJOR (both native — no host-side to_layout/pad/slice)
  * dtypes: bfloat16 and float32
  * gamma: absent, present (TILE gamma, ROW_MAJOR gamma, mixed dtype)
  * alignment: tile_aligned, w_non_aligned, h_non_aligned, both
  * ranks: 2, 3, 4
  * regime pinning: one shape per compile-time regime named in op_design.md §5.2
    (X_RESIDENT / GAMMA_RESIDENT / NW>1 / HT_BLOCK>1 / streaming fallback), so a
    regime cannot silently go untested on a particular grid size.

PCC thresholds are keyed by dtype and match the golden suite exactly:
  float32 -> 0.999, bfloat16 -> 0.995, bfloat8_b -> 0.99
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from ttnn.operations.rms_norm import rms_norm


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #

TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

# Same thresholds as the golden suite — do NOT tighten these.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """RMSNorm reference, computed in fp32, returned in the input dtype."""
    original_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(original_dtype)


def _pcc_for(dtype, gamma_dtype=None):
    """Loosest of the involved dtypes governs."""
    thresholds = [PCC[dtype]]
    if gamma_dtype is not None:
        thresholds.append(PCC[gamma_dtype])
    return min(thresholds)


def _run(
    device,
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    with_gamma=True,
    gamma_dtype=None,
    gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    torch.manual_seed(42)

    torch_dtype = TORCH_DTYPE[dtype]
    torch_x = torch.randn(shape, dtype=torch_dtype)

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)

    torch_gamma = None
    tt_gamma = None
    if with_gamma:
        g_dtype = gamma_dtype if gamma_dtype is not None else dtype
        width = shape[-1]
        torch_gamma = torch.randn(width, dtype=TORCH_DTYPE[g_dtype])
        tt_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, width),
            dtype=g_dtype,
            layout=gamma_layout,
            device=device,
        )

    expected = torch_rms_norm(torch_x, gamma=torch_gamma, epsilon=epsilon)

    kwargs = {}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=epsilon, **kwargs)

    # The op must preserve shape, dtype and layout — no host-side fixups.
    assert tuple(tt_out.shape) == tuple(shape), f"shape {tuple(tt_out.shape)} != {tuple(shape)}"
    assert tt_out.dtype == dtype, f"dtype {tt_out.dtype} != {dtype}"
    assert tt_out.layout == layout, f"layout {tt_out.layout} != {layout}"

    actual = ttnn.to_torch(tt_out)

    threshold = _pcc_for(dtype, gamma_dtype if with_gamma else None)
    passed, message = check_with_pcc(expected.to(torch.float32), actual.to(torch.float32), threshold)
    assert passed, message


# --------------------------------------------------------------------------- #
# 1. Core shape / dtype / layout matrix
# --------------------------------------------------------------------------- #

CORE_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile, non-square
    (2, 4, 128, 512),  # multi-batch, multi-channel
    (1, 1, 32, 1024),  # wide hidden, single tile-row
    (4, 1, 512, 256),  # tall, multi-batch
]


@pytest.mark.parametrize("shape", CORE_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_core(device, shape, dtype, layout):
    _run(device, shape, dtype=dtype, layout=layout, with_gamma=True)


@pytest.mark.parametrize("shape", CORE_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_no_gamma(device, shape, layout):
    """rms_norm(x) — the no-weight call pattern."""
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout, with_gamma=False)


# --------------------------------------------------------------------------- #
# 2. Rank coverage (2D / 3D / 4D) — leading dims collapse into the row axis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        (32, 64),  # 2D
        (128, 512),  # 2D, larger
        (1, 32, 128),  # 3D
        (4, 128, 512),  # 3D, multi-batch
        (1, 1, 64, 128),  # 4D
        (4, 8, 32, 256),  # 4D, multi-batch/channel
    ],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_rank(device, shape, layout):
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout, with_gamma=True)


# --------------------------------------------------------------------------- #
# 3. Non-tile-aligned shapes — must be native, no host padding/slicing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 50),  # W non-aligned, H aligned
        (1, 1, 64, 17),  # W non-aligned, very narrow
        (4, 8, 32, 47),  # W non-aligned, multi-batch
        (1, 1, 17, 64),  # H non-aligned, W aligned
        (1, 1, 50, 128),  # H non-aligned
        (1, 1, 17, 50),  # both non-aligned
        (2, 1, 100, 47),  # both non-aligned, multi-batch
        (32, 17),  # 2D, W non-aligned
        (1, 17, 128),  # 3D, H non-aligned
    ],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_non_aligned(device, shape, layout):
    """The RMS denominator must reflect only valid (non-padding) elements."""
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout, with_gamma=True)


# --------------------------------------------------------------------------- #
# 4. gamma format matrix — layout and dtype are independent of the activation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["g_bf16", "g_fp32"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_gamma_formats(device, dtype, gamma_dtype, gamma_layout):
    """Mixed-precision weights (bf16 activations + fp32 gamma) and both gamma layouts."""
    _run(
        device,
        (1, 1, 64, 128),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        with_gamma=True,
        gamma_dtype=gamma_dtype,
        gamma_layout=gamma_layout,
    )


# --------------------------------------------------------------------------- #
# 5. Regime pinning (op_design.md §5.2)
#
#    Each entry forces a distinct compile-time regime. A regime that only
#    triggers on some grid sizes can pass on one device and fail on another,
#    so every one is pinned by an explicit case here.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape,layout,with_gamma",
    [
        # X_RESIDENT + GAMMA_RESIDENT, NW == 1 (whole row is one chunk)
        pytest.param((1, 1, 64, 128), ttnn.TILE_LAYOUT, True, id="resident_nw1"),
        # X_RESIDENT + GAMMA_RESIDENT, NW > 1 (chunked reduce with cross-call accumulate)
        pytest.param((1, 1, 32, 4096), ttnn.TILE_LAYOUT, True, id="resident_nw_gt1"),
        # X_RESIDENT, gamma too wide to stay resident
        pytest.param((1, 1, 32, 8192), ttnn.TILE_LAYOUT, True, id="x_resident_gamma_streaming"),
        # neither resident -> the bounded streaming fallback (two read passes)
        pytest.param((1, 1, 32, 16384), ttnn.TILE_LAYOUT, True, id="streaming_fallback"),
        # HT_BLOCK > 1: narrow W, many tile-rows -> multi-row compute blocks
        pytest.param((1, 1, 2048, 64), ttnn.TILE_LAYOUT, True, id="ht_block_gt1"),
        # HT_BLOCK > 1 without gamma
        pytest.param((1, 1, 1024, 64), ttnn.TILE_LAYOUT, False, id="ht_block_gt1_no_gamma"),
        # more tile-rows than a single core can hold -> multi-block-per-core loop
        pytest.param((1, 1, 4096, 256), ttnn.TILE_LAYOUT, True, id="many_row_blocks"),
        # HAS_PARTIAL_W with a chunked reduce (mask lands on the final chunk)
        pytest.param((1, 1, 32, 4050), ttnn.TILE_LAYOUT, True, id="partial_w_nw_gt1"),
        # ROW_MAJOR chunked path (tilize/untilize per W-chunk)
        pytest.param((1, 1, 32, 4096), ttnn.ROW_MAJOR_LAYOUT, True, id="rm_nw_gt1"),
        # ROW_MAJOR with both axes non-aligned
        pytest.param((1, 1, 17, 50), ttnn.ROW_MAJOR_LAYOUT, True, id="rm_both_non_aligned"),
    ],
)
def test_rms_norm_regimes(device, shape, layout, with_gamma):
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout, with_gamma=with_gamma)


# --------------------------------------------------------------------------- #
# 6. Call patterns from the op contract
# --------------------------------------------------------------------------- #


def test_rms_norm_default_epsilon(device):
    """rms_norm(x) — no gamma, default epsilon."""
    _run(device, (1, 1, 64, 128), with_gamma=False)


@pytest.mark.parametrize("epsilon", [1e-2, 1e-5, 1e-6, 1e-8], ids=["e1em2", "e1em5", "e1em6", "e1em8"])
def test_rms_norm_epsilon(device, epsilon):
    """rms_norm(x, epsilon=...) and rms_norm(x, gamma=g, epsilon=...)."""
    _run(device, (1, 1, 64, 128), with_gamma=True, epsilon=epsilon)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_maxed_precision_config(device, dtype):
    """The Phase-0 maxed-out precision corner, passed explicitly."""
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    _run(device, (1, 1, 64, 512), dtype=dtype, with_gamma=True, compute_kernel_config=config)


def test_rms_norm_default_config_matches_factory(device):
    """`compute_kernel_config=None` must resolve through default_compute_kernel_config()."""
    from ttnn.operations.rms_norm import default_compute_kernel_config

    cfg = default_compute_kernel_config()
    assert bool(cfg.fp32_dest_acc_en), "Phase 0 default must be the maxed-out corner"
    # A fresh descriptor per call — never a shared mutable constant.
    assert default_compute_kernel_config() is not cfg


def test_rms_norm_small_epsilon_dominated_row(device):
    """A near-zero row makes epsilon load-bearing: rsqrt(0 + eps) must be finite."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_x[0, 0, 0, :] = 0.0

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch_rms_norm(torch_x, epsilon=1e-6)
    tt_out = rms_norm(tt_x, epsilon=1e-6)
    actual = ttnn.to_torch(tt_out)

    assert torch.isfinite(actual.to(torch.float32)).all(), "non-finite output on a zero row"
    passed, message = check_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[ttnn.bfloat16])
    assert passed, message


# --------------------------------------------------------------------------- #
# 7. Validation
# --------------------------------------------------------------------------- #


def test_rms_norm_rejects_rank_1(device, expect_error):
    """`validate()` must reject rank < 2. Its message must mention "rank"."""
    torch.manual_seed(42)
    tt_x = ttnn.from_torch(
        torch.randn((64,), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), r"(?i)rank"):
        rms_norm(tt_x)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    """`validate()` must reject gamma whose last dim != input's. Message mentions "gamma"."""
    torch.manual_seed(42)
    tt_x = ttnn.from_torch(
        torch.randn((1, 1, 64, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_gamma = ttnn.from_torch(
        torch.randn((1, 1, 1, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), r"(?i)gamma"):
        rms_norm(tt_x, gamma=tt_gamma)
