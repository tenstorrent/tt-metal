# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical configurability + TILE-layout affine weights for groupnorm_sc_N_1_HW_C.

  * test_groupnorm_sc_N_1_HW_C_precision_matrix   — every supported dtype x shapes (aligned and
    non-aligned, small to large) x input distribution at the default compute config (HiFi4, fp32
    DEST, exact SFPU); PCC and relative RMS asserted.
  * test_groupnorm_sc_N_1_HW_C_precision_matrix_fidelity — the compute_kernel_config surface:
    math_fidelity x math_approx_mode x dst_full_sync_en round-trips on two shapes per dtype.
  * test_compute_kernel_config_refusals — fp32_dest_acc_en=False / packer_l1_acc=True are refused
    (the statistics path accumulates in fp32 DEST into fp32 CBs; matmul_block fidelity rule #38306).
  * test_affine_layout_matrix — TILE-layout gamma/beta in bf16 / fp32 / bf8b across the channel
    geometries the reader's lane gather has to serve: interleaved TILE / ROW_MAJOR (identity lane
    map, C % 32 != 0 clipping), RM model shards (periodic c_period = 40 on the direct view), a
    staged hw_mask shard (c0 not a tile multiple), TILE shards in_place, N > 1 straddles.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C_sharded import (
    block_shard_config,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

# (pcc, relative rms) gates per dtype.
PCC_GATE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
RMS_GATE = {ttnn.float32: 0.01, ttnn.bfloat16: 0.02, ttnn.bfloat8_b: 0.10}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def torch_groupnorm_n_1_hw_c(x, num_groups, *, gamma=None, beta=None, eps=1e-5):
    xf = x.to(torch.float32)
    N, _, HW, C = xf.shape
    x_nchw = xf.squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


def metrics(actual, expected):
    a, e = actual.float(), expected.float()
    abs_err = (a - e).abs()
    _, pcc = comp_pcc(e, a, 0.0)  # (passed, pcc) — the value, not a message
    return dict(
        pcc=float(pcc),
        rms=float((abs_err.pow(2).mean().sqrt() / e.pow(2).mean().sqrt().clamp(min=1e-10))),
        finite=bool(torch.isfinite(a).all()),
    )


def run_case(
    device,
    shape,
    num_groups,
    *,
    dtype,
    layout,
    affine="gamma_beta",
    affine_dtype=None,
    affine_layout=None,
    distribution="randn",
    memory_config=None,
    in_place=False,
    compute_kernel_config=None,
):
    torch.manual_seed(0)
    N, _, HW, C = shape
    affine_dtype = affine_dtype or dtype
    if affine_layout is None:  # a block-format weight has no ROW_MAJOR representation
        affine_layout = ttnn.TILE_LAYOUT if affine_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    x = torch.rand(shape) if distribution == "rand" else torch.randn(shape)
    x = x.to(TORCH_DTYPE[dtype])
    gamma = beta = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[affine_dtype])
    if affine == "gamma_beta":
        beta = torch.randn(1, 1, 1, C).to(TORCH_DTYPE[affine_dtype])
    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta)

    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    kwargs = {}
    for name, w in (("gamma", gamma), ("beta", beta)):
        if w is not None:
            kwargs[name] = ttnn.from_torch(w, dtype=affine_dtype, layout=affine_layout, device=device)
    tt_y = groupnorm_sc_N_1_HW_C(
        tt_x, num_groups, in_place=in_place, compute_kernel_config=compute_kernel_config, **kwargs
    )
    assert list(tt_y.shape) == list(shape)
    assert tt_y.dtype == dtype and tt_y.layout == layout
    m = metrics(ttnn.to_torch(tt_y), expected)
    assert m["finite"], "non-finite values in output"
    assert m["pcc"] >= PCC_GATE[dtype], f"PCC {m['pcc']:.6f} < {PCC_GATE[dtype]}"
    return m


DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, id="bfp8"),
]
DISTRIBUTIONS = [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")]
LAYOUTS = [pytest.param(ttnn.TILE_LAYOUT, id="tile"), pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm")]

# (shape, G) — aligned and non-aligned HW / C, whole-tile and straddling groups, small to large.
MATRIX_SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="32x32_small"),
    pytest.param((1, 1, 64, 128), 4, id="64x128"),
    pytest.param((1, 1, 64, 320), 32, id="64x320_straddling"),
    pytest.param((2, 1, 256, 256), 8, id="2x256x256_batch"),
    pytest.param((1, 1, 4096, 640), 32, id="4096x640_sdxl_large"),
    pytest.param((1, 1, 64, 50), 1, id="64x50_C_non_aligned"),
    pytest.param((1, 1, 50, 128), 1, id="50x128_HW_non_aligned"),
    pytest.param((1, 1, 64, 200), 8, id="64x200_both_straddling_C_non_aligned"),
]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape,num_groups", MATRIX_SHAPES)
def test_groupnorm_sc_N_1_HW_C_precision_matrix(device, shape, num_groups, layout, dtype, distribution):
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat8_b:
        pytest.skip("Block formats do not support ROW_MAJOR layout")
    m = run_case(device, shape, num_groups, dtype=dtype, layout=layout, distribution=distribution)
    assert m["rms"] <= RMS_GATE[dtype], f"rel RMS {m['rms']:.5f} > {RMS_GATE[dtype]}"


FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
    pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
    pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
]


@pytest.mark.parametrize("approx", [pytest.param(False, id="exact"), pytest.param(True, id="approx")])
@pytest.mark.parametrize("math_fidelity", FIDELITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,num_groups", [MATRIX_SHAPES[2], MATRIX_SHAPES[7]])
def test_groupnorm_sc_N_1_HW_C_precision_matrix_fidelity(device, shape, num_groups, dtype, math_fidelity, approx):
    # fp32_dest_acc_en stays True (mandatory, see test_compute_kernel_config_refusals); the matmul
    # operands are fp32 CBs so every fidelity is documented-correct, just less precise. LoFi drops
    # mantissa bits of the (1/n)-scaled statistics and of x*scale — PCC still clears the dtype gate.
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity, math_approx_mode=approx, fp32_dest_acc_en=True, dst_full_sync_en=approx
    )
    run_case(device, shape, num_groups, dtype=dtype, layout=ttnn.TILE_LAYOUT, compute_kernel_config=cfg)


def test_compute_kernel_config_refusals(device, expect_error):
    x = ttnn.from_torch(
        torch.randn(1, 1, 64, 64).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(ValueError, "fp32_dest_acc_en"):
        groupnorm_sc_N_1_HW_C(x, 2, compute_kernel_config=ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=False))
    with expect_error(ValueError, "packer_l1_acc"):
        groupnorm_sc_N_1_HW_C(
            x, 2, compute_kernel_config=ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True, packer_l1_acc=True)
        )


# --- TILE-layout affine weights across channel geometries -------------------------------------
AFFINE_DTYPES = [
    pytest.param(ttnn.bfloat16, id="w_bf16"),
    pytest.param(ttnn.float32, id="w_fp32"),
    pytest.param(ttnn.bfloat8_b, id="w_bfp8"),
]
# (shape, G, dtype, layout, shard [h, w] or None, grid, in_place)
AFFINE_GEOMETRIES = [
    pytest.param((1, 1, 64, 320), 32, ttnn.bfloat16, ttnn.TILE_LAYOUT, None, None, False, id="interleaved_tile"),
    pytest.param((1, 1, 64, 320), 32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, None, None, False, id="interleaved_rm_fp32"),
    pytest.param((1, 1, 64, 47), 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, None, None, False, id="interleaved_rm_C47"),
    pytest.param(
        (1, 1, 1024, 320),
        32,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        [128, 40],
        (8, 8),
        True,
        id="rm_model_w40_direct_view",
    ),
    pytest.param(
        (1, 1, 1024, 640), 32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, [128, 80], (8, 8), False, id="rm_model_w80_fp32"
    ),
    pytest.param(
        (1, 1, 1024, 640),
        32,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        [103, 64],
        (10, 10),
        False,
        id="rm_staged_h103_hw_mask",
    ),
    pytest.param(
        (1, 1, 1024, 1280),
        32,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        [128, 160],
        (8, 8),
        True,
        id="tile_model_bfp8_in_place",
    ),
    pytest.param(
        (1, 1, 64, 4096), 32, ttnn.float32, ttnn.TILE_LAYOUT, [32, 384], (11, 2), False, id="tile_ragged_last_col_fp32"
    ),
    pytest.param(
        (2, 1, 256, 1280), 32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, [52, 128], (10, 10), False, id="rm_batch2_straddle"
    ),
]


@pytest.mark.parametrize("affine_dtype", AFFINE_DTYPES)
@pytest.mark.parametrize("shape,num_groups,dtype,layout,shard_shape,grid,in_place", AFFINE_GEOMETRIES)
def test_affine_layout_matrix(device, shape, num_groups, dtype, layout, shard_shape, grid, in_place, affine_dtype):
    mc = block_shard_config(shard_shape, grid) if shard_shape is not None else None
    m = run_case(
        device,
        shape,
        num_groups,
        dtype=dtype,
        layout=layout,
        affine_dtype=affine_dtype,
        affine_layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
        in_place=in_place,
    )
    # bf8b weights quantize gamma/beta at 2^-8 of the 16-lane block maximum; the output error is a
    # per-channel scale error, well inside the bf8b rms gate even on a bf16 activation.
    rms_gate = RMS_GATE[ttnn.bfloat8_b] if affine_dtype == ttnn.bfloat8_b else RMS_GATE[dtype]
    assert m["rms"] <= rms_gate, f"rel RMS {m['rms']:.5f} > {rms_gate}"


@pytest.mark.parametrize("affine", ["gamma_only", "gamma_beta"])
def test_affine_tile_gamma_only_fp32(device, affine):
    run_case(
        device,
        (1, 1, 64, 160),
        5,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        affine=affine,
        affine_dtype=ttnn.bfloat8_b,
        affine_layout=ttnn.TILE_LAYOUT,
    )
