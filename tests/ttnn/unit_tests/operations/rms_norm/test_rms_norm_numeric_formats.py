# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — numerical configurability: bfloat8_b (input and gamma) + fp32_dest_acc_en=False.

Pins:
- fp32_dest_acc_en=False for bf16 / bf8b inputs (16-bit DEST, Float16_b intermediate CBs) on both regimes
  (R1 row-local, R2 W-split root combine) and both layouts;
- bfloat8_b input (TILE only — bf8b has no ROW_MAJOR form) on tile-aligned shapes;
- bfloat8_b TILE gamma (whole-page writer load) under bf16 / bf8b inputs, both layouts of the input;
- the perf-case precision corner (bf16, HiFi2, fp32_dest_acc_en=False, bf16 TILE gamma, PCC >= 0.9995);
- {float32, fp32_dest_acc_en=False} and bf8b x non-aligned still raise ExcludedCell.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.rms_norm import rms_norm

PCC_THRESHOLD = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def _pcc(expected, actual):
    e = expected.to(torch.float32).flatten()
    a = actual.to(torch.float32).flatten()
    assert torch.isfinite(a).all(), "output contains non-finite values"
    e = e - e.mean()
    a = a - a.mean()
    return float((e * a).sum() / (torch.sqrt((e * e).sum()) * torch.sqrt((a * a).sum())))


def _run(device, shape, dtype, layout, *, gamma_dtype=None, gamma_layout=None, fp32_acc, fidelity=None, pcc=None):
    torch.manual_seed(7)
    W = shape[-1]
    torch_x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Reference sees exactly what the device sees (bf8b quantization of the input / gamma included).
    torch_x = ttnn.to_torch(tt_x).to(torch.float32)
    kwargs = {}
    torch_gamma = None
    if gamma_dtype is not None:
        g = torch.randn((1, 1, 1, W), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tt_g = ttnn.from_torch(
            g, dtype=gamma_dtype, layout=gamma_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        torch_gamma = ttnn.to_torch(tt_g).to(torch.float32).reshape(-1)
        kwargs["gamma"] = tt_g
    kwargs["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity or ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc
    )
    tt_out = rms_norm(tt_x, **kwargs)
    assert tt_out.dtype == dtype and tt_out.layout == layout
    actual = ttnn.to_torch(tt_out)
    expected = torch_x * torch.rsqrt(torch_x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    if torch_gamma is not None:
        expected = expected * torch_gamma
    value = _pcc(expected, actual)
    rel_rms = float((actual.float() - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt())
    print(
        f"\nNUMERIC_FORMATS shape={shape} dtype={dtype} layout={layout} gamma={gamma_dtype} fp32_acc={fp32_acc} "
        f"pcc={value:.6f} rel_rms={rel_rms:.5f}"
    )
    threshold = pcc if pcc is not None else PCC_THRESHOLD[dtype]
    assert value >= threshold, f"PCC {value:.6f} < {threshold}"


LAYOUTS = [pytest.param(ttnn.TILE_LAYOUT, id="TILE"), pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="RM")]

SHAPES = [
    pytest.param((1, 1, 64, 128), id="R1_small"),
    pytest.param((2, 1, 64, 320), id="R1_multi_batch"),
    pytest.param((1, 1, 2048, 1024), id="R1_tall"),
    pytest.param((1, 1, 32, 7168), id="R2_decode"),
    pytest.param((1, 1, 32, 16384), id="R2_wide"),
]


@pytest.mark.parametrize("gamma", [None, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["no_gamma", "tile_g", "rm_g"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_bf16_fp32_dest_off(device, shape, layout, gamma):
    _run(
        device,
        shape,
        ttnn.bfloat16,
        layout,
        gamma_dtype=None if gamma is None else ttnn.bfloat16,
        gamma_layout=gamma,
        fp32_acc=False,
    )


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize(
    "gamma_dtype, gamma_layout",
    [
        (None, None),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.float32, ttnn.TILE_LAYOUT),
    ],
    ids=["no_gamma", "bf8b_tile_g", "bf16_rm_g", "fp32_tile_g"],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_bf8b_input(device, shape, gamma_dtype, gamma_layout, fp32_acc):
    _run(
        device,
        shape,
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        gamma_dtype=gamma_dtype,
        gamma_layout=gamma_layout,
        fp32_acc=fp32_acc,
    )


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 128), id="R1"),
        pytest.param((1, 1, 47, 50), id="R1_non_aligned"),
        pytest.param((1, 1, 32, 16300), id="R2_non_aligned"),
    ],
)
def test_bf8b_tile_gamma_bf16_input(device, shape, layout, fp32_acc):
    _run(
        device,
        shape,
        ttnn.bfloat16,
        layout,
        gamma_dtype=ttnn.bfloat8_b,
        gamma_layout=ttnn.TILE_LAYOUT,
        fp32_acc=fp32_acc,
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 1024), (1, 1, 32, 7168), (1, 1, 640, 6144), (1, 1, 8192, 1024)],
    ids=["decode_1024", "decode_7168", "t640_6144", "prefill_1024"],
)
def test_perf_case_precision_corner(device, shape):
    # _perf_case corner: bf16 / HiFi2 / fp32_dest_acc_en=False / bf16 TILE gamma, soft PCC 0.9995.
    _run(
        device,
        shape,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.TILE_LAYOUT,
        fp32_acc=False,
        fidelity=ttnn.MathFidelity.HiFi2,
        pcc=0.9995,
    )


@pytest.mark.parametrize(
    "dtype, shape, fp32_acc",
    [
        (ttnn.float32, (1, 1, 32, 64), False),
        (ttnn.bfloat8_b, (1, 1, 32, 50), True),
        (ttnn.bfloat8_b, (1, 1, 17, 64), True),
    ],
    ids=["fp32_fp32_dest_off", "bf8b_w_non_aligned", "bf8b_h_non_aligned"],
)
def test_excluded_cells_refused(device, dtype, shape, fp32_acc, expect_error):
    tt_x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(ExcludedCell, r"(?s).*"):
        rms_norm(tt_x, compute_kernel_config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_acc))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
def test_row_local_chunked_rows_fp32_dest_off(device, dtype):
    # Selector shift: 16-bit intermediates free L1, so (1,1,8192,4096) bf16 no longer chunks its rows at
    # fp32_dest_acc_en=False. W=8192 still does on a 1.5 MiB-L1 grid of >= 120 cores (bf16: 5 chunks per row,
    # bf8b: 2), keeping the R1 chunks_per_row > 1 path pinned on the 16-bit-DEST datapath.
    _run(
        device,
        (1, 1, 4096, 8192),
        dtype,
        ttnn.TILE_LAYOUT,
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        fp32_acc=False,
    )
