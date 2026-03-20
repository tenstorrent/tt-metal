# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the row-major binary eltwise kernel.

Each test compares the custom kernel output (row-major path, no tilize/untilize)
against the torch reference, checking both numeric correctness and that the
output remains in ROW_MAJOR_LAYOUT.
"""

import csv
import math
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger


from tests.ttnn.utils_for_testing import assert_with_ulp

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# (B, T, H, W, C) — all sticks = B*T*H*W, stick_size = C
SHAPES = [
    (1, 32),
    (2, 32),
    # (1, 2, 4, 4, 1024),   # typical VAE block shape, C = tile width
    # (1, 1, 4, 4, 512),    # half-width stick
    # (2, 1, 2, 2, 256),    # batch=2, smaller C
    # (1, 4, 1, 1, 1024),   # temporal dimension only
]


OPS = ["add", "mul"]

DTYPES = [ttnn.bfloat16, ttnn.float32]

# Minimum acceptable PSNR (dB) per dtype.
# bf16 has ~2–3 significant decimal digits, so quantisation noise limits PSNR
# to roughly 40–50 dB for typical randn tensors.  fp32 should be near-exact.
_MIN_PSNR = {
    ttnn.bfloat16: 40.0,
    ttnn.float32: 70.0,
}


def _torch_dtype(ttnn_dtype):
    return torch.bfloat16 if ttnn_dtype == ttnn.bfloat16 else torch.float32


def compute_psnr(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio between *out* and *ref*.

    PSNR = 10 * log10(peak² / MSE)

    *peak* is the maximum absolute value in *ref*, which normalises the metric
    to the dynamic range of the reference signal.  Returns ``float('inf')``
    when the tensors are identical.
    """
    out_f = out.float()
    ref_f = ref.float()
    mse = (out_f - ref_f).pow(2).mean().item()
    if mse == 0.0:
        return float("inf")
    peak = ref_f.abs().max().item()
    if peak == 0.0:
        peak = 1.0
    return 10.0 * math.log10(peak**2 / mse)


def save_tensor_to_csv(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a torch tensor to a CSV file with aligned row/column indices.

    The output has one header row with column indices (col_0, col_1, …) and
    one header column with the flat row index.  The tensor is first cast to
    float32 so that bfloat16 values are printed as readable decimals.

    For tensors with more than 2 dimensions the trailing two dimensions are
    used as (rows, cols); all leading dimensions are flattened into the row
    axis so the file is always 2-D.
    """
    t = tensor.detach().float()
    if t.dim() == 1:
        t = t.unsqueeze(0)  # treat a 1-D vector as a single row
    t = t.reshape(-1, t.shape[-1])  # flatten leading dims → (N_rows, N_cols)

    n_rows, n_cols = t.shape
    # Width needed to right-align the row-index column
    row_idx_width = len(str(n_rows - 1))
    col_idx_width = max(len(str(n_cols - 1)), 8)  # at least 8 chars per value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Header row: blank corner cell + column indices
        header = ["row \\ col"] + [f"col_{c:>{col_idx_width - 4}}" for c in range(n_cols)]
        writer.writerow(header)
        for r in range(n_rows):
            row_label = f"row_{r:>{row_idx_width}}"
            values = [f"{v:.6g}" for v in t[r].tolist()]
            writer.writerow([row_label] + values)
    logger.info(f"Saved tensor {list(tensor.shape)} → {path}")


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_minimal_rm_binary(device, shape, op, dtype):
    """Validate that ttnn.experimental.dit_minimal_rm_binary matches torch element-wise arithmetic."""
    tdtype = _torch_dtype(dtype)

    torch.manual_seed(0)

    # Generate random inputs; keep values small to avoid large fp32/bf16 errors
    # A_torch = torch.rand(shape, dtype=tdtype) * 2.0 - 1.0
    # B_torch = torch.rand(shape, dtype=tdtype) * 2.0 - 1.0

    # A_torch = torch.randint(0, 10, shape, dtype=tdtype)
    # B_torch = torch.randint(0, 10, shape, dtype=tdtype)

    A_torch = torch.full(shape, 1.0, dtype=tdtype)
    B_torch = torch.full(shape, 2.0, dtype=tdtype)

    # print(f"A_torch: {A_torch}")
    # print(f"B_torch: {B_torch}")

    # Torch ground truth
    if op == "add":
        C_ref = A_torch + B_torch
    else:
        C_ref = A_torch * B_torch

    # Build row-major device tensors
    A_rm = ttnn.from_torch(
        A_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B_rm = ttnn.from_torch(
        B_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run the custom kernel
    out = ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op=op)

    print(f"out.padded_shape: {out.padded_shape}")

    # Output must stay in row-major layout
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR output, got {out.layout}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} != {dtype}"
    assert out.padded_shape == A_rm.padded_shape, "Shape mismatch in output"

    # Compare values
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)

    num_sticks = math.prod(shape[:-1])
    max_abs_err = (C_out - C_ref_cast).abs().max().item()
    psnr = compute_psnr(C_out, C_ref_cast)
    min_psnr = _MIN_PSNR[dtype]

    logger.info(
        f"shape={shape} op={op} dtype={dtype} " f"sticks={num_sticks} max_abs_err={max_abs_err:.3e} PSNR={psnr:.2f} dB"
    )
    assert psnr >= min_psnr, (
        f"ttnn.experimental.dit_minimal_rm_binary({op}, {dtype}) PSNR too low — "
        f"{psnr:.2f} dB < {min_psnr} dB (max_abs_err={max_abs_err:.3e})"
    )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
    ids=[
        "bf16",
        # "fp32"
    ],
)
def test_rm_binary_add_single_stick(device, dtype):
    """Edge case: tensor with exactly 1 stick (a single row)."""
    shape = (1, 1, 1, 1, 1024)
    tdtype = _torch_dtype(dtype)

    # A_torch = torch.randint(0, 10, shape, dtype=tdtype)
    # B_torch = torch.randint(0, 10, shape, dtype=tdtype)

    A_torch = torch.full(shape, 1.0, dtype=tdtype)
    B_torch = torch.full(shape, 2.0, dtype=tdtype)

    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op="add")
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)

    save_tensor_to_csv(C_out, "debug_csv/C_out.csv")
    save_tensor_to_csv(C_ref_cast, "debug_csv/C_ref.csv")

    psnr = compute_psnr(C_out, C_ref_cast)
    min_psnr = _MIN_PSNR[dtype]
    max_abs_err = (C_out - C_ref_cast).abs().max().item()
    logger.info(f"single_stick add dtype={dtype} PSNR={psnr:.2f} dB max_abs_err={max_abs_err:.3e}")
    assert psnr >= min_psnr, (
        f"ttnn.experimental.dit_minimal_rm_binary(add, {dtype}) single-stick PSNR too low — "
        f"{psnr:.2f} dB < {min_psnr} dB (max_abs_err={max_abs_err:.3e})"
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rm_binary_mul_large(device, dtype):
    """Large tensor stressing multi-core distribution."""
    # shape = [1, 1, 32, 110, 96]  # 512 sticks
    shape = [1, 1, 32, 220, 96]  # 512 sticks
    tdtype = _torch_dtype(dtype)

    torch.manual_seed(0)

    # Use [0.5, 1.5] range to avoid large accumulation errors in bf16
    # A_torch = torch.randn(shape, dtype=tdtype)
    # B_torch = torch.randn(shape, dtype=tdtype)

    A_torch = torch.randint(0, 10, shape, dtype=tdtype)
    B_torch = torch.randint(0, 10, shape, dtype=tdtype)

    # Debug
    # A_torch = torch.full(shape, 1.0, dtype=tdtype)
    # B_torch = torch.full(shape, 2.0, dtype=tdtype)

    # print(f"A_torch: \n{A_torch}")
    # print(f"B_torch: \n{B_torch}")

    C_ref = A_torch * B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op="mul")
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)

    pos = (0, 0, 0, 1, 9)
    print(f"A[0, 0, 0, 1, 9] = {A_torch[pos]}")
    print(f"B[0, 0, 0, 1, 9] = {B_torch[pos]}")
    print(f"C_ref[0, 0, 0, 1, 9] = {C_ref[pos]}")
    print(f"C_out[0, 0, 0, 1, 9] = {C_out[pos]}")

    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)


# ---------------------------------------------------------------------------
# Wan VAE residual-add shapes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wan VAE production shapes
#
# These correspond to the residual add at WanResidualBlock.forward line 571-573:
#   x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
#   x_tile_BTHWC = ttnn.add(h_tile_BTHWC, x_tile_BTHWC)
#
# Both operands have identical shape (B, T, H, W, out_dim) so ttnn.experimental.dit_minimal_rm_binary
# can replace this add directly (once h_tile_BTHWC is kept in ROW_MAJOR).
#
# Channels (96, 192, 384) are all multiples of ALIGNMENT=32, so padded_shape
# equals logical_shape for ROW_MAJOR tensors — no hidden padding to worry about.
#
# NOTE: The WanConv2d mask multiply at line 765-766:
#   x_BTHWC = ttnn.mul(x_BTHWC, mask)   # mask shape: (1, 1, H, 1, 1)
# uses broadcasting (mask is NOT the same shape as x). ttnn.experimental.dit_minimal_rm_binary does not
# support broadcast, so that call cannot be replaced with the current kernel.
# ---------------------------------------------------------------------------
WAN_RESIDUAL_ADD_SHAPES = [
    (1, 1, 90, 160, 384),  # decoder.mid_block.resnets.0          — 14 400 sticks
    (1, 2, 180, 320, 384),  # decoder.up_blocks.1.resnets.{0,1}    — 115 200 sticks
    (1, 4, 360, 640, 192),  # decoder.up_blocks.2.resnets.0        — 921 600 sticks
    (1, 4, 720, 1280, 96),  # decoder.up_blocks.3.resnets.0        — 3 686 400 sticks
]


@pytest.mark.parametrize(
    "shape",
    WAN_RESIDUAL_ADD_SHAPES,
    ids=[
        "mid_block-(1,1,90,160,384)",
        "upblock1-(1,2,180,320,384)",
        "upblock2-(1,4,360,640,192)",
        "upblock3-(1,4,720,1280,96)",
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_rm_binary_wan_residual_add(device, shape, dtype):
    """Validate ttnn.experimental.dit_minimal_rm_binary add on the exact WanDecoder3D residual-add shapes.

    Mirrors WanResidualBlock.forward lines 571-573:
        x_tile = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        x_tile = ttnn.add(h_tile_BTHWC, x_tile)

    Both operands have the same BTHWC shape; all C values are multiples of 32,
    so padded_shape == logical_shape for ROW_MAJOR (no hidden tile-padding).
    """
    tdtype = _torch_dtype(dtype)
    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype) * 2.0 - 1.0
    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_rm_binary(A_rm, B_rm, op="add")

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR output, got {out.layout}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} != {dtype}"
    assert out.padded_shape == A_rm.padded_shape, "Shape mismatch in output"

    print(f"out.padded_shape: {out.padded_shape}")

    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)

    num_sticks = math.prod(shape[:-1])
    max_abs_err = (C_out - C_ref_cast).abs().max().item()
    psnr = compute_psnr(C_out, C_ref_cast)
    min_psnr = _MIN_PSNR[dtype]
    logger.info(f"shape={shape} dtype={dtype} sticks={num_sticks} " f"max_abs_err={max_abs_err:.3e} PSNR={psnr:.2f} dB")
    assert psnr >= min_psnr, (
        f"ttnn.experimental.dit_minimal_rm_binary(add, {dtype}) Wan-shape PSNR too low — "
        f"{psnr:.2f} dB < {min_psnr} dB (max_abs_err={max_abs_err:.3e})"
    )
