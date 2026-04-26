# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the row-major binary eltwise kernel.

Each test compares the custom kernel output (row-major path, no tilize/untilize)
against the torch reference, checking both numeric correctness and that the
output remains in ROW_MAJOR_LAYOUT.
"""

import pytest
import torch
import ttnn


from tests.ttnn.utils_for_testing import assert_with_ulp

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# (B, T, H, W, C) — all sticks = B*T*H*W, stick_size = C
SHAPES = [
    (1, 32),
    (2, 32),
    (32, 3, 5, 96),
    # (1, 1, 4, 4, 512),    # half-width stick
    # (2, 1, 2, 2, 256),    # batch=2, smaller C
    # (1, 4, 1, 1, 1024),   # temporal dimension only
]


OPS = ["add", "mul"]

DTYPES = [ttnn.bfloat16, ttnn.float32]


def _torch_dtype(ttnn_dtype):
    return torch.bfloat16 if ttnn_dtype == ttnn.bfloat16 else torch.float32


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_minimal_binary(device, shape, op, dtype):
    """Validate that ttnn.experimental.dit_minimal_binary matches torch element-wise arithmetic."""
    tdtype = _torch_dtype(dtype)

    torch.manual_seed(0)

    # Generate random inputs; keep values small to avoid large fp32/bf16 errors
    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype)

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
    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op=op)

    print(f"out.padded_shape: {out.padded_shape}")

    # Output must stay in row-major layout
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR output, got {out.layout}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} != {dtype}"
    assert out.padded_shape == A_rm.padded_shape, "Shape mismatch in output"

    # Compare values
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)
    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32],
    ids=["bf16", "fp32"],
)
def test_minimal_binary_add_single_stick(device, dtype):
    """Edge case: tensor with exactly 1 stick (a single row)."""
    shape = (1, 1, 1, 1, 1024)
    tdtype = _torch_dtype(dtype)

    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype)

    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op="add")
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)
    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"], ids=["bf16", "fp32"])
def test_minimal_binary_mul_large(device, dtype):
    """Large tensor stressing multi-core distribution."""
    shape = [1, 12, 32, 111, 96]

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    torch.manual_seed(0)

    A_torch = torch.randn(shape, dtype=torch_dtype)
    B_torch = torch.randn(shape, dtype=torch_dtype)

    C_ref = A_torch * B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op="mul")
    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)

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
# Both operands have identical shape (B, T, H, W, out_dim) so ttnn.experimental.dit_minimal_binary
# can replace this add directly (once h_tile_BTHWC is kept in ROW_MAJOR).
#
# Channels (96, 192, 384) are all multiples of ALIGNMENT=32, so padded_shape
# equals logical_shape for ROW_MAJOR tensors — no hidden padding to worry about.
#
# NOTE: The WanConv2d mask multiply at line 765-766:
#   x_BTHWC = ttnn.mul(x_BTHWC, mask)   # mask shape: (1, 1, H, 1, 1)
# uses broadcasting (mask is NOT the same shape as x). ttnn.experimental.dit_minimal_binary does not
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
def test_minimal_binary_wan_residual_add(device, shape, dtype):
    """Validate ttnn.experimental.dit_minimal_binary add on the exact WanDecoder3D residual-add shapes.

    Mirrors WanResidualBlock.forward lines 571-573:
        x_tile = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        x_tile = ttnn.add(h_tile_BTHWC, x_tile)

    Both operands have the same BTHWC shape; all C values are multiples of 32,
    so padded_shape == logical_shape for ROW_MAJOR (no hidden tile-padding).
    """
    tdtype = _torch_dtype(dtype)
    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype)
    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op="add")

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected ROW_MAJOR output, got {out.layout}"
    assert out.dtype == dtype, f"dtype mismatch: {out.dtype} != {dtype}"
    assert out.padded_shape == A_rm.padded_shape, "Shape mismatch in output"

    print(f"out.padded_shape: {out.padded_shape}")

    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)
    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)


# ---------------------------------------------------------------------------
# preallocated_output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_minimal_binary_preallocated_output_input_b(device, dtype):
    """preallocated_output=input_b: result written in-place into input_b's buffer.

    NCRISC reads input_b[block k] then writes result[block k] back to the same
    buffer — sequential within each block, safe. BRISC reads input_a independently.
    This is the approach used by WanResidualBlock to avoid DRAM free-pool aliasing.
    """
    shape = (1, 1, 60, 104, 512)  # encoder mid-block shape at 480p
    tdtype = _torch_dtype(dtype)
    torch.manual_seed(42)

    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype)
    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op="add", preallocated_output=B_rm)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.dtype == dtype
    assert out.padded_shape == A_rm.padded_shape

    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)
    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_minimal_binary_preallocated_output_separate(device, dtype):
    """preallocated_output is a fresh pre-allocated tensor distinct from both inputs.

    Verifies that result is correctly written when output, input_a, and input_b are
    all distinct DRAM buffers.
    """
    shape = (1, 1, 60, 104, 512)  # encoder mid-block shape at 480p
    tdtype = _torch_dtype(dtype)
    torch.manual_seed(7)

    A_torch = torch.randn(shape, dtype=tdtype)
    B_torch = torch.randn(shape, dtype=tdtype)
    C_ref = A_torch + B_torch

    A_rm = ttnn.from_torch(
        A_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    B_rm = ttnn.from_torch(
        B_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    C_pre = ttnn.from_torch(
        torch.zeros(shape, dtype=tdtype),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.dit_minimal_binary(A_rm, B_rm, op="add", preallocated_output=C_pre)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.dtype == dtype
    assert out.padded_shape == A_rm.padded_shape

    C_out = ttnn.to_torch(out)
    C_ref_cast = C_ref.to(C_out.dtype)
    assert_with_ulp(C_ref_cast, C_out, ulp_threshold=2)
