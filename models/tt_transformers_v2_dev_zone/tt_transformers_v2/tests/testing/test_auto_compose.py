#!/usr/bin/env python3
"""
Tests for automatic composition of multi-device sharded tensors using TensorTopology.

This test module validates that the auto-composition logic correctly infers
MeshToTensor composers from a sharded ttnn.Tensor's topology metadata.

It validates both host-sharded and device-sharded cases.
"""

import os
import struct

import pytest
import torch
from tt_transformers_v2.src.testing.auto_compose import to_torch_auto_compose

import ttnn

# ======================================================================================
# Test Parameters
# ======================================================================================


# Try a variety of mesh shapes and tensor layouts; tests skip if device can't be opened
pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            (1, 1),  # single device # [INFO] apply auto_compose on single device would incur error in c++ code
            (1, 2),  # 1D mesh, 2 devices
            (1, 8),  # 1D mesh, 8 devices
            (2, 4),  # 2D mesh, 8 devices
        ],
        ids=[
            "1x1",
            "1x2",
            "1x8",
            "2x4",
        ],
        indirect=True,
    ),
    pytest.mark.parametrize(
        "layout,dtype",
        [
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat16),
            (ttnn.TILE_LAYOUT, ttnn.bfloat8_b),  # bfloat8_b only works with TILE_LAYOUT
            (ttnn.TILE_LAYOUT, ttnn.bfloat4_b),  # bfloat4_b only works with TILE_LAYOUT
        ],
        ids=["row_major_bf16", "tile_bf16", "tile_bf8b", "tile_bf4b"],
    ),
]


# ======================================================================================
# Helper Functions
# ======================================================================================


def _make_known_pattern(num_chunks: int) -> torch.Tensor:
    """
    Produces shape [num_chunks, 1, 3, 1] with per-chunk distinct values.
    Chunk i contains [i*1, i*2, i*3].
    """
    rows = []
    for i in range(num_chunks):
        rows.append(torch.tensor([[[i * 1.0], [i * 2.0], [i * 3.0]]]).transpose(0, 1))  # [1,3,1]
    data = torch.stack(rows, dim=0)  # [num_chunks,1,3,1]
    return data.to(torch.bfloat16)


def _make_arange_dtype(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Create a deterministic tensor with arange data and bfloat16 dtype."""
    numel = 1
    for s in shape:
        numel *= s
    data = torch.arange(numel, dtype=torch.float32).reshape(shape)
    return data.to(dtype=dtype)


def _pos_dim(dim: int, rank: int) -> int:
    """Convert possibly-negative dim to positive index for given rank."""
    return dim % rank


def _get_hw_shard_unit() -> int:
    """
    Hardware-related shard unit threshold (default 32).
    Override via env var TT_TEST_SHARD_UNIT for future hardware.
    """
    try:
        return int(os.environ.get("TT_TEST_SHARD_UNIT", "32"))
    except Exception:
        return 32


def _get_max_exp(u32_vec: list[int], is_exp_a: bool = False) -> int:
    """
    Compute the maximum exponent from a list of uint32 float32 representations.
    Matches C++ get_max_exp function.
    """
    max_exp = 0
    for u32_val in u32_vec:
        exp = (u32_val & 0x7F800000) >> 23
        if is_exp_a:
            se = int(exp) - 127 + 15
            if se > 31:
                se = 31
            elif se < 0:
                se = 0
            exp = se
        max_exp = max(max_exp, exp)
    return max_exp


# todo)) may want to refactor the quantization code into a separate file and add tests for it
def _convert_float32_to_bfloat4_b(input_val: float, shared_exp: int, is_exp_a: bool = False) -> int:
    """
    Convert a single float32 value to bfloat4_b format.
    Matches C++ convert_u32_to_bfp<Bfp4_b, false> function.

    Args:
        input_val: Input float32 value
        shared_exp: Shared exponent for the group
        is_exp_a: Whether to use exp_a rebias format

    Returns:
        uint8 value representing the bfloat4_b quantized value
    """
    # Convert float32 to uint32 representation
    u32_input = struct.unpack("I", struct.pack("f", input_val))[0]

    # Constants for bfloat4_b
    MANTISSA_BFP_WIDTH = 3
    MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH  # 21
    MANTISSA_BFP_MAX_VAL = (1 << MANTISSA_BFP_WIDTH) - 1  # 7

    # Extract sign, exponent, mantissa from float32
    mantissa = u32_input & 0x007FFFFF
    exp = (u32_input & 0x7F800000) >> 23
    sign = (u32_input & 0x80000000) >> 31

    # Check for zero or denormal
    is_zero_or_denormal = exp == 0
    if is_zero_or_denormal:
        return 0

    # Handle exp_a rebias
    if is_exp_a:
        se = int(exp) - 127 + 15
        # Check for saturation
        if se > 31:
            se = 31
            mantissa = 0x007FFFFF
        elif se < 0:
            se = 0
            mantissa = 0x0
        exp = se

    # Add hidden bit (float mantissa is 23 bits + hidden bit = 24 bits)
    mantissa = (1 << 23) | mantissa

    # Adjust mantissa if shared_exp > exp
    if shared_exp > exp:
        exp_diff = shared_exp - exp
        # Handle large shifts (undefined if shift >= bit width)
        while exp_diff > 31:
            mantissa = mantissa >> 31
            exp_diff -= 31
        mantissa = mantissa >> exp_diff

    # Round mantissa to nearest; ties round to even
    MANTISSA_ROUND_MASK = (1 << MANTISSA_BFP_SHIFT) - 1
    TIE_VALUE = 1 << (MANTISSA_BFP_SHIFT - 1)
    round_value = mantissa & MANTISSA_ROUND_MASK
    mantissa = mantissa >> MANTISSA_BFP_SHIFT
    guard_bit = mantissa & 0x1

    if round_value > TIE_VALUE or (round_value == TIE_VALUE and guard_bit == 1):
        # Round up
        mantissa += 1

    mantissa = min(mantissa, MANTISSA_BFP_MAX_VAL)

    # Add sign bit only if result is not 0
    if mantissa == 0:
        sign = 0
    mantissa = (sign << MANTISSA_BFP_WIDTH) | mantissa

    return mantissa


def _convert_bfloat4_b_to_float32(bfp4_data: int, shared_exp: int, is_exp_a: bool = False) -> float:
    """
    Convert a bfloat4_b uint8 value back to float32.
    Matches C++ convert_bfp_to_u32 function for Bfp4_b format.

    Args:
        bfp4_data: uint8 value representing bfloat4_b quantized value
        shared_exp: Shared exponent for the group
        is_exp_a: Whether to use exp_a rebias format

    Returns:
        float32 value
    """
    sign = bfp4_data >> 3
    man = bfp4_data & 0x7

    # Shift mantissa up until there is a 1 in bit 3
    shift_cnt = 0
    if man == 0:
        man = 0
        exp = 0
    else:
        while (man & 0x04) == 0:
            man = man << 1
            shift_cnt += 1
        # Shift one more time and zero the hidden top mantissa bit
        man = man << 1
        man = man & 0x7

        # Adjust exponent (C++ code asserts exp >= shift_cnt, but handle edge case)
        if shared_exp < shift_cnt:
            # Denormal case: flush to zero (matches SIMD unpacking behavior)
            return 0.0

        exp = shared_exp - shift_cnt

        # If exp_a rebias exp to 127
        if is_exp_a:
            exp = exp - 15 + 127

    # Put s, e, m together
    out_num = (sign << 31) | (exp << 23) | (man << 20)

    # Convert uint32 back to float32
    return struct.unpack("f", struct.pack("I", out_num))[0]


def _convert_row_major_to_tile_nfaces(
    tensor_flat: torch.Tensor,
    H: int,
    W: int,
    B: int,
    tile_H: int = 32,
    tile_W: int = 32,
    face_H: int = 16,
    face_W: int = 16,
) -> torch.Tensor:
    """
    Convert row-major tensor to tile_nfaces layout.
    Mirrors the layout expected by pack_as_bfp_tiles with row_major_input=False.
    """
    row_tiles = H // tile_H
    col_tiles = W // tile_W
    row_faces = tile_H // face_H
    col_faces = tile_W // face_W

    result = torch.zeros_like(tensor_flat)
    batch_size = H * W

    for b in range(B):
        batch_start = b * batch_size
        tile_row_base = batch_start

        for row_tile in range(row_tiles):
            tile_col_base = tile_row_base
            for col_tile in range(col_tiles):
                # Faces are stored sequentially per tile
                for face_h_idx in range(row_faces):
                    for face_w_idx in range(col_faces):
                        # Source in row-major
                        src_face_row0 = tile_col_base + (face_h_idx * face_H) * W + (face_w_idx * face_W)
                        # Destination in tile_nfaces
                        tile_offset = (row_tile * col_tiles + col_tile) * (tile_H * tile_W)
                        face_offset = (face_h_idx * col_faces + face_w_idx) * (face_H * face_W)
                        dst_face_row0 = batch_start + tile_offset + face_offset

                        # Copy face rows
                        for r in range(face_H):
                            src_idx = src_face_row0 + r * W
                            dst_idx = dst_face_row0 + r * face_W
                            result[dst_idx : dst_idx + face_W] = tensor_flat[src_idx : src_idx + face_W]
                tile_col_base += tile_W
            tile_row_base += tile_H * W

    return result


def _convert_tile_nfaces_to_row_major(
    tensor_flat: torch.Tensor,
    H: int,
    W: int,
    B: int,
    tile_H: int = 32,
    tile_W: int = 32,
    face_H: int = 16,
    face_W: int = 16,
) -> torch.Tensor:
    """
    Convert tile_nfaces layout back to row-major.
    Mirrors the inverse of _convert_row_major_to_tile_nfaces.
    """
    row_faces = tile_H // face_H
    col_faces = tile_W // face_W
    tile_rows = H // tile_H
    tile_cols = W // tile_W

    result = torch.zeros_like(tensor_flat)
    batch_size = H * W

    for b in range(B):
        batch_start = b * batch_size
        for tile_row in range(tile_rows):
            for tile_col in range(tile_cols):
                tile_offset = (tile_row * tile_cols + tile_col) * (tile_H * tile_W)
                for face_h_idx in range(row_faces):
                    for face_w_idx in range(col_faces):
                        face_offset = (face_h_idx * col_faces + face_w_idx) * (face_H * face_W)
                        src_face_row0 = batch_start + tile_offset + face_offset
                        dst_face_row0 = (
                            batch_start
                            + (tile_row * tile_H + face_h_idx * face_H) * W
                            + (tile_col * tile_W + face_w_idx * face_W)
                        )
                        for r in range(face_H):
                            src_idx = src_face_row0 + r * face_W
                            dst_idx = dst_face_row0 + r * W
                            result[dst_idx : dst_idx + face_W] = tensor_flat[src_idx : src_idx + face_W]
    return result


def _convert_tensor_to_bfloat4_b_and_back(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a torch tensor from float32/bfloat16 to bfloat4_b quantization and back to float32.

    Important: TT layout flattens all dims except the last into the 2D height, and uses the last dim as width.
    We mirror that exactly here, so exponent sharing groups run along the last (channel) dimension in
    rows of 16 within each tile face.

    Args:
        tensor: Input tensor (will be converted to float32)

    Returns:
        Tensor with same shape, representing the dequantized float32 values
    """
    # Convert to float32 for processing
    if tensor.dtype != torch.float32:
        tensor_fp32 = tensor.to(torch.float32)
    else:
        tensor_fp32 = tensor.clone()

    original_shape = tensor_fp32.shape

    # Ensure rank >= 1. Treat last dim as width; all preceding dims multiply into height
    t = tensor_fp32
    if t.ndim == 0:
        t = t.view(1)
    if t.ndim == 1:
        # Height=1, Width=C
        t = t.view(1, t.shape[0])

    # Normalize to (..., C)
    rank = t.ndim
    C = t.shape[-1]
    H_flat = int(torch.tensor(t.shape[:-1]).prod().item()) if rank > 1 else 1
    t2d = t.reshape(H_flat, C)

    # Tile and face sizes match hardware defaults
    TILE_H, TILE_W = 32, 32
    FACE_H, FACE_W = 16, 16

    # Pad 2D to tile boundaries: height to 32, width to 32
    H_padded = ((H_flat + TILE_H - 1) // TILE_H) * TILE_H
    W_padded = ((C + TILE_W - 1) // TILE_W) * TILE_W

    if H_padded != H_flat or W_padded != C:
        pad_H = H_padded - H_flat
        pad_W = W_padded - C
        t2d = torch.nn.functional.pad(t2d, (0, pad_W, 0, pad_H))  # pad last dim then first dim

    # Single 2D image to tilize
    B_images = 1
    tensor_flat = t2d.flatten()

    # Convert to tile_nfaces layout
    tiled = _convert_row_major_to_tile_nfaces(tensor_flat, H_padded, W_padded, B_images, TILE_H, TILE_W, FACE_H, FACE_W)

    # Process rows of 16 within each face
    result_tiled = torch.zeros_like(tiled, dtype=torch.float32)
    is_exp_a = False  # BF4_b uses non-rebiased exponent

    tile_HW = TILE_H * TILE_W
    face_HW = FACE_H * FACE_W
    row_faces = TILE_H // FACE_H
    col_faces = TILE_W // FACE_W
    batch_size = H_padded * W_padded
    num_tiles_per_batch = batch_size // tile_HW

    for b in range(B_images):
        batch_base = b * batch_size
        for tile_idx in range(num_tiles_per_batch):
            tile_base = batch_base + tile_idx * tile_HW
            for fh in range(row_faces):
                for fw in range(col_faces):
                    face_base = tile_base + (fh * col_faces + fw) * face_HW
                    for r in range(FACE_H):
                        row_start = face_base + r * FACE_W
                        group = tiled[row_start : row_start + FACE_W]
                        group_u32 = [struct.unpack("I", struct.pack("f", float(val.item())))[0] for val in group]
                        shared_exp = _get_max_exp(group_u32, is_exp_a)
                        for j, val in enumerate(group):
                            bfp4_val = _convert_float32_to_bfloat4_b(float(val.item()), shared_exp, is_exp_a)
                            deq = _convert_bfloat4_b_to_float32(bfp4_val, shared_exp, is_exp_a)
                            result_tiled[row_start + j] = deq

    # Convert back to row-major 2D
    result_rowmajor = _convert_tile_nfaces_to_row_major(
        result_tiled, H_padded, W_padded, B_images, TILE_H, TILE_W, FACE_H, FACE_W
    )

    # Reshape and unpad to 2D (H_flat, C)
    result_2d = result_rowmajor.reshape(H_padded, W_padded)[:H_flat, :C]

    # Restore the original shape (..., C)
    result_nd = result_2d.reshape(*t.shape[:-1], C)

    # Finally, reshape back to the original shape
    return result_nd.reshape(original_shape)


def _build_and_compose_sharded(
    torch_in: torch.Tensor,
    device: ttnn.MeshDevice | None,
    layout,
    ttnn_mesh_device: ttnn.MeshDevice,
    shard_dim: int,
    dtype: torch.dtype = ttnn.bfloat16,
) -> tuple[ttnn.Tensor, torch.Tensor, torch.Tensor]:
    """Build sharded tensor and compose it back to torch."""
    tt_sharded = ttnn.from_torch(
        torch_in,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=shard_dim),
    )
    torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device if device is None else None)
    torch_ref = ttnn.to_torch(tt_sharded, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_mesh_device, dim=shard_dim))
    return tt_sharded, torch_auto, torch_ref


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
def test_sharded_1d_basic(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, storage: str) -> None:
    """Basic 1D sharding auto-composition for both host and device storage."""
    num_devices = ttnn_mesh_device.get_num_devices()

    # Input tensor of shape [num_devices, 1, 3, 1]
    torch_in = _make_known_pattern(num_devices)

    # Build sharded tensor on host or device along dim=0 and compose back
    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(
        torch_in, device, layout, ttnn_mesh_device, shard_dim=0, dtype=dtype
    )

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in), "Explicit composer mismatch"
        assert torch.equal(torch_auto, torch_in), "Auto-composer mismatch"
    else:
        # For quantized dtypes, compare auto vs explicit composed results
        assert torch.equal(torch_auto, torch_ref), "Auto vs explicit composer mismatch for quantized dtype"


@pytest.mark.parametrize("storage", ["host", "device"])  # where the replicated tensor lives
def test_replicate_1d_basic(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, storage: str) -> None:
    """Replicated 1D distribution should compose to identity for host and device storage."""
    # Any shape works; replication does not change global shape
    # use float32 to use _convert_tensor_to_bfloat4_b_and_back as it converts to torch.float32
    torch_in = _make_arange_dtype((2, 3, 4, 5), dtype=torch.float32)

    device = None if storage == "host" else ttnn_mesh_device
    tt_replicated = ttnn.from_torch(
        torch_in,
        device=device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
    )

    # Auto-composition should detect full replication and yield identity
    if device is None:
        torch_auto = to_torch_auto_compose(tt_replicated, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_replicated)

    if dtype != ttnn.bfloat4_b:
        assert torch.equal(torch_auto, torch_in)
    else:
        # bfloat4_b quantizes, so compute expected quantized result
        expected = _convert_tensor_to_bfloat4_b_and_back(torch_in)
        assert torch.equal(torch_auto, expected)


# --------------------------------------------------------------------------------------
# Shard various tensor dims on 1D meshes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
def test_sharded_various_dims(ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dim: int, storage: str) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    torch_in = _make_arange_dtype(tuple(shape))

    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(torch_in, device, layout, ttnn_mesh_device, dim, dtype=dtype)

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)


# --------------------------------------------------------------------------------------
# Coverage for 2D mesh sharding: shard-shard and replicate-shard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
def test_sharded_2d_basic(
    ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, dims_pair: tuple[int, int], storage: str
) -> None:
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] <= 1 or mesh_shape[1] <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1, "Shard dims for 2D sharding must be distinct"

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    torch_in = _make_arange_dtype(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    device = None if storage == "host" else ttnn_mesh_device
    tt_sharded = ttnn.from_torch(torch_in, device=device, dtype=dtype, layout=layout, mesh_mapper=mapper)

    if device is None:
        torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_sharded)
    composer = ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    torch_ref = ttnn.to_torch(tt_sharded, mesh_composer=composer)

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)


@pytest.mark.parametrize(
    "dims_pair",
    [
        (None, -1),  # replicate along mesh dim 0, shard along last tensor dim
        (1, None),  # shard along tensor dim 1 on mesh dim 0, replicate mesh dim 1
    ],
)
@pytest.mark.parametrize("storage", ["host", "device"])  # host vs device sharded tensor
def test_sharded_2d_with_replicate(
    ttnn_mesh_device: ttnn.MeshDevice,
    layout,
    dtype,
    dims_pair: tuple[object, object],
    storage: str,
) -> None:
    # None indicates replicate axis
    replicate_axis = [i for i, d in enumerate(dims_pair) if d is None][0]
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 and torch.prod(torch.tensor(mesh_shape)).item() <= 1:
        pytest.skip("Requires a 2D mesh with at least one dim > 1 to observe replication")

    rank = 4
    # Determine which tensor axis is sharded (the non-None entry)
    shard_dim = [d for d in dims_pair if d is not None][0]
    shard_axis = _pos_dim(shard_dim, rank)
    shape = [2, 3, 4, 5]
    # Set size along sharded axis rounded up to a multiple of the other mesh dim
    other_mesh_dim = mesh_shape[1 - replicate_axis]
    shape[shard_axis] = ((shape[shard_axis] + other_mesh_dim - 1) // other_mesh_dim) * other_mesh_dim

    # use float32 to use _convert_tensor_to_bfloat4_b_and_back as it converts to torch.float32
    torch_in = _make_arange_dtype(tuple(shape), dtype=torch.float32)

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=dims_pair)  # type: ignore[arg-type]
    device = None if storage == "host" else ttnn_mesh_device
    tt_sharded = ttnn.from_torch(torch_in, device=device, dtype=dtype, layout=layout, mesh_mapper=mapper)

    if device is None:
        torch_auto = to_torch_auto_compose(tt_sharded, device=ttnn_mesh_device)
    else:
        torch_auto = to_torch_auto_compose(tt_sharded)

    if dtype != ttnn.bfloat4_b:
        assert torch.equal(torch_auto, torch_in)
    else:
        # bfloat4_b quantizes, so compute expected quantized result
        expected = _convert_tensor_to_bfloat4_b_and_back(torch_in)
        assert torch.equal(torch_auto, expected)


# --------------------------------------------------------------------------------------
# Tensor shape categories around hardware threshold (e.g., 32)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["lt", "eq", "gt"])  # per-shard length relative to threshold
@pytest.mark.parametrize("storage", ["host", "device"])  # where the sharded tensor lives
def test_sharded_shape_thresholds(
    ttnn_mesh_device: ttnn.MeshDevice, layout, dtype, category: str, storage: str
) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    unit = _get_hw_shard_unit()
    if category == "lt":
        per_shard = max(1, unit - 1)
    elif category == "eq":
        per_shard = unit
    else:
        per_shard = unit + 1

    shard_dim = -1  # test last dimension as sharded axis (rank=4)
    rank = 4
    axis = _pos_dim(shard_dim, rank)
    # Global size across sharded dim = per_shard_len * num_devices
    shape = [2, 3, 4, 5]
    shape[axis] = per_shard * num_devices
    torch_in = _make_arange_dtype(tuple(shape))

    device = None if storage == "host" else ttnn_mesh_device
    _, torch_auto, torch_ref = _build_and_compose_sharded(
        torch_in, device, layout, ttnn_mesh_device, shard_dim, dtype=dtype
    )

    if dtype == ttnn.bfloat16:
        assert torch.equal(torch_ref, torch_in)
        assert torch.equal(torch_auto, torch_in)
    else:
        assert torch.equal(torch_auto, torch_ref)
