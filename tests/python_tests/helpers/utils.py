# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .format_config import DataFormat, FormatConfig
from .llk_params import format_dict
from .logger import logger
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
)
from .tile_shape import construct_tile_shape

torch.set_printoptions(linewidth=500, sci_mode=False, precision=2, threshold=10000)

Tolerance = namedtuple("Tolerance", ["atol", "rtol"])
tolerances = {
    DataFormat.Float16: Tolerance(atol=0.05, rtol=0.05),
    DataFormat.Float16_b: Tolerance(atol=0.05, rtol=0.05),
    DataFormat.Float32: Tolerance(atol=0.05, rtol=0.05),
    DataFormat.Int32: Tolerance(atol=0, rtol=0),
    DataFormat.UInt32: Tolerance(atol=0, rtol=0),
    DataFormat.Int16: Tolerance(atol=0, rtol=0),
    DataFormat.UInt16: Tolerance(atol=0, rtol=0),
    DataFormat.Int8: Tolerance(atol=0, rtol=0),
    DataFormat.UInt8: Tolerance(atol=0, rtol=0),
    DataFormat.Bfp8_b: Tolerance(atol=0.1, rtol=0.2),
    DataFormat.Bfp4_b: Tolerance(atol=0.25, rtol=0.3),
    DataFormat.MxFp8R: Tolerance(atol=0.2, rtol=0.3),
    DataFormat.MxFp8P: Tolerance(atol=0.2, rtol=0.3),
    DataFormat.Fp8_e4m3: Tolerance(atol=0.2, rtol=0.2),
}


def print_faces(operand1, tile_shape=None):
    if tile_shape is None:
        tile_shape = construct_tile_shape((DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM))

    face_size = tile_shape.face_r_dim * tile_shape.face_c_dim
    num_faces = tile_shape.total_num_faces()

    # Extract faces
    faces = []
    for face_idx in range(num_faces):
        start = face_idx * face_size
        end = start + face_size
        faces.append(
            operand1[start:end].view(tile_shape.face_r_dim, tile_shape.face_c_dim)
        )

    lines = []
    # Print faces row by row
    faces_per_row = tile_shape.num_faces_c_dim
    num_face_rows = tile_shape.num_faces_r_dim

    for face_row in range(num_face_rows):
        # Print all rows within this face row
        for i in range(tile_shape.face_r_dim):
            row_parts = []
            for face_col in range(faces_per_row):
                face_idx = face_row * faces_per_row + face_col
                if face_idx < num_faces:
                    row_parts.append(
                        " ".join(f"{x:6.2f}" for x in faces[face_idx][i].tolist())
                    )
            lines.append("  |  ".join(row_parts))

        # Add separator between face rows
        if face_row < num_face_rows - 1:
            lines.append(
                "-"
                * (faces_per_row * tile_shape.face_c_dim * 7 + (faces_per_row - 1) * 5)
            )

    logger.debug("Tile faces:\n{}", "\n".join(lines))


def run_shell_command(
    command: str, cwd: str | None = None, stdin_data: str | bytes = None, text=True
):
    result = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=text,
        input=stdin_data,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command:\n{command}\n\nCommand's stderr:\n{result.stderr}")
    return result


def calculate_read_byte_count(format: FormatConfig, array_size: int, sfpu=False) -> int:
    total_bytes = array_size * format.output_format.size
    if format.output_format == DataFormat.Bfp8_b:
        total_bytes += total_bytes // 16
    return total_bytes


def reverse_endian_chunk(input_list, chunk_size=4):
    output_list = []
    for j in range(0, len(input_list), chunk_size):
        output_list.extend(input_list[j : j + chunk_size][::-1])
    return output_list


def format_kernel_list(kernels, as_hex=False):
    formatter = hex if as_hex else str
    return ",".join(formatter(i) for i in kernels)


def calculate_pcc(golden, input):
    """Calculate Pearson Correlation Coefficient between two tensors.

    Handles special cases like NaN/Inf values and returns correlation coefficient.
    1.0 indicates perfect correlation, 0.0 indicates no correlation.

    Args:
        golden: Reference tensor
        input: Tensor to compare against golden

    Returns:
        float: Pearson correlation coefficient
    """
    golden = torch.as_tensor(golden)
    input = torch.as_tensor(input)

    if golden.dtype != input.dtype:
        input = input.type(golden.dtype)

    # Handle special cases
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(input)):
        return 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(input)):
        return 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(input.bool()):
        return 0.0

    # Handle exact equality case
    if torch.equal(golden, input):
        return 1.0

    # Mask invalid values (nan/inf)
    def mask_invalid(x):
        x = x.clone()
        mask = torch.logical_or(
            torch.isnan(x), torch.logical_or(torch.isinf(x), torch.isneginf(x))
        )
        x[mask] = 0
        return x

    golden = mask_invalid(golden)
    input = mask_invalid(input)

    # Convert bfloat16 to float32 for correlation calculation
    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        input = input.type(torch.float32)

    # Calculate correlation
    pcc_result = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(input).detach().numpy()).flatten(),
        )
    )

    if isinstance(pcc_result, np.ma.core.MaskedConstant):
        return 1.0

    return pcc_result


def _bfp4_block_aware_compare(
    golden: torch.Tensor, result: torch.Tensor, max_ulp_diff: int = 2
) -> torch.Tensor:
    """Compare two BFP4_b tensors allowing small ULP differences per 16-element block.

    BFP4_b shares an exponent across each 16-element block, so the ULP size
    depends on the block's max magnitude.  The SFPU hardware computes in FP32
    with internal approximations (e.g. Newton-Raphson for rsqrt) that can
    differ slightly from the golden model.  After Bfp4 quantization (3-bit
    mantissa), these small intermediate differences can shift a value by up
    to ``max_ulp_diff`` quantization steps.

    For full tiles (n a multiple of 1024), we tilize before block-wise check and
    untilize the validity mask. For partial tiles (num_faces 1 or 2: 256/512
    elements), we compare block-by-block on flat data without tilize/untilize.
    """
    from helpers.tilize_untilize import tilize_block, untilize_block

    BLOCK = 16
    BFP4_MANTISSA_BITS = 3
    TILE_SIZE = 1024

    g_flat = golden.float().flatten()
    r_flat = result.float().flatten()
    n = g_flat.numel()

    if n % TILE_SIZE == 0 and n > 0:
        num_tiles = n // TILE_SIZE
        tile_dim = (32 * num_tiles, 32)
        g_til = tilize_block(g_flat, tile_dim, DataFormat.Float32).flatten()
        r_til = tilize_block(r_flat, tile_dim, DataFormat.Float32).flatten()
    else:
        g_til = g_flat
        r_til = r_flat

    is_valid_til = torch.ones(n, dtype=torch.bool)

    for blk_start in range(0, n, BLOCK):
        blk_end = min(blk_start + BLOCK, n)
        g_blk = g_til[blk_start:blk_end]
        r_blk = r_til[blk_start:blk_end]

        both_nan = torch.isnan(g_blk) & torch.isnan(r_blk)

        finite_vals = torch.cat(
            [
                g_blk[torch.isfinite(g_blk)].abs(),
                r_blk[torch.isfinite(r_blk)].abs(),
            ]
        )
        if finite_vals.numel() == 0:
            is_valid_til[blk_start:blk_end] = True
            continue

        block_max = finite_vals.max().item()
        if block_max == 0:
            is_valid_til[blk_start:blk_end] = (g_blk == r_blk) | both_nan
            continue

        block_exp = math.floor(math.log2(block_max))
        one_ulp = 2.0 ** (block_exp - BFP4_MANTISSA_BITS + 1)

        diff = (g_blk - r_blk).abs()
        is_valid_til[blk_start:blk_end] = (diff <= max_ulp_diff * one_ulp) | both_nan

    if n % TILE_SIZE == 0 and n > 0:
        num_tiles = n // TILE_SIZE
        tile_dim = (32 * num_tiles, 32)
        is_valid = (
            untilize_block(is_valid_til.float(), DataFormat.Float32, tile_dim)
            .flatten()
            .bool()
        )
    else:
        is_valid = is_valid_til

    return is_valid


def passed_test(
    golden_tensor,
    res_tensor,
    output_data_format: DataFormat = DataFormat.Float16_b,
    L1_to_L1_iterations: int = 1,
    print_errors: bool = True,
    print_pcc: bool = False,
    custom_atol=None,
    custom_rtol=None,
    custom_pcc_threshold=None,
    tile_shape=None,
):

    if tile_shape is None:
        tile_shape = construct_tile_shape((DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM))

    def get_tolerance(output_data_format):
        try:
            return tolerances[output_data_format]
        except KeyError:
            raise ValueError(f"Unsupported output data format: {output_data_format}")

    tolerance = get_tolerance(output_data_format)

    if custom_atol is not None and custom_atol >= 0:
        logger.info("Overriding atol from {} to {}", tolerance.atol, custom_atol)
        tolerance = tolerance._replace(atol=custom_atol)

    if custom_rtol is not None and custom_rtol >= 0:
        logger.info("Overriding rtol from {} to {}", tolerance.rtol, custom_rtol)
        tolerance = tolerance._replace(rtol=custom_rtol)

    golden_tensor = golden_tensor.type(format_dict[output_data_format])
    res_tensor = res_tensor.type(format_dict[output_data_format])

    if output_data_format == DataFormat.Bfp4_b:
        is_valid = _bfp4_block_aware_compare(golden_tensor, res_tensor)
    else:
        is_close = torch.isclose(
            golden_tensor, res_tensor, rtol=tolerance.rtol, atol=tolerance.atol
        )
        is_nan = torch.isnan(golden_tensor) & torch.isnan(res_tensor)
        is_valid = is_close | is_nan

    is_within_tolerance = torch.all(is_valid)

    if print_errors:
        try:
            if not is_within_tolerance:
                diff_indices = torch.where(~is_valid)[0]
                num_tiles = (res_tensor.size()[0]) // (tile_shape.total_tile_size())
                tile_shape_for_torch = (
                    tile_shape.total_row_dim(),
                    tile_shape.total_col_dim(),
                )

                def bg(r, g, b):
                    return f"\033[48;2;{r};{g};{b}m"

                BLUE = bg(0, 0, 100)
                RED = bg(160, 0, 0)
                PURPLE = bg(50, 0, 50)

                RESET = "\033[0m"

                def format_tile(
                    tile_data, error_tile, tile_no, golden: bool = False
                ) -> list[str]:
                    if torch.all(error_tile == 0):
                        return []

                    label = "Golden tile" if golden else "Result tile"
                    background = PURPLE if golden else BLUE
                    tile_lines = [f"Row\t === {label} {tile_no+1} ==="]
                    for row in range(tile_shape.total_row_dim()):
                        row_values = []
                        for col in range(tile_shape.total_col_dim()):
                            colour = RED if error_tile[row, col] else background
                            row_values.append(
                                f"{colour}{tile_data[row, col]:7.2f}{RESET}{' ' if col == tile_shape.face_c_dim - 1 else '' }"
                            )
                        tile_lines.append(f"{(row+1):02d}. {''.join(row_values)}")
                    return tile_lines

                formatted_error = []

                for tile_no in range(num_tiles):
                    result_tile = res_tensor[
                        tile_no
                        * tile_shape.total_tile_size() : (tile_no + 1)
                        * tile_shape.total_tile_size()
                    ].view(tile_shape_for_torch)
                    golden_tile = golden_tensor[
                        tile_no
                        * tile_shape.total_tile_size() : (tile_no + 1)
                        * tile_shape.total_tile_size()
                    ].view(tile_shape_for_torch)
                    error_tile = ~is_valid[
                        tile_no
                        * tile_shape.total_tile_size() : (tile_no + 1)
                        * tile_shape.total_tile_size()
                    ].view(tile_shape_for_torch)

                    lines = format_tile(result_tile, error_tile, tile_no)
                    if not lines:
                        continue

                    formatted_error.extend(lines)
                    formatted_error.append("")
                    formatted_error.extend(
                        format_tile(golden_tile, error_tile, tile_no, True)
                    )

                logger.error("\n{}", "\n".join(formatted_error))

        except RuntimeError:
            logger.error(
                "Could not reshape to {}x{} matrix, showing linear indices: {}",
                tile_shape.total_row_dim(),
                tile_shape.total_col_dim(),
                res_tensor.size()[0],
            )
            for idx in diff_indices[:10]:
                logger.error(
                    "Failed at index {} with result={}, golden={}",
                    idx,
                    res_tensor[idx],
                    golden_tensor[idx],
                )

    pcc = calculate_pcc(res_tensor, golden_tensor)

    if print_pcc:
        logger.info("PCC: {:.6f} | format={}", pcc, output_data_format.name)
    else:
        logger.debug("PCC: {:.6f} | format={}", pcc, output_data_format.name)

    target_pcc = 0.99
    # Once we iterate L1-L1 more than once the loss in precision is accumulated because the result from the first run is transferred as input to the next run
    # We don't have a robust accuracy model to determine exact precision loss from each run and accumulate as such per test, so we use a heuristic
    #   - This reduction in precision occurs primarily when copying results from the first L1-to-L1 stage, and is further compounded when truncating
    #     values with less precision (Bfp8_b) and drops below 99% in that case
    if output_data_format == DataFormat.Bfp8_b:
        target_pcc = pow(0.99, L1_to_L1_iterations)
    elif output_data_format == DataFormat.Bfp4_b:
        target_pcc = 0.98

    if custom_pcc_threshold is not None:
        logger.info(
            "Overriding PCC threshold from {} to {}", target_pcc, custom_pcc_threshold
        )
        target_pcc = custom_pcc_threshold

    return is_within_tolerance and (pcc > target_pcc)


def create_directories(dirs: list[Path]):
    """Create directories with file lock to handle race conditions in parallel execution."""

    # If all directories exist, skip locking entirely
    if all(dir.exists() for dir in dirs):
        return

    # Acquire lock and create using os.makedirs (more robust than pathlib.mkdir)
    lock = FileLock("/tmp/tt-llk-build.lock")
    with lock:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
