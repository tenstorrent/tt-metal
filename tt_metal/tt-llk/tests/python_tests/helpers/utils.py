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
    DataFormat.Bfp4_b: Tolerance(atol=0.125, rtol=0.3),
    DataFormat.Bfp2_b: Tolerance(atol=0.5, rtol=0.4),
    DataFormat.MxFp8R: Tolerance(atol=0.2, rtol=0.3),
    DataFormat.MxFp8P: Tolerance(atol=0.2, rtol=0.3),
    DataFormat.MxFp4: Tolerance(atol=0.5, rtol=0.35),
    DataFormat.MxInt8: Tolerance(atol=0.05, rtol=0.05),
    DataFormat.MxInt4: Tolerance(atol=0.5, rtol=0.35),
    DataFormat.MxInt2: Tolerance(atol=1.0, rtol=0.5),
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
    total = array_size * format.output_format.size
    assert total.denominator == 1, (
        f"array_size={array_size} not aligned to {format.output_format} "
        f"element size ({format.output_format.size}); would drop a partial byte"
    )
    total_bytes = total.numerator
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


def _bfp_block_aware_compare(
    golden: torch.Tensor,
    result: torch.Tensor,
    mantissa_bits: int,
    max_ulp_diff: int = 1,
) -> torch.Tensor:
    """Compare two block-float tensors allowing small ULP differences per 16-element block.

    BFP formats share an exponent across each 16-element block, so the ULP size
    depends on the block's max magnitude.  Hardware and the Python golden can
    disagree at BFP truncation boundaries (e.g. 0.0 vs 0.5 printed as bf16).
    After quantization, treat differences up to ``max_ulp_diff`` steps as acceptable.

    ``mantissa_bits`` is the number of mantissa bits the format reconstructs
    per element (3 for Bfp4_b, 1 for Bfp2_b).

    Golden and result must already be in the same flat buffer order (tilized
    layout as produced by the tests).  We do not re-tilize here: inputs are
    already ordered in 16-element BFP blocks for the device.
    """
    BLOCK = 16

    g_flat = golden.float().flatten()
    r_flat = result.float().flatten()
    n = g_flat.numel()

    is_valid = torch.ones(n, dtype=torch.bool)

    for blk_start in range(0, n, BLOCK):
        blk_end = min(blk_start + BLOCK, n)
        g_blk = g_flat[blk_start:blk_end]
        r_blk = r_flat[blk_start:blk_end]

        both_nan = torch.isnan(g_blk) & torch.isnan(r_blk)

        finite_vals = torch.cat(
            [
                g_blk[torch.isfinite(g_blk)].abs(),
                r_blk[torch.isfinite(r_blk)].abs(),
            ]
        )
        if finite_vals.numel() == 0:
            is_valid[blk_start:blk_end] = True
            continue

        block_max = finite_vals.max().item()
        if block_max == 0:
            is_valid[blk_start:blk_end] = (
                torch.isclose(g_blk, r_blk, atol=1e-5, rtol=0.0, equal_nan=True)
                | both_nan
            )
            continue

        block_exp = math.floor(math.log2(block_max))
        one_ulp = 2.0 ** (block_exp - mantissa_bits + 1)

        diff = (g_blk - r_blk).abs()
        ulp_ok = diff <= max_ulp_diff * one_ulp
        # Padding / zero lanes can disagree slightly after pack-unpack while still
        # printing as 0.00; ULP sizing from a large value elsewhere in the block
        # can make those tiny residuals look like multi-ULP failures. Accept when
        # both sides are negligible magnitude and close in absolute terms.
        max_abs = torch.maximum(g_blk.abs(), r_blk.abs())
        tiny_ok = (max_abs < 1e-4) & torch.isclose(
            g_blk, r_blk, atol=1e-5, rtol=0.0, equal_nan=True
        )
        is_valid[blk_start:blk_end] = ulp_ok | both_nan | tiny_ok

    return is_valid


def _mxint2_block_aware_compare(
    golden: torch.Tensor, result: torch.Tensor
) -> torch.Tensor:
    """Compare two MxInt2 tensors allowing single-lattice-step disagreements.

    MxInt2 represents each element as raw -1/0/+1 times the block's
    2^(scale-127). The per-block lattice has only three points, so any
    intermediate-precision divergence between HW's dest->pack path and the
    golden's fp32 model can flip a near-midpoint element by exactly one lattice
    step (e.g. golden picks +scale, HW picks 0). The two are equally valid
    quantizations of a true value sitting on the midpoint; what looks like a
    failure is the coarse lattice amplifying sub-ULP precision noise.

    Acceptance rule (per 32-element block): a position is valid iff
        |g[i] - r[i]| <= block_scale_max
    where ``block_scale_max = max(|g|, |r|)`` over the block (this equals the
    block's lattice spacing whenever the block has any non-zero element).
    Larger differences (sign flips, 2-step jumps) still fail.

    Tilizes first to match HW's block layout (32-element block = one face
    row-pair), so the comparison is done in the same coordinate system HW
    used when deriving block scales.
    """
    from helpers.tilize_untilize import tilize_block, untilize_block

    BLOCK = 32
    TILE_SIZE = 1024

    g_flat = golden.float().flatten()
    r_flat = result.float().flatten()
    n = g_flat.numel()

    if n == 0:
        return torch.ones(0, dtype=torch.bool)

    if n % TILE_SIZE == 0:
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
        diff = (g_blk - r_blk).abs()

        block_scale = torch.max(
            g_blk.abs().nan_to_num(nan=0.0).max(),
            r_blk.abs().nan_to_num(nan=0.0).max(),
        ).item()

        if block_scale == 0.0:
            is_valid_til[blk_start:blk_end] = (g_blk == r_blk) | both_nan
        else:
            # Small fp slack on the bound; the legitimate disagreements we
            # observed sit at diff == block_scale (e.g. golden=+2 vs HW=0 at
            # scaled=0.5 midpoint), well inside this margin.
            is_valid_til[blk_start:blk_end] = (diff <= block_scale + 1e-6) | both_nan

    if n % TILE_SIZE == 0:
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


_RECORD_TEST_ORDER: bool = False

# Per-format params for _mxfp_block_aware_compare: (mantissa_bits, max_steps).
#   mantissa_bits of the SxEyMz element -> local step = 2^(floor(log2|v|) - mantissa_bits).
#   max_steps = accepted adjacent-representable steps (same role as MxInt's
#     max_ulp_steps). HW flushes subnormals to 0, so the smallest representable
#     magnitude is the min normal and the a==0 branch handles flushed values.
_MXFP_COMPARE_PARAMS = {
    DataFormat.MxFp4: (1, 2),  # E2M1
    DataFormat.MxFp8R: (2, 2),  # E5M2
    DataFormat.MxFp8P: (3, 2),  # E4M3
}


def _mxfp_block_aware_compare(
    golden: torch.Tensor,
    result: torch.Tensor,
    mantissa_bits: int,
    max_steps: int = 2,
) -> torch.Tensor:
    """Compare two MX-float tensors allowing small representable-adjacency diffs.

    Unlike the MxInt formats (uniform integer lattice) and BFP4 (one shared
    exponent per block -> uniform ULP), MX-float elements (MxFp4 E2M1, MxFp8R
    E5M2, MxFp8P E4M3) carry their own exponent on top of the block's E8M0
    scale, so the representable lattice is non-uniform. For an element format
    with `mantissa_bits` mantissa bits the local step at a value v is exactly
    2^(floor(log2|v|) - mantissa_bits) -- derivable per element from v's own
    magnitude, no block scale needed. A position is valid iff |g-r| is within
    `max_steps` such local steps (golden and HW within `max_steps` adjacent
    representable values). Sign flips and larger jumps still fail.

    HW flushes subnormals to 0, so a flushed value is 0 on both sides and is
    caught by the a==0 branch (exact match) -- no separate subnormal handling.
    """
    g = golden.float().flatten()
    r = result.float().flatten()
    n = g.numel()
    if n == 0:
        return torch.ones(0, dtype=torch.bool)

    both_nan = torch.isnan(g) & torch.isnan(r)

    # Per-element local lattice step from the larger magnitude.
    a = torch.nan_to_num(
        torch.maximum(g.abs(), r.abs()), nan=0.0, posinf=0.0, neginf=0.0
    )
    safe = a > 0
    exp = torch.zeros_like(a)
    exp[safe] = torch.floor(torch.log2(a[safe]))
    local_ulp = torch.where(
        safe, torch.pow(2.0, exp - mantissa_bits), torch.zeros_like(a)
    )

    diff = (g - r).abs()
    # Relative float32-rounding guard (~1 ULP at the comparison magnitude) instead of a
    # fixed absolute slack. A constant would dominate `max_steps * local_ulp` for small
    # magnitudes (tiny block scales) and let sign flips / multi-step jumps pass.
    is_valid = torch.where(
        safe, diff <= max_steps * local_ulp + torch.finfo(torch.float32).eps * a, g == r
    )
    return is_valid | both_nan


def passed_test(
    golden_tensor,
    res_tensor,
    output_data_format: DataFormat = DataFormat.Float16_b,
    L1_to_L1_iterations: int = 1,
    custom_bfp4_max_ulp_diff=None,
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
        logger.debug("Overriding atol from {} to {}", tolerance.atol, custom_atol)
        tolerance = tolerance._replace(atol=custom_atol)

    if custom_rtol is not None and custom_rtol >= 0:
        logger.debug("Overriding rtol from {} to {}", tolerance.rtol, custom_rtol)
        tolerance = tolerance._replace(rtol=custom_rtol)

    golden_tensor = golden_tensor.type(format_dict[output_data_format])
    res_tensor = res_tensor.type(format_dict[output_data_format])

    if output_data_format == DataFormat.Bfp4_b:
        ulp = custom_bfp4_max_ulp_diff if custom_bfp4_max_ulp_diff is not None else 1
        is_valid = _bfp_block_aware_compare(
            golden_tensor, res_tensor, mantissa_bits=3, max_ulp_diff=ulp
        )
    elif output_data_format == DataFormat.Bfp2_b:
        is_valid = _bfp_block_aware_compare(
            golden_tensor, res_tensor, mantissa_bits=1, max_ulp_diff=1
        )
    elif output_data_format.is_mx_fp_format():
        # Non-uniform float lattice (E2M1 / E5M2 / E4M3): per-element adjacency
        # check instead of a fixed ULP. Replaces the loose torch.isclose +
        # count fallback; per-format (mantissa_bits, max_steps) in
        # _MXFP_COMPARE_PARAMS.
        mantissa_bits, max_steps = _MXFP_COMPARE_PARAMS[output_data_format]
        is_valid = _mxfp_block_aware_compare(
            golden_tensor,
            res_tensor,
            mantissa_bits=mantissa_bits,
            max_steps=max_steps,
        )
    else:
        is_close = torch.isclose(
            golden_tensor, res_tensor, rtol=tolerance.rtol, atol=tolerance.atol
        )
        is_nan = torch.isnan(golden_tensor) & torch.isnan(res_tensor)
        is_valid = is_close | is_nan

    is_within_tolerance = torch.all(is_valid)

    if output_data_format.is_mx_format():
        # Every MX low-bit format is judged by its lattice-aware compare
        # (MxFp* via _mxfp_block_aware_compare on the E2M1/E5M2/E4M3 float
        # lattices), which accepts disagreements up to a few lattice steps. At
        # power-of-2 block-max boundaries the golden (fp32 amax) and HW (lower-
        # precision amax) can pick block exponents one spec-legal step apart
        # (OCP MX: scale = largest pow2 <= amax) — The per-element lattice
        # check is the principled correctness criterion here, so trust its
        # verdict rather than re-gating on PCC (sign flips and gross multi-step
        # jumps still fail the lattice-aware check).
        return bool(is_within_tolerance)

    if print_errors and not _RECORD_TEST_ORDER:
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
        target_pcc = 0.97
    elif output_data_format == DataFormat.Bfp2_b:
        target_pcc = (
            0.90  # Bfp2_b has only 1 mantissa bit; precision is severely limited
        )

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
