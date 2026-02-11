# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .format_config import DataFormat, FormatConfig
from .llk_params import format_dict

torch.set_printoptions(linewidth=500, sci_mode=False, precision=2, threshold=10000)


def print_faces(operand1):
    f0 = operand1[:256].view(16, 16)
    f1 = operand1[256:512].view(16, 16)
    f2 = operand1[512:768].view(16, 16)
    f3 = operand1[768:].view(16, 16)

    # Print the first set with proper alignment
    for i in range(16):
        print(
            " ".join(f"{x:6.2f}" for x in f0[i].tolist()),
            " | ",
            " ".join(f"{x:6.2f}" for x in f1[i].tolist()),
        )

    print("-" * 250)

    # Print the second set with proper alignment
    for i in range(16):
        print(
            " ".join(f"{x:6.2f}" for x in f2[i].tolist()),
            " | ",
            " ".join(f"{x:6.2f}" for x in f3[i].tolist()),
        )

    print("\n" * 3)


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


def passed_test(
    golden_tensor,
    res_tensor,
    output_data_format: DataFormat = DataFormat.Float16_b,
    L1_to_L1_iterations: int = 1,
    print_erros: bool = True,
    print_pcc: bool = False,
):
    Tolerance = namedtuple("Tolerance", ["atol", "rtol"])

    def get_tolerance(output_data_format):
        tolerances = {
            DataFormat.Float16: Tolerance(atol=0.05, rtol=0.05),
            DataFormat.Float16_b: Tolerance(atol=0.05, rtol=0.05),
            DataFormat.Float32: Tolerance(atol=0.05, rtol=0.05),
            DataFormat.Int32: Tolerance(atol=0, rtol=0),
            DataFormat.UInt32: Tolerance(atol=0, rtol=0),
            DataFormat.UInt16: Tolerance(atol=0, rtol=0),
            DataFormat.Int8: Tolerance(atol=0, rtol=0),
            DataFormat.UInt8: Tolerance(atol=0, rtol=0),
            DataFormat.Bfp8_b: Tolerance(atol=0.1, rtol=0.2),
            DataFormat.MxFp8R: Tolerance(atol=0.2, rtol=0.3),
            DataFormat.MxFp8P: Tolerance(atol=0.2, rtol=0.3),
        }

        try:
            return tolerances[output_data_format]
        except KeyError:
            raise ValueError(f"Unsupported output data format: {output_data_format}")

    tolerance = get_tolerance(output_data_format)

    golden_tensor = golden_tensor.type(format_dict[output_data_format])
    res_tensor = res_tensor.type(format_dict[output_data_format])

    is_close = torch.isclose(
        golden_tensor, res_tensor, rtol=tolerance.rtol, atol=tolerance.atol
    )
    is_nan = torch.isnan(golden_tensor) & torch.isnan(res_tensor)

    is_valid = is_close | is_nan
    is_within_tolerance = torch.all(is_valid)

    if print_erros:
        try:
            if not is_within_tolerance:
                diff_indices = torch.where(~is_valid)[0]
                num_tiles = (res_tensor.size()[0]) // 1024
                tile_shape = (32, 32)

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
                    for row in range(32):
                        row_values = []
                        for col in range(32):
                            colour = RED if error_tile[row, col] else background
                            row_values.append(
                                f"{colour}{tile_data[row, col]:7.2f}{RESET}{' ' if col == 15 else '' }"
                            )

                        tile_lines.append(f"{(row+1):02d}. {''.join(row_values)}")

                        if row == 15:
                            tile_lines.append("")
                    return tile_lines

                formatted_error = []

                for tile_no in range(num_tiles):
                    result_tile = res_tensor[
                        tile_no * 1024 : (tile_no + 1) * 1024
                    ].view(tile_shape)
                    golden_tile = golden_tensor[
                        tile_no * 1024 : (tile_no + 1) * 1024
                    ].view(tile_shape)
                    error_tile = ~is_valid[tile_no * 1024 : (tile_no + 1) * 1024].view(
                        tile_shape
                    )

                    lines = format_tile(result_tile, error_tile, tile_no)
                    if not lines:
                        continue

                    formatted_error.extend(lines)
                    formatted_error.append("")
                    formatted_error.extend(
                        format_tile(golden_tile, error_tile, tile_no, True)
                    )

                print("\n".join(formatted_error), file=sys.stderr)

        except RuntimeError:
            print(
                f"Could not reshape to 32x32 matrix, showing linear indices: {res_tensor.size()[0]}"
            )
            for idx in diff_indices[:10]:
                print(
                    f"Failed at index {idx} with result={res_tensor[idx]}, golden={golden_tensor[idx]}"
                )

    pcc = calculate_pcc(res_tensor, golden_tensor)

    if print_pcc:
        print("PCC:", pcc)

    target_pcc = 0.99
    # Once we iterate L1-L1 more than once the loss in precision is accumulated because the result from the first run is transferred as input to the next run
    # We don't have a robust accuracy model to determine exact precision loss from each run and accumulate as such per test, so we use a heuristic
    #   - This reduction in precision occurs primarily when copying results from the first L1-to-L1 stage, and is further compounded when truncating
    #     values with less precision (Bfp8_b) and drops below 99% in that case
    if output_data_format == DataFormat.Bfp8_b:
        target_pcc = pow(0.99, L1_to_L1_iterations)
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
