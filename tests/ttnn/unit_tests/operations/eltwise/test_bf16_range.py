# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
import numpy as np
import time
import pandas as pd
import sys
import os
import traceback
import scipy
import models.utility_functions as util
import matplotlib.pyplot as plt
import random


device_id = 0
device = ttnn.open_device(device_id=device_id)

EPSILON = 2**-9

# ttnn.enable_program_cache(device)  # Useful: we are going to call the same kernel several times


def generate_bfloat16_scalars(num_of_scalar=5):
    raw_uint16_array = np.array(
        random.sample(range(0, 2**16), num_of_scalar), dtype=np.uint16
    )  # Generate randim bf16 values
    raw_int16_array = raw_uint16_array.view(np.int16)
    torch_int16_tensor = torch.from_numpy(raw_int16_array)
    bfloat16_tensor = torch_int16_tensor.view(torch.bfloat16)
    float_list = bfloat16_tensor.tolist()

    # Add special values to the list
    float_list += [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
    ]

    return float_list


datatypes_parameters = {
    "float32": {
        "numpy_type": np.float32,
        "torch_type": torch.float32,
        "ttnn_type": ttnn.float32,
        "numpy_int_type": np.int32,
        "torch_int_type": torch.int32,
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 23,
        "tensor_width": 2**12,
        "tensor_height": 2**11,
    },
    "bfloat16": {
        "numpy_type": np.float32,  # Note: a conversion will be needed
        "torch_type": torch.bfloat16,
        "ttnn_type": ttnn.bfloat16,
        "numpy_int_type": np.int16,
        "torch_int_type": torch.int16,
        "sign_bits": 1,
        "exponent_bits": 8,
        "mantissa_bits": 7,
        "tensor_width": 2**4,  # Not great (< 32) => tiles will have padding
        "tensor_height": 2**3,
    },
}

operations_dict = {
    "maximum": (
        torch.maximum,
        ttnn.maximum,
        None,
        generate_bfloat16_scalars(3),
        "maximum",
    ),
    "minimum": (
        torch.minimum,
        ttnn.minimum,
        None,
        generate_bfloat16_scalars(),
        "minimum",
    ),
    "deg2rad": (
        torch.deg2rad,
        ttnn.deg2rad,
        None,
        None,
        "deg2rad",
    ),
    "exp": (
        torch.exp,
        ttnn.exp,
        None,
        None,
        "exp",
    ),
    "expm1": (
        torch.special.expm1,
        ttnn.expm1,
        None,
        None,
        "expm1",
    ),
    "elu": (
        torch.nn.functional.elu,
        ttnn.elu,
        None,
        [0.5, 1.0],
        "elu",
    ),
    "selu": (
        torch.nn.functional.selu,
        ttnn.selu,
        None,
        None,  # Default values to be used for now
        "selu",
    ),
}


def measure_op_accuracy_bf16_eq_only(operation_name, dest_dir):
    parameters = datatypes_parameters["bfloat16"]

    # Create 2^9 x 2^7 tensor (2^16 elements total)
    TENSOR_WIDTH = 2**7
    TENSOR_HEIGHT = 2**9
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    SIGN_BITS = parameters["sign_bits"]  # should be 1
    EXPONENT_BITS = parameters["exponent_bits"]
    MANTISSA_BITS = parameters["mantissa_bits"]

    NUMPY_TYPE = parameters["numpy_type"]
    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TORCH_INT_TYPE = parameters["torch_int_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Create input tensors
    input_np = np.arange(0, 2**16, dtype=NUMPY_INT_TYPE)  # All possible bfloat16 values
    torch_value = torch.from_numpy(input_np).reshape(size)
    torch_input_bf16 = torch_value.view(TORCH_TYPE)  # reinterpret data as bfloat16
    # torch_input_bf16[torch.isnan(torch_input_bf16)] = 0.0 # replace nan to 0.0

    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    (torch_op, ttnn_op, _, scalar_list, parent_op) = operations_dict[operation_name]

    match_percentages = []
    scalar_labels = []

    scalar_list = scalar_list if scalar_list is not None else [None]
    for scalar in scalar_list:
        print(f"  Testing with scalar = {scalar}")
        scalar_tensor = torch.full(size, scalar, dtype=TORCH_TYPE)

        torch_output_ref = torch_op(torch_input_bf16, scalar_tensor)

        ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

        def launch_ttnn_op(x):
            ttnn_value = ttnn.from_torch(x, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
            return ttnn.to_torch(ttnn_op(ttnn_value, scalar, output_tensor=ttnn_output))

        torch_output_actual = launch_ttnn_op(torch_input_bf16)

        match_mask = torch.eq(torch_output_ref, torch_output_actual) | (
            torch.isnan(torch_output_ref) & torch.isnan(torch_output_actual)
        )
        percent_match = 100.0 * match_mask.sum().item() / match_mask.numel()
        print(f"\t Match % for scalar {scalar}: {percent_match:.2f}%")

        # Save to CSV
        df = pd.DataFrame(
            {
                "Input": torch_input_bf16.flatten().tolist(),
                "Torch result": torch_output_ref.flatten().tolist(),
                "TTNN Result": torch_output_actual.flatten().tolist(),
                "Equal check": match_mask.flatten().tolist(),
            }
        )
        filename = f"{operation_name}-bfloat16-eq-scalar={scalar}.csv"
        csv_path = os.path.join(dest_dir, filename)
        df.to_csv(csv_path, index_label="index")
        print(f"\t Saved results to {csv_path}")

        scalar_labels.append(str(scalar))
        match_percentages.append(percent_match)

    # Plot bar graph of scalar vs mistmatch %
    mismatch_percentages = [100.0 - match for match in match_percentages]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scalar_labels, mismatch_percentages, color="salmon")
    plt.yscale("log")  # Using log scale to emphasize small mismatches

    plt.title(f"{operation_name} - Mismatch % per Scalar (bfloat16)")
    plt.xlabel("Scalar")
    plt.ylabel("Mismatch Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5)
    for bar, mismatch in zip(bars, mismatch_percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{mismatch:.2f}%", ha="center", va="bottom", fontsize=8)

    # Save plot
    plot_dir = os.path.join(os.path.dirname(dest_dir.rstrip("/")), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{operation_name}-bfloat16-eq-barplot-mismatch.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"----> Overall result : Histogram plot saved to: {plot_path}")


def measure_op_accuracy_bf16(operation_name, dest_dir, group_size=None):
    # Use bfloat16 parameters
    parameters = datatypes_parameters["bfloat16"]

    # Ensure group_size is a power of 2
    if group_size is not None and (group_size & (group_size - 1)) != 0:
        raise ValueError(f"Number of samples ({group_size}) must be a power of 2")

    # Create 2^9 x 2^7 tensor (2^16 elements total)
    TENSOR_WIDTH = 2**7
    TENSOR_HEIGHT = 2**9
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    SIGN_BITS = parameters["sign_bits"]  # should be 1
    EXPONENT_BITS = parameters["exponent_bits"]
    MANTISSA_BITS = parameters["mantissa_bits"]

    NUMPY_TYPE = parameters["numpy_type"]
    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TORCH_INT_TYPE = parameters["torch_int_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Group by exponent if group_size is not specified
    sub_batches = 2**9 if group_size is None else 2**16 // group_size

    # Create input tensors
    input_np = np.arange(0, 2**16, dtype=NUMPY_INT_TYPE)  # All possible bfloat16 values
    torch_value = torch.from_numpy(input_np).reshape(size)
    torch_input_bf16 = torch_value.view(TORCH_TYPE)  # reinterpret data as bfloat16
    # torch_input_bf16[torch.isnan(torch_input_bf16)] = 0.0 # replace nan to 0.0
    torch_input_f64 = torch_input_bf16.to(torch.float64)  # Convert to float64 for torch golden function

    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    # Get the operations to test
    (torch_unary_op, ttnn_unary_op, python_unary_op, scalar_list, parent_op) = operations_dict[operation_name]

    plt.figure(figsize=(12, 6))
    scalar_list = scalar_list if scalar_list is not None else [None]
    for scalar in scalar_list:
        print(f"  Testing with scalar = {scalar}")

        # Initialize arrays for measurements
        [
            x_array,
            y_array,
            yref_array,
            max_abs_error_array,
            mean_abs_error_array,
            max_ulp_error_array,
            mean_ulp_error_array,
            max_rel_error_array,
            mean_rel_error_array,
        ] = [np.zeros([sub_batches], dtype=np.float64) for _ in range(9)]

        start_time = time.time()

        # Launch TTNN operation
        def launch_ttnn_op(torch_tensor, scalar, ttnn_unary, ttnn_output):
            ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
            # check of optional output tensor support is provided for the op
            if scalar is not None:
                ttnn_output = ttnn_unary(ttnn_value, scalar)
            else:
                ttnn_output = ttnn_unary(ttnn_value)
            return ttnn.to_torch(ttnn_output)

        # Run reference and actual operations
        if scalar is not None:
            torch_golden_f64 = torch_unary_op(torch_input_f64, scalar)
        else:
            torch_golden_f64 = torch_unary_op(torch_input_f64)

        torch_ttnn_output_bf16 = launch_ttnn_op(torch_input_bf16, scalar, ttnn_unary_op, ttnn_output)

        torch_golden_bf16 = torch_golden_f64.to(torch.bfloat16)
        torch_ttnn_output_f64 = torch_ttnn_output_bf16.to(torch.float64)

        # Compute errors
        np_golden_f64 = torch_golden_f64.flatten().numpy()
        np_ttnn_output_f64 = torch_ttnn_output_f64.flatten().numpy()
        np_diff = np.abs(np_golden_f64 - np_ttnn_output_f64)

        torch_ulp_value = util.ulp(torch_golden_bf16).to(torch.float64)
        torch_eps = torch.full(torch_input_bf16.size(), EPSILON, dtype=torch.float64)
        np_eps = np.full(2**16, EPSILON)

        np_rel_error = np_diff / np.maximum(np.abs(np_golden_f64), np_eps)
        np_ulp_error = np_diff / torch_ulp_value.flatten().numpy()

        finite_mask = np.isfinite(np_golden_f64) & np.isfinite(np_ttnn_output_f64)
        # pcc = scipy.stats.pearsonr(np_golden_f64[finite_mask], np_ttnn_output_f64[finite_mask])
        if np.count_nonzero(finite_mask) >= 2:
            pcc = scipy.stats.pearsonr(np_golden_f64[finite_mask], np_ttnn_output_f64[finite_mask])
        else:
            pcc = (float("nan"), float("nan"))
            print(f"{operation_name} [bfloat16] PCC could not be computed due to insufficient finite values.")

        # Flatten tensors and convert to ndarray for analysis
        np_flat_input = torch_input_f64.flatten().numpy()
        np_flat_output = torch_ttnn_output_f64.flatten().numpy()
        np_flat_golden = torch_golden_f64.flatten().numpy()

        # Process each sub-batch
        for j in range(0, sub_batches):
            chunk_size = TENSOR_WIDTH * TENSOR_HEIGHT // sub_batches
            (beg_index, end_index) = (j * chunk_size, (j + 1) * chunk_size)

            # Get sub-range
            np_sub_input = np_flat_input[beg_index:end_index]
            np_sub_output = np_flat_output[beg_index:end_index]
            np_sub_ref = np_flat_golden[beg_index:end_index]
            np_sub_diff = np_diff[beg_index:end_index]
            np_sub_rel_error = np_rel_error[beg_index:end_index]
            np_sub_ulp_error = np_ulp_error[beg_index:end_index]

            # Calculate errors

            finite_mask = np.isfinite(np_sub_diff)
            if np.any(finite_mask):
                max_abs_error = np.max(np_sub_diff[finite_mask])
                mean_abs_error = np.mean(np_sub_diff[finite_mask])

                max_ulp_error = np.max(np_sub_ulp_error[finite_mask])
                mean_ulp_error = np.mean(np_sub_ulp_error[finite_mask])

                max_rel_error = np.max(np_sub_rel_error[finite_mask])
                mean_rel_error = np.mean(np_sub_rel_error[finite_mask])

            else:
                max_abs_error = np.max(np_sub_diff)
                mean_abs_error = np.mean(np_sub_diff)

                max_ulp_error = np.max(np_sub_ulp_error)
                mean_ulp_error = np.mean(np_sub_ulp_error)

                max_rel_error = np.max(np_sub_rel_error)
                mean_rel_error = np.mean(np_sub_rel_error)

            # Store results
            x_array[j] = np_sub_input[0].item()
            y_array[j] = np_sub_output[0].item()
            yref_array[j] = np_sub_ref[0].item()
            max_abs_error_array[j] = max_abs_error.item()
            mean_abs_error_array[j] = mean_abs_error.item()
            max_ulp_error_array[j] = max_ulp_error.item()
            mean_ulp_error_array[j] = mean_ulp_error.item()
            max_rel_error_array[j] = max_rel_error.item()
            mean_rel_error_array[j] = mean_rel_error.item()

        # Create and save DataFrame
        accuracy_df = pd.DataFrame(
            {
                "base_x": x_array,
                "base_y": y_array,
                "base_yref": yref_array,
                "max_abs_error": max_abs_error_array,
                "mean_abs_error": mean_abs_error_array,
                "max_ulp_error": max_ulp_error_array,
                "mean_ulp_error": mean_ulp_error_array,
                "max_rel_error": max_rel_error_array,
                "mean_rel_error": mean_rel_error_array,
            }
        )
        accuracy_df["operation"] = operation_name
        accuracy_df["dtype"] = "bfloat16"
        accuracy_df["parent_op"] = parent_op

        # Compute PCC on [-1e5; 1e5]
        np_finite_mask = np.isfinite(np_flat_output) & np.isfinite(np_flat_golden)
        df = pd.DataFrame(
            {
                "x": np_flat_input[np_finite_mask],
                "y": np_flat_output[np_finite_mask],
                "yref": np_flat_golden[np_finite_mask],
            }
        )

        df = df[df["x"].between(-1e5, 1e5)]
        # pcc = scipy.stats.pearsonr(df["y"], df["yref"])
        if len(df) >= 2:
            pcc = scipy.stats.pearsonr(df["y"], df["yref"])
        else:
            pcc = (float("nan"), float("nan"))
            print(
                f"{operation_name} [bfloat16] PCC could not be computed after filtering (length={len(df)}). Duration = {elapsed_s:.4f}s"
            )

        golden_std = np.std(np_flat_golden)
        ttnn_std = np.std(np_flat_output)

        np_finite_ulp_mask = np.isfinite(np_ulp_error) & (
            np.greater(np_flat_input, -(2**6)) & np.less(np_flat_input, 2**6)
        )
        mean_ulp_error = np.mean(np_ulp_error[np_finite_ulp_mask])
        print(f"Finite ulp error = {np_ulp_error[np_finite_ulp_mask]}")

        print(f"Mean ulp error = {mean_ulp_error}")

        covar = np.cov(np_flat_golden, np_flat_output)
        print(f"Golden std = {golden_std}, TTNN std = {ttnn_std}")
        print(f"Covar = {covar}")

        # accuracy_df.to_csv(f"{dest_dir}/{operation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")
        if scalar is not None:
            filename = f"{operation_name}-bfloat16_{scalar}_{group_size}.csv"
        else:
            filename = f"{operation_name}-bfloat16_{group_size}.csv"
        csv_path = os.path.join(dest_dir, filename)
        accuracy_df.to_csv(csv_path, na_rep="NaN", index_label="index")
        print(f"--> Results written to: {csv_path}")

        end_time = time.time()
        elapsed_s = end_time - start_time
        print(f"{operation_name} [bfloat16] PCC = {pcc[0]}, Duration = {elapsed_s:.4f}s")

        # Single plot
        print(f"Plotting max ULP for {operation_name}, x in [-360, 360], y in [0, 20]")
        plot_df = pd.read_csv(csv_path)

        label = f"scalar={scalar}" if scalar is not None else operation_name
        plt.plot(plot_df["base_x"], plot_df["max_ulp_error"], label=label)

        plt.title(f"TTNN vs PyTorch: Max ULP Error for {operation_name} (bfloat16)")
        plt.xlabel("Input Value (base_x)")
        plt.ylabel("Max ULP Error")
        plt.ylim(0, 20)
        if operation_name == "deg2rad":
            plt.xlim(-360, 360)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.title(f"TTNN vs PyTorch: Mean ULP Error for {operation_name} (bfloat16)")
    plt.xlabel("Input Value (base_x)")
    plt.ylabel("Mean ULP Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(dest_dir.rstrip("/")), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if scalar is not None:
        plot_path = os.path.join(plot_dir, f"{operation_name}-bfloat16_all_scalars_{group_size}.png")
    else:
        plot_path = os.path.join(plot_dir, f"{operation_name}-bfloat16_{group_size}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"----> Overall result : plot saved to: {plot_path}")


def main(args):
    dest_dir = "accuracy_results/results/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")

    #   Ops that require equal check
    equal_check_operations = [
        "maximum",
        "minimum",
    ]

    all_operations = [
        "deg2rad",
        "exp",
        "expm1",
        "elu",
        "selu",
    ]

    highres_operations = [
        #
    ]

    success_count = 0
    successfull_operations = []
    failed_operations = []
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    cnt = 0
    total_operation_cnt = len(all_operations) + len(highres_operations)

    for operation in equal_check_operations:
        cnt += 1
        print(f"\nRunning operation {operation}  #{cnt}/{total_operation_cnt}")
        try:
            measure_op_accuracy_bf16_eq_only(operation, dest_dir)
            success_count += 1
            successfull_operations.append(operation)
        except Exception as e:
            print(f"{RED}Could not run operation {operation}: {e}{RESET}")
            print(f"{RED}{traceback.format_exc()}{RESET}")
            failed_operations.append(operation)

    for operation in all_operations:
        cnt += 1
        print(f"\nRunning operation {operation}  #{cnt}/{total_operation_cnt}")
        try:
            measure_op_accuracy_bf16(operation, dest_dir, group_size=32)
            success_count += 1
            successfull_operations.append(operation)
        except Exception as e:
            print(f"{RED}Could not run operation {operation}: {e}{RESET}")
            print(f"{RED}{traceback.format_exc()}{RESET}")
            failed_operations.append(operation)

    # High-resolution ops
    print(f"Now measuring high-resolution operations")
    for operation in highres_operations:
        cnt += 1
        print(f"Running operation {operation} [highres] #{cnt}/{total_operation_cnt}")
        try:
            measure_op_accuracy_bf16(operation, dest_dir, group_size=1)
            success_count += 1
            successfull_operations.append(f"{operation}[highres]")
        except Exception as e:
            print(f"{RED}Could not run operation {operation}: {e}{RESET}")
            print(f"{RED}{traceback.format_exc()}{RESET}")
            failed_operations.append(f"{operation}[highres]")

    # Summary
    print(f"\nRan {success_count} / {total_operation_cnt} operations successfully.")
    if successfull_operations:
        print(f"{GREEN}SUCCESS: {successfull_operations}{RESET}")
    if failed_operations:
        print(f"{RED}FAILED: {failed_operations}{RESET}")


if __name__ == "__main__":
    args = sys.argv
    try:
        main(args)
    finally:
        ttnn.close_device(device)
