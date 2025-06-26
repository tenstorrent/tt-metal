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
import utils
import matplotlib.pyplot as plt


device_id = 0
device = ttnn.open_device(device_id=device_id)

EPSILON = 2**-9

# ttnn.enable_program_cache(device)  # Useful: we are going to call the same kernel several times


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
        [0.0, 1.0, -1.0, -0.0],
        "maximum",
    ),
    "minimum": (
        torch.minimum,
        ttnn.minimum,
        None,
        [1.0],
        "minimum",
    ),
}


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

    for scalar in scalar_list:
        print(f"  Testing with scalar = {scalar}")
        # Create scalar tensor for torch
        scalar_tensor_f64 = torch.full(size, scalar, dtype=torch.float64)

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
            ttnn_output = ttnn_unary(ttnn_value, scalar, output_tensor=ttnn_output)
            return ttnn.to_torch(ttnn_output)

        # Run reference and actual operations
        torch_golden_f64 = torch_unary_op(torch_input_f64, scalar_tensor_f64, out=torch_output_ref)
        torch_ttnn_output_bf16 = launch_ttnn_op(torch_input_bf16, scalar, ttnn_unary_op, ttnn_output)

        torch_golden_bf16 = torch_golden_f64.to(torch.bfloat16)
        torch_ttnn_output_f64 = torch_ttnn_output_bf16.to(torch.float64)

        # Compute errors
        np_golden_f64 = torch_golden_f64.flatten().numpy()
        np_ttnn_output_f64 = torch_ttnn_output_f64.flatten().numpy()
        np_diff = np.abs(np_golden_f64 - np_ttnn_output_f64)

        torch_ulp_value = utils.ulp_bf16(torch_golden_bf16).to(torch.float64)
        torch_eps = torch.full(torch_input_bf16.size(), EPSILON, dtype=torch.float64)
        np_eps = np.full(2**16, EPSILON)

        np_rel_error = np_diff / np.maximum(np.abs(np_golden_f64), np_eps)
        np_ulp_error = np_diff / torch_ulp_value.flatten().numpy()

        finite_mask = np.isfinite(np_golden_f64) & np.isfinite(
            np_ttnn_output_f64
        )  # Don't compute PCC on NaN and infinity
        pcc = scipy.stats.pearsonr(np_golden_f64[finite_mask], np_ttnn_output_f64[finite_mask])

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
        pcc = scipy.stats.pearsonr(df["y"], df["yref"])

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
        filename = f"{operation_name}-bfloat16_{scalar}_{group_size}.csv"
        csv_path = os.path.join(dest_dir, filename)
        accuracy_df.to_csv(csv_path, na_rep="NaN", index_label="index")
        print(f"--> Results written to: {csv_path}")

        end_time = time.time()
        elapsed_s = end_time - start_time
        print(f"{operation_name} [bfloat16] PCC = {pcc[0]}, Duration = {elapsed_s:.4f}s")

        # Plot Mean ULP Error vs Input Value (base_x) for individual scalar values
        plot_df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df["base_x"], plot_df["mean_ulp_error"], label="Mean ULP Error", color="blue")
        plt.title(f"TTNN vs PyTorch: Mean ULP Error for {operation_name}")
        plt.xlabel("Input Value (base_x)")
        plt.ylabel("Mean ULP Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        parent_dir = os.path.dirname(dest_dir.rstrip("/"))
        plot_dir = os.path.join(parent_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"{operation_name}-bfloat16_{scalar}_{group_size}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"--> Plot saved to: {plot_path}")


def main(args):
    dest_dir = "accuracy_results/results/"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")

    all_operations = [
        "maximum",
        "minimum",
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
