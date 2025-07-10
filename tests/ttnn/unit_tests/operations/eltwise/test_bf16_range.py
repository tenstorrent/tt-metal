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
import csv


device_id = 0
device = ttnn.open_device(device_id=device_id)

EPSILON = 2**-9

# ttnn.enable_program_cache(device)  # Useful: we are going to call the same kernel several times


def test_plot_binary_op_pow_looping_tensor():
    torch_binary_op = torch.pow
    ttnn_op = ttnn.pow
    low = -100
    high = 100
    x = torch.arange(low, high, 0.1, dtype=torch.float32)
    x_bf16 = x.to(torch.bfloat16)

    plot_dir = "accuracy_results/plots/pow_results/"
    csv_dir = "accuracy_results/csvs/pow_results/"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_results.csv")

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["input_x", "input_y", "torch_result", "ttnn_result", "abs_error", "ulp_error"])

    scalar_values = []
    mean_ulp_errors = []
    max_ulp_errors = []

    for y_scalar in np.arange(1.0, 10.5, 0.5):
        y = torch.full_like(x, fill_value=y_scalar, dtype=torch.float32)
        y_bf16 = y.to(torch.bfloat16)

        torch_out = torch_binary_op(x, y)

        ttnn_x = ttnn.from_torch(x_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn_y = ttnn.from_torch(y_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn_out_res = ttnn.multiply(
            ttnn_x,
            ttnn_y,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
            use_legacy=False,
        )
        ttnn_out = ttnn.to_torch(ttnn_out_res).to(torch.float32)

        abs_error = torch.abs(torch_out - ttnn_out)
        ulp_spacing = util.ulp(torch_out.to(torch.bfloat16)).to(torch.float32)
        ulp_error = abs_error / ulp_spacing

        # Filter valid ULP values to avoid NaNs/infs
        valid_mask = torch.isfinite(ulp_error)
        filtered_ulp_error = ulp_error[valid_mask]

        scalar_values.append(y_scalar)
        mean_ulp_errors.append(filtered_ulp_error.mean().item() if filtered_ulp_error.numel() > 0 else float("nan"))
        max_ulp_errors.append(filtered_ulp_error.max().item() if filtered_ulp_error.numel() > 0 else float("nan"))

        # CSV
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for i in range(len(x)):
                writer.writerow(
                    [
                        x[i].item(),
                        y_scalar,
                        torch_out[i].item(),
                        ttnn_out[i].item(),
                        abs_error[i].item(),
                        ulp_error[i].item(),
                    ]
                )

        # Output Comparison
        plt.plot(x.numpy(), torch_out.numpy(), label="torch", linewidth=1)
        plt.plot(x.numpy(), ttnn_out.numpy(), label="ttnn", linestyle="--", linewidth=1)
        plt.title(f"Output Comparison: {torch_binary_op.__name__}(x, y={y_scalar})\nInput Range: x ∈ [{low}, {high}]")
        plt.xlabel(f"x (with y = {y_scalar})")
        plt.ylabel(f"{torch_binary_op.__name__}(x, {y_scalar})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_y_{y_scalar}.png"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"\n[x, y={y_scalar}] Output graph saved to {os.path.abspath(save_path)}")

        # ULP
        plt.figure(figsize=(10, 5))
        plt.plot(x.numpy(), ulp_error.numpy(), label="ULP Error", color="red", linewidth=1)
        plt.title(f"ULP Error: {ttnn_op.__name__} vs Torch\nInput Range: x ∈ [{low}, {high}], y = {y_scalar}")
        plt.xlabel(f"x (with y = {y_scalar})")
        plt.ylabel("ULP Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        ulp_filename = f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_y_{y_scalar}_ulp.png"
        ulp_path = os.path.join(plot_dir, ulp_filename)
        plt.savefig(ulp_path)
        plt.close()
        print(f"[x, y={y_scalar}] ULP graph saved to {os.path.abspath(ulp_path)}")

    plt.figure(figsize=(10, 5))
    plt.plot(scalar_values, mean_ulp_errors, label="Mean ULP Error", marker="o")
    plt.plot(scalar_values, max_ulp_errors, label="Max ULP Error", marker="x")
    plt.title(f"ULP Error Summary across Scalar y ∈ [1.0, 10.0]")
    plt.xlabel("Scalar y Value")
    plt.ylabel("ULP Error")
    plt.xticks(scalar_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    summary_path = os.path.join(plot_dir, f"{ttnn_op.__name__}_ulp_summary_vs_scalar.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"\nULP summary graph saved to {os.path.abspath(summary_path)}")

    print(f"\nAll results saved to CSV: {os.path.abspath(csv_path)}")


def plot_using_arange(torch_unary_op, ttnn_op, scalar=None, low=-100, high=100):
    if low > high:
        low, high = high, low
    x = torch.arange(low, high, 0.1, dtype=torch.bfloat16)

    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()

    if scalar is not None:
        torch_out = torch_unary_op(x, alpha=scalar)
    else:
        torch_out = torch_unary_op(x)

    ttnn_value = ttnn.from_torch(x.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    if scalar is not None:
        ttnn_out = ttnn.to_torch(ttnn_op(ttnn_value, scalar))
    else:
        ttnn_out = ttnn.to_torch(ttnn_op(ttnn_value))

    torch_label = f"torch.{torch_unary_op.__name__}"
    ttnn_label = f"{ttnn_op.__name__}"
    y_label = f"{torch_unary_op.__name__}(x)"
    title = f"Comparison: {torch_label} vs {ttnn_label}"

    x_plot = x.to(torch.float32).numpy()
    torch_out_plot = torch_out.to(torch.float32).numpy()
    ttnn_out_plot = ttnn_out.to(torch.float32).numpy()

    plt.plot(x_plot, torch_out_plot, label=torch_label, linewidth=1)
    plt.plot(x_plot, ttnn_out_plot, label=ttnn_label, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel(y_label)
    if torch_unary_op.__name__ in ["exp", "expm1", "log", "log1p"]:
        plt.yscale("log")

    plt.grid(True)
    plt.legend()

    plot_dir = "accuracy_results/plots/arange_comparison/"
    os.makedirs(plot_dir, exist_ok=True)

    scalar_str = f"{scalar}" if scalar is not None else "noscalar"
    range_str = f"{int(low)}_{int(high)}"
    filename = f"{ttnn_op.__name__}_bf16_range_{range_str}_{scalar_str}.png"

    save_path = os.path.join(plot_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print("\tfor range [-100,100]")
    print(f"\t\t\tTorch vs TTNN output graph saved to {os.path.abspath(save_path)}")

    ulp_spacing = util.ulp(torch_out.to(torch.bfloat16)).to(torch.float32)
    abs_diff = torch.abs(torch_out - ttnn_out)
    ulp_error = abs_diff / ulp_spacing

    # ULP
    plt.plot(x_plot, ulp_error.numpy(), label="ULP Error", color="red", linewidth=1)
    plt.title(f"ULP Error: {ttnn_op.__name__} vs Torch")
    plt.xlabel("x")
    plt.ylabel("ULP Error")
    plt.grid(True)
    plt.legend()

    ulp_dir = "accuracy_results/plots/arange_comparison/"
    os.makedirs(ulp_dir, exist_ok=True)
    filename = f"{ttnn_op.__name__}_bf16_range_{range_str}_{scalar_str}_ulp.png"
    ulp_path = os.path.join(ulp_dir, filename)
    plt.savefig(ulp_path)
    plt.close()
    print(f"\t\t\tULP error graph saved to {os.path.abspath(ulp_path)}")

    # CSV
    csv_data = {
        "x": x_plot,
        "ttnn_out": ttnn_out_plot,
        "torch_out_bf16": torch_out_plot,
        "abs_diff": abs_diff.to(torch.float32).numpy(),
        "ulp_error": ulp_error.numpy(),
    }
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(plot_dir, f"{ttnn_op.__name__}_bf16__{scalar_str}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\t\t\tResults CSV saved to {os.path.abspath(csv_path)}")


def plot_torch_vs_ttnn_outputs_full_range(
    torch_output, ttnn_output, torch_input_bf16, operation_name, scalar=None, value_range=None
):
    torch_input_flat = torch_input_bf16.flatten().to(torch.float32)
    torch_output_flat = torch_output.flatten().to(torch.float32)
    ttnn_output_flat = ttnn_output.flatten().to(torch.float32)

    if value_range is not None:
        min_val, max_val = value_range
        mask = (torch_input_flat >= min_val) & (torch_input_flat <= max_val)

        if mask.sum() == 0:
            print(f"No input values found in specified range: {value_range}. Skipping plot.")
            return

        x_axis = torch_input_flat[mask].numpy()
        y_torch = torch_output_flat[mask].numpy()
        y_ttnn = ttnn_output_flat[mask].numpy()

        x_label = f"Input (filtered: {min_val} to {max_val})"
    else:
        x_axis = np.arange(torch_input_flat.numel())  # Use raw bit index (0 to 65535)
        y_torch = torch_output_flat.numpy()
        y_ttnn = ttnn_output_flat.numpy()
        x_label = "Bit Pattern Index (0 to 65535)"

    plt.figure(figsize=(14, 6))
    plt.plot(x_axis, y_torch, label="Torch Output", color="blue", linewidth=1)
    plt.plot(x_axis, y_ttnn, label="TTNN Output", color="orange", linewidth=1, linestyle="--")

    label_suffix = f" (scalar={scalar})" if scalar is not None else ""
    range_suffix = f"_range={value_range}" if value_range else "fullrange"
    plt.title(f"{operation_name}{label_suffix}: Torch vs TTNN Output")
    plt.xlabel(x_label)
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_dir = "accuracy_results/plots/full_range_comparison/"
    os.makedirs(plot_dir, exist_ok=True)
    scalar_str = f"{scalar}" if scalar is not None else "noscalar"
    filename = f"{operation_name}-bfloat16-output-{range_suffix}-{scalar_str}.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    print(f"==> Torch vs TTNN output graph saved to {os.path.join(plot_dir, filename)}")


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
        [1.0],
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

        # Plot input vs output graph
        plot_torch_vs_ttnn_outputs_full_range(
            torch_output_ref, torch_output_actual, torch_input_bf16, operation_name, scalar
        )

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

        # Plot input vs output graph
        plot_torch_vs_ttnn_outputs_full_range(
            torch_golden_bf16, torch_ttnn_output_bf16, torch_input_bf16, operation_name, scalar
        )
        if operation_name == "elu":
            plot_torch_vs_ttnn_outputs_full_range(
                torch_golden_bf16, torch_ttnn_output_bf16, torch_input_bf16, "elu", value_range=(-100.0, 100.0)
            )
            plot_using_arange(torch_unary_op, ttnn_unary_op, scalar=scalar, low=-100, high=100)
        elif operation_name == "selu":
            plot_torch_vs_ttnn_outputs_full_range(
                torch_golden_bf16, torch_ttnn_output_bf16, torch_input_bf16, "selu", value_range=(-100.0, 100.0)
            )
            plot_using_arange(torch_unary_op, ttnn_unary_op, scalar=None, low=-100, high=100)
        elif operation_name == "exp":
            plot_torch_vs_ttnn_outputs_full_range(
                torch_golden_bf16, torch_ttnn_output_bf16, torch_input_bf16, "exp", value_range=(-100.0, 100.0)
            )
            plot_using_arange(torch_unary_op, ttnn_unary_op, scalar=None, low=-100, high=100)
        elif operation_name == "expm1":
            plot_torch_vs_ttnn_outputs_full_range(
                torch_golden_bf16, torch_ttnn_output_bf16, torch_input_bf16, "expm1", value_range=(-100.0, 100.0)
            )
            plot_using_arange(torch_unary_op, ttnn_unary_op, scalar=None, low=-100, high=100)

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
        plt.plot(plot_df["base_x"], plot_df["max_ulp_error"], label=f"{label} (max ULP)")
        plt.plot(plot_df["base_x"], plot_df["mean_ulp_error"], label=f"{label} (mean ULP)", linestyle="--")

        plt.title(f"TTNN vs PyTorch: Max ULP Error for {operation_name} (bfloat16)")
        plt.xlabel("Input Value (base_x)")
        plt.ylabel("Max ULP Error")
        plt.ylim(0, 20)
        plt.xlim(-360, 360)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.title(f"TTNN vs PyTorch: Max ULP Error for {operation_name} (bfloat16)")
    plt.xlabel("Input Value (base_x)")
    plt.ylabel("Max ULP Error")
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
        # "maximum",
        # "minimum",
    ]

    all_operations = [
        # "deg2rad",
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
