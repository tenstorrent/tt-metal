import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from models.common.utility_functions import ulp
import os
from loguru import logger
import ttnn

# Disable TT logging
os.environ["TT_LOGGER_LEVEL"] = "off"
os.environ["TT_METAL_CACHE"] = "/localdev/astancov/tt-metal/built"
logger.disable("ttnn")

default_fig_size = (8, 5)


def ulp_error(res, ref):
    """
    Compute the ULP error (in ref ULPs) between two tensors.

    Args:
        res: Result tensor (calculated/actual)
        ref: Reference tensor (golden/expected)

    Returns:
        tuple: (ulp_errors, ulp_details) where:
            - ulp_errors: ULP error tensor between res and ref (in ULPs of ref)
            - ulp_details: dict containing max ULP error location and values
    """
    ref_ulp = ulp(ref)
    ulp_errors = torch.abs((res.to(torch.float64) - ref.to(torch.float64)) / ref_ulp.to(torch.float64))
    # ulp_errors = torch.abs((res - ref) / ref_ulp)

    # Track details about max ULP error
    max_ulp = torch.max(ulp_errors)
    ulp_index = torch.argmax(ulp_errors)
    ulp_index_tuple = tuple(int(idx) for idx in torch.unravel_index(ulp_index, ref.shape))

    ulp_details = {
        "max_ulp": float(max_ulp),
        "max_ulp_index": ulp_index_tuple,
        "calculated_value": float(res[ulp_index_tuple]),
        "golden_value": float(ref[ulp_index_tuple]),
        "ulp_value": float(ref_ulp[ulp_index_tuple]),
    }

    return ulp_errors, ulp_details


def run_ttnn_conv2d(
    input_tensor, weight_tensor, bias_tensor, padding, stride, fp32_acc=False, device=None, use_bias=True
):
    """Run conv2d in ttnn."""
    create_local_device = device is None
    if create_local_device:
        device = ttnn.CreateDevice(0, l1_small_size=16384)

    # Extract dimensions
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    input_height = input_tensor.shape[2]
    input_width = input_tensor.shape[3]
    out_channels = weight_tensor.shape[0]
    kernel_h = weight_tensor.shape[2]
    kernel_w = weight_tensor.shape[3]

    # Handle padding and stride
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding[0], padding[1]

    if isinstance(stride, int):
        str_h, str_w = stride, stride
    else:
        str_h, str_w = stride[0], stride[1]

    # Convert input from NCHW to NHWC format
    input_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()

    # Prepare bias - reshape to (1, 1, 1, out_channels) if needed
    if use_bias:
        if bias_tensor.dim() == 1:
            bias_tensor = bias_tensor.reshape(1, 1, 1, -1)
        elif bias_tensor.shape != (1, 1, 1, out_channels):
            bias_tensor = bias_tensor.reshape(1, 1, 1, -1)

    # Determine ttnn dtype
    ttnn_dtype = ttnn.bfloat16 if input_tensor.dtype == torch.bfloat16 else ttnn.float32

    # Convert to ttnn tensors
    tt_input = ttnn.from_torch(input_nhwc, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(weight_tensor, dtype=ttnn_dtype)
    tt_bias = ttnn.from_torch(bias_tensor, dtype=ttnn_dtype) if use_bias else None

    # Configure compute kernel
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    # Run conv2d
    [tt_output, out_dims, _] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(kernel_h, kernel_w),
        stride=(str_h, str_w),
        padding=(pad_h, pad_h, pad_w, pad_w),
        dilation=(1, 1),
        groups=1,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    # Convert back to torch and reshape to NCHW
    output_torch = ttnn.to_torch(tt_output, dtype=input_tensor.dtype)
    out_height, out_width = out_dims
    output_nchw = output_torch.reshape(batch_size, out_height, out_width, out_channels).permute(0, 3, 1, 2)

    if create_local_device:
        ttnn.close_device(device)

    return output_nchw


def run_ttnn_conv_transpose2d(
    input_tensor,
    weight_tensor,
    bias_tensor,
    padding,
    stride,
    output_padding=(0, 0),
    fp32_acc=False,
    device=None,
    use_bias=True,
):
    """Run conv_transpose2d in ttnn."""
    create_local_device = device is None
    if create_local_device:
        device = ttnn.CreateDevice(0, l1_small_size=16384)

    # Extract dimensions
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    input_height = input_tensor.shape[2]
    input_width = input_tensor.shape[3]
    out_channels = weight_tensor.shape[1]  # Note: weight is IOHW for transposed conv
    kernel_h = weight_tensor.shape[2]
    kernel_w = weight_tensor.shape[3]

    # Handle padding and stride
    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding[0], padding[1]

    if isinstance(stride, int):
        str_h, str_w = stride, stride
    else:
        str_h, str_w = stride[0], stride[1]

    if isinstance(output_padding, int):
        out_pad_h, out_pad_w = output_padding, output_padding
    else:
        out_pad_h, out_pad_w = output_padding[0], output_padding[1]

    # Convert input from NCHW to NHWC format
    input_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()

    # Prepare bias - reshape to (1, 1, 1, out_channels) if needed
    if use_bias:
        if bias_tensor.dim() == 1:
            bias_tensor = bias_tensor.reshape(1, 1, 1, -1)
        elif bias_tensor.shape != (1, 1, 1, out_channels):
            bias_tensor = bias_tensor.reshape(1, 1, 1, -1)

    # Determine ttnn dtype
    ttnn_dtype = ttnn.bfloat16 if input_tensor.dtype == torch.bfloat16 else ttnn.float32

    # Convert to ttnn tensors
    tt_input = ttnn.from_torch(input_nhwc, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(weight_tensor, dtype=ttnn_dtype)
    tt_bias = ttnn.from_torch(bias_tensor, dtype=ttnn_dtype) if use_bias else None

    # Configure compute kernel
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    # Run conv_transpose2d
    [tt_output, out_dims, _] = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(kernel_h, kernel_w),
        stride=(str_h, str_w),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
        dilation=(1, 1),
        groups=1,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn_dtype,
    )

    # Convert back to torch and reshape to NCHW
    output_torch = ttnn.to_torch(tt_output, dtype=input_tensor.dtype)
    out_height, out_width = out_dims
    output_nchw = output_torch.reshape(batch_size, out_height, out_width, out_channels).permute(0, 3, 1, 2)
    # Trim to actual output channels
    output_nchw = output_nchw[:, :out_channels, :, :]

    if create_local_device:
        ttnn.close_device(device)

    return output_nchw


def run_conv_k_sweep_with_seed_variation(
    inner_channels_values, num_iterations=5, input_generator="rand", use_bias=False, conv_type="conv2d"
):
    """
    Run convolution with varying inner channels values, repeating each K with different random seeds.

    Args:
        inner_channels_values: List of inner channels values to test
        num_iterations: Number of times to run each K value with different random data
        input_generator: 'rand' or 'randn' for input generation
        use_bias: Whether to use bias
        conv_type: 'conv2d' or 'conv_transpose2d'

    Returns:
        Dictionary with results for each configuration
    """
    # Configurations to test: (dtype, fp32_acc, name)
    configs = [
        (torch.bfloat16, False, "bfp16"),
        (torch.bfloat16, True, "bfp16+fp32acc"),
        (torch.float32, False, "fp32"),
        (torch.float32, True, "fp32+fp32acc"),
    ]

    results = {
        config[2]: {
            "k_values": [],
            "max_ulp_mean": [],  # Mean of max_ulp across iterations
            "max_ulp_std": [],  # Std dev of max_ulp across iterations
            "max_ulp_min": [],  # Min of max_ulp across iterations
            "max_ulp_max": [],  # Max of max_ulp across iterations
            "max_ulp_all_iterations": [],  # List of lists: all max_ulp values for each K
            "max_ulp_details": [],  # List of lists: ulp_details for each K and iteration
            "p99_ulp_mean": [],
            "p99_ulp_std": [],
        }
        for config in configs
    }

    # Set up conv parameters based on type
    if conv_type == "conv2d":
        kernel_h, kernel_w = 3, 3
        padding = (1, 1)
        stride = (1, 1)
        output_padding = None
    else:  # conv_transpose2d
        kernel_h, kernel_w = 2, 2
        padding = (0, 0)
        stride = (2, 2)
        output_padding = (0, 0)

    for inner_channels in inner_channels_values:
        print(f"\nTesting inner channels={inner_channels}", flush=True)

        # Create convolution configuration
        batch_size = 1
        in_channels = inner_channels
        out_channels = 96
        h = 12
        w = 8

        k_value = inner_channels * kernel_h * kernel_w

        # Store max_ulp for each iteration for each config
        iteration_results = {config[2]: {"max_ulp": [], "p99_ulp": [], "ulp_details": []} for config in configs}

        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}", flush=True)

            # Generate new random data for this iteration (no seed set)
            if input_generator == "rand":
                input_tensor_fp32 = torch.rand(batch_size, in_channels, h, w)
                if conv_type == "conv2d":
                    weight_tensor_fp32 = torch.rand(out_channels, in_channels, kernel_h, kernel_w)
                else:
                    weight_tensor_fp32 = torch.rand(in_channels, out_channels, kernel_h, kernel_w)
                bias_tensor_fp32 = torch.rand(out_channels) if use_bias else torch.zeros(out_channels)
            elif input_generator == "randn":
                input_tensor_fp32 = torch.randn(batch_size, in_channels, h, w)
                if conv_type == "conv2d":
                    weight_tensor_fp32 = torch.randn(out_channels, in_channels, kernel_h, kernel_w)
                else:
                    weight_tensor_fp32 = torch.randn(in_channels, out_channels, kernel_h, kernel_w)
                bias_tensor_fp32 = torch.randn(out_channels) if use_bias else torch.zeros(out_channels)
            else:
                raise ValueError(f"Unknown input_generator: {input_generator}")

            # Test each configuration with this random data
            for dtype, fp32_acc, config_name in configs:
                # Cast tensors to target dtype
                input_tensor = input_tensor_fp32.to(dtype)
                weight_tensor = weight_tensor_fp32.to(dtype)
                bias_tensor = bias_tensor_fp32.to(dtype)

                # Compute reference
                with torch.no_grad():
                    if conv_type == "conv2d":
                        reference = torch.nn.functional.conv2d(
                            input_tensor,
                            weight_tensor,
                            bias=bias_tensor if use_bias else None,
                            padding=padding,
                            stride=stride,
                        )
                    else:  # conv_transpose2d
                        reference = torch.nn.functional.conv_transpose2d(
                            input_tensor,
                            weight_tensor,
                            bias=bias_tensor if use_bias else None,
                            padding=padding,
                            stride=stride,
                            output_padding=output_padding,
                        )

                # Run TTNN convolution (device will be created and closed internally)
                try:
                    if conv_type == "conv2d":
                        ttnn_output = run_ttnn_conv2d(
                            input_tensor,
                            weight_tensor,
                            bias_tensor,
                            padding=padding,
                            stride=stride,
                            fp32_acc=fp32_acc,
                            device=None,
                            use_bias=use_bias,
                        )
                    else:  # conv_transpose2d
                        ttnn_output = run_ttnn_conv_transpose2d(
                            input_tensor,
                            weight_tensor,
                            bias_tensor,
                            padding=padding,
                            stride=stride,
                            output_padding=output_padding,
                            fp32_acc=fp32_acc,
                            device=None,
                            use_bias=use_bias,
                        )

                    # Compute ULP errors
                    ulp_errors_tensor, ulp_details = ulp_error(ttnn_output, reference)
                    ulp_errors = ulp_errors_tensor.detach().cpu().numpy().flatten()

                    max_ulp = float(np.max(ulp_errors))
                    p99_ulp = float(np.percentile(ulp_errors, 99))

                    iteration_results[config_name]["max_ulp"].append(max_ulp)
                    iteration_results[config_name]["p99_ulp"].append(p99_ulp)
                    iteration_results[config_name]["ulp_details"].append(ulp_details)

                except Exception as e:
                    print(f"    Error in {config_name}: {e}", flush=True)
                    iteration_results[config_name]["max_ulp"].append(np.nan)
                    iteration_results[config_name]["p99_ulp"].append(np.nan)
                    iteration_results[config_name]["ulp_details"].append(None)

        # Compute statistics across iterations for each config
        for config_name in [config[2] for config in configs]:
            max_ulp_values = iteration_results[config_name]["max_ulp"]
            p99_ulp_values = iteration_results[config_name]["p99_ulp"]
            ulp_details_list = iteration_results[config_name]["ulp_details"]

            # Filter out NaNs and corresponding details
            valid_indices = [i for i, v in enumerate(max_ulp_values) if not np.isnan(v)]
            max_ulp_values = [max_ulp_values[i] for i in valid_indices]
            p99_ulp_values = [p99_ulp_values[i] for i in valid_indices]
            ulp_details_list = [ulp_details_list[i] for i in valid_indices]

            if len(max_ulp_values) > 0:
                results[config_name]["k_values"].append(k_value)
                results[config_name]["max_ulp_mean"].append(np.mean(max_ulp_values))
                results[config_name]["max_ulp_std"].append(np.std(max_ulp_values))
                results[config_name]["max_ulp_min"].append(np.min(max_ulp_values))
                results[config_name]["max_ulp_max"].append(np.max(max_ulp_values))
                results[config_name]["max_ulp_all_iterations"].append(max_ulp_values)
                results[config_name]["max_ulp_details"].append(ulp_details_list)
                results[config_name]["p99_ulp_mean"].append(np.mean(p99_ulp_values))
                results[config_name]["p99_ulp_std"].append(np.std(p99_ulp_values))

                # Print breakdown of individual seed values
                print(f"    {config_name}: K={k_value}", flush=True)
                print(f"      Individual max_ulp values: {[f'{v:.2f}' for v in max_ulp_values]}", flush=True)
                print(
                    f"      Min: {np.min(max_ulp_values):.2f}, Max: {np.max(max_ulp_values):.2f}, "
                    f"Mean: {np.mean(max_ulp_values):.2f}, Std: {np.std(max_ulp_values):.2f}",
                    flush=True,
                )

    # Print summary breakdown with calculation details
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: Max ULP Breakdown by K and Iteration with Calculation Details", flush=True)
    print("=" * 80, flush=True)

    for config_name in [config[2] for config in configs]:
        if len(results[config_name]["k_values"]) > 0:
            print(f"\n{config_name}:", flush=True)
            print("-" * 80, flush=True)
            for i, k_value in enumerate(results[config_name]["k_values"]):
                max_ulp_values = results[config_name]["max_ulp_all_iterations"][i]
                ulp_details_list = results[config_name]["max_ulp_details"][i]
                print(f"  K={k_value}:", flush=True)
                for iter_idx, (val, details) in enumerate(zip(max_ulp_values, ulp_details_list)):
                    if details is not None:
                        res_val = details["calculated_value"]
                        ref_val = details["golden_value"]
                        ulp_val = details["ulp_value"]
                        diff = abs(res_val - ref_val)
                        print(f"    Iteration {iter_idx}: max_ulp = {val:.4f}", flush=True)
                        print(
                            f"      |res - ref| / ulp(ref) = |{res_val:.6e} - {ref_val:.6e}| / {ulp_val:.6e}",
                            flush=True,
                        )
                        print(f"                            = {diff:.6e} / {ulp_val:.6e} = {val:.4f}", flush=True)
                    else:
                        print(f"    Iteration {iter_idx}: max_ulp = {val:.4f}", flush=True)
                print(f"    → Min={min(max_ulp_values):.4f}, Max={max(max_ulp_values):.4f}", flush=True)

    return results


def plot_seed_variation_results(results, input_generator, conv_type="conv2d"):
    """
    Plot max ULP variation across different random seeds.

    Args:
        results: Dictionary with results from run_conv_k_sweep_with_seed_variation
        input_generator: Name of input generator used
        conv_type: Type of convolution ('conv2d' or 'conv_transpose2d')
    """
    configs = ["bfp16", "bfp16+fp32acc", "fp32", "fp32+fp32acc"]

    kernel_str = "3x3" if conv_type == "conv2d" else "2x2"
    op_str = "Conv2D" if conv_type == "conv2d" else "Conv2D Transposed"

    # Create subplots: one for each configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        ax = axes[idx]
        data = results[config_name]

        if len(data["k_values"]) == 0:
            continue

        k_values = data["k_values"]
        max_ulp_min = data["max_ulp_min"]
        max_ulp_max = data["max_ulp_max"]

        # Plot min line
        ax.plot(
            k_values,
            max_ulp_min,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label="Min",
            alpha=0.85,
            color="#2ca02c",
            linestyle="-",
            markeredgewidth=1.5,
            markeredgecolor="white",
        )

        # Plot max line
        ax.plot(
            k_values,
            max_ulp_max,
            marker="s",
            linewidth=2.5,
            markersize=7,
            label="Max",
            alpha=0.85,
            color="#d62728",
            linestyle="--",
            markeredgewidth=1.5,
            markeredgecolor="white",
        )

        # Plot shaded area between min and max
        ax.fill_between(k_values, max_ulp_min, max_ulp_max, alpha=0.2, color="#1f77b4", label="Min-Max Range")

        ax.set_xlabel("K (inner_channels × kernel_h × kernel_w)", fontsize=11)
        ax.set_ylabel("Max ULP", fontsize=11)
        ax.set_title(f"{config_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{op_str} ({kernel_str} kernel) Max ULP Min-Max Range Across Random Seeds - {input_generator} inputs",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(f"{conv_type}_seed_variation_{input_generator}_max_ulp.png", dpi=150)
    plt.show()


def plot_seed_variation_comparison(all_results, conv_type="conv2d"):
    """
    Compare seed variation across different input generators.

    Args:
        all_results: Dictionary mapping input generator names to their results
        conv_type: Type of convolution ('conv2d' or 'conv_transpose2d')
    """
    configs = ["bfp16", "bfp16+fp32acc", "fp32", "fp32+fp32acc"]

    kernel_str = "3x3" if conv_type == "conv2d" else "2x2"
    op_str = "Conv2D" if conv_type == "conv2d" else "Conv2D Transposed"

    # Define distinct line styles and colors
    line_styles = ["-", "--"]
    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]

    # Create subplots: one for each configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        ax = axes[idx]

        for style_idx, (input_name, results) in enumerate(all_results.items()):
            if config_name in results:
                data = results[config_name]

                if len(data["k_values"]) == 0:
                    continue

                k_values = data["k_values"]
                max_ulp_min = data["max_ulp_min"]
                max_ulp_max = data["max_ulp_max"]

                color = colors[style_idx % len(colors)]
                marker = markers[style_idx % len(markers)]
                linestyle = line_styles[style_idx % len(line_styles)]

                # Plot max line
                ax.plot(
                    k_values,
                    max_ulp_max,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=2.5,
                    markersize=7,
                    label=f"{input_name} (max)",
                    alpha=0.85,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                )

                # Plot shaded area between min and max
                ax.fill_between(k_values, max_ulp_min, max_ulp_max, alpha=0.15, color=color)

        ax.set_xlabel("K (inner_channels × kernel_h × kernel_w)", fontsize=11)
        ax.set_ylabel("Max ULP", fontsize=11)
        ax.set_title(f"{config_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{op_str} ({kernel_str} kernel) Max ULP Min-Max Range Comparison - All Input Types",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(f"{conv_type}_seed_variation_comparison_max_ulp.png", dpi=150)
    plt.show()


def plot_p99_comparison(all_results, conv_type="conv2d"):
    """
    Plot P99 ULP values across K for different input generators.

    Args:
        all_results: Dictionary mapping input generator names to their results
        conv_type: Type of convolution ('conv2d' or 'conv_transpose2d')
    """
    configs = ["bfp16", "bfp16+fp32acc", "fp32", "fp32+fp32acc"]

    kernel_str = "3x3" if conv_type == "conv2d" else "2x2"
    op_str = "Conv2D" if conv_type == "conv2d" else "Conv2D Transposed"

    # Define distinct line styles and colors
    line_styles = ["-", "--"]
    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]

    # Create subplots: one for each configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        ax = axes[idx]

        for style_idx, (input_name, results) in enumerate(all_results.items()):
            if config_name in results:
                data = results[config_name]

                if len(data["k_values"]) == 0:
                    continue

                k_values = data["k_values"]
                p99_ulp_mean = data["p99_ulp_mean"]
                p99_ulp_std = data["p99_ulp_std"]

                color = colors[style_idx % len(colors)]
                marker = markers[style_idx % len(markers)]
                linestyle = line_styles[style_idx % len(line_styles)]

                # Plot P99 mean with error bars for std
                ax.errorbar(
                    k_values,
                    p99_ulp_mean,
                    yerr=p99_ulp_std,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=2.5,
                    markersize=7,
                    capsize=4,
                    label=input_name,
                    alpha=0.85,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                )

        ax.set_xlabel("K (inner_channels × kernel_h × kernel_w)", fontsize=11)
        ax.set_ylabel("P99 ULP (Mean ± Std)", fontsize=11)
        ax.set_title(f"{config_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{op_str} ({kernel_str} kernel) P99 ULP Across Random Seeds - All Input Types",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(f"{conv_type}_seed_variation_p99_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Number of iterations per K value
    num_iterations = 5

    # Define K values to test
    # For Conv2D (3x3): K = inner_channels * 9, so increment inner_channels by ~111 for K increment of ~1000
    # For Conv2D Transposed (2x2): K = inner_channels * 4, so increment inner_channels by ~250 for K increment of ~1000
    # Using 128 increment as a compromise that works reasonably for both
    inner_channels_values = [
        64,
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1152,
        1280,
        1408,
        1536,
        1664,
        1792,
        1920,
        2048,
    ]

    # Test Conv2D
    print("=" * 80, flush=True)
    print(f"Testing Conv2D with 3x3 kernel - {num_iterations} iterations per K", flush=True)
    print("=" * 80, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Testing with RAND input generator", flush=True)
    print("=" * 80, flush=True)
    conv2d_results_rand = run_conv_k_sweep_with_seed_variation(
        inner_channels_values,
        num_iterations=num_iterations,
        input_generator="rand",
        # use_bias=True,
        use_bias=False,
        conv_type="conv2d",
    )

    print("\n" + "=" * 80, flush=True)
    print("Testing with RANDN input generator", flush=True)
    print("=" * 80, flush=True)
    conv2d_results_randn = run_conv_k_sweep_with_seed_variation(
        inner_channels_values,
        num_iterations=num_iterations,
        input_generator="randn",
        # use_bias=True,
        use_bias=False,
        conv_type="conv2d",
    )

    # Test Conv Transpose2D
    print("\n" + "=" * 80, flush=True)
    print(f"Testing Conv2D Transposed with 2x2 kernel - {num_iterations} iterations per K", flush=True)
    print("=" * 80, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Testing with RAND input generator", flush=True)
    print("=" * 80, flush=True)
    conv_transpose_results_rand = run_conv_k_sweep_with_seed_variation(
        inner_channels_values,
        num_iterations=num_iterations,
        input_generator="rand",
        # use_bias=True,
        use_bias=False,
        conv_type="conv_transpose2d",
    )

    print("\n" + "=" * 80, flush=True)
    print("Testing with RANDN input generator", flush=True)
    print("=" * 80, flush=True)
    conv_transpose_results_randn = run_conv_k_sweep_with_seed_variation(
        inner_channels_values,
        num_iterations=num_iterations,
        input_generator="randn",
        # use_bias=True,
        use_bias=False,
        conv_type="conv_transpose2d",
    )

    # Generate plots
    print("\nGenerating plots...", flush=True)

    # Individual plots for each input type and conv type
    print("  Plotting Conv2D rand seed variation...", flush=True)
    plot_seed_variation_results(conv2d_results_rand, "rand", conv_type="conv2d")

    print("  Plotting Conv2D randn seed variation...", flush=True)
    plot_seed_variation_results(conv2d_results_randn, "randn", conv_type="conv2d")

    print("  Plotting Conv2D Transposed rand seed variation...", flush=True)
    plot_seed_variation_results(conv_transpose_results_rand, "rand", conv_type="conv_transpose2d")

    print("  Plotting Conv2D Transposed randn seed variation...", flush=True)
    plot_seed_variation_results(conv_transpose_results_randn, "randn", conv_type="conv_transpose2d")

    # Comparison plots
    print("  Plotting Conv2D comparison across input types...", flush=True)
    plot_seed_variation_comparison(
        {"rand [0,1)": conv2d_results_rand, "randn (-∞,∞)": conv2d_results_randn}, conv_type="conv2d"
    )

    print("  Plotting Conv2D Transposed comparison across input types...", flush=True)
    plot_seed_variation_comparison(
        {"rand [0,1)": conv_transpose_results_rand, "randn (-∞,∞)": conv_transpose_results_randn},
        conv_type="conv_transpose2d",
    )

    # P99 plots
    print("  Plotting Conv2D P99 ULP comparison...", flush=True)
    plot_p99_comparison({"rand [0,1)": conv2d_results_rand, "randn (-∞,∞)": conv2d_results_randn}, conv_type="conv2d")

    print("  Plotting Conv2D Transposed P99 ULP comparison...", flush=True)
    plot_p99_comparison(
        {"rand [0,1)": conv_transpose_results_rand, "randn (-∞,∞)": conv_transpose_results_randn},
        conv_type="conv_transpose2d",
    )

    print("\nAll plots generated successfully!", flush=True)
