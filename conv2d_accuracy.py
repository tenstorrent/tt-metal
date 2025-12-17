import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from models.common.utility_functions import ulp, comp_ulp
import os
from loguru import logger
import ttnn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Disable TT logging
os.environ["TT_LOGGER_LEVEL"] = "off"
os.environ["TT_METAL_CACHE"] = "/localdev/astancov/tt-metal/built"
logger.disable("ttnn")

default_fig_size = (8, 5)


def ulp_error(res, ref):
    """
    Compute the ULP error (in ref ULPs) between two tensors.
    This matches the standard ULP calculation used in utility_functions.py comp_ulp.

    Args:
        res: Result tensor (calculated/actual)
        ref: Reference tensor (golden/expected)

    Returns:
        ULP error between res and ref (in ULPs of ref)
        Formula: |res - ref| / ULP(ref)
    """
    # ref_as_res_dtype = ref.to(res.dtype)
    ref_ulp = ulp(ref)
    return torch.abs((res.to(torch.float64) - ref.to(torch.float64)) / ref_ulp.to(torch.float64))


def compute_ulp_statistics(res, ref):
    """
    Compute ULP error statistics.

    Args:
        res: Result tensor
        ref: Reference tensor (numpy or torch)

    Returns:
        Dictionary with ULP statistics
    """
    # Convert numpy to torch if needed
    if isinstance(ref, np.ndarray):
        ref = torch.from_numpy(ref)

    ulp_errors = ulp_error(res, ref)
    ulp_vals = ulp_errors.detach().cpu().numpy().flatten()

    return {
        "mean": float(np.mean(ulp_vals)),
        "median": float(np.median(ulp_vals)),
        "max": float(np.max(ulp_vals)),
        "min": float(np.min(ulp_vals)),
        "p95": float(np.percentile(ulp_vals, 95)),
        "p99": float(np.percentile(ulp_vals, 99)),
    }


def plot_ulp_histogram_comparison(conv2d_res, conv2d_ref, matmul_res, matmul_ref, config_name, ref_type, n_bins=100):
    """
    Plot ULP error histograms for both conv2d and matmul side by side.
    """
    # Convert numpy to torch if needed
    if isinstance(conv2d_ref, np.ndarray):
        conv2d_ref = torch.from_numpy(conv2d_ref)
    if isinstance(matmul_ref, np.ndarray):
        matmul_ref = torch.from_numpy(matmul_ref)

    conv2d_ulp = ulp_error(conv2d_res, conv2d_ref).detach().cpu().numpy().flatten()
    matmul_ulp = ulp_error(matmul_res, matmul_ref).detach().cpu().numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Conv2D histogram
    ax1.hist(conv2d_ulp, bins=n_bins, alpha=0.7, edgecolor="black", color="blue")
    ax1.set_xlabel("ULP Error")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Conv2D ULP Error Histogram\n{config_name} vs {ref_type}")
    ax1.grid(True, alpha=0.3)

    # Matmul histogram
    ax2.hist(matmul_ulp, bins=n_bins, alpha=0.7, edgecolor="black", color="green")
    ax2.set_xlabel("ULP Error")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Matmul ULP Error Histogram\n{config_name} vs {ref_type}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ulp_cdf_comparison(conv2d_res, conv2d_ref, matmul_res, matmul_ref, config_name, ref_type):
    """
    Plot ULP error CDFs for both conv2d and matmul on the same plot.
    """
    # Convert numpy to torch if needed
    if isinstance(conv2d_ref, np.ndarray):
        conv2d_ref = torch.from_numpy(conv2d_ref)
    if isinstance(matmul_ref, np.ndarray):
        matmul_ref = torch.from_numpy(matmul_ref)

    conv2d_ulp = ulp_error(conv2d_res, conv2d_ref).detach().cpu().numpy().flatten()
    matmul_ulp = ulp_error(matmul_res, matmul_ref).detach().cpu().numpy().flatten()

    conv2d_sorted = np.sort(conv2d_ulp)
    matmul_sorted = np.sort(matmul_ulp)

    conv2d_cdf = np.arange(1, len(conv2d_sorted) + 1) / len(conv2d_sorted)
    matmul_cdf = np.arange(1, len(matmul_sorted) + 1) / len(matmul_sorted)

    plt.figure(figsize=default_fig_size)
    plt.plot(conv2d_sorted, conv2d_cdf, linewidth=2, label="Conv2D", alpha=0.8)
    plt.plot(matmul_sorted, matmul_cdf, linewidth=2, label="Matmul", alpha=0.8)
    plt.xlabel("ULP Error")
    plt.ylabel("Cumulative Probability")
    plt.title(f"ULP Error CDF: TTNN vs {ref_type}\n{config_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_ttnn_conv2d(
    input_tensor, weight_tensor, bias_tensor, padding, stride, fp32_acc=False, device=None, use_bias=True
):
    """
    Run conv2d in ttnn.

    Args:
        input_tensor: Input tensor (torch) in NCHW format
        weight_tensor: Weight tensor (torch) in OIHW format
        bias_tensor: Bias tensor (torch)
        padding: Padding tuple (h, w)
        stride: Stride tuple (h, w)
        fp32_acc: Whether to use fp32 accumulation
        device: TTNN device
        use_bias: Whether to use bias

    Returns:
        Output tensor (torch) in NCHW format
    """
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
        packer_l1_acc=True,  # l1_acc is turned on
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
    output_torch = ttnn.to_torch(tt_output)
    out_height, out_width = out_dims
    output_nchw = output_torch.reshape(batch_size, out_height, out_width, out_channels).permute(0, 3, 1, 2)

    if create_local_device:
        ttnn.close_device(device)

    return output_nchw


def run_conv_k_sweep(inner_channels_values, input_generator="rand", use_bias=False):
    """
    Run convolution with varying inner channels values where:
    - K = inner_channels * kernel_h * kernel_w
    - Activation matrix: 32 x K
    - Weight matrix: K x 32

    Args:
        inner_channels_values: List of inner channels values to test
        input_generator: 'rand' or 'randn' for input generation
        use_bias: Whether to use bias

    Returns:
        Dictionary with results for each configuration
    """
    device = ttnn.CreateDevice(0, l1_small_size=16384)

    # Configurations to test: (dtype, fp32_acc, name)
    configs = [
        (torch.bfloat16, False, "bfp16"),
        (torch.bfloat16, True, "bfp16+fp32acc"),
        (torch.float32, False, "fp32"),
        (torch.float32, True, "fp32+fp32acc"),
    ]
    results = {
        config[2]: {"k_values": [], "mean_ulp": [], "median_ulp": [], "max_ulp": [], "all_ulp_errors": []}
        for config in configs
    }

    for inner_channels in inner_channels_values:
        print(f"\nTesting inner channels={inner_channels}", flush=True)

        # Create convolution configuration
        # Input: (N, C_in, H, W) where C_in=32 and H*W=K
        # For simplicity, use H=K, W=1 (or factorize K for more realistic spatial dims)
        batch_size = 1
        in_channels = inner_channels
        out_channels = 96

        # Factorize K into H x W for more realistic spatial dimensions
        h = 12
        w = 8

        # 1x1 convolution kernel
        kernel_h, kernel_w = 3, 3
        padding = (1, 1)
        stride = (1, 1)

        # Select generation function based on input_generator
        if input_generator == "rand":
            gen_fn = torch.rand
        elif input_generator == "randn":
            gen_fn = torch.randn
        else:
            raise ValueError(f"Unknown input_generator: {input_generator}")

        # Generate input tensor
        input_tensor_fp32 = gen_fn(batch_size, in_channels, h, w)

        # Generate weight and bias using the same generation function
        weight_tensor_fp32 = gen_fn(out_channels, in_channels, kernel_h, kernel_w)
        bias_tensor_fp32 = gen_fn(out_channels) if use_bias else torch.zeros(out_channels)

        # Test each configuration
        for dtype, fp32_acc, config_name in configs:
            print(f"  Config: {config_name}", flush=True)

            # Convert tensors to target dtype
            input_tensor = input_tensor_fp32.to(dtype)
            weight_tensor = weight_tensor_fp32.to(dtype)
            if use_bias:
                bias_tensor = bias_tensor_fp32.to(dtype)
            else:
                bias_tensor = None

            # Compute reference in the same dtype as the configuration
            with torch.no_grad():
                reference = torch.nn.functional.conv2d(
                    input_tensor, weight_tensor, bias=bias_tensor if use_bias else None, padding=padding, stride=stride
                )

            # Run TTNN conv2d
            try:
                ttnn_output = run_ttnn_conv2d(
                    input_tensor,
                    weight_tensor,
                    bias_tensor,
                    padding=padding,
                    stride=stride,
                    fp32_acc=fp32_acc,
                    device=device,
                    use_bias=use_bias,
                )

                # Compute ULP errors
                ulp_errors = ulp_error(ttnn_output, reference).detach().cpu().numpy().flatten()

                # Compute statistics for this K
                ulp_stats = {
                    "mean": float(np.mean(ulp_errors)),
                    "median": float(np.median(ulp_errors)),
                    "max": float(np.max(ulp_errors)),
                }

                results[config_name]["k_values"].append(inner_channels * kernel_h * kernel_w)
                results[config_name]["mean_ulp"].append(ulp_stats["mean"])
                results[config_name]["median_ulp"].append(ulp_stats["median"])
                results[config_name]["max_ulp"].append(ulp_stats["max"])

                # Store ULP errors for aggregated analysis
                results[config_name]["all_ulp_errors"].append(ulp_errors)

                print(
                    f"    Mean ULP: {ulp_stats['mean']:.2f}, Median ULP: {ulp_stats['median']:.2f}, Max ULP: {ulp_stats['max']:.2f}",
                    flush=True,
                )
            except Exception as e:
                print(f"    Error: {e}", flush=True)
                results[config_name]["k_values"].append(inner_channels * kernel_h * kernel_w)
                results[config_name]["mean_ulp"].append(np.nan)
                results[config_name]["median_ulp"].append(np.nan)
                results[config_name]["max_ulp"].append(np.nan)

    # Compute aggregated statistics across all K values
    print("\n" + "=" * 80, flush=True)
    print(
        f"AGGREGATED STATISTICS (All K values combined) - values: {'[0,1)' if input_generator == 'rand' else '(-inf,inf)'}",
        flush=True,
    )
    print("=" * 80, flush=True)

    aggregated_stats = {}
    for config_name in results.keys():
        if len(results[config_name]["all_ulp_errors"]) > 0:
            # Concatenate all ULP error arrays
            all_ulp_errors = np.concatenate(results[config_name]["all_ulp_errors"])

            # Compute statistics on combined ULP errors
            agg_stats = {
                "mean": float(np.mean(all_ulp_errors)),
                "median": float(np.median(all_ulp_errors)),
                "max": float(np.max(all_ulp_errors)),
                "min": float(np.min(all_ulp_errors)),
                "p95": float(np.percentile(all_ulp_errors, 95)),
                "p99": float(np.percentile(all_ulp_errors, 99)),
            }
            aggregated_stats[config_name] = agg_stats

            print(f"\n{config_name}:", flush=True)
            print(f"  K values tested: {results[config_name]['k_values']}", flush=True)
            print(f"  Total samples: {len(all_ulp_errors)}", flush=True)
            print(f"  Mean ULP:   {agg_stats['mean']:.4f}", flush=True)
            print(f"  Median ULP: {agg_stats['median']:.4f}", flush=True)
            print(f"  Max ULP:    {agg_stats['max']:.4f}", flush=True)
            print(f"  P95 ULP:    {agg_stats['p95']:.4f}", flush=True)
            print(f"  P99 ULP:    {agg_stats['p99']:.4f}", flush=True)

    results["_aggregated"] = aggregated_stats

    ttnn.close_device(device)
    return results


def plot_k_sweep_results(results, input_generator, metric="mean_ulp"):
    """
    Plot ULP results across K values for different configurations.

    Args:
        results: Dictionary with results from test_conv_k_sweep
        input_generator: Name of input generator used ('rand' or 'randn')
        metric: Which ULP metric to plot ('mean_ulp', 'median_ulp', or 'max_ulp')
    """
    plt.figure(figsize=(12, 7))

    for config_name, data in results.items():
        if config_name != "_aggregated":  # Skip aggregated stats
            plt.plot(data["k_values"], data[metric], marker="o", linewidth=2, label=config_name, markersize=8)

    plt.xlabel("K (Activation: 32xK, Weight: Kx32)", fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    plt.title(f"Conv2D ULP vs K - {input_generator} inputs", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"conv_k_sweep_{input_generator}_{metric}.png", dpi=150)
    plt.show()


def plot_aggregated_comparison(results, input_generator):
    """
    Plot aggregated ULP statistics comparison across all K values.

    Args:
        results: Dictionary with results from test_conv_k_sweep
        input_generator: Name of input generator used ('rand' or 'randn')
    """
    if "_aggregated" not in results:
        print("No aggregated statistics available", flush=True)
        return

    aggregated = results["_aggregated"]
    config_names = list(aggregated.keys())
    metrics = ["mean", "median", "p95", "p99", "max"]

    # Prepare data for plotting
    data_by_metric = {metric: [] for metric in metrics}
    for config_name in config_names:
        for metric in metrics:
            data_by_metric[metric].append(aggregated[config_name][metric])

    # Create bar plot
    x = np.arange(len(config_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics) / 2 + 0.5)
        bars = ax.bar(x + offset, data_by_metric[metric], width, label=metric.upper())

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("ULP Error", fontsize=12)
    ax.set_title(f"Aggregated ULP Statistics (All K values) - {input_generator} inputs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"conv_k_sweep_{input_generator}_aggregated.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Define K values to test
    # inner_channels_values = [64,128]
    inner_channels_values = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 8092]

    print("=" * 80, flush=True)
    print("Testing with RAND input generator", flush=True)
    print("=" * 80, flush=True)
    results_rand = run_conv_k_sweep(inner_channels_values, input_generator="rand", use_bias=True)

    print("\n" + "=" * 80, flush=True)
    print("Testing with RANDN input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn = run_conv_k_sweep(inner_channels_values, input_generator="randn", use_bias=True)

    # # Plot results
    # print("\nGenerating plots...", flush=True)

    # # Per-K plots
    # for metric in ['mean_ulp', 'median_ulp', 'max_ulp']:
    #     print(f"  Plotting {metric} for rand inputs...", flush=True)
    #     plot_k_sweep_results(results_rand, 'rand', metric=metric)
    #     print(f"  Plotting {metric} for randn inputs...", flush=True)
    #     plot_k_sweep_results(results_randn, 'randn', metric=metric)

    # # Aggregated plots
    # print(f"  Plotting aggregated statistics for rand inputs...", flush=True)
    # plot_aggregated_comparison(results_rand, 'rand')
    # print(f"  Plotting aggregated statistics for randn inputs...", flush=True)
    # plot_aggregated_comparison(results_randn, 'randn')
