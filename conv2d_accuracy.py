import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from models.common.utility_functions import ulp, comp_ulp
import os
from loguru import logger
import ttnn

# Set random seed for reproducibility
SEED = 0
# torch.manual_seed(SEED)
# np.random.seed(SEED)

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
              Formula: |res - ref| / ULP(ref)
            - ulp_details: dict containing:
                - max_ulp: maximum ULP error value
                - max_ulp_index: index tuple where max ULP occurs
                - calculated_value: res value at max ULP location
                - golden_value: ref value at max ULP location
                - ulp_value: ULP(ref) value at max ULP location (to detect near-zero division)
    """
    ref_ulp = ulp(ref)
    ulp_errors = torch.abs((res.to(torch.float64) - ref.to(torch.float64)) / ref_ulp.to(torch.float64))

    # Track details about max ULP error (similar to comp_ulp)
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


def compute_ulp_statistics(res, ref):
    """
    Compute ULP error statistics.

    Args:
        res: Result tensor
        ref: Reference tensor (numpy or torch)

    Returns:
        Dictionary with ULP statistics including details about max ULP error
    """
    # Convert numpy to torch if needed
    if isinstance(ref, np.ndarray):
        ref = torch.from_numpy(ref)

    ulp_errors, ulp_details = ulp_error(res, ref)
    ulp_vals = ulp_errors.detach().cpu().numpy().flatten()

    stats = {
        "mean": float(np.mean(ulp_vals)),
        "median": float(np.median(ulp_vals)),
        "max": float(np.max(ulp_vals)),
        "min": float(np.min(ulp_vals)),
        "p95": float(np.percentile(ulp_vals, 95)),
        "p99": float(np.percentile(ulp_vals, 99)),
    }

    # Add ULP details to track near-zero division
    stats.update(ulp_details)

    return stats


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


def make_rand_seeded():
    def rand_seeded(*args, **kwargs):
        torch.manual_seed(SEED)
        return torch.rand(*args, **kwargs)

    return rand_seeded


def make_randn_seeded():
    def randn_seeded(*args, **kwargs):
        torch.manual_seed(SEED)
        return torch.randn(*args, **kwargs)

    return randn_seeded


def make_randn_clamped(min_val, max_val):
    """
    Create a randn generator function that clamps values to [min_val, max_val].

    Args:
        min_val: Minimum value to clamp to
        max_val: Maximum value to clamp to

    Returns:
        Function that generates clamped random normal values
    """

    def randn_clamped(*args, **kwargs):
        torch.manual_seed(SEED)
        return torch.clamp(torch.randn(*args, **kwargs), min=min_val, max=max_val)

    return randn_clamped


def make_randn_clamped_to_dtype_max(dtype=torch.float32):
    """
    Generate random normal values clamped to the maximum finite value of the given dtype.
    This prevents infinity values from appearing in the tensors.

    Args:
        dtype: PyTorch dtype to use for clamping range

    Returns:
        Function that generates clamped random normal values
    """

    def randn_clamped(*args, **kwargs):
        torch.manual_seed(SEED)
        max_val = torch.finfo(dtype).max
        min_val = torch.finfo(dtype).min
        return torch.clamp(torch.randn(*args, **kwargs), min=min_val, max=max_val)

    return randn_clamped


def run_conv_k_sweep(inner_channels_values, input_generator="rand", use_bias=False, clamp_tuple=None):
    """
    Run convolution with varying inner channels values where:
    - K = inner_channels * kernel_h * kernel_w
    - Activation matrix: 32 x K
    - Weight matrix: K x 32

    Args:
        inner_channels_values: List of inner channels values to test
        input_generator: 'rand' or 'randn' for input generation
        use_bias: Whether to use bias
        clamp_tuple: Tuple of (min_val, max_val) to clamp the input to

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
        config[2]: {
            "k_values": [],
            "mean_ulp": [],
            "median_ulp": [],
            "max_ulp": [],
            "p99_ulp": [],
            "all_ulp_errors": [],
            "ulp_details": [],  # Track ULP details for each K value
        }
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
            gen_fn = make_rand_seeded()
        elif input_generator == "randn" and clamp_tuple is not None:
            gen_fn = make_randn_clamped(clamp_tuple[0], clamp_tuple[1])
        elif input_generator == "randn":
            gen_fn = make_randn_seeded()
        elif input_generator == "randn_clamped_fp32_max":
            gen_fn = make_randn_clamped_to_dtype_max(torch.float32)
        else:
            raise ValueError(f"Unknown input_generator: {input_generator}")

        # Generate input tensors in fp32 once for this K value
        # All configurations will use the same input data for fair comparison
        input_tensor_fp32 = gen_fn(batch_size, in_channels, h, w)
        weight_tensor_fp32 = gen_fn(out_channels, in_channels, kernel_h, kernel_w)
        bias_tensor_fp32 = gen_fn(out_channels) if use_bias else torch.zeros(out_channels)

        # Test each configuration with the same input data
        for dtype, fp32_acc, config_name in configs:
            print(f"  Config: {config_name}", flush=True)

            # Cast tensors to target dtype for this configuration
            input_tensor = input_tensor_fp32.to(dtype)
            weight_tensor = weight_tensor_fp32.to(dtype)
            bias_tensor = bias_tensor_fp32.to(dtype)

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

                # Compute ULP errors and details
                ulp_errors_tensor, ulp_details = ulp_error(ttnn_output, reference)
                ulp_errors = ulp_errors_tensor.detach().cpu().numpy().flatten()

                # Compute statistics manually for this K
                ulp_stats = {
                    "mean": float(np.mean(ulp_errors)),
                    "median": float(np.median(ulp_errors)),
                    "max": float(np.max(ulp_errors)),
                    "p99": float(np.percentile(ulp_errors, 99)),
                }

                results[config_name]["k_values"].append(inner_channels * kernel_h * kernel_w)
                results[config_name]["mean_ulp"].append(ulp_stats["mean"])
                results[config_name]["median_ulp"].append(ulp_stats["median"])
                results[config_name]["max_ulp"].append(ulp_stats["max"])
                results[config_name]["p99_ulp"].append(ulp_stats["p99"])

                # Store ULP errors and details for aggregated analysis
                results[config_name]["all_ulp_errors"].append(ulp_errors)
                results[config_name]["ulp_details"].append(ulp_details)

                # Print detailed ULP information including near-zero division detection
                print(
                    f"    Mean ULP: {ulp_stats['mean']:.2f}, Median ULP: {ulp_stats['median']:.2f}, P99 ULP: {ulp_stats['p99']:.2f}, Max ULP: {ulp_stats['max']:.2f}",
                    flush=True,
                )
                print(
                    f"    Max ULP details @ {list(ulp_details['max_ulp_index'])}: "
                    f"|{ulp_details['calculated_value']} - {ulp_details['golden_value']}| / {ulp_details['ulp_value']}",
                    flush=True,
                )
            except Exception as e:
                print(f"    Error: {e}", flush=True)
                results[config_name]["k_values"].append(inner_channels * kernel_h * kernel_w)
                results[config_name]["mean_ulp"].append(np.nan)
                results[config_name]["median_ulp"].append(np.nan)
                results[config_name]["max_ulp"].append(np.nan)
                results[config_name]["p99_ulp"].append(np.nan)

    # Compute aggregated statistics across all K values
    print("\n" + "=" * 80, flush=True)
    if input_generator == "rand":
        values_range = "[0,1)"
    elif input_generator == "randn" and clamp_tuple is not None:
        values_range = f"[{clamp_tuple[0]:.2e},{clamp_tuple[1]:.2e}]"
    elif input_generator == "randn_clamped_fp32_max":
        max_fp32 = torch.finfo(torch.float32).max
        values_range = f"[{torch.finfo(torch.float32).min:.2e},{max_fp32:.2e}]"
    else:
        values_range = "(-inf,inf)"
    print(
        f"AGGREGATED STATISTICS (All K values combined) - values: {values_range}",
        flush=True,
    )
    print("=" * 80, flush=True)

    aggregated_stats = {}
    for config_name in results.keys():
        if len(results[config_name]["all_ulp_errors"]) > 0:
            # Concatenate all ULP error arrays
            all_ulp_errors = np.concatenate(results[config_name]["all_ulp_errors"])

            # Find the worst-case ULP details across all K values
            ulp_details_list = results[config_name]["ulp_details"]
            worst_case_idx = max(range(len(ulp_details_list)), key=lambda i: ulp_details_list[i]["max_ulp"])
            worst_case_details = ulp_details_list[worst_case_idx]
            worst_case_k = results[config_name]["k_values"][worst_case_idx]

            # Compute statistics on combined ULP errors
            agg_stats = {
                "mean": float(np.mean(all_ulp_errors)),
                "median": float(np.median(all_ulp_errors)),
                "max": float(np.max(all_ulp_errors)),
                "min": float(np.min(all_ulp_errors)),
                "p95": float(np.percentile(all_ulp_errors, 95)),
                "p99": float(np.percentile(all_ulp_errors, 99)),
                "worst_case_details": worst_case_details,
                "worst_case_k": worst_case_k,
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
            print(f"  Worst case ULP details (K={worst_case_k}):", flush=True)
            print(
                f"    @ {list(worst_case_details['max_ulp_index'])}: "
                f"|{worst_case_details['calculated_value']:.6e} - {worst_case_details['golden_value']:.6e}| / {worst_case_details['ulp_value']:.6e}",
                flush=True,
            )

    results["_aggregated"] = aggregated_stats

    ttnn.close_device(device)
    return results


def plot_k_sweep_results(results, input_generator, metric="mean_ulp"):
    """
    Plot ULP results across K values for different configurations.

    Args:
        results: Dictionary with results from run_conv_k_sweep
        input_generator: Name of input generator used ('rand' or 'randn')
        metric: Which ULP metric to plot ('mean_ulp', 'median_ulp', or 'max_ulp')
    """
    plt.figure(figsize=(12, 7))

    for config_name, data in results.items():
        if config_name != "_aggregated":  # Skip aggregated stats
            plt.plot(data["k_values"], data[metric], marker="o", linewidth=2, label=config_name, markersize=8)

    plt.xlabel("K (inner_channels * kernel_h * kernel_w)", fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    plt.title(f"Conv2D (3x3 kernel) ULP vs K - {input_generator} inputs", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"conv_k_sweep_{input_generator}_{metric}.png", dpi=150)
    plt.show()


def plot_aggregated_comparison(results, input_generator):
    """
    Plot aggregated ULP statistics comparison across all K values.

    Args:
        results: Dictionary with results from run_conv_k_sweep
        input_generator: Name of input generator used ('rand' or 'randn')
    """
    if "_aggregated" not in results:
        print("No aggregated statistics available", flush=True)
        return

    aggregated = results["_aggregated"]
    config_names = list(aggregated.keys())
    metrics = ["p95", "p99", "max"]

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
    ax.set_title(f"Conv2D (3x3 kernel) Aggregated ULP Statistics - {input_generator} inputs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"conv_k_sweep_{input_generator}_aggregated.png", dpi=150)
    plt.show()


def plot_all_inputs_comparison(all_results_dict, metric="max_ulp"):
    """
    Plot ULP results across K values for all input types on the same graph.

    Args:
        all_results_dict: Dictionary mapping input type names to their results
        metric: Which ULP metric to plot ('mean_ulp', 'median_ulp', or 'max_ulp')
    """
    configs = ["bfp16", "bfp16+fp32acc", "fp32", "fp32+fp32acc"]

    # Define distinct line styles, markers, and colors for each input type
    line_styles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Create subplots: one for each configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        ax = axes[idx]

        for style_idx, (input_name, results) in enumerate(all_results_dict.items()):
            if config_name in results:
                data = results[config_name]
                ax.plot(
                    data["k_values"],
                    data[metric],
                    marker=markers[style_idx % len(markers)],
                    linestyle=line_styles[style_idx % len(line_styles)],
                    color=colors[style_idx % len(colors)],
                    linewidth=2.5,
                    label=input_name,
                    markersize=7,
                    alpha=0.85,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                )

        ax.set_xlabel("K (inner_channels × kernel_h × kernel_w)", fontsize=11)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=11)
        ax.set_title(f"{config_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Conv2D (3x3 kernel) {metric.replace('_', ' ').title()} Comparison - All Input Types",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(f"conv_all_inputs_comparison_{metric}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Define K values to test
    # For 3x3 kernel: K = inner_channels * 3 * 3 = inner_channels * 9
    inner_channels_values = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 8092, 10140]

    print("=" * 80, flush=True)
    print("Testing Conv2D with 3x3 kernel", flush=True)
    print("=" * 80, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Testing with RAND input generator", flush=True)
    print("=" * 80, flush=True)
    results_rand = run_conv_k_sweep(inner_channels_values, input_generator="rand", use_bias=True)

    print("\n" + "=" * 80, flush=True)
    clamp_tuple = (torch.finfo(torch.bfloat16).min, torch.finfo(torch.bfloat16).max)
    print(f"Testing with RANDN CLAMPED [{clamp_tuple[0]:.2e},{clamp_tuple[1]:.2e}] input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn_clamped = run_conv_k_sweep(
        inner_channels_values, input_generator="randn", use_bias=True, clamp_tuple=clamp_tuple
    )

    print("\n" + "=" * 80, flush=True)
    print("Testing with RANDN input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn = run_conv_k_sweep(inner_channels_values, input_generator="randn", use_bias=True)

    print("\n" + "=" * 80, flush=True)
    eps = torch.finfo(torch.bfloat16).eps
    clamp_tuple = (-1 + eps, 1 - eps)
    print(f"Testing with RANDN CLAMPED [{clamp_tuple[0]},{clamp_tuple[1]}] input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn_clamped_eps = run_conv_k_sweep(
        inner_channels_values, input_generator="randn", use_bias=True, clamp_tuple=clamp_tuple
    )

    print("\n" + "=" * 80, flush=True)
    eps = torch.finfo(torch.bfloat16).eps
    clamp_tuple = (0, 1 - eps)
    print(f"Testing with RANDN CLAMPED [{clamp_tuple[0]},{clamp_tuple[1]}] input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn_clamped_0_1 = run_conv_k_sweep(
        inner_channels_values, input_generator="randn", use_bias=True, clamp_tuple=clamp_tuple
    )

    # Generate plots
    print("\nGenerating plots...", flush=True)

    # Create a dictionary with all results for comparison
    all_results = {
        "rand [0,1)": results_rand,
        "randn (-∞,∞)": results_randn,
        f"randn clamped [bf16_min, bf16_max]": results_randn_clamped,
        f"randn clamped [-1+ε, 1-ε]": results_randn_clamped_eps,
        f"randn clamped [0, 1-ε]": results_randn_clamped_0_1,
    }

    # Plot comparison across all input types
    for metric in ["max_ulp", "p99_ulp"]:
        print(f"  Plotting {metric} comparison for all input types...", flush=True)
        plot_all_inputs_comparison(all_results, metric=metric)

    print("\nAll plots generated successfully!", flush=True)
