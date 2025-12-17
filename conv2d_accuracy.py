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
logger.disable("ttnn")


def ulp_error(res, ref):
    """
    Compute the ULP error (in ref ULPs) between two tensors.

    Args:
        res: Result tensor (calculated/actual)
        ref: Reference tensor (golden/expected)

    Returns:
        ULP error between res and ref (in ULPs of ref)
        Formula: |res - ref| / ULP(ref)
    """
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


if __name__ == "__main__":
    inner_channels_values = [64, 128, 256, 512, 1024, 2048, 3072, 4096, 8092]

    print("=" * 80, flush=True)
    print("Testing with RAND input generator", flush=True)
    print("=" * 80, flush=True)
    results_rand = run_conv_k_sweep(inner_channels_values, input_generator="rand", use_bias=True)

    print("\n" + "=" * 80, flush=True)
    print("Testing with RANDN input generator", flush=True)
    print("=" * 80, flush=True)
    results_randn = run_conv_k_sweep(inner_channels_values, input_generator="randn", use_bias=True)
