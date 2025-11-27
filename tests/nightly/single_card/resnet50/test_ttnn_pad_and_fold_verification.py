#!/usr/bin/env python3

import pytest
from loguru import logger

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import ttnn

# Add the models directory to path to import utility functions
# Go up 4 levels from tests/nightly/single_card/resnet50/ to reach tt-metal root
tt_metal_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
models_path = os.path.join(tt_metal_root, "models")
sys.path.append(models_path)
from utility_functions import pad_and_fold_conv_activation_for_unity_stride, pad_and_fold_conv_filters_for_unity_stride


def torch_fast_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden).flatten()
    calculated = torch.Tensor(calculated).flatten()
    if torch.any(torch.isinf(calculated)) or torch.any(torch.isnan(calculated)):
        logger.error("Output tensor contains inf or nan values")
        return False, 0.0
    cov_input = torch.concat([calculated, golden])
    calc_pcc = torch.corrcoef(cov_input)
    return calc_pcc >= pcc, calc_pcc


def check_with_fast_pcc_without_tensor_printout(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    pcc_passed, pcc_message = torch_fast_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message


def write_to_file(file_name, tensor):
    tensor = tensor.float()
    tensor = tensor.cpu().detach().numpy()
    with open(file_name, "w") as f:
        for i in range(1):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        # f.write(str(round(tensor[i][j][k][l]), 2) + " ")
                        f.write("{:.2f}".format(tensor[i][j][k][l]) + " ")
                    f.write("\n")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_ttnn_normal_vs_folded(device):
    """
    Direct comparison: Normal TTNN conv2d (stride=2) vs Folded TTNN conv2d (stride=1)
    Both use TTNN for fair comparison
    Input: [1, 3, 1024, 1024], Conv: kernel=7x7, stride=2, padding=3
    """
    print("\n" + "=" * 80)
    print("TTNN Normal vs Folded Convolution Test")
    print("Input: [1, 3, 1024, 1024], Conv: kernel=7x7, stride=2, padding=3")
    print("=" * 80)

    try:
        torch.manual_seed(42)

        # Parameters
        batch_size = 1
        in_channels = 3
        out_channels = 64
        input_h, input_w = 1024, 1024
        kernel_size = 7
        stride = 2
        padding = 3

        # Create input and weights
        # Use seed for reproducible results
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16).float() * 0.01
        conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16).float()
        conv_bias = torch.randn(out_channels, dtype=torch.bfloat16).float()

        print(f"Input shape: {input_tensor.shape}")
        print(f"Weight shape: {conv_weight.shape}")
        print(f"Bias shape: {conv_bias.shape}")

        # ============================================================================
        # Method 1: Normal TTNN Conv2d (stride=2)
        # ============================================================================
        print(f"\n{'='*50}")
        print("Method 1: Normal TTNN Conv2d (stride=2)")
        print(f"{'='*50}")

        # Convert input to TTNN format (NCHW -> NHWC)
        input_nhwc = torch.permute(input_tensor, (0, 2, 3, 1))
        ttnn_input_normal = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16)
        ttnn_input_normal = ttnn.to_device(ttnn_input_normal, device)

        # Convert weights and bias to TTNN
        ttnn_weight_normal = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
        ttnn_bias_normal = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

        # Conv config for normal conv
        conv_kwargs_normal = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "batch_size": batch_size,
            "input_height": input_h,
            "input_width": input_w,
            "kernel_size": (kernel_size, kernel_size),
            "stride": (stride, stride),
            "padding": (padding, padding),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat8_b,
                deallocate_activation=True,
                reallocate_halo_output=True,
                act_block_h_override=32,
            ),
        }

        # Prepare normal conv weights
        prepared_weight_normal = ttnn.prepare_conv_weights(
            weight_tensor=ttnn_weight_normal,
            weights_format="OIHW",
            input_memory_config=ttnn_input_normal.memory_config(),
            input_layout=ttnn_input_normal.get_layout(),
            has_bias=True,
            **conv_kwargs_normal,
            input_dtype=ttnn.bfloat8_b,
        )
        prepared_bias_normal = ttnn.prepare_conv_bias(
            bias_tensor=ttnn_bias_normal,
            input_memory_config=ttnn_input_normal.memory_config(),
            input_layout=ttnn_input_normal.get_layout(),
            **conv_kwargs_normal,
            input_dtype=ttnn.bfloat8_b,
        )

        prepared_weight_normal = ttnn.to_device(prepared_weight_normal, device)
        prepared_bias_normal = ttnn.to_device(prepared_bias_normal, device)

        # Run normal TTNN conv2d
        print("Running normal TTNN conv2d...")
        ttnn_output_normal, [out_h, out_w], [weight_tensor_device, bias_tensor_device] = ttnn.conv2d(
            input_tensor=ttnn_input_normal,
            weight_tensor=prepared_weight_normal,
            bias_tensor=prepared_bias_normal,
            **conv_kwargs_normal,
            compute_config=ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat8_b,
        )
        weight_tensor_host = ttnn.to_torch(weight_tensor_device)
        write_to_file("weight_tensor_normal.txt", weight_tensor_host.float())
        print(f"Normal TTNN output shape: {ttnn_output_normal.shape}")
        print(f"Output dimensions: h={out_h}, w={out_w}")

        # Convert to torch for comparison
        # TTNN outputs in format [1, 1, n*h*w, c], need to reshape to [n, c, h, w]
        ttnn_normal_torch = ttnn.to_torch(ttnn_output_normal)
        print(f"Raw TTNN normal output shape: {ttnn_normal_torch.shape}")
        print(f"Expected output size: batch={batch_size}, h={out_h}, w={out_w}, c={out_channels}")
        # Reshape from [1, 1, n*h*w, c] to [n, h, w, c] then permute to [n, c, h, w]
        ttnn_normal_torch = ttnn_normal_torch.reshape(batch_size, out_h, out_w, ttnn_normal_torch.shape[-1])
        ttnn_normal_torch = ttnn_normal_torch[:, :, :, :out_channels]  # Take only the required channels
        ttnn_normal_torch = torch.permute(ttnn_normal_torch, (0, 3, 1, 2))
        print(f"Normal output (NCHW): {ttnn_normal_torch.shape}")

        # ============================================================================
        # Method 2: Folded TTNN Conv2d (stride=1)
        # ============================================================================
        print(f"\n{'='*50}")
        print("Method 2: Folded TTNN Conv2d (stride=1)")
        print(f"{'='*50}")

        # Apply folding to input and weights
        print("Applying pad_and_fold to input...")
        folded_input = pad_and_fold_conv_activation_for_unity_stride(
            input_tensor, pad_h=padding, pad_w=padding, stride_h=stride, stride_w=stride
        )
        print(f"Folded input shape: {folded_input.shape}")

        print("Applying pad_and_fold to weights...")
        folded_weight = pad_and_fold_conv_filters_for_unity_stride(conv_weight, stride_h=stride, stride_w=stride)
        print(f"Folded weight shape: {folded_weight.shape}")

        # Convert folded tensors to TTNN format
        folded_input_nhwc = torch.permute(folded_input, (0, 2, 3, 1))
        ttnn_input_folded = ttnn.from_torch(folded_input_nhwc, dtype=ttnn.bfloat16)
        ttnn_input_folded = ttnn.to_device(ttnn_input_folded, device)

        ttnn_weight_folded = ttnn.from_torch(folded_weight, dtype=ttnn.bfloat16)
        ttnn_bias_folded = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

        # Conv config for folded conv (unity stride, no padding)
        conv_kwargs_folded = {
            "in_channels": folded_input.shape[1],  # Folded channels
            "out_channels": out_channels,
            "batch_size": batch_size,
            "input_height": folded_input.shape[2],  # Folded height
            "input_width": folded_input.shape[3],  # Folded width
            "kernel_size": (folded_weight.shape[2], folded_weight.shape[3]),
            "stride": (1, 1),  # Unity stride
            "padding": (0, 0),  # No padding
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat8_b,
                deallocate_activation=True,
                reallocate_halo_output=True,
                act_block_h_override=32,
            ),
        }

        print(f"Folded conv config:")
        print(f"  Input channels: {conv_kwargs_folded['in_channels']}")
        print(f"  Input size: {conv_kwargs_folded['input_height']}x{conv_kwargs_folded['input_width']}")
        print(f"  Kernel size: {conv_kwargs_folded['kernel_size']}")

        # Prepare folded conv weights
        prepared_weight_folded = ttnn.prepare_conv_weights(
            weight_tensor=ttnn_weight_folded,
            weights_format="OIHW",
            input_memory_config=ttnn_input_folded.memory_config(),
            input_layout=ttnn_input_folded.get_layout(),
            has_bias=True,
            **conv_kwargs_folded,
            input_dtype=ttnn.bfloat8_b,
        )
        prepared_bias_folded = ttnn.prepare_conv_bias(
            bias_tensor=ttnn_bias_folded,
            input_memory_config=ttnn_input_folded.memory_config(),
            input_layout=ttnn_input_folded.get_layout(),
            **conv_kwargs_folded,
            input_dtype=ttnn.bfloat8_b,
        )

        prepared_weight_folded = ttnn.to_device(prepared_weight_folded, device)
        prepared_bias_folded = ttnn.to_device(prepared_bias_folded, device)

        # Run folded TTNN conv2d
        print("Running folded TTNN conv2d...")
        ttnn_output_folded, [folded_out_h, folded_out_w], [weight_tensor_device, bias_tensor_device] = ttnn.conv2d(
            input_tensor=ttnn_input_folded,
            weight_tensor=prepared_weight_folded,
            bias_tensor=prepared_bias_folded,
            **conv_kwargs_folded,
            compute_config=ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat8_b,
        )
        weight_tensor_host = ttnn.to_torch(weight_tensor_device)
        write_to_file("weight_tensor_folder.txt", weight_tensor_host.float())
        print(f"Folded TTNN output shape: {ttnn_output_folded.shape}")
        print(f"Folded output dimensions: h={folded_out_h}, w={folded_out_w}")

        # Convert to torch for comparison
        # TTNN outputs in format [1, 1, n*h*w, c], need to reshape to [n, c, h, w]
        ttnn_folded_torch = ttnn.to_torch(ttnn_output_folded)
        print(f"Raw TTNN folded output shape: {ttnn_folded_torch.shape}")
        print(f"Expected folded output size: batch={batch_size}, h={folded_out_h}, w={folded_out_w}, c={out_channels}")
        # Reshape from [1, 1, n*h*w, c] to [n, h, w, c] then permute to [n, c, h, w]
        ttnn_folded_torch = ttnn_folded_torch.reshape(
            batch_size, folded_out_h, folded_out_w, ttnn_folded_torch.shape[-1]
        )
        ttnn_folded_torch = ttnn_folded_torch[:, :, :, :out_channels]  # Take only the required channels
        ttnn_folded_torch = torch.permute(ttnn_folded_torch, (0, 3, 1, 2))
        print(f"Folded output (NCHW): {ttnn_folded_torch.shape}")

        # ============================================================================
        # Method 3: PyTorch Conv2d Reference (stride=2) - COMMENTED OUT
        # ============================================================================
        # print(f"\n{'='*50}")
        # print("Method 3: PyTorch Conv2d Reference (stride=2)")
        # print(f"{'='*50}")
        #
        # # Run PyTorch conv2d as reference
        # torch_output = F.conv2d(
        #     input_tensor,
        #     conv_weight,
        #     bias=conv_bias,
        #     stride=stride,
        #     padding=padding
        # )
        # print(f"PyTorch output shape: {torch_output.shape}")

        # ============================================================================
        # Verification using PCC
        # ============================================================================
        print(f"\n{'='*50}")
        print("VERIFICATION: Normal TTNN vs Folded TTNN (PCC-based)")
        print(f"{'='*50}")

        print(f"Normal TTNN output shape: {ttnn_normal_torch.shape}")
        print(f"Folded TTNN output shape: {ttnn_folded_torch.shape}")

        # Convert to float32 for precise comparison
        normal_fp32 = ttnn_normal_torch.float()
        folded_fp32 = ttnn_folded_torch.float()

        # PCC verification with 0.99 threshold
        pcc_threshold = 0.99

        print(f"\n{'='*30}")
        print("PCC VERIFICATION RESULTS")
        print(f"{'='*30}")

        # PyTorch vs Normal TTNN - COMMENTED OUT
        # torch_normal_passing, torch_normal_pcc = check_with_fast_pcc_without_tensor_printout(
        #     torch_fp32, normal_fp32, pcc=pcc_threshold
        # )
        # print(f"PyTorch vs Normal TTNN:")
        # print(f"  PCC = {torch_normal_pcc:.6f}, Threshold = {pcc_threshold}")
        # print(f"  Result: {'✅ PASS' if torch_normal_passing else '❌ FAIL'}")

        # PyTorch vs Folded TTNN - COMMENTED OUT
        # torch_folded_passing, torch_folded_pcc = check_with_fast_pcc_without_tensor_printout(
        #     torch_fp32, folded_fp32, pcc=pcc_threshold
        # )
        # print(f"PyTorch vs Folded TTNN:")
        # print(f"  PCC = {torch_folded_pcc:.6f}, Threshold = {pcc_threshold}")
        # print(f"  Result: {'✅ PASS' if torch_folded_passing else '❌ FAIL'}")

        # Normal vs Folded TTNN
        normal_folded_passing, normal_folded_pcc = check_with_fast_pcc_without_tensor_printout(
            normal_fp32, folded_fp32, pcc=pcc_threshold
        )
        print(f"Normal TTNN vs Folded TTNN:")
        print(f"  PCC = {normal_folded_pcc:.6f}, Threshold = {pcc_threshold}")
        print(f"  Result: {'✅ PASS' if normal_folded_passing else '❌ FAIL'}")

        all_match = normal_folded_passing  # Only checking Normal vs Folded now

        if all_match:
            print("\n✅ SUCCESS: Normal TTNN and Folded TTNN produce equivalent results!")
        else:
            print(f"\n⚠️  WARNING: PCC check failed")

        # Sample comparison
        print(f"\nSample values comparison (first 3x3 from output[0,0,:,:]):")
        print("Normal TTNN:")
        print(normal_fp32[0, 0, :3, :3])
        print("Folded TTNN:")
        print(folded_fp32[0, 0, :3, :3])

        return all_match

    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    # Run direct comparison first
    print("Running direct comparison test...")
    direct_result = test_direct_comparison()

    # Run the new TTNN comparison test
    print("\nRunning TTNN normal vs folded test...")
    ttnn_comparison_result = test_ttnn_normal_vs_folded()

    # if direct_result:
    #     print("\nRunning full TTNN test...")
    #     main_result = test_ttnn_pad_and_fold_conv()
    # else:
    #     print("\nSkipping full TTNN test due to direct comparison failure")
    #     main_result = False

    # print("\n" + "="*80)
    # print("FINAL RESULT")
    # print("="*80)
    # print(f"Direct PyTorch comparison: {'✅ PASSED' if direct_result else '❌ FAILED'}")
    # print(f"TTNN normal vs folded: {'✅ PASSED' if ttnn_comparison_result else '❌ FAILED'}")
    # print(f"Full TTNN test: {'✅ PASSED' if main_result else '❌ FAILED'}")

    # if direct_result and ttnn_comparison_result:
    #     print("\n✅ SUCCESS: Pad-and-fold functions work correctly!")
    #     print("Both PyTorch and TTNN implementations produce equivalent results.")
    # elif direct_result:
    #     print("\n⚠️  PyTorch folding works, but TTNN implementation has issues.")
    # else:
    #     print("\n❌ FAILURE: Basic pad-and-fold logic is incorrect.")
    # print("="*80)
