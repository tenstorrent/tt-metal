# #File models/utility_functions.py for print all values with ULP > 1
# def comp_ulp_check(input, golden, calculated, ulp_threshold, allow_nonfinite=False):
#     """
#     Compute absolute error between two tensors in Units of Least Precision (ULP)
#     """

#     # If both tensors are empty, then we can return True
#     if torch.numel(golden) == 0 and torch.numel(calculated) == 0:
#         return True, "Both tensors are empty"

#     if not allow_nonfinite and not torch.all(torch.isfinite(calculated)):
#         return False, "Calculated tensor contains non-finite values"

#     if not _comp_nonfinite(golden, calculated):
#         return False, "Tensors are not finite at the same positions"
#     # nonfinite elments can intefere with ULP error calculation
#     # To avoid this, replace nan, +inf, -inf with 0
#     # (we have already checked that both tensors have the same nonfinite elements)
#     mask_finite = ~torch.isfinite(golden)
#     golden = golden.clone()
#     calculated = calculated.clone()
#     golden[mask_finite] = 0
#     calculated[mask_finite] = 0

#     # ULP is measured according to the golden tensor
#     # In most cases, data type of golden tensor should be the same as calculated tensor.
#     # However, in some cases, we may want to measure < 1 ULP differences, which requires golden tensor
#     # to have higher precision than calculated tensor.
#     # If we passed golden tensor to ulp() as is, we would get ULP of higher precision.
#     # e.g. ulp of float32 rather bfloat16 calculation, which would give us a wrong value.
#     ulp_value = ulp(golden.type(calculated.dtype))

#     if golden.dtype != calculated.dtype:  # Note: assumes that golden has higher precision than calculated tensor
#         calculated = calculated.type(golden.dtype)
#         ulp_value = ulp_value.type(golden.dtype)  # Convert ULP to higher precision (for sub-1 ULP measurements)

#     ULP_Cond = torch.abs(calculated - golden) / ulp_value
#     mask = ULP_Cond > 1.0

#     if mask.any():
#         indices = torch.nonzero(mask, as_tuple=False)
#         output_lines = []
#         output_lines.append(f"Found {indices.shape[0]} values with ULP > 1:\n")

#         seen_inputs = set()

#         for idx in indices:
#             idx_tuple = tuple(idx.tolist())
#             inp_val = input[idx_tuple].item()

#             if inp_val in seen_inputs:
#                 continue
#             seen_inputs.add(inp_val)

#             calc_val = calculated[idx_tuple].item()
#             golden_val = golden[idx_tuple].item()
#             ulp_val = ULP_Cond[idx_tuple].item()

#             line = f"Input: {inp_val}, " f"Calculated: {calc_val}, " f"Golden: {golden_val}, " f"ULP: {ulp_val}"
#             output_lines.append(line)

#         file_path = "ulp_mismatches.txt"
#         with open(file_path, "w") as f:
#             f.write("\n".join(output_lines))

#         print(f"\nSaved mismatch details to {file_path}")
#     else:
#         print("No values with ULP > 1 found.")

#     ulp_delta = torch.max(ULP_Cond)


# # Test file with arange testing
# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

# from loguru import logger
# import random
# import pytest
# import torch
# import ttnn
# import traceback
# from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_allclose
# from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
# from models.utility_functions import comp_ulp_custom


# def run_eltwise_selu_tests(
#     input_shape,
#     dtype,
#     dlayout,
#     in_mem_config,
#     output_mem_config,
#     data_seed,
#     device,
# ):
#     torch.manual_seed(data_seed)
#     x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

#     try:
#         # get ref result
#         ref_value = torch.nn.functional.selu(x)

#         x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

#         tt_result = ttnn.selu(x)
#         tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

#     except Exception as e:
#         logger.warning(f"Test execution crashed: {e}")
#         print(traceback.format_exc())
#         raise e

#     assert len(tt_result.shape) == len(ref_value.shape)
#     assert tt_result.shape == ref_value.shape
#     assert_with_pcc(ref_value, tt_result, 0.99)


# test_sweep_args = [
#     (
#         [(3, 2, 192, 32)],
#         [ttnn.bfloat16],
#         [ttnn.TILE_LAYOUT],
#         [ttnn.DRAM_MEMORY_CONFIG],
#         ttnn.L1_MEMORY_CONFIG,
#         6861134,
#     ),
#     (
#         [(12, 224, 224)],
#         [ttnn.bfloat8_b],
#         [ttnn.TILE_LAYOUT],
#         [ttnn.DRAM_MEMORY_CONFIG],
#         ttnn.L1_MEMORY_CONFIG,
#         6411147,
#     ),
#     (
#         [(3, 2, 191, 31)],
#         [ttnn.bfloat16],
#         [ttnn.TILE_LAYOUT],
#         [ttnn.DRAM_MEMORY_CONFIG],
#         ttnn.L1_MEMORY_CONFIG,
#         6861134,
#     ),
#     (
#         [(12, 225, 223)],
#         [ttnn.bfloat8_b],
#         [ttnn.TILE_LAYOUT],
#         [ttnn.DRAM_MEMORY_CONFIG],
#         ttnn.L1_MEMORY_CONFIG,
#         6411147,
#     ),
# ]


# @pytest.mark.parametrize(
#     "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
#     (test_sweep_args),
# )
# def test_eltwise_selu(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
#     run_eltwise_selu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)


# def test_selu_arange(device):
#     # Generate all possible bit pattersn for bf16
#     all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
#     input_tensor = all_bitpatterns.view(torch.bfloat16)
#     input_tensor = input_tensor.to(torch.float32)

#     # Mask for NaN
#     # Input: 3.2300240297573456e+38, Calculated: 3.3895313892515355e+38, Golden: 3.393789478323726e+38, ULP: inf
#     mask = (
#         torch.isnan(input_tensor)
#         | ((input_tensor >= -0.30859375) & (input_tensor <= 1.1663108012064884e-38))
#         | (input_tensor == 3.2300240297573456e38)
#         | (input_tensor == -0.0)
#     )
#     input_tensor[mask] = 1.0

#     tt_in = ttnn.from_torch(
#         input_tensor,
#         dtype=ttnn.bfloat16,
#         device=device,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )

#     golden_function = ttnn.get_golden_function(ttnn.selu)
#     golden = golden_function(input_tensor, device=device)

#     tt_result = ttnn.selu(tt_in)
#     result = ttnn.to_torch(tt_result)
#     comp_ulp_custom(input_tensor, golden, result, 1, allow_nonfinite=False)
#     assert_with_ulp(golden, result, 1, allow_nonfinite=False)


# @pytest.mark.parametrize(
#     "input_shapes",
#     (
#         (torch.Size([1, 1, 32, 32])),
#         (torch.Size([1, 2, 64, 120])),
#         (torch.Size([1, 3, 320, 320])),
#     ),
# )
# @pytest.mark.parametrize(
#     "low, high",
#     [
#         (-0.30859375, 1.1663108012064884e-38),
#     ],
# )
# def test_selu_atol(input_shapes, low, high, device):
#     # Max ATOL Delta: 0.0078125, Max RTOL Delta: 1.0
#     num_elements = torch.prod(torch.tensor(input_shapes)).item()
#     torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
#     torch_input = torch_input[:num_elements].reshape(input_shapes)

#     golden_function = ttnn.get_golden_function(ttnn.selu)
#     golden = golden_function(torch_input, device=device)

#     tt_in = ttnn.from_torch(
#         torch_input,
#         dtype=ttnn.bfloat16,
#         device=device,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )

#     tt_result = ttnn.selu(tt_in)
#     result = ttnn.to_torch(tt_result)

#     assert_allclose(tt_result, golden, rtol=1.0, atol=0.0078125)


# #Test file for graph plot
# import torch
# import matplotlib.pyplot as plt
# import os
# import ttnn
# import pandas as pd

# import models.utility_functions as util

# def test_selu_arange_plot(device, out_dir="plots"):
#     os.makedirs(out_dir, exist_ok=True)
#     all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
#     input_tensor = all_bitpatterns.view(torch.bfloat16).to(torch.float32)

#     # Mask NaNs
#     mask = torch.isnan(input_tensor)
#     input_tensor[mask] = 1.0

#     x_vals = input_tensor.cpu().numpy()

#     golden_function = ttnn.get_golden_function(ttnn.selu)
#     golden = golden_function(input_tensor, device=device)

#     tt_in = ttnn.from_torch(
#         input_tensor,
#         dtype=ttnn.bfloat16,
#         device=device,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     )
#     tt_result = ttnn.selu(tt_in)
#     result = ttnn.to_torch(tt_result)

#     # ULP
#     abs_error = torch.abs(golden - result)
#     ulp_spacing = util.ulp(golden.to(torch.bfloat16)).to(torch.float32)
#     ulp_error = abs_error / ulp_spacing

#     y_vals = ulp_error.cpu().numpy()

#     plt.figure(figsize=(12, 6))
#     plt.scatter(x_vals, y_vals, s=1, alpha=0.6)
#     plt.xlabel("BF16 Input Value (converted to FP32)")
#     plt.ylabel("ULP Error")
#     # plt.ylim(0, 5)
#     # plt.xlim(0, 5)
#     plt.title("SELU BF16 vs Golden: ULP Error per Input Value")
#     plt.grid(True, alpha=0.3)

#     out_file = os.path.join(out_dir, "selu_bf16_ulp_errors.png")
#     plt.savefig(out_file, dpi=300)
#     plt.close()
#     print(f"ULP scatter plot saved to: {out_file}")

#     df = pd.DataFrame({"input_value": x_vals, "ulp_error": y_vals})
#     csv_file = os.path.join(out_dir, "selu_bf16_ulp_errors.csv")
#     df.to_csv(csv_file, index=False)
#     print(f"ULP error data saved to: {csv_file}")

#     golden_vals = golden.to(torch.float32).cpu().numpy()
#     result_vals = result.to(torch.float32).cpu().numpy()

#     #TTNN vs Torch
#     plt.figure(figsize=(12, 6))

#     plt.scatter(x_vals, golden_vals, s=12, c="blue", label="PyTorch Golden", alpha=0.6, marker="x")
#     plt.scatter(x_vals, result_vals, s=12, c="orange", label="TTNN Result", alpha=0.6, marker="o")


#     plt.xlabel("Input (bfloat16 values)")
#     plt.ylabel("SELU Output")
#     plt.title("TTNN vs PyTorch (SELU outputs for all bf16 values)")
#     # plt.ylim(0, 5)
#     # plt.xlim(0, 5)
#     plt.legend()
#     save_path = os.path.abspath("ttnn_vs_torch_selu.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"Plot saved at: {save_path}")
