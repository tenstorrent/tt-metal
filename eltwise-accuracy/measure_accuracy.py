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
from operations import UNARY_OPERATIONS

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


TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"


def exp_regression_0p5_to_1(tensor_input):
    tensor_constA = torch.full(tensor_input.size(), 0.863281, dtype=torch.bfloat16)
    tensor_const1 = torch.full(tensor_input.size(), 1.0, dtype=torch.bfloat16)
    tensor_const0p8373 = torch.full(tensor_input.size(), 0.837300003, dtype=torch.bfloat16)

    # print(f"tensor_constA =\n{tensor_constA}")
    # print(f"tensor_const1 =\n{tensor_const1}")
    # print(f"tensor_const0p8373 =\n{tensor_const0p8373}")

    # Linear regression approximation of torch.exp on [0.5; 1]
    # y = x * (0.8373 * x + 0.863281) + 1
    tensor_tmp = torch.add(torch.mul(tensor_input, tensor_const0p8373), tensor_constA)
    # print(f"tensor_tmp =\n{tensor_tmp}")
    tensor_val = torch.add(torch.mul(tensor_input, tensor_tmp), tensor_const1)

    return tensor_val


def exp_regression_0p5_to_1_alt1(tensor_input):
    fun = lambda x: 1.1173422532270 + 0.535411571001 * x + 1.063234614758 * x**2

    tensor_output = tensor_input.clone()
    tensor_output = tensor_output.apply_(fun)

    return tensor_output


def exp_accurate_python(ttnn_input, output_tensor, exp_regression=exp_regression_0p5_to_1):
    # setsgn => get sign of input and then multiply input by it to make numbers positive
    # this shouldn't alter accuracy as -1 and 1 are exact and so are their multiples
    torch.set_printoptions(precision=6)

    input_tensor = ttnn.to_torch(ttnn_input)

    tensor_positive_val = input_tensor.clone()

    tensor_positive_val = utils.setsgn_bf16_(tensor_positive_val, 0)
    tensor_exponent = utils.exexp_bf16(tensor_positive_val)

    # print(f"tensor_positive_val =\n{tensor_positive_val}")
    # print(f"tensor_exponent =\n{tensor_exponent}")
    mask_positive_exponents = tensor_exponent >= 0

    tensor_input_adjusted_exponent = tensor_positive_val.clone()
    tensor_input_adjusted_exponent = utils.setexp_bf16_(tensor_input_adjusted_exponent, -1)
    tensor_positive_val = torch.where(mask_positive_exponents, tensor_input_adjusted_exponent, tensor_positive_val)

    # print(f"tensor_positive_val =\n{tensor_positive_val}")

    tensor_positive_val = exp_regression(tensor_positive_val)

    # print(f"tensor regression 0 =\n{tensor_positive_val}")

    # Expand values with squaring method
    mask_positive_exponent = tensor_exponent >= 0

    tensor_square = torch.mul(tensor_positive_val, tensor_positive_val)
    tensor_positive_val = torch.where(mask_positive_exponent, tensor_square, tensor_positive_val)

    for s_iter in range(0, 7):
        tensor_exponent = torch.where(mask_positive_exponent, tensor_exponent - 1, tensor_exponent)

        # print(f"[{s_iter}] tensor_positive_val =\n{tensor_positive_val}")

        mask_positive_exponent = (tensor_exponent >= 0) & mask_positive_exponent
        tensor_square = torch.mul(tensor_positive_val, tensor_positive_val)
        tensor_positive_val = torch.where(mask_positive_exponent, tensor_square, tensor_positive_val)

    tensor_out = tensor_positive_val
    mask_negative_input = input_tensor < 0
    tensor_reciprocal = torch.reciprocal(tensor_out)
    tensor_out = torch.where(mask_negative_input, tensor_reciprocal, tensor_out)

    output_tensor = ttnn.from_torch(tensor_out, device=device, dtype=ttnn_input.dtype, layout=ttnn_input.layout)

    return output_tensor


def test_exp_accurate_python():
    tensor_input = torch.from_numpy(np.array([4.25], dtype=np.float32)).type(torch.bfloat16)
    ttnn_input = ttnn.from_torch(tensor_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tensor_output = None
    print(f"tensor_input =\n{tensor_input}")

    tensor_output = exp_accurate_python(ttnn_input, tensor_output)

    ttnn_output = ttnn.exp(ttnn_input)

    tensor_output_torch = ttnn.to_torch(tensor_output)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    print(f"torch.exp() =\n{torch.exp(tensor_input)}")
    print(f"tensor_output =\n{tensor_output_torch}")
    print(f"ttnn_output =\n{ttnn_output_torch}")


# Add powers of [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10] into dictionary
exponents = [2, 3, 4, 5, 6, 7, 8, 9, 10]

operations_dict = UNARY_OPERATIONS

# a**x functions
for exponent in exponents:
    operations_dict[f"pow_{exponent}"] = (
        lambda x, out, e=exponent: torch.pow(e, x),
        lambda x, output_tensor, e=exponent: ttnn.pow(e, x, output_tensor=output_tensor),
        None,
        "pow",
    )
    operations_dict[f"pow21f_{exponent}"] = (
        lambda x, out, e=exponent: torch.pow(e, x),
        lambda x, output_tensor, e=exponent: ttnn.pow(e, x, output_tensor=output_tensor),
        None,
        "pow",
    )

operations_dict["pow21f_tiny"] = (
    lambda x, out, e=0.000001: torch.pow(e, x),
    lambda x, output_tensor, e=0.000001: ttnn.pow(e, x, output_tensor=output_tensor),
    None,
    "pow",
)

# x**a functions
powers = [0, 0.5, 1, 2, 5, 7, 10]
for power in powers:
    operations_dict[f"pow(x,{power})"] = (
        lambda x, out, p=power: torch.pow(x, p),
        lambda x, output_tensor, p=power: ttnn.pow(x, p, output_tensor=output_tensor),
        None,
        "pow",
    )
    operations_dict[f"pow21f(x,{power})"] = (
        lambda x, out, p=power: torch.pow(x, p),
        lambda x, output_tensor, p=power: ttnn.pow(x, p, output_tensor=output_tensor),
        None,
        "pow",
    )


def measure_op_accuracy(operation_name, target_dtype, dest_dir, samples=None):
    parameters = datatypes_parameters[target_dtype]

    # For each tensor, pick several values:
    # e.g. for sub_batch=4, pick, for each i (2**i, 2**i + 2**i/4, 2**i + 2**i/2, 2**i + 3*2**i/4)
    # Within the tensor (2**11, 2**13), these would be indices [(0, 0), (2**9, 0), (2**10), (2**10 + 2**10, 0)]
    # Or, with a general formula: [(0, 0), (TENSOR_HEIGHT//4, 0), (TENSOR_HEIGHT//2, 0), (3*TENSOR_HEIGHT//4, 0)]
    sub_batches = 4
    if samples is not None:
        sub_batches = samples  # TODO: Compute sub_batches from samples instead of just copying data

    TENSOR_WIDTH = parameters["tensor_width"]
    TENSOR_HEIGHT = parameters["tensor_height"]

    SIGN_BITS = parameters["sign_bits"]  # should be 1
    EXPONENT_BITS = parameters["exponent_bits"]
    MANTISSA_BITS = parameters["mantissa_bits"]

    NUMPY_TYPE = parameters["numpy_type"]
    NUMPY_INT_TYPE = parameters["numpy_int_type"]
    TORCH_TYPE = parameters["torch_type"]
    TORCH_INT_TYPE = parameters["torch_int_type"]
    TTNN_TYPE = parameters["ttnn_type"]

    # Tile layout seem to be the main ttnn data layout
    # We could keep data 1D, but with Tile layout, tiles would mostly contain padding data
    # By having 2D tensors, we maximize the filling of each tile
    size = [TENSOR_HEIGHT, TENSOR_WIDTH]

    repeats = 2**EXPONENT_BITS * 2**SIGN_BITS  # sign + exp

    # Use integer => we build floating point numbers exhaustively using their bianry representation
    input_np = np.arange(0, 2**MANTISSA_BITS, dtype=NUMPY_INT_TYPE)

    (host_dtype, dev_dtype) = (TORCH_TYPE, TTNN_TYPE)

    # Create input tensors
    torch_mantissa = torch.from_numpy(input_np).reshape(size)

    torch_exponent = torch.zeros(size, dtype=TORCH_INT_TYPE)
    torch_value = torch.zeros(size, dtype=TORCH_INT_TYPE)
    torch_output_ref = torch.zeros(size, dtype=TORCH_TYPE)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    mse_loss = torch.nn.MSELoss()

    start_time = time.time()

    # Define operations to run

    (torch_unary_op, ttnn_unary_op, python_unary_op, parent_op) = operations_dict[operation_name]

    # Measurements

    [
        x_array,
        y_array,
        yref_array,
        mse_array,
        max_abs_error_array,
        mean_abs_error_array,
        max_rel_error_array,
        mean_rel_error_array,
    ] = [np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE) for _ in range(8)]

    for i in range(0, repeats):
        print(f"{operation_name} [{target_dtype}] iteration #{i} / {repeats}", end="\r")

        # Compute exponent / bit using integer operations
        # Here, we build the binary representation of a set of floating point numbers
        # With this pattern, we have a tensor that contains a set of contiguous floating point numbers
        # All floating point numbers (`torch_input_f32`) will share the same exponent & sign but will have unique mantissa

        torch.full(size, i, dtype=TORCH_INT_TYPE, out=torch_exponent)  # exp bits = i
        torch.bitwise_left_shift(torch_exponent, MANTISSA_BITS, out=torch_exponent)
        torch.bitwise_or(torch_exponent, torch_mantissa, out=torch_value)  # combine sign/exponent with mantissa

        torch_input_f32 = torch_value.view(TORCH_TYPE)  # reinterpret data as float32

        # print(f"Torch Value =\n{torch_value}, size = {torch_value.size()}")
        # print(f"Torch Value f32 =\n{torch_value_f32}, size = {torch_value_f32.size()}")

        # Launch a TTNN operation from a torch tensor and returns output in torch tensor
        def launch_ttnn_op(torch_tensor, ttnn_unary, ttnn_output):
            ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)

            ttnn_output = ttnn_unary(ttnn_value, output_tensor=ttnn_output)

            # Convert back to torch
            torch_output = ttnn.to_torch(ttnn_output)

            return torch_output

        def launch_scalar_op(torch_tensor, python_unary):
            np_input = torch_tensor.to(torch.float32).flatten().numpy()

            def run_unary_op(x):
                try:
                    return python_unary(x)
                except:
                    return math.nan

            np_output = np.vectorize(run_unary_op)(np_input)

            torch_output = torch.from_numpy(np_output).to(TORCH_TYPE).reshape(size)
            return torch_output

        # Run operation
        torch_output_ref = torch_unary_op(torch_input_f32, out=torch_output_ref)

        if True:
            actual_torch_output = launch_ttnn_op(torch_input_f32, ttnn_unary_op, ttnn_output)
        else:  # Launch scalar op (used to evaluate accuracy of torch)
            actual_torch_output = launch_scalar_op(torch_input_f32, python_unary_op)

        # Flatten tensors for data analysis (we only used 2D for ttnn and TILE_LAYOUT)
        np_flat_input = torch_input_f32.to(torch.float32).flatten().numpy()
        np_flat_output = actual_torch_output.to(torch.float32).flatten().numpy()
        np_flat_ref = torch_output_ref.to(torch.float32).flatten().numpy()
        # np_flat_exponent = torch_exponent.flatten().view(TORCH_TYPE).numpy()

        # TODO: Just switch to pandas and do groupby() & cie
        for j in range(0, sub_batches):
            chunk_size = TENSOR_WIDTH * TENSOR_HEIGHT // sub_batches
            (beg_index, end_index) = (j * chunk_size, (j + 1) * chunk_size)
            res_i = i * sub_batches + j

            # TODO: Handle NaN/inf

            # Get sub-range
            np_sub_input = np_flat_input[beg_index:end_index]
            np_sub_output = np_flat_output[beg_index:end_index]
            np_sub_ref = np_flat_ref[beg_index:end_index]

            # Measure abs error
            np_diff = np.abs(np_sub_ref - np_sub_output)

            # Compare actual and expected output
            # mse_value      = mse_loss(actual_sub_output, torch_sub_ref)
            np_diff_curated = np_diff[~np.isfinite(np_diff)]
            np_sub_ref_abs = np.abs(np_sub_ref[~np.isfinite(np_sub_ref)])

            if len(np_diff) > 0 and len(np_sub_ref_abs) > 0:
                # Reduces edge cases
                max_abs_error = np_diff_curated.max()
                mean_abs_error = np_diff_curated.mean()
                rel_error = np_diff_curated / max(np_sub_ref_abs.max(), EPSILON)
                max_rel_error = np.max(rel_error)  # Ignore NaN
                mean_rel_error = np.mean(rel_error)

            else:  # Batch only contains infinite value
                max_abs_error = np_diff.max()
                mean_abs_error = np_diff.mean()
                max_rel_error = np.max(np_diff / np.abs(np_sub_ref))  # Ignore NaN
                mean_rel_error = np.mean(np_diff / np.abs(np_sub_ref))

            # Write output data at given sub-batches / sub-samples
            x_array[res_i] = np_sub_input[0].item()
            y_array[res_i] = np_sub_output[0].item()
            yref_array[res_i] = np_sub_ref[0].item()
            # mse_array           [res_i] = mse_value.item()
            max_abs_error_array[res_i] = max_abs_error.item()
            mean_abs_error_array[res_i] = mean_abs_error.item()
            max_rel_error_array[res_i] = max_rel_error.item()
            mean_rel_error_array[res_i] = mean_rel_error.item()

    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "mse": mse_array,
            "max_abs_error": max_abs_error_array,
            "mean_abs_error": mean_abs_error_array,
            "max_rel_error": max_rel_error_array,
            "mean_rel_error": mean_rel_error_array,
        }
    )
    accuracy_df["operation"] = operation_name
    accuracy_df["dtype"] = target_dtype

    accuracy_df.to_csv(f"{dest_dir}/{operation_name}-{target_dtype}-[{samples}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    elapsed_ms = (elapsed_s) * 1000
    print(f"Duration = {elapsed_s}s, {elapsed_ms/repeats} ms/iteration")


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
    torch_input_f64 = torch_input_bf16.to(torch.float64)  # Convert to float64 for torch golden function

    torch_output_ref = torch.zeros(size, dtype=torch.float64)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    # Get the operations to test
    (torch_unary_op, ttnn_unary_op, python_unary_op, parent_op) = operations_dict[operation_name]

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
    def launch_ttnn_op(torch_tensor, ttnn_unary, ttnn_output):
        ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
        ttnn_output = ttnn_unary(ttnn_value, output_tensor=ttnn_output)
        return ttnn.to_torch(ttnn_output)

    # Run reference and actual operations
    torch_golden_f64 = torch_unary_op(torch_input_f64, out=torch_output_ref)
    torch_ttnn_output_bf16 = launch_ttnn_op(torch_input_bf16, ttnn_unary_op, ttnn_output)

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

    finite_mask = np.isfinite(np_golden_f64) & np.isfinite(np_ttnn_output_f64)  # Don't compute PCC on NaN and infinity
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

    accuracy_df.to_csv(f"{dest_dir}/{operation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"{operation_name} [bfloat16] PCC = {pcc[0]}, Duration = {elapsed_s:.4f}s")


def main(args):
    dest_dir = "accuracy_results/results/"
    if not os.path.exists(dest_dir):  # TODO: Check if recursive
        os.makedirs(dest_dir)

    # Set numpy floating point warning to reduce stdout clutter
    # Since we test *all* possible floating point values, invalid values
    # are expected.
    # TODO: Log warnings into file
    np.seterr(divide="ignore")
    np.seterr(invalid="ignore")
    np.seterr(over="ignore")

    # Add powers into operations
    powers_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    powers = [f"pow_{power}" for power in powers_vals]

    # Unused: atan2, logaddexp, logaddexp2
    all_operations = [
        # "exp",
        # "exp_hybrid",
        # "exp_cond",
        # "exp_approx",
        # "exp_approx0",
        # "exp_approx_21f",
        # "exp_accurate_python",
        # "exp_python_alt1",
        # "pow_2",
        # "pow_3",
        # "pow_5",
        # "pow_10",
        "pow21f_tiny",
        # "pow21f_2",
        # "pow21f_3",
        # "pow21f_5",
        # "pow21f_10",
        # "pow(x,-10)",
        # "pow(x,-2)",
        # "pow(x,-1)",
        # "pow(x,0)",
        # "pow(x,2)",
        # "pow(x,5)",
        # "pow(x,10)",
        # "pow(x,0)",
        # "pow(x,0.5)",
        # "pow(x,2)",
        # "pow(x,5)",
        # "pow(x,10)",
        # "pow21f(x,0)",
        # "pow21f(x,0)",
        # "pow21f(x,0.5)",
        # "pow21f(x,2)",
        # "pow21f(x,5)",
        # "pow21f(x,10)",
        # "log",
        # "tanh",
        # "tanh_accurate",
        # "cosh",
        # "sinh",
        # "log10",
        # "log2",
        # "log1p",
        # "silu",
        # "gelu",
        # "gelu_approx",
        # "logit",
        # "swish",
        # "mish",
        # "elu",
        # "selu",
        # "softplus",
        # "softsign",
        # "tan",
        # "atan",
        # "sin",
        # "cos",
        # "sqrt",
        # "rsqrt",
        # "rsqrt_approx",
        # "reciprocal",
        # "digamma",
        # "lgamma",
        # "tanhshrink",
    ]

    highres_operations = [
        # "exp_hybrid",
        # "exp",
        # "exp_cond",
        # "exp_approx_21f",
        # "exp_approx",
        # "exp_approx0",
        # "exp_accurate_python",
        # "exp_python_alt1",
        "pow21f_tiny",
        # "pow_2",
        # "pow_3",
        # "pow_5",
        # "pow_10",
        # "pow21f_2",
        # "pow21f_3",
        # "pow21f_5",
        # "pow21f_10",
        # "pow(x,-10)",
        # "pow(x,-2)",
        # "pow(x,-1)",
        # "pow(x,0)",
        # "pow(x,2)",
        # "pow(x,5)",
        # "pow(x,10)",
        # "pow(x,0)",
        # "pow(x,0.5)",
        # "pow(x,2)",
        # "pow(x,5)",
        # "pow(x,10)",
        # "pow21f(x,0)",
        # "pow21f(x,0.5)",
        # "pow21f(x,2)",
        # "pow21f(x,5)",
        # "pow21f(x,10)",
        # "log",
        # "log10",
        # "log2",
        # "log1p",
        # "tanh",
        # "tanh_accurate",
        # "cosh",
        # "sinh",
        # "tan",
        # "atan",
        # "cos",
        # "sin",
        # "silu",
        # "gelu",
        # "gelu_approx",
        # "logit",
        # "swish",
        # "mish",
        # "elu",
        # "selu",
        # "softplus",
        # "softsign",
        # "sqrt",
        # "rsqrt",
        # "rsqrt_approx",
        # "reciprocal",
        # "digamma",
        # "lgamma",
        # "tanhshrink",
    ]

    # all_operations += powers
    # highres_operations += powers

    success_count = 0
    successfull_operations = []
    failed_operations = []

    cnt = 0
    total_operation_cnt = len(all_operations) + len(highres_operations)
    for operation in all_operations:
        cnt += 1
        print(f"Running operation {operation}  #{cnt} / {total_operation_cnt}", end="\r")
        try:
            measure_op_accuracy_bf16(operation, dest_dir, group_size=32)
            success_count += 1
            successfull_operations += [operation]
        except Exception as e:
            print(f"{TERM_RED}Could not run operation {operation}: {e}{TERM_RESET}")
            print(f"{TERM_RED}{traceback.format_exc()}{TERM_RESET}")
            failed_operations += [operation]

    print(f"Now measuring high-resolution operations")
    for operation in highres_operations:
        cnt += 1
        print(f"Running operation {operation} [highres] #{cnt}/{total_operation_cnt}", end="\r")
        try:
            measure_op_accuracy_bf16(operation, dest_dir, group_size=1)
            success_count += 1
            successfull_operations += [f"{operation}[highres]"]
        except Exception as e:
            print(f"{TERM_RED}Could not run operation {operation}: {e}{TERM_RESET}")
            print(f"{TERM_RED}{traceback.format_exc()}{TERM_RESET}")
            failed_operations += [f"{operation}[highres]"]

    print(f"Sucessfully ran {success_count} / {len(all_operations)} operations")
    print(f"{TERM_GREEN}SUCCESS: {successfull_operations}{TERM_RESET}")
    print(f"{TERM_RED}FAILED: {failed_operations}{TERM_RESET}")


args = sys.argv
main(args)

# test_exp_accurate_python()

ttnn.close_device(device)
