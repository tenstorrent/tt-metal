import ttnn
import torch
import math
import numpy as np
import time
import pandas as pd
import sys
import os

device_id = 0
device = ttnn.open_device(device_id=device_id)

ttnn.enable_program_cache(device)  # Useful: we are going to call the same kernel several times


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
    # Exponential functions
    "exp": (torch.exp, ttnn.exp, math.exp),
    "exp_approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
    ),
    "tanh": (
        torch.tanh,
        lambda x, output_tensor: ttnn.tanh(x),
        math.tanh,
    ),  # ttnn.tanh() does not support output_tensor ?
    "cosh": (
        torch.cosh,
        lambda x, output_tensor: ttnn.cosh(x),
        math.cosh,
    ),  # ttnn.cosh() does not support output_tensor ?
    "sinh": (
        torch.sinh,
        lambda x, output_tensor: ttnn.sinh(x),
        math.sinh,
    ),  # ttnn.sinh() does not support output_tensor ?
    # Logarithmic functions
    "log": (torch.log, ttnn.log, math.log),
    "log10": (torch.log10, ttnn.log10, math.log10),
    "log2": (torch.log2, ttnn.log2, math.log2),
    "log1p": (torch.log1p, ttnn.log1p, math.log1p),
    "logaddexp": (torch.logaddexp, ttnn.logaddexp, None),
    "logaddexp2": (torch.logaddexp2, ttnn.logaddexp2, None),
    # Activation functions
    "silu": (lambda x, out: torch.nn.SiLU()(x), ttnn.silu, None),
    "gelu": (lambda x, out: torch.nn.GELU()(x), ttnn.gelu, None),
    "logit": (torch.logit, lambda x, output_tensor: ttnn.logit(x), None),  # ttnn.logit does not support output_tensor ?
    "swish": (
        lambda x, out: torch.nn.SiLU()(x),
        lambda x, output_tensor: ttnn.swish(x),
        None,
    ),  # ttnn.swish does not support output_tensor ?
    "mish": (lambda x, out: torch.nn.Mish()(x), ttnn.mish, None),
    "elu": (
        lambda x, out: torch.nn.ELU()(x),
        lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0),
        None,
    ),  # Unlike torch, ttnn.elu does not use alpha=1 by default
    "selu": (
        lambda x, out: torch.nn.SELU()(x),
        lambda x, output_tensor: ttnn.selu(x),
        None,
    ),  # ttnn.selu does not support output_tensor ?
    "softplus": (lambda x, out: torch.nn.Softplus()(x), ttnn.softplus, None),
    "softsign": (
        lambda x, out: torch.nn.Softsign()(x),
        lambda x, output_tensor: ttnn.softsign(x),
        None,
    ),  # ttnn.softsign does not support output_tensor ?
    # Trigonometric functions
    "tan": (torch.tan, ttnn.tan, math.tan),
    "atan": (torch.atan, ttnn.atan, math.atan),
    "atan2": (torch.atan2, ttnn.atan2, math.atan2),
    "sin": (torch.sin, ttnn.sin, math.sin),
    "cos": (torch.cos, ttnn.cos, math.cos),
    # Miscellaneous functions
    "sqrt": (torch.sqrt, ttnn.sqrt, math.sqrt),
    "rsqrt": (torch.rsqrt, ttnn.rsqrt, None),
    "rsqrt_approx": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
    ),
    "digamma": (
        torch.digamma,
        lambda x, output_tensor: ttnn.digamma(x),
        None,
    ),  # ttnn.digamma does not support output_tensor ?
    "lgamma": (
        torch.lgamma,
        lambda x, output_tensor: ttnn.lgamma(x),
        math.lgamma,
    ),  # ttnn.lgamma does not support output_tensor ?
    "tanhshrink": (
        lambda x, out: torch.nn.Tanhshrink()(x),
        lambda x, output_tensor: ttnn.tanhshrink(x),
        None,
    ),  # ttnn.tan
}


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

    (torch_unary_op, ttnn_unary_op, python_unary_op) = operations_dict[operation_name]

    # Measurements
    x_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    y_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    yref_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    mse_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    max_abs_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    max_rel_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
    mean_rel_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)

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

        # Flatten tensors for data analsis (we only used 2D for ttnn and TILE_LAYOUT)
        np_flat_input = torch_input_f32.to(torch.float32).flatten().numpy()
        np_flat_output = actual_torch_output.to(torch.float32).flatten().numpy()
        np_flat_ref = torch_output_ref.to(torch.float32).flatten().numpy()
        # np_flat_exponent = torch_exponent.flatten().view(TORCH_TYPE).numpy()

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
            # TODO: Cast data to float32 or even float64 to reduce measurement errors
            # mse_value      = mse_loss(actual_sub_output, torch_sub_ref)
            np_diff_curated = np_diff[~np.isfinite(np_diff)]
            np_sub_ref_abs = np.abs(np_sub_ref[~np.isfinite(np_sub_ref)])

            if len(np_diff) > 0 and len(np_sub_ref_abs) > 0:
                # Reduces edge cases
                max_abs_error = np_diff_curated.max()
                max_rel_error = np.max(np_diff_curated / np_sub_ref_abs)  # Ignore NaN
                mean_rel_error = np.mean(np_diff_curated / np_sub_ref_abs)

            else:  # Batch only contains infinite value
                max_abs_error = np_diff.max()
                max_rel_error = np.max(np_diff / np.abs(np_sub_ref))  # Ignore NaN
                mean_rel_error = np.mean(np_diff / np.abs(np_sub_ref))

            # Write output data at given sub-batches / sub-samples
            x_array[res_i] = np_sub_input[0].item()
            y_array[res_i] = np_sub_output[0].item()
            yref_array[res_i] = np_sub_ref[0].item()
            # mse_array           [res_i] = mse_value.item()
            max_abs_error_array[res_i] = max_abs_error.item()
            max_rel_error_array[res_i] = max_rel_error.item()
            mean_rel_error_array[res_i] = mean_rel_error.item()

    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "mse": mse_array,
            "max_abs_error": max_abs_error_array,
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

    torch_output_ref = torch.zeros(size, dtype=TORCH_TYPE)
    ttnn_output = ttnn.zeros(size, dtype=TTNN_TYPE, device=device, layout=ttnn.TILE_LAYOUT)

    # Get the operations to test
    (torch_unary_op, ttnn_unary_op, python_unary_op) = operations_dict[operation_name]

    # Initialize arrays for measurements
    x_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)
    y_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)
    yref_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)
    max_abs_error_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)
    max_rel_error_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)
    mean_rel_error_array = np.zeros([sub_batches], dtype=NUMPY_TYPE)

    start_time = time.time()

    # Launch TTNN operation
    def launch_ttnn_op(torch_tensor, ttnn_unary, ttnn_output):
        ttnn_value = ttnn.from_torch(torch_tensor, device=device, dtype=TTNN_TYPE, layout=ttnn.TILE_LAYOUT)
        ttnn_output = ttnn_unary(ttnn_value, output_tensor=ttnn_output)
        return ttnn.to_torch(ttnn_output)

    # Run reference and actual operations
    torch_output_ref = torch_unary_op(torch_input_bf16, out=torch_output_ref)
    actual_torch_output = launch_ttnn_op(torch_input_bf16, ttnn_unary_op, ttnn_output)

    # Flatten tensors for analysis
    np_flat_input = torch_input_bf16.to(torch.float32).flatten().numpy()
    np_flat_output = actual_torch_output.to(torch.float32).flatten().numpy()
    np_flat_ref = torch_output_ref.to(torch.float32).flatten().numpy()

    # Process each sub-batch
    for j in range(0, sub_batches):
        chunk_size = TENSOR_WIDTH * TENSOR_HEIGHT // sub_batches
        (beg_index, end_index) = (j * chunk_size, (j + 1) * chunk_size)

        # Get sub-range
        np_sub_input = np_flat_input[beg_index:end_index]
        np_sub_output = np_flat_output[beg_index:end_index]
        np_sub_ref = np_flat_ref[beg_index:end_index]

        # Calculate errors
        np_diff = np.abs(np_sub_ref - np_sub_output)
        np_sub_ref_abs = np.abs(np_sub_ref)

        # Handle edge cases
        finite_mask = np.isfinite(np_diff) & np.isfinite(np_sub_ref_abs)
        if np.any(finite_mask):
            max_abs_error = np.max(np_diff[finite_mask])
            max_rel_error = np.max(np_diff[finite_mask] / np_sub_ref_abs[finite_mask])
            mean_rel_error = np.mean(np_diff[finite_mask] / np_sub_ref_abs[finite_mask])
        else:
            max_abs_error = np.max(np_diff)
            max_rel_error = np.max(np_diff / np_sub_ref_abs)
            mean_rel_error = np.mean(np_diff / np_sub_ref_abs)

        # Store results
        x_array[j] = np_sub_input[0].item()
        y_array[j] = np_sub_output[0].item()
        yref_array[j] = np_sub_ref[0].item()
        max_abs_error_array[j] = max_abs_error.item()
        max_rel_error_array[j] = max_rel_error.item()
        mean_rel_error_array[j] = mean_rel_error.item()

    # Create and save DataFrame
    accuracy_df = pd.DataFrame(
        {
            "base_x": x_array,
            "base_y": y_array,
            "base_yref": yref_array,
            "max_abs_error": max_abs_error_array,
            "max_rel_error": max_rel_error_array,
            "mean_rel_error": mean_rel_error_array,
        }
    )
    accuracy_df["operation"] = operation_name
    accuracy_df["dtype"] = "bfloat16"

    accuracy_df.to_csv(f"{dest_dir}/{operation_name}-bfloat16-[{group_size}].csv", na_rep="NaN", index_label="index")

    end_time = time.time()
    elapsed_s = end_time - start_time
    print(f"{operation_name} [bfloat16] Duration = {elapsed_s}s")


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

    # Unused: atan2, logaddexp, logaddexp2
    all_operations = [
        "exp",
        "exp_approx",
        "log",
        "tanh",
        "cosh",
        "sinh",
        "log10",
        "log2",
        "log1p",
        "silu",
        "gelu",
        "logit",
        "swish",
        "mish",
        "elu",
        "selu",
        "softplus",
        "softsign",
        "tan",
        "atan",
        "sin",
        "cos",
        "sqrt",
        "rsqrt",
        "rsqrt_approx",
        "digamma",
        "lgamma",
        "tanhshrink",
    ]

    highres_operations = set(
        [
            "exp",
            "exp_approx",
            "log",
            "log10",
            "log2",
            "log1p",
            "tanh",
            "cosh",
            "sinh",
            "tan",
            "atan",
            "cos",
            "sin",
            "silu",
            "gelu",
            "logit",
            "swish",
            "mish",
            "elu",
            "selu",
            "softplus",
            "softsign",
            "digamma",
            "lgamma",
            "tanhshrink",
            "sqrt",
            "rsqrt",
            "rsqrt_approx",
        ]
    )
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
        print(f"Running operation {operation}  #{cnt} / {total_operation_cnt}", end="\r")
        try:
            measure_op_accuracy_bf16(operation, dest_dir, group_size=32)
            success_count += 1
            successfull_operations += [operation]
        except Exception as e:
            print(f"{RED}Could not run operation {operation}: {e}{RESET}")
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
            print(f"{RED}Could not run operation {operation}: {e}{RESET}")
            failed_operations += [f"{operation}[highres]"]

    print(f"Sucessfully ran {success_count} / {len(all_operations)} operations")
    print(f"{GREEN}SUCCESS: {successfull_operations}{RESET}")
    print(f"{RED}FAILED: {failed_operations}{RESET}")


args = sys.argv
main(args)

ttnn.close_device(device)
