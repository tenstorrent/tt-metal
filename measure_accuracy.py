import ttnn
import torch
import math
import numpy as np
import time
import pandas as pd


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

target_type = "bfloat16"
operation_name = "tan"


parameters = datatypes_parameters[target_type]


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


def exp_ttnn(tensor_input, output_tensor):
    return ttnn.exp(tensor_input, output_tensor=output_tensor)


def exp_ttnn_approx(tensor_input, output_tensor):
    return ttnn.exp(tensor_input, fast_and_approximate_mode=True, output_tensor=output_tensor)


def log_ttnn(tensor_input, output_tensor):
    return ttnn.log(tensor_input, output_tensor=output_tensor)


def tan_ttnn(tensor_input, output_tensor):
    return ttnn.tan(tensor_input, output_tensor=output_tensor)


def silu_ttnn(tensor_input, output_tensor):
    return ttnn.silu(tensor_input, output_tensor=output_tensor)


def silu_torch(tensor_input, out):
    silu = torch.nn.SiLU()
    return silu(tensor_input)  # Does not support output tensor


operations_dict = {
    "exp": (torch.exp, exp_ttnn, math.exp),
    "exp_approx": (torch.exp, exp_ttnn_approx, None),
    "log": (torch.log, log_ttnn, math.log),
    "silu": (silu_torch, silu_ttnn, None),
    "tan": (torch.tan, tan_ttnn, None),
}


(torch_unary_op, ttnn_unary_op, python_unary_op) = operations_dict[operation_name]


# For each tensor, pick several values:
# e.g. for sub_batch=4, pick, for each i (2**i, 2**i + 2**i/4, 2**i + 2**i/2, 2**i + 3*2**i/4)
# Within the tensor (2**11, 2**13), these would be indices [(0, 0), (2**9, 0), (2**10), (2**10 + 2**10, 0)]
# Or, with a general formula: [(0, 0), (TENSOR_HEIGHT//4, 0), (TENSOR_HEIGHT//2, 0), (3*TENSOR_HEIGHT//4, 0)]
sub_batches = 4

# Measurements
x_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
y_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
yref_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
mse_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
max_abs_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
max_rel_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)
mean_rel_error_array = np.zeros([repeats * sub_batches], dtype=NUMPY_TYPE)


# repeats = 10
for i in range(0, repeats):
    print(f"Iteration #{i} / {repeats}")

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
accuracy_df["dtype"] = target_type

accuracy_df.to_csv(f"accuracy-{operation_name}-{target_type}.csv", na_rep="NaN", index_label="index")

end_time = time.time()
elapsed_s = end_time - start_time
elapsed_ms = (elapsed_s) * 1000
print(f"Duration = {elapsed_s}s, {elapsed_ms/repeats} ms/iteration")

ttnn.close_device(device)
