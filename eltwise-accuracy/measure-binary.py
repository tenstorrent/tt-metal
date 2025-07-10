import ttnn
import torch
import pandas as pd
import sys
import os

import utils
from utils import TERM_RED, TERM_GREEN, TERM_RESET

from models.utility_functions import ulp
import operations

device = ttnn.open_device(device_id=0)


# Take 3D tensor and return 1D tensor
# Input: [batch, rows, cols]
# Output: [rows]
def reduce_batch_and_cols(tensor):
    (tensor_tmp, _) = torch.max(tensor, dim=0)
    (tensor_max, _) = tensor_tmp.max(dim=1)
    return tensor_max


def reduce_on_batch_and_cols(tensor):
    # Compute max
    (tensor_tmp, _) = torch.max(tensor, dim=0)
    (tensor_max, _) = torch.max(tensor_tmp, dim=1)

    # Compute min
    (tensor_tmp, _) = torch.min(tensor, dim=0)
    (tensor_min, _) = torch.min(tensor_tmp, dim=1)

    # Compute average
    tensor_tmp = torch.mean(tensor, dim=0)
    tensor_mean = torch.mean(tensor_tmp, dim=1)

    return {"min": tensor_min, "max": tensor_max, "mean": tensor_mean}


def convert_to_ttn(torch_tensor):
    # Shard data on all cores to speed-up computation
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.Layout.TILE, device=device)

    return ttnn_tensor


def bench_binary_op(operation_name, dest_dir):
    df_all_results = pd.DataFrame()
    batch_size = 128

    operation = operations.BINARY_OPERATIONS[operation_name]
    ttnn_op = operation["ttnn"]
    torch_op = operation["torch"]

    i = 0
    for tensor_a, tensor_b in utils.generate_binary_tensors_bf16():
        print(f"Iteration = {i}")
        ttnn_tensor_a = convert_to_ttn(tensor_a)
        ttnn_tensor_b = convert_to_ttn(tensor_b)

        # Run OP
        golden_result = torch_op(tensor_a.to(torch.float32), tensor_b.to(torch.float32))
        ttnn_result = ttnn_op(ttnn_tensor_a, ttnn_tensor_b)

        torch_result = ttnn.to_torch(ttnn_result).to(torch.float32)

        # Compute ULP error
        golden_ulp = ulp(golden_result.to(torch.bfloat16)).to(torch.float32)

        golden_result_f32 = golden_result.to(torch.float32)
        ulp_delta = torch.abs(torch_result - golden_result_f32) / golden_ulp

        # Reduce values to same mantissa
        # This should give 1D tensors with 2**9 elements,
        # Each elements is the min/max/mean/... error for a pair of (mantissa_a, mantissa_b)
        ulp_batch = reduce_on_batch_and_cols(ulp_delta)

        rows = 2**9

        # print(f"ULP Batch[min]: {ulp_batch['min']}, size = {ulp_batch['min'].size()}")
        assert ulp_batch["min"].size() == torch.Size([rows])
        assert ulp_batch["max"].size() == torch.Size([rows])
        assert ulp_batch["mean"].size() == torch.Size([rows])

        # We must reduce on batch + columns
        series_a_reduced = reduce_batch_and_cols(tensor_a.to(torch.float32))  # Note: 128 different mantissas
        series_b_reduced = reduce_batch_and_cols(
            tensor_b.to(torch.float32)
        )  # Note: tensor has 128 elements, but all have same mantissa
        assert series_a_reduced.size() == torch.Size([rows])
        assert series_b_reduced.size() == torch.Size([rows])

        # Insert into results
        df_results = pd.DataFrame({"a": series_a_reduced, "b": series_b_reduced, "max_ulp_error": ulp_batch["max"]})

        df_all_results = pd.concat([df_all_results, df_results])

        i += 1

    # Write results to CSV
    df_all_results.to_csv(f"{dest_dir}/{operation_name}[bfloat16].csv", na_rep="NaN", index_label="index")


def main(args):
    dest_dir = "accuracy_results/results/binary"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # np.seterr(divide="ignore")
    # np.seterr(invalid="ignore")
    # np.seterr(over="ignore")

    operation_names = ["pow21f"]
    all_operations = {name: operations.BINARY_OPERATIONS[name] for name in operation_names}

    (successes, failed) = utils.execute_benchmarks(bench_binary_op, all_operations, dest_dir)

    print(f"Sucessfully ran {len(successes)} / {len(all_operations)} operations")
    print(f"{TERM_GREEN}SUCCESS: {successes}{TERM_RESET}")
    print(f"{TERM_RED}FAILED: {failed}{TERM_RESET}")


args = sys.argv
main(args)

ttnn.close_device(device)
