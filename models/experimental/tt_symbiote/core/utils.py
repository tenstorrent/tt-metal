# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

TTNN_TO_TORCH = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.uint32: torch.uint32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.uint16,
    ttnn.uint8: torch.uint8,
}


def ttnn_dtype_to_torch_dtype(ttnn_dtype):
    if ttnn_dtype in TTNN_TO_TORCH:
        return TTNN_TO_TORCH[ttnn_dtype]
    print(f"Warning: Unsupported ttnn dtype for conversion to torch dtype: {ttnn_dtype}. Using float32 as fallback.")
    return torch.float32  # Default fallback, extend as needed


TORCH_TO_TTNN = {
    torch.float32: ttnn.float32,
    torch.bfloat16: ttnn.bfloat16,
    torch.float16: ttnn.bfloat16,
    torch.uint32: ttnn.uint32,
    torch.int32: ttnn.int32,
    torch.uint16: ttnn.uint16,
    torch.uint8: ttnn.uint8,
    torch.int16: ttnn.int32,
    torch.bool: ttnn.uint8,
    torch.uint8: ttnn.uint8,
    torch.int64: ttnn.int32,
}


def torch_dtype_to_ttnn_dtype(torch_dtype):
    if torch_dtype in TORCH_TO_TTNN:
        return TORCH_TO_TTNN[torch_dtype]
    elif "float" in str(torch_dtype).lower():
        print(f"Warning: Generic 'float' torch dtype detected, using float32 for ttnn dtype.")
        return ttnn.float32
    elif "bool" in str(torch_dtype).lower():
        return ttnn.uint8
    elif "int" in str(torch_dtype).lower():
        print(f"Warning: {torch_dtype} torch dtype detected, using int32 for ttnn dtype.")
        return ttnn.int32
    else:
        raise RuntimeError(f"Unsupported torch dtype for conversion to ttnn dtype: {torch_dtype}")


def compare_fn_outputs(torch_output, ttnn_output, func_name):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    torch_output_tensors = []
    ttnn_output_tensors = []
    if isinstance(torch_output, TorchTTNNTensor):
        torch_output_tensors.append(torch_output.to_torch)
    elif isinstance(torch_output, torch.Tensor):
        torch_output_tensors.append(torch_output)
    elif isinstance(torch_output, (list, tuple)):
        for item in torch_output:
            if isinstance(item, TorchTTNNTensor):
                torch_output_tensors.append(item.to_torch)
            elif isinstance(item, torch.Tensor):
                torch_output_tensors.append(item)
    if isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_output.elem = None
        ttnn_output_tensors.append(ttnn_output.to_torch)
        if not isinstance(torch_output, TorchTTNNTensor):
            print("Mismatched output types between TTNN and Torch.")
    elif isinstance(ttnn_output, (list, tuple)):
        assert isinstance(torch_output, (list, tuple)), "Mismatched output types between TTNN and Torch."
        assert len(ttnn_output) == len(torch_output), "Mismatched output lengths between TTNN and Torch."
        for index, item in enumerate(ttnn_output):
            if isinstance(item, TorchTTNNTensor):
                if not isinstance(torch_output[index], TorchTTNNTensor):
                    print("Mismatched output types between TTNN and Torch.")
                item.elem = None
                ttnn_output_tensors.append(item.to_torch)

    passed = True
    for t_tensor, n_tensor in zip(torch_output_tensors, ttnn_output_tensors):
        # calculate PCC between t_tensor and n_tensor
        t_tensor = t_tensor.to(torch.float32)
        n_tensor = n_tensor.to(torch.float32)
        assert t_tensor.shape == n_tensor.shape, "Mismatched output shapes between TTNN and Torch."
        pcc = torch.corrcoef(torch.stack([t_tensor.flatten(), n_tensor.flatten()]))[0, 1]
        print(f"PCC: {pcc.item()}")
        diff = torch.abs(t_tensor - n_tensor)
        if (
            pcc < 0.999
            or (torch.median(diff) > torch.mean(diff) and torch.max(diff).item() > 1)
            or pcc.isnan().any()
            or diff.isnan().any()
        ):
            passed = False
            print(
                f"Warning: High discrepancy detected in operation {func_name}. "
                f"PCC: {pcc.item()}, Max Abs Diff: {torch.max(diff).item()}, Median Abs Diff: {torch.median(diff).item()}, Mean Abs Diff: {torch.mean(diff).item()}"
            )
        if torch.logical_xor((n_tensor == 0).all(), (t_tensor == 0).all()):
            passed = False
            print(f"Warning: One of the outputs is all zeros while the other is not in operation {func_name}.")

        if func_name == "aten::topk":
            break
    if not passed:
        print(f"Operation {func_name} PCC < 0.99.")


def ensure_tile_layout(tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Convert tensor to TILE_LAYOUT if needed.

    Args:
        tensor: TTNN tensor to convert

    Returns:
        Tensor in TILE_LAYOUT
    """
    if tensor.layout != ttnn.TILE_LAYOUT:
        return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return tensor


def optimized_tree_map_with_only_dict_list(*args, **kwargs):
    # don't use pytorch
    from collections.abc import Mapping, Sequence

    func = args[0]
    data_structures = args[1:]
    if all(isinstance(ds, Mapping) for ds in data_structures):
        keys = data_structures[0].keys()
        if not all(ds.keys() == keys for ds in data_structures):
            raise ValueError("All dicts must have the same keys")
        return {
            key: optimized_tree_map_with_only_dict_list(func, *(ds[key] for ds in data_structures), **kwargs)
            for key in keys
        }
    elif all(isinstance(ds, Sequence) and not isinstance(ds, str) for ds in data_structures):
        if not all(len(ds) == len(data_structures[0]) for ds in data_structures):
            raise ValueError("All lists must have the same length")
        return [
            optimized_tree_map_with_only_dict_list(func, *(ds[i] for ds in data_structures), **kwargs)
            for i in range(len(data_structures[0]))
        ]
    return func(*data_structures, **kwargs)


def tree_map(*args, **kwargs):
    import time
    from models.experimental.tt_symbiote.core.run_config import DispatchManager

    start_time = time.time()
    result = optimized_tree_map_with_only_dict_list(*args, **kwargs)
    end_time = time.time()
    DispatchManager.record_timing(
        "Torch",
        (
            ""
            if DispatchManager.current_module_name is None
            else DispatchManager.current_module_name + f".{args[0].__name__}"
        ),
        args[0].__name__,
        {},
        end_time - start_time,
    )
    return result


def wrap_from_torch(data_structure):
    from models.experimental.tt_symbiote.core.run_config import wrap_from_torch as wrap_from_torch_impl

    return tree_map(wrap_from_torch_impl, data_structure)
