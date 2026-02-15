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
    elif isinstance(torch_output, (list, tuple)):
        for item in torch_output:
            if isinstance(item, TorchTTNNTensor):
                torch_output_tensors.append(item.to_torch)
    if isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_output.elem = None
        ttnn_output_tensors.append(ttnn_output.to_torch)
        assert isinstance(torch_output, TorchTTNNTensor), "Mismatched output types between TTNN and Torch."
    elif isinstance(ttnn_output, (list, tuple)):
        assert isinstance(torch_output, (list, tuple)), "Mismatched output types between TTNN and Torch."
        assert len(ttnn_output) == len(torch_output), "Mismatched output lengths between TTNN and Torch."
        for index, item in enumerate(ttnn_output):
            if isinstance(item, TorchTTNNTensor):
                assert isinstance(
                    torch_output[index], TorchTTNNTensor
                ), "Mismatched output types between TTNN and Torch."
                item.elem = None
                ttnn_output_tensors.append(item.to_torch)

    passed = True
    for t_tensor, n_tensor in zip(torch_output_tensors, ttnn_output_tensors):
        # calculate PCC between t_tensor and n_tensor
        t_tensor = t_tensor.to(torch.float32)
        n_tensor = n_tensor.to(torch.float32)
        assert t_tensor.shape == n_tensor.shape, "Mismatched output shapes between TTNN and Torch."
        pcc = torch.corrcoef(torch.stack([t_tensor.flatten(), n_tensor.flatten()]))[0, 1]
        diff = torch.abs(t_tensor - n_tensor)
        if pcc < 0.999 or (torch.median(diff) > torch.mean(diff) and torch.max(diff).item() > 1):
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
