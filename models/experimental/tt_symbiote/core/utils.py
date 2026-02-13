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

    def find_first_tensor(obj):
        if isinstance(obj, TorchTTNNTensor):
            return obj
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            for item in obj:
                if isinstance(item, TorchTTNNTensor):
                    return item
        return obj

    if isinstance(ttnn_output, TorchTTNNTensor) and not isinstance(torch_output, TorchTTNNTensor):
        torch_output = find_first_tensor(torch_output)

    if isinstance(torch_output, TorchTTNNTensor) and not isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_output = find_first_tensor(ttnn_output)

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
        assert isinstance(
            torch_output, TorchTTNNTensor
        ), f"Type mismatch in {func_name}: TTNN is Tensor, Torch is {type(torch_output)}"
    elif isinstance(ttnn_output, (list, tuple)):
        assert isinstance(
            torch_output, (list, tuple)
        ), f"Type mismatch in {func_name}: TTNN is List/Tuple, Torch is {type(torch_output)}"
        assert len(ttnn_output) == len(torch_output), "Mismatched output lengths between TTNN and Torch."
        for index, item in enumerate(ttnn_output):
            if isinstance(item, TorchTTNNTensor):
                if not isinstance(torch_output[index], TorchTTNNTensor):
                    print("Mismatched output types between TTNN and Torch.")
                assert isinstance(torch_output[index], TorchTTNNTensor), "Mismatched item types"
                item.elem = None
                ttnn_output_tensors.append(item.to_torch)

    passed = True
    for t_tensor, n_tensor in zip(torch_output_tensors, ttnn_output_tensors):
        t_tensor = t_tensor.to(torch.float32)
        n_tensor = n_tensor.to(torch.float32)
        assert t_tensor.shape == n_tensor.shape, "Mismatched output shapes between TTNN and Torch."
        assert t_tensor.shape == n_tensor.shape, f"Shape mismatch in {func_name}: {t_tensor.shape} vs {n_tensor.shape}"

        pcc = torch.corrcoef(torch.stack([t_tensor.flatten(), n_tensor.flatten()]))[0, 1]
        diff = torch.abs(t_tensor - n_tensor)
        print(f"[PCC REPORT] {func_name} -> PCC: {pcc.item():.6f}")
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
        print(f"!!! {func_name} FAILED ACCURACY CHECK (PCC < 0.999) !!!")


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


# ==============================================================================
# Groot: Hardware Assimilation & Tensor Unwrapping
# These functions handle specialized tensor wrappers and complex-to-real
# conversions required for the GR00T-N1.6 model architecture on Tenstorrent.
# ==============================================================================


def unwrap_ttnn(tensor):
    """Recursively extracts the core tensor from TTNN or Symbiote wrappers."""
    if tensor is None:
        return None
    curr = tensor
    while hasattr(curr, "ttnn_tensor") or hasattr(curr, "value") or hasattr(curr, "tensor"):
        curr = getattr(curr, "ttnn_tensor", getattr(curr, "value", getattr(curr, "tensor", curr)))
    return curr


def assimilate_to_device(tensor, device):
    """
    Prepares tensors for GR00T inference by handling unwrapping,
    complex-to-real conversion, and moving to the Tenstorrent device.
    """
    if tensor is None:
        return None

    curr = unwrap_ttnn(tensor)

    if isinstance(curr, ttnn.Tensor) and curr.storage_type() == ttnn.StorageType.DEVICE:
        return ensure_tile_layout(curr)

    torch_t = curr if isinstance(curr, torch.Tensor) else ttnn.to_torch(curr)
    if torch.is_complex(torch_t):
        torch_t = torch_t.real

    return ttnn.from_torch(torch_t.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
