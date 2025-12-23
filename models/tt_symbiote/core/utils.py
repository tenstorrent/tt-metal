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
