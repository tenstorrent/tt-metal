"""Pure-PyTorch reference of g6_clone. No ttnn, no device.

Reference implementation followed:
pytorch/pytorch@70d99e998b4955e0049d13a98d77ae1b14db1f45 (torch 2.11.0, the build installed
here), pinned and cited as sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/TensorFactories.cpp:2270-2293
(the function) and sources/github/pytorch/pytorch@70d99e998b49/aten/src/ATen/native/native_functions.yaml:7085-7088
(schema and dispatch).
Deliberate differences from it: the memory_format keyword of the reference is not a parameter
(a device tensor has no strides to preserve; DOSSIER.md D4). The oracle returns the input dtype
without a float32 round trip: int32 and uint32 values above 2^24 in magnitude are not exact in
float32, and a copy must be bit-identical (DOSSIER.md D6).
"""

import torch

# Dtypes whose bit pattern is read through a same-width signed integer view.
_BITS_VIEW = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float64: torch.int64,
    torch.uint32: torch.int32,
    torch.int32: torch.int32,
}


def reference(input_tensor):
    """The definition, written out (never the built-in it is compared with).

    S1: out has the shape and dtype of input_tensor and lives in new storage.
    S2: out[i] = input_tensor[i] for every logical index i, bit for bit.
    """
    flat = input_tensor.reshape(-1)
    view = _BITS_VIEW.get(flat.dtype)
    bits = flat.view(view) if view is not None and view != flat.dtype else flat
    out_bits = torch.empty(bits.shape, dtype=bits.dtype)  # S1: new storage
    index = torch.arange(bits.numel(), dtype=torch.int64)
    out_bits[index] = bits[index]  # S2: every logical element, as its bit pattern
    out = out_bits.view(flat.dtype) if out_bits.dtype != flat.dtype else out_bits
    return out.reshape(input_tensor.shape)  # S1: the input's shape


def pytorch_g6_clone(input_tensor):
    """Oracle with the op's signature: the input dtype in and out, no float32 round trip (D6)."""
    return input_tensor.clone()
