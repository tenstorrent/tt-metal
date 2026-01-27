# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff key is not in cacheDict.
def constEvalFuncWrapper(constEvalFunc, inputs, cacheDict, key, device):
    if key not in cacheDict:
        cacheDict[key] = constEvalFunc(inputs, device)
    return cacheDict[key]


# Wrapper to abstract const-eval logic out of runtime funcs to keep them
# cleaner. Invokes constEvalFunc iff key is not in cacheDict.
# This is an overload of constEvalFuncWrapper for const-eval functions that
# take zero arguments (but still need device).
def constEvalFuncWrapperZeroArg(constEvalFunc, cacheDict, key, device):
    if key not in cacheDict:
        cacheDict[key] = constEvalFunc(device)
    return cacheDict[key]


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()


def load_weight_from_pytorch(
    state_dict: dict,
    weight_name: str,
    layout,
    dtype,
    device,
    memory_config,
) -> ttnn.Tensor:
    """Load a weight from PyTorch state_dict and convert to TTNN tensor."""
    pt_tensor = state_dict[weight_name]

    # Convert PyTorch tensor to TTNN tensor
    ttnn_tensor = ttnn.from_torch(pt_tensor, dtype=dtype, layout=layout)

    if device is not None:
        ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config)

    return ttnn_tensor


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}")

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
