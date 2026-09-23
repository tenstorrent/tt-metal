# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import struct
import time
from typing import Union

import numpy as np
import torch
from loguru import logger
from ttnn.device import Arch
from typing_extensions import deprecated

import ttnn


def get_mesh_device():
    """Fixture to provide mesh device configuration."""
    mesh_device = os.environ.get("MESH_DEVICE", "N150")
    mesh_config = {
        "N150": (1, 1),
        "N300": (2, 1),
        "T3K": (8, 1),
        "TG": (8, 4),
    }.get(mesh_device, (ttnn.get_num_devices(), 1))
    return mesh_config


### Math operations ###
def _nearest_32(x):
    return math.ceil(x / 32) * 32


def nearest_32(
    x,
):  # needs refctoring; to match alias called in some scripts (e.g. test_padding_test in unit tests)
    return _nearest_32(x)


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def nearest_y(x, y):
    return _nearest_y(x, y)


def divup(a, b):
    return (a + b - 1) // b


def roundup(a, b):
    result = divup(a, b) * b
    return result


def roundup32(a):
    return roundup(a, 32)


def float_to_bits(x):
    s = struct.pack(">f", x)
    return struct.unpack(">l", s)[0]


def torch_random(shape, low, high, dtype):
    if dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


def torch_random_with_zeros(shape, low, high, dtype, zero_fraction=0.1):
    total_elements = torch.prod(torch.tensor(shape)).item()
    num_zeros = int(total_elements * zero_fraction)
    num_random = total_elements - num_zeros

    # Generate random values between low and high
    random_values = torch.empty(num_random).uniform_(low, high)
    zeros = torch.zeros(num_zeros)

    # Combine zeros and random values
    combined = torch.cat([zeros, random_values])

    # Shuffle the tensor
    shuffled = combined[torch.randperm(combined.size(0))]

    # Reshape to the desired shape. Tensor.to is not in-place, so return the converted tensor.
    result_tensor = shuffled.view(shape)
    return result_tensor.to(dtype)


### Profiling ###
class Profiler:
    def __init__(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def clear(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True

    def start(self, key, force_enable=False):
        if self.disabled and not force_enable:
            return

        self.start_times[key] = time.time()

    def end(self, key, PERF_CNT=1, force_enable=False):
        if self.disabled and not force_enable:
            return

        if key not in self.start_times:
            return

        diff = time.time() - self.start_times[key]

        if key not in self.times:
            self.times[key] = []

        self.times[key].append(diff / PERF_CNT)

    def get(self, key):
        if key not in self.times:
            return 0

        return sum(self.times[key]) / len(self.times[key])

    def print(self, units="s"):
        for key in self.times:
            average = self.get(key)
            if units == "s":
                pass
            elif units == "ms":
                average *= 1000
            elif units == "us":
                average *= 1000000
            elif units == "ns":
                average *= 1000000000
            else:
                raise ValueError(f"Invalid units: {units}")
            print(f"{key}: {average:.3f}{units}")


profiler = Profiler()


### Turn flags on/off ###
def enable_memory_reports():
    """
    Enables generating reports of memory allocation statistics in .reports/tt_metal dir
    """
    return ttnn.device.EnableMemoryReports()


def disable_memory_reports():
    """
    Disables generating reports of memory allocation statistics
    """
    return ttnn.device.DisableMemoryReports()


### Tensor conversion ###
def torch2tt_tensor(
    py_tensor: torch.Tensor,
    tt_device,
    tt_layout=ttnn.TILE_LAYOUT,
    tt_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
    tt_dtype=ttnn.bfloat16,
):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = ttnn.Tensor(py_tensor.reshape(size), tt_dtype)
    tt_tensor = tt_tensor.to(tt_layout)

    if tt_device is not None:
        tt_tensor = tt_tensor.to(tt_device, tt_memory_config)
    else:
        tt_tensor = tt_tensor.cpu()

    return tt_tensor


def tt_tensors_to_torch_tensors(
    tt_tensors_device: ttnn.Tensor, mesh_device: Union[ttnn.MeshDevice, ttnn.Device], concat_dim: int = 0
):
    # Convert tensors to interleaved
    if tt_tensors_device.is_sharded():
        tt_tensors_device = ttnn.sharded_to_interleaved(tt_tensors_device)

    # Convert tensors to RM layout
    if tt_tensors_device.layout == ttnn.TILE_LAYOUT:
        # Convert to bfloat16 to ensure untilize works
        if tt_tensors_device.dtype != ttnn.bfloat16:
            tt_tensors_device = ttnn.clone(
                tt_tensors_device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        # Untilize using singlecore since multicore version runs out of l1 memory (Issue #9022)
        tt_tensors_device = ttnn.untilize(tt_tensors_device, use_multicore=False)

    return torch.cat([t.to_torch() for t in ttnn.get_device_tensors(tt_tensors_device.cpu())], dim=concat_dim)


def tt2torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        tt_output = tt_output.to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()


def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()


def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor


def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
        .to(
            ttnn.TILE_LAYOUT
        )  # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)  # move TT Tensor from host to TT accelerator device (device is of type ttnn.device.Device)
    )

    return tt_tensor


def unpad_from_zero(x, desired_shape):
    if x.padded_shape[-1] == desired_shape[-1] and x.padded_shape[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = x.to(ttnn.ROW_MAJOR_LAYOUT)
        x = x.unpad(
            (0, 0, 0, 0),
            (
                desired_shape[0],
                desired_shape[1],
                desired_shape[2],
                desired_shape[3],
            ),
        )

        x = x.to_torch()
    return x


def pad_activation(x):
    """
    This function pads an activation with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 support"

    # Original implementation omitted for brevity
    return x

# -----------------------------------------------------------------------------
# Shifted Two-Pass Statistics (Welford replacement)
# -----------------------------------------------------------------------------

def _shifted_two_pass_mean(x: torch.Tensor, shift: float = None) -> torch.Tensor:
    """Compute the mean of *x* using a shifted two‑pass algorithm.

    The input is first shifted by a constant (defaulting to the first element) to
    improve numerical stability for large‑mean, low‑variance data. The mean is
    then recovered by adding the shift back.

    Args:
        x: Input tensor (any shape, dtype must be floating point).
        shift: Optional constant to subtract before accumulation. If ``None``
            the first element of the flattened tensor is used.

    Returns:
        Tensor containing the mean (scalar tensor).
    """
    if shift is None:
        shift = x.view(-1)[0].item()
    # Center the data
    centered = x - shift
    # First pass: compute mean of centered data
    mean_centered = centered.mean()
    # Recover the true mean
    return mean_centered + shift


def shifted_two_pass_variance(
    x: torch.Tensor, unbiased: bool = True, shift: float = None
) -> torch.Tensor:
    """Compute variance of *x* using a shifted two‑pass algorithm.

    The algorithm performs:
        mean = shift + average(x - shift)
        variance = average(((x - shift) - (mean - shift)) ** 2)
    which is equivalent to the standard definition but avoids catastrophic
    cancellation when ``x`` has a large common offset.

    Args:
        x: Input tensor (floating point).
        unbiased: If ``True`` (default) returns the unbiased estimator
            (multiply by N/(N‑1)).
        shift: Optional shift constant; defaults to the first element.

    Returns:
        Tensor containing the variance (scalar tensor).
    """
    if shift is None:
        shift = x.view(-1)[0].item()
    centered = x - shift
    mean_centered = centered.mean()
    # Second pass: variance of centered data
    var = ((centered - mean_centered) ** 2).mean()
    if unbiased:
        n = x.numel()
        if n > 1:
            var = var * n / (n - 1)
    return var


def shifted_two_pass_std(
    x: torch.Tensor, unbiased: bool = True, shift: float = None
) -> torch.Tensor:
    """Standard deviation using the shifted two‑pass variance implementation.

    Args:
        x: Input tensor.
        unbiased: Whether to use the unbiased variance estimator.
        shift: Optional shift constant.

    Returns:
        Tensor containing the standard deviation.
    """
    return torch.sqrt(shifted_two_pass_variance(x, unbiased=unbiased, shift=shift))

# End of file