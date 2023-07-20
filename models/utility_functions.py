import time
import tt_lib
import torch
import numpy as np
from loguru import logger
from tt_lib.utils import _nearest_32
from os import environ



def enable_compile_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built_kernels/.../hash directory with kernel binaries is present.
    """
    tt_lib.device.EnableCompileCache()


def disable_compile_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    tt_lib.device.DisableCompileCache()


def get_compile_cache_enabled():
    """
    Returns the current state of persistent compile cache on/off switch.
    """
    return tt_lib.device.GetCompileCacheEnabled()


def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableCompilationReports()


def disable_compilation_reports():
    """
    Disables generating reports of compilation statistics
    """
    return tt_lib.device.DisableCompilationReports()


def enable_memory_reports():
    """
    Enables generating reports of memory allocation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableMemoryReports()


def disable_memory_reports():
    """
    Disables generating reports of memory allocation statistics
    """
    return tt_lib.device.DisableMemoryReports()

def torch2tt_tensor(py_tensor: torch.Tensor, tt_device, tt_layout=tt_lib.tensor.Layout.TILE, tt_memory_config=tt_lib.tensor.MemoryConfig(True)):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_layout).to(tt_device, tt_memory_config)

    return tt_tensor


def tt2torch_tensor(tt_tensor, tt_host=None):
    if tt_host == None:
        host = tt_lib.device.GetHost()
    else:
        host = tt_host
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)

    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).reshape(tt_output.shape())
    return py_output


def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    pad_shape = list(x.shape)
    while len(pad_shape) < 4:
        pad_shape.insert(0, 1)
    if pad_shape[3] % 32 != 0 or pad_shape[2] % 32 != 0:
        tt_tensor = tt_lib.tensor.Tensor(
        x.reshape(-1).tolist(),
        pad_shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
        )
        x = tt_tensor.pad((pad_shape[0], pad_shape[1], _nearest_32(pad_shape[2]), _nearest_32(pad_shape[3])), (0, 0, 0, 0), 0)
        x = x.to(tt_lib.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape


def unpad_from_zero(x, desired_shape, host):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2] :
        x = tt2torch_tensor(x)
    else:
        x = x.to(host)
        if(x.layout() != tt_lib.tensor.Layout.ROW_MAJOR):
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1) )
        x = torch.Tensor(x.data()).reshape(x.shape())
    return x


def tt_to_torch_tensor(tt_tensor, host):
    tt_tensor = tt_tensor.to(host).to(tt_lib.tensor.Layout.ROW_MAJOR)
    # create a 1D PyTorch tensor from values in TT Tensor obtained with data() member function
    # and then reshape PyTorch tensor to shape of TT Tensor

    # py_tensor = torch.Tensor(tt_tensor.data()).reshape(tt_tensor.shape())

    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).reshape(tt_output.shape())

    return py_tensor


def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = (
        tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(), # PyTorch tensor flatten into a list of floats
            shape,               # shape of TT Tensor that will be created
            tt_lib.tensor.DataType.BFLOAT16, # data type that will be used in created TT Tensor
            tt_lib.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
        )
    )
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor


def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(), # PyTorch tensor flatten into a list of floats
            shape,               # shape of TT Tensor that will be created
            tt_lib.tensor.DataType.BFLOAT16, # data type that will be used in created TT Tensor
            tt_lib.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
        )
        .to(tt_lib.tensor.Layout.TILE)     # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                         # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )

    return tt_tensor
