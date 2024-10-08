# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union
import time
import ttnn
import torch
import numpy as np
from loguru import logger
import os
import math
import struct
import pytest

from ttnn.device import Arch


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
def enable_persistent_kernel_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built_kernels/.../hash directory with kernel binaries is present.
    """
    logger.warning(
        "Persistent kernel cache is enabled. Cache invalidation may fail after a rebase and may require deleting the built directory."
    )
    ttnn.device.EnablePersistentKernelCache()


def disable_persistent_kernel_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    ttnn.device.DisablePersistentKernelCache()


def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return ttnn.device.EnableCompilationReports()


def disable_compilation_reports():
    """
    Disables generating reports of compilation statistics
    """
    return ttnn.device.DisableCompilationReports()


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

    tt_tensors_device = ttnn.to_torch(
        tt_tensors_device, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=concat_dim), device=mesh_device
    )

    return tt_tensors_device


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


### Padding / Unpadding ###
def pad_by_zero(
    x: torch.Tensor,
    device=None,
    tt_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
    tt_dtype=ttnn.bfloat16,
):
    initial_shape = x.shape
    pad_shape = list(x.shape)
    while len(pad_shape) < 4:
        pad_shape.insert(0, 1)
    if pad_shape[-1] % 32 != 0 or pad_shape[-2] % 32 != 0:
        # Pad in torch before creating TT tensor.
        # Certain datatypes like BFP8_B requires inputs to already be a specific size when creating the tensor, so we need to pad first
        x = torch.nn.functional.pad(
            x.reshape(pad_shape),
            (
                0,
                _nearest_32(pad_shape[-1]) - pad_shape[-1],
                0,
                _nearest_32(pad_shape[-2]) - pad_shape[-2],
            ),
        )
        x = ttnn.Tensor(x, tt_dtype)
        x = x.to(ttnn.TILE_LAYOUT)
        if device is not None:
            x = x.to(device, tt_memory_config)

    else:
        x = torch2tt_tensor(x, device, tt_memory_config=tt_memory_config, tt_dtype=tt_dtype)
    return x, initial_shape


def unpad_from_zero(x, desired_shape):
    if x.shape.with_tile_padding()[-1] == desired_shape[-1] and x.shape.with_tile_padding()[-2] == desired_shape[-2]:
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
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"
    if len(x.shape) == 1:  # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, : x.shape[0]] = x
    elif len(x.shape) == 2:  # (batch, num features)
        padded_tensor = torch.zeros(x.shape[0], 1, 32, nearest_32(x.shape[1]))
        padded_tensor[:, 0, 0, : x.shape[1]] = x
    elif len(x.shape) == 3:  # (batch, num features y, num features x)
        padded_tensor = torch.zeros(x.shape[0], 1, nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., 0, : x.shape[-2], : x.shape[-1]] = x
    else:  # (batch, num channels, num features y, num features x)
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., : x.shape[-2], : x.shape[-1]] = x
    return padded_tensor


def pad_weight(x):
    """
    This function pads a weight/bias with 0s as a pre-preprocessing step to tilization.

    tt_tensor = ttnn.Tensor(
        py_tensor.reshape(shape), ttnn.bfloat16
    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"

    if len(x.shape) == 1:  # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, : x.shape[0]] = x
    elif len(x.shape) == 2:  # (r_features, c_features)
        padded_tensor = torch.zeros(1, 1, nearest_32(x.shape[0]), nearest_32(x.shape[1]))
        padded_tensor[:, 0, : x.shape[0], : x.shape[1]] = x
    else:
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., : x.shape[-2], : x.shape[-1]] = x

    return padded_tensor


def convert_weights_2d_matrix(weights, w_shape):
    """
    :param weights: Input PyTorch Tensor
    :type weights: class:`torch.Tensor`
    """
    ret_shape = [1, 1, w_shape[0], w_shape[1] * w_shape[2] * w_shape[3]]
    if isinstance(weights, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for k in range(w_shape[0]):
        for r in range(w_shape[2]):
            for s in range(w_shape[3]):
                for c in range(w_shape[1]):
                    ret[idx] = weights[k][c][r][s]
                    idx += 1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape).transpose(2, 3)


def convert_act_2d_matrix(activation, kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x):
    """
    :param activation: Input PyTorch Tensor
    :type activation: class:`torch.Tensor`
    """
    N = activation.shape[0]
    C = activation.shape[1]
    H = activation.shape[2]
    W = activation.shape[3]

    OH = (int)((H - kernel_y + 2 * pad_y) // stride_y) + 1
    OW = ((W - kernel_x + 2 * pad_x) // stride_x) + 1
    nrows = OH * OW
    ncols = C * kernel_x * kernel_y
    ret_shape = [1, N, nrows, ncols]
    if isinstance(activation, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for n in range(N):
        for h in range(-1 * pad_y, H + pad_y - kernel_y + 1, stride_y):
            for w in range(-1 * pad_x, W + pad_x - kernel_x + 1, stride_x):
                for r in range(kernel_y):
                    for s in range(kernel_x):
                        for c in range(C):
                            h_offs = h + r
                            w_offs = w + s
                            pad = h_offs < 0 or h_offs >= H or w_offs < 0 or w_offs >= W
                            ret[idx] = 0 if pad else activation[n][c][h_offs][w_offs]
                            idx += 1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape)


### Tilizing / Untilizing ###
def tilize(x):
    """
    This function tilizes a tensor. The last two tensor dims must be divisible by 32, after which this function
    produces row major tiles and creates faces. The output of this function is a flattened list that
    we can send to the device.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(
        x, (torch.Tensor, np.ndarray)
    ), "Input to this function must be an instance of torch.Tensor or np.array"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (
        x.shape[-1] % 32
    ) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(np.prod(x.shape))
    else:
        ret = np.zeros(np.prod(x.shape))

    idx = 0
    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            for H in range(0, x.shape[2], 32):
                for W in range(0, x.shape[3], 32):
                    unfaced_tile = x[B, C, H : H + 32, W : W + 32]

                    face0 = unfaced_tile[:16, :16]
                    face1 = unfaced_tile[:16, 16:]
                    face2 = unfaced_tile[16:, :16]
                    face3 = unfaced_tile[16:, 16:]

                    for face in (face0, face1, face2, face3):
                        ret[idx : idx + 256] = face.reshape(-1)
                        idx += 256

    return ret.reshape(x.shape)


def tilize_to_list(x):
    """
    Tilize a PyTorch and then return the values as a flat list. The last two
    tensor dims must be divisible by 32, after which this function produces row
    major tiles and creates faces.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """

    return tilize(x).reshape(-1).tolist()


def untilize(x):
    """
    This function untilizes a tensor to row major format.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (
        x.shape[-1] % 32
    ) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(x.shape)
    else:
        ret = np.zeros(x.shape)

    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            x_hw = x[B, C, :].reshape(-1)
            hw = 0
            for h in range(0, x.shape[2], 32):
                for w in range(0, x.shape[3], 32):
                    f_tile = x_hw[hw : hw + 256].reshape(16, 16)
                    ret[B, C, h : h + 16, w : w + 16] = f_tile

                    f_tile = x_hw[hw + 256 : hw + 512].reshape(16, 16)
                    ret[B, C, h : h + 16, w + 16 : w + 32] = f_tile

                    f_tile = x_hw[hw + 512 : hw + 768].reshape(16, 16)
                    ret[B, C, h + 16 : h + 32, w : w + 16] = f_tile

                    f_tile = x_hw[hw + 768 : hw + 1024].reshape(16, 16)
                    ret[B, C, h + 16 : h + 32, w + 16 : w + 32] = f_tile
                    hw += 1024  # traverse tiles in RM-order

    return ret


### Measuring accuracy and other metrics ###
def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag=2.0, max_mag_fraction=0.02):
    """
    A variant of np.isclose with logging.
    """
    absdiff = (a - b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs() + 1.0) / (b.abs() + 1.0) - 1.0  # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs() < rtol, reldiff2.abs() < rtol)
    max_mag_ok = absdiff < max_mag * max_mag_fraction

    or_abs_rel = torch.logical_or(absdiff < atol, reldiff_or)
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()

    if not or_abs_rel.reshape(-1)[debug_index]:
        logger.info(f"isclose mismatch at index={debug_index}")
        logger.info(a.reshape(-1)[debug_index])
        logger.info(b.reshape(-1)[debug_index])
        logger.info(f"reldiff1={reldiff1.reshape(-1)[debug_index]}")
        logger.info(f"reldiff2={reldiff2.reshape(-1)[debug_index]}")
        logger.info(f"absdiff={absdiff.reshape(-1)[debug_index]}")

        HT = a.shape[-2] // 32
        WT = a.shape[-1] // 32
        hwt = debug_index // 1024
        wt = hwt % WT
        ht = hwt // WT
        h = (debug_index % 1024) // 32
        w = (debug_index % 1024) % 32

        logger.info(f"****    at {debug_index} --- HTWT={ht} {wt} HW={h} {w}")

    return torch.all(or_abs_rel)


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    passing = True
    output = ""
    passing_allclose, output_allclose = comp_allclose(golden, calculated, rtol, atol)
    passing &= passing_allclose
    output += output_allclose
    if torch.numel(golden) != 1:
        passing_pcc, output_pcc = comp_pcc(golden, calculated, pcc)
        passing &= passing_pcc
        output += f", {output_pcc}"

    return passing, output


def comp_equal(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()
    return (
        torch.equal(golden, calculated),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    ooms = []
    for el in float_lst:
        str_el = str(el)
        if "e" in str_el:
            oom = int(str_el.split("e")[1])
        elif str_el[:2] == "0.":
            str_el = str_el.split(".")[1]

            oom = -1
            for e in str_el:
                if e != "0":
                    break
                oom -= 1
        else:
            oom = len(str_el.split(".")[0])

        ooms.append(oom)

    return ooms


def print_diff_argmax(a, b, annotation=""):
    """
    Prints out the value of both tensors at a point where the absolute difference is the largest.
    """
    absdiff = (a - b).abs()
    argmax = absdiff.argmax().item()
    diff = absdiff.reshape(-1)[argmax]
    rela = a.abs() / (torch.max(a.abs(), b.abs()))
    relb = b.abs() / (torch.max(a.abs(), b.abs()))
    HT = a.shape[-2] // 32
    WT = a.shape[-1] // 32
    hwt = argmax // 1024
    wt = hwt % WT
    ht = hwt // WT
    h = (argmax % 1024) // 32
    w = (argmax % 1024) % 32
    print(
        "Abs diff=",
        diff,
        " at ",
        argmax,
        " --- ",
        annotation,
        "HTWT=",
        ht,
        wt,
        "HW=",
        h,
        w,
    )
    print("  (a=", a.reshape(-1)[argmax].item(), ")")
    print("  (b=", b.reshape(-1)[argmax].item(), ")")
    print("  Rel a=", rela.reshape(-1)[argmax], " at ", argmax)
    print("  Rel b=", relb.reshape(-1)[argmax], " at ", argmax)
    return diff.item()


def print_diff_tt_pyt(a, b, annotation=""):
    # first convert a pytorch tensor argument b to tt
    padded_b = pad_weight(b)
    pyt_a = tt2torch(a)  # untilizes also
    return print_diff_argmax(pyt_a, padded_b, annotation)


def ttP(x, count=4, offset=0, stride=1):
    if type(x) == torch.Tensor:
        t1 = x.reshape(-1)
    else:
        tt_out = x.cpu()
        torch_out = untilize(tt_out.to_torch())
        t1 = torch_out.reshape(-1)
    print("Tensor vals: (", end="")
    for j in range(offset, offset + count * stride, stride):
        print(t1[j].item(), " ", end="")
    print(")")


### Conv related helpers ###
def read_conv_act_into_mm_act_block(
    conv_act,
    act_address_map_index,
    address_map,
    address_map_this_block_size,
    act_block_h,
    act_block_w,
):
    mm_act_block_shape = [1, 1, act_block_h * 32, act_block_w * 32]
    mm_act_block_size = act_block_h * act_block_w * 1024
    mm_act_block = torch.zeros(mm_act_block_size, dtype=torch.bfloat16).float()
    for i in range(0, address_map_this_block_size, 4):
        src_address = address_map[act_address_map_index]
        dst_address = address_map[act_address_map_index + 1]
        read_size = address_map[act_address_map_index + 2]
        pad = address_map[act_address_map_index + 3]
        for s in range(read_size):
            assert dst_address + s < mm_act_block_size
            if pad:
                mm_act_block[dst_address + s] = 0
            else:
                assert src_address + s < len(conv_act)
                mm_act_block[dst_address + s] = conv_act[src_address + s]
        act_address_map_index += 4
    return (mm_act_block.reshape(mm_act_block_shape), act_address_map_index)


def read_conv_weight_into_mm_weight_block(
    conv_weight,
    weight_address_map_index,
    weight_address_map,
    weight_address_map_this_block_size,
    weight_block_h,
    weight_block_w,
):
    mm_weight_block_shape = [1, 1, weight_block_h * 32, weight_block_w * 32]
    mm_weight_block_size = weight_block_h * weight_block_w * 1024
    mm_weight_block = torch.zeros(mm_weight_block_size, dtype=torch.bfloat16).float()
    for i in range(0, weight_address_map_this_block_size, 4):
        src_address = weight_address_map[weight_address_map_index]
        dst_address = weight_address_map[weight_address_map_index + 1]
        read_size = weight_address_map[weight_address_map_index + 2]
        pad = weight_address_map[weight_address_map_index + 3]
        for s in range(read_size):
            assert dst_address + s < mm_weight_block_size
            if pad:
                mm_weight_block[dst_address + s] = 0
            else:
                assert src_address + s < len(conv_weight)
                mm_weight_block[dst_address + s] = conv_weight[src_address + s]
        weight_address_map_index += 4
    return (mm_weight_block.reshape(mm_weight_block_shape), weight_address_map_index)


def blocked_mm_with_conv_act(
    conv_act,
    mm_weight,
    act_address_map,
    weight_address_map,
    num_blocks_act_h,
    num_blocks_act_w,
    num_blocks_weight_w,
    act_block_h,
    act_block_w,
    weight_block_w,
):
    # act refers to conv activation tensor
    # weight refers to conv weight tensor
    mm_output_shape = [
        1,
        1,
        num_blocks_act_h * act_block_h * 32,
        num_blocks_weight_w * weight_block_w * 32,
    ]
    ret = torch.zeros(mm_output_shape, dtype=torch.bfloat16).float()
    mm_output_block_shape = [1, 1, act_block_h * 32, weight_block_w * 32]
    act_address_map_index = 0
    weight_address_map_index = 0
    weight_block_h = act_block_w
    num_groups = act_address_map[act_address_map_index]
    assert num_groups == num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w
    weight_num_groups = act_address_map[weight_address_map_index]
    assert weight_num_groups == num_groups
    act_address_map_index += 1
    weight_address_map_index += 1
    for block_act_h in range(num_blocks_act_h):
        # Reset weight (weight) to the starting tile in this column
        for block_weight_w in range(num_blocks_weight_w):
            output_block = torch.zeros(mm_output_block_shape, dtype=torch.bfloat16).float()
            for block_act_w in range(num_blocks_act_w):
                address_map_this_block_size = act_address_map[act_address_map_index]
                act_address_map_index += 1
                weight_address_map_this_block_size = weight_address_map[weight_address_map_index]
                weight_address_map_index += 1
                (mm_act_block, act_address_map_index) = read_conv_act_into_mm_act_block(
                    conv_act,
                    act_address_map_index,
                    act_address_map,
                    address_map_this_block_size,
                    act_block_h,
                    act_block_w,
                )
                (
                    mm_weight_block,
                    weight_address_map_index,
                ) = read_conv_weight_into_mm_weight_block(
                    mm_weight,
                    weight_address_map_index,
                    weight_address_map,
                    weight_address_map_this_block_size,
                    weight_block_h,
                    weight_block_w,
                )
                # Untilize weight block (this CPU reference does matmul on untilized blocks)
                mm_weight_block = untilize(mm_weight_block)
                for out_h_block in range(act_block_h * 32):
                    for out_w_block in range(weight_block_w * 32):
                        output_block[0][0][out_h_block][out_w_block] += torch.dot(
                            mm_act_block[0, 0, out_h_block, :].reshape(-1),
                            mm_weight_block[0, 0, :, out_w_block].reshape(-1),
                        )
            start_oh = block_act_h * act_block_h * 32
            start_ow = block_weight_w * weight_block_w * 32
            end_oh = start_oh + (act_block_h * 32)
            end_ow = start_ow + (weight_block_w * 32)
            ret[0, 0, start_oh:end_oh, start_ow:end_ow] = output_block

    return ret


def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    if K % 32 != 0 or dilation != 1 or groups != 1:
        logger.warning("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        logger.warning(
            "K="
            + str(K)
            + " C="
            + str(C)
            + " R="
            + str(R)
            + " S="
            + str(S)
            + " U="
            + str(U)
            + " V="
            + str(V)
            + " PH="
            + str(P_H)
            + " PW="
            + str(P_W)
            + " dilation="
            + str(dilation)
            + " groups="
            + str(groups)
        )
        return False

    return True


# detect E75 Grayskull card
def is_e75(device):
    compute_grid_size = device.compute_with_storage_grid_size()
    return (device.arch() == Arch.GRAYSKULL) and (compute_grid_size.x * compute_grid_size.y == 88)


def is_x2_harvested(device):
    grid = device.compute_with_storage_grid_size()
    return device.arch() == Arch.WORMHOLE_B0 and (grid.x, grid.y) == (8, 7)


def is_blackhole():
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "blackhole" in ARCH_NAME


def is_wormhole_b0():
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "wormhole_b0" in ARCH_NAME


def is_grayskull():
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "grayskull" in ARCH_NAME


def skip_for_blackhole(reason_str="not working for Blackhole"):
    return pytest.mark.skipif(is_blackhole(), reason=reason_str)


def skip_for_wormhole_b0(reason_str="not working for Wormhole B0"):
    return pytest.mark.skipif(is_wormhole_b0(), reason=reason_str)


def skip_for_grayskull(reason_str="not working for Grayskull"):
    return pytest.mark.skipif(is_grayskull(), reason=reason_str)


def run_for_wormhole_b0(reason_str="only runs for Wormhole B0"):
    return pytest.mark.skipif(not is_wormhole_b0(), reason=reason_str)


def run_for_grayskull(reason_str="only runs for Grayskull"):
    return pytest.mark.skipif(not is_grayskull(), reason=reason_str)


def get_devices_for_t3000(all_devices, num_devices):
    """
    all_devices comes from fixture which devices in order from 0 to 7.
    First 4 devices are PCIE devices so we can just extract and return.
    For 8 devices, return in a ring pattern.
    """
    assert num_devices <= len(all_devices), "Not enough devices detected!"

    if num_devices <= 4:
        return all_devices[:num_devices]
    elif num_devices == 8:
        # Temporary until we move request for ring order to CCL operations directly.
        # This is better because we no longer need to manually manage the ring order.
        ring_indices = ttnn.get_t3k_physical_device_ids_ring()
        return [all_devices[i] for i in ring_indices]
    else:
        raise NotImplementedError("Only supports 1, 2, 3, 4, and 8 chip configurations!")


def ttl_complex_2_torch_complex(tt_tensor):
    torch_tensor = tt2torch_tensor(tt_tensor)

    # extract real and imag parts of the complex tensor
    real = torch_tensor[:, :, :, : torch_tensor.shape[-1] // 2].to(torch.bfloat16).to(torch.float)
    imag = torch_tensor[:, :, :, torch_tensor.shape[-1] // 2 :].to(torch.bfloat16).to(torch.float)

    # create torch complex tensor
    result = torch.complex(real, imag)
    return result


def pad_and_fold_conv_activation_for_unity_stride(activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w):
    assert stride_h == stride_w
    assert activation_pyt_nchw_tensor.shape[2] == activation_pyt_nchw_tensor.shape[3]
    # Fold activation for unity stride
    # Pad channel size to 4. This is to make sure L1 read addresses are 16 bit aligned
    C = _nearest_y(activation_pyt_nchw_tensor.shape[1], 4)
    # Also, pre-pad the conv left right and top bottom padding
    activation_pyt_padded = torch.nn.functional.pad(
        activation_pyt_nchw_tensor, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_pyt_nchw_tensor.shape[1])
    )
    # Fold the activation face by stride depth wise i.e. C,H,W -> C*stride_h*stride_w, H/stride_h, W/stride_w
    assert activation_pyt_padded.shape[2] % stride_h == 0
    activation_pyt_padded_folded = torch.zeros(
        [
            activation_pyt_padded.shape[0],
            C * stride_h * stride_w,
            (int)(activation_pyt_padded.shape[2] / stride_h),
            (int)(activation_pyt_padded.shape[3] / stride_w),
        ]
    )
    for h in range(0, activation_pyt_padded.shape[2], stride_h):
        for w in range(0, activation_pyt_padded.shape[3], stride_w):
            folded_h = (int)(h / stride_h)
            folded_w = (int)(w / stride_w)
            for i in range(stride_h * stride_w):
                start_c = i * C
                activation_pyt_padded_folded[:, start_c : start_c + C, folded_h, folded_w] = activation_pyt_padded[
                    :, :, h + (int)(i / stride_w), w + (int)(i % stride_w)
                ]

    return activation_pyt_padded_folded


def pad_and_fold_conv_filters_for_unity_stride(filter_pyt_nchw_tensor, stride_h, stride_w):
    assert stride_h == stride_w
    assert filter_pyt_nchw_tensor.shape[2] == filter_pyt_nchw_tensor.shape[3]
    # Fold activation for unity stride
    # Pad channel size to 4. This is to make sure L1 read addresses are 16 bit aligned
    C = _nearest_y(filter_pyt_nchw_tensor.shape[1], 4)
    # Pad filter to nearest stride
    Padded_filter_height = _nearest_y(filter_pyt_nchw_tensor.shape[2], stride_h)
    Padded_filter_width = _nearest_y(filter_pyt_nchw_tensor.shape[3], stride_w)
    filter_pyt_padded = torch.nn.functional.pad(
        filter_pyt_nchw_tensor,
        (
            0,
            Padded_filter_width - filter_pyt_nchw_tensor.shape[3],
            0,
            Padded_filter_height - filter_pyt_nchw_tensor.shape[2],
            0,
            C - filter_pyt_nchw_tensor.shape[1],
        ),
    )
    # Fold filter for unity stride.
    filter_pyt_padded_folded = torch.zeros(
        [
            filter_pyt_padded.shape[0],
            C * stride_h * stride_w,
            (int)(filter_pyt_padded.shape[2] / stride_h),
            (int)(filter_pyt_padded.shape[3] / stride_w),
        ]
    )
    for h in range(0, filter_pyt_padded.shape[2], stride_h):
        for w in range(0, filter_pyt_padded.shape[3], stride_w):
            folded_h = (int)(h / stride_h)
            folded_w = (int)(w / stride_w)
            for i in range(4):
                start_c = i * C
                filter_pyt_padded_folded[:, start_c : start_c + C, folded_h, folded_w] = filter_pyt_padded[
                    :, :, h + (int)(i / stride_w), w + (int)(i % stride_w)
                ]
    return filter_pyt_padded_folded


# produces a tensor where each element in a page is the page number
# this tensor is easy to debug and visualize
def get_debug_tensor(num_pages_width, num_pages_height, dtype, page_width=32, page_height=32):
    torch_tensor = None
    for row_idx in range(0, int(num_pages_height)):
        tile_row = None
        for col_idx in range(0, int(num_pages_width)):
            tile_idx = col_idx + num_pages_width * row_idx
            tile = torch.full((1, 1, page_width, page_height), tile_idx + 1, dtype=dtype)
            if tile_row == None:
                tile_row = tile
            else:
                tile_row = torch.cat((tile_row, tile), 3)
        if torch_tensor == None:
            torch_tensor = tile_row
        else:
            torch_tensor = torch.cat((torch_tensor, tile_row), 2)

    return torch_tensor
