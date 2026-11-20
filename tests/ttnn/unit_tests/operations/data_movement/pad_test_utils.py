# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Torch references and mismatch reports for pad/unpad tests, shared by host and device checks.

Every failure message describes the ttnn tensor (storage, layout, dtype, logical and padded shape), the first
mismatching elements, and its physical data, so a host/device divergence is readable from the pytest output.
"""

import math

import torch
import ttnn

MAX_REPORTED_MISMATCHES = 8


def make_ramp_input(shape, dtype):
    """Ramp starting at 1: never equal to 0, negative or inf pad values, exact in every listed dtype.

    Float values wrap below 32, where bfloat16 still represents every multiple of 1/8 exactly.
    """
    numel = math.prod(shape)
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return ((torch.arange(numel, dtype=torch.float32) % 248) / 8 + 1).reshape(shape)
    data = torch.arange(numel, dtype=torch.int32) + 1
    if dtype == ttnn.uint16:
        data = data.to(torch.uint16)
    return data.reshape(shape)


def reference_pad(torch_input, padding, value):
    """``padding`` is ((before, after), ...) per dim. Built in float64 so every dtype compares exactly."""
    output_shape = [before + size + after for size, (before, after) in zip(torch_input.shape, padding)]
    expected = torch.full(output_shape, float(value), dtype=torch.float64)
    region = tuple(slice(before, before + size) for size, (before, _) in zip(torch_input.shape, padding))
    expected[region] = torch_input.to(torch.float64)
    return expected


def describe(tt_tensor):
    storage = "device" if ttnn.is_tensor_storage_on_device(tt_tensor) else "host"
    return (
        f"storage={storage} layout={tt_tensor.layout} dtype={tt_tensor.dtype} "
        f"logical_shape={list(tt_tensor.shape)} padded_shape={list(tt_tensor.padded_shape)}"
    )


def physical_data(tt_tensor):
    host = ttnn.from_device(tt_tensor) if ttnn.is_tensor_storage_on_device(tt_tensor) else tt_tensor
    try:
        return host.to_torch_with_padded_shape()
    except Exception as error:
        return f"<to_torch_with_padded_shape failed: {error}>"


def _to_float64(tt_tensor):
    try:
        return ttnn.to_torch(tt_tensor).to(torch.float64), None
    except Exception as error:
        return None, error


def mismatch_report(label, tt_tensor, expected, torch_input=None):
    lines = [f"[{label}]", f"actual:   {describe(tt_tensor)}", f"expected: logical_shape={list(expected.shape)}"]
    actual, error = _to_float64(tt_tensor)
    if error is not None:
        lines.append(f"to_torch failed: {error}")
    elif actual.shape == expected.shape:
        wrong = actual != expected.to(torch.float64)
        lines.append(f"{int(wrong.sum())} of {expected.numel()} values wrong")
        for coord in wrong.nonzero()[:MAX_REPORTED_MISMATCHES].tolist():
            index = tuple(coord)
            lines.append(f"  at {coord}: got {actual[index].item()}, expected {expected[index].item()}")
    else:
        lines.append(f"to_torch shape {list(actual.shape)} != expected shape {list(expected.shape)}")
        if torch_input is not None and actual.shape == torch_input.shape:
            differ = int((actual != torch_input.to(torch.float64)).sum())
            lines.append(f"to_torch result vs unpadded input: {differ} of {torch_input.numel()} values differ")
    lines.append(f"physical data (to_torch_with_padded_shape):\n{physical_data(tt_tensor)}")
    return "\n".join(lines)


def assert_matches(label, tt_tensor, expected, torch_input=None):
    """Logical shape and values of ``tt_tensor`` (through ``ttnn.to_torch``) equal ``expected``."""
    actual, _ = _to_float64(tt_tensor)
    matches = (
        actual is not None
        and list(tt_tensor.shape) == list(expected.shape)
        and actual.shape == expected.shape
        and torch.equal(actual, expected.to(torch.float64))
    )
    assert matches, mismatch_report(label, tt_tensor, expected, torch_input)


def assert_physical_matches(label, tt_tensor, expected_physical, expected_logical_shape=None):
    """Padded shape and physical data (through ``to_torch_with_padded_shape``) equal ``expected_physical``."""
    physical = physical_data(tt_tensor)
    lines = [f"[{label}]", f"actual:   {describe(tt_tensor)}"]
    matches = isinstance(physical, torch.Tensor)
    if matches:
        physical = physical.to(torch.float64)
        matches = physical.shape == expected_physical.shape and torch.equal(
            physical, expected_physical.to(torch.float64)
        )
    if expected_logical_shape is not None and list(tt_tensor.shape) != list(expected_logical_shape):
        matches = False
        lines.append(f"logical_shape {list(tt_tensor.shape)} != expected {list(expected_logical_shape)}")
    lines.append(f"expected padded_shape={list(expected_physical.shape)}")
    lines.append(f"physical data:\n{physical}\nexpected physical data:\n{expected_physical}")
    assert matches, "\n".join(lines)


def assert_same_spec(host_tensor, device_tensor):
    host_spec = (list(host_tensor.shape), list(host_tensor.padded_shape), host_tensor.layout, host_tensor.dtype)
    device_spec = (
        list(device_tensor.shape),
        list(device_tensor.padded_shape),
        device_tensor.layout,
        device_tensor.dtype,
    )
    assert host_spec == device_spec, f"host:   {describe(host_tensor)}\ndevice: {describe(device_tensor)}"
