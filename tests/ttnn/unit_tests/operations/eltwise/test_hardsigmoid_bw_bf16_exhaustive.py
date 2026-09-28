# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Generated from the declared mathematical mask; no fitted-output oracle.
import numpy as np
import pytest
import torch
import ttnn

_MASK_INTERVALS = {
    "blackhole": ((0, 16448), (32641, 49216), (65409, 65536)),
    "wormhole_b0": ((0, 16448), (32641, 49216), (65409, 65536)),
}
_GRADIENT_TRANSPORT = {
    "blackhole": "bf16_signed_nonfinite_positive_zero",
    "wormhole_b0": "bf16_signed_nonfinite_positive_zero",
}


def _mask(words, architecture):
    intervals = _MASK_INTERVALS[architecture]  # Unknown architectures refuse.
    selected = np.zeros(words.shape, dtype=bool)
    for first, stop in intervals:
        selected |= (words >= first) & (words.astype(np.uint32) < stop)
    return selected


_FACTOR_INTERVALS = {
    "blackhole": ((0, 16448, 16256), (32641, 49216, 16256), (65409, 65536, 16256)),
    "wormhole_b0": ((0, 16448, 16256), (32641, 49216, 16256), (65409, 65536, 16256)),
}
_GRADIENT_SCALE_BITS = {"blackhole": 1042983595, "wormhole_b0": 1042983595}
_NONFINITE_GRADIENT_OUTPUTS = {
    "blackhole": {0: (0, 0, 0, 0), 16256: (32640, 32640, 65408, 32640)},
    "wormhole_b0": {0: (0, 0, 0, 0), 16256: (32640, 32640, 65408, 65408)},
}
_GRADIENT_REPRESENTATIVES = {"blackhole": (16448, 0), "wormhole_b0": (16448, 0)}


def _expected_words(activation, gradient, architecture):
    assert activation.dtype == gradient.dtype == np.uint16
    assert activation.shape == gradient.shape
    factors = np.zeros(activation.shape, np.uint16)
    for first, stop, word in _FACTOR_INTERVALS[architecture]:
        factors[(activation >= first) & (activation.astype(np.uint32) < stop)] = word
    gradient_exponent = gradient & np.uint16(0x7F80)
    values = (gradient.astype(np.uint32) << np.uint32(16)).view(np.float32)
    values = np.where(gradient_exponent == 0, np.float32(0), values)
    scale = _GRADIENT_SCALE_BITS[architecture]
    multiplier = (
        (factors.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if scale is None
        else np.where(factors != 0, np.asarray(scale, np.uint32).view(np.float32), np.float32(0))
    )
    with np.errstate(all="ignore"):
        product = np.asarray(values * multiplier, np.float32).view(np.uint32)
    # Target multiplication flushes before its explicit BF16 conversion.
    product = np.where((product & np.uint32(0x7F800000)) == 0, np.uint32(0), product)
    bias = (
        np.uint32(0x7FFF) + ((product >> np.uint32(16)) & np.uint32(1))
        if architecture == "blackhole"
        else np.uint32(0x8000)
    )
    rounded = ((product + bias) >> np.uint32(16)).astype(np.uint16)
    exponent = rounded & np.uint16(0x7F80)
    result = np.where(
        exponent == 0, np.uint16(0), np.where(exponent == 0x7F80, rounded & np.uint16(0xFF80), rounded)
    ).astype(np.uint16)
    kinds = ((gradient >> np.uint16(15)) * np.uint16(2) + ((gradient & np.uint16(0x7F)) != 0)).astype(np.uint16)
    for factor, outputs in _NONFINITE_GRADIENT_OUTPUTS[architecture].items():
        for kind, word in enumerate(outputs):
            selected = (factors == factor) & (gradient_exponent == 0x7F80) & (kinds == kind)
            result[selected] = word
    return result


@pytest.mark.skipif(ttnn.get_arch_name() not in _MASK_INTERVALS, reason="generated target unavailable")
def test_hardsigmoid_bw_bf16_exhaustive(device):
    architecture = ttnn.get_arch_name()
    raw = np.arange(65536, dtype=np.uint16)
    predicate = _mask(raw, architecture)
    assert _GRADIENT_REPRESENTATIVES[architecture]

    def run(activation, gradient):
        host = torch.from_numpy(activation.copy()).view(torch.bfloat16).reshape(256, 256)
        grad = torch.from_numpy(gradient.copy()).view(torch.bfloat16).reshape(256, 256)
        device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        device_gradient = ttnn.from_torch(grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        result = ttnn.to_torch(
            ttnn.hardsigmoid_bw(device_gradient, device_input, **{"memory_config": ttnn.DRAM_MEMORY_CONFIG})[0]
        ).to(torch.bfloat16)
        actual = result.contiguous().view(torch.uint16).cpu().numpy().reshape(-1)
        expected = _expected_words(activation, gradient, architecture)
        np.testing.assert_array_equal(actual, expected)

    # Exhaustive activation predicate, then every gradient carrier on each
    # branch. This is separated coverage, not exhaustive tensor-pair sampling.
    run(raw, np.full(raw.shape, 0x3F80, dtype=np.uint16))
    for representative in _GRADIENT_REPRESENTATIVES[architecture]:
        run(np.full(raw.shape, representative, dtype=np.uint16), raw)
    # Correlate hostile gradients with every activation to catch role swaps
    # and accidental factor multiplication across either branch.
    for gradient in (0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0xBF80):
        run(raw, np.full(raw.shape, gradient, dtype=np.uint16))
