# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Generated from the declared mathematical mask; no fitted-output oracle.
import numpy as np
import pytest
import torch
import ttnn

_MASK_INTERVALS = {
    "blackhole": ((0, 16256), (32641, 49024), (65409, 65536)),
    "wormhole_b0": ((0, 16256), (32641, 49024), (65409, 65536)),
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


def _expected_words(activation, gradient, architecture):
    assert activation.dtype == gradient.dtype == np.uint16
    assert activation.shape == gradient.shape
    assert _GRADIENT_TRANSPORT[architecture] == "bf16_signed_nonfinite_positive_zero"
    chosen = np.where(_mask(activation, architecture), gradient, np.uint16(0))
    exponent = chosen & np.uint16(0x7F80)
    return np.where(
        exponent == 0, np.uint16(0), np.where(exponent == 0x7F80, chosen & np.uint16(0xFF80), chosen)
    ).astype(np.uint16)


@pytest.mark.skipif(ttnn.get_arch_name() not in _MASK_INTERVALS, reason="generated target unavailable")
def test_hardtanh_bw_bf16_exhaustive(device):
    architecture = ttnn.get_arch_name()
    raw = np.arange(65536, dtype=np.uint16)
    predicate = _mask(raw, architecture)
    assert np.any(predicate) and np.any(~predicate)

    def run(activation, gradient):
        host = torch.from_numpy(activation.copy()).view(torch.bfloat16).reshape(256, 256)
        grad = torch.from_numpy(gradient.copy()).view(torch.bfloat16).reshape(256, 256)
        device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        device_gradient = ttnn.from_torch(grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        result = ttnn.to_torch(
            ttnn.hardtanh_bw(
                device_gradient, device_input, **{"min": -1.0, "max": 1.0, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
            )[0]
        ).to(torch.bfloat16)
        actual = result.contiguous().view(torch.uint16).cpu().numpy().reshape(-1)
        expected = _expected_words(activation, gradient, architecture)
        np.testing.assert_array_equal(actual, expected)

    # Exhaustive activation predicate, then every gradient carrier on each
    # branch. This is separated coverage, not exhaustive tensor-pair sampling.
    run(raw, np.full(raw.shape, 0x3F80, dtype=np.uint16))
    for selected in (False, True):
        representative = raw[np.flatnonzero(predicate == selected)[0]]
        run(np.full(raw.shape, representative, dtype=np.uint16), raw)
    # Correlate hostile gradients with every activation to catch role swaps
    # and accidental factor multiplication across either branch.
    for gradient in (0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0xBF80):
        run(raw, np.full(raw.shape, gradient, dtype=np.uint16))
