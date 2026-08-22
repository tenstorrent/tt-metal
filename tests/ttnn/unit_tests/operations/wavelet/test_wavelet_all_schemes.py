# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import pywt
import torch
import ttnn

BOUNDARY_MODES: tuple[str, ...] = (
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_all_106_discrete_schemes_jit_forward_inverse(device: ttnn.MeshDevice, boundary_mode: str) -> None:
    schemes = pywt.wavelist(kind="discrete")
    assert len(schemes) == 106

    signal = torch.sin(torch.arange(257, dtype=torch.float32) * 0.113)
    input_tensor = ttnn.from_torch(
        signal,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for scheme in schemes:
        approximation, detail = ttnn.dwt(input_tensor, scheme, boundary_mode=boundary_mode)
        reconstructed = ttnn.idwt(
            approximation,
            detail,
            scheme,
            signal.numel(),
            boundary_mode=boundary_mode,
        )

        coefficient_length = pywt.dwt_coeff_len(signal.numel(), pywt.Wavelet(scheme).dec_len, mode=boundary_mode)
        assert ttnn.dwt_coeff_len(signal.numel(), scheme) == coefficient_length
        coefficient_sticks = (coefficient_length + 31) // 32
        signal_sticks = (signal.numel() + 31) // 32
        assert tuple(approximation.shape) == (coefficient_sticks, 32)
        assert tuple(detail.shape) == (coefficient_sticks, 32)
        assert tuple(reconstructed.shape) == (signal_sticks, 32)
        assert torch.isfinite(ttnn.to_torch(approximation)).all(), scheme
        assert torch.isfinite(ttnn.to_torch(detail)).all(), scheme
        assert torch.isfinite(ttnn.to_torch(reconstructed)).all(), scheme


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_all_106_discrete_schemes_jit_forward_inverse_2d(
    device: ttnn.MeshDevice,
    boundary_mode: str,
) -> None:
    schemes = pywt.wavelist(kind="discrete")
    assert len(schemes) == 106

    shape = (33, 35)
    y = torch.arange(shape[0], dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(shape[1], dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.17 * x) + torch.cos(0.11 * y)
    input_tensor = ttnn.from_torch(
        signal,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for scheme in schemes:
        outputs = ttnn.dwt_2d(input_tensor, scheme, boundary_mode=boundary_mode)
        reconstructed = ttnn.idwt_2d(
            *outputs,
            scheme,
            shape,
            boundary_mode=boundary_mode,
        )

        wavelet = pywt.Wavelet(scheme)
        coefficient_shape = (
            pywt.dwt_coeff_len(shape[0], wavelet.dec_len, mode=boundary_mode),
            pywt.dwt_coeff_len(shape[1], wavelet.dec_len, mode=boundary_mode),
        )
        for output in outputs:
            assert tuple(output.shape) == coefficient_shape
            assert torch.isfinite(ttnn.to_torch(output)).all(), scheme
        assert tuple(reconstructed.shape) == shape
        assert torch.isfinite(ttnn.to_torch(reconstructed)).all(), scheme
