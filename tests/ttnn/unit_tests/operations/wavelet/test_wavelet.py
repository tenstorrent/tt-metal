# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import pywt
import torch
import ttnn

BOUNDARY_MODES: list[str] = [
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
]

REPRESENTATIVE_SCHEMES: list[str] = ["db1", "db7", "bior3.9", "dmey", "coif17"]


def to_device_1d(
    device: ttnn.MeshDevice,
    value: torch.Tensor,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        value,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def to_device_2d(
    device: ttnn.MeshDevice,
    value: torch.Tensor,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        value,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def assert_fp32_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 5e-5) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=atol)


def assert_fp32_close_1d(
    actual_tensor: ttnn.Tensor,
    expected_tensor: torch.Tensor,
    atol: float = 5e-5,
) -> None:
    actual = ttnn.to_torch(actual_tensor)
    if expected_tensor.ndim == 1:
        actual_valid = actual.flatten()[: expected_tensor.numel()].reshape(expected_tensor.shape)
    else:
        batch = expected_tensor.shape[0]
        valid_length = expected_tensor.shape[-1]
        actual_valid = actual.reshape(batch, 1, -1)[..., :valid_length].reshape(expected_tensor.shape)
    torch.testing.assert_close(actual_valid, expected_tensor, rtol=1e-5, atol=atol)


def assert_fp32_identical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def assert_fp32_identical_1d(actual: torch.Tensor, expected: torch.Tensor, valid_length: int) -> None:
    def valid_prefix(value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 2:
            return value.flatten()[:valid_length]
        return value.reshape(value.shape[0], -1)[:, :valid_length]

    assert_fp32_identical(valid_prefix(actual), valid_prefix(expected))


def stick_shape(valid_length: int, batch: int | None = None) -> tuple[int, ...]:
    sticks = (valid_length + 31) // 32
    return (sticks, 32) if batch is None else (batch, 1, sticks, 32)


@pytest.mark.parametrize("boundary_mode", ["symmetric", "antireflect"])
@pytest.mark.parametrize("wavelet", REPRESENTATIVE_SCHEMES)
def test_representative_schemes_jit_execute_all_operations(
    device: ttnn.MeshDevice, wavelet: str, boundary_mode: str
) -> None:
    signal_1d = torch.sin(torch.arange(257, dtype=torch.float32) * 0.113)
    approximation, detail = ttnn.dwt(to_device_1d(device, signal_1d), wavelet, boundary_mode=boundary_mode)
    reconstructed_1d = ttnn.idwt(
        approximation,
        detail,
        wavelet,
        signal_1d.numel(),
        boundary_mode=boundary_mode,
    )

    shape_2d = (33, 35)
    y = torch.arange(shape_2d[0], dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(shape_2d[1], dtype=torch.float32).reshape(1, -1)
    signal_2d = torch.sin(0.17 * x) + torch.cos(0.11 * y)
    bands = ttnn.dwt_2d(to_device_2d(device, signal_2d), wavelet, boundary_mode=boundary_mode)
    reconstructed_2d = ttnn.idwt_2d(
        *bands,
        wavelet,
        shape_2d,
        boundary_mode=boundary_mode,
    )

    wavelet_spec = pywt.Wavelet(wavelet)
    coefficient_length = pywt.dwt_coeff_len(signal_1d.numel(), wavelet_spec.dec_len, mode=boundary_mode)
    coefficient_shape_2d = tuple(
        pywt.dwt_coeff_len(size, wavelet_spec.dec_len, mode=boundary_mode) for size in shape_2d
    )
    coeff_sticks = (coefficient_length + 31) // 32
    signal_sticks = (signal_1d.numel() + 31) // 32
    assert tuple(approximation.shape) == (coeff_sticks, 32)
    assert tuple(detail.shape) == (coeff_sticks, 32)
    assert tuple(reconstructed_1d.shape) == (signal_sticks, 32)
    assert all(tuple(band.shape) == coefficient_shape_2d for band in bands)
    assert tuple(reconstructed_2d.shape) == shape_2d
    for tensor in (*bands, approximation, detail, reconstructed_1d, reconstructed_2d):
        assert torch.isfinite(ttnn.to_torch(tensor)).all(), (wavelet, boundary_mode)


@pytest.mark.parametrize("wavelet", ["db1", "db7", "bior3.9"])
@pytest.mark.parametrize("length", [32, 33])
def test_batched_1d_matches_independent_samples(device: ttnn.MeshDevice, wavelet: str, length: int) -> None:
    batch = 2
    index = torch.arange(batch * length, dtype=torch.float32).reshape(batch, 1, 1, length)
    signal = torch.sin(index * 0.071) + 0.01 * index

    approximation, detail = ttnn.dwt(to_device_1d(device, signal), wavelet, boundary_mode="antireflect")
    reconstructed = ttnn.idwt(approximation, detail, wavelet, length, boundary_mode="antireflect")
    approximation_host = ttnn.to_torch(approximation)
    detail_host = ttnn.to_torch(detail)
    reconstructed_host = ttnn.to_torch(reconstructed)

    assert approximation_host.shape[:2] == (batch, 1)
    assert detail_host.shape == approximation_host.shape
    assert reconstructed_host.shape[:2] == (batch, 1)
    assert reconstructed_host.shape[3] == 32
    coefficient_length = ttnn.dwt_coeff_len(length, wavelet)
    for batch_index in range(batch):
        sample = signal[batch_index, 0, 0]
        sample_approximation, sample_detail = ttnn.dwt(
            to_device_1d(device, sample), wavelet, boundary_mode="antireflect"
        )
        sample_reconstructed = ttnn.idwt(
            sample_approximation,
            sample_detail,
            wavelet,
            length,
            boundary_mode="antireflect",
        )
        assert_fp32_identical_1d(
            approximation_host[batch_index, 0],
            ttnn.to_torch(sample_approximation),
            coefficient_length,
        )
        assert_fp32_identical_1d(
            detail_host[batch_index, 0],
            ttnn.to_torch(sample_detail),
            coefficient_length,
        )
        assert_fp32_identical_1d(
            reconstructed_host[batch_index, 0],
            ttnn.to_torch(sample_reconstructed),
            length,
        )


@pytest.mark.parametrize("shape", [(32, 34), (33, 35)])
def test_batched_2d_matches_independent_samples(device: ttnn.MeshDevice, shape: tuple[int, int]) -> None:
    batch = 2
    height, width = shape
    values = torch.arange(batch * height * width, dtype=torch.float32).reshape(batch, 1, height, width)
    signal = torch.sin(values * 0.017) + torch.cos(values * 0.003)

    bands = ttnn.dwt_2d(to_device_2d(device, signal), "db7", boundary_mode="antireflect")
    reconstructed = ttnn.idwt_2d(*bands, "db7", shape, boundary_mode="antireflect")
    band_hosts = [ttnn.to_torch(band) for band in bands]
    reconstructed_host = ttnn.to_torch(reconstructed)
    assert all(band.shape[:2] == (batch, 1) for band in band_hosts)
    assert reconstructed_host.shape == signal.shape

    for batch_index in range(batch):
        sample = signal[batch_index, 0]
        sample_bands = ttnn.dwt_2d(to_device_2d(device, sample), "db7", boundary_mode="antireflect")
        sample_reconstructed = ttnn.idwt_2d(*sample_bands, "db7", shape, boundary_mode="antireflect")
        for batch_band, sample_band in zip(band_hosts, sample_bands):
            assert_fp32_identical(batch_band[batch_index, 0], ttnn.to_torch(sample_band))
        assert_fp32_identical(reconstructed_host[batch_index, 0], ttnn.to_torch(sample_reconstructed))


def test_large_batched_inputs_and_interleaved_l1(device: ttnn.MeshDevice) -> None:
    batch = 2
    length = 65_537
    values = torch.arange(batch * length, dtype=torch.float32).reshape(batch, 1, 1, length)
    signal = torch.sin(values * 0.013) + values * 1.0e-5
    dram = ttnn.dwt(to_device_1d(device, signal), "bior3.9", boundary_mode="symmetric")
    l1_input = to_device_1d(device, signal, ttnn.L1_MEMORY_CONFIG)
    l1 = ttnn.dwt(l1_input, "bior3.9", boundary_mode="symmetric")
    coefficient_length = ttnn.dwt_coeff_len(length, "bior3.9")
    for actual, expected in zip(l1, dram):
        assert_fp32_identical_1d(ttnn.to_torch(actual), ttnn.to_torch(expected), coefficient_length)

    reconstructed_dram = ttnn.idwt(*dram, "bior3.9", length, boundary_mode="symmetric")
    l1_coefficients = tuple(to_device_1d(device, ttnn.to_torch(tensor), ttnn.L1_MEMORY_CONFIG) for tensor in dram)
    reconstructed_l1 = ttnn.idwt(*l1_coefficients, "bior3.9", length, boundary_mode="symmetric")
    assert_fp32_identical_1d(ttnn.to_torch(reconstructed_l1), ttnn.to_torch(reconstructed_dram), length)


def test_batch_larger_than_worker_count_and_coif17_execution(
    device: ttnn.MeshDevice,
) -> None:
    batch = 113
    length = 17
    signal = torch.arange(batch * length, dtype=torch.float32).reshape(batch, 1, 1, length)
    approximation, detail = ttnn.dwt(to_device_1d(device, signal), "coif17")
    reconstructed = ttnn.idwt(approximation, detail, "coif17", length)
    assert tuple(reconstructed.shape) == stick_shape(length, batch)
    assert torch.isfinite(ttnn.to_torch(reconstructed)).all()


def test_rank_four_batch_one_preserves_shapes(device: ttnn.MeshDevice) -> None:
    signal_1d = torch.arange(33, dtype=torch.float32).reshape(1, 1, 1, 33)
    coefficients = ttnn.dwt(to_device_1d(device, signal_1d), "db1")
    reconstructed_1d = ttnn.idwt(*coefficients, "db1", 33)
    assert tuple(reconstructed_1d.shape) == (1, 1, 2, 32)

    signal_2d = torch.arange(33 * 35, dtype=torch.float32).reshape(1, 1, 33, 35)
    bands = ttnn.dwt_2d(to_device_2d(device, signal_2d), "db1")
    reconstructed_2d = ttnn.idwt_2d(*bands, "db1", (33, 35))
    assert tuple(reconstructed_2d.shape) == (1, 1, 33, 35)


def test_large_batched_2d_interleaved_l1_matches_dram(device: ttnn.MeshDevice) -> None:
    batch, height, width = 2, 257, 259
    values = torch.arange(batch * height * width, dtype=torch.float32).reshape(batch, 1, height, width)
    signal = torch.sin(values * 0.017) + torch.cos(values * 0.019)
    dram_bands = ttnn.dwt_2d(to_device_2d(device, signal), "bior1.3", boundary_mode="antireflect")
    l1_bands = ttnn.dwt_2d(
        to_device_2d(device, signal, ttnn.L1_MEMORY_CONFIG),
        "bior1.3",
        boundary_mode="antireflect",
    )
    for actual, expected in zip(l1_bands, dram_bands):
        assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))

    dram_reconstructed = ttnn.idwt_2d(*dram_bands, "bior1.3", (height, width), boundary_mode="antireflect")
    input_bands_l1 = tuple(to_device_2d(device, ttnn.to_torch(band), ttnn.L1_MEMORY_CONFIG) for band in dram_bands)
    l1_reconstructed = ttnn.idwt_2d(
        *input_bands_l1,
        "bior1.3",
        (height, width),
        boundary_mode="antireflect",
    )
    assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))


def test_batched_preallocated_outputs_and_program_cache(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal_1d = torch.arange(2 * 33, dtype=torch.float32).reshape(2, 1, 1, 33)
        coefficients = ttnn.dwt(to_device_1d(device, signal_1d), "db1")
        reconstructed_1d = ttnn.idwt(*coefficients, "db1", 33)

        signal_2d = torch.arange(2 * 33 * 35, dtype=torch.float32).reshape(2, 1, 33, 35)
        bands = ttnn.dwt_2d(to_device_2d(device, signal_2d), "db1")
        reconstructed_2d = ttnn.idwt_2d(*bands, "db1", (33, 35))

        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for scale in (-0.5, 0.25):
            next_1d = signal_1d * scale + 3.0
            next_coefficients = ttnn.dwt(to_device_1d(device, next_1d), "db1", output_tensors=coefficients)
            next_reconstructed_1d = ttnn.idwt(*next_coefficients, "db1", 33, output_tensor=reconstructed_1d)
            assert_fp32_close_1d(next_reconstructed_1d, next_1d)

            next_2d = signal_2d * scale + 3.0
            next_bands = ttnn.dwt_2d(to_device_2d(device, next_2d), "db1", output_tensors=bands)
            next_reconstructed_2d = ttnn.idwt_2d(*next_bands, "db1", (33, 35), output_tensor=reconstructed_2d)
            assert_fp32_close(ttnn.to_torch(next_reconstructed_2d), next_2d)

        assert device.num_program_cache_entries() == 4
    finally:
        device.disable_and_clear_program_cache()


def test_batched_channel_validation(device: ttnn.MeshDevice, expect_error) -> None:
    invalid_1d = torch.zeros((2, 2, 1, 33), dtype=torch.float32)
    with expect_error(RuntimeError, "C == 1"):
        ttnn.dwt(to_device_1d(device, invalid_1d), "db1")

    invalid_2d = torch.zeros((2, 2, 33, 35), dtype=torch.float32)
    with expect_error(RuntimeError, "C == 1"):
        ttnn.dwt_2d(to_device_2d(device, invalid_2d), "db1")


@pytest.mark.parametrize("length", [20, 31, 32, 33])
def test_lwt_ilwt_1d_stick_padding_regression(device: ttnn.MeshDevice, length: int) -> None:
    indices = torch.arange(length, dtype=torch.float32)
    signal = 0.125 * indices + torch.sin(0.7 * indices)
    approximation_ref, detail_ref = pywt.dwt(signal.numpy(), "bior1.3", mode="symmetric")

    approximation, detail = ttnn.dwt(to_device_1d(device, signal), "bior1.3", boundary_mode="symmetric")

    coeff_sticks = (len(approximation_ref) + 31) // 32
    assert tuple(approximation.shape) == (coeff_sticks, 32)
    assert tuple(detail.shape) == (coeff_sticks, 32)
    assert_fp32_close_1d(approximation, torch.from_numpy(approximation_ref))
    assert_fp32_close_1d(detail, torch.from_numpy(detail_ref))

    reconstructed = ttnn.idwt(
        approximation,
        detail,
        "bior1.3",
        length,
        boundary_mode="symmetric",
    )
    assert_fp32_close_1d(reconstructed, signal)


@pytest.mark.parametrize(
    ("signal_length", "coefficient_length"),
    [(61, 31), (63, 32), (65, 33), (125, 63), (127, 64), (129, 65)],
)
def test_1d_explicit_valid_length_and_page_contract(
    device: ttnn.MeshDevice, signal_length: int, coefficient_length: int
) -> None:
    signal = torch.sin(torch.arange(signal_length, dtype=torch.float32) * 0.071)
    input_tensor = to_device_1d(device, signal)
    approximation, detail = ttnn.dwt(input_tensor, "db1")
    reconstructed = ttnn.idwt(approximation, detail, "db1", signal_length)

    assert ttnn.dwt_coeff_len(signal_length, "db1") == coefficient_length
    assert tuple(approximation.shape) == stick_shape(coefficient_length)
    assert tuple(detail.shape) == stick_shape(coefficient_length)
    assert tuple(reconstructed.shape) == stick_shape(signal_length)
    for tensor in (approximation, detail, reconstructed):
        assert tensor.buffer_page_size() == 32 * torch.float32.itemsize
        assert tensor.buffer_num_pages() == tensor.shape[-2]

    approximation_ref, detail_ref = pywt.dwt(signal.numpy(), "db1", mode="symmetric")
    assert_fp32_close_1d(approximation, torch.from_numpy(approximation_ref))
    assert_fp32_close_1d(detail, torch.from_numpy(detail_ref))
    assert_fp32_close_1d(reconstructed, signal)


def test_batched_1d_explicit_valid_length_and_page_contract(
    device: ttnn.MeshDevice,
) -> None:
    batch, signal_length = 3, 65
    signal = torch.arange(batch * signal_length, dtype=torch.float32).reshape(batch, 1, 1, signal_length)
    approximation, detail = ttnn.dwt(to_device_1d(device, signal), "db1")
    reconstructed = ttnn.idwt(approximation, detail, "db1", signal_length)

    coefficient_length = ttnn.dwt_coeff_len(signal_length, "db1")
    assert tuple(approximation.shape) == stick_shape(coefficient_length, batch)
    assert tuple(detail.shape) == stick_shape(coefficient_length, batch)
    assert tuple(reconstructed.shape) == stick_shape(signal_length, batch)
    assert approximation.buffer_page_size() == 32 * torch.float32.itemsize
    assert approximation.buffer_num_pages() == batch * approximation.shape[-2]
    assert reconstructed.buffer_page_size() == 32 * torch.float32.itemsize
    assert reconstructed.buffer_num_pages() == batch * reconstructed.shape[-2]
    reconstructed_valid = ttnn.to_torch(reconstructed).reshape(batch, 1, -1)[..., :signal_length]
    assert_fp32_close(reconstructed_valid.reshape(signal.shape), signal)


@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_lwt_ilwt_1d_boundary_modes(device: ttnn.MeshDevice, boundary_mode: str) -> None:
    signal = torch.linspace(-1.25, 2.75, 33, dtype=torch.float32)
    approximation_ref, detail_ref = pywt.dwt(signal.numpy(), "bior1.3", mode=boundary_mode)

    approximation, detail = ttnn.dwt(
        to_device_1d(device, signal),
        "bior1.3",
        boundary_mode=boundary_mode,
    )
    assert_fp32_close_1d(approximation, torch.from_numpy(approximation_ref))
    assert_fp32_close_1d(detail, torch.from_numpy(detail_ref))

    reconstructed = ttnn.idwt(
        approximation,
        detail,
        "bior1.3",
        signal.numel(),
        boundary_mode=boundary_mode,
    )
    assert_fp32_close_1d(reconstructed, signal)


def test_ilwt_1d_external_coefficients_shorter_than_one_stick(
    device: ttnn.MeshDevice,
) -> None:
    signal = torch.arange(20, dtype=torch.float32) ** 2 * 0.03125 - torch.arange(20, dtype=torch.float32) * 0.25
    approximation, detail = pywt.dwt(signal.numpy(), "bior1.3", mode="symmetric")
    assert approximation.size < 32
    assert detail.size < 32

    reconstructed = ttnn.idwt(
        to_device_1d(device, torch.from_numpy(approximation)),
        to_device_1d(device, torch.from_numpy(detail)),
        "bior1.3",
        signal.numel(),
        boundary_mode="symmetric",
    )
    assert_fp32_close_1d(reconstructed, signal)


def test_ilwt_1d_batched_external_canonical_coefficients_more_than_one_stick(
    device: ttnn.MeshDevice,
) -> None:
    batch = 2
    original_length = 65
    wavelet = "db4"
    values = torch.arange(batch * original_length, dtype=torch.float32).reshape(batch, 1, 1, original_length)
    signals = torch.sin(values * 0.071) + values * 0.002

    approximation_references: list[torch.Tensor] = []
    detail_references: list[torch.Tensor] = []
    reconstruction_references: list[torch.Tensor] = []
    for batch_index in range(batch):
        approximation, detail = pywt.dwt(
            signals[batch_index, 0, 0].numpy(),
            wavelet,
            mode="symmetric",
        )
        reconstructed = pywt.idwt(approximation, detail, wavelet, mode="symmetric")[:original_length]
        approximation_references.append(torch.from_numpy(approximation))
        detail_references.append(torch.from_numpy(detail))
        reconstruction_references.append(torch.from_numpy(reconstructed))

    coefficient_length = approximation_references[0].numel()
    assert coefficient_length == ttnn.dwt_coeff_len(original_length, wavelet)
    assert coefficient_length > 32

    approximation_values = torch.stack(approximation_references).reshape(batch, 1, 1, coefficient_length)
    detail_values = torch.stack(detail_references).reshape(batch, 1, 1, coefficient_length)
    assert tuple(approximation_values.shape) == (batch, 1, 1, coefficient_length)
    assert tuple(detail_values.shape) == (batch, 1, 1, coefficient_length)

    reconstructed = ttnn.idwt(
        to_device_1d(device, approximation_values),
        to_device_1d(device, detail_values),
        wavelet,
        original_length,
        boundary_mode="symmetric",
    )
    reconstructed_values = ttnn.to_torch(reconstructed).reshape(batch, 1, -1)[..., :original_length]

    for batch_index, reference in enumerate(reconstruction_references):
        assert_fp32_close(reconstructed_values[batch_index, 0], reference)


def test_wavelet_1d_interleaved_l1_input_matches_dram_multichunk(
    device: ttnn.MeshDevice,
) -> None:
    length = 65_537
    indices = torch.arange(length, dtype=torch.float32)
    signal = torch.sin(indices * 0.013) + indices * 1.0e-5

    dram_outputs = ttnn.dwt(to_device_1d(device, signal), "bior1.3", boundary_mode="antireflect")
    l1_input = to_device_1d(device, signal, ttnn.L1_MEMORY_CONFIG)
    l1_outputs = ttnn.dwt(l1_input, "bior1.3", boundary_mode="antireflect")
    coefficient_length = ttnn.dwt_coeff_len(length, "bior1.3")
    for actual, expected in zip(l1_outputs, dram_outputs):
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert_fp32_identical_1d(ttnn.to_torch(actual), ttnn.to_torch(expected), coefficient_length)
    ttnn.deallocate(l1_input)

    dram_reconstructed = ttnn.idwt(*dram_outputs, "bior1.3", length, boundary_mode="antireflect")
    approximation_host, detail_host = (ttnn.to_torch(tensor) for tensor in dram_outputs)
    approximation_l1 = to_device_1d(device, approximation_host, ttnn.L1_MEMORY_CONFIG)
    detail_l1 = to_device_1d(device, detail_host, ttnn.L1_MEMORY_CONFIG)

    l1_reconstructed = ttnn.idwt(
        approximation_l1,
        detail_l1,
        "bior1.3",
        length,
        boundary_mode="antireflect",
    )
    mixed_reconstructed = ttnn.idwt(
        approximation_l1,
        dram_outputs[1],
        "bior1.3",
        length,
        boundary_mode="antireflect",
    )
    assert_fp32_identical_1d(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed), length)
    assert_fp32_identical_1d(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed), length)
    assert l1_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert mixed_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("shape", [(32, 32), (33, 31), (35, 37)])
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_lwt_ilwt_2d_shapes_and_boundary_modes(
    device: ttnn.MeshDevice, shape: tuple[int, int], boundary_mode: str
) -> None:
    height, width = shape
    y = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(width, dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.17 * x) + torch.cos(0.11 * y) + 0.01 * x - 0.02 * y

    ll_ref, (hl_ref, lh_ref, hh_ref) = pywt.dwt2(signal.numpy(), "bior1.3", mode=boundary_mode)
    ll, lh, hl, hh = ttnn.dwt_2d(
        to_device_2d(device, signal),
        "bior1.3",
        boundary_mode=boundary_mode,
    )

    # TTNN names bands by (vertical, horizontal) result. PyWavelets returns
    # its horizontal-detail band before its vertical-detail band.
    references = [ll_ref, lh_ref, hl_ref, hh_ref]
    for result, reference in zip((ll, lh, hl, hh), references):
        assert tuple(result.shape) == reference.shape
        assert_fp32_close(ttnn.to_torch(result), torch.from_numpy(reference))

    reconstructed = ttnn.idwt_2d(
        ll,
        lh,
        hl,
        hh,
        "bior1.3",
        shape,
        boundary_mode=boundary_mode,
    )
    assert_fp32_close(ttnn.to_torch(reconstructed), signal)


def test_wavelet_2d_interleaved_l1_input_matches_dram_multichunk(
    device: ttnn.MeshDevice,
) -> None:
    shape = (257, 259)
    y = torch.arange(shape[0], dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(shape[1], dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.017 * x) + torch.cos(0.019 * y) + 1.0e-4 * x * y

    dram_outputs = ttnn.dwt_2d(to_device_2d(device, signal), "bior1.3", boundary_mode="antireflect")
    l1_input = to_device_2d(device, signal, ttnn.L1_MEMORY_CONFIG)
    l1_outputs = ttnn.dwt_2d(l1_input, "bior1.3", boundary_mode="antireflect")
    for actual, expected in zip(l1_outputs, dram_outputs):
        assert actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))
    ttnn.deallocate(l1_input)

    dram_reconstructed = ttnn.idwt_2d(
        *dram_outputs,
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    l1_bands = tuple(to_device_2d(device, ttnn.to_torch(tensor), ttnn.L1_MEMORY_CONFIG) for tensor in dram_outputs)
    l1_reconstructed = ttnn.idwt_2d(
        *l1_bands,
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    mixed_reconstructed = ttnn.idwt_2d(
        l1_bands[0],
        dram_outputs[1],
        l1_bands[2],
        dram_outputs[3],
        "bior1.3",
        shape,
        boundary_mode="antireflect",
    )
    assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    assert l1_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert mixed_reconstructed.memory_config() == ttnn.DRAM_MEMORY_CONFIG


def test_wavelet_preallocated_outputs_and_program_cache(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.arange(20, dtype=torch.float32)
        input_tensor = to_device_1d(device, signal)
        approximation, detail = ttnn.dwt(input_tensor, "bior1.3")
        reconstructed = ttnn.idwt(approximation, detail, "bior1.3", signal.numel())

        # Retain correctly specified output tensors, then isolate the two
        # preallocated operation cache entries from the allocation run above.
        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for scale in (-2.5, 0.375):
            next_signal = signal * scale + 7.0
            approximation_out, detail_out = ttnn.dwt(
                to_device_1d(device, next_signal),
                "bior1.3",
                output_tensors=(approximation, detail),
            )
            reconstructed_out = ttnn.idwt(
                approximation_out,
                detail_out,
                "bior1.3",
                signal.numel(),
                output_tensor=reconstructed,
            )
            assert approximation_out.buffer_address() == approximation.buffer_address()
            assert detail_out.buffer_address() == detail.buffer_address()
            assert reconstructed_out.buffer_address() == reconstructed.buffer_address()
            assert_fp32_close_1d(reconstructed_out, next_signal)

        assert device.num_program_cache_entries() == 2
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("batch", [None, 2])
def test_python_allocated_1d_preallocated_outputs(device: ttnn.MeshDevice, batch: int | None) -> None:
    length = 65
    shape = (length,) if batch is None else (batch, 1, 1, length)
    element_count = length if batch is None else batch * length
    signal = torch.sin(torch.arange(element_count, dtype=torch.float32).reshape(shape) * 0.071)
    input_tensor = to_device_1d(device, signal)
    expected_approximation, expected_detail = ttnn.dwt(input_tensor, "db4", boundary_mode="symmetric")
    coefficient_length = ttnn.dwt_coeff_len(length, "db4")

    coefficient_spec = ttnn.TensorSpec(expected_approximation.shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    approximation = ttnn.allocate_tensor_on_device(coefficient_spec, device)
    detail = ttnn.allocate_tensor_on_device(coefficient_spec, device)
    actual_approximation, actual_detail = ttnn.dwt(
        input_tensor,
        "db4",
        boundary_mode="symmetric",
        output_tensors=(approximation, detail),
    )
    assert actual_approximation.buffer_address() == approximation.buffer_address()
    assert actual_detail.buffer_address() == detail.buffer_address()
    assert_fp32_identical_1d(
        ttnn.to_torch(actual_approximation),
        ttnn.to_torch(expected_approximation),
        coefficient_length,
    )
    assert_fp32_identical_1d(
        ttnn.to_torch(actual_detail),
        ttnn.to_torch(expected_detail),
        coefficient_length,
    )

    expected_reconstructed = ttnn.idwt(
        expected_approximation,
        expected_detail,
        "db4",
        length,
        boundary_mode="symmetric",
    )
    output_spec = ttnn.TensorSpec(expected_reconstructed.shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.allocate_tensor_on_device(output_spec, device)
    actual_reconstructed = ttnn.idwt(
        actual_approximation,
        actual_detail,
        "db4",
        length,
        boundary_mode="symmetric",
        output_tensor=output,
    )
    assert actual_reconstructed.buffer_address() == output.buffer_address()
    assert_fp32_identical_1d(
        ttnn.to_torch(actual_reconstructed),
        ttnn.to_torch(expected_reconstructed),
        length,
    )


def test_wavelet_2d_preallocated_outputs_and_program_cache(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.001
        input_tensor = to_device_2d(device, signal)
        outputs = ttnn.dwt_2d(input_tensor, "bior1.3")
        reconstructed = ttnn.idwt_2d(*outputs, "bior1.3", shape)

        device.disable_and_clear_program_cache()
        device.enable_program_cache()
        for scale in (-0.25, 0.5):
            next_signal = 1.0 + signal * scale
            next_outputs = ttnn.dwt_2d(
                to_device_2d(device, next_signal),
                "bior1.3",
                output_tensors=outputs,
            )
            reconstructed_out = ttnn.idwt_2d(
                *next_outputs,
                "bior1.3",
                shape,
                output_tensor=reconstructed,
            )
            assert all(
                result.buffer_address() == output.buffer_address() for result, output in zip(next_outputs, outputs)
            )
            assert reconstructed_out.buffer_address() == reconstructed.buffer_address()
            assert_fp32_close(ttnn.to_torch(reconstructed_out), next_signal)

        assert device.num_program_cache_entries() == 2
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_operations_with_program_cache_disabled(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    try:
        signal_1d = torch.linspace(-1.0, 1.0, 33, dtype=torch.float32)
        approximation, detail = ttnn.dwt(to_device_1d(device, signal_1d), "bior1.3")
        reconstructed_1d = ttnn.idwt(approximation, detail, "bior1.3", signal_1d.numel())
        assert_fp32_close_1d(reconstructed_1d, signal_1d)

        signal_2d = signal_1d.reshape(3, 11)
        bands = ttnn.dwt_2d(to_device_2d(device, signal_2d), "bior1.3")
        reconstructed_2d = ttnn.idwt_2d(*bands, "bior1.3", signal_2d.shape)
        assert_fp32_close(ttnn.to_torch(reconstructed_2d), signal_2d)
        assert device.num_program_cache_entries() == 0
    finally:
        device.enable_program_cache()


def test_wavelet_1d_program_cache_keys_and_address_override(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.sin(torch.arange(33, dtype=torch.float32) * 0.17)
        first_input = to_device_1d(device, signal)
        first_outputs = ttnn.dwt(first_input, "db7")
        assert device.num_program_cache_entries() == 1

        # Identical tensors and new buffers with identical specs both reuse the
        # program. Tensor addresses are runtime arguments, not cache-key data.
        ttnn.dwt(first_input, "db7")
        second_input = to_device_1d(device, signal + 0.25)
        second_outputs = ttnn.dwt(second_input, "db7")
        assert device.num_program_cache_entries() == 1

        db8_outputs = ttnn.dwt(second_input, "db8")
        assert device.num_program_cache_entries() == 2

        reconstructed = ttnn.idwt(*first_outputs, "db7", signal.numel())
        ttnn.idwt(*first_outputs, "db7", signal.numel())
        ttnn.idwt(*second_outputs, "db7", signal.numel())
        assert device.num_program_cache_entries() == 3
        assert_fp32_close_1d(reconstructed, signal, atol=2e-4)

        ttnn.idwt(*db8_outputs, "db8", signal.numel())
        assert device.num_program_cache_entries() == 4
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_2d_program_cache_keys_and_address_override(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.sin(torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.013)
        first_input = to_device_2d(device, signal)
        first_outputs = ttnn.dwt_2d(first_input, "db7")
        assert device.num_program_cache_entries() == 1

        ttnn.dwt_2d(first_input, "db7")
        second_input = to_device_2d(device, signal + 0.25)
        second_outputs = ttnn.dwt_2d(second_input, "db7")
        assert device.num_program_cache_entries() == 1

        db8_outputs = ttnn.dwt_2d(second_input, "db8")
        assert device.num_program_cache_entries() == 2

        reconstructed = ttnn.idwt_2d(*first_outputs, "db7", shape)
        ttnn.idwt_2d(*first_outputs, "db7", shape)
        ttnn.idwt_2d(*second_outputs, "db7", shape)
        assert device.num_program_cache_entries() == 3
        assert_fp32_close(ttnn.to_torch(reconstructed), signal, atol=2e-4)

        ttnn.idwt_2d(*db8_outputs, "db8", shape)
        assert device.num_program_cache_entries() == 4
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_1d_interleaved_l1_program_cache_keys_and_address_override(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal = torch.sin(torch.arange(33, dtype=torch.float32) * 0.17)
        dram_input_a = to_device_1d(device, signal)
        dram_outputs = ttnn.dwt(dram_input_a, "bior1.3")
        ttnn.dwt(to_device_1d(device, signal + 0.25), "bior1.3")
        assert device.num_program_cache_entries() == 1

        l1_input_a = to_device_1d(device, signal, ttnn.L1_MEMORY_CONFIG)
        l1_input_b = to_device_1d(device, signal + 0.25, ttnn.L1_MEMORY_CONFIG)
        l1_outputs = ttnn.dwt(l1_input_a, "bior1.3")
        ttnn.dwt(l1_input_b, "bior1.3")
        assert device.num_program_cache_entries() == 2
        coefficient_length = ttnn.dwt_coeff_len(signal.numel(), "bior1.3")
        for actual, expected in zip(l1_outputs, dram_outputs):
            assert_fp32_identical_1d(ttnn.to_torch(actual), ttnn.to_torch(expected), coefficient_length)

        coefficient_values = tuple(ttnn.to_torch(tensor) for tensor in dram_outputs)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()

        dram_coefficients_a = tuple(to_device_1d(device, tensor) for tensor in coefficient_values)
        dram_coefficients_b = tuple(to_device_1d(device, tensor + 0.125) for tensor in coefficient_values)
        dram_reconstructed = ttnn.idwt(*dram_coefficients_a, "bior1.3", signal.numel())
        ttnn.idwt(*dram_coefficients_b, "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 1

        l1_coefficients_a = tuple(to_device_1d(device, tensor, ttnn.L1_MEMORY_CONFIG) for tensor in coefficient_values)
        l1_coefficients_b = tuple(
            to_device_1d(device, tensor + 0.125, ttnn.L1_MEMORY_CONFIG) for tensor in coefficient_values
        )
        l1_reconstructed = ttnn.idwt(*l1_coefficients_a, "bior1.3", signal.numel())
        ttnn.idwt(*l1_coefficients_b, "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 2

        mixed_reconstructed = ttnn.idwt(l1_coefficients_a[0], dram_coefficients_a[1], "bior1.3", signal.numel())
        ttnn.idwt(l1_coefficients_b[0], dram_coefficients_b[1], "bior1.3", signal.numel())
        assert device.num_program_cache_entries() == 3
        assert_fp32_identical_1d(
            ttnn.to_torch(l1_reconstructed),
            ttnn.to_torch(dram_reconstructed),
            signal.numel(),
        )
        assert_fp32_identical_1d(
            ttnn.to_torch(mixed_reconstructed),
            ttnn.to_torch(dram_reconstructed),
            signal.numel(),
        )
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_2d_interleaved_l1_program_cache_keys_and_address_override(
    device: ttnn.MeshDevice,
) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        shape = (35, 37)
        signal = torch.sin(torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) * 0.013)
        dram_input_a = to_device_2d(device, signal)
        dram_outputs = ttnn.dwt_2d(dram_input_a, "bior1.3")
        ttnn.dwt_2d(to_device_2d(device, signal + 0.25), "bior1.3")
        assert device.num_program_cache_entries() == 1

        l1_input_a = to_device_2d(device, signal, ttnn.L1_MEMORY_CONFIG)
        l1_input_b = to_device_2d(device, signal + 0.25, ttnn.L1_MEMORY_CONFIG)
        l1_outputs = ttnn.dwt_2d(l1_input_a, "bior1.3")
        ttnn.dwt_2d(l1_input_b, "bior1.3")
        assert device.num_program_cache_entries() == 2
        for actual, expected in zip(l1_outputs, dram_outputs):
            assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))

        band_values = tuple(ttnn.to_torch(tensor) for tensor in dram_outputs)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()

        dram_bands_a = tuple(to_device_2d(device, tensor) for tensor in band_values)
        dram_bands_b = tuple(to_device_2d(device, tensor + 0.125) for tensor in band_values)
        dram_reconstructed = ttnn.idwt_2d(*dram_bands_a, "bior1.3", shape)
        ttnn.idwt_2d(*dram_bands_b, "bior1.3", shape)
        assert device.num_program_cache_entries() == 1

        l1_bands_a = tuple(to_device_2d(device, tensor, ttnn.L1_MEMORY_CONFIG) for tensor in band_values)
        l1_bands_b = tuple(to_device_2d(device, tensor + 0.125, ttnn.L1_MEMORY_CONFIG) for tensor in band_values)
        l1_reconstructed = ttnn.idwt_2d(*l1_bands_a, "bior1.3", shape)
        ttnn.idwt_2d(*l1_bands_b, "bior1.3", shape)
        assert device.num_program_cache_entries() == 2

        mixed_reconstructed = ttnn.idwt_2d(
            l1_bands_a[0],
            dram_bands_a[1],
            l1_bands_a[2],
            dram_bands_a[3],
            "bior1.3",
            shape,
        )
        ttnn.idwt_2d(
            l1_bands_b[0],
            dram_bands_b[1],
            l1_bands_b[2],
            dram_bands_b[3],
            "bior1.3",
            shape,
        )
        assert device.num_program_cache_entries() == 3
        assert_fp32_identical(ttnn.to_torch(l1_reconstructed), ttnn.to_torch(dram_reconstructed))
        assert_fp32_identical(ttnn.to_torch(mixed_reconstructed), ttnn.to_torch(dram_reconstructed))
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_program_cache_specializes_for_available_l1_budget(device: ttnn.MeshDevice) -> None:
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        signal_1d = torch.sin(torch.arange(257, dtype=torch.float32) * 0.017)
        input_1d = to_device_1d(device, signal_1d)
        coefficients = ttnn.dwt(input_1d, "db1")
        reconstructed_1d = ttnn.idwt(*coefficients, "db1", signal_1d.numel())

        shape_2d = (65, 67)
        signal_2d = torch.sin(torch.arange(shape_2d[0] * shape_2d[1], dtype=torch.float32).reshape(shape_2d) * 0.013)
        input_2d = to_device_2d(device, signal_2d)
        bands = ttnn.dwt_2d(input_2d, "db1")
        reconstructed_2d = ttnn.idwt_2d(*bands, "db1", shape_2d)
        assert device.num_program_cache_entries() == 4

        pressure = ttnn.allocate_tensor_on_device(
            (512, 1024),
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )
        assert pressure.buffer_address() != 0

        pressured_coefficients = ttnn.dwt(input_1d, "db1")
        pressured_reconstructed_1d = ttnn.idwt(*coefficients, "db1", signal_1d.numel())
        pressured_bands = ttnn.dwt_2d(input_2d, "db1")
        pressured_reconstructed_2d = ttnn.idwt_2d(*bands, "db1", shape_2d)
        assert device.num_program_cache_entries() == 8

        ttnn.dwt(input_1d, "db1")
        ttnn.idwt(*coefficients, "db1", signal_1d.numel())
        ttnn.dwt_2d(input_2d, "db1")
        ttnn.idwt_2d(*bands, "db1", shape_2d)
        assert device.num_program_cache_entries() == 8

        coefficient_length = ttnn.dwt_coeff_len(signal_1d.numel(), "db1")
        for actual, expected in zip(pressured_coefficients, coefficients):
            assert_fp32_identical_1d(ttnn.to_torch(actual), ttnn.to_torch(expected), coefficient_length)
        assert_fp32_identical_1d(
            ttnn.to_torch(pressured_reconstructed_1d),
            ttnn.to_torch(reconstructed_1d),
            signal_1d.numel(),
        )
        for actual, expected in zip(pressured_bands, bands):
            assert_fp32_identical(ttnn.to_torch(actual), ttnn.to_torch(expected))
        assert_fp32_identical(ttnn.to_torch(pressured_reconstructed_2d), ttnn.to_torch(reconstructed_2d))
    finally:
        device.disable_and_clear_program_cache()


def test_wavelet_1d_validation_errors(device: ttnn.MeshDevice, expect_error) -> None:
    signal = torch.arange(20, dtype=torch.float32)
    input_tensor = to_device_1d(device, signal)

    with expect_error(RuntimeError, "wavelet"):
        ttnn.dwt(input_tensor, "not-a-wavelet")
    with expect_error(RuntimeError, "boundary"):
        ttnn.dwt(input_tensor, "bior1.3", boundary_mode="not-a-mode")
    with expect_error(RuntimeError, "device tensor"):
        ttnn.dwt(ttnn.from_torch(signal, layout=ttnn.ROW_MAJOR_LAYOUT), "bior1.3")
    with expect_error(RuntimeError, "FLOAT32"):
        ttnn.dwt(
            ttnn.from_torch(signal, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            "bior1.3",
        )
    with expect_error(RuntimeError, "got rank 2"):
        ttnn.dwt(to_device_1d(device, torch.zeros((2, 32), dtype=torch.float32)), "bior1.3")
    with expect_error(RuntimeError, "requires H == 1"):
        ttnn.dwt(to_device_1d(device, torch.zeros((2, 1, 2, 32), dtype=torch.float32)), "bior1.3")
    with expect_error(RuntimeError, "DRAM-interleaved outputs"):
        ttnn.dwt(input_tensor, "bior1.3", memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error(RuntimeError, "greater than one"):
        ttnn.dwt(to_device_1d(device, torch.ones(1)), "bior1.3", boundary_mode="reflect")

    approximation, detail = ttnn.dwt(input_tensor, "bior1.3")
    wrong_detail = to_device_1d(device, torch.zeros(detail.shape[0] + 1))
    with expect_error(RuntimeError, "identical shapes"):
        ttnn.idwt(approximation, wrong_detail, "bior1.3", signal.numel())
    with expect_error(RuntimeError, "greater than zero"):
        ttnn.idwt(approximation, detail, "bior1.3", 0)

    wrong_output = to_device_1d(device, torch.empty(approximation.shape[0] + 1))
    with expect_error(RuntimeError, "does not match"):
        ttnn.dwt(input_tensor, "bior1.3", output_tensors=(wrong_output, wrong_output))
    with expect_error(RuntimeError, "must not alias"):
        ttnn.dwt(input_tensor, "bior1.3", output_tensors=(approximation, approximation))

    alias_input = to_device_1d(device, torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32))
    _, alias_detail = ttnn.dwt(alias_input, "db1")
    with expect_error(RuntimeError, "must not alias the input"):
        ttnn.dwt(alias_input, "db1", output_tensors=(alias_input, alias_detail))


def test_wavelet_1d_rejects_sharded_input(device: ttnn.MeshDevice, expect_error) -> None:
    sharded_memory_config = ttnn.create_sharded_memory_config(
        shape=(2, 32),
        core_grid=ttnn.CoreGrid(x=1, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )
    sharded_input = to_device_1d(
        device,
        torch.arange(64, dtype=torch.float32).reshape(2, 1, 1, 32),
        sharded_memory_config,
    )

    with expect_error(RuntimeError, "sharded inputs are unsupported"):
        ttnn.dwt(sharded_input, "bior1.3")


def test_wavelet_2d_validation_errors(device: ttnn.MeshDevice, expect_error) -> None:
    signal = torch.arange(35 * 37, dtype=torch.float32).reshape(35, 37)
    input_tensor = to_device_2d(device, signal)

    with expect_error(RuntimeError, "TILE layout"):
        ttnn.dwt_2d(to_device_1d(device, signal), "bior1.3")
    with expect_error(RuntimeError, "both dimensions greater than one"):
        ttnn.dwt_2d(
            to_device_2d(device, torch.ones(1, 8)),
            "bior1.3",
            boundary_mode="antireflect",
        )

    sharded_memory_config = ttnn.create_sharded_memory_config(
        shape=(64, 64),
        core_grid=ttnn.CoreGrid(x=1, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )
    sharded_input = to_device_2d(
        device,
        torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64),
        sharded_memory_config,
    )
    with expect_error(RuntimeError, "sharded inputs are unsupported"):
        ttnn.dwt_2d(sharded_input, "bior1.3")

    bands = ttnn.dwt_2d(input_tensor, "bior1.3")
    wrong_band = to_device_2d(device, torch.zeros(bands[0].shape[0] + 1, bands[0].shape[1]))
    with expect_error(RuntimeError, "identical shapes"):
        ttnn.idwt_2d(bands[0], wrong_band, bands[2], bands[3], "bior1.3", signal.shape)
    with expect_error(RuntimeError, "must be positive"):
        ttnn.idwt_2d(*bands, "bior1.3", (0, signal.shape[1]))
    with expect_error(RuntimeError, "does not match expected shape"):
        ttnn.idwt_2d(*bands, "bior1.3", (signal.shape[0] + 2, signal.shape[1]))
    with expect_error(RuntimeError, "must not alias"):
        ttnn.dwt_2d(input_tensor, "bior1.3", output_tensors=(bands[0],) * 4)
