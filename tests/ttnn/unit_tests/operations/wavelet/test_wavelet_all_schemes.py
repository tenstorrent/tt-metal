# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
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
SCHEMES: tuple[str, ...] = tuple(pywt.wavelist(kind="discrete"))
YELLOW_PRECISION_WAVELETS: set[str] = {
    "bior4.4",
    "bior5.5",
    "bior6.8",
    "rbio4.4",
    "rbio5.5",
    "rbio6.8",
    *(f"sym{order}" for order in range(2, 21)),
}
FP32_EPS = float(np.finfo(np.float32).eps)
DEVICE_FP32_TOLERANCE_1D = {
    "zero": 1e-5,
    "constant": 1e-5,
    "symmetric": 1e-5,
    "reflect": 1e-5,
    "periodic": 1e-5,
    "smooth": 2e-5,
    "antisymmetric": 1e-5,
    "antireflect": 1e-5,
}
DEVICE_FP32_TOLERANCE_2D = {
    "zero": 2e-5,
    "constant": 2e-5,
    "symmetric": 2e-5,
    "reflect": 2e-5,
    "periodic": 2e-5,
    "smooth": 2e-4,
    "antisymmetric": 1e-5,
    "antireflect": 5e-5,
}
FACTORIZATION_ALLOWANCE = {
    "GREEN": 8 * FP32_EPS,
    "YELLOW": 32 * FP32_EPS,
}


@dataclass(frozen=True)
class PrecisionResult:
    wavelet: str
    boundary_mode: str
    operation: str
    output_name: str
    input_shape: tuple[int, ...]
    precision_class: str
    cpu_fp32_error: float
    tt_fp32_error: float
    normalized_excess: float
    direct_fp32_difference: float
    allowed_tolerance: float

    @property
    def score(self) -> float:
        return max(self.normalized_excess, self.direct_fp32_difference)


def measure_fp32_precision(
    reference_fp64: np.ndarray,
    pywt_fp32: np.ndarray,
    tt_fp32: np.ndarray,
    *,
    wavelet: str,
    boundary_mode: str,
    operation: str,
    output_name: str,
    input_shape: tuple[int, ...],
    device_tolerance: float,
) -> PrecisionResult | None:
    if wavelet == "dmey":
        return None

    assert reference_fp64.dtype == np.float64
    assert pywt_fp32.dtype == np.float32
    assert tt_fp32.dtype == np.float32
    assert reference_fp64.shape == pywt_fp32.shape == tt_fp32.shape

    precision_class = "YELLOW" if wavelet in YELLOW_PRECISION_WAVELETS else "GREEN"
    allowed_tolerance = device_tolerance + FACTORIZATION_ALLOWANCE[precision_class]
    pywt_result = pywt_fp32.astype(np.float64)
    tt_result = tt_fp32.astype(np.float64)
    scale = max(1.0, float(np.max(np.abs(reference_fp64))))
    cpu_fp32_error = float(np.max(np.abs(reference_fp64 - pywt_result)))
    tt_fp32_error = float(np.max(np.abs(reference_fp64 - tt_result)))
    normalized_excess = abs(tt_fp32_error - cpu_fp32_error) / scale
    direct_fp32_difference = float(np.max(np.abs(tt_result - pywt_result))) / scale

    return PrecisionResult(
        wavelet=wavelet,
        boundary_mode=boundary_mode,
        operation=operation,
        output_name=output_name,
        input_shape=input_shape,
        precision_class=precision_class,
        cpu_fp32_error=cpu_fp32_error,
        tt_fp32_error=tt_fp32_error,
        normalized_excess=normalized_excess,
        direct_fp32_difference=direct_fp32_difference,
        allowed_tolerance=allowed_tolerance,
    )


def format_precision_result(result: PrecisionResult) -> str:
    return (
        f"wavelet={result.wavelet}\n"
        f"mode={result.boundary_mode}\n"
        f"operation={result.operation}\n"
        f"output={result.output_name}\n"
        f"input_shape={result.input_shape}\n"
        f"class={result.precision_class}\n"
        f"PyWT FP64 -> PyWT FP32: {result.cpu_fp32_error:.8e}\n"
        f"PyWT FP64 -> TT FP32: {result.tt_fp32_error:.8e}\n"
        f"normalized excess: {result.normalized_excess:.8e}\n"
        f"TT FP32 -> PyWT FP32: {result.direct_fp32_difference:.8e}\n"
        f"allowed: {result.allowed_tolerance:.8e}"
    )


def assert_precision_results(results: list[PrecisionResult]) -> None:
    failures = [result for result in results if result.score > result.allowed_tolerance]
    if failures:
        diagnostics = [f"{len(failures)} precision regressions"]
        for precision_class in FACTORIZATION_ALLOWANCE:
            class_failures = [result for result in failures if result.precision_class == precision_class]
            if class_failures:
                worst = max(class_failures, key=lambda result: result.score / result.allowed_tolerance)
                diagnostics.append(
                    f"{precision_class}: {len(class_failures)} failures; worst case:\n{format_precision_result(worst)}"
                )
        pytest.fail("\n\n".join(diagnostics))


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_all_discrete_schemes_forward_inverse_precision_1d(device: ttnn.MeshDevice, boundary_mode: str) -> None:
    assert len(SCHEMES) == 106

    signal = torch.sin(torch.arange(257, dtype=torch.float32) * 0.113)
    signal_fp32 = signal.numpy()
    signal_fp64 = signal_fp32.astype(np.float64)
    input_tensor = ttnn.from_torch(
        signal,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    precision_results: list[PrecisionResult] = []
    for scheme in SCHEMES:
        approximation, detail = ttnn.dwt(input_tensor, scheme, boundary_mode=boundary_mode)
        reconstructed = ttnn.idwt(
            approximation,
            detail,
            scheme,
            signal.numel(),
            boundary_mode=boundary_mode,
        )

        coefficient_length = pywt.dwt_coeff_len(signal.numel(), pywt.Wavelet(scheme).dec_len, mode=boundary_mode)
        approximation_ref64, detail_ref64 = pywt.dwt(signal_fp64, scheme, mode=boundary_mode)
        approximation_pywt32, detail_pywt32 = pywt.dwt(signal_fp32, scheme, mode=boundary_mode)
        reconstructed_ref64 = pywt.idwt(approximation_ref64, detail_ref64, scheme, mode=boundary_mode)[: signal.numel()]
        reconstructed_pywt32 = pywt.idwt(approximation_pywt32, detail_pywt32, scheme, mode=boundary_mode)[
            : signal.numel()
        ]
        approximation_host = ttnn.to_torch(approximation)
        detail_host = ttnn.to_torch(detail)
        reconstructed_host = ttnn.to_torch(reconstructed)
        approximation_tt32 = approximation_host.flatten()[:coefficient_length].numpy()
        detail_tt32 = detail_host.flatten()[:coefficient_length].numpy()
        reconstructed_tt32 = reconstructed_host.flatten()[: signal.numel()].numpy()

        for output_name, reference_fp64, pywt_fp32, tt_fp32 in (
            ("cA", approximation_ref64, approximation_pywt32, approximation_tt32),
            ("cD", detail_ref64, detail_pywt32, detail_tt32),
            ("reconstructed", reconstructed_ref64, reconstructed_pywt32, reconstructed_tt32),
        ):
            result = measure_fp32_precision(
                reference_fp64,
                pywt_fp32,
                tt_fp32,
                wavelet=scheme,
                boundary_mode=boundary_mode,
                operation="DWT" if output_name != "reconstructed" else "DWT+IDWT",
                output_name=output_name,
                input_shape=tuple(signal.shape),
                device_tolerance=DEVICE_FP32_TOLERANCE_1D[boundary_mode],
            )
            if result is not None:
                precision_results.append(result)

        assert ttnn.dwt_coeff_len(signal.numel(), scheme) == coefficient_length
        coefficient_sticks = (coefficient_length + 31) // 32
        signal_sticks = (signal.numel() + 31) // 32
        assert tuple(approximation.shape) == (coefficient_sticks, 32)
        assert tuple(detail.shape) == (coefficient_sticks, 32)
        assert tuple(reconstructed.shape) == (signal_sticks, 32)
        assert torch.isfinite(approximation_host).all(), scheme
        assert torch.isfinite(detail_host).all(), scheme
        assert torch.isfinite(reconstructed_host).all(), scheme

    assert_precision_results(precision_results)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
def test_all_discrete_schemes_forward_inverse_precision_2d(
    device: ttnn.MeshDevice,
    boundary_mode: str,
) -> None:
    shape = (33, 35)
    y = torch.arange(shape[0], dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(shape[1], dtype=torch.float32).reshape(1, -1)
    signal = torch.sin(0.17 * x) + torch.cos(0.11 * y)
    signal_fp32 = signal.numpy()
    signal_fp64 = signal_fp32.astype(np.float64)
    input_tensor = ttnn.from_torch(
        signal,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    precision_results: list[PrecisionResult] = []
    for scheme in SCHEMES:
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
        ll_ref64, (hl_ref64, lh_ref64, hh_ref64) = pywt.dwt2(signal_fp64, scheme, mode=boundary_mode)
        ll_pywt32, (hl_pywt32, lh_pywt32, hh_pywt32) = pywt.dwt2(signal_fp32, scheme, mode=boundary_mode)
        reconstructed_ref64 = pywt.idwt2((ll_ref64, (hl_ref64, lh_ref64, hh_ref64)), scheme, mode=boundary_mode)[
            : shape[0], : shape[1]
        ]
        reconstructed_pywt32 = pywt.idwt2((ll_pywt32, (hl_pywt32, lh_pywt32, hh_pywt32)), scheme, mode=boundary_mode)[
            : shape[0], : shape[1]
        ]
        output_hosts = tuple(ttnn.to_torch(output) for output in outputs)
        tt_outputs = tuple(output.numpy() for output in output_hosts)
        reconstructed_host = ttnn.to_torch(reconstructed)
        reconstructed_tt32 = reconstructed_host.numpy()
        references = (
            ("LL", ll_ref64, ll_pywt32),
            ("LH", lh_ref64, lh_pywt32),
            ("HL", hl_ref64, hl_pywt32),
            ("HH", hh_ref64, hh_pywt32),
        )
        for (output_name, reference_fp64, pywt_fp32), tt_fp32 in zip(references, tt_outputs):
            result = measure_fp32_precision(
                reference_fp64,
                pywt_fp32,
                tt_fp32,
                wavelet=scheme,
                boundary_mode=boundary_mode,
                operation="DWT2D",
                output_name=output_name,
                input_shape=shape,
                device_tolerance=DEVICE_FP32_TOLERANCE_2D[boundary_mode],
            )
            if result is not None:
                precision_results.append(result)
        reconstructed_result = measure_fp32_precision(
            reconstructed_ref64,
            reconstructed_pywt32,
            reconstructed_tt32,
            wavelet=scheme,
            boundary_mode=boundary_mode,
            operation="DWT2D+IDWT2D",
            output_name="reconstructed",
            input_shape=shape,
            device_tolerance=DEVICE_FP32_TOLERANCE_2D[boundary_mode],
        )
        if reconstructed_result is not None:
            precision_results.append(reconstructed_result)

        for output, output_host in zip(outputs, output_hosts):
            assert tuple(output.shape) == coefficient_shape
            assert torch.isfinite(output_host).all(), scheme
        assert tuple(reconstructed.shape) == shape
        assert torch.isfinite(reconstructed_host).all(), scheme

    assert_precision_results(precision_results)
