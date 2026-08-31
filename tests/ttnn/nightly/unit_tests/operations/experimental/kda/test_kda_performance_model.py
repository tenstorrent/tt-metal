# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace

import pytest

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.kda_performance_model_test_utils import (
    KdaTensorTraffic,
    KdaWork,
    affine_exclusive_scan_work,
    clock_mhz_from_frequency_ghz,
    estimate,
    estimate_for_tensors,
    profile_realtime_program,
    math_fidelity_factor,
    prepare_chunk_recurrence_work,
    qkv_causal_conv1d_silu_work,
    recurrent_chunk_scan_work,
    reduce_affine_transforms_work,
    sigmoid_gated_rms_norm_work,
    summarize_chunk_recurrence_work,
    tensor_traffic,
    utilization,
)


def test_all_mathematical_work_closed_forms() -> None:
    rms = sigmoid_gated_rms_norm_work(2, 3, 5, 7)
    assert rms == KdaWork(
        multiply_results=4 * 2 * 3 * 5 * 7,
        add_results=2 * 3 * 5,
        reduction_input_elements=2 * 3 * 5 * 7,
        omitted_sfpu_results=2 * 3 * 5 + 2 * 3 * 5 * 7,
    )

    elements = 2 * 5 * (7 + 11 + 13)
    assert qkv_causal_conv1d_silu_work(2, 5, 7, 11, 13) == KdaWork(
        multiply_results=4 * elements,
        add_results=3 * elements,
        omitted_sfpu_results=elements,
    )

    compositions = 3 * (4 - 1)
    assert reduce_affine_transforms_work(3, 4, 5, 7) == KdaWork(
        dense_flops=compositions * (2 * 5**3 + 2 * 5**2 * 7),
        add_results=compositions * 5 * 7,
    )
    assert affine_exclusive_scan_work(3, 4, 5, 7) == KdaWork(
        dense_flops=compositions * 2 * 5**2 * 7,
        add_results=compositions * 5 * 7,
    )

    heads, chunks, key, value, chunk = 2, 3, 5, 7, 32
    instances = heads * chunks
    assert prepare_chunk_recurrence_work(heads, chunks, key, value) == KdaWork(
        dense_flops=instances * (4 * chunk**2 * key + chunk * (chunk - 1) * (chunk + 1) // 3),
        multiply_results=instances * (10 * chunk * key + chunk * value),
        add_results=instances * (2 * chunk + (chunk - 1) * key + chunk * key + chunk**2),
        reduction_input_elements=instances * 2 * chunk * key,
        omitted_sfpu_results=instances * (2 * chunk + 3 * chunk * key + key),
    )
    assert recurrent_chunk_scan_work(heads, chunks, key, value) == KdaWork(
        dense_flops=instances * (6 * chunk * key * value + 4 * chunk**2 * value),
        multiply_results=instances * key * value,
        add_results=instances * (2 * chunk * value + key * value),
    )
    assert summarize_chunk_recurrence_work(heads, chunks, key, value) == KdaWork(
        dense_flops=instances * (8 * chunk * key * value + 4 * chunk**2 * value),
        multiply_results=instances * 2 * key * value,
        add_results=instances * (2 * chunk * value + 2 * key * value) + heads * key * value,
    )


def test_production_large_exact_and_unbounded_work() -> None:
    rows = 1 * 48 * 1280
    elements = rows * 128
    assert sigmoid_gated_rms_norm_work(1, 48, 1280, 128) == KdaWork(
        multiply_results=4 * elements,
        add_results=rows,
        reduction_input_elements=elements,
        omitted_sfpu_results=rows + elements,
    )

    heads, chunks, key, value, chunk = 12, 160, 128, 128, 32
    instances = heads * chunks
    production = prepare_chunk_recurrence_work(heads, chunks, key, value)
    assert production.dense_flops == instances * (4 * chunk**2 * key + chunk * (chunk - 1) * (chunk + 1) // 3)
    assert production.omitted_sfpu_results == instances * (2 * chunk + 3 * chunk * key + key)

    above_float_exactness = sigmoid_gated_rms_norm_work(1 << 53, 1, 1, 1)
    assert above_float_exactness == KdaWork(
        multiply_results=1 << 55,
        add_results=1 << 53,
        reduction_input_elements=1 << 53,
        omitted_sfpu_results=1 << 54,
    )
    huge_instances = 1 << 124
    huge = prepare_chunk_recurrence_work(1 << 62, 1 << 62, 0, 1)
    assert huge == KdaWork(
        dense_flops=huge_instances * (32 * 31 * 33 // 3),
        multiply_results=huge_instances * 32,
        add_results=huge_instances * (2 * 32 + 32**2),
        omitted_sfpu_results=huge_instances * (2 * 32),
    )


@pytest.mark.parametrize(
    ("work_fn", "args"),
    [
        (sigmoid_gated_rms_norm_work, (-1, 1, 1, 1)),
        (qkv_causal_conv1d_silu_work, (1, 1, -1, 1, 1)),
        (reduce_affine_transforms_work, (1, -1, 1, 1)),
        (affine_exclusive_scan_work, (1, -1, 1, 1)),
        (prepare_chunk_recurrence_work, (1, 1, -1, 1)),
        (recurrent_chunk_scan_work, (1, 1, -1, 1)),
        (summarize_chunk_recurrence_work, (1, 1, -1, 1)),
    ],
)
def test_mathematical_work_requires_nonnegative_dimensions(
    work_fn, args: tuple[int, ...], expect_error: Callable
) -> None:
    with expect_error(AssertionError, "must be non-negative"):
        work_fn(*args)


def test_single_group_has_no_composition_or_transition_work() -> None:
    assert reduce_affine_transforms_work(8, 1, 128, 128) == KdaWork()
    assert affine_exclusive_scan_work(8, 1, 128, 128) == KdaWork()


@pytest.mark.parametrize("fidelity_factor", [1, 2, 3, 4])
def test_exact_cycles_use_all_harvested_non_square_cores(fidelity_factor: int) -> None:
    work = KdaWork(
        dense_flops=4096 * 10,
        multiply_results=128 * 10,
        add_results=128 * 10,
        reduction_input_elements=256 * 10,
    )
    numerator = (
        work.dense_flops * fidelity_factor
        + 32 * work.multiply_results * fidelity_factor
        + 32 * work.add_results
        + 16 * work.reduction_input_elements * fidelity_factor
    )
    denominator = 4096 * (13 * 9)
    expected_cycles = (numerator + denominator - 1) // denominator
    result = estimate(work, (), (), core_count=13 * 9, clock_mhz=1350, fidelity_factor=fidelity_factor)
    assert result.valid
    assert result.ideal_fpu_cycles == expected_cycles
    assert result.ideal_fpu_ns == (expected_cycles * 1000 + 1349) // 1350


def test_dram_traffic_sums_deduplicates_aliases_and_rounds_decimal_bandwidth() -> None:
    inputs = (
        KdaTensorTraffic(0x1000, 513, True),
        KdaTensorTraffic(0x1000, 513, True),
        KdaTensorTraffic(0x2000, 4096, False),
        KdaTensorTraffic(0x3000, 511, True),
    )
    outputs = (
        KdaTensorTraffic(0x1000, 513, True),
        KdaTensorTraffic(0x4000, 8192, False),
    )
    result = estimate(KdaWork(), inputs, outputs, core_count=130, clock_mhz=1350, fidelity_factor=4)
    assert result.valid
    assert result.mandatory_dram_bytes == 513 + 511 + 513
    assert result.ideal_dram_ns == 4
    assert result.ideal_ns == 4
    assert result.input_bytes == (513, 513, 0, 511)
    assert result.output_bytes == (513, 0)


def test_profiler_field_overflow_raises(expect_error: Callable) -> None:
    oversized = KdaTensorTraffic(0x1000, 1 << 31, True)
    with expect_error(OverflowError, "input bytes"):
        estimate(KdaWork(), (oversized,), (), core_count=117, clock_mhz=1350, fidelity_factor=2)


def test_l1_only_and_invalid_fallbacks_preserve_slots() -> None:
    l1 = (KdaTensorTraffic(0x1000, 4096, False),)
    valid = estimate(KdaWork(omitted_sfpu_results=17), l1, l1, core_count=117, clock_mhz=1350, fidelity_factor=2)
    assert valid.valid
    assert valid.ideal_ns == 0
    assert valid.input_bytes == (0,)
    assert valid.output_bytes == (0,)
    assert valid.omitted_sfpu_results == 17

    invalid = estimate(KdaWork(dense_flops=4096), l1, l1, core_count=117, clock_mhz=1350, fidelity_factor=0)
    assert not invalid.valid
    assert invalid.input_bytes == (0,)
    assert invalid.output_bytes == (0,)


class _FakeTensor:
    def __init__(self, *, storage_type, volume: int, element_size: int, buffer_type, allocated: bool = True) -> None:
        self._storage_type = storage_type
        self._volume = volume
        self._element_size = element_size
        self._buffer_type = buffer_type
        self._allocated = allocated

    def storage_type(self):
        return self._storage_type

    def is_allocated(self) -> bool:
        return self._allocated

    def volume(self) -> int:
        return self._volume

    def element_size(self) -> int:
        return self._element_size

    def buffer_address(self) -> int:
        return 0x1234

    def memory_config(self):
        return SimpleNamespace(buffer_type=self._buffer_type)


def test_tensor_traffic_uses_physical_padded_volume_and_rejects_host_tensor() -> None:
    import torch

    logical_shape = (3, 17, 17)
    padded_host_tensor = ttnn.Tensor(torch.zeros(logical_shape), ttnn.bfloat16).pad_to_tile(0.0)
    assert padded_host_tensor.volume() == 3 * 32 * 32
    assert padded_host_tensor.volume() != 3 * 17 * 17

    device_tensor = _FakeTensor(
        storage_type=ttnn.StorageType.DEVICE,
        volume=padded_host_tensor.volume(),
        element_size=2,
        buffer_type=ttnn.BufferType.DRAM,
    )
    assert tensor_traffic(device_tensor) == KdaTensorTraffic(0x1234, 3 * 32 * 32 * 2, True)

    unallocated_device_tensor = _FakeTensor(
        storage_type=ttnn.StorageType.DEVICE,
        volume=32 * 32,
        element_size=2,
        buffer_type=ttnn.BufferType.DRAM,
        allocated=False,
    )
    assert tensor_traffic(padded_host_tensor) is None
    assert tensor_traffic(unallocated_device_tensor) is None


@pytest.mark.parametrize(
    "math_fidelity, expected_factor",
    [
        (ttnn.MathFidelity.LoFi, 1),
        (ttnn.MathFidelity.HiFi2, 2),
        (ttnn.MathFidelity.HiFi3, 3),
        (ttnn.MathFidelity.HiFi4, 4),
        (ttnn.MathFidelity.Invalid, 0),
    ],
)
def test_math_fidelity_factor(math_fidelity, expected_factor: int) -> None:
    assert math_fidelity_factor(math_fidelity) == expected_factor


def test_estimate_for_tensors_derives_grid_clock_fidelity_and_traffic() -> None:
    grid = SimpleNamespace(x=13, y=9)
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: grid)
    input_tensor = _FakeTensor(
        storage_type=ttnn.StorageType.DEVICE,
        volume=3 * 32 * 32,
        element_size=2,
        buffer_type=ttnn.BufferType.DRAM,
    )
    output_tensor = _FakeTensor(
        storage_type=ttnn.StorageType.DEVICE,
        volume=32 * 32,
        element_size=2,
        buffer_type=ttnn.BufferType.L1,
    )
    result = estimate_for_tensors(
        KdaWork(dense_flops=4096 * 13 * 9 * 1350),
        (input_tensor,),
        (output_tensor,),
        device=device,
        frequency_ghz=1.3495,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )

    assert result.valid
    assert result.ideal_fpu_cycles == 2700
    assert result.ideal_fpu_ns == 2000
    assert result.mandatory_dram_bytes == 3 * 32 * 32 * 2
    assert result.ideal_dram_ns == 12
    assert result.ideal_ns == 2000
    assert result.input_bytes == (3 * 32 * 32 * 2,)
    assert result.output_bytes == (0,)


@pytest.mark.parametrize(
    "frequency_ghz, expected_mhz",
    [(1.3494, 1349), (1.3495, 1350), (1.3505, 1351), (0.0, 0), (float("nan"), 0)],
)
def test_realtime_frequency_converts_once_to_nearest_integer_mhz(frequency_ghz: float, expected_mhz: int) -> None:
    assert clock_mhz_from_frequency_ghz(frequency_ghz) == expected_mhz


def test_realtime_profile_record_exposes_frequency_ghz(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = None

    def register(collector):
        nonlocal callback
        callback = collector
        return 7

    monkeypatch.setattr(ttnn.device, "RegisterProgramRealtimeProfilerCallback", register)
    monkeypatch.setattr(ttnn.device, "UnregisterProgramRealtimeProfilerCallback", lambda _handle: None)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _device: None)

    record = SimpleNamespace(
        runtime_id=19,
        chip_id=0,
        start_timestamp=100,
        end_timestamp=1450,
        frequency=1.35,
        kernel_sources=("reader.cpp",),
    )
    batch = SimpleNamespace(dropped=0, records=(record,))

    def run():
        assert callback is not None
        callback(batch)
        return "result"

    result, profile = profile_realtime_program(object(), run)
    assert result == "result"
    assert profile["frequency_ghz"] == pytest.approx(1.35)
    assert profile["duration_ns"] == pytest.approx(1000.0)


def test_utilization_uses_fpu_dram_and_roofline_times() -> None:
    modeled = estimate(
        KdaWork(dense_flops=4096 * 117),
        (KdaTensorTraffic(0x1000, 1024, True),),
        (),
        core_count=117,
        clock_mhz=1000,
        fidelity_factor=1,
    )
    percentages = utilization(modeled, measured_ns=4)
    assert percentages.fpu_utilization_pct == pytest.approx(25.0)
    assert percentages.dram_utilization_pct == pytest.approx(50.0)
    assert percentages.roofline_utilization_pct == pytest.approx(50.0)
