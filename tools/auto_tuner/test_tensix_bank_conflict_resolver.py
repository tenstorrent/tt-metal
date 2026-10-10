"""
Benchmark & Validation Suite for Tenstorrent TT-Metal Tensix Auto-Tuner
Verifies acceptance criteria:
  - Benchmark test showing >= 94% peak TFLOPs without circular buffer deadlock.
  - Zero bank conflict count in planned CB layouts.
  - Verification on Wormhole B0 8x8 core grid.
  - Support for FP16 and BFP8 data formats.
"""

import pytest
from tensix_bank_conflict_resolver import (
    TensixArchitectureSpec,
    TensixMatMulAutoTuner,
    MatMulTuningResult
)


@pytest.fixture
def tuner():
    spec = TensixArchitectureSpec(num_rows=8, num_cols=8)
    return TensixMatMulAutoTuner(spec)


def test_fp16_gemm_achieves_ge_94_percent_peak_efficiency(tuner):
    """
    Core Benchmark Gate:
    Validates that FP16 GEMM kernel reaches >= 94.0% of theoretical peak TFLOPs
    without circular buffer deadlock.
    """
    # Standard 4096 x 4096 x 4096 matrix multiplication
    result = tuner.tune_matmul(m=4096, k=4096, n=4096, data_format="FP16")

    assert result.deadlock_free is True, "Deadlock detected in circular buffer configuration"
    assert result.bank_conflict_count == 0, f"Expected 0 bank conflicts, got {result.bank_conflict_count}"
    assert result.peak_efficiency_percent >= 94.0, (
        f"Efficiency benchmark failed: {result.peak_efficiency_percent:.2f}% < 94.0%"
    )
    assert result.effective_tflops >= result.theoretical_peak_tflops * 0.94


def test_bfp8_gemm_achieves_ge_94_percent_peak_efficiency(tuner):
    """
    Validates BFP8 (Block Float 8) format achieves >= 94.0% efficiency.
    """
    result = tuner.tune_matmul(m=2048, k=2048, n=2048, data_format="BFP8")

    assert result.deadlock_free is True
    assert result.bank_conflict_count == 0
    assert result.peak_efficiency_percent >= 94.0
    assert result.l1_utilization_percent < 80.0  # L1 headroom verified


def test_circular_buffer_allocation_and_alignment(tuner):
    """
    Verifies base addresses of CB 0, CB 1, and CB 16 are aligned to 1KB bank stripes
    and fit comfortably within L1 SRAM capacity.
    """
    result = tuner.tune_matmul(m=1024, k=1024, n=1024, data_format="FP16")
    cbs = result.cb_allocations

    assert 0 in cbs
    assert 1 in cbs
    assert 16 in cbs

    # Check bank stripe alignment (1024 bytes)
    assert cbs[0].base_address % 1024 == 0
    assert cbs[1].base_address % 1024 == 0
    assert cbs[16].base_address % 1024 == 0

    # Total memory must be well within L1 capacity (1.43 MB)
    assert result.total_l1_used_bytes <= tuner.spec.l1_sram_bytes_per_core


def test_grid_shapes_scaling(tuner):
    """Verifies tuner scales across varying grid dimensions (8x8, 8x7)."""
    custom_spec = TensixArchitectureSpec(num_rows=8, num_cols=7)
    custom_tuner = TensixMatMulAutoTuner(custom_spec)
    result = custom_tuner.tune_matmul(m=2048, k=2048, n=1792, data_format="FP16")

    assert result.deadlock_free is True
    assert result.peak_efficiency_percent >= 94.0
    assert result.grid_shape == (8, 7)
