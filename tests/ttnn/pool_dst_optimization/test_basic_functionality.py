#!/usr/bin/env python3
"""
Level 1 Tests: Basic Functionality Validation
Test minimal cases to verify core pool functionality before optimization
"""

import pytest
import ttnn
import torch
from debug_utilities import (
    PoolTestConfig,
    PoolValidator,
    PerformanceBenchmark,
    print_test_header,
    print_validation_results,
    print_benchmark_results,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestBasicPoolFunctionality:
    """Basic functionality tests for pool operations"""

    def test_minimal_case(self, device):
        """Test absolute minimal case - 1x4x8x8 input"""
        config = PoolTestConfig(input_shape=(1, 4, 8, 8), kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))

        print_test_header("Minimal Case", config)

        # Create test data
        torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)

        # Run reference
        torch_reference = PoolValidator.run_reference_implementation(torch_input, config)

        # Run TTNN implementation
        ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

        # Validate results
        results = PoolValidator.validate_results(ttnn_output, torch_reference)
        print_validation_results(results)

        # Assertions
        assert results["shape_match"], f"Shape mismatch: {results['ttnn_shape']} vs {results['reference_shape']}"
        assert results["values_close"], f"Values not close, max diff: {results['max_diff']}"
        assert results["pcc"] > 0.99, f"PCC too low: {results['pcc']}"

    def test_small_channel_count(self, device):
        """Test with small channel count"""
        config = PoolTestConfig(input_shape=(1, 8, 16, 16), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        print_test_header("Small Channel Count", config)

        torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)
        torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
        ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

        results = PoolValidator.validate_results(ttnn_output, torch_reference)
        print_validation_results(results)

        assert results["shape_match"]
        assert results["values_close"]
        assert results["pcc"] > 0.99

    def test_different_strides(self, device):
        """Test with different stride values"""
        configs = [
            PoolTestConfig((1, 16, 16, 16), (2, 2), (1, 1), (0, 0)),
            PoolTestConfig((1, 16, 16, 16), (2, 2), (2, 2), (0, 0)),
            PoolTestConfig((1, 16, 16, 16), (3, 3), (2, 2), (1, 1)),
        ]

        for i, config in enumerate(configs):
            print_test_header(f"Different Strides Test {i+1}", config)

            torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)
            torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
            ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

            results = PoolValidator.validate_results(ttnn_output, torch_reference)
            print_validation_results(results)

            assert results["shape_match"]
            assert results["values_close"]
            assert results["pcc"] > 0.99

    def test_baseline_performance(self, device):
        """Establish baseline performance metrics"""
        config = PoolTestConfig(input_shape=(1, 32, 32, 32), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        print_test_header("Baseline Performance", config)

        torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)

        # Benchmark TTNN implementation
        def run_ttnn():
            return PoolValidator.run_ttnn_implementation(ttnn_input, config)

        benchmark_results = PerformanceBenchmark.benchmark_operation(run_ttnn, warmup_runs=3, timing_runs=10)

        print_benchmark_results(benchmark_results, "TTNN AvgPool2D")

        # Save baseline for later comparison
        baseline_time = benchmark_results["mean_time"]
        print(f"\n🎯 BASELINE ESTABLISHED: {baseline_time*1000:.3f} ms")

        # Verify correctness
        ttnn_output = run_ttnn()
        torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
        results = PoolValidator.validate_results(ttnn_output, torch_reference)

        assert results["pcc"] > 0.99, "Baseline test must be functionally correct"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
