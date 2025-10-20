#!/usr/bin/env python3
"""
Level 2 Tests: Target Scenario Validation
Test the specific scenarios mentioned in GitHub issue #26407
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
class TestTargetScenarios:
    """Tests for specific scenarios mentioned in the GitHub issue"""

    def test_c64_scenario(self, device):
        """Test the C=64 scenario specifically mentioned in the issue"""
        config = PoolTestConfig(input_shape=(1, 64, 16, 16), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        print_test_header("C=64 Target Scenario", config)

        torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)
        torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
        ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

        results = PoolValidator.validate_results(ttnn_output, torch_reference)
        print_validation_results(results)

        assert results["shape_match"]
        assert results["values_close"]
        assert results["pcc"] > 0.99

        print(f"\n🎯 C=64 scenario validation: PASSED")

    def test_channel_tile_scenarios(self, device):
        """Test scenarios with different numbers of channel tiles per core"""
        # Test cases designed to trigger different channel tile counts
        configs = [
            # 32 channels - 1 tile per channel group
            PoolTestConfig((1, 32, 16, 16), (3, 3), (1, 1), (1, 1)),
            # 64 channels - 2 tiles per channel group (issue scenario)
            PoolTestConfig((1, 64, 16, 16), (3, 3), (1, 1), (1, 1)),
            # 128 channels - 4 tiles per channel group
            PoolTestConfig((1, 128, 16, 16), (3, 3), (1, 1), (1, 1)),
            # 256 channels - 8 tiles per channel group
            PoolTestConfig((1, 256, 16, 16), (3, 3), (1, 1), (1, 1)),
        ]

        for i, config in enumerate(configs):
            print_test_header(f"Channel Tiles Test {i+1} (C={config.input_shape[1]})", config)

            torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)
            torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
            ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

            results = PoolValidator.validate_results(ttnn_output, torch_reference)
            print_validation_results(results)

            assert results["shape_match"]
            assert results["values_close"]
            assert results["pcc"] > 0.99

    def test_performance_critical_sizes(self, device):
        """Test performance on sizes that should benefit from DST optimization"""
        configs = [
            # Small but typical CNN sizes
            PoolTestConfig((1, 64, 32, 32), (3, 3), (1, 1), (1, 1)),
            PoolTestConfig((1, 128, 16, 16), (3, 3), (2, 2), (1, 1)),
            # Larger sizes that should show clear performance differences
            PoolTestConfig((1, 256, 14, 14), (3, 3), (1, 1), (1, 1)),
        ]

        baseline_times = {}

        for i, config in enumerate(configs):
            print_test_header(f"Performance Critical Size {i+1}", config)

            torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)

            # Benchmark current implementation
            def run_ttnn():
                return PoolValidator.run_ttnn_implementation(ttnn_input, config)

            benchmark_results = PerformanceBenchmark.benchmark_operation(run_ttnn, warmup_runs=3, timing_runs=10)

            print_benchmark_results(benchmark_results, f"Config {i+1}")

            # Store baseline for later optimization comparison
            baseline_times[str(config)] = benchmark_results["mean_time"]

            # Verify correctness
            ttnn_output = run_ttnn()
            torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
            results = PoolValidator.validate_results(ttnn_output, torch_reference)

            assert results["pcc"] > 0.99

        print(f"\n📊 PERFORMANCE BASELINES RECORDED:")
        for config_str, time in baseline_times.items():
            print(f"  {config_str}: {time*1000:.3f} ms")

    def test_different_kernel_sizes(self, device):
        """Test different kernel sizes that affect reduction complexity"""
        configs = [
            # Small kernels
            PoolTestConfig((1, 64, 16, 16), (2, 2), (1, 1), (0, 0)),
            # Medium kernels (target case)
            PoolTestConfig((1, 64, 16, 16), (3, 3), (1, 1), (1, 1)),
            # Larger kernels
            PoolTestConfig((1, 64, 16, 16), (5, 5), (1, 1), (2, 2)),
        ]

        for i, config in enumerate(configs):
            print_test_header(f"Kernel Size Test {i+1} (K={config.kernel_size})", config)

            torch_input, ttnn_input = PoolValidator.create_test_tensor(config, device)
            torch_reference = PoolValidator.run_reference_implementation(torch_input, config)
            ttnn_output = PoolValidator.run_ttnn_implementation(ttnn_input, config)

            results = PoolValidator.validate_results(ttnn_output, torch_reference)
            print_validation_results(results)

            assert results["shape_match"]
            assert results["values_close"]
            assert results["pcc"] > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
