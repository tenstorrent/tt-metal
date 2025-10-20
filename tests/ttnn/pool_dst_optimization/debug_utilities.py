#!/usr/bin/env python3
"""
Debug utilities for Pool DST Optimization testing
"""

import ttnn
import torch
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional


class PoolTestConfig:
    """Configuration for pool testing"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def __str__(self):
        return f"Pool[{self.input_shape}]_K{self.kernel_size}_S{self.stride}_P{self.padding}"


class PoolValidator:
    """Validator for pool operations"""

    @staticmethod
    def create_test_tensor(config: PoolTestConfig, device) -> Tuple[torch.Tensor, ttnn.Tensor]:
        """Create test tensors for validation"""
        # Create deterministic test data for reproducible results
        torch.manual_seed(42)
        torch_input = torch.randn(config.input_shape, dtype=torch.bfloat16)

        # Convert to NHWC format for ttnn (following existing test patterns)
        torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C

        # Convert to ttnn tensor with proper layout
        ttnn_input = ttnn.from_torch(
            torch_input_permuted,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        return torch_input, ttnn_input

    @staticmethod
    def run_reference_implementation(torch_input: torch.Tensor, config: PoolTestConfig) -> torch.Tensor:
        """Run PyTorch reference implementation"""
        return torch.nn.functional.avg_pool2d(
            torch_input,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            ceil_mode=config.ceil_mode,
            count_include_pad=config.count_include_pad,
        )

    @staticmethod
    def run_ttnn_implementation(ttnn_input: ttnn.Tensor, config: PoolTestConfig) -> ttnn.Tensor:
        """Run TTNN implementation"""
        return ttnn.avg_pool2d(
            ttnn_input,
            batch_size=config.input_shape[0],
            input_h=config.input_shape[2],
            input_w=config.input_shape[3],
            channels=config.input_shape[1],
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            ceil_mode=config.ceil_mode,
            count_include_pad=config.count_include_pad,
        )

    @staticmethod
    def validate_results(
        ttnn_output: ttnn.Tensor, torch_reference: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2
    ) -> Dict[str, Any]:
        """Validate TTNN output against PyTorch reference"""
        ttnn_torch = ttnn.to_torch(ttnn_output)

        # Shape validation
        shape_match = ttnn_torch.shape == torch_reference.shape

        # Value validation
        values_close = torch.allclose(ttnn_torch, torch_reference, rtol=rtol, atol=atol)

        # Calculate PCC (Pearson Correlation Coefficient)
        def calculate_pcc(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            if len(a_flat) < 2:
                return 1.0
            return np.corrcoef(a_flat.numpy(), b_flat.numpy())[0, 1]

        pcc = calculate_pcc(ttnn_torch, torch_reference)

        # Calculate max absolute difference
        max_diff = torch.max(torch.abs(ttnn_torch - torch_reference)).item()

        return {
            "shape_match": shape_match,
            "values_close": values_close,
            "pcc": pcc,
            "max_diff": max_diff,
            "ttnn_shape": ttnn_torch.shape,
            "reference_shape": torch_reference.shape,
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities"""

    @staticmethod
    def benchmark_operation(operation_func, *args, warmup_runs: int = 3, timing_runs: int = 10) -> Dict[str, float]:
        """Benchmark an operation with warmup"""

        # Warmup runs
        for _ in range(warmup_runs):
            _ = operation_func(*args)

        # Timing runs
        times = []
        for _ in range(timing_runs):
            start_time = time.perf_counter()
            _ = operation_func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "times": times,
        }


def print_test_header(test_name: str, config: PoolTestConfig):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"CONFIG: {config}")
    print(f"{'='*60}")


def print_validation_results(results: Dict[str, Any]):
    """Print formatted validation results"""
    print(f"\nVALIDATION RESULTS:")
    print(f"  Shape Match: {results['shape_match']}")
    print(f"  Values Close: {results['values_close']}")
    print(f"  PCC: {results['pcc']:.6f}")
    print(f"  Max Diff: {results['max_diff']:.6f}")
    print(f"  TTNN Shape: {results['ttnn_shape']}")
    print(f"  Reference Shape: {results['reference_shape']}")

    # Status summary
    if results["shape_match"] and results["values_close"] and results["pcc"] > 0.99:
        print(f"  STATUS: ✅ PASS")
    else:
        print(f"  STATUS: ❌ FAIL")


def print_benchmark_results(results: Dict[str, float], operation_name: str):
    """Print formatted benchmark results"""
    print(f"\nBENCHMARK RESULTS ({operation_name}):")
    print(f"  Mean Time: {results['mean_time']*1000:.3f} ms")
    print(f"  Std Time: {results['std_time']*1000:.3f} ms")
    print(f"  Min Time: {results['min_time']*1000:.3f} ms")
    print(f"  Max Time: {results['max_time']*1000:.3f} ms")
