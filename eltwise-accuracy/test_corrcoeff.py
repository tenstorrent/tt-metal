#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


# ANSI color codes
class Colors:
    RED = "\033[91m"
    RESET = "\033[0m"


def _print_corrcoeff_results(test_case_num, test_name, golden, calculated, corr_coeff, notes=None):
    """
    Helper function to print correlation coefficient results in a consistent format

    Args:
        test_case_num (int): Test case number
        test_name (str): Name of the test case
        golden (np.ndarray): Golden reference array
        calculated (np.ndarray): Calculated array
        corr_coeff (np.ndarray): Correlation coefficient matrix from np.corrcoef
        notes (str, optional): Additional notes to display
    """
    print("=" * 60)
    print(f"Test Case {test_case_num}: {test_name}")
    print(f"Golden shape: {golden.shape}")
    print(f"Calculated shape: {calculated.shape}")

    # Print array-specific information
    if np.all(golden == golden[0]):  # Check if constant
        print(f"Golden: constant value = {golden[0]}")
    else:
        print(f"Golden range: [{golden.min():.1f}, {golden.max():.1f}]")
        if hasattr(golden, "mean"):
            print(f"Golden stats: mean={golden.mean():.4f}, std={golden.std():.4f}")

    if np.all(calculated == calculated[0]):  # Check if constant
        print(f"Calculated: constant value = {calculated[0]}")
    else:
        print(f"Calculated range: [{calculated.min():.1f}, {calculated.max():.1f}]")
        if hasattr(calculated, "mean"):
            print(f"Calculated stats: mean={calculated.mean():.4f}, std={calculated.std():.4f}")

    # Check for special relationships
    if np.array_equal(golden, calculated):
        print(f"Arrays are identical: True")
    elif np.allclose(calculated, 2 * golden, rtol=1e-10):
        print(f"Relationship: calculated = 2 * golden")

    # Print correlation matrix with red highlighting
    print(f"Correlation coefficient matrix:")
    print(f"{Colors.RED}{corr_coeff}{Colors.RESET}")

    # Print individual correlation coefficient if not NaN
    if not np.isnan(corr_coeff[0, 1]):
        print(f"Correlation coefficient: {corr_coeff[0, 1]:.6f}")

    # Print additional notes
    if notes:
        print(f"Note: {notes}")

    print()


def test_random_vs_random():
    """
    Test case: golden = random data, calculated = random data (golden != calculated)
    """
    N = 1024
    np.random.seed(42)  # For reproducibility
    golden = np.random.randn(N)
    np.random.seed(123)  # Different seed for different data
    calculated = np.random.randn(N)

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=1,
        test_name="Random vs Random (different)",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
    )


def test_constant_vs_constant():
    """
    Test case: golden = constant, calculated = constant
    """
    N = 1024
    golden = np.full(N, 5.0)  # Constant value 5.0
    calculated = np.full(N, 3.0)  # Constant value 3.0

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=2,
        test_name="Constant vs Constant",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
        notes="Correlation is undefined for constant arrays (NaN expected)",
    )


def test_identical_arange():
    """
    Test case: golden = calculated = arange(0, N)
    """
    N = 1024
    golden = np.arange(N, dtype=np.float32)
    calculated = golden.copy()  # Identical arrays

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=3,
        test_name="Identical arrays (arange)",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
    )


def test_arange_vs_scaled():
    """
    Test case: golden = arange(...), calculated = 2 * golden
    """
    N = 1024
    golden = np.arange(N, dtype=np.float32)
    calculated = 2 * golden  # Perfect linear relationship

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=4, test_name="Arange vs Scaled (2x)", golden=golden, calculated=calculated, corr_coeff=corr_coeff
    )


def test_arange_vs_shifted():
    """
    Test case: golden = arange(...), calculated = golden + 100
    """
    N = 1024
    golden = np.arange(N, dtype=np.float32)
    calculated = golden + 100  # Perfect linear relationship

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=4,
        test_name="Arange vs Shifted (100)",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
    )


def test_constant_vs_arange():
    """
    Test case: golden = constant, calculated = arange(0, N)
    """
    N = 1024
    golden = np.full(N, 7.5)  # Constant value
    calculated = np.arange(N, dtype=np.float32)

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=5,
        test_name="Constant vs Arange",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
        notes="Correlation is undefined when one array is constant (NaN expected)",
    )


def test_arange_vs_constant():
    """
    Test case: golden = arange(0, N), calculated = constant
    """
    N = 1024
    golden = np.arange(N, dtype=np.float32)
    calculated = np.full(N, 10.0)  # Constant value

    corr_coeff = np.corrcoef(golden, calculated)

    _print_corrcoeff_results(
        test_case_num=6,
        test_name="Arange vs Constant",
        golden=golden,
        calculated=calculated,
        corr_coeff=corr_coeff,
        notes="Correlation is undefined when one array is constant (NaN expected)",
    )


def main():
    """
    Main function to run all test cases and display results
    """
    print("Testing numpy.corrcoef behavior between golden and calculated tensors")
    print("N = 1024 for all test cases")
    print()

    # Run all test cases
    test_random_vs_random()
    test_constant_vs_constant()
    test_identical_arange()
    test_arange_vs_scaled()
    test_arange_vs_shifted()
    test_constant_vs_arange()
    test_arange_vs_constant()

    print("=" * 60)
    print("Summary:")
    print("- Perfect correlation (r=1.0): Identical arrays or perfect linear relationship")
    print("- Undefined correlation (NaN): When one or both arrays have zero variance (constant)")
    print("- Weak correlation (r≈0): Random, uncorrelated data")
    print("- The correlation coefficient matrix is symmetric with 1.0 on the diagonal")


if __name__ == "__main__":
    main()
