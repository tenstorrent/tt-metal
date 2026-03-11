import os
import csv
import math
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose, assert_relative_frobenius, assert_with_ulp
from tests.ttnn.utils_for_testing import logger, _normalize_tensor, comp_relative_frobenius
from models.common.utility_functions import comp_pcc, comp_allclose, comp_allclose_custom, comp_ulp


def _get_bool_env(env_var, default):
    """Get boolean from environment variable, with default fallback."""
    value = os.environ.get(env_var, "").lower()
    return True if value in ("1", "true", "yes") else False if value in ("0", "false", "no") else default


USE_PCC = _get_bool_env("USE_PCC", True)  # Pearson Correlation Coefficient (existing default)
USE_ALLCLOSE = _get_bool_env("USE_ALLCLOSE", False)  # Per-element closeness check
USE_RELATIVE_FROBENIUS = _get_bool_env("USE_RELATIVE_FROBENIUS", False)  # Global error matrix norm
USE_ULP = _get_bool_env("USE_ULP", False)  # Units in the Last Place (not recommended for matmul)


# Default tolerance values
ALLCLOSE_ATOL = 1e-08  # Absolute tolerance for allclose
ALLCLOSE_RTOL = 1e-05  # Relative tolerance for allclose
FROBENIUS_THRESHOLD = 0.01  # Threshold for relative Frobenius (1%)
ULP_THRESHOLD = 10  # ULP threshold (not recommended for matmul)
NEAR_ZERO_THRESHOLD = 0.0001
# # Padding validation configuration
# USE_FILL_PAD = False  # Enable fill_implicit_tile_padding with TEST_PADDING_VALUE for validation
# TEST_PADDING_VALUE = 0  # Value to fill implicit tile padding for validation


# def assert_matmul_accuracy(
#     expected,
#     actual,
#     test_name="",
#     allclose_atol=ALLCLOSE_ATOL,
#     allclose_rtol=ALLCLOSE_RTOL,
#     frobenius_threshold=FROBENIUS_THRESHOLD,
#     ulp_threshold=ULP_THRESHOLD,
# ):
#     """
#     Apply configured accuracy metrics to matmul test results.
#     Filters out near-zero values in expected for allclose and ULP to avoid division issues.
#     Uses ATOL threshold to determine what counts as "near-zero".

#     Args:
#         expected: Expected tensor (PyTorch or TTNN)
#         actual: Actual tensor (PyTorch or TTNN)
#         test_name: Optional test name for logging
#         allclose_atol: Absolute tolerance for allclose (default: ALLCLOSE_ATOL)
#         allclose_rtol: Relative tolerance for allclose (default: ALLCLOSE_RTOL)
#         frobenius_threshold: Threshold for relative Frobenius (default: FROBENIUS_THRESHOLD)
#         ulp_threshold: ULP threshold (default: ULP_THRESHOLD)
#     """
#     # Normalize tensors
#     expected = _normalize_tensor(expected)
#     actual = _normalize_tensor(actual)

#     # Calculate statistics for near-zero values
#     # Use ATOL as threshold: if |expected| < ATOL, it's effectively zero for RTOL/ULP purposes
#     total_elements = expected.numel()
#     near_zero_threshold = allclose_atol
#     near_zero_mask_expected = torch.abs(expected) < near_zero_threshold
#     near_zero_count_expected = near_zero_mask_expected.sum().item()
#     near_zero_percentage = (near_zero_count_expected / total_elements * 100) if total_elements > 0 else 0

#     if near_zero_percentage > 0:
#         logger.info(f"{test_name}: {near_zero_percentage:.2f}% of values have |expected| < {near_zero_threshold}")

#     # Apply PCC (if enabled) - works with all values including zeros
#     if USE_PCC:
#         assert_with_pcc(expected, actual, 0.999)

#     # Apply allclose (if enabled) - filter out positions where expected is near-zero
#     if USE_ALLCLOSE:
#         # Filter out positions where |expected| < ATOL (to avoid huge RTOL errors from small expected values)
#         non_near_zero_mask = torch.abs(expected) >= near_zero_threshold

#         if non_near_zero_mask.any():
#             expected_non_near_zero = expected[non_near_zero_mask]
#             actual_non_near_zero = actual[non_near_zero_mask]
#             assert_allclose(expected_non_near_zero, actual_non_near_zero, atol=allclose_atol, rtol=allclose_rtol)

#         # For positions where expected is near-zero, check using ATOL only
#         if near_zero_mask_expected.any():
#             expected_near_zero = expected[near_zero_mask_expected]
#             actual_near_zero = actual[near_zero_mask_expected]
#             # Check if actual is close to expected using ATOL
#             abs_diff = torch.abs(actual_near_zero - expected_near_zero)
#             max_atol_for_near_zeros = torch.max(abs_diff).item()
#             if max_atol_for_near_zeros > allclose_atol:
#                 raise AssertionError(
#                     f"{test_name}: For positions where |expected| < {near_zero_threshold}, "
#                     f"actual values exceed ATOL. Max |actual - expected|: {max_atol_for_near_zeros} > {allclose_atol}"
#                 )

#     # Apply relative Frobenius (if enabled) - handles zeros automatically
#     if USE_RELATIVE_FROBENIUS:
#         assert_relative_frobenius(expected, actual, threshold=frobenius_threshold)

#     # Apply ULP (if enabled) - filter out positions where expected is near-zero
#     if USE_ULP:
#         # Filter out positions where |expected| < ATOL (to avoid division by zero in ULP)
#         non_near_zero_mask = torch.abs(expected) >= near_zero_threshold

#         if non_near_zero_mask.any():
#             expected_non_near_zero = expected[non_near_zero_mask]
#             actual_non_near_zero = actual[non_near_zero_mask]
#             assert_with_ulp(expected_non_near_zero, actual_non_near_zero, ulp_threshold=ulp_threshold)

#         # For positions where expected is near-zero, ULP doesn't make sense, skip
#         if near_zero_mask_expected.any():
#             logger.info(
#                 f"{test_name}: Skipping ULP check for {near_zero_count_expected} positions where |expected| < {near_zero_threshold}"
#             )


def collect_and_dump_numeric_metrics(
    expected,
    actual,
    test_name="",
    csv_filename=None,
    csv_dir=None,
    test_params=None,
    allclose_atol=ALLCLOSE_ATOL,
    allclose_rtol=ALLCLOSE_RTOL,
    ulp_threshold=ULP_THRESHOLD,
    pcc_threshold=0.999,
):
    """
    Collect all numeric accuracy metrics (PCC, Allclose, Frobenius, ULP) and dump to CSV.
    Also calls assert_matmul_accuracy for assertions based on env vars.

    This is a reusable function that can be used by any test to collect comprehensive
    numeric metrics and write them to a CSV file for analysis.

    Args:
        expected: Expected tensor (PyTorch or TTNN)
        actual: Actual tensor (PyTorch or TTNN)
        test_name: Test name for logging and CSV (e.g., "test_var[batch=1,h=32,w=32]")
        csv_filename: Name of CSV file (default: "numeric_results.csv")
        csv_dir: Directory to write CSV (default: same directory as calling test file)
        test_params: Dict of test parameters to include in CSV (e.g., {"batch_size": 1, "h": 32})
        allclose_atol: Absolute tolerance for allclose
        allclose_rtol: Relative tolerance for allclose
        ulp_threshold: ULP threshold
        pcc_threshold: PCC threshold

    Returns:
        dict: Dictionary containing all collected metrics
    """
    # Normalize tensors

    # PCC
    pcc_passed, pcc_val = comp_pcc(expected, actual, pcc_threshold)

    ###################
    # Allclose (using existing comp_allclose_custom function)

    expected_allclose = _normalize_tensor(expected)
    actual_allclose = _normalize_tensor(actual)

    # Remove the elements where either expected or actual is 0
    # mask_nonzero = expected_allclose != 0 & actual_allclose != 0
    # expected_allclose = expected_allclose[mask_nonzero]
    # actual_allclose = actual_allclose[mask_nonzero]

    near_zero_threshold = NEAR_ZERO_THRESHOLD
    non_near_zero_mask = torch.abs(expected_allclose) >= near_zero_threshold
    if non_near_zero_mask.any():
        expected_allclose = expected_allclose[non_near_zero_mask]
        actual_allclose = actual_allclose[non_near_zero_mask]

    if expected_allclose.numel() > 0 and actual_allclose.numel() > 0:
        allclose_passed, allclose_message = comp_allclose_custom(
            expected_allclose, actual_allclose, rtol=allclose_rtol, atol=allclose_atol
        )
        # Parse values from message: "Max ATOL Delta: {max_atol}, Mean ATOL Delta: {mean_atol}, Max RTOL Delta: {max_rtol}, Mean RTOL Delta: {mean_rtol}"
        max_atol = float(allclose_message.split("Max ATOL Delta: ")[1].split(",")[0])
        mean_atol = float(allclose_message.split("Mean ATOL Delta: ")[1].split(",")[0])
        max_rtol = float(allclose_message.split("Max RTOL Delta: ")[1].split(",")[0])
        mean_rtol = float(allclose_message.split("Mean RTOL Delta: ")[1])
    else:
        allclose_passed = False
        allclose_message = "Both tensors are empty"
        max_atol = 0
        mean_atol = 0
        max_rtol = 0
        mean_rtol = 0

    # Relative Frobenius
    frob_val, frob_expected_zero = comp_relative_frobenius(expected, actual)

    if 1:
        # Filter out positions where |expected| < ATOL (to avoid division by zero in ULP)
        near_zero_threshold = NEAR_ZERO_THRESHOLD
        non_near_zero_mask = torch.abs(expected) >= near_zero_threshold
        total_elems = expected.numel()
        near_zero_count = (torch.abs(expected) < near_zero_threshold).sum().item()
        near_zero_pct = (near_zero_count / total_elems * 100) if total_elems > 0 else 0
        # near_zero_pct =5

        if non_near_zero_mask.any():
            expected_non_near_zero = expected[non_near_zero_mask]
            actual_non_near_zero = actual[non_near_zero_mask]
            ulp_passed, ulp_message = assert_with_ulp(
                expected_non_near_zero, actual_non_near_zero, ulp_threshold=ulp_threshold
            )
            ulp_passed = bool(ulp_passed)
            # max_ulp=float(ulp_message.split("Max ULP Delta: ")[1].split(" ")[0])
            if "Max ULP Delta:" in ulp_message:
                ulp_str = ulp_message.split("Max ULP Delta: ")[1].split(" ")[0]
                # Handle tensor(...) format
                if ulp_str.startswith("tensor("):
                    ulp_str = ulp_str[len("tensor(") :].rstrip(")")
                try:
                    max_ulp = float(ulp_str)
                except ValueError:
                    max_ulp = float("nan")

            if "Avg ulp Delta:" in ulp_message:
                avg_ulp = float(ulp_message.split("Avg ulp Delta: ")[1].split(",")[0])
                # if avg_ulp.startswith("tensor("):
                #     avg_ulp = ulp_str[len("tensor(") :].rstrip(")")
                # try:
                #     avg_ulp = float(avg_ulp)
                # except ValueError:
                #     avg_ulp = float("nan")
        else:
            avg_ulp = 0
            max_ulp = 0
            ulp_passed = False
            # near_zero_pct=5
        # # # For positions where expected is near-zero, ULP doesn't make sense, skip
        # if near_zero_mask_expected.any():
        #     logger.info(
        #         f"{test_name}: Skipping ULP check for {near_zero_count_expected} positions where |expected| < {near_zero_threshold}"
        #     )

    # Prepare metrics dict
    metrics = {
        "pcc_passed": pcc_passed,
        "pcc_value": float(pcc_val) if isinstance(pcc_val, (int, float)) else pcc_val,
        "allclose_passed": allclose_passed,
        "max_atol": max_atol,
        "mean_atol": mean_atol,
        "max_rtol": max_rtol,
        "mean_rtol": mean_rtol,
        "frobenius_value": frob_val,
        "frobenius_expected_zero": frob_expected_zero,
        "ulp_passed": ulp_passed,
        "max_ulp": max_ulp,
        "avg_ulp": avg_ulp,
        "near_zero_pct": near_zero_pct,
    }
    # print(metrics)

    # Dump to CSV if filename provided
    if csv_filename:
        if csv_dir is None:
            # Try to get the calling file's directory
            import inspect

            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back
                caller_file = caller_frame.f_globals.get("__file__", "")
                if caller_file:
                    csv_dir = os.path.dirname(os.path.abspath(caller_file))
                else:
                    csv_dir = os.getcwd()
            finally:
                del frame

        csv_path = os.path.join(csv_dir, csv_filename)
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                # Build header row
                header = ["test_name"]
                if test_params:
                    header.extend(test_params.keys())
                header.extend(
                    [
                        "pcc_passed",
                        "pcc_value",
                        "allclose_passed",
                        "max_atol",
                        "mean_atol",
                        "max_rtol",
                        "mean_rtol",
                        "frobenius_value",
                        "frobenius_expected_zero",
                        "ulp_passed",
                        "max_ulp",
                        "avg_ulp",
                        "near_zero_pct",
                    ]
                )
                writer.writerow(header)

            # Build data row
            row = [test_name]
            if test_params:
                row.extend(test_params.values())
            row.extend(
                [
                    pcc_passed,
                    f"{metrics['pcc_value']:.6f}"
                    if isinstance(metrics["pcc_value"], (int, float))
                    else str(metrics["pcc_value"]),
                    allclose_passed,
                    f"{max_atol:.6e}",
                    f"{mean_atol:.6e}",
                    f"{max_rtol:.6e}",
                    f"{mean_rtol:.6e}",
                    f"{frob_val:.6e}",
                    frob_expected_zero,
                    ulp_passed,
                    f"{max_ulp:.1f}" if not math.isnan(max_ulp) else "N/A",
                    f"{avg_ulp:.1f}" if not math.isnan(avg_ulp) else "N/A",
                    f"{near_zero_pct:.2f}",
                ]
            )
            # print(row)
            writer.writerow(row)

    return metrics
