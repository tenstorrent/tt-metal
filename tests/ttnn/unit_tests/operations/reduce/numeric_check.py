import os
import csv
import json
import math
import torch
from tests.ttnn.utils_for_testing import assert_with_ulp
from tests.ttnn.utils_for_testing import logger, _normalize_tensor, comp_relative_frobenius
from models.common.utility_functions import comp_pcc, comp_allclose_custom


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
NEAR_ZERO_THRESHOLD = 1e-04
FROBENIUS_GRID_SPLITS = 4  # Split each dim into N pieces → NxN tile grid
FROBENIUS_TOP_K = 5  # Report the K worst tiles
# # Padding validation configuration
# USE_FILL_PAD = False  # Enable fill_implicit_tile_padding with TEST_PADDING_VALUE for validation
# TEST_PADDING_VALUE = 0  # Value to fill implicit tile padding for validation


def _to_float_cpu_tensor(tensor):
    tensor = _normalize_tensor(tensor)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.detach().cpu().to(torch.float32)


def _reshape_to_2d(tensor):
    """Reshape any tensor to 2D for grid-based analysis."""
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def _compute_frobenius_error_clusters(
    expected,
    actual,
    n_splits=FROBENIUS_GRID_SPLITS,
    top_k=FROBENIUS_TOP_K,
):
    """
    Divide the tensor into an NxN grid and call comp_relative_frobenius on
    each tile.  Return tiles sorted worst-first.

    This is intentionally simple: no thresholds to tune, no flood-fill,
    no allclose mixing.  Every tile uses the exact same Frobenius formula
    as the global metric so the numbers are directly comparable.

    Args:
        expected: Expected tensor (PyTorch or TTNN).
        actual:   Actual tensor (PyTorch or TTNN).
        n_splits: Number of splits along each dimension (grid is n_splits x n_splits).
        top_k:    How many worst tiles to keep in the output.

    Returns:
        dict with metadata and a ``tiles`` list (worst first, up to *top_k*).
    """
    expected = _to_float_cpu_tensor(expected)
    actual = _to_float_cpu_tensor(actual)

    expected_2d = _reshape_to_2d(expected)
    actual_2d = _reshape_to_2d(actual)

    rows, cols = expected_2d.shape
    tile_h = max(1, math.ceil(rows / n_splits))
    tile_w = max(1, math.ceil(cols / n_splits))
    grid_rows = math.ceil(rows / tile_h)
    grid_cols = math.ceil(cols / tile_w)

    # Collect per-tile Frobenius + squared error (for error_share)
    tiles = []
    for gr in range(grid_rows):
        r0, r1 = gr * tile_h, min(rows, (gr + 1) * tile_h)
        for gc in range(grid_cols):
            c0, c1 = gc * tile_w, min(cols, (gc + 1) * tile_w)

            exp_tile = expected_2d[r0:r1, c0:c1]
            act_tile = actual_2d[r0:r1, c0:c1]

            tile_frob, tile_exp_zero = comp_relative_frobenius(exp_tile, act_tile)

            err = act_tile - exp_tile
            error_sq = float(torch.sum(err * err).item())
            max_abs = float(torch.abs(err).max().item()) if exp_tile.numel() > 0 else 0.0

            tiles.append(
                {
                    "grid_pos": [gr, gc],
                    "bbox": [r0, r1, c0, c1],
                    "elements": int(exp_tile.numel()),
                    "frobenius_value": float(tile_frob),
                    # "expected_norm_is_zero": bool(tile_exp_zero),
                    # "error_sq": error_sq,
                    # "max_abs_error": max_abs,
                }
            )

    # Compute error_share now that we have the total
    # total_error_sq = sum(t["error_sq"] for t in tiles)
    # for t in tiles:
    #     t["error_share"] = (t["error_sq"] / total_error_sq) if total_error_sq > 0 else 0.0

    # Sort worst-first, then assign rank
    tiles.sort(key=lambda t: t["frobenius_value"], reverse=True)
    for rank, t in enumerate(tiles, 1):
        t["rank"] = rank

    return {
        "original_shape": list(expected.shape),
        # "matrix_shape": [rows, cols],
        "tile_shape": [tile_h, tile_w],
        "grid_shape": [grid_rows, grid_cols],
        # "total_tiles": len(tiles),
        "cluster_count": len(tiles),  # back-compat alias
        "clusters": tiles[:top_k],  # back-compat alias (worst tiles)
    }


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
    k=None,
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

    # Mask out values less than NEAR_ZERO_THRESHOLD
    non_near_zero_mask = torch.abs(expected_allclose) >= NEAR_ZERO_THRESHOLD
    expected_allclose = expected_allclose[non_near_zero_mask]
    actual_allclose = actual_allclose[non_near_zero_mask]
    generate_graph = False
    if generate_graph:
        import matplotlib.pyplot as plt

        x_plot = expected_allclose.reshape(-1).detach().cpu().float().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, ".")
        plt.xlabel("Index")
        plt.ylabel("Expected Value")
        plt.title("Expected Allclose Values")
        plt.savefig("expected_allclose_plot.png")
        plt.close()

        x_plot = actual_allclose.reshape(-1).detach().cpu().float().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, ".")
        plt.xlabel("Index")
        plt.ylabel("Actual Value")
        plt.title("Actual Allclose Values")
        plt.savefig("actual_allclose_plot.png")
        plt.close()
    # print(f"expected_allclose.numel(): {expected_allclose.numel()}, expected.numel(): {expected.numel()}")
    if expected_allclose.numel() > 0 and actual_allclose.numel() > 0:
        allclose_passed, allclose_message = comp_allclose_custom(
            expected_allclose, actual_allclose, rtol=allclose_rtol, atol=allclose_atol
        )
        # Parse values from message: "Max ATOL Delta: {max_atol}, Mean ATOL Delta: {mean_atol}, Max RTOL Delta: {max_rtol}, Mean RTOL Delta: {mean_rtol}"
        max_abs_dif = float(allclose_message.split("Max ATOL Delta: ")[1].split(",")[0])
        mean_abs_dif = float(allclose_message.split("Mean ATOL Delta: ")[1].split(",")[0])
        max_rel_dif = float(allclose_message.split("Max RTOL Delta: ")[1].split(",")[0])
        mean_rel_dif = float(allclose_message.split("Mean RTOL Delta: ")[1].split(",")[0])
        max_abs_idx = int(allclose_message.split("Max Absolute Difference Index: ")[1])
        # print(f"max_abs_idx: {max_abs_idx}")
        # print(f"expected_allclose[{max_abs_idx}]: {expected_allclose.reshape(-1)[max_abs_idx]}")
        # print(f"actual_allclose[{max_abs_idx}]: {actual_allclose.reshape(-1)[max_abs_idx]}")
    else:
        allclose_passed = True
        allclose_message = "Both tensors are empty"
        max_abs_dif = 0
        mean_abs_dif = 0
        max_rel_dif = 0
        mean_rel_dif = 0
        max_abs_idx = 0

    # Relative Frobenius
    frob_val, frob_expected_zero = comp_relative_frobenius(expected, actual)
    if isinstance(frob_expected_zero, torch.Tensor):
        frob_expected_zero = bool(frob_expected_zero.item() if frob_expected_zero.numel() == 1 else frob_expected_zero)
    from_cluster_analysis = False
    if from_cluster_analysis:
        frob_cluster_summary = _compute_frobenius_error_clusters(expected, actual)
    else:
        frob_cluster_summary = None

    # ulp
    max_ulp = 0.0
    avg_ulp = 0.0
    near_zero_pct = 0.0
    ulp_passed = False

    # Filter out positions where |expected| < ATOL (to avoid division by zero in ULP)
    # near_zero_threshold = NEAR_ZERO_THRESHOLD
    total_elems = expected.numel()
    non_near_zero_mask = torch.abs(expected) >= NEAR_ZERO_THRESHOLD
    expected_non_near_zero = expected[non_near_zero_mask]
    actual_non_near_zero = actual[non_near_zero_mask]
    # for i in range(len(expected_non_near_zero)):
    #     print(f"expected_non_near_zero[i]: {expected_non_near_zero[i].item():.20f}, actual_non_near_zero[i]: {actual_non_near_zero[i].item():.20f}")
    # for i in range(len(expected_non_near_zero)):
    #     diff = expected_non_near_zero[i].item() - actual_non_near_zero[i].item()
    #     print(f"difference: {diff:.20f}")
    # print(f"expected_non_near_zer{expected_non_near_zero[i]).item():.20f}")
    # expected_non_near_zero = expected[non_near_zero_mask]
    # near_zero_count = (torch.abs(expected) < NEAR_ZERO_THRESHOLD).sum().item()
    near_zero_count = (non_near_zero_mask == False).sum().item()
    near_zero_pct = (near_zero_count / total_elems * 100) if total_elems > 0 else 0

    # if non_near_zero_mask.any():
    if expected_non_near_zero.numel() > 0 and actual_non_near_zero.numel() > 0:
        # expected_non_near_zero = expected[non_near_zero_mask]
        # actual_non_near_zero = actual[non_near_zero_mask]
        ulp_passed, ulp_message, ulp_message_index = assert_with_ulp(
            expected_non_near_zero, actual_non_near_zero, ulp_threshold=ulp_threshold, find_ulp_index=max_abs_idx
        )
        # print(f"ulp_passed: {ulp_passed}")
        # print(f"ulp_message: {ulp_message}")
        ulp_passed = bool(ulp_passed)
        if "Max ULP Delta:" in ulp_message:
            ulp_str = ulp_message.split("Max ULP Delta: ")[1].split(" ")[0]
            ulp_max_str = ulp_message.split("Max ULP Delta: ")[-1]
            # Handle tensor(...) format
            if ulp_str.startswith("tensor("):
                ulp_str = ulp_str[len("tensor(") :].rstrip(")")
            try:
                max_ulp = float(ulp_str)
            except ValueError:
                max_ulp = float("nan")

        if "Avg ulp Delta:" in ulp_message:
            avg_ulp = float(ulp_message.split("Avg ulp Delta: ")[1].split(",")[0])
    else:
        avg_ulp = 0
        max_ulp = 0
        ulp_passed = True
        ulp_max_str = None
        ulp_message_index = None

    # Prepare metrics dict
    metrics = {
        "pcc_passed": pcc_passed,
        "pcc_value": float(pcc_val) if isinstance(pcc_val, (int, float)) else pcc_val,
        "allclose_passed": allclose_passed,
        # "max_adiff": max_atol,
        "max_abs_dif": max_abs_dif,
        # "mean_atol": mean_atol,
        "mean_abs_dif": mean_abs_dif,
        # "max_rtol": max_rtol,
        "max_rel_dif": max_rel_dif,
        # "mean_rtol": mean_rtol,
        "mean_rel_dif": mean_rel_dif,
        "frobenius_value": frob_val,
        # "frobenius_expected_zero": frob_expected_zero,
        # "frobenius_cluster_count": frob_cluster_count,
        # "frobenius_largest_cluster_value": largest_frob_cluster_value,
        # "frobenius_largest_cluster_error_share": largest_frob_cluster_error_share,
        # "frobenius_cluster_summary": frob_cluster_summary,
        "ulp_passed": ulp_passed,
        "max_ulp": max_ulp,
        "avg_ulp": avg_ulp,
        "near_zero_pct": near_zero_pct,
        "ulp_max_str": ulp_max_str,
        "max_diff_ulp_str": ulp_message_index,
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
                        "max_abs_dif",
                        "mean_abs_dif",
                        "max_rel_dif",
                        "mean_rel_dif",
                        "ulp_passed",
                        "max_ulp",
                        "avg_ulp",
                        "near_zero_pct",
                        "frobenius_value",
                        # "frobenius_expected_zero",
                        # "frobenius_cluster_count",
                        # "frobenius_largest_cluster_value",
                        # "frobenius_largest_cluster_error_share",
                        # "frobenius_cluster_summary",
                        "ulp_max_data: (calculated-golden)/ulp_value",
                        "max_abs_diff_ulp: (calculated-golden)/ulp_value @ [max abs_diff_index]",
                    ]
                )
                if k is not None:
                    header.append("k")
                    if k >= 1:
                        header.extend(
                            [
                                "max_atol_div_k",
                                "mean_atol_div_k",
                                "max_rtol_div_k",
                                "mean_rtol_div_k",
                                "frobenius_value_div_k",
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
                    f"{max_abs_dif:.6e}",
                    f"{mean_abs_dif:.6e}",
                    f"{max_rel_dif:.6e}",
                    f"{mean_rel_dif:.6e}",
                    ulp_passed,
                    f"{max_ulp:.1f}" if not math.isnan(max_ulp) else "N/A",
                    f"{avg_ulp:.1f}" if not math.isnan(avg_ulp) else "N/A",
                    f"{near_zero_pct:.2f}",
                    f"{frob_val:.6e}",
                    # frob_expected_zero,
                    ulp_max_str,
                    ulp_message_index,
                    # frob_cluster_count,
                    # f"{largest_frob_cluster_value:.6e}",
                    # f"{largest_frob_cluster_error_share:.6e}",
                    # json.dumps(frob_cluster_summary, separators=(",", ":")),
                ]
            )
            if k is not None:
                row.append(str(k))
                if k >= 1:
                    row.extend(
                        [
                            f"{max_atol / k:.6e}",
                            f"{mean_atol / k:.6e}",
                            f"{max_rtol / k:.6e}",
                            f"{mean_rtol / k:.6e}",
                            f"{frob_val / k:.6e}",
                        ]
                    )
            # print(row)
            writer.writerow(row)

    return metrics
