"""
Evaluation metrics for LLVC (Low-Latency Low-Resource Voice Conversion).

This module provides functions to compute standard metrics:
    - Mel Cepstral Distortion (MCD)
    - Inference latency statistics (mean, std, percentiles)

All implementations are designed for production use: type-annotated, documented,
and with appropriate error handling.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MCD_SCALE = (10.0 / np.log(10.0))  # scaling factor for MCD in dB
_EPSILON = 1e-10                     # avoid log(0) if needed in other metrics

# ---------------------------------------------------------------------------
# MCD (Mel Cepstral Distortion)
# ---------------------------------------------------------------------------

def compute_mcd(
    target_mc: np.ndarray,
    predicted_mc: np.ndarray,
    axis: int = 1
) -> float:
    """
    Compute Mel Cepstral Distortion between two sets of mel-cepstral coefficients.

    Parameters
    ----------
    target_mc : np.ndarray
        Reference mel-cepstral coefficients. Shape (frames, order) or (order,).
    predicted_mc : np.ndarray
        Synthesized mel-cepstral coefficients. Same shape as `target_mc`.
    axis : int, default=1
        Axis along which the coefficients are arranged (0 for order, 1 for frames).
        Only used when `target_mc` is 2-D.

    Returns
    -------
    float
        Mean MCD in dB over all frames. Lower is better.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or dimensions.
    """
    target_mc = np.asarray(target_mc, dtype=np.float64)
    predicted_mc = np.asarray(predicted_mc, dtype=np.float64)

    if target_mc.shape != predicted_mc.shape:
        raise ValueError(
            f"Shape mismatch: target {target_mc.shape} vs predicted {predicted_mc.shape}"
        )

    # Ensure one-dimensional case
    if target_mc.ndim == 1:
        diff_sq = (target_mc - predicted_mc) ** 2
        # For single frame, MCD = (10/ln(10)) * sqrt(2 * sum(diff_sq))
        # Typically the first coefficient (c0) is excluded because it represents
        # energy; we include it by default, but this can be overridden.
        mcd = _MCD_SCALE * np.sqrt(2.0 * float(np.sum(diff_sq)))
        return float(mcd)

    if target_mc.ndim == 2:
        if axis == 0:
            # Coefficients organized as (order, frames) -> transpose
            diff = target_mc - predicted_mc
            sum_sq = np.sum(diff ** 2, axis=0)  # sum over order per frame
        elif axis == 1:
            # Coefficients organized as (frames, order)
            diff = target_mc - predicted_mc
            sum_sq = np.sum(diff ** 2, axis=1)  # sum over order per frame
        else:
            raise ValueError(f"Axis must be 0 or 1, got {axis}")

        # Per-frame MCD
        mcd_per_frame = _MCD_SCALE * np.sqrt(2.0 * sum_sq)
        return float(np.mean(mcd_per_frame))

    raise ValueError(
        f"Input arrays must be 1-D or 2-D, got {target_mc.ndim} dimensions."
    )

def compute_mcd_with_skip_first_coeff(
    target_mc: np.ndarray,
    predicted_mc: np.ndarray,
    axis: int = 1,
    skip_c0: bool = True
) -> float:
    """
    Compute MCD optionally excluding the first coefficient (c0, energy).

    Parameters
    ----------
    target_mc, predicted_mc : np.ndarray
        Mel-cepstral coefficient arrays.
    axis : int, default=1
        Axis for coefficients (if 2-D).
    skip_c0 : bool, default=True
        If True, exclude the first coefficient (index 0) from computation.

    Returns
    -------
    float
        Mean MCD in dB.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes.
    """
    target_mc = np.asarray(target_mc, dtype=np.float64)
    predicted_mc = np.asarray(predicted_mc, dtype=np.float64)

    if target_mc.shape != predicted_mc.shape:
        raise ValueError("Shape mismatch between target and predicted.")

    if skip_c0:
        if target_mc.ndim == 1:
            if target_mc.shape[0] < 2:
                raise ValueError("Need at least 2 coefficients to skip c0.")
            return compute_mcd(target_mc[1:], predicted_mc[1:], axis=axis)
        elif target_mc.ndim == 2:
            if axis == 1:
                if target_mc.shape[1] < 2:
                    raise ValueError("Need at least 2 coefficients to skip c0.")
                return compute_mcd(target_mc[:, 1:], predicted_mc[:, 1:], axis=axis)
            else:  # axis=0
                if target_mc.shape[0] < 2:
                    raise ValueError("Need at least 2 coefficients to skip c0.")
                return compute_mcd(target_mc[1:, :], predicted_mc[1:, :], axis=axis)
        else:
            raise ValueError("Input arrays must be 1-D or 2-D.")
    else:
        return compute_mcd(target_mc, predicted_mc, axis=axis)

# ---------------------------------------------------------------------------
# Inference latency metrics
# ---------------------------------------------------------------------------

def compute_latency_metrics(
    latencies: Union[List[float], np.ndarray],
    percentiles: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute statistics for inference latency measurements.

    Parameters
    ----------
    latencies : list or np.ndarray
        Per-invocation latency values in seconds (or any consistent unit).
    percentiles : list of float, optional
        Additional percentile values to compute (0–100). Default is [50, 95, 99].

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'min', 'max', 'p50', 'p95', 'p99'
        plus any user‑requested percentiles as 'p<value>'.

    Raises
    ------
    ValueError
        If `latencies` is empty or contains negative values.
    """
    if percentiles is None:
        percentiles = [50.0, 95.0, 99.0]

    latencies = np.asarray(latencies, dtype=np.float64)
    if latencies.size == 0:
        raise ValueError("Latency array is empty.")
    if np.any(latencies < 0):
        raise ValueError("Negative latency values are not allowed.")

    results = {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies, ddof=1) if latencies.size > 1 else 0.0),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
    }

    # Compute percentiles ensuring keys are sorted for readability
    for p in sorted(percentiles):
        key = f'p{p:.0f}'
        if p < 0 or p > 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {p}.")
        results[key] = float(np.percentile(latencies, p))

    return results

# ---------------------------------------------------------------------------
# Combined evaluation function
# ---------------------------------------------------------------------------

def evaluate_llvc(
    target_mc_list: List[np.ndarray],
    predicted_mc_list: List[np.ndarray],
    latencies: Optional[Union[List[float], np.ndarray]] = None,
    skip_c0: bool = True
) -> Dict[str, float]:
    """
    Compute full set of evaluation metrics for LLVC.

    Parameters
    ----------
    target_mc_list : list of np.ndarray
        Reference mel-cepstral sequences for each utterance.
    predicted_mc_list : list of np.ndarray
        Synthesized mel-cepstral sequences for each utterance.
    latencies : list or np.ndarray, optional
        Per-utterance inference latencies (in seconds).
    skip_c0 : bool, default=True
        Whether to skip the first coefficient (c0) in MCD.

    Returns
    -------
    dict
        Keys:
            - 'mcd_mean' : overall mean MCD across all frames of all utterances.
            - 'mcd_std'   : standard deviation of per‑utterance mean MCD.
            - 'latency_mean', 'latency_std', ... (if `latencies` provided).

    Raises
    ------
    ValueError
        If any input list is empty or lengths mismatch.
    """
    if len(target_mc_list) != len(predicted_mc_list):
        raise ValueError(
            f"Number of target ({len(target_mc_list)}) and predicted "
            f"({len(predicted_mc_list)}) sequences must match."
        )
    if len(target_mc_list) == 0:
        raise ValueError("No utterances provided.")

    # Compute per-utterance MCD
    mcd_per_utterance = []
    for t, p in zip(target_mc_list, predicted_mc_list):
        mcd_val = compute_mcd_with_skip_first_coeff(t, p, skip_c0=skip_c0)
        mcd_per_utterance.append(mcd_val)

    mcd_array = np.array(mcd_per_utterance, dtype=np.float64)

    results = {
        'mcd_mean': float(np.mean(mcd_array)),
        'mcd_std': float(np.std(mcd_array, ddof=1) if len(mcd_array) > 1 else 0.0),
        'mcd_min': float(np.min(mcd_array)),
        'mcd_max': float(np.max(mcd_array)),
    }

    if latencies is not None:
        lat_dict = compute_latency_metrics(latencies)
        # Prefix with 'latency_' to avoid key collision
        results.update({f'latency_{k}': v for k, v in lat_dict.items()})

    return results

# ---------------------------------------------------------------------------
# Example usage (demonstrative, not executed during import)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Simple demonstration with synthetic data
    np.random.seed(42)
    n_frames, order = 100, 24

    # Create dummy target and predicted coefficients
    target = np.random.randn(n_frames, order) * 0.5
    predicted = target + np.random.randn(n_frames, order) * 0.1

    # Compute MCD
    mcd = compute_mcd(target, predicted)
    print(f"MCD (all coeffs): {mcd:.4f} dB")

    mcd_skip = compute_mcd_with_skip_first_coeff(target, predicted, skip_c0=True)
    print(f"MCD (skip c0): {mcd_skip:.4f} dB")

    # Latency metrics
    latencies = np.random.exponential(scale=0.02, size=200)  # ~20 ms mean
    lat_metrics = compute_latency_metrics(latencies)
    print(f"Latency stats: {lat_metrics}")

    # Full evaluation
    all_targets = [target[:50], target[50:]]
    all_predicted = [predicted[:50], predicted[50:]]
    eval_results = evaluate_llvc(all_targets, all_predicted, latencies)
    print(f"Evaluation results: {eval_results}")