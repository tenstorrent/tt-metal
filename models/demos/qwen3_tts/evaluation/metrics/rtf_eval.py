# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RTF (Real-Time Factor) and latency evaluation.
RTF = wall_clock_time / audio_duration. RTF < 1.0 means faster than real-time.
"""

import numpy as np


def compute_rtf_summary(results):
    """
    Compute RTF statistics from benchmark results.

    Args:
        results: List of dicts with 'id', 'elapsed', 'duration', 'rtf' keys

    Returns:
        Dict with RTF statistics
    """
    rtfs = [r["rtf"] for r in results if r["rtf"] != float("inf")]
    durations = [r["duration"] for r in results]
    elapsed_times = [r["elapsed"] for r in results]

    if not rtfs:
        return {"error": "No valid RTF measurements"}

    return {
        "mean_rtf": float(np.mean(rtfs)),
        "median_rtf": float(np.median(rtfs)),
        "min_rtf": float(np.min(rtfs)),
        "max_rtf": float(np.max(rtfs)),
        "std_rtf": float(np.std(rtfs)),
        "p90_rtf": float(np.percentile(rtfs, 90)),
        "p99_rtf": float(np.percentile(rtfs, 99)),
        "total_audio_duration_s": float(sum(durations)),
        "total_elapsed_s": float(sum(elapsed_times)),
        "num_samples": len(rtfs),
        "realtime_capable": float(np.mean(rtfs)) < 1.0,
    }
