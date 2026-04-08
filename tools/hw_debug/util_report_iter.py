# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared steady-state iteration filter for utilization reports (NOC-based heuristic)."""


def filter_last_steady_state_iteration(df, log=print):
    """
    Keep only the last steady-state iteration by detecting repeating OP CODE suffixes.

    If log is None, status messages are suppressed.
    """
    npe_col = "NOC UTIL (%)"
    if npe_col not in df.columns:
        return df

    valid = df[df[npe_col].notna()].copy()
    if len(valid) == 0:
        if log:
            log("  Warning: No rows with NOC UTIL data found")
        return df

    op_codes = valid["OP CODE"].tolist()
    n = len(op_codes)
    best_size = None
    best_reps = 0
    for size in range(1, n // 2 + 1):
        reps = 1
        while reps * size + size <= n:
            if op_codes[n - (reps + 1) * size : n - reps * size] == op_codes[n - size :]:
                reps += 1
            else:
                break
        if reps >= 2 and (best_size is None or reps > best_reps or (reps == best_reps and size < best_size)):
            best_size = size
            best_reps = reps

    if best_size is not None:
        result = valid.iloc[-best_size:]
        if log:
            log(f"  Detected iteration size: {best_size} ops ({best_reps} repetitions found), keeping last iteration")
        return result.reset_index(drop=True)
    if log:
        log("  Warning: No repeating iteration pattern found, keeping all rows with NOC data")
    return valid.reset_index(drop=True)
