# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared steady-state iteration filter for utilization reports.

Works with or without NOC metrics.  When NOC UTIL (%) is present and
populated, only rows with valid NOC data are considered (original
behaviour).  When the column is absent or all-NaN (e.g. the NOC trace
pass timed out), all rows are used for pattern detection so the correct
iteration is still selected.
"""


def _detect_repeating_suffix(op_codes):
    """Find the longest-repeating suffix pattern in a list of OP CODE strings.

    Returns (iteration_size, repetition_count) or (None, 0).
    """
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
    return best_size, best_reps


def filter_last_steady_state_iteration(df, log=print):
    """
    Keep only the last steady-state iteration by detecting repeating OP CODE suffixes.

    If NOC UTIL (%) is present with valid data, rows are pre-filtered to
    those with NOC data before pattern detection (preserves legacy behaviour).
    Otherwise all rows are used, so the filter still works when the NOC
    trace pass was skipped or timed out.

    If log is None, status messages are suppressed.
    """
    npe_col = "NOC UTIL (%)"
    has_noc = npe_col in df.columns and df[npe_col].notna().any()

    if has_noc:
        valid = df[df[npe_col].notna()].copy()
    else:
        if log:
            log("  NOC UTIL column missing or empty — using all rows for iteration detection")
        valid = df.copy()

    if len(valid) == 0:
        if log:
            log("  Warning: No valid rows for iteration detection")
        return df

    op_codes = valid["OP CODE"].tolist()
    best_size, best_reps = _detect_repeating_suffix(op_codes)

    if best_size is not None:
        result = valid.iloc[-best_size:]
        if log:
            log(f"  Detected iteration size: {best_size} ops ({best_reps} repetitions found), keeping last iteration")
        return result.reset_index(drop=True)
    if log:
        log("  Warning: No repeating iteration pattern found, keeping all rows")
    return valid.reset_index(drop=True)
