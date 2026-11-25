# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only canonical spans; logical lengths never become API restrictions."""


def prefill_plan(start, end):
    """Yield absolute half-open spans from the finite 1..512/2048 family."""
    if not 0 <= start < end:
        raise ValueError("Expected a nonempty nonnegative prefill span")
    while start < end:
        remaining = end - start
        if start % 512:
            count = min(remaining, 512 - start % 512)
        elif remaining >= 2048:
            count = 2048
        elif remaining >= 512:
            count = 512
        else:
            count = remaining
        yield start, start + count
        start += count
