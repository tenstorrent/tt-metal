# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixed-tail capacity search planning; never certifies capacity."""


def search_residue(probe, residue, lower, upper):
    """Search lengths32*b+r, assuming (without proving) monotonicity per r.

    A failed minimum stops this residue: later passing holes remain possible.
    Every inferred boundary requires fresh-process and full public/HF checks.
    """
    if residue not in range(1, 33) or lower < 32 or upper < lower:
        raise ValueError("Require residue1..32 and32 <= lower <= upper")
    first = max(0, (lower - residue + 31) // 32)
    last = (upper - residue) // 32
    if first > last:
        raise ValueError("Search interval contains no length for this residue")
    result = {
        "residue": residue,
        "length_formula": "32 * block_index + residue; residue32 means aligned",
        "first_block_index": first,
        "last_block_index": last,
        "observations": [],
        "largest_passing_candidate": None,
        "next_observed_failure": None,
        "candidate_only": True,
        "monotonicity_assumed_not_proven": True,
    }

    def check(block):
        length = 32 * block + residue
        passed = bool(probe(length))
        result["observations"].append({"block_index": block, "length": length, "passed": passed})
        return passed

    if not check(first):
        result.update(next_observed_failure=32 * first + residue, stopped_at_failed_minimum=True)
        return result
    if last == first or check(last):
        result.update(largest_passing_candidate=32 * last + residue, upper_search_bound_passed=True)
        return result
    low, high = first, last
    while high - low > 1:
        middle = (low + high) // 2
        if check(middle):
            low = middle
        else:
            high = middle
    result.update(largest_passing_candidate=32 * low + residue, next_observed_failure=32 * high + residue)
    return result
