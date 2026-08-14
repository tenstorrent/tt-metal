# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for the Quasar SFPU parity gate. Host-only: no kernel, no device.

The parity tests are written ahead of the kernels they exercise, and each op is gated on
its header existing. That gate is only trustworthy if two things stay true, and neither is
checked by any test that runs a kernel:

  * a kernel that lands must actually turn its tests on -- if the header appears but the
    C++ dispatcher has no branch for it, the op silently stays untested while the Python
    side reports it as ported;
  * a parity header name must not collide with an unrelated Quasar kernel -- a collision
    would open the gate for an op nobody ported, and the dispatch would reference symbols
    that do not exist.

Both are cheap to assert and expensive to notice by hand, so they are pinned here. These
tests pass today with zero kernels ported, and keep passing as kernels arrive; they fail
only when the table and the tree disagree.
"""

import re

import pytest
from helpers.sfpu_port_quasar import (
    CONVERSIONS_COVERAGE,
    QUASAR_SFPU_PARITY,
    Arity,
    entry_for,
    resolve_header,
)

# The dispatcher whose branches the gate switches on.
_DISPATCHER = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "helpers"
    / "include"
    / "sfpu_operations_quasar.h"
)

# The parity set as published by the SFPU parity dashboard at the commit the table was
# built from. Pinned so that trimming an op out of the table is a deliberate edit here
# rather than a silent loss of coverage.
_EXPECTED_KERNEL_COUNT = 57


@pytest.fixture(scope="module")
def dispatcher_source() -> str:
    assert _DISPATCHER.is_file(), f"dispatcher not found at {_DISPATCHER}"
    return _DISPATCHER.read_text()


def test_parity_table_matches_the_published_set():
    """The table still describes the full parity set, with no duplicates."""
    assert len(QUASAR_SFPU_PARITY) == _EXPECTED_KERNEL_COUNT, (
        f"parity table has {len(QUASAR_SFPU_PARITY)} kernels, expected "
        f"{_EXPECTED_KERNEL_COUNT}. If the dashboard's set genuinely changed, update "
        "_EXPECTED_KERNEL_COUNT along with the table."
    )

    kernels = [e.kernel for e in QUASAR_SFPU_PARITY]
    duplicates = sorted({k for k in kernels if kernels.count(k) > 1})
    assert (
        not duplicates
    ), f"kernels listed more than once in the parity table: {duplicates}"

    ops = [op for e in QUASAR_SFPU_PARITY for op in e.ops]
    dup_ops = sorted({o.name for o in ops if ops.count(o) > 1})
    assert not dup_ops, (
        "these MathOperations are claimed by more than one parity kernel, so "
        f"entry_for() would resolve one of them arbitrarily: {dup_ops}"
    )


def test_only_the_helper_row_has_no_ops():
    """Every dispatchable entry names at least one op; only HELPER rows may be empty."""
    empty = [
        e.kernel
        for e in QUASAR_SFPU_PARITY
        if not e.ops and e.arity is not Arity.HELPER
    ]
    assert not empty, (
        "these parity entries declare no MathOperation but are not marked "
        f"Arity.HELPER, so nothing can ever drive them: {empty}"
    )


def test_conversions_coverage_points_at_live_parity_ops():
    """The helper header's coverage claim must be backed by ops that still exist.

    ckernel_sfpu_conversions.h has no entry point of its own and is covered through its
    callers. If one of those callers left the parity set, the claim would be quietly
    false -- the helper would be listed as covered by a test that no longer runs.
    """
    for op in CONVERSIONS_COVERAGE:
        assert entry_for(op) is not None, (
            f"{op.name} is named as a conversions-helper carrier but is no longer in "
            "the parity set, so ckernel_sfpu_conversions.h would be uncovered"
        )

    helper_rows = [e for e in QUASAR_SFPU_PARITY if e.arity is Arity.HELPER]
    for entry in helper_rows:
        assert entry.covered_by, (
            f"helper row {entry.kernel} declares no covered_by, so nothing records how "
            "it is exercised"
        )
        for op in entry.covered_by:
            assert op in CONVERSIONS_COVERAGE, (
                f"{entry.kernel} names {op.name} as a carrier but CONVERSIONS_COVERAGE "
                "does not say what that op injects, so the claim is unauditable"
            )


def test_parity_headers_do_not_collide_with_existing_quasar_kernels():
    """No parity header name may already be taken by an unrelated Quasar kernel.

    A collision opens the gate for an op nobody ported: the Python side would start
    emitting variants and the C++ dispatcher would compile a branch against symbols the
    colliding header does not define.
    """
    collisions = []
    for entry in QUASAR_SFPU_PARITY:
        found = resolve_header(entry)
        if found is None:
            continue
        # The header exists. That is legitimate only if it really is the ported kernel,
        # which we approximate by requiring it to define at least one of the entry's
        # expected calculate symbols.
        text = found.read_text()
        if not any(sym in text for sym in entry.call):
            collisions.append(
                f"{entry.kernel}: {found} exists but defines none of {list(entry.call)}"
            )
    assert not collisions, (
        "these parity header names are already taken by something that is not the "
        "ported kernel, so the gate would open spuriously:\n  "
        + "\n  ".join(collisions)
    )


def test_ported_kernels_have_a_dispatch_branch(dispatcher_source):
    """A kernel that landed must actually be reachable from the dispatcher.

    Without this, a port that adds the header but forgets the dispatch branch leaves the
    op reported as ported and silently untested.
    """
    missing = []
    for entry in QUASAR_SFPU_PARITY:
        if entry.arity is Arity.HELPER or resolve_header(entry) is None:
            continue
        if f"#define {entry.guard_macro}" not in dispatcher_source:
            missing.append(f"{entry.kernel}: no '#define {entry.guard_macro}' guard")
            continue
        for sym in entry.call:
            if sym not in dispatcher_source:
                missing.append(
                    f"{entry.kernel}: header is present but the dispatcher never "
                    f"calls {sym}"
                )
    assert (
        not missing
    ), "ported parity kernels are not wired into the dispatcher:\n  " + "\n  ".join(
        missing
    )


def test_every_dispatchable_entry_has_a_guard(dispatcher_source):
    """Each non-helper entry has a guard defined and consumed in the dispatcher.

    This is the structural half of the gate, and unlike the test above it holds whether or
    not the kernel exists yet -- so it catches a table entry added without the matching
    C++ plumbing at the moment it is added, rather than months later when the port lands.
    """
    defined = set(re.findall(r"#define (QSR_HAS_\w+)", dispatcher_source))
    used = set(re.findall(r"#ifdef (QSR_HAS_\w+)", dispatcher_source))

    for entry in QUASAR_SFPU_PARITY:
        if entry.arity is Arity.HELPER:
            assert entry.guard_macro not in defined, (
                f"{entry.kernel} is a helper-only header with no entry point, so it must "
                "not have a dispatch guard"
            )
            continue
        assert entry.guard_macro in defined, (
            f"parity kernel {entry.kernel} has no '#define {entry.guard_macro}' in "
            f"{_DISPATCHER.name}; its tests could never activate"
        )
        assert entry.guard_macro in used, (
            f"{entry.guard_macro} is defined but never consumed by an #ifdef, so the "
            f"{entry.kernel} dispatch branch is unreachable"
        )

    orphans = sorted(defined - {e.guard_macro for e in QUASAR_SFPU_PARITY})
    assert (
        not orphans
    ), f"{_DISPATCHER.name} defines guards with no parity table entry: {orphans}"
