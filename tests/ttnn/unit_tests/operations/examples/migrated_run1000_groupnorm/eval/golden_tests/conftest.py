# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest config for the registry-model golden-test suites.

Hosts the one thing every op's translated suite needs identically: a
runtime xfail-gate for cells the op refuses via SUPPORTED / EXCLUSIONS.

Why a runtime hook (and not collection-time marks like test_golden.py)
-----------------------------------------------------------------------
test_golden.py generates its cells from TARGET x INPUTS, so it owns an
explicit per-cell `axes` dict at collection and can xfail-strict the
unsupported ones up front (see eval/golden_harness._decorate).

test_translated.py cells are reference-shaped: arbitrary parametrize
(the case is translated verbatim from a production ttnn test), and the
support-relevant axes are *shape-derived* — they only come into
existence when the op's INPUT_TAGGERS reduce the concrete tensor shape
inside validate(), at run time. There is no per-cell axes dict at
collection. So the only place the refusal is knowable is when the call
actually runs and validate() raises NotImplementedError.

This hookwrapper catches that specific refusal and converts it to a
**lenient** xfail, scoped two ways:
  1. only on tests carrying @pytest.mark.translated_from, and
  2. only on a typed SupportRefusal (ttnn.operations._op_contract) — the
     exception validate() raises for a deliberate refusal. Matched by
     isinstance, so the message wording is free to change.

Anything else — a numeric assertion failure, or an unrelated
NotImplementedError (a genuine op bug) — stays a real failure. Golden
remains the strict support contract; this is its lenient mirror for the
reference-shaped suite. Lenient (not strict) is deliberate: a translated
cell silently flips to pass the moment the op grows to support it, with
no per-suite XPASS bookkeeping — golden already polices XPASS drift.

Note on loop-bearing cells: a translated test whose body loops over
several shapes (e.g. `for shape in input_shapes:`) is one pytest case.
If any iteration hits a refused cell, validate() raises and the whole
case xfails — pytest cannot partially xfail within a single case. This
matches the prior hard-fail granularity; it is not a regression.
"""

import pytest

from ttnn.operations._op_contract import SupportRefusal


def _release_device_tensors_in_traceback(tb):
    """Deallocate every on-device ttnn tensor bound as a local in traceback
    `tb`'s frames.

    Why this exists — the L1-clash cascade. Under the module-scoped device, a
    cell that raises (op TT_THROWs at CB finalize, a SupportRefusal xfail, any
    error) leaves its exception's __traceback__ holding the run_<op>() frame,
    whose locals still point at the cell's on-device input/output tensors.
    pytest retains that excinfo, and those frames are trapped in an
    exc<->tb<->frame reference cycle whose retaining structures live in gc's
    oldest generation — so plain refcounting can't reclaim them, a young-gen
    gc.collect() misses them, and only a full gc.collect() (whole-heap walk,
    far too slow per cell) frees them. The pinned L1 then collides with the
    next same-shape cell's static circular-buffer region (program.cpp CB-vs-L1
    clash), failing that cell too -> a self-sustaining cascade over a whole
    shape group. (Interleaved cells escape as the *seed* because their tensors
    live in DRAM, not L1; but they still clash as victims against a sharded
    seed.)

    The pinned tensors are exactly the failing call's frame locals, so they are
    freed there directly: walk the traceback's frames and deallocate any ttnn
    tensor. Deterministic, general across allocation paths (inputs, op output,
    compound-op intermediates — anything the failing stack held), no gc
    heap-walk, no per-op edits, no function-scoped-device reopen. The report's
    longrepr is already built by the time this runs, so only the live device
    buffers are dropped, not the failure report.
    """
    import ttnn

    def _dealloc_if_device(obj):
        if isinstance(obj, ttnn.Tensor):
            try:
                if obj.storage_type() == ttnn.StorageType.DEVICE:
                    ttnn.deallocate(obj)
            except Exception:
                pass

    while tb is not None:
        for val in tb.tb_frame.f_locals.values():
            _dealloc_if_device(val)
            # shallow scan of the common containers ops return tensors in
            if isinstance(val, (list, tuple)):
                for el in val:
                    _dealloc_if_device(el)
            elif isinstance(val, dict):
                for el in val.values():
                    _dealloc_if_device(el)
        tb = tb.tb_next


def _is_support_refusal(exc: BaseException) -> bool:
    """True iff `exc` is the op's validate() refusing an unsupported cell.

    Typed match: validate() raises a SupportRefusal subclass
    (UnsupportedAxisValue / ExcludedCell, defined in ttnn) for a deliberate
    refusal. An unrelated NotImplementedError — a real bug — is NOT a
    SupportRefusal, so it still surfaces as a failure. No message matching.
    """
    return isinstance(exc, SupportRefusal)


def pytest_configure(config):
    # Registered here once for every op's translated suite; per-op
    # conftests only register their own op-specific markers (e.g.
    # `numerics`).
    config.addinivalue_line(
        "markers",
        "translated_from(source, commit): test translated from a reference "
        "ttnn op test at the given source path and tt-metal commit (both "
        "carried on the marker args)",
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if (
        call.when == "call"
        and "translated_from" in item.keywords
        and call.excinfo is not None
        and _is_support_refusal(call.excinfo.value)
    ):
        # Mark as xfail the same way the xfail machinery does internally:
        # outcome "skipped" + a `wasxfail` reason → reported and counted
        # as xfailed ('x'), not failed.
        rep.outcome = "skipped"
        rep.wasxfail = f"op refused (SUPPORTED/EXCLUSIONS): {call.excinfo.value}"
    # Free device tensors pinned by a failing/xfail cell's traceback frames so
    # their L1 can't clash the next module-device cell (see the helper's docs).
    if call.excinfo is not None:
        _release_device_tensors_in_traceback(call.excinfo.tb)
    return rep
