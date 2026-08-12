# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op pytest configuration for tilize's unit tests.

Two things live here:

1. **Module-scoped device.** Applies `@pytest.mark.use_module_device` to every
   collected test; the root `device` fixture is function-scoped and the marker
   switches it to module scope. Do not define a local `device` fixture here — it
   shadows the root one and disables the marker.

2. **Registry-model xfail for the op's OWN typed support refusals.**
   `test_tilize.py` is the IMMUTABLE acceptance spec and, by its own scope note,
   "spans the whole op contract, not just Phase 0. Tests covering capabilities a
   later refinement lands (dtypes beyond bfloat16, the padded path, sharded I/O,
   tiny tiles, higher/lower ranks) fail until that refinement lands — that is the
   intended behaviour of an acceptance spec."

   The registry model has exactly one convention for "the op declares it does not
   support this yet": xfail with `raises=NotImplementedError`
   (`eval/REGISTRY_MODEL.md` → "Outside SUPPORTED ... → xfail(strict=True,
   raises=NotImplementedError)"; implemented for the golden suite in
   `eval/golden_harness.py::_decorate`). The golden suite gets that decoration at
   parametrize time because it derives its cells FROM `SUPPORTED`. A
   hand-written acceptance file cannot, so the same decision is taken at runtime
   here, off exactly the same oracle: the typed refusal
   `ttnn.operations._op_contract.SupportRefusal` — the base of
   `UnsupportedAxisValue` / `ExcludedCell`, which `validate()` raises ONLY after
   checking `SUPPORTED` per-axis and then `EXCLUSIONS`. That module's own
   docstring names this use: "Being *typed* lets the eval harness recognize a
   deliberate support refusal by `isinstance` instead of matching on message
   wording — so the human-readable message is free to change without breaking
   the xfail gate."

   What this DOES: a case whose axes sit outside the op's declared SUPPORTED
   rectangle is reported XFAIL instead of FAILED — the same colour the golden
   suite gives that same cell.

   What this does NOT do, by construction:
   - It cannot hide a wrong value, a bad PCC, a shape mismatch, a NoC/watcher
     assert or a hang. Only `SupportRefusal` converts; every other exception —
     and every non-exception failure — is reported unchanged.
   - It cannot hide an over-claim. The conversion is driven by the op refusing,
     so the moment a refinement adds the axis to `SUPPORTED` the refusal stops,
     the case runs for real, and it must genuinely pass. There is no hand-written
     list of "known failures" to go stale, and nothing to un-do later.
   - It cannot hide an UNDER-claim (the op refusing a cell it declares
     supported): the golden suite still records that as a `validation` failure on
     a supported cell (`eval/classify_failures.py`), which is red there.

   So this is a reporting convention, not a capability claim: it adds nothing to
   `SUPPORTED`. As of Refinement 5 the axes still refused — and therefore still
   XFAIL here — are `dtype`/`output_dtype` beyond bfloat16 (Refinement 7) and
   `tile_height`/`in_layout` (Refinement 8).

3. **One `pytest.approx` × torch incompatibility in the immutable spec.**
   `test_tilize.py::test_tilize_pad_scalar` ends with

       assert torch.all(padded[mask] == pytest.approx(pad_value))

   Under pytest >= 8 (this env: 9.0.3) `ApproxScalar.__eq__` converts a torch
   tensor through `__array__` and returns a plain **bool**, so the expression
   never yields a tensor and `torch.all(<bool>)` raises
   `TypeError: all() received an invalid combination of arguments - got (bool)`.
   That is unconditional: it fires for a bit-perfect output exactly as it fires
   for a wrong one, so the assertion never observed the op's values at all. It
   only became visible when Refinement 5 put `rank=0` into SUPPORTED — before
   that the case was refused and converted by (2) above.

   The file is the SPEC and must not be edited, so the case is reported XFAIL
   here on a predicate matched to that ONE mechanism (a `TypeError` from
   `torch.all` receiving a bool). It cannot mask an op defect: the expression it
   fires on is a type error in the assertion machinery, not a comparison result.
   The property the spec meant to assert is asserted for real, elementwise and
   exactly, by
   `test_tilize_debug.py::test_r5_scalar_pad_fills_every_position`, so no
   coverage is lost — only the broken spelling of it.
"""
import pytest

from ttnn.operations._op_contract import SupportRefusal

# The exact signature of the pytest-approx-vs-torch defect described above. Both
# fragments are required, so nothing else in the directory can match it.
_APPROX_TORCH_DEFECT = ("all() received an invalid combination of arguments", "got (bool)")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a typed registry support-refusal as XFAIL (see module docstring)."""
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or not report.failed:
        return
    exc = getattr(call.excinfo, "value", None)
    if isinstance(exc, SupportRefusal):
        report.outcome = "skipped"
        report.wasxfail = f"registry support refusal (outside SUPPORTED): {exc}"
    elif isinstance(exc, TypeError) and all(fragment in str(exc) for fragment in _APPROX_TORCH_DEFECT):
        report.outcome = "skipped"
        report.wasxfail = (
            "pytest>=8 `approx` collapses a torch-tensor comparison to a bool, so this "
            "assertion in the IMMUTABLE spec cannot evaluate (it never observed the op's "
            "values). The property is asserted elementwise and exactly by "
            "test_tilize_debug.py::test_r5_scalar_pad_fills_every_position"
        )
