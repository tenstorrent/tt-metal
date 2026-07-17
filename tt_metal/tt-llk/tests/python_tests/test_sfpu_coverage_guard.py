# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SFPU functional-test coverage guard.

Turns "every SFPU op has a functional test" into a machine-checked invariant:
parse the authoritative ``SfpuType`` enum, subtract the ops that are actually
registered in ``MathOperation`` (i.e. exercised by a functional test), and
subtract the explicitly-waived ops in ``WAIVED``. Anything left over is a new,
untested SFPU op and fails this test.

When you add a new ``SfpuType`` enumerator you must either:
  * wire it to a functional test (add a ``MathOperation`` + dispatch/golden), or
  * add it to ``WAIVED`` below with a concrete reason.

See optimization_results/SFPU_COVERAGE_GAP_AND_REMEDIATION.md for the burn-down
plan behind this ledger.
"""

import re
from pathlib import Path

import pytest
from helpers.llk_params import MathOperation

# ---------------------------------------------------------------------------
# WAIVED: SfpuType enumerators that are intentionally NOT registered as a
# functional-test MathOperation. Every entry needs a concrete reason.
# ---------------------------------------------------------------------------
WAIVED = {
    # -- covered through a non-unary / separate harness (not a gap) -----------
    "div_int32_trunc": "exercised via binary DIV_INT32 (calculate_div_int32_trunc)",
    "reduce": "covered by the dedicated reduce harness (test_sfpu_reduce.py)",
    # -- dead / placeholder enum ---------------------------------------------
    "unused": "placeholder enumerator, no kernel behind it",
    # -- GROUP B: integer / format-typed variants -----------------------------
    # The integer *unary* harness now exists (test_eltwise_unary_sfpu_int): shifts and
    # unary max/min int32/uint32 are wired and removed from this ledger. The entries
    # below are the remaining integer gaps.
    #   unary-with-scalar arithmetic: kernels exist (calculate_add_int32 / binop_with_scalar
    #   unsigned variants) and can now use the int harness, but each needs its own scalar
    #   plumbing + golden; left as a focused follow-up.
    "add_uint32": "gap(B): uint32 add-with-scalar; int harness ready, needs scalar plumbing + golden",
    "add_uint16": "gap(B): uint16 add-with-scalar; int harness ready, needs scalar plumbing + golden",
    "sub_uint16": "gap(B): uint16 sub-with-scalar; int harness ready, needs scalar plumbing + golden",
    "mul_uint16": "gap(B): uint16 mul-with-scalar; int harness ready, needs scalar plumbing + golden",
    "remainder_int32": "gap(B): int32 remainder; int harness ready, needs golden+dispatch",
    "remainder_uint32": "gap(B): uint32 remainder; int harness ready, needs golden+dispatch",
    "fmod_int32": "gap(B): int32 fmod; int harness ready, needs golden+dispatch",
    #   binary integer ops belong in the binary harness (test_sfpu_binary.py), not the
    #   unary int harness added here.
    "eq_int": "gap(B): int32 binary eq; needs binary-int path",
    "ne_int": "gap(B): int32 binary ne; needs binary-int path",
    "max_int32": "gap(B): int32 binary max; needs binary-int path",
    "min_int32": "gap(B): int32 binary min; needs binary-int path",
    "max_uint32": "gap(B): uint32 binary max; needs binary-int path",
    "min_uint32": "gap(B): uint32 binary min; needs binary-int path",
    # -- GROUP C: quantization -----------------------------------------------
    "quant_int32": "gap(C): quantization; scalar scale/zero-point golden pending",
    "requant_int32": "gap(C): requantization; scalar scale/zero-point golden pending",
    "dequant_int32": "gap(C): dequantization; scalar scale/zero-point golden pending",
    # -- GROUP D: stateful / cross-lane; need a dedicated driver -------------
    "cumsum": "gap(D): running-sum, cross-tile carry; needs dedicated driver",
    "reshuffle_rows": "gap(D): cross-row permute; needs dedicated driver",
    "tiled_prod": "gap(D): tile-wide product reduction; needs dedicated driver",
    "alt_complex_rotate90": "gap(D): complex rotate; needs dedicated driver",
    "cpy_values": "gap(D): cross-lane copy; needs dedicated driver",
    "max_pool_with_indices": "gap(D): pooling+indices; needs dedicated driver",
    # -- GROUP E: non-deterministic; need a statistical (non-PCC) test -------
    "dropout": "gap(E): stochastic; needs distributional test (no PCC golden)",
}


def _find_sfpu_types_header() -> Path:
    """Locate llk_sfpu_types.h by walking up from this test file."""
    rel = Path("hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tt_metal" / rel
        if candidate.is_file():
            return candidate
        candidate = parent / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("llk_sfpu_types.h not found walking up from test file")


def _parse_sfpu_types(header: Path) -> set[str]:
    body = re.search(
        r"enum class SfpuType\s*\{(.*?)\};", header.read_text(), re.S
    ).group(1)
    names = set()
    for line in body.splitlines():
        token = line.split("//")[0].strip().rstrip(",").strip()
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            names.add(token)
    return names


def _tested_cpp_values() -> set[str]:
    """cpp_enum_value of every registered MathOperation, lowercased.

    Lowercasing lets a unary SfpuType (`tan`) match, and also lets binary/ternary
    ops whose enum name differs only in case (`MAX` -> `max`) count as covered.
    """
    return {op.cpp_enum_value.lower() for op in MathOperation}


def test_every_sfpu_type_is_tested_or_waived():
    try:
        header = _find_sfpu_types_header()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    universe = _parse_sfpu_types(header)
    tested = _tested_cpp_values()
    waived = {name.lower() for name in WAIVED}

    missing = {
        s for s in universe if s.lower() not in tested and s.lower() not in waived
    }

    assert not missing, (
        "New/untested SFPU ops found. For each, either wire a functional test "
        "(add a MathOperation + dispatch/golden) or add it to WAIVED with a "
        f"reason:\n  {sorted(missing)}"
    )


def test_waived_entries_are_still_untested():
    """Keep WAIVED honest: once an op gets a real test, drop it from WAIVED."""
    tested = _tested_cpp_values()
    stale = sorted(name for name in WAIVED if name.lower() in tested)
    assert not stale, (
        "These ops are now registered as MathOperations but are still listed in "
        f"WAIVED — remove them from WAIVED:\n  {stale}"
    )


def test_waived_reasons_are_nonempty():
    bad = sorted(name for name, reason in WAIVED.items() if not reason.strip())
    assert not bad, f"WAIVED entries missing a reason: {bad}"
