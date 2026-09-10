# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static guard: the two custom_mm uninit bodies, and the driver's copy of them.

``test_custom_mm_uninit_restore.py`` covers what ``custom_mm_block_uninit`` and
``compressed_custom_mm_block_uninit`` do, but it does it by **replicating** their shared
body in ``sources/custom_mm_uninit_restore_test.cpp`` -- a tt-llk driver cannot include
``tt_metal/hw/inc/api/compute``. Two blind spots follow, and neither is visible to any
runtime test:

1. **Divergence.** The two compute-API bodies are currently identical, so one driver covers
   both. If they diverge, every existing test keeps passing and the driver silently stops
   describing one of them.
2. **Staleness.** The driver hardcodes the W-stride expressions rather than deriving them.
   If a header's constants change, the driver keeps asserting the old behaviour and still
   passes.

This file closes both textually. It is the cheap interim guard for what really wants a
metal-side test calling the real entry points -- one that can include
``tt_metal/hw/inc/api/compute``, which a tt-llk test cannot; it
does not replace that, because a text match cannot tell you the functions *work* -- only
that they still say the same thing.

Static and device-free, in the same spirit as ``test_perf_header_gate.py``. It reaches
outside the tt-llk tree to read the compute API, which no other test here does, so it skips
cleanly when those headers are absent -- as they are in a standalone tt-llk checkout.
"""

import re
from pathlib import Path

import pytest

# tests/python_tests -> tests -> tt-llk -> tt_metal
_TT_METAL = Path(__file__).resolve().parents[3]
_COMPUTE_API = _TT_METAL / "hw" / "inc" / "api" / "compute" / "experimental"
_DRIVER = (
    Path(__file__).resolve().parents[1]
    / "sources"
    / "custom_mm_uninit_restore_test.cpp"
)

# (header, uninit function) pairs whose bodies must stay in lockstep.
_UNINIT_PAIR = [
    ("custom_mm.h", "custom_mm_block_uninit"),
    ("compressed_custom_mm.h", "compressed_custom_mm_block_uninit"),
]

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _strip_comments(text):
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _normalize(text):
    """Comments out, whitespace collapsed -- so only the code itself is compared."""
    return " ".join(_strip_comments(text).split())


def _extract_body(source, function):
    """Return the brace-delimited body of `function`, by brace matching.

    Deliberately not a C++ parse: the bodies here are a handful of statements, and a
    dependency-free matcher keeps this test cheap enough to be a static gate.
    """
    match = re.search(
        rf"\bALWI\s+void\s+{re.escape(function)}\s*\([^)]*\)\s*\{{", source
    )
    assert match, f"could not find 'ALWI void {function}(...)' -- has it been renamed?"

    depth = 0
    start = source.index("{", match.start())
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"unbalanced braces in {function}")


def _read_headers():
    missing = [name for name, _ in _UNINIT_PAIR if not (_COMPUTE_API / name).is_file()]
    if missing:
        pytest.skip(
            f"compute API not present ({', '.join(missing)}) -- expected in a standalone "
            "tt-llk checkout, where there is nothing to guard"
        )
    return {name: (_COMPUTE_API / name).read_text() for name, _ in _UNINIT_PAIR}


def test_custom_mm_uninit_bodies_have_not_diverged():
    """The two uninit bodies must stay identical, because one driver covers both."""
    sources = _read_headers()
    bodies = {
        function: _normalize(_extract_body(sources[header], function))
        for header, function in _UNINIT_PAIR
    }

    (first_fn, first), (second_fn, second) = bodies.items()
    assert first == second, (
        f"{first_fn} and {second_fn} no longer share a body, so "
        "custom_mm_uninit_restore_test.cpp -- which replicates that shared body rather than "
        "calling either function -- now describes at most one of them, while continuing to "
        "pass for both.\n\n"
        f"  {first_fn}:\n    {first}\n\n"
        f"  {second_fn}:\n    {second}\n\n"
        "Either restore the shared body, or split the driver and its test so each family "
        "is covered on its own -- properly, that means a metal-side test that includes the "
        "compute API headers and calls both entry points, which a tt-llk test cannot do."
    )


def test_driver_wstride_constants_match_the_compute_api():
    """The driver's replicated W-stride expressions must still be the header's.

    The driver spells these out so a change on either side shows up as a diff; this asserts
    the diff is actually noticed. Without it the driver can keep asserting a stride the
    compute API no longer programs -- and pass, since it programs that stride itself.
    """
    sources = _read_headers()
    if not _DRIVER.is_file():
        pytest.skip(f"driver not found: {_DRIVER}")
    driver = _normalize(_DRIVER.read_text())

    # As spelled in custom_mm_uninit_restore_test.cpp, and in both headers.
    expressions = {
        "DENSE_WSTRIDE": "(TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2",
        "DEFAULT_WSTRIDE": "TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2",
    }

    for name, expression in expressions.items():
        assert f"{name} = {expression}" in driver, (
            f"custom_mm_uninit_restore_test.cpp no longer defines {name} as "
            f"'{expression}'. If that is deliberate, update this test and check the "
            "headers agree; if not, the driver has drifted from the compute API."
        )
        for header in sources:
            assert expression in _normalize(sources[header]), (
                f"{header} no longer contains the W-stride expression '{expression}' that "
                f"the driver replicates as {name}. The driver is now asserting a stride "
                "the compute API does not program, and will keep passing because it "
                "programs that stride itself. Reconcile the two."
            )
