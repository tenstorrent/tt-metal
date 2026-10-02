# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Keep custom-mm compute APIs and the LLK restore driver on one PACK helper.

The device tests call the production ``llk_pack_custom_mm.h`` helper directly and
check stride restoration, including skip-uninit and caller-MOP negative controls.
This static guard catches either compute family or the driver bypassing that
shared implementation. It does not replace device coverage of its behavior.

Skip in standalone tt-llk checkouts, where the compute API headers are absent.
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

# Both compute APIs must delegate to the helper exercised by the device tests.
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


def _extract_body(source, function, offset=0):
    """Return the brace-delimited body of `function`, by brace matching.

    Deliberately not a C++ parse: the bodies here are a handful of statements, and a
    dependency-free matcher keeps this test cheap enough to be a static gate.
    """
    match = re.compile(
        rf"\bALWI\s+void\s+{re.escape(function)}\s*\([^)]*\)\s*\{{"
    ).search(source, offset)
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
    """Both uninits must consist only of the shared PACK-helper call."""
    sources = _read_headers()
    expected = "PACK((_llk_pack_custom_mm_uninit_<dense_packing>()));"
    for header, function in _UNINIT_PAIR:
        body = _normalize(_extract_body(sources[header], function))
        assert body == expected, (
            f"{function} no longer delegates only to the shared PACK helper. "
            "The LLK restore driver must exercise the same implementation as both "
            f"compute APIs. Found: {body}"
        )


def test_driver_and_compute_apis_use_the_shared_pack_helper():
    """The driver and all compute variants must use the same stride setup."""
    sources = _read_headers()
    if not _DRIVER.is_file():
        pytest.skip(f"driver not found: {_DRIVER}")
    driver = _normalize(_DRIVER.read_text())

    assert '#include "experimental/llk_pack_custom_mm.h"' in driver
    for operation in ("init", "uninit"):
        call = f"_llk_pack_custom_mm_{operation}_<UNINIT_DENSE_PACKING>();"
        assert call in driver, (
            f"The restore driver no longer calls {call}; its coverage must exercise "
            "the same PACK helper as the compute APIs."
        )

    # Typed overloads inherit the legacy header's PACK-helper include.
    for header in list(sources):
        typed_header = _COMPUTE_API / "2_0" / header
        if typed_header.is_file():
            sources[f"2_0/{header}"] = typed_header.read_text()
    for header, source in sources.items():
        code = _strip_comments(source)
        function = f"{Path(header).stem}_block_init"
        overloads = list(
            re.finditer(
                rf"\bALWI\s+void\s+({re.escape(function)}(?:_short)?)\s*\(", code
            )
        )
        assert overloads, f"{header} has no custom-mm init overloads to check."
        for index, match in enumerate(overloads):
            body = _normalize(_extract_body(code, match.group(1), match.start()))
            assert (
                body.count("PACK((_llk_pack_custom_mm_init_<dense_packing>()));") == 1
            ), (
                f"{header}: {match.group(1)} overload {index + 1} must configure "
                "the stride exactly once through the shared PACK helper."
            )

    # Duplicating the register write would restore the old blind spot: the driver
    # could pass while a compute API programs a different stride.
    for name, source in {**sources, _DRIVER.name: driver}.items():
        assert "PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW" not in _strip_comments(
            source
        ), f"{name} writes the stride directly instead of using the shared PACK helper."
