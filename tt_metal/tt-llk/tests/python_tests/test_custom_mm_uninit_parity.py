# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Keep every custom-mm dense-stride programmer on one PACK helper.

The device tests call the production ``llk_pack_custom_mm.h`` helper directly and
check the stride it programs and restores, including skip-uninit and caller-MOP
negative controls. This static guard catches any compute family, the Blackhole
``llk_api`` wrapper pair, the deepseek model-private init or a driver bypassing that
shared implementation, and any raw write of the packer tile-stride register in the
model kernel trees that use custom_mm. It does not replace device coverage.

Skip in standalone tt-llk checkouts, where the compute API headers are absent.
"""

import re
from pathlib import Path

import pytest

# tests/python_tests -> tests -> tt-llk -> tt_metal
_TT_METAL = Path(__file__).resolve().parents[3]
_REPO = _TT_METAL.parent
_COMPUTE_API = _TT_METAL / "hw" / "inc" / "api" / "compute" / "experimental"
# The Blackhole llk_api wrapper pair: the one place the CB-to-pack-format lookup lives.
_PACK_API = (
    _TT_METAL
    / "hw"
    / "ckernels"
    / "blackhole"
    / "metal"
    / "llk_api"
    / "experimental"
    / "llk_pack_custom_mm_api.h"
)
_SOURCES = Path(__file__).resolve().parents[1] / "sources"
_DRIVER = _SOURCES / "custom_mm_uninit_restore_test.cpp"
# The matmul drivers pack dense through the same helper, so their format sweeps cover the
# stride it programs.
_MATMUL_DRIVERS = [
    _SOURCES / "custom_mm_test.cpp",
    _SOURCES / "compressed_custom_mm_test.cpp",
]
# Model kernel trees that call the custom_mm family. The deepseek tree carries a private
# compressed short init that programs the dense stride itself; the whole trees are swept
# for raw writes of the stride register.
_MODELS = _REPO / "models" / "demos"
_MODEL_TREES = [_MODELS / "deepseek_v3_b1", _MODELS / "gpt_oss"]
_DEEPSEEK_COMMON = (
    _MODELS
    / "deepseek_v3_b1"
    / "kernel_includes"
    / "tt_metal"
    / "hw"
    / "ckernels"
    / "blackhole"
    / "metal"
    / "llk_api"
    / "llk_custom_mm_compressed_common.h"
)
_MODEL_INIT = (_DEEPSEEK_COMMON, "custom_mm_compressed_block_init_short")
_KERNEL_SUFFIXES = {".h", ".hpp", ".cpp"}

# Both compute APIs must delegate to the wrapper exercised by the device tests.
_UNINIT_PAIR = [
    ("custom_mm.h", "custom_mm_block_uninit"),
    ("compressed_custom_mm.h", "compressed_custom_mm_block_uninit"),
]

# Any field of the packer tile-stride register (RMW, ADDR32, SHAMT, MASK ...).
_STRIDE_REGISTER = "PCK0_ADDR_CTRL_ZW_REG_0_Wstride"

_API_INIT_CALL = "PACK((llk_pack_custom_mm_init<dense_packing>(out_cb_id)));"
_API_UNINIT_CALL = "PACK((llk_pack_custom_mm_uninit<dense_packing>(out_cb_id)));"
_HELPER_CALL = {
    "init": "_llk_pack_custom_mm_init_<dense_packing>(pack_src_format[output_id]);",
    "uninit": "_llk_pack_custom_mm_uninit_<dense_packing>(pack_src_format[output_id]);",
}

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
        rf"\b(?:ALWI|inline)\s+void\s+{re.escape(function)}\s*\([^)]*\)\s*\{{"
    ).search(source, offset)
    assert match, f"could not find 'void {function}(...)' -- has it been renamed?"

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


def _read_pack_api():
    if not _PACK_API.is_file():
        pytest.skip(f"llk_api wrapper not found: {_PACK_API}")
    return _PACK_API.read_text()


def _assert_no_raw_stride_write(name, source):
    assert _STRIDE_REGISTER not in _strip_comments(source), (
        f"{name} writes the packer tile-stride register ({_STRIDE_REGISTER}*) directly "
        "instead of going through the shared PACK helper, so the device tests of that "
        "helper no longer cover the stride it programs."
    )


def test_pack_api_wrappers_delegate_to_the_shared_helper():
    """The llk_api pair is the only CB-to-format lookup; each calls the helper once."""
    source = _strip_comments(_read_pack_api())
    for operation, call in _HELPER_CALL.items():
        body = _normalize(_extract_body(source, f"llk_pack_custom_mm_{operation}"))
        assert body.count(call) == 1, (
            f"llk_pack_custom_mm_{operation} must call {call} exactly once; the restore "
            f"driver exercises that helper with the same arguments. Found: {body}"
        )
    _assert_no_raw_stride_write(_PACK_API.name, source)


def test_custom_mm_uninit_bodies_have_not_diverged():
    """Both uninits must consist only of the shared PACK-wrapper call."""
    sources = _read_headers()
    for header, function in _UNINIT_PAIR:
        body = _normalize(_extract_body(sources[header], function))
        assert body == _API_UNINIT_CALL, (
            f"{function} no longer delegates only to the shared PACK wrapper. "
            "The LLK restore driver must exercise the same implementation as both "
            f"compute APIs. Found: {body}"
        )


def test_driver_and_compute_apis_use_the_shared_pack_helper():
    """The drivers and all compute variants must use the same stride setup."""
    sources = _read_headers()
    if not _DRIVER.is_file():
        pytest.skip(f"driver not found: {_DRIVER}")
    driver = _normalize(_DRIVER.read_text())

    assert '#include "experimental/llk_pack_custom_mm.h"' in driver
    for operation in ("init", "uninit"):
        call = (
            f"_llk_pack_custom_mm_{operation}_<UNINIT_DENSE_PACKING>(formats.pack_src);"
        )
        assert call in driver, (
            f"The restore driver no longer calls {call}; its coverage must exercise "
            "the same PACK helper as the compute APIs."
        )

    matmul_drivers = {}
    for path in _MATMUL_DRIVERS:
        if not path.is_file():
            pytest.skip(f"driver not found: {path}")
        matmul_drivers[path.name] = _normalize(path.read_text())
        assert (
            '#include "experimental/llk_pack_custom_mm.h"' in matmul_drivers[path.name]
        )
        for operation in ("init", "uninit"):
            call = f"_llk_pack_custom_mm_{operation}_<DENSE_PACKING>(formats.pack_src);"
            assert call in matmul_drivers[path.name], (
                f"{path.name} no longer calls {call}; its format sweep must exercise "
                "the same PACK helper as the compute APIs."
            )

    # Typed overloads inherit the legacy header's PACK-wrapper include.
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
            assert body.count(_API_INIT_CALL) == 1, (
                f"{header}: {match.group(1)} overload {index + 1} must configure "
                "the stride exactly once through the shared PACK wrapper."
            )

    # Duplicating the register write would restore the old blind spot: the driver
    # could pass while a compute API programs a different stride.
    for name, source in {**sources, _DRIVER.name: driver, **matmul_drivers}.items():
        _assert_no_raw_stride_write(name, source)


def test_model_private_init_uses_the_shared_pack_helper():
    """The deepseek compressed short init programs the dense stride like the API does."""
    path, function = _MODEL_INIT
    if not path.is_file():
        pytest.skip(f"model-private header not found: {path}")
    code = _strip_comments(path.read_text())
    body = _normalize(_extract_body(code, function))
    assert body.count(_API_INIT_CALL) == 1, (
        f"{path.name}: {function} must configure the stride exactly once through the "
        f"shared PACK wrapper, as the compute APIs do. Found: {body}"
    )
    _assert_no_raw_stride_write(path.name, code)


def test_model_kernels_do_not_write_the_stride_register():
    """No kernel in the custom_mm model trees programs the packer tile stride itself."""
    present = [tree for tree in _MODEL_TREES if tree.is_dir()]
    if not present:
        pytest.skip(f"model trees not found: {[str(tree) for tree in _MODEL_TREES]}")
    for tree in present:
        for path in sorted(tree.rglob("*")):
            if path.suffix in _KERNEL_SUFFIXES and path.is_file():
                _assert_no_raw_stride_write(
                    str(path.relative_to(_REPO)), path.read_text(errors="replace")
                )
