# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compile the opt-in untilize guard with LLK assertions disabled."""

import re
import subprocess
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[2]
_TT_METAL = Path(__file__).resolve().parents[4]
_LLK = _TT_METAL / "tt-llk"
_BLACKHOLE = _LLK / "tt_llk_blackhole"
_PACK_HEADER_DIR = _BLACKHOLE / "llk_lib"
_COMPILER = _TESTS_ROOT / "sfpi/compiler/bin/riscv-tt-elf-g++"

_SOURCE = """
#ifdef ENABLE_LLK_ASSERT
#error This regression must compile with LLK assertions disabled
#endif
#include "llk_pack_untilize.h"
extern "C" {
void invalid_mop() { _llk_pack_untilize_mop_config_<1, false, false, true>(16, 1); }
void invalid_init() { _llk_pack_untilize_init_<1, 1, false, TILE_C_DIM, false, true>(5, 5, 16, 1); }
void invalid_two_faces_mop() { _llk_pack_untilize_mop_config_<1, false, false, true>(16, 2); }
void invalid_two_faces_init() { _llk_pack_untilize_init_<1, 1, false, TILE_C_DIM, false, true>(5, 5, 16, 2); }
void valid_four_faces() { _llk_pack_untilize_mop_config_<1, false, false, true>(16, 4); }
void default_mop() { _llk_pack_untilize_mop_config_<1>(16, 1); }
void default_init() { _llk_pack_untilize_init_<1>(5, 5, 16, 1); }
void default_runtime(unsigned n) { _llk_pack_untilize_mop_config_<1>(16, n); }
void tilize_runtime(unsigned n) { _llk_pack_untilize_mop_config_<1, false, false, true>(16, n); }
}
"""


@pytest.fixture(scope="module")
def assembly(tmp_path_factory):
    if not (_TT_METAL / "hw/inc").is_dir():
        pytest.skip("tt-metal hw/inc is absent; the probe includes those headers")
    if not _COMPILER.exists():
        pytest.skip(f"sfpi compiler not present at {_COMPILER}")

    folder = tmp_path_factory.mktemp("blackhole_untilize_source")
    source = folder / "probe.cpp"
    source.write_text(_SOURCE)
    output = folder / "probe.s"
    includes = [
        _PACK_HEADER_DIR,
        _TESTS_ROOT / "sfpi/include",
        _BLACKHOLE,
        _BLACKHOLE / "common/inc",
        _BLACKHOLE / "common/inc/sfpu",
        _BLACKHOLE / "llk_lib",
        _LLK / "common",
        _TT_METAL / "hw/inc",
        _TT_METAL / "hw/inc/internal/tt-1xx/blackhole",
    ]
    subprocess.run(
        [
            str(_COMPILER),
            "-mcpu=tt-bh-tensix",
            "-std=c++17",
            "-S",
            "-O2",
            "-DARCH_BLACKHOLE",
            "-DCOMPILE_FOR_TRISC=2",
            "-DTENSIX_FIRMWARE",
            "-DENV_LLK_INFRA",
            "-UENABLE_LLK_ASSERT",
            *[f"-I{path}" for path in includes],
            "-o",
            str(output),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return output.read_text()


def _body(assembly, name):
    match = re.search(rf"^{name}:\n(.*?)^\s*\.size\s+{name},", assembly, re.M | re.S)
    assert match is not None, f"Missing probe function {name}"
    return match.group(1)


@pytest.mark.parametrize(
    "entrypoint",
    ["invalid_mop", "invalid_init", "invalid_two_faces_mop", "invalid_two_faces_init"],
)
def test_unsupported_tilize_source_traps_without_asserts(assembly, entrypoint):
    body = _body(assembly, entrypoint)
    assert re.search(r"^\s*ebreak\s*$", body, re.M), body
    assert not re.search(r"^\s*ret\s*$", body, re.M), body


@pytest.mark.parametrize(
    "entrypoint", ["valid_four_faces", "default_mop", "default_init"]
)
def test_supported_and_default_layouts_do_not_trap(assembly, entrypoint):
    body = _body(assembly, entrypoint)
    assert not re.search(r"^\s*ebreak\s*$", body, re.M), body
    assert re.search(r"^\s*ret\s*$", body, re.M), body


def test_runtime_face_count_guard_is_opt_in(assembly):
    assert re.search(r"^\s*ebreak\s*$", _body(assembly, "tilize_runtime"), re.M)
    assert not re.search(r"^\s*ebreak\s*$", _body(assembly, "default_runtime"), re.M)
