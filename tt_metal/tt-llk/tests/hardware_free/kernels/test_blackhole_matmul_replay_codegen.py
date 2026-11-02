# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check Blackhole matmul's emitted RDCFG dependencies and replay boundaries."""

import itertools
import re
import subprocess
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[2]
_LLK = _TESTS.parent
_METAL = _LLK.parent
_COMPILER = _TESTS / "sfpi/compiler/bin/riscv-tt-elf-g++"
_CASES = list(itertools.product((False, True), repeat=3))


def _name(reuse_a, partial, broadcast):
    return f"probe_{int(reuse_a)}_{int(partial)}_{int(broadcast)}"


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    if not _COMPILER.is_file():
        pytest.skip(f"sfpi compiler not present at {_COMPILER}")
    source = ['#include "llk_unpack_AB_matmul.h"']
    for reuse_a, partial, broadcast in _CASES:
        ct, rt = (4, 1) if reuse_a else (1, 4)
        ba, bb = (0, int(broadcast)) if reuse_a else (int(broadcast), 0)
        pa, pb = (partial, False) if reuse_a else (False, partial)
        source.append(
            f'extern "C" __attribute__((flatten)) void {_name(reuse_a, partial, broadcast)}() {{'
            f"_llk_unpack_AB_matmul_mop_config_<{ba}, {bb}>"
            f"({ct}, {rt}, {str(pa).lower()}, {str(pb).lower()}); }}"
        )
    source.append(
        'extern "C" void ref_stall() {'
        "TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::CFGEXU); }"
    )
    source.append('extern "C" void ref_nop() { TTI_NOP; }')
    tmp = tmp_path_factory.mktemp("blackhole_matmul_replay")
    cpp, assembly = tmp / "probe.cpp", tmp / "probe.s"
    cpp.write_text("\n".join(source))
    bh = _LLK / "tt_llk_blackhole"
    includes = [
        bh,
        bh / "common/inc",
        bh / "common/inc/sfpu",
        bh / "llk_lib",
        _LLK / "common",
        _TESTS / "sfpi/include",
        _METAL / "hw/inc",
        _METAL / "hw/inc/internal/tt-1xx",
        _METAL / "hw/inc/internal/tt-1xx/blackhole",
    ]
    subprocess.run(
        [
            str(_COMPILER),
            "-mcpu=tt-bh-tensix",
            "-std=c++17",
            "-S",
            "-O2",
            "-DCOMPILE_FOR_TRISC=0",
            "-DTENSIX_FIRMWARE",
            *[f"-I{path}" for path in includes],
            "-o",
            str(assembly),
            str(cpp),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    functions = {}
    current = None
    for line in assembly.read_text().splitlines():
        label = re.match(r"^(probe_\d_\d_\d|ref_stall|ref_nop):", line)
        if label:
            current = functions[label[1]] = {"words": [], "records": []}
        if current is None:
            continue
        word = re.search(r"\.ttinsn\s+(0x[0-9a-fA-F]+|-?\d+)", line)
        if word:
            current["words"].append(int(word[1], 0) & 0xFFFFFFFF)
        record = re.search(r"TTREPLAY\s+(\d+),\s*(\d+),\s*0,\s*1", line)
        if record:
            current["records"].append((int(record[1]), int(record[2])))
    return functions


@pytest.mark.parametrize("reuse_a,partial", itertools.product((False, True), repeat=2))
def test_rdcfg_waits_for_configuration_unit(emitted, reuse_a, partial):
    words = emitted[_name(reuse_a, partial, False)]["words"]
    (stall,) = emitted["ref_stall"]["words"]
    reads = [i for i, word in enumerate(words) if word >> 24 == 0xB1]
    assert (
        len(reads) == 2
    ), "Both configuration contexts must advance their base address"
    for index in reads:
        assert (
            words[index + 1] == stall
        ), "RDCFG must complete before ADDDMAREG reads its result"
        assert (
            words[index + 2] >> 24 == 0x58
        ), "Expected the dependent ADDDMAREG after the wait"


@pytest.mark.parametrize("reuse_a,partial,broadcast", _CASES)
def test_replay_length_and_context_boundaries(emitted, reuse_a, partial, broadcast):
    actual = emitted[_name(reuse_a, partial, broadcast)]
    words = actual["words"]
    assert actual["records"] == [
        (0, len(words))
    ], "Record exactly the emitted replay body"
    assert len(words) % 2 == 0
    half = len(words) // 2
    (nop,) = emitted["ref_nop"]["words"]
    assert (
        words[half - 1] == words[-1] == nop
    ), "Each context must end after its WRCFG delay"
    assert len(words) == len(emitted[_name(reuse_a, partial, not broadcast)]["words"])
    for context in (words[:half], words[half:]):
        reads = sum(word >> 24 == 0xB1 for word in context)
        assert reads == (0 if broadcast else 1)
