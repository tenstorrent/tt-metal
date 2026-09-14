# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
``t6_semaphore_post``/``t6_semaphore_get`` (Quasar) take three independently
optional pre-stall resources. Each wrapper carries its own copy of the gate, so
every case is asserted on both: each slot on its own produces a stall, all three
together produce one, naming none produces none, and the emitted word is exactly
what a hand-written ``TTI_STALLWAIT`` with the same slots produces.

Compile-only. It builds a probe translation unit against the Quasar headers with
the sfpi compiler and reads the ``.ttinsn`` words back out of the assembly, so it
needs no Quasar device -- the Quasar compile job is the one that runs it, which is
also the job a change to the guarded header triggers. Every probe is paired
with a reference function that hand-writes the expected stall, which keeps the
expected encoding in the C++ where the constants live rather than duplicating
opcode and field values here.
"""

import re
import subprocess
from pathlib import Path

import pytest

# The Quasar compile job selects on this marker; without it the guard would not run
# on a change to the header it guards.
pytestmark = pytest.mark.quasar

_TESTS_ROOT = Path(__file__).resolve().parents[2]
_TT_METAL = Path(__file__).resolve().parents[4]
_COMPILER = _TESTS_ROOT / "sfpi/compiler/bin/riscv-tt-elf-g++"
_QUASAR_LLK = _TT_METAL / "tt-llk/tt_llk_quasar"

# Each probe_* calls the wrapper; the ref_* beside it hand-writes the stall that
# wrapper must emit. TTI_STALLWAIT takes (stall_res, idx_2, idx_1, idx_0), which
# is why the reference argument order mirrors the template arguments reversed.
_PROBE_SOURCE = """
#include "ckernel_trisc_common.h"

using namespace ckernel;
using namespace ckernel::trisc;

extern "C" void probe_none() { t6_semaphore_post<>(3); }
extern "C" void ref_none() {}

extern "C" void probe_w0() { t6_semaphore_post<p_stall::MATH>(3); }
extern "C" void ref_w0() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH); }

extern "C" void probe_w1() { t6_semaphore_post<p_stall::NOTHING, p_stall::MATH>(3); }
extern "C" void ref_w1() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::MATH, p_stall::NOTHING); }

extern "C" void probe_w2() { t6_semaphore_post<p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH>(3); }
extern "C" void ref_w2() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::MATH, p_stall::NOTHING, p_stall::NOTHING); }

extern "C" void probe_all() { t6_semaphore_post<p_stall::THCON, p_stall::MATH, p_stall::PACK0>(3); }
extern "C" void ref_all() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::PACK0, p_stall::MATH, p_stall::THCON); }

extern "C" void probe_get_none() { t6_semaphore_get<>(3); }
extern "C" void ref_get_none() {}

extern "C" void probe_get_w0() { t6_semaphore_get<p_stall::MATH>(3); }
extern "C" void ref_get_w0() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH); }

extern "C" void probe_get_w1() { t6_semaphore_get<p_stall::NOTHING, p_stall::MATH>(3); }
extern "C" void ref_get_w1() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::MATH, p_stall::NOTHING); }

extern "C" void probe_get_w2() { t6_semaphore_get<p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH>(3); }
extern "C" void ref_get_w2() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::MATH, p_stall::NOTHING, p_stall::NOTHING); }

extern "C" void probe_get_all() { t6_semaphore_get<p_stall::THCON, p_stall::MATH, p_stall::PACK0>(3); }
extern "C" void ref_get_all() { TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::PACK0, p_stall::MATH, p_stall::THCON); }
"""

_LABEL = re.compile(r"^(probe_\w+|ref_\w+):")
_TTINSN = re.compile(r"\.ttinsn\s+(-?\d+|0x[0-9a-fA-F]+)")


def _emitted_words(assembly: str) -> dict:
    """Map each ``extern "C"`` function to the instruction words it emits."""
    words, current = {}, None
    for line in assembly.splitlines():
        stripped = line.strip()
        label = _LABEL.match(stripped)
        if label:
            current = label.group(1)
            words[current] = []
        insn = _TTINSN.search(stripped)
        if insn and current is not None:
            words[current].append(int(insn.group(1), 0) & 0xFFFFFFFF)
    return words


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    if not _COMPILER.exists():
        pytest.skip(f"sfpi compiler not present at {_COMPILER}")

    tmp = tmp_path_factory.mktemp("quasar_stall_probe")
    source = tmp / "probe.cpp"
    source.write_text(_PROBE_SOURCE)
    assembly = tmp / "probe.s"

    subprocess.run(
        [
            str(_COMPILER),
            "-std=c++17",
            "-S",
            "-O2",
            # TRISC id and the firmware guard that keeps host-only headers out.
            "-DCOMPILE_FOR_TRISC=0",
            "-DTENSIX_FIRMWARE",
            f"-I{_QUASAR_LLK / 'common/inc'}",
            f"-I{_QUASAR_LLK / 'common/inc/sfpu'}",
            f"-I{_QUASAR_LLK / 'llk_lib'}",
            f"-I{_TT_METAL / 'tt-llk/common'}",
            f"-I{_TT_METAL / 'hw/inc'}",
            f"-I{_TT_METAL / 'hw/inc/internal/tt-2xx/quasar'}",
            "-o",
            str(assembly),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return _emitted_words(assembly.read_text())


@pytest.mark.parametrize(
    "probe",
    ["w0", "w1", "w2", "all", "get_w0", "get_w1", "get_w2", "get_all"],
)
def test_named_resource_emits_its_stall(emitted, probe):
    got, expected = emitted[f"probe_{probe}"], emitted[f"ref_{probe}"]
    assert expected, f"ref_{probe} emitted no stall -- the reference itself is wrong"
    assert got == expected, (
        f"t6_semaphore_{'get' if probe.startswith('get') else 'post'} with slots "
        f"'{probe}' emitted {[hex(w) for w in got]}, expected "
        f"{[hex(w) for w in expected]}. A named wait resource in any slot must "
        f"produce the same STALLWAIT as writing it by hand."
    )


@pytest.mark.parametrize("wrapper", ["post", "get"])
def test_no_named_resource_emits_no_stall(emitted, wrapper):
    probe = "probe_none" if wrapper == "post" else "probe_get_none"
    assert emitted[probe] == [], (
        f"t6_semaphore_{wrapper}<> named no wait resource but still emitted "
        f"{[hex(w) for w in emitted[probe]]}"
    )
