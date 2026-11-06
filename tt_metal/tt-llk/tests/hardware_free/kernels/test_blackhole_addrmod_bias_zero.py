# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Invariant: no Blackhole code sets a nonzero address mod bias increment.

The Blackhole matmul inits (llk_math_matmul.h) program the source and dest words of an address mod with
addr_mod_t::set_no_bias and never write its bias word, which saves a third of the SETC16 of every init. That is only
correct while the bias register of every address mod stays 0, so this test fails on any code that could change it:
a nonzero bias increment in an addr_mod_t, a direct write of an ADDR_MOD_BIAS register, or a use of addr_mod_bias_t
outside the addr mod header. A writer that is needed must keep the bias write on its own path and extend this test.
"""

import re
from pathlib import Path

import pytest

TT_METAL = Path(__file__).resolve().parents[4]
ROOTS = [TT_METAL / "tt-llk" / "tt_llk_blackhole", TT_METAL / "hw"]
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".inc"}
# the addr mod header defines the bias fields and set(); the generated register definitions name the registers
ALLOWED = {"ckernel_addrmod.h", "cfg_defines.h"}

NONZERO_BIAS = re.compile(r"\.bias\s*=\s*\{[^}]*\bincr\s*=\s*(?!0\b)[0-9A-Za-z_]+")
BIAS_REGISTER = re.compile(r"ADDR_MOD_BIAS_SEC\d")
BIAS_TYPE = re.compile(r"\baddr_mod_bias_t\b")


def _code_lines(path):
    text = re.sub(
        r"/\*.*?\*/",
        lambda m: "\n" * m.group(0).count("\n"),
        path.read_text(errors="replace"),
        flags=re.S,
    )
    for number, line in enumerate(text.splitlines(), 1):
        yield number, line.split("//", 1)[0]


def _sources():
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (
                path.suffix in SUFFIXES
                and path.name not in ALLOWED
                and "build" not in path.parts[len(root.parts) :]
            ):
                yield path


def test_no_nonzero_addr_mod_bias_in_blackhole_code():
    if not any(root.is_dir() for root in ROOTS):
        pytest.skip("no tt_llk_blackhole or hw tree")
    found = []
    for path in _sources():
        for number, line in _code_lines(path):
            if (
                NONZERO_BIAS.search(line)
                or BIAS_REGISTER.search(line)
                or BIAS_TYPE.search(line)
            ):
                found.append(f"{path.relative_to(TT_METAL)}:{number}: {line.strip()}")
    assert not found, (
        "address mod bias is written, so the matmul inits must write it too:\n"
        + "\n".join(found)
    )
