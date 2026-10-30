# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoding against the headers: every TT_OP_* macro expanded by the real preprocessor."""

import random
import re
import subprocess

import pytest
from conftest import needs_toolchain
from hal_explorer.isa import instructions
from hal_explorer.toolchain import build


def expand(calls):
    """Integer value of each TT_OP_* call, expanded with the compiler's preprocessor."""
    flags = [f for f in build.compile_flags() if f.startswith(("-I", "-D"))]
    source = (
        '#include "ckernel_ops.h"\n' + "\n".join(f"@@ {call}" for call in calls) + "\n"
    )
    output = subprocess.run(
        [build.compile_flags()[0], "-E", "-P", "-x", "c++", *flags, "-"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    values = []
    for line in output.splitlines():
        if line.startswith("@@ "):
            expression = line[3:]
            assert re.fullmatch(r"[\s0-9a-fA-Fx()+<>|&]*", expression), expression
            values.append(eval(expression) & 0xFFFFFFFF)  # pylint: disable=eval-used
    return values


@needs_toolchain
def test_every_macro_round_trips(tables):
    rng = random.Random(0)
    calls, expected = [], []
    for layout in tables.opcodes.values():
        widths = {f.name: f.width for f in layout.fields}
        for pattern in ("max", "random", "zero"):
            values = []
            for parameter in layout.parameters:
                width = widths.get(parameter, 0)
                top = (1 << width) - 1
                values.append(
                    {"max": top, "random": rng.randint(0, top), "zero": 0}[pattern]
                )
            calls.append(
                f"TT_OP_{layout.mnemonic}({', '.join(map(str, values))})"
                if layout.parameters
                else f"TT_OP_{layout.mnemonic}"
            )
            expected.append((layout, dict(zip(layout.parameters, values))))
    assert len(tables.opcodes) > 100
    for word, (layout, values) in zip(expand(calls), expected):
        mnemonic, fields = instructions.decode(word, tables)
        assert mnemonic == layout.mnemonic
        assert {f["name"]: f["value"] for f in fields} == {
            k: v for k, v in values.items() if k in {f["name"] for f in fields}
        }


def test_replay_encoding(tables):
    word, mask = instructions.encode(
        tables,
        "REPLAY",
        {"start_idx": 4, "len": 3, "execute_while_loading": 0, "load_mode": 1},
    )
    assert (word, mask) == (0x04010031, 0)
    word, mask = instructions.encode(
        tables,
        "REPLAY",
        {"start_idx": 0, "len": None, "execute_while_loading": 0, "load_mode": 1},
    )
    mnemonic, fields = instructions.decode(word, tables, mask)
    assert mnemonic == "REPLAY"
    assert [f["name"] for f in fields if f["runtime"]] == ["len"]


@pytest.mark.parametrize(
    "address, name",
    [
        (0xFFEF0000 + 76 * 4, "cfg state 0: THCON_SEC0_REG3_Base_address"),
        (0xFFEF0000 + (224 + 76) * 4, "cfg state 1: THCON_SEC0_REG3_Base_address"),
        (0xFFB80000 + 0x14, "TENSIX_MOP_CFG_BASE + 0x14"),
        (0xFFE80000 + 40, "PC_BUF_BASE + 0x28"),
    ],
)
def test_address_names(tables, address, name):
    assert instructions.address_name(address, tables).startswith(name)


def test_thread_registers_are_named_from_the_thread_section(tables):
    first_thread_word = 0xFFEF0000 + 3 * tables.state_words * 4
    assert instructions.address_name(first_thread_word, tables).startswith(
        "cfg thread 0: CFG_STATE_ID"
    )


def test_digit_separators_in_bases():
    regions = instructions.parse_regions("#define RISC_PIC_BASE 0xFFB1'3000\n")
    assert regions == [(0xFFB13000, "RISC_PIC_BASE")]
