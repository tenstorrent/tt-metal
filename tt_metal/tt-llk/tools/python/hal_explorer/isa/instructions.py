# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tensix instruction layouts and MMIO address names, read from the checked-out headers.

Opcodes and field layouts come from the TT_OP_* macros in ckernel_ops.h, MMIO
region names from tensix.h and configuration register names from cfg_defines.h.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

OPCODE_SHIFT = 24
REGION_SPAN = 0x40000


@dataclass(frozen=True)
class FieldLayout:
    name: str
    shift: int
    width: int

    @property
    def mask(self):
        return ((1 << self.width) - 1) << self.shift


@dataclass(frozen=True)
class Layout:
    mnemonic: str
    opcode: int
    fields: Tuple[FieldLayout, ...]
    parameters: Tuple[str, ...]  # macro parameter order, as builtins take them


@dataclass
class Tables:
    opcodes: Dict[int, Layout]
    by_mnemonic: Dict[str, Layout]
    regions: List[Tuple[int, str]]
    state_registers: Dict[int, str]
    thread_registers: Dict[int, str]
    state_words: int
    thread_words: int


def headers(llk_root):
    llk_root = Path(llk_root)
    metal = llk_root.parent / "hw" / "inc" / "internal" / "tt-1xx" / "blackhole"
    return {
        "ops": llk_root / "tt_llk_blackhole" / "common" / "inc" / "ckernel_ops.h",
        "tensix": metal / "tensix.h",
        "cfg": metal / "cfg_defines.h",
    }


def load_tables(llk_root):
    paths = headers(llk_root)
    opcodes = parse_opcodes(paths["ops"].read_text(encoding="utf-8"))
    regions = parse_regions(paths["tensix"].read_text(encoding="utf-8"))
    cfg = parse_cfg_registers(paths["cfg"].read_text(encoding="utf-8"))
    return Tables(opcodes, {l.mnemonic: l for l in opcodes.values()}, regions, *cfg)


def parse_opcodes(text):
    text = text.replace("\\\n", " ")
    table = {}
    for line in text.splitlines():
        m = re.match(
            r"\s*#define\s+TT_OP_(\w+)\s*(\([^)]*\))?\s+"
            r"TT_OP\(\s*(0x[0-9a-fA-F]+|\d+)\s*,(.*)$",
            line,
        )
        if not m:
            continue
        name, opcode, encoding = m.group(1), int(m.group(3), 0), m.group(4)
        shifts = re.findall(r"\(\s*\(\s*(\w+)\s*\)\s*<<\s*(\d+)\s*\)", encoding)
        fields, top = [], OPCODE_SHIFT
        for field, shift in sorted(shifts, key=lambda f: -int(f[1])):
            fields.append(FieldLayout(field, int(shift), max(top - int(shift), 0)))
            top = int(shift)
        if opcode in table:
            raise ValueError(
                f"opcode 0x{opcode:02x} defined by both {table[opcode].mnemonic} and {name}"
            )
        parameters = tuple(
            p.strip() for p in (m.group(2) or "()")[1:-1].split(",") if p.strip()
        )
        table[opcode] = Layout(name, opcode, tuple(fields), parameters)
    return table


def parse_regions(text):
    return sorted(
        (int(value.replace("'", ""), 16), name)
        for name, value in re.findall(r"#define (\w+_BASE) (0x[0-9A-Fa-f']+)", text)
    )


def parse_cfg_registers(text):
    """(state names, thread names, state words per bank, thread words) by 32-bit word."""
    state, thread, section = {}, {}, None
    for line in text.splitlines():
        m = re.match(r"// Registers for (\w+)", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"#define (\w+)_ADDR32 (\d+)", line)
        if m and not m.group(1).endswith("_CFGREG_BASE"):
            names = thread if section == "THREAD" else state
            names.setdefault(int(m.group(2)), []).append(m.group(1))
    state_size = int(re.search(r"#define CFG_STATE_SIZE (\d+)", text).group(1)) * 4
    thread_size = int(re.search(r"#define THD_STATE_SIZE (\d+)", text).group(1))
    return word_names(state), word_names(thread), state_size, thread_size


def word_names(names):
    return {
        word: fields[0] + (f" (+{len(fields) - 1} fields)" if len(fields) > 1 else "")
        for word, fields in names.items()
    }


def decode(word, tables, runtime_mask=0):
    """Mnemonic and fields of a 32-bit word; fields under runtime_mask have no value.

    The opcode is read from word even when runtime bits could reach it: a runtime
    operand wider than its field is a source bug, not a different instruction.
    """
    layout = tables.opcodes.get(word >> OPCODE_SHIFT)
    if layout is None:
        return f"op 0x{word >> OPCODE_SHIFT:02x}", []
    fields = []
    for f in layout.fields:
        runtime = bool(f.mask & runtime_mask)
        value = None if runtime else (word & f.mask) >> f.shift
        fields.append({"name": f.name, "value": value, "runtime": runtime})
    return layout.mnemonic, fields


def encode(tables, mnemonic, values):
    """(word, runtime mask) from field values by name; None marks a runtime field."""
    layout = tables.by_mnemonic[mnemonic]
    word, runtime = layout.opcode << OPCODE_SHIFT, 0
    for f in layout.fields:
        value = values.get(f.name)
        if value is None:
            runtime |= f.mask
        else:
            word |= (value << f.shift) & f.mask
    return word, runtime


def address_name(address, tables, width=32):
    """A named configuration register, a region offset or a plain hex address."""
    region = next(
        (r for r in reversed(tables.regions) if r[0] <= address < r[0] + REGION_SPAN),
        None,
    )
    if region is None:
        return f"0x{address:08x}"
    offset = address - region[0]
    if region[1] != "TENSIX_CFG_BASE":
        return f"{region[1]} + 0x{offset:x}"
    word, byte = divmod(offset, 4)
    suffix = f" byte {byte}" if byte or width < 32 else ""
    bank, index = divmod(word, tables.state_words)
    state_name = tables.state_registers.get(index, f"word {index}")
    if bank < 2:
        return f"cfg state {bank}: {state_name}{suffix}"
    if bank == 2:
        return f"cfg both states: {state_name}{suffix}"
    thread, index = divmod(word - 3 * tables.state_words, tables.thread_words)
    if thread < 3:
        thread_name = tables.thread_registers.get(index, f"word {index}")
        return f"cfg thread {thread}: {thread_name}{suffix}"
    return f"TENSIX_CFG_BASE + 0x{offset:x}"
