# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Turn one parsed statement into what it issues, given the values known in a Scope."""

from hal_explorer.gimple.expr import parse_atom
from hal_explorer.gimple.reader import INSTRUCTION_BUFFER as INSTRUCTION_BUFFER_NAME
from hal_explorer.gimple.types import Pointer
from hal_explorer.isa.instructions import address_name, decode, encode
from hal_explorer.model import Access, Instruction

ISSUING = {"ttinsn", "store", "ttreplay", "ttbuiltin"}
ACCESSES = {"mmio_write", "mmio_read", "asm"}
# One instruction buffer per TRISC thread, 64 KiB apart.
INSTRUCTION_BUFFER_BYTES = 3 * 0x10000
HIDING = {"call", "unsupported"}


def warning(stmt, sources):
    if stmt.kind == "call":
        message = (
            f"call to {stmt.data['callee']} is not inlined; "
            "instructions it issues are not visible"
        )
    else:
        message = stmt.data["reason"]
    return {"message": message, "origin": sources.origin(stmt.location)}


def describe(stmt, scope, tables, sources):
    """Instruction or Access for an issuing or MMIO statement, else None."""
    origin = sources.origin(stmt.location)
    if stmt.kind == "ttinsn":
        mnemonic, fields = decode(stmt.data["word"], tables)
        return Instruction(
            mnemonic, stmt.data["word"], 0, None, "immediate", fields, origin
        )
    if stmt.kind == "store":
        return runtime_word(
            parse_atom(stmt.data["value"]), scope, tables, "store", origin
        )
    if stmt.kind == "ttreplay":
        return replay(stmt, scope, tables, origin)
    if stmt.kind == "ttbuiltin":
        return builtin(stmt, scope, tables, origin)
    if stmt.kind in ACCESSES:
        return access(stmt, scope, tables, origin)
    return None


def runtime_word(node, scope, tables, issue, origin):
    value = scope.value(node)
    if isinstance(value, int):
        word = value & 0xFFFFFFFF
        mnemonic, fields = decode(word, tables)
        return Instruction(mnemonic, word, 0, None, issue, fields, origin)
    mask = scope.runtime_mask(node)
    word = scope.constant_part(node)
    mnemonic, fields = decode(word, tables, mask)
    return Instruction(mnemonic, word, mask, scope.text(node), issue, fields, origin)


def replay(stmt, scope, tables, origin):
    data = stmt.data
    values = {
        "start_idx": scope.value(data["start"]),
        "len": scope.value(data["count"]),
        "execute_while_loading": scope.value(data["exec"]),
        "load_mode": scope.value(data["load"]),
    }
    # The builtin's dynamic flag numbers each runtime REPLAY; nonzero means a store.
    dynamic = scope.value(data["dynamic"])
    issue = "immediate" if dynamic == 0 else "store"
    if issue == "store":
        # The compiler builds the runtime part of the word itself, masking the count.
        word, mask = encode(tables, "REPLAY", {**values, "len": 0})
        part = parse_atom(data["word"])
        runtime = scope.value(part)
        if isinstance(runtime, int):
            word |= runtime & 0xFFFFFFFF
            values["len"] = (runtime >> 4) & 0x3F
        else:
            word |= scope.constant_part(part)
            mask = scope.runtime_mask(part)
    else:
        word, mask = encode(tables, "REPLAY", values)
    mnemonic, fields = decode(word, tables, mask)
    length = values["len"]
    runtime_expr = None if length is not None else scope.text(data["count"])
    info = {
        "load": values["load_mode"] == 1,
        "start": values["start_idx"],
        # REPLAY encodes a count of 64 as 0.
        "count": (length & 63 or 64) if length is not None else None,
        "count_text": runtime_expr,
    }
    return Instruction(mnemonic, word, mask, runtime_expr, issue, fields, origin, info)


def builtin(stmt, scope, tables, origin):
    name = stmt.data["name"].upper()
    layout = tables.by_mnemonic.get(name[2:]) or tables.by_mnemonic.get(name)
    args = [scope.value(a) for a in stmt.data["args"]]
    if layout is None or len(args) < len(layout.parameters):
        text = f"{stmt.data['name']}({', '.join(stmt.data['args'])})"
        return Instruction(name, 0, 0xFFFFFFFF, text, "builtin", [], origin)
    values = dict(zip(layout.parameters, args[len(args) - len(layout.parameters) :]))
    word, mask = encode(tables, layout.mnemonic, values)
    mnemonic, fields = decode(word, tables, mask)
    text = ", ".join(stmt.data["args"]) if mask else None
    return Instruction(mnemonic, word, mask, text, "builtin", fields, origin)


def in_instruction_buffer(address, tables):
    """True for an address (int or Pointer) inside a thread's instruction buffer."""
    if isinstance(address, Pointer):
        return address.symbol.lstrip("_") == "instrn_buffer"
    base = next((v for v, name in tables.regions if name == "INSTRN_BUF_BASE"), None)
    return base is not None and base <= address < base + INSTRUCTION_BUFFER_BYTES


def asm(stmt, scope, tables, origin):
    """A blocking store (store_blocking: sw %0, (%1)) into the buffer issues an instruction."""
    operands = stmt.data["operands"]
    expressions = [expression for _, expression in operands]
    if stmt.data["template"].startswith("sw %0, (%1)") and len(operands) >= 2:
        address = scope.value(expressions[1])
        named = INSTRUCTION_BUFFER_NAME.search(expressions[1])
        if named or (address is not None and in_instruction_buffer(address, tables)):
            # The stored value is the input tied to output %0.
            value = next((e for c, e in operands if c == "0"), expressions[0])
            return runtime_word(parse_atom(value), scope, tables, "store", origin)
    targets = ", ".join(scope.text(expression) for expression in expressions)
    return Access("asm", 32, None, targets, None, stmt.data["template"], origin)


def access(stmt, scope, tables, origin):
    if stmt.kind == "asm":
        return asm(stmt, scope, tables, origin)
    address = stmt.data["address"]
    width = address["width"]
    location, target = None, address["text"]
    if address["base"] is not None:
        base = scope.value(address["base"])
        offset = scope.value(address["offset"])
        if isinstance(base, int) and isinstance(offset, int):
            location = (base + offset) & 0xFFFFFFFF
            target = address_name(location, tables, width)
        elif isinstance(base, Pointer) and isinstance(offset, int):
            index, rest = divmod(base.offset + offset, max(width // 8, 1))
            target = f"{base.symbol}[{index}]" + (f" + {rest}" if rest else "")
        else:
            offset_text = scope.text(address["offset"])
            target = scope.text(address["base"]) + (
                "" if offset_text == "0" else f" + {offset_text}"
            )
    if stmt.kind == "mmio_read":
        return Access("read", width, location, target, None, "", origin)
    base = scope.value(address["base"]) if address["base"] is not None else None
    if (location is not None and in_instruction_buffer(location, tables)) or (
        isinstance(base, Pointer) and in_instruction_buffer(base, tables)
    ):
        return runtime_word(
            parse_atom(stmt.data["value"]), scope, tables, "store", origin
        )
    value = scope.value(stmt.data["value"])
    value_text = (
        hex(value) if isinstance(value, int) else scope.text(stmt.data["value"])
    )
    return Access(
        "write",
        width,
        location,
        target,
        value if isinstance(value, int) else None,
        value_text,
        origin,
    )
