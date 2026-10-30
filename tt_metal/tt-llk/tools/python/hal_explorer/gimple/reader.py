# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parser for GCC's -fdump-tree-optimized-lineno-blocks output.

Each function becomes a control-flow graph of basic blocks. Successors come from
the dump's explicit ";; succ:" lists, so no fall-through is inferred. Statements
that matter for the Tensix are classified: instruction words (.ttinsn asm,
volatile stores to the instruction buffer, REPLAY and other tt builtins), MMIO
accesses (any other volatile load or store, blocking-store asm), and anything that
would hide issued instructions (un-inlined calls, SFPI builtins).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hal_explorer.gimple.types import type_info

# "[file:line:col]", "[file:line:col discrim N]" or the location-less "[0:0]".
LOCATION = re.compile(
    r"\[(?:([^\[\]:]+(?::[^\[\]:]+)*?):(\d+):(\d+)(?: discrim \d+)?|0:0)\]\s*"
)
SSA = r"[A-Za-z_$.][\w$.]*(?:\(D\))?"
BLOCK_HEADER = re.compile(r";;\s+basic block (\d+), loop depth (\d+)")
SUCC = re.compile(r";;\s+succ:\s*(.*)$")
SUCC_MORE = re.compile(r";;\s+(\d+|EXIT)\s*$")
LABEL = re.compile(r"^<(L\d+)>:$")
CASE = re.compile(r"(default|case (-?\d+)(?: \.\.\. (-?\d+))?): <(L\d+)>")
VOLATILE_MEM = re.compile(
    r"^MEM(?: <[^>]*>)?\s*\[\((?P<type>[^)]*\*)\)(?P<base>[^\s\]]+)(?: \+ (?P<offset>[^\]\s]+))?\]$"
)
ANNOTATION = re.compile(r"\s*\[(?:tail call|must tail call|return slot optimization)\]")
MAY_ISSUE = re.compile(r"__builtin_rvtt|=\{v\}|__asm__|instrn_buffer")
INSTRUCTION_BUFFER = re.compile(r"\b_?_?instrn_buffer\b")
IGNORED_CALLS = {"__builtin_unreachable", "__builtin_trap", "__builtin_expect"}


@dataclass
class Statement:
    kind: str
    location: Optional[Tuple[str, int, int]]
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class Block:
    index: int
    loop_depth: int
    statements: List[Statement] = field(default_factory=list)
    phis: List[Tuple[str, Dict[int, str]]] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    exits: bool = False
    terminator: Optional[tuple] = None


@dataclass
class Function:
    name: str
    mangled: str
    return_type: str
    parameters: List[Tuple[str, str]]
    declarations: Dict[str, str]
    blocks: Dict[int, Block]
    entry: Optional[int]
    text: str
    files: set

    @property
    def signature(self):
        return f"{self.name}({', '.join(t for t, _ in self.parameters)})"


def strip_locations(line):
    """Statement text without location tags, and the first real location."""
    location = None
    for m in LOCATION.finditer(line):
        if m.group(1):
            location = (m.group(1), int(m.group(2)), int(m.group(3)))
            break
    return LOCATION.sub("", line).strip(), location


def balanced(text):
    """True when quotes, parentheses and angle-bracketed operands are closed."""
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
    return not quoted and depth <= 0


def split_functions(dump):
    """(name, mangled, lines) for every function in the dump."""
    chunks = []
    for chunk in dump.split(";; Function ")[1:]:
        header, _, rest = chunk.partition("\n")
        m = re.match(r"(.+?) \(([^,\s]+)", header)
        if m:
            chunks.append((m.group(1), m.group(2), rest.splitlines()))
    return chunks


def parse_signature(line):
    """('void', [('unsigned int', 'x'), ...]) from 'void f (unsigned int x, bool a)'."""
    m = re.match(r"^(.*?)\s*\b[^\s(]+ \((.*)\)$", line.strip())
    if not m:
        return "", []
    parameters = []
    for parameter in split_top_level(m.group(2)):
        parameter = parameter.strip()
        if not parameter or parameter == "void":
            continue
        pm = re.match(r"^(.*?[\s*&])(\w+)$", parameter)
        parameters.append((pm.group(1).strip(), pm.group(2)) if pm else (parameter, ""))
    return m.group(1), parameters


def split_top_level(text, separator=","):
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == separator and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return parts + ([current] if current.strip() else [])


def parse_functions(dump):
    functions = []
    for name, mangled, lines in split_functions(dump):
        if "{" not in lines:
            continue
        start = lines.index("{")
        end = len(lines) - 1 - lines[::-1].index("}") if "}" in lines else len(lines)
        return_type, parameters = (
            parse_signature(lines[start - 1]) if start else ("", [])
        )
        text = "\n".join(line for line in lines[start - 1 : end + 1] if line.strip())
        functions.append(
            parse_body(
                name, mangled, return_type, parameters, lines[start + 1 : end], text
            )
        )
    return functions


def parse_body(name, mangled, return_type, parameters, lines, text):
    declarations, blocks, labels, files = {}, {}, {}, set()
    block = None
    pending = ""
    in_succ = False
    for raw in lines:
        m = BLOCK_HEADER.match(raw.strip())
        if m:
            block = Block(int(m.group(1)), int(m.group(2)))
            blocks[block.index] = block
            in_succ = False
            continue
        if block is None:
            decl = re.match(r"^\s*(.+?)\s+(\S+);$", raw)
            if decl and not raw.strip().startswith(";;"):
                declarations[decl.group(2)] = decl.group(1)
            continue
        stripped = raw.strip()
        m = SUCC.match(stripped)
        if m:
            in_succ = True
            add_successors(block, m.group(1))
            continue
        if in_succ and SUCC_MORE.match(stripped):
            add_successors(block, SUCC_MORE.match(stripped).group(1))
            continue
        if stripped.startswith(";;") or not stripped:
            continue
        pending = f"{pending}\n{raw}" if pending else raw
        if not balanced(pending):
            continue
        statement_text, location = strip_locations(pending)
        pending = ""
        if location:
            files.add(location[0])
        classify(block, statement_text, location, labels, declarations)

    for block in blocks.values():
        resolve_terminator(block, labels)
    entry = min(blocks) if blocks else None
    return Function(
        name, mangled, return_type, parameters, declarations, blocks, entry, text, files
    )


def add_successors(block, text):
    for token in text.split():
        if token == "EXIT":
            block.exits = True
        elif token.isdigit():
            block.successors.append(int(token))


def classify(block, stmt, location, labels, declarations):
    """Append one statement (or terminator piece) to block."""
    stmt = ANNOTATION.sub("", stmt)
    m = LABEL.match(stmt)
    if m:
        labels[m.group(1)] = block.index
        return
    m = re.match(r"^# (" + SSA + r") = PHI <(.*)>$", stmt)
    if m:
        incoming = re.findall(r"\s*([^,\s]+?)\((\d+)\)(?:,|$)", m.group(2))
        block.phis.append((m.group(1), {int(pred): arg for arg, pred in incoming}))
        return
    m = re.match(r"^if \((.*)\)$", stmt, re.S)
    if m:
        block.terminator = ("if", m.group(1), [])
        return
    m = re.match(r"^goto <bb (\d+)>;", stmt)
    if m:
        if block.terminator and block.terminator[0] == "if":
            block.terminator[2].append(int(m.group(1)))
        else:
            block.terminator = ("goto", int(m.group(1)))
        return
    if stmt == "else":
        return
    m = re.match(r"^switch \((\S+)\) <(.*)>$", stmt, re.S)
    if m:
        cases = []
        for case in CASE.finditer(m.group(2)):
            low = None if case.group(1) == "default" else int(case.group(2))
            high = low if case.group(3) is None else int(case.group(3))
            cases.append((low, high, case.group(4)))
        block.terminator = ("switch", m.group(1), cases)
        return
    m = re.match(r"^return(?: (.+))?;$", stmt)
    if m:
        block.terminator = ("return",)
        return

    add = lambda kind, **data: block.statements.append(
        Statement(kind, location, stmt, data)
    )

    m = re.search(r'\.ttinsn %0"\s*:\s*:\s*"n"\s*(-?\d+)', stmt)
    if m:
        add("ttinsn", word=int(m.group(1)) & 0xFFFFFFFF)
        return
    m = re.match(r"^(?:(" + SSA + r") = )?__builtin_rvtt_ttreplay \((.*)\);$", stmt)
    if m:
        args = [a.strip() for a in split_top_level(m.group(2))]
        add(
            "ttreplay",
            count=args[1],
            word=args[2],
            dynamic=args[3],
            start=args[4],
            exec=args[5],
            load=args[6],
        )
        return
    m = re.match(r"^(?:(" + SSA + r") = )?(__builtin_rvtt_(\w+)) \((.*)\);$", stmt)
    if m:
        name, short = m.group(2), m.group(3)
        args = [a.strip() for a in split_top_level(m.group(4))]
        if short == "synth_opcode":
            # The opcode template a dynamic builtin adds its runtime operands to; the
            # explorer encodes the constant fields itself, so the template counts as zero.
            if m.group(1):
                declarations.setdefault(m.group(1), "unsigned int")
                definition = {"name": m.group(1), "rhs": "0"}
                block.statements.append(Statement("def", location, stmt, definition))
            return
        if short.startswith("tt"):
            add("ttbuiltin", name=short, args=args)
        else:
            add(
                "unsupported",
                reason=f"{name} (SFPI or other non-volatile builtin) is not counted",
            )
        return

    # A store or load of a volatile location: "LHS ={v} RHS;".
    m = re.match(r"^(.+?) =\{v\} (.+);$", stmt, re.S)
    if m:
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        if re.fullmatch(SSA, lhs):
            add("mmio_read", target=lhs, address=parse_address(rhs, declarations))
        elif INSTRUCTION_BUFFER.search(lhs):
            add("store", value=rhs)
        else:
            add("mmio_write", address=parse_address(lhs, declarations), value=rhs)
        return

    # An empty asm with a tied operand only hides the value from the optimizer.
    m = re.match(r'^__asm__\(""\s*:\s*"=r"\s*(' + SSA + r')\s*:\s*"0"\s*(.+)\);$', stmt)
    if m:
        block.statements.append(
            Statement(
                "def", location, stmt, {"name": m.group(1), "rhs": m.group(2).strip()}
            )
        )
        return
    m = re.match(r'^__asm__(?: __volatile__)?\("(.*?)"\s*(?::(.*))?\);$', stmt, re.S)
    if m:
        if m.group(1).strip():
            template = "; ".join(t.strip() for t in m.group(1).split("\n") if t.strip())
            sections = (m.group(2) or "").split(":")
            outputs = asm_operands(sections[0] if sections else "")
            inputs = asm_operands(sections[1] if len(sections) > 1 else "")
            for _, output in outputs:
                definition = {"name": output, "rhs": "OPAQUE"}
                block.statements.append(Statement("def", location, stmt, definition))
            # Numbered as the template refers to them: outputs first, then inputs.
            add("asm", template=template, operands=outputs + inputs)
        return

    m = re.match(r"^(?:(" + SSA + r") = )?([\w:~<>,.$ ]+?) \((.*)\);$", stmt, re.S)
    if (
        m
        and not m.group(2).startswith(("MEM", "__asm__"))
        and m.group(2).strip() not in IGNORED_CALLS
    ):
        if m.group(1):
            block.statements.append(
                Statement("def", location, stmt, {"name": m.group(1), "rhs": "OPAQUE"})
            )
        add("call", callee=m.group(2).strip())
        return
    m = re.match(r"^(" + SSA + r") = (.+);$", stmt, re.S)
    if m:
        block.statements.append(
            Statement(
                "def", location, stmt, {"name": m.group(1), "rhs": m.group(2).strip()}
            )
        )
        return
    if MAY_ISSUE.search(stmt):
        add(
            "unsupported",
            reason=f"unrecognised statement, not counted: {' '.join(stmt.split())[:160]}",
        )
    # Remaining forms (non-volatile memory stores, clobbers, ignored builtins) issue nothing.


def asm_operands(section):
    """[(constraint, expression)] of one asm operand list, e.g. '"ptr" "r" _6, "0" 0'."""
    operands = []
    for item in split_top_level(section):
        m = re.match(r'\s*(?:"\w+"\s+)?"([^"]*)"\s+(.+?)\s*$', item, re.S)
        if m:
            operands.append((m.group(1), m.group(2)))
    return operands


def parse_address(text, declarations):
    """{'base', 'offset', 'width', 'text'} for a volatile memory reference."""
    m = VOLATILE_MEM.match(text)
    if m:
        return {
            "base": m.group("base"),
            "offset": (m.group("offset") or "0").rstrip("B"),
            "width": pointee_width(m.group("type")),
            "text": text,
        }
    m = re.match(r"^\*(" + SSA + r")$", text)
    if m:
        return {
            "base": m.group(1),
            "offset": "0",
            "width": pointee_width(
                declarations.get(m.group(1), "")
                or declarations.get(base_name(m.group(1)), "")
            ),
            "text": text,
        }
    return {"base": None, "offset": "0", "width": 32, "text": text}


def pointee_width(pointer_type):
    return (
        type_info(pointer_type.replace("*", "", 1).strip()).bits if pointer_type else 32
    )


def base_name(ssa):
    """The variable an SSA name versions: x_3(D) -> x, cfg_state_id.0_9 -> cfg_state_id."""
    m = re.match(r"^(.+?)(?:_\d+\(D\)|\.\d+_\d+|_\d+)$", ssa)
    return m.group(1) if m else ssa


def resolve_terminator(block, labels):
    """Turn the parsed terminator into explicit successor edges."""
    term = block.terminator
    if term and term[0] == "if":
        targets = term[2]
        if len(targets) != 2:
            raise ValueError(
                f"bb {block.index}: conditional with {len(targets)} targets"
            )
        block.terminator = ("if", term[1], targets[0], targets[1])
    elif term and term[0] == "switch":
        cases = [(low, high, labels[label]) for low, high, label in term[2]]
        block.terminator = ("switch", term[1], cases)
    elif term and term[0] == "return":
        block.terminator = ("return",)
    elif block.exits and not block.successors:
        block.terminator = ("return",)
    elif not block.successors:
        block.terminator = ("noreturn",)
    elif len(block.successors) == 1:
        block.terminator = ("goto", block.successors[0])
    elif term is None:
        raise ValueError(
            f"bb {block.index}: {len(block.successors)} successors but no branch"
        )


def classify_line(line):
    """'insn', 'mmio' or '' for one line of the dump, used to mark it in the page."""
    if re.search(r"\.ttinsn %0|__builtin_rvtt_tt\w* \(", line):
        return "insn"
    if "={v}" in line:
        return "insn" if INSTRUCTION_BUFFER.search(line.split("={v}")[0]) else "mmio"
    if re.search(r'__asm__ __volatile__\("(?!\.ttinsn)[^"]', line):
        return "mmio"
    if re.search(r"__builtin_rvtt_(?!tt|synth)\w+ \(", line):
        return "unsupported"
    return ""
