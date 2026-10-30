# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ground truth from the object file: run a function's RISC-V code and record its Tensix words.

Independent of GIMPLE: compiles with the tool's flags, disassembles with objdump and
executes the instructions. A tt* instruction issues its word (objdump shows it
rotated right by 2); a store into the instruction buffer issues the stored value.
Loads of anything else read zero, so MMIO reads and globals behave as zero.
"""

import os
import re
import subprocess
from collections import defaultdict

from hal_explorer.toolchain import build

M32 = 0xFFFFFFFF
INSTRUCTION_BUFFER = (0xFFE40000, 0xFFE40000 + 3 * 0x10000)
SYMBOLS = {"__instrn_buffer": 0xFFE40000}
LINE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)\s*(.*)$")
RELOCATION = re.compile(r"^\s+([0-9a-f]+):\s+(R_RISCV_\w+)\s+(\S+)")
LABEL = re.compile(r"^([0-9a-f]+) <(.+)>:$")


class EmulationError(Exception):
    pass


def objdump_path():
    return os.path.join(
        os.path.dirname(build.compile_flags()[0]), "riscv-tt-elf-objdump"
    )


def compile_object(source, workdir):
    """Object file for already-wrapped source, compiled with the tool's flags."""
    path = os.path.join(workdir, "oracle.cpp")
    obj = os.path.join(workdir, "oracle.o")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    subprocess.run(
        build.compile_flags() + ["-c", path, "-o", obj],
        check=True,
        capture_output=True,
        cwd=workdir,
    )
    return obj


def functions(obj):
    """{symbol: [instruction dict]} from objdump -dr."""
    text = subprocess.run(
        [objdump_path(), "-dr", "-w", obj], capture_output=True, text=True, check=True
    ).stdout
    result, current, last = {}, None, None
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            if not m.group(2).startswith(".L"):
                current = result.setdefault(m.group(2), [])
            continue
        if current is None:
            continue
        m = LINE.match(line)
        if m:
            rest = re.split(r"\t[0-9a-f]+: R_RISCV", m.group(4))[0]
            inline = re.findall(r"[0-9a-f]+: (R_RISCV_\w+)\t(\S+)", m.group(4))
            last = {
                "address": int(m.group(1), 16),
                "raw": int(m.group(2), 16),
                "mnemonic": m.group(3),
                "operands": rest.split("#")[0].strip(),
                "relocations": [r for r in inline if r[0] != "R_RISCV_RELAX"],
            }
            current.append(last)
            continue
        m = RELOCATION.match(line)
        if m and last is not None and m.group(2) != "R_RISCV_RELAX":
            last["relocations"].append((m.group(2), m.group(3)))
    return result


def immediate(token):
    try:
        return int(token.strip(), 0)
    except ValueError:
        return None


def signed(value):
    value &= M32
    return value - (1 << 32) if value & 0x80000000 else value


def symbol_address(name):
    name = re.match(r"[^+-]+", name).group(0)
    if name not in SYMBOLS:
        SYMBOLS[name] = 0x20000000 + 0x10000 * len(SYMBOLS)
    return SYMBOLS[name]


def branch_target(operands):
    m = re.search(r"\b([0-9a-f]+) <", operands)
    return int(m.group(1), 16) if m else None


ALU = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "and": lambda a, b: a & b,
    "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b, "sll": lambda a, b: a << (b & 31),
    "srl": lambda a, b: a >> (b & 31), "sra": lambda a, b: signed(a) >> (b & 31),
    "slt": lambda a, b: int(signed(a) < signed(b)), "sltu": lambda a, b: int(a < b),
    "mul": lambda a, b: a * b, "mulhu": lambda a, b: (a * b) >> 32,
    "divu": lambda a, b: a // b if b else M32, "remu": lambda a, b: a % b if b else a,
    "andn": lambda a, b: a & ~b, "orn": lambda a, b: a | ~b, "xnor": lambda a, b: ~(a ^ b),
    "min": lambda a, b: min(signed(a), signed(b)), "max": lambda a, b: max(signed(a), signed(b)),
    "minu": min, "maxu": max,
    "rol": lambda a, b: (a << (b & 31)) | (a >> (32 - (b & 31))),
    "ror": lambda a, b: (a >> (b & 31)) | (a << (32 - (b & 31))),
    "sh1add": lambda a, b: (a << 1) + b, "sh2add": lambda a, b: (a << 2) + b,
    "sh3add": lambda a, b: (a << 3) + b,
}  # fmt: skip
ALU_IMMEDIATE = {
    "addi": "add", "andi": "and", "ori": "or", "xori": "xor", "slli": "sll", "srli": "srl",
    "srai": "sra", "slti": "slt", "sltiu": "sltu", "rori": "ror",
}  # fmt: skip
UNARY = {
    "seqz": lambda a: int(a == 0), "snez": lambda a: int(a != 0), "sltz": lambda a: int(signed(a) < 0),
    "sgtz": lambda a: int(signed(a) > 0), "neg": lambda a: -a, "not": lambda a: ~a,
    "zext.b": lambda a: a & 0xFF, "zext.h": lambda a: a & 0xFFFF,
    "sext.b": lambda a: signed((a & 0xFF) << 24) >> 24, "sext.h": lambda a: signed((a & 0xFFFF) << 16) >> 16,
    "mv": lambda a: a,
}  # fmt: skip
BRANCH = {
    "beq": lambda a, b: a == b, "bne": lambda a, b: a != b, "blt": lambda a, b: signed(a) < signed(b),
    "bge": lambda a, b: signed(a) >= signed(b), "bltu": lambda a, b: a < b, "bgeu": lambda a, b: a >= b,
    "ble": lambda a, b: signed(a) <= signed(b), "bgt": lambda a, b: signed(a) > signed(b),
    "bleu": lambda a, b: a <= b, "bgtu": lambda a, b: a > b,
}  # fmt: skip
BRANCH_ZERO = {
    "beqz": "beq",
    "bnez": "bne",
    "blez": "ble",
    "bgez": "bge",
    "bltz": "blt",
    "bgtz": "bgt",
}
TRAPS = re.compile(r"(abort|_exit|exit|__assert_func|__stack_chk_fail)\b")


def run(instructions, arguments, max_steps=200000):
    """Tensix words issued by one call with integer arguments in a0, a1, ..."""
    index = {insn["address"]: k for k, insn in enumerate(instructions)}
    registers = defaultdict(int)
    for k, value in enumerate(arguments):
        registers[f"a{k}"] = value & M32
    memory = defaultdict(int)
    words = []

    def read(register):
        return 0 if register == "zero" else registers[register] & M32

    def write(register, value):
        if register != "zero":
            registers[register] = value & M32

    pc, steps = 0, 0
    while pc < len(instructions):
        steps += 1
        if steps > max_steps:
            raise EmulationError("step limit")
        insn = instructions[pc]
        mnemonic, text = insn["mnemonic"], insn["operands"]
        ops = [o.strip() for o in re.split(r",(?![^()]*\))", text)] if text else []
        relocated = bool(insn["relocations"])
        next_pc = pc + 1
        if mnemonic.startswith(("tt", "sfp")) or mnemonic in (".insn", ".word"):
            raw = insn["raw"]
            words.append(((raw >> 2) | (raw << 30)) & M32)
        elif mnemonic == "lui":
            write(
                ops[0],
                (
                    symbol_address(insn["relocations"][0][1])
                    if relocated
                    else immediate(ops[1]) << 12
                ),
            )
        elif mnemonic == "auipc":
            if relocated and insn["relocations"][0][0].startswith("R_RISCV_CALL"):
                if TRAPS.match(insn["relocations"][0][1]):
                    return words
                raise EmulationError(f"call {insn['relocations'][0][1]}")
            write(ops[0], symbol_address(insn["relocations"][0][1]) if relocated else 0)
        elif mnemonic == "li":
            write(ops[0], immediate(ops[1]))
        elif mnemonic in ALU:
            write(ops[0], ALU[mnemonic](read(ops[1]), read(ops[2])))
        elif mnemonic in ALU_IMMEDIATE:
            value = 0 if relocated else immediate(ops[2])
            if mnemonic == "addi" and relocated:
                write(ops[0], read(ops[1]))
            else:
                write(
                    ops[0],
                    ALU[ALU_IMMEDIATE[mnemonic]](
                        read(ops[1]), value & M32 if mnemonic == "sltiu" else value
                    ),
                )
        elif mnemonic in UNARY:
            write(ops[0], UNARY[mnemonic](read(ops[1])))
        elif mnemonic in BRANCH or mnemonic in BRANCH_ZERO:
            a = read(ops[0])
            b = 0 if mnemonic in BRANCH_ZERO else read(ops[1])
            if BRANCH[BRANCH_ZERO.get(mnemonic, mnemonic)](a, b):
                next_pc = index[branch_target(text)]
        elif mnemonic == "j":
            next_pc = index[branch_target(text)]
        elif mnemonic == "ret" or (mnemonic == "jr" and ops == ["ra"]):
            return words
        elif mnemonic == "jr":
            raise EmulationError("indirect jump (jump table)")
        elif mnemonic in ("jal", "jalr", "call", "tail"):
            target = insn["relocations"][0][1] if relocated else "?"
            if TRAPS.match(target):
                return words
            raise EmulationError(f"call {target}")
        elif mnemonic in ("lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb"):
            m = re.fullmatch(r"(\S+?),\s*(-?\w+)?\((\w+)\)", text.replace(" ", ""))
            address = (
                read(m.group(3)) + (0 if relocated else immediate(m.group(2) or "0"))
            ) & M32
            if mnemonic.startswith("l"):
                write(m.group(1), memory[address & ~3])
            elif INSTRUCTION_BUFFER[0] <= address < INSTRUCTION_BUFFER[1]:
                words.append(read(m.group(1)))
            else:
                memory[address & ~3] = read(m.group(1))
        elif mnemonic == "ebreak":
            return words
        elif mnemonic in ("nop", "fence", "fence.i", "ecall") or mnemonic.startswith(
            "csr"
        ):
            if mnemonic.startswith("csrr") and ops:
                write(ops[0], 0)
        else:
            raise EmulationError(f"unsupported instruction {mnemonic} {text}")
        pc = next_pc
    raise EmulationError("ran past the end of the function")
