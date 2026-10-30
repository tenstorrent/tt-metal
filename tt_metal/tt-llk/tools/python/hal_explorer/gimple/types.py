# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""C types of GIMPLE values on RISC-V, and the runtime value forms the walkers use."""

import re
from dataclasses import dataclass

POINTER_BITS = 32


@dataclass(frozen=True)
class TypeInfo:
    bits: int
    signed: bool
    pointer: bool = False

    def wrap(self, value):
        if isinstance(value, Pointer):
            return value
        value &= (1 << self.bits) - 1
        if self.signed and value >> (self.bits - 1):
            value -= 1 << self.bits
        return value


UNSIGNED32 = TypeInfo(32, False)
SCALAR_TYPES = {
    "bool": TypeInfo(1, False),
    "_Bool": TypeInfo(1, False),
    "char": TypeInfo(8, False),  # RISC-V char is unsigned
    "signed char": TypeInfo(8, True),
    "unsigned char": TypeInfo(8, False),
    "short int": TypeInfo(16, True),
    "short": TypeInfo(16, True),
    "short unsigned int": TypeInfo(16, False),
    "unsigned short": TypeInfo(16, False),
    "int": TypeInfo(32, True),
    "unsigned int": TypeInfo(32, False),
    "long int": TypeInfo(32, True),
    "long": TypeInfo(32, True),
    "long unsigned int": TypeInfo(32, False),
    "unsigned long": TypeInfo(32, False),
    "long long int": TypeInfo(64, True),
    "long long unsigned int": TypeInfo(64, False),
    "size_t": TypeInfo(32, False),
}


def type_info(text):
    text = re.sub(r"\b(const|volatile|struct|enum|class)\b", " ", text or "")
    text = " ".join(text.split())
    if "*" in text or "&" in text:
        return TypeInfo(POINTER_BITS, False, True)
    text = text.removeprefix("std::")
    m = re.fullmatch(r"(u?)int(8|16|32|64)_t", text)
    if m:
        return TypeInfo(int(m.group(2)), not m.group(1))
    return SCALAR_TYPES.get(text, UNSIGNED32)


@dataclass(frozen=True)
class Pointer:
    """A runtime address known only relative to a named symbol."""

    symbol: str
    offset: int = 0
