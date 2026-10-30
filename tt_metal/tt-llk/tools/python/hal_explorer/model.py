# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Result types. to_json() gives the documented JSON returned by the CLI and the server."""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Union

from hal_explorer.isa.origin import Origin


@dataclass
class Instruction:
    """One Tensix instruction issued by the RISC-V core."""

    mnemonic: str
    word: int  # constant bits; runtime bits read as zero
    runtime_mask: int  # bits of word that depend on runtime values
    runtime_expr: Optional[str]
    issue: str  # "immediate" (.ttinsn), "store" (__instrn_buffer) or "builtin"
    fields: List[dict]
    origin: Optional[Origin]
    replay: Optional[dict] = None  # {"load", "start", "count"} for REPLAY
    kind: str = "insn"

    def key(self):
        # How the word reached the Tensix (immediate or store) does not change what it receives.
        runtime = self.runtime_expr if self.runtime_mask else None
        return ("insn", self.word, self.runtime_mask, runtime)

    def line(self):
        runtime = f" + rt({self.runtime_expr})" if self.runtime_mask else ""
        return f"{self.mnemonic} 0x{self.word:08x}{runtime}"


@dataclass
class Access:
    """An MMIO access or opaque volatile asm: issues no Tensix instruction."""

    operation: str  # "write", "read" or "asm"
    width: int
    address: Optional[int]
    target: str
    value: Optional[int]
    value_text: str
    origin: Optional[Origin]
    kind: str = "mmio"

    def key(self):
        address = self.address if self.address is not None else self.target
        value = self.value if self.value is not None else self.value_text
        return ("mmio", self.operation, self.width, address, value)

    def line(self):
        return f"MMIO {self.operation} {self.width}b {self.target} = {self.value_text}"


@dataclass
class Loop:
    """A loop whose trip count is only known at runtime."""

    repeat_while: str
    body: List["Item"]
    # When the body differs between iterations: [{"when": [[conditions]], "body": [items]}].
    variants: List[dict] = field(default_factory=list)
    kind: str = "loop"

    def key(self):
        bodies = [self.body] + [v["body"] for v in self.variants]
        return (
            "loop",
            self.repeat_while,
            tuple(tuple(i.key() for i in b) for b in bodies),
        )


Item = Union[Instruction, Access, Loop]


@dataclass
class Group:
    """One distinct issued sequence and every path condition that produces it."""

    fingerprint: str
    sequence: List[Item]
    count: Optional[int]  # None when the count depends on a runtime loop
    count_text: str
    when: List[List[str]]


@dataclass
class Sample:
    """One concrete run: input values and what they issue."""

    inputs: dict
    count: int
    fingerprint: Optional[str]
    complete: bool  # False when a step or branch limit stopped the run
    note: str = ""  # e.g. that MMIO-dependent loops were assumed to end


@dataclass
class Function:
    name: str
    signature: str
    mangled: str
    path_count: int
    truncated: bool
    groups: List[Group]
    samples: List[Sample]
    count_formula: Optional[str]
    warnings: List[dict]  # {"message", "origin"}: reasons the count may be incomplete
    gimple: List[dict]  # {"text", "mark"} per dump line


@dataclass
class Comparison:
    name: str
    reference: str
    verdict: str  # "identical", "differs", "inconclusive" or "unpaired"
    samples: int
    reasons: List[str]
    counterexample: Optional[dict] = None  # {"inputs", "diff"}


@dataclass
class Result:
    ok: bool
    mode: str
    diagnostics: str
    errors: List[dict] = field(
        default_factory=list
    )  # {"file", "line", "column", "message"}
    functions: List[Function] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)

    @property
    def incomplete(self):
        return any(f.warnings for f in self.functions) or any(
            c.verdict == "inconclusive" for c in self.comparisons
        )


def to_json(value):
    """Plain JSON data; instruction words are also given as hex text."""
    if dataclasses.is_dataclass(value):
        data = {
            f.name: to_json(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
        if isinstance(value, Instruction):
            data["word_text"] = f"0x{value.word:08x}" + (
                " + rt" if value.runtime_mask else ""
            )
        return data
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    return value
