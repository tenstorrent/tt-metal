# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Check base class + the uniform Finding envelope.

Every checker is an AUGMENTOR: it emits recall *candidates* over KNOWN patterns,
never a merge verdict. That contract is encoded here structurally:

  * A Finding has a `hint` (a recall bucket) and NO verdict field. (The optional
    `safety` field is a SUB-annotation, e.g. SAFE_BY_MASKING — explicitly not a
    pass/fail and never a clearance.)
  * Every Check must declare `blind_spots` — the patterns it structurally cannot
    see — which is echoed into the output so the LLM/human knows what to hunt
    beyond the tool.
  * The output envelope carries `authority: "advisory"`.

A tool result is input to the `/*-audit` skill, not a clearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..factbase import FactBase


@dataclass
class Finding:
    file: str
    line: int
    function: str
    kind: str  # what was found (e.g. write kind, "post/get imbalance")
    hint: str  # recall bucket — NEVER a verdict
    detail: str  # one-line human explanation
    evidence: list = field(default_factory=list)  # related facts (file:line: what)
    #: optional SUB-annotation on a finding (never replaces the hint / never
    #: clears it). e.g. cfg-word attaches safety="SAFE_BY_MASKING" to a still-
    #: reported CROSS_THREAD_SHARED_WORD so the access is visible AND the
    #: race-safety is known. "" when a check doesn't annotate.
    safety: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("safety"):
            d.pop("safety", None)  # keep output clean for checks that don't annotate
        return d


class Check:
    #: short id used on the CLI (--checks=<name>)
    name: str = ""
    #: one-line description
    description: str = ""
    #: the patterns this check structurally cannot see (echoed to output)
    blind_spots: str = ""

    def run(self, fb: FactBase) -> list[Finding]:  # pragma: no cover - interface
        raise NotImplementedError

    # convenience for subclasses
    @staticmethod
    def _ev(fact: dict, what: str) -> str:
        return f'{fact["file"].split("/")[-1]}:{fact.get("line","?")}: {what}'
