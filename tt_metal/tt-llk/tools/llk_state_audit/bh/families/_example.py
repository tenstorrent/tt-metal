# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Template family plugin: copy to ``bh/families/<family>.py`` and edit.

Files whose name starts with ``_`` are never loaded by ``effects_db`` (this one
is only imported by the tests). A real family module must define ``FAMILY`` and
may define any of the other names below; everything is optional except FAMILY.
See ``bh/SCHEMA.md`` for the full contract.
"""

from __future__ import annotations

import re

from ..model import Pattern, PatternMatch
from ..resources import Normalized

#: Unique family name. Every pattern id and normalizer name must start with "<FAMILY>.".
FAMILY = "example"

#: Optional kill switch (a loaded family with ENABLED = False contributes no patterns).
ENABLED = True


def _lreg_operand(m: PatternMatch) -> str | None:
    """Spec callables receive the PatternMatch and return the operand text (or None)."""
    return m.args[0] if m.args else None


#: Declarative recognizers. Kinds: call | member | ctor | regex | assign.
PATTERNS = [
    # (1) call: instruction macro; full-match on the final identifier of each call site.
    #     SFPLOADI(lreg, mod, imm16) writes LREG<lreg>; TT_OP_ forms are MOP/replay words.
    Pattern(
        id="example.sfploadi",
        kind="call",
        match=r"TT(?:I|_OP)?_SFPLOADI",
        resource=_lreg_operand,  # or the spec string "arg:0"
        normalizer="lreg",  # resources.ResourceTables.normalizers key
        access="write",
        value="fmt:mod={arg1};imm={arg2}",
        when=lambda m: len(m.args) == 3,
    ),
    # (2) call with a qualifier: lltt::replay(start, len) reads the replay buffer.
    Pattern(
        id="example.lltt_replay",
        kind="call",
        match=r"replay",
        qualifier=r"(?:::)?lltt",
        resource="literal:REPLAY",
        normalizer="replay",
        access="read",
        detail="fmt:start={arg0};len={arg1}",
    ),
    # (3) member: receiver_type is the inferred class of the object the method is called on.
    Pattern(
        id="example.unpack_mop_run",
        kind="member",
        match=r"run",
        receiver_type=r"ckernel_unpack_template",
        resource="literal:MOP",
        normalizer="mop",
        access="read",
        stop=False,  # still descend into ckernel_unpack_template::run
    ),
    # (4) regex: searched in the masked body and in every macro expansion; groups are specs.
    Pattern(
        id="example.l_reg_write",
        kind="regex",
        match=r"\bl_reg\s*\[\s*(?P<reg>(?:sfpi::)?LRegs::LReg\d+)\s*\]\s*=(?!=)",
        resource="group:reg",
        normalizer="lreg",
        access="write",
    ),
    # (5) assign: NAME[index] op= rhs / NAME op= rhs (index, op, rhs available to specs).
    Pattern(
        id="example.regfile_write",
        kind="assign",
        match=r"regfile",
        resource="index",
        normalizer="gpr",
        value="rhs",
        when=lambda m: bool(m.index),
    ),
    # (6) family-defined normalizer (registered via NORMALIZERS below).
    Pattern(
        id="example.relu_cfg",
        kind="call",
        match=r"_llk_pack_relu_config_",
        resource="arg:0",
        normalizer="example.relu",
        value="arg:0",
        stop=False,  # keep the transitive cfg_reg_rmw_tensix effects as well
    ),
]

#: Core pattern ids this family replaces with its own (see effects_db.CORE_PATTERNS).
DISABLE_CORE: tuple[str, ...] = ()

#: Extra semantic groups: (group name, regex over canonical resource names).
GROUPS = [("example.relu", r"^PACK\.RELU$")]


def _relu(expr: str | None, thread: str | None = None):
    """Normalizer: (operand text, thread) -> Normalized | list[Normalized] | str | None."""
    return Normalized("PACK.RELU", "whole", detail=expr or "")


NORMALIZERS = {"example.relu": _relu}


def extract(fn, ctx):
    """Optional imperative hook, called once per scanned function.

    ``fn`` is a ``scan.FunctionDef`` (name, key, path, layer, params, masked_body,
    calls, expansions, line_at(), condition_at(), ...); ``ctx`` is an
    ``effects_db.FamilyContext`` whose ``effects(...)`` builds Effects exactly like a
    declarative pattern. Return an iterable of ``model.Effect``.
    """
    found = []
    for match in re.finditer(r"\bTTI_SFPENCC\s*\(", fn.masked_body):
        found.extend(
            ctx.effects(
                fn,
                pattern_id="example.sfpencc",
                resource_expr="SFPU.CC",
                normalizer="literal",
                access="write",
                offset=match.start(),
            )
        )
    return found


def role(fn, default: str) -> str | None:
    """Optional role override: return one of model.ROLES, or None to keep ``default``."""
    if fn.name.endswith("_reinit"):
        return "init"
    return None


def thread(fn, default: str) -> str | None:
    """Optional thread override: return unpack|math|pack|any, or None to keep ``default``."""
    return None
