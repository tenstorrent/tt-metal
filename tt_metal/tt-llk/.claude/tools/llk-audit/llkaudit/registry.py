# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Declarative pattern registry — the single place that maps LLK *names/signatures*
to their *meaning* for the race-audit checkers.

============================  EDIT HERE WHEN SIGNATURES CHANGE  ================
The C++ extractor is deliberately semantics-free: it reports raw names (the
function that produced a pointer, the macro that was expanded, the callee of a
call). ALL knowledge of "get_cfg_pointer() yields a CONFIG pointer" or
"TTI_UNPACR is a consumer" lives in the tables below. So when an LLK function is
renamed, a macro is added, or a new cfg-pointer accessor appears, you update one
table here — never the C++ and rarely a checker.
================================================================================

Everything is matched case-insensitively unless noted. Where a rule is a
substring test it is documented as such (macros come in families like
TTI_UNPACR / TTI_UNPACR_NOP / TT_UNPACR_VALID, so substring matching is the
robust choice for them).
"""

from __future__ import annotations

import re

# --- Thread attribution -------------------------------------------------------
# A CONFIG/GPR write's owning Tensix thread is inferred from the file it lives in
# (the LLK convention). Order matters: first substring hit wins.
THREAD_BY_FILE = [
    ("cunpack", "UNPACK"),
    ("llk_unpack", "UNPACK"),
    ("unpack", "UNPACK"),
    ("cmath", "MATH"),
    ("llk_math", "MATH"),
    ("math", "MATH"),
    ("cpack", "PACK"),
    ("llk_pack", "PACK"),
    ("pack", "PACK"),
    # SFPU (the vector unit) runs on the MATH thread; its files carry no
    # unpack/math/pack token, so without this they resolve to UNKNOWN and their
    # (real, MATH-side) config writes drop out of the cross-thread analysis.
    # Checked last so a "math_..._sfpu" file still hits MATH via "math" first.
    ("sfpu", "MATH"),
]


def thread_of(path: str) -> str:
    p = path.lower()
    for sub, thr in THREAD_BY_FILE:
        if sub in p:
            return thr
    return "UNKNOWN"


def thread_of_fact(fact: dict) -> str:
    """Thread for a write fact: the FILE's thread, else (token-less file) fall back
    to the FUNCTION name. Quasar common/lib headers (ckernel_trisc_common.h,
    llk_srcs.h, ...) carry no unpack/math/pack token so their config writers resolve
    UNKNOWN by file alone and drop out of the cross-thread / intra-thread-clobber
    analysis — but the WRITING FUNCTION usually does carry it (_set_packer_dest_
    registers_<PACK1>, _llk_pack_srcs_config_). WH/BH files resolve by path, so the
    fallback only ever fires for a token-less file and never changes their result."""
    t = thread_of(fact.get("file", ""))
    return t if t != "UNKNOWN" else thread_of(fact.get("function", "") or "")


# --- CONFIG/GPR MMIO write provenance ----------------------------------------
# A pointer_write fact carries provenance (how its base pointer was produced).
# Map the producer name -> the register space written. These are the UNSAFE
# (RISC-MMIO) write producers.
CFG_POINTER_PRODUCERS = {
    "get_cfg_pointer": "cfg32",
    "get_cfg16_pointer": "cfg16",
    "get_regfile_pointer": "regfile_gpr",
}
# MMIO writes expressed as a direct call (callee name -> write kind).
MMIO_WRITE_CALLS = {
    "reg_write": "reg_write",
    "cfg_rmw": "cfg_rmw",
    "cfg_rmw_gpr": "cfg_rmw",
}
# Variable-name substrings that identify a raw MMIO pointer when provenance is
# only a variable (lower-confidence "name-fallback").
CFG_VAR_NAME_HINTS = {
    "regfile": "regfile_gpr",
    "gpr": "regfile_gpr",
    "mop_cfg": "mmio_ptr",
    "cfg": "cfg_ptr",
}
# Cast-target substrings that identify a raw MMIO pointer write.
CFG_CAST_HINTS = ("cfg", "mop", "base")


def classify_write(pw: dict):
    """Return (kind, detected_by) for a pointer_write fact, or (None, None)."""
    prov, prod = pw.get("provenance_kind"), pw.get("producer", "")
    if prov == "call":
        k = CFG_POINTER_PRODUCERS.get(prod)
        if k:
            return k, "ast-provenance"
        # a producer call we don't recognize as cfg — not an MMIO cfg/gpr write
        return None, None
    if prov == "cast":
        t = prod.lower()
        if "volatile" in t and any(h in t for h in CFG_CAST_HINTS):
            return "mmio_ptr", "ast-provenance"
        return None, None
    if prov == "var":
        vn = prod.lower()
        for sub, kind in CFG_VAR_NAME_HINTS.items():
            if sub in vn:
                return kind, "name-fallback"
    return None, None


def write_call_kind(callee: str):
    return MMIO_WRITE_CALLS.get(callee)


CFG_WRITE_KINDS = {"cfg32", "cfg16", "cfg_ptr", "mmio_ptr", "reg_write", "cfg_rmw"}
GPR_WRITE_KINDS = {"regfile_gpr"}


# --- Tensix instruction / primitive macros -----------------------------------
# Consumers must be genuine instruction macros; require a TTI_/TT_ prefix so that
# address macros (TENSIX_MOP_CFG_BASE) are never mistaken for a MOP run.
INSTR_PREFIXES = ("TTI_", "TT_")  # a genuine Tensix instruction macro


def _instr(name: str) -> bool:
    return name.startswith(INSTR_PREFIXES)


def is_thread_config_write(macro_name: str) -> bool:
    """SETC16 targets the per-thread ThreadConfig array (namespace THREAD), not the
    shared Config space — the one ordered-write macro that is NOT Config-namespace."""
    return "SETC16" in (macro_name or "").upper()


def is_gpr_source_write(macro_name: str) -> bool:
    """SETDMAREG writes a GPR source VALUE, not a sampled cfg word — so it is
    excluded from the cfg-word shared-word map (centralized here, not in the check)."""
    return "SETDMAREG" in (macro_name or "").upper()


# Ordered in-stream cfg/GPR writes (SAFE — Tensix instructions through the config
# unit). Substrings, must be instruction macros.
ORDERED_WRITE_MACRO_SUBSTR = (
    "REG2FLOP",
    "WRCFG",
    "SETC16",
    "RMWCIB",
    "SETADC",
    "SETDMAREG",
    "CFGSHIFTMASK",  # Quasar: TTI_CFGSHIFTMASK(CfgRegAddr, ...) — a config RMW write
)
# Calls that are ordered cfg writes.
ORDERED_WRITE_CALLS = ("cfg_reg_rmw_tensix",)  # substring match on callee text

# The config-RMW helper calls that take a FIELD (cfg_reg_rmw_tensix carries it in
# the template text `<FIELD_RMW>`; cfg_rmw / cfg_rmw_gpr carry it in arg0). These
# names were duplicated verbatim in cfg_word_overlap + reconfig_stall; centralized
# here (the single name→meaning table) so a rename touches one place.
CFG_RMW_TEMPLATE_CALL = "cfg_reg_rmw_tensix"  # matched as a substring of callee text
CFG_RMW_ARG0_CALLS = (
    "cfg_rmw",
    "cfg_rmw_gpr",
)  # matched on callee name (FIELD in arg0)


def is_cfg_rmw_helper(fact: dict) -> bool:
    """True if a call fact is a config-RMW helper (cfg_reg_rmw_tensix / cfg_rmw[_gpr])."""
    return (
        CFG_RMW_TEMPLATE_CALL in fact.get("text", "")
        or fact.get("name", "") in CFG_RMW_ARG0_CALLS
    )


STALL_MACRO_SUBSTR = ("STALLWAIT",)
# The STALLWAIT condition that orders a RISC cfg/GPR write. Matched by NAME, not
# value, because the encoding is PER-ARCH (WH = condition C13; BH = bit 10 / 0x400;
# Quasar = index 21) — the token spelling `TRISC_CFG` is stable across archs.
TRISC_CFG_TOKEN = "TRISC_CFG"

CONSUMER_UNPACK_SUBSTR = ("UNPACR",)
CONSUMER_PACK_SUBSTR = ("PACR",)  # checked after UNPACR
CONSUMER_MOP_SUBSTR = ("MOP", "REPLAY")  # excluding CFG/BASE address macros
# Matrix / FPU instruction macros that CONSUME config the RISC just wrote (ALU
# format, DEST, accumulation control). Best-effort CURATED set — SFPU vector ops
# and less-common matrix ops are a known partial (see mmio-race blind_spots).
# Deliberately excludes NOP-like macros (TTI_NOP / TTI_SFPNOP do NOT consume
# config), which now reach the fact base as object-like macros.
CONSUMER_MATH_SUBSTR = (
    "MVMUL",
    "GMPOOL",
    "GAPOOL",
    "DOTPV",
    "ELWADD",
    "ELWSUB",
    "ELWMUL",
    "GATESRCRST",
)
CONSUMER_CALLS = ("mop_run",)  # substring match on callee

# RISC-blocking drains / fences (functions).
DRAIN_CALLS = {
    "sync_regfile_write": "sync_regfile_write",  # drains GPR (regfile) writes
    "mop_sync": "mop_sync",  # drains in-flight MOPs
    "tensix_sync": "tensix_sync",  # drains the whole Tensix thread
}


def classify_macro(name: str):
    """Return a role string for a macro name, or None if not of interest.
    Roles: stall | ordered_write | consumer_unpack | consumer_pack | consumer_mop
    | consumer_math (matrix/FPU issue that consumes config — used by mmio-race and
    reconfig_stall.reissues_unit)
    """
    up = name.upper()
    has = lambda s: s in up
    if _instr(name) and any(has(s) for s in STALL_MACRO_SUBSTR):
        return "stall"
    if _instr(name) and any(has(s) for s in ORDERED_WRITE_MACRO_SUBSTR):
        return "ordered_write"
    if not _instr(name):
        return None
    if any(has(s) for s in CONSUMER_UNPACK_SUBSTR):
        return "consumer_unpack"
    if any(has(s) for s in CONSUMER_PACK_SUBSTR):
        return "consumer_pack"
    if any(has(s) for s in CONSUMER_MOP_SUBSTR) and not has("CFG") and not has("BASE"):
        return "consumer_mop"
    if any(has(s) for s in CONSUMER_MATH_SUBSTR):
        return "consumer_math"
    return None


def classify_call(callee: str, callee_text: str):
    """Return a role for a call, or None. Roles: drain:<what> | ordered_write |
    consumer_mop."""
    if callee in DRAIN_CALLS:
        return f"drain:{DRAIN_CALLS[callee]}"
    if any(s in callee_text for s in ORDERED_WRITE_CALLS):
        return "ordered_write"
    if any(s in callee for s in CONSUMER_CALLS):
        return "consumer_mop"
    return None


def is_consumer(role: str) -> bool:
    return role is not None and role.startswith("consumer")


# --- Semaphore / mutex protocol (semaphore-handshake checker) ------------------
# Wrapper functions (callee name -> op). RISC-MMIO wrappers + t6 in-stream ones.
SEMAPHORE_CALLS = {
    "semaphore_post": "post",
    "semaphore_get": "get",
    "semaphore_read": "read",
    "t6_semaphore_post": "post",
    "t6_semaphore_get": "get",
    "t6_semaphore_wait_on_max": "wait",
    "t6_semaphore_wait_on_zero": "wait",
    "t6_semaphore_init": "init",
    "t6_mutex_acquire": "mutex_acquire",
    "t6_mutex_release": "mutex_release",
}
# Raw ISA macros (substring, instruction-prefixed).
SEMAPHORE_MACRO_SUBSTR = {
    "SEMINIT": "init",
    "SEMPOST": "post",
    "SEMGET": "get",
    "SEMWAIT": "wait",
    "ATGETM": "mutex_acquire",
    "ATRELM": "mutex_release",
}


def classify_semaphore_call(callee: str):
    return SEMAPHORE_CALLS.get(callee)


def classify_semaphore_macro(name: str):
    if not _instr(name):
        return None
    up = name.upper()
    for sub, op in SEMAPHORE_MACRO_SUBSTR.items():
        if sub in up:
            return op
    return None


def is_semaphore_wrapper_def(fn_name: str) -> bool:
    """True if fn_name is the DEFINITION of a semaphore/mutex wrapper (its body
    contains the primitive by definition) — not a use site."""
    return fn_name in SEMAPHORE_CALLS


def is_ctor_or_dtor(fn_name: str) -> bool:
    """RAII acquire-in-ctor / release-in-dtor is balanced at the object level,
    not per-function, so those are skipped. A destructor's name starts with '~';
    a constructor's captured name is its (CamelCase) TYPE name. Match a CamelCase
    type-like name (leading uppercase AND no underscore). This still classifies a
    bare no-underscore `Capitalized` helper as a ctor/dtor (accepted — LLK funcs
    are snake_case/`_llk_*`, so this is vanishingly unlikely); it does NOT swallow
    a `Capitalized_snake` name (it has an underscore). It is tighter than the
    earlier bare `fn_name[0].isupper()` test, which dropped EVERY capitalized-named
    function (incl. capitalized_snake) from the balance check."""
    if not fn_name:
        return False
    if fn_name.startswith("~"):
        return True
    return fn_name[0].isupper() and "_" not in fn_name


# Semaphore identity. Waits name the semaphore as `semaphore::NAME`; SEMINIT
# names it as `p_stall::SEMAPHORE_n`; the generic wrapper uses a parameterized
# `t6_sem(index)`. These vocabularies do not reconcile statically, so identity
# matching is best-effort and a *parameterized* (generic) init is treated as a
# wildcard that inits any semaphore (see the semaphore-handshake checker).
_SEM_NAME_RE = re.compile(r"semaphore::\s*([A-Za-z_][A-Za-z0-9_]*)")
_SEM_PSTALL_RE = re.compile(r"p_stall::\s*(SEM[A-Za-z0-9_]*)")


def semaphore_target(fact: dict):
    """Return (semaphore_id | None, is_parameterized). `is_parameterized` marks
    a generic init/wait over an index variable (e.g. t6_sem(index))."""
    text = (
        fact.get("arg0", "") if fact.get("family") == "call" else fact.get("text", "")
    )
    # A parameterized/generic op is the `t6_sem(index)` wrapper form ONLY. A bare
    # `"index" in text` test was over-broad: a concrete op like
    # `wait(semaphore::MATH_PACK, tile_index)` would be misread as a wildcard and
    # its real identity discarded.
    if "t6_sem(" in text:
        return None, True
    m = _SEM_NAME_RE.search(text)
    if m:
        return m.group(1), False
    m = _SEM_PSTALL_RE.search(text)  # e.g. p_stall::SEMAPHORE_1 (NOT p_stall::STALL_*)
    if m:
        return m.group(1), False
    return None, False


def condition_drains_unit(operand: str, tokens) -> bool:
    """True if the STALLWAIT condition `operand` waits on any of `tokens`,
    matched at WORD boundaries. Substring matching would be wrong: 'PACK' is a
    substring of 'UNPACK', so `p_stall::UNPACK` (unpacker drain, C1/C2) would be
    misread as a packer drain (C3-C6)."""
    return any(re.search(rf"\b{re.escape(t)}\b", operand) for t in tokens)


def stallwait_wait_operand(text: str) -> str:
    """The DRAIN condition of a STALLWAIT is its wait_res operand(s); the FIRST
    operand (stall_res, e.g. STALL_UNPACK) names the block being held, not the unit
    being drained. Return every top-level argument AFTER the first, joined — WH/BH
    have a single wait_res (2-operand macro), but Quasar's macro is 4-operand
    (`TTI_STALLWAIT(stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0)`) and
    splits the wait condition across the last three, so returning only operand 2
    (which is often `0` on Quasar) would miss the real drain tokens."""
    lp = text.find("(")
    rp = text.rfind(")")
    if lp < 0 or rp <= lp:
        return ""
    inner = text[lp + 1 : rp]
    # Track (), [], <> and {} nesting so a comma inside a template-arg / brace-init
    # (e.g. STALLWAIT(foo<a, b>, cond)) does not split an argument. STALLWAIT
    # operands are plain p_stall:: enums today, so this is hardening, not a live
    # fix.
    depth, start, args = 0, 0, []
    for i, ch in enumerate(inner):
        if ch in "([<{":
            depth += 1
        elif ch in ")]>}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(inner[start:i])
            start = i + 1
    args.append(inner[start:])
    # Join operands 2..N (all wait_res, skipping stall_res at index 0) so the token
    # match sees the Quasar 4-operand condition (idx_2/idx_1/idx_0), not just one.
    return " ".join(a.strip() for a in args[1:]) if len(args) >= 2 else ""


# --- cfg-word-overlap: cfg_defines.h resolution -------------------------------
# The register field written is named in a pointer_write's index_text (e.g.
# THCON_SEC0_REG1_Row_start_section_size_ADDR32) or a cfg_reg_rmw_tensix<FIELD>.
# We resolve those to their 32-bit ADDR32 word via the arch cfg_defines.h — see
# resolve_word (below), which uses _WORD_OFFSET_RE / _RMW_ALIAS_RE.

# Relative path (from the metal repo root) to each arch's cfg_defines.h.
CFG_DEFINES_REL = {
    "wormhole": "tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
    "blackhole": "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
    "quasar": "tt_metal/hw/inc/internal/tt-2xx/quasar/cfg_defines.h",
}

# NAME -> integer literal (decimal OR hex), tolerating an integer-suffix
# (U/L/UL/…) and a trailing // comment. Captures _ADDR32 (decimal word index),
# _SHAMT (decimal), and _MASK (hex, pre-positioned 32-bit field mask).
_DEFINE_RE = re.compile(
    r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*(?://.*)?$"
)


def load_addr32(cfg_defines_path: str) -> dict:
    """Parse `#define NAME <int|hex>` lines from a cfg_defines.h into {name: int}.
    Includes the *_ADDR32 word indices (used by resolve_word) and the *_MASK
    field masks (used by field_bitmask for the masking annotation). A value that
    does not parse as base-0 (e.g. a leading-zero octal) is skipped, not fatal."""
    out = {}
    try:
        with open(cfg_defines_path) as fh:
            for ln in fh:
                m = _DEFINE_RE.match(ln)
                if m:
                    try:
                        out[m.group(1)] = int(m.group(2), 0)  # base 0 -> dec/hex
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def wrcfg_word_count(text: str) -> int:
    """A TTI_WRCFG(..., WRCFG_128b, ...) overwrites FOUR consecutive 32-bit config
    words (base..base+3); WRCFG_32b (and everything else) writes one. Returns the
    number of words the write spans so the overlap checker can enumerate them."""
    return 4 if "WRCFG_128B" in (text or "").upper() else 1


def field_bitmask(field_token: str, defines: dict):
    """The pre-positioned 32-bit mask of the bits a field occupies, from its
    sibling *_MASK define. `field_token` is the *_ADDR32 token (e.g.
    STACC_RELU_ApplyRelu_ADDR32). Returns int mask, or None if unresolved."""
    if not field_token:
        return None
    _S = "_ADDR32"
    base = field_token[: -len(_S)] if field_token.endswith(_S) else field_token
    return defines.get(base + "_MASK")


def write_is_atomic_masked(how: str) -> bool:
    """True if the write is a byte-atomic masked RMW (cfg_reg_rmw_tensix / RMWCIB,
    which lower to TT_RMWCIB) — the only mechanism that both preserves sibling
    bits AND is atomic across threads. A full-word write clobbers siblings; a
    software cfg_rmw preserves them but is NOT atomic across threads (needs a
    mutex); REG2FLOP writes bytes without a preserving RMW."""
    h = how.lower()
    return "cfg_reg_rmw_tensix" in h or "rmwcib" in h


def write_is_full_word(how: str) -> bool:
    h = how.upper()
    return "CFG32" in h or "CFG_PTR" in h or "MMIO_PTR" in h or "WRCFG" in h


_WORD_OFFSET_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*_ADDR32)\s*(?:\+\s*(0[xX][0-9a-fA-F]+|\d+))?"
)
# cfg_rmw / cfg_reg_rmw_tensix take a FIELD_RMW composite alias; the sibling
# word-address macro is FIELD_ADDR32.
_RMW_ALIAS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)_RMW\b")


def resolve_word(text: str, addr32: dict):
    """Resolve the 32-bit CONFIG word an expression writes: find the first
    *_ADDR32 token (+ optional literal offset) and look it up. Falls back to a
    *_RMW alias (FIELD_RMW -> FIELD_ADDR32). Returns
    (word:int, field:str) or (None, field_or_None)."""
    m = _WORD_OFFSET_RE.search(text or "")
    if m:
        field = m.group(1)
        base = addr32.get(field)
        if base is None:
            return None, field
        try:  # a malformed offset (e.g. leading-zero '+ 08') must not abort the
            off = int(m.group(2), 0) if m.group(2) else 0  # whole audit (cli.py
        except ValueError:  # doesn't wrap chk.run) — treat as unresolved instead.
            return None, field
        return base + off, field
    a = _RMW_ALIAS_RE.search(text or "")
    if a:
        field = a.group(1) + "_ADDR32"
        base = addr32.get(field)
        return (base, field) if base is not None else (None, field)
    return None, None


# NOTE: the register-FILE separation (Config vs ThreadConfig) is decided by the
# WRITE INSTRUCTION, not the field-name prefix — see cfg_word_overlap.py. THCON
# is a sub-range of the Config array (per tt-isa-docs BackendConfiguration.md),
# NOT a separate file, so a field-name/THCON split would be wrong.


# --- reconfig-stall -----------------------------------------------------------
# Candidate functions: names that rewrite config a running unit samples.
RECONFIG_FN_SUBSTR = (
    "reconfig",
    "_uninit",
    "set_packer_strides",
    "program_packer_destination",
    "configure_unpack",
    "configure_pack",
    "set_packer_config",
    "set_unpack_config",
)
# STALLWAIT condition tokens that actually DRAIN an execution unit (not THCON,
# which only orders the GPR->cfg write). Keyed by the writing thread.
DRAIN_UNIT_TOKENS = {
    "UNPACK": ("UNPACK", "UNPACK0", "UNPACK1"),
    "MATH": ("MATH", "WAIT_SFPU"),
    "PACK": ("PACK", "PACK0", "PACK1"),  # PACK0/PACK1 are Quasar's per-packer tokens
}
# The MATH thread drives TWO independent engines: the FPU (matrix, `p_stall::MATH`)
# and the SFPU (vector, `p_stall::WAIT_SFPU` == `SFPU1`; other archs add `SFPU0`/…).
# Draining one does NOT drain the other, so a math reconfig fully drains only when it
# waits on BOTH. UNPACK/PACK tokens are aliases of a single unit (any one = full).
MATH_FPU_TOKENS = ("MATH",)


def unit_drain_state(operand: str, thread: str):
    """(drains_any, drains_full) for a STALLWAIT wait-condition wrt `thread`'s unit.
    For MATH: FPU + SFPU are separate engines (SFPU matched by the 'SFPU' substring so
    WAIT_SFPU/SFPU1/SFPU0/... all count) — full drain needs BOTH, partial drains one.
    For UNPACK/PACK the tokens alias one unit, so any drain is a full drain."""
    if thread == "MATH":
        fpu = any(re.search(rf"\b{re.escape(t)}\b", operand) for t in MATH_FPU_TOKENS)
        sfpu = "SFPU" in operand.upper()
        return (fpu or sfpu, fpu and sfpu)
    d = condition_drains_unit(operand, DRAIN_UNIT_TOKENS.get(thread, ()))
    return (d, d)


# Config-write macros that write the UNIT-SAMPLED CONFIG register file (the ones
# a reconfig must drain the unit before). Deliberately EXCLUDES SETDMAREG, which
# writes a GPR (a source value for a later REG2FLOP), not a sampled config reg —
# so it must not be mistaken for "the reconfig write".
RECONFIG_WRITE_MACRO_SUBSTR = (
    "REG2FLOP",
    "WRCFG",
    "SETC16",
    "RMWCIB",
    "SETADC",
    "CFGSHIFTMASK",
)

# Latched (not sampled) registers where a THCON-only order is correct and adding
# a unit drain was reverted as over-sync. Substring match on the field name.
LATCHED_FIELDS = ("L1_Dest_addr",)


def is_reconfig_fn(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in RECONFIG_FN_SUBSTR)


# --- srcreg-bank: SrcA/SrcB data-valid handshake ------------------------------
# The dvalid flow-control primitives are the handshake CONTROL POINTS of the
# unpacker<->Matrix-Unit bank handoff. The tool RECALLS these sites (few, clean)
# and flags the one concrete ISA-grounded pattern — a *raw* SETDVALID on
# Blackhole (ISA-unsupported: it corrupts ImpliedSrcBFmt; the supported form is
# UNPACR_NOP(...,SET_DVALID,...)). The bank-flip lockstep, dvalid placement, and
# single-thread-ownership *verdicts* are semantic and stay with the LLM (see the
# SrcRegBank check's blind_spots).
SRCREG_DVALID_OPS = {
    "SETDVALID": "DVALID_SET",
    "CLEARDVALID": "DVALID_CLEAR",
}


def classify_srcreg_macro(name: str):
    """Return (op_token, role) for a dvalid ISSUE macro, or (None, None).

    Excludes ``TT_OP_*`` (opcode-VALUE constants, not an issued instruction) and
    any ``UNPACR`` family macro — ``UNPACR_NOP(...,SET_DVALID,...)`` is the
    *supported* way to set data-valid, NOT a raw ``SETDVALID``, so it must never
    be mistaken for the raw form the Blackhole flag targets."""
    if not (name.startswith("TTI_") or name.startswith("TT_")) or name.startswith(
        "TT_OP_"
    ):
        return None, None
    up = name.upper()
    if "UNPACR" in up:
        return None, None
    for tok, role in SRCREG_DVALID_OPS.items():
        if tok in up:
            return tok, role
    return None, None


# --- mailbox-sync (lite): in-tree RISC<->RISC FIFO endpoints -------------------
# The functional mailbox surface INSIDE tt-llk is tiny (the T1->T0 dst_index
# hand-off + the debug halt/unhalt handshake); the large mailbox surface (the CB
# tile-address/value broadcast) lives in the compute API / kernels, OUTSIDE the
# headers this tool parses (the kernel/JIT tier — see run.sh --full-jit). So the
# tool RECALLS the in-tree endpoints and their pairing; the balance-over-
# iterations / ordering / deadlock verdict stays with the LLM.
#
# Convention (Mailboxes.md): a mailbox is a DIRECTED FIFO. `mailbox_write(dest)`
# is issued on the SOURCE core and names the DESTINATION thread; `mailbox_read`/
# `mailbox_not_empty(src)` are issued on the DESTINATION core and name the SOURCE
# thread. So a write and a read of the SAME physical FIFO carry DIFFERENT thread
# args — the pairing key is the resolved (source_thread -> dest_thread) channel.
MAILBOX_FIFO_CALLS = {
    "mailbox_write": "write",  # push  — arg0 = DEST thread, issued on SOURCE core
    "mailbox_read": "read",  # blocking pop — arg0 = SOURCE thread, on DEST core
    "mailbox_not_empty": "query",  # non-blocking — arg0 = SOURCE thread, on DEST core
}
# (record_mailbox_value / clear_mailbox_values / debug_mailbox are the L1 debug
# SCRATCH mailbox — a different thing; excluded by matching only the 3 names.)
_THREADID_RE = re.compile(r"ThreadId::\s*([A-Za-z_][A-Za-z0-9_]*)")


def mailbox_op(callee: str):
    return MAILBOX_FIFO_CALLS.get(callee)


# --- cb-sync (kernel tier): circular-buffer credit primitives ------------------
# These live in JIT-compiled kernels OUTSIDE tt-llk; the checker is deterministic
# and committed, but only produces findings when fed a KERNEL fact base (the
# on-request capture — see run.sh --full-jit). Over the tt-llk fact base there are
# no cb_* calls, so it is trivially empty there.
CB_CALLS = {
    "cb_reserve_back": "reserve",  # producer: claim space
    "cb_push_back": "push",  # producer: commit (hand a credit to the consumer)
    "cb_wait_front": "wait",  # consumer: wait for pages
    "cb_pop_front": "pop",  # consumer: release the pages
    # Remote/sharded CB family (asymmetric names — the push is fused with the
    # page write). Balance still pairs reserve<->push / wait<->pop by count.
    "remote_cb_reserve_back": "reserve",
    "remote_cb_push_back_and_write_pages": "push",
    "remote_cb_wait_front": "wait",
    "remote_cb_pop_front": "pop",
    # Thin CB-op wrappers seen in ttnn kernels (SDPA). Curated — non-canonical
    # wrappers are a known partial (see cb_sync blind_spots); add here as found.
    "cb_push_back_hold_wr_ptr": "push",
    "cb_wait_fronts": "wait",
}
# Modern ttnn kernels use the OBJECT/METHOD API: `CircularBuffer cb(id);
# cb.reserve_back(n)`. The callee is the METHOD name (no cb_ prefix) and the CB
# identity is the RECEIVER object, not arg0 (which is the page count). Gate on the
# receiver TYPE so `push_back` on a std::vector is NOT mistaken for a CB push.
CB_METHOD_CALLS = {
    "reserve_back": "reserve",
    "push_back": "push",
    "wait_front": "wait",
    "pop_front": "pop",
}
_CB_RECV_TYPES = ("CircularBuffer", "CBInterface")


def _is_cb_receiver(recv_type: str) -> bool:
    return any(t in (recv_type or "") for t in _CB_RECV_TYPES)


def cb_classify(fact: dict):
    """Return (op, cb_id_key) for a CB flow-control call, or (None, None).
    Free function `cb_reserve_back(cb, n)` -> cb id is arg0; method
    `cb_obj.reserve_back(n)` on a CircularBuffer -> cb id is the RECEIVER object."""
    name = fact.get("name", "")
    if name in CB_CALLS:
        return CB_CALLS[name], (fact.get("arg0", "") or "?").strip() or "?"
    if name in CB_METHOD_CALLS and _is_cb_receiver(fact.get("recv_type", "")):
        return CB_METHOD_CALLS[name], (fact.get("recv", "") or "?").strip() or "?"
    return None, None


# --- noc-sync (kernel tier): NoC semaphore + write-flush primitives ------------
# Remote CREDIT signals — a NoC write that posts a credit (the data-before-credit
# target). NOTE: plain `noc_semaphore_set` is DELIBERATELY EXCLUDED — it is a
# LOCAL store `(*sem_addr)=val` (reset/init of a local semaphore), NOT a remote
# signal, so flagging it would fire NOC_SIGNAL_NO_FLUSH on every local reset.
# `noc_semaphore_set_remote` is the remote (4-byte-write) form and IS a signal.
NOC_SIGNAL_CALLS = {
    "noc_semaphore_inc": "inc",
    "noc_semaphore_inc_multicast": "mcast",
    "noc_semaphore_set_remote": "set",
    "noc_semaphore_set_multicast": "mcast",
    "noc_semaphore_set_multicast_loopback_src": "mcast",
}
NOC_WAIT_CALLS = ("noc_semaphore_wait", "noc_semaphore_wait_min")
# A write is "landed" (safe to credit) only after one of these drains the NoC.
NOC_FLUSH_CALLS = (
    "noc_async_write_barrier",
    "noc_async_writes_flushed",
    "noc_async_write_barrier_with_trid",
    "noc_async_write_flushed_with_trid",
)
# The POSTED-writes flush is a SEPARATE, weaker primitive. NoC writes are tracked by
# TWO independent HW counters — posted (NIU_MST_POSTED_WR_REQ_SENT) and non-posted
# (NIU_MST_NONPOSTED_WR_REQ_SENT) — and `noc_async_posted_writes_flushed` polls ONLY
# the posted counter. If the preceding write/inc was issued NON-posted (the common
# default path, e.g. a remote_cb write/inc whose `posted` template arg is false), this
# flush waits on a counter that was never incremented -> it returns immediately and
# provides NO ordering. The tool cannot see a write's posted-ness (a constexpr/template
# property), so a posted-flush is recognized as a flush but classified DISTINCTLY
# ("posted"): it is NEVER placed in the clearing sets, and a credit whose only
# preceding flush is a posted-flush is surfaced with safety=POSTED_FLUSH_ONLY for the
# skill to confirm the write/credit posted-ness (a wrong-counter flush is a silent
# false-clear otherwise). See the dataflow-API posted/non-posted counter split.
NOC_POSTED_FLUSH_CALLS = ("noc_async_posted_writes_flushed",)


def noc_op(callee: str):
    """Return inc|set|mcast (a credit SIGNAL) | wait | flush | None."""
    if callee in NOC_SIGNAL_CALLS:
        return NOC_SIGNAL_CALLS[callee]
    if callee in NOC_WAIT_CALLS:
        return "wait"
    if callee in NOC_FLUSH_CALLS or callee in NOC_POSTED_FLUSH_CALLS:
        return "flush"
    return None


# The WRITE-flush methods on the `Noc` accessor object (noc.async_write_barrier()).
# Read barriers are excluded — they drain reads, not the writes a credit refers to.
NOC_METHOD_FLUSH = (
    "async_write_barrier",
    "async_writes_flushed",
    "async_write_barrier_with_trid",
    "async_write_flushed_with_trid",
)
# Object-method form of the posted-only flush (see NOC_POSTED_FLUSH_CALLS).
NOC_METHOD_POSTED_FLUSH = ("async_posted_writes_flushed",)
_NOC_RECV_TYPES = ("Noc",)

# The modern object API also has credit-SIGNAL methods on the `Semaphore` object
# (noc_semaphore.h) — the object-form parallel of NOC_SIGNAL_CALLS, NOT free
# functions. The multicast forms are unambiguously remote signals. `up` is
# OVERLOADED: `up(uint32_t value)` is a LOCAL increment (no NoC, needs no flush)
# while `up(const Noc&, x, y, value, vc)` is the REMOTE signal — distinguished by
# argument count (local=1, remote>=4), so `up` is a signal only when argc>=2.
# `set(value)` is likewise a LOCAL set (excluded); the remote set is the multicast
# form. Gated on the Semaphore receiver TYPE so an unrelated `.up()` can't match.
NOC_METHOD_SIGNAL_MCAST = (
    "set_multicast",
    "set_multicast_loopback_src",
    "relay_multicast",
    "inc_multicast",
)
# Object-method write credits that are NOT multicast: relay_unicast lowers to
# noc_semaphore_set_remote (a 4-byte remote WRITE) — the object twin of set_remote.
NOC_METHOD_SIGNAL_WRITE = ("relay_unicast",)
_NOC_SIGNAL_RECV_TYPES = ("Semaphore",)


def _is_noc_signal_receiver(recv_type: str) -> bool:
    return any(t in (recv_type or "") for t in _NOC_SIGNAL_RECV_TYPES)


def noc_op_of(fact: dict):
    """Fact-aware noc_op: free-function form, the Noc-method write-flush, OR the
    Semaphore-object credit signals (mcast / write / remote `up`)."""
    op = noc_op(fact.get("name", ""))
    if op:
        return op
    name = fact.get("name", "")
    if (name in NOC_METHOD_FLUSH or name in NOC_METHOD_POSTED_FLUSH) and any(
        t in (fact.get("recv_type", "") or "") for t in _NOC_RECV_TYPES
    ):
        return "flush"
    if _is_noc_signal_receiver(fact.get("recv_type", "")):
        if name in NOC_METHOD_SIGNAL_MCAST:
            return "mcast"
        if name in NOC_METHOD_SIGNAL_WRITE:
            return "set"
        # `up` is a remote signal only in its NoC-carrying overload (argc>=2);
        # argc==-1 (unknown, pre-argc fact base) is treated as NOT a signal to
        # avoid false NOC_SIGNAL_NO_FLUSH on the local up(value) form.
        if name == "up" and fact.get("argc", -1) >= 2:
            return "inc"
    return None


# Which credit signals are ATOMIC increments vs write-based. `inc_multicast` /
# `noc_semaphore_inc_multicast` lower to a multicast ATOMIC increment, so they are
# atomic despite the "mcast" op label; set/set_multicast/relay_* are writes. The
# checker CLEARS an atomic credit only on a preceding ack BARRIER (the conservative
# choice — see NOC_WRITE_BARRIERS); it clears a write credit on any flush.
NOC_ATOMIC_SIGNALS = (
    "noc_semaphore_inc",
    "noc_semaphore_inc_multicast",
    "inc_multicast",  # Semaphore-object form
)


def noc_signal_is_atomic(fact: dict) -> bool:
    """True if this credit signal is an atomic increment (a remote atomic, not a
    plain write)."""
    name = fact.get("name", "")
    if name in NOC_ATOMIC_SIGNALS:
        return True
    # remote Semaphore::up(noc, x, y, v) is an atomic increment
    return (
        name == "up"
        and _is_noc_signal_receiver(fact.get("recv_type", ""))
        and fact.get("argc", -1) >= 2
    )


# The two write-completion primitives differ in what they WAIT FOR (see the
# dataflow-API headers under tt_metal/hw/inc/api/dataflow/ and the data-movement doc
# data_movement_doc/general/posted_writes.md): a *barrier* waits for the remote ACK
# (the data has LANDED / been COMMITTED at the destination); a *flushed* only drains
# the initiator's outgoing queue (the write has DEPARTED, not yet committed).
#
# The checker CLEARS an atomic credit only on a preceding barrier — the CONSERVATIVE
# choice, grounded in posted_writes.md: the data-before-credit race is the increment
# observed before the write is COMMITTED, and writes_flushed does not guarantee
# commit. Whether a same-NoC/VC UNICAST atomic is nonetheless safe with just a flush
# (in-issue-order delivery landing the payload first) is a doc-grounded VERDICT —
# the /noc-sync skill decides it against <arch>/NoC/Ordering.md + posted_writes.md,
# the tool does NOT assume it. A MULTICAST credit, or writes crossing separate
# command-buffer FIFOs / different NoCs / a different VC, definitely need the barrier.
# So the tool keeps flagging a flush-but-no-barrier atomic (never miss a real race)
# and TAGS it FLUSH_NOT_BARRIER for the skill to confirm, rather than clearing it.
NOC_WRITE_BARRIERS = (
    "noc_async_write_barrier",
    "noc_async_write_barrier_with_trid",
    "async_write_barrier",  # Noc-object method form
    "async_write_barrier_with_trid",
)


def noc_flush_kind(fact: dict):
    """For a fact noc_op_of classifies as 'flush', return the flush strength:
    'barrier' — waits for the remote ACK (data LANDED / committed at the dest)
    'flushed' — initiator's NON-posted queue drained (write DEPARTED, not committed)
    'posted'  — ONLY the posted-writes counter drained; a no-op for non-posted
                writes/incs (a wrong-counter flush). Never clears a credit — see
                NOC_POSTED_FLUSH_CALLS. Distinguished by name so the checker can tag
                a posted-only-cleared credit rather than silently treat it as safe."""
    if noc_op_of(fact) != "flush":
        return None
    name = fact.get("name", "")
    if name in NOC_WRITE_BARRIERS:
        return "barrier"
    if name in NOC_POSTED_FLUSH_CALLS or name in NOC_METHOD_POSTED_FLUSH:
        return "posted"
    return "flushed"


# --- noc-atomic-exit: the ATOMIC-drain barrier + kernel-entry recognition ------
# A non-posted NoC atomic (noc_semaphore_inc / remote Semaphore::up) is tracked by a
# SEPARATE HW counter from writes (noc_nonposted_atomics_acked / NIU_MST_ATOMIC_RESP_
# RECEIVED). A write barrier / writes_flushed does NOT drain it — ONLY these do. Used
# by the noc-atomic-exit check: if a kernel entry issues an atomic and returns without
# draining it, the atomic is in flight at exit — the post-kernel_main firmware epilogue
# does NOT drain atomics in release/Watcher-off builds (it only ASSERTs idle under
# DM_DEDICATED_NOC), so the kernel itself must, or the readiness signal is left pending
# across program teardown / noc_init.
NOC_ATOMIC_BARRIERS = ("noc_async_atomic_barrier",)
NOC_METHOD_ATOMIC_BARRIER = ("async_atomic_barrier",)  # Noc-object method form


def noc_is_atomic_barrier(fact: dict) -> bool:
    """True if the call drains the non-posted ATOMIC counter (free-fn or Noc method)."""
    name = fact.get("name", "")
    if name in NOC_ATOMIC_BARRIERS:
        return True
    return name in NOC_METHOD_ATOMIC_BARRIER and any(
        t in (fact.get("recv_type", "") or "") for t in _NOC_RECV_TYPES
    )


# A data-movement kernel's outermost entry body. The NOC-idle-at-exit contract applies
# at the point this returns. (Compute kernels use MAIN/main and rarely issue raw NoC
# atomics; scoped to kernel_main to avoid flagging helper functions whose caller drains.)
KERNEL_ENTRY_NAMES = ("kernel_main",)


def is_kernel_entry(fn_name: str) -> bool:
    return (fn_name or "") in KERNEL_ENTRY_NAMES


# --- noc-read-barrier: inbound READ drained before it is consumed --------------
# An inbound async READ fills a local L1 buffer; its data is present only after a READ
# barrier drains it (noc_async_read_barrier also invalidates the L1 cache on BH). The
# barrier is a DIFFERENT primitive from the write flushes above and drains the read
# counter, not the write counter. Consuming the buffer before the read barrier reads
# stale/partial L1.
NOC_METHOD_READ_BARRIER = ("async_read_barrier", "async_read_barrier_with_trid")


def noc_is_read(fact: dict) -> bool:
    """A free-fn `noc_async_read*` (any variant) or Noc-method `async_read`, EXCLUDING
    the barrier/flush forms (noc_async_read_barrier / noc_async_reads_flushed)."""
    n = fact.get("name", "")
    if n.startswith("noc_async_read") and "barrier" not in n and "flush" not in n:
        return True
    return n == "async_read" and any(
        t in (fact.get("recv_type", "") or "") for t in _NOC_RECV_TYPES
    )


def noc_is_read_barrier(fact: dict) -> bool:
    """A read-drain barrier: free-fn noc_async_read_barrier[_with_trid] or the Noc
    method form. (A bare noc_async_reads_flushed is NOT treated as the drain — it does
    not invalidate the L1 cache, so the checker keeps flagging conservatively.)"""
    n = fact.get("name", "")
    if n.startswith("noc_async_read_barrier"):
        return True
    return n in NOC_METHOD_READ_BARRIER and any(
        t in (fact.get("recv_type", "") or "") for t in _NOC_RECV_TYPES
    )


# Ops that CONSUME/forward the just-read buffer downstream: a CB push hands it to the
# compute consumer; tt_memmove forwards it (its guaranteed-16B-aligned path is itself a
# NoC write whose SOURCE is the read buffer). A plain noc_async_write is deliberately
# NOT included — its source may be unrelated to the read (a miss documented in the
# check's blind_spots) — to keep the recall from firing on every read+write kernel.
READ_FORWARD_CALLS = ("tt_memmove",)


def is_read_consumer(fact: dict) -> bool:
    """True if the call consumes/forwards a just-read L1 buffer (CB push or tt_memmove)."""
    if fact.get("name", "") in READ_FORWARD_CALLS:
        return True
    op, _ = cb_classify(fact)
    return op == "push"


def mailbox_thread_arg(arg0: str):
    """Decode a `ThreadId::MathThreadId`-style arg0 to a short thread name
    (UNPACK/MATH/PACK), or None if it isn't a recognizable ThreadId literal
    (e.g. a runtime variable — then the channel end is unresolved)."""
    m = _THREADID_RE.search(arg0 or "")
    if not m:
        return None
    n = m.group(1).upper()
    for key in (
        "UNPACK",
        "MATH",
        "PACK",
    ):  # order-independent: none is a prefix of another
        if n.startswith(key):
            return key
    return None
