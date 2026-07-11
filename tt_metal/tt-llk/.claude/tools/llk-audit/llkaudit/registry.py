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
def _instr(name: str) -> bool:
    return name.startswith("TTI_") or name.startswith("TT_")


# Ordered in-stream cfg/GPR writes (SAFE — Tensix instructions through the config
# unit). Substrings, must be instruction macros.
ORDERED_WRITE_MACRO_SUBSTR = (
    "REG2FLOP",
    "WRCFG",
    "SETC16",
    "RMWCIB",
    "SETADC",
    "SETDMAREG",
)
# Calls that are ordered cfg writes.
ORDERED_WRITE_CALLS = ("cfg_reg_rmw_tensix",)  # substring match on callee text

STALL_MACRO_SUBSTR = ("STALLWAIT",)
TRISC_CFG_TOKEN = (
    "TRISC_CFG"  # the STALLWAIT condition (ISA C13) that orders a RISC cfg/GPR write
)

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
    a constructor's captured name is its (CamelCase) TYPE name. Match only a
    CamelCase type-like name (leading uppercase AND no underscore) so we do NOT
    also swallow a genuine capitalized_snake or `Capitalized` helper function —
    the earlier `fn_name[0].isupper()` test was too broad and could silently drop
    a real acquire/release imbalance in any capitalized-named function."""
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
    """The DRAIN condition of a STALLWAIT is its SECOND operand (wait_res); the
    first (stall_res, e.g. STALL_UNPACK) names the block being held, not the unit
    being drained. Return the text of the 2nd top-level argument (or "")."""
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
    return args[1].strip() if len(args) >= 2 else ""


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
    base = field_token[:-7] if field_token.endswith("_ADDR32") else field_token
    return defines.get(base + "_MASK")


# Full-word writes (cfg[]= / WRCFG_32b) touch all 32 bits.
_FULL_WORD_MASK = 0xFFFFFFFF


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
        return base + (int(m.group(2), 0) if m.group(2) else 0), field
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
    "PACK": ("PACK",),
}
# Config-write macros that write the UNIT-SAMPLED CONFIG register file (the ones
# a reconfig must drain the unit before). Deliberately EXCLUDES SETDMAREG, which
# writes a GPR (a source value for a later REG2FLOP), not a sampled config reg —
# so it must not be mistaken for "the reconfig write".
RECONFIG_WRITE_MACRO_SUBSTR = ("REG2FLOP", "WRCFG", "SETC16", "RMWCIB", "SETADC")

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


def mailbox_thread_arg(arg0: str):
    """Decode a `ThreadId::MathThreadId`-style arg0 to a short thread name
    (UNPACK/MATH/PACK), or None if it isn't a recognizable ThreadId literal
    (e.g. a runtime variable — then the channel end is unresolved)."""
    m = _THREADID_RE.search(arg0 or "")
    if not m:
        return None
    n = m.group(1).upper()
    for key in ("UNPACK", "MATH", "PACK"):  # UNPACK before PACK (startswith)
        if n.startswith(key):
            return key
    return None
