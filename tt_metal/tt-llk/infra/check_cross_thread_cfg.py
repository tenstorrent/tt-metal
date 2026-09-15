#!/usr/bin/env python3
"""Flag a Tensix config write that can corrupt a config field another thread owns.

Most Tensix backend config lives in ONE copy shared by all three threads, packed several fields to
a 32-bit word. Two consequences:

  CLOBBER    A whole-word write (WRCFG_32b, `cfg[w] = ...`) writes all 32 bits, so it destroys every
             other field in that word -- including fields belonging to another thread.
  SAME-FIELD Two threads doing a masked read-modify-write of the SAME bits still race on the value.
             (DIFFERENT bits are safe without a mutex: the config RMW is per-byte atomic.)

The per-thread register file (written by SETC16) is exempt: each thread has its own copy. Note the
two spaces numerically ALIAS -- THREAD_CFGREG_BASE_ADDR32 and ALU_CFGREG_BASE_ADDR32 are both 0 --
so classification is by the WRITING INSTRUCTION, never by the address.

Field ownership is derived from the tree rather than hand-maintained: the checker first records
which thread writes which field (from the enclosing llk_unpack_* / llk_math_* / llk_pack_* file or
function), then flags writes that cross those boundaries.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

# --- config-register definitions ----------------------------------------------------------------
DEF_ADDR = re.compile(r"^#define\s+(\w+?)_ADDR32\s+(\d+)")
DEF_SHAMT = re.compile(r"^#define\s+(\w+?)_SHAMT\s+(\d+)")
DEF_MASK = re.compile(r"^#define\s+(\w+?)_MASK\s+(0x[0-9a-fA-F]+|\d+)")


def load_defs(path):
    """field -> (word, shamt, mask). Only fields with all three are usable."""
    addr, shamt, mask = {}, {}, {}
    for line in open(path, errors="ignore"):
        for rx, d in ((DEF_ADDR, addr), (DEF_SHAMT, shamt), (DEF_MASK, mask)):
            m = rx.match(line)
            if m:
                d[m.group(1)] = int(m.group(2), 0)
    return {k: (addr[k], shamt[k], mask[k]) for k in addr if k in shamt and k in mask}


# --- write mechanisms ---------------------------------------------------------------------------
# Whole-word: every bit of the word is overwritten.
# Addressing is frequently BASE + OFFSET. Three pitfalls, all of which silently corrupt a result:
#   cfg[REG + i]        a variable offset -- the target word is not statically known
#   WRCFG(.., REG + 1)  a literal offset the address pattern must not drop
#   WRCFG_128b          writes FOUR consecutive words, not one
# Offsets are parsed as expressions; a non-literal offset is reported UNRESOLVED, never assumed.
# The mode argument is the wr128b bit: spelled p_cfg::WRCFG_32b/WRCFG_128b at most sites and as
# the bare literal at the rest. TT_OP_WRCFG builds a MOP word, which executes a WRCFG when the MOP
# runs, so it is a write like any other -- the macro definitions in ckernel_ops.h pass the mode
# through as `wr128b` and so do not match.
WRCFG_ANY = re.compile(
    r"\b(?:TTI?|TT_OP)_WRCFG\s*\(\s*[^,]+,\s*(?:p_cfg::)?(WRCFG_32b|WRCFG_128b|0|1)\s*,\s*([^)]+)\)"
)
WRCFG_WIDE = ("WRCFG_128b", "1")
CFG_INDEX = re.compile(r"\bcfg\s*\[([^\]]+)\]\s*=")
ADDR_EXPR = re.compile(r"^\s*([A-Za-z_0-9]+)_ADDR32\s*(?:\+\s*(.+?))?\s*$")


def resolve_addr(expr, defs):
    """(word, n_words, ok). ok=False when the offset is not a compile-time constant."""
    m = ADDR_EXPR.match(expr)
    if not m or m.group(1) not in defs:
        return None, 0, False
    base = defs[m.group(1)][0]
    off = m.group(2)
    if off is None:
        return base, 1, True
    off = off.strip()
    if re.fullmatch(r"\d+", off):
        return base + int(off), 1, True
    return base, 1, False  # variable offset: target not statically known


# Masked read-modify-write: only the field's bits change.
MASKED = [
    re.compile(r"\bcfg_reg_rmw_tensix\s*<\s*([A-Za-z_0-9]+?)_RMW\s*>"),
    re.compile(
        r"\bcfg_reg_rmw_tensix\s*<\s*([A-Za-z_0-9]+)_ADDR32\s*,\s*[^,]+,\s*([A-Za-z_0-9]+)\s*>"
    ),
    re.compile(r"\bcfg_reg_rmw_tensix\s*<\s*([A-Za-z_0-9]+)_ADDR32\s*,"),
]
# A combined write names one field for the address but masks several: the reconfig helpers build
# `config_mask = A_MASK | B_MASK`. Reading only the named field's mask misses the other fields in
# the same write -- which is how INT8_math_enabled was invisible in the first version.
MASK_CONST = re.compile(r"\b(\w*mask\w*)\s*=\s*([^;]+);", re.I)


def resolve_mask(name, text, defs, depth=0, seen=None):
    """Resolve a mask identifier to a bitmask, following local constants transitively.

    Masks are built in layers: `alu_mask = alu_format_mask | alu_stoch_rnd_mask`, and only the
    leaves are cfg_defines `*_MASK` names. A non-recursive resolver returns nothing here, and
    falling back to the NAMED field's mask then attributes a write to bits it never touches --
    which is how a disjoint-bit (and therefore safe) unpack write looked like a format-bit race.

    Returns None when the mask cannot be resolved, so the caller can decline to judge rather
    than guess.
    """
    if depth > 6:
        return None
    seen = seen or set()
    if name in seen:
        return None
    seen.add(name)
    if name in defs:
        return defs[name][2]
    # `text` runs from the top of the file to the write being resolved, and a name like
    # `config_mask` is redeclared in sibling functions with different bits. The definition that
    # governs is the nearest one above the write, which is the LAST match, not the first.
    hits = [m for m in MASK_CONST.finditer(text) if m.group(1) == name]
    for m in hits[-1:]:
        bits, ok = 0, False
        for ref in re.findall(r"\b(\w+)\b", m.group(2)):
            # defs is keyed on the FIELD name; the source writes `<FIELD>_MASK`. Strip the
            # suffix before lookup or every leaf reference fails and the mask never resolves.
            leaf = ref[:-5] if ref.endswith("_MASK") else ref
            if leaf in defs:
                bits |= defs[leaf][2]
                ok = True
            elif ref.lower().endswith("mask"):
                sub = resolve_mask(ref, text, defs, depth + 1, seen)
                if sub is not None:
                    bits |= sub
                    ok = True
        return bits if ok else None
    return None


# Per-thread register file -- each thread has its own copy, so it cannot cross threads.
PER_THREAD = re.compile(r"\bTTI?_SETC16\b")
MUTEX_ACQ = re.compile(r"t6_mutex_acquire\s*\(\s*mutex::REG_RMW")
MUTEX_REL = re.compile(r"t6_mutex_release\s*\(\s*mutex::REG_RMW")

THREADS = ("UNPACK", "MATH", "PACK")


# --- consumer map ------------------------------------------------------------------------------
# Which THREAD's instructions consume a field is not derivable from the LLK source; it is a
# hardware fact, so it is supplied as a vendored table rather than inferred here.
READER_THREADS = [
    ("unpacker", "UNPACK"),
    ("matrix unit", "MATH"),
    ("sfpu", "MATH"),  # SFPU instructions are issued by the math thread
    ("vector unit", "MATH"),
    ("packer", "PACK"),  # includes "the Dest-read output stage that feeds the packer"
    ("pack path", "PACK"),
]


CONSUMERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cfg_consumers.yaml"
)


def load_consumers(arch, path=None):
    """field -> set(consuming threads), from the vendored table.

    The table is generated data, committed so the check needs no external input at run time.
    A field ABSENT from it has no reader recorded for it: that is UNKNOWN, not 'nobody consumes
    it'.
    """
    try:
        import yaml
    except ImportError:
        return None
    try:
        doc = yaml.safe_load(open(path or CONSUMERS_FILE))
    except Exception:
        return None
    table = (doc or {}).get(arch) or {}
    return {k: set(v) for k, v in table.items()} or None


def thread_of(path, text_before):
    """Which Tensix thread executes this line, from the file and enclosing function."""
    base = os.path.basename(path)
    for name, keys in (
        ("UNPACK", ("llk_unpack", "cunpack_common")),
        ("MATH", ("llk_math", "cmath_common")),
        ("PACK", ("llk_pack", "cpack_common")),
    ):
        if any(k in base for k in keys):
            return name
    m = None
    for fn in re.finditer(r"_llk_(unpack|math|pack)\w*_\s*\(", text_before):
        m = fn
    return (
        {"unpack": "UNPACK", "math": "MATH", "pack": "PACK"}[m.group(1)] if m else None
    )


def writes_in(path, defs):
    """Yield (line_no, thread, kind, field, word, bitmask, guarded)."""
    lines = open(path, errors="ignore").read().split("\n")
    joined = ""
    # Track the mutex as acquire/release STATE, not a fixed lookback window: a protected write can
    # sit far below its acquire (cunpack_common.h has 26 lines between them), and a window-based
    # check reports those as unguarded.
    held = False
    for i, raw in enumerate(lines):
        line = raw.split("//")[0]
        joined += raw + "\n"
        if not line.strip():
            continue
        if PER_THREAD.search(line):
            continue  # per-thread file; cannot cross threads
        # A write whose thread cannot be determined is REPORTED, not dropped: the file/function
        # naming conventions thread_of() keys on do not cover every spelling (a ThreadId template
        # parameter dispatches by value, shared headers carry no thread in the name). Silently
        # skipping those is the one outcome the rest of this checker is written to avoid. It is a
        # note rather than a finding -- an unclassified write is a gap in this tool's coverage, not
        # evidence of a hazard, and must not fail anyone's commit.
        th = thread_of(path, joined)
        if MUTEX_ACQ.search(line):
            held = True
        if MUTEX_REL.search(line):
            held = False
        guarded = held
        hit = False
        m = WRCFG_ANY.search(line)
        if m:
            word, _n, ok = resolve_addr(m.group(2), defs)
            if word is not None:
                hit = True
                # WRCFG_128b writes an aligned group of FOUR words.
                span = 4 if m.group(1) in WRCFG_WIDE else 1
                for k in range(span):
                    yield i + 1, th, "whole-word" if ok else "addr-unresolved", m.group(
                        2
                    ).strip(), word + k, 0xFFFFFFFF, guarded
        if not hit:
            m = CFG_INDEX.search(line)
            if m:
                word, _n, ok = resolve_addr(m.group(1), defs)
                if word is not None:
                    hit = True
                    yield i + 1, th, "whole-word" if ok else "addr-unresolved", m.group(
                        1
                    ).strip(), word, 0xFFFFFFFF, guarded
        if not hit:
            for rx in MASKED:
                m = rx.search(line)
                if m and m.group(1) in defs:
                    w, _sh, mk = defs[m.group(1)]
                    if m.lastindex and m.lastindex > 1 and m.group(2):
                        combined = resolve_mask(m.group(2), joined, defs)
                        if combined is None:
                            # Do not fall back to the named field's mask: that attributes the
                            # write to bits it may never touch. Report it as unresolved instead.
                            yield i + 1, th, "mask-unresolved", m.group(
                                1
                            ), w, 0, guarded
                            break
                        mk = combined
                    # _MASK is already positioned within the word (e.g. SrcA: SHAMT 17,
                    # MASK 0x1e0000). Shifting by SHAMT again would double-shift and compare
                    # the wrong bits, producing both false positives and false negatives.
                    yield i + 1, th, "masked", m.group(1), w, mk, guarded
                    break


def collect(paths, defs):
    sites = []
    for p in paths:
        sites += [(p,) + w for w in writes_in(p, defs)]
    return sites


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--defs", required=True, help="cfg_defines.h for the architecture")
    ap.add_argument(
        "--tree",
        required=True,
        help="LLK arch tree; ownership is derived from all of it",
    )
    ap.add_argument(
        "--arch", default="wormhole_b0", help="key into the vendored consumer table"
    )
    ap.add_argument("--consumers", help="override the vendored consumer table")
    ap.add_argument("--baseline")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    # Quasar writes config with cfg_rmw / RMWCIB rather than cfg_reg_rmw_tensix, and cfg_rmw is a
    # DIFFERENT mechanism there (RMWCIB) than on Wormhole/Blackhole (MMIO). A parser tuned to the
    # 1xx spelling sees almost nothing on Quasar and reports a comfortable, meaningless zero.
    SUPPORTED = ("wormhole_b0", "blackhole")
    if args.arch not in SUPPORTED:
        print(
            f"REFUSING: arch '{args.arch}' is not supported (supported: {', '.join(SUPPORTED)})."
        )
        print(
            "  Quasar uses cfg_rmw/RMWCIB for config writes; this checker does not model them,"
        )
        print("  so a clean result here would mean 'unmeasured', not 'no hazards'.")
        return 2

    defs = load_defs(args.defs)
    consumers = load_consumers(args.arch, args.consumers)
    all_h = [
        os.path.join(d, f)
        for d, _, fs in os.walk(args.tree)
        for f in fs
        if f.endswith(".h")
    ]
    every = collect(all_h, defs)

    # Derive ownership from the tree: which threads write which bits of which word.
    owners = defaultdict(lambda: defaultdict(int))  # word -> thread -> bitmask
    for _, _, th, kind, _, word, bits, _ in every:
        # th is None for an unclassified write; it must not become an owner, or every real write
        # to the word would read it as another thread and report against it.
        if (
            kind == "masked" and th is not None
        ):  # whole-word/unresolved writes tell us nothing about ownership
            owners[word][th] |= bits

    # Vacuity guard: if the tree has headers but essentially no masked writes were recognised,
    # the parser is blind to this tree's write API and a zero result proves nothing.
    n_masked = sum(
        1 for e in every if e[3] == "masked"
    )  # tuple: path,ln,thread,kind,field,...
    if all_h and n_masked == 0:
        print(
            f"REFUSING: parsed {len(all_h)} header(s) but recognised no masked config writes."
        )
        print(
            "  The checker does not understand this tree's config-write API; a zero result would"
        )
        print(
            "  be vacuous. Extend the write-mechanism patterns before trusting any output."
        )
        return 2

    findings, unclassified = [], []
    for path, ln, th, kind, field, word, bits, guarded in collect(
        args.files or all_h, defs
    ):
        if th is None:
            unclassified.append((path, ln, field, word))
            continue
        others = {t: b for t, b in owners[word].items() if t != th and b}
        if not others:
            continue
        if kind == "addr-unresolved":
            findings.append(
                (
                    path,
                    ln,
                    "UNRESOLVED",
                    field,
                    word,
                    "offset is not a compile-time constant; review by hand",
                )
            )
        elif kind == "mask-unresolved":
            findings.append(
                (
                    path,
                    ln,
                    "UNRESOLVED",
                    field,
                    word,
                    "mask could not be resolved; review by hand",
                )
            )
        elif kind == "whole-word":
            # Name who CONSUMES the destroyed bits, not merely who else writes them: a whole-word
            # write destroys every field in the word, and the damage lands on the thread whose
            # instructions read those fields.
            victims = ", ".join(
                f"{t} (bits 0x{b:08x})" for t, b in sorted(others.items())
            )
            harmed = set()
            if consumers is not None:
                for fname, (fw, _fs, _fm) in defs.items():
                    if fw == word:
                        harmed |= {t for t in consumers.get(fname, set()) if t != th}
            extra = (
                f"; other fields in this word are consumed by {', '.join(sorted(harmed))}"
                if harmed
                else ""
            )
            findings.append(
                (
                    path,
                    ln,
                    "CLOBBER",
                    field,
                    word,
                    f"whole-word write from {th} destroys {victims}{extra}",
                )
            )
        elif not guarded:
            overlap = {t: b & bits for t, b in others.items() if b & bits}
            if overlap:
                v = ", ".join(f"{t} (0x{b:08x})" for t, b in sorted(overlap.items()))
                findings.append(
                    (
                        path,
                        ln,
                        "SAME-FIELD",
                        field,
                        word,
                        f"{th} RMWs bits also written by {v}, no mutex::REG_RMW",
                    )
                )

    accepted = set()
    if args.baseline and not args.write_baseline:
        try:
            accepted = {
                l.split("#")[0].strip()
                for l in open(args.baseline)
                if l.strip() and not l.startswith("#")
            }
        except FileNotFoundError:
            pass

    keys = [f"{os.path.relpath(p)}:{k}:{f}" for p, _, k, f, _, _ in findings]
    if args.write_baseline:
        with open(args.baseline, "w") as fh:
            fh.write("# Cross-thread config-register writes accepted for now.\n")
            for k in sorted(set(keys)):
                fh.write(k + "\n")
        print(f"baseline written: {len(set(keys))} site(s)")
        return 0

    new = [(f, k) for f, k in zip(findings, keys) if k not in accepted]
    for (path, ln, kind, field, word, why), _ in new:
        print(f"{path}:{ln}: [{kind}] config word {word} shared across threads")
        print(f"  field: {field}")
        print(f"  {why}")
        if consumers is not None:
            cs = consumers.get(field.replace("_ADDR32", ""))
            print(
                f"  consumed by: {', '.join(sorted(cs))}"
                if cs
                else "  consumed by: NOT RECORDED -- treat as unknown, not safe"
            )
        print(
            "  fix: masked cfg_reg_rmw_tensix for your own bits; for a shared FIELD, hold mutex::REG_RMW\n"
        )
    if unclassified:
        print(
            f"note: {len(unclassified)} config write(s) whose thread could not be determined, "
            "so they were not checked:"
        )
        for path, ln, field, word in unclassified:
            print(f"  {os.path.relpath(path)}:{ln}: {field} (config word {word})")
        print(
            "  These are a coverage gap in this checker, not hazards. thread_of() keys on the\n"
            "  llk_unpack/llk_math/llk_pack file and function naming; a write dispatched by a\n"
            "  ThreadId template parameter, or sitting in a shared header, carries neither.\n"
            "  Review by hand, or teach thread_of() the spelling. Never fails the commit.\n"
        )

    if new:
        print(f"{len(new)} new cross-thread config write(s).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
