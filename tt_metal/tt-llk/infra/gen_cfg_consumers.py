#!/usr/bin/env python3
"""Generate the vendored config-consumer table from the Tensix hazard database.

Which THREAD's instructions consume a config field is a hardware fact. It is not derivable from
the LLK source, and the ISA pages are incomplete (notably for Blackhole), so it is taken from the
hazard database's RTL-derived reader rows.

The database is not available to CI, so the facts are vendored here as a small data file. Run this
when the database changes:

    python3 gen_cfg_consumers.py --db /path/to/tensix-hazard-db --out cfg_consumers.yaml

A field ABSENT from the table means "no reader row recorded upstream" -- i.e. UNKNOWN, not "nobody
consumes it". The checker reports it that way.
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import date

# reader phrase -> Tensix thread that issues the instructions doing the reading
ALL_THREADS = ["MATH", "PACK", "UNPACK"]

# Reader phrase -> consuming thread(s). Ordered; first match wins.
#
# Three categories beyond the obvious units, each of which was silently dropped in the first
# version and together accounted for ~half of the database's reader rows:
#
#  * THREAD-RELATIVE readers ("the issuing thread's instruction issue stage", the stall unit, the
#    MOP/replay expanders, the sync unit and wait gate). These consume config on behalf of
#    WHICHEVER thread issued. The register is a single shared copy, so a write by one thread is
#    read by any of them -> all three threads.
#  * DEST PATH readers. Dest is written by the Matrix Unit and read out on the pack path.
#  * RISC / JTAG debug readers. Not a Tensix thread; recorded as RISC so they are not silently
#    counted as one of the three.
READER_THREADS = [
    ("unpack front-end", ["UNPACK"]),
    ("unpacker", ["UNPACK"]),
    ("matrix unit", ["MATH"]),
    ("matrix-unit", ["MATH"]),
    (
        "matrixunit",
        ["MATH"],
    ),  # Blackhole spells it unseparated; without this the arch scores 0
    ("vectorunit", ["MATH"]),
    ("vector unit", ["MATH"]),
    ("sfpu", ["MATH"]),  # SFPU instructions are issued by the math thread
    ("packer", ["PACK"]),  # includes "the Dest-read output stage that feeds the packer"
    ("pack path", ["PACK"]),
    ("dest read", ["MATH", "PACK"]),
    ("dest bank write", ["MATH", "PACK"]),
    ("jtag", ["RISC"]),
    ("risc", ["RISC"]),
    ("instruction-stall unit", ALL_THREADS),
    ("issue stage", ALL_THREADS),
    ("expander", ALL_THREADS),
    ("replay", ALL_THREADS),
    ("sync unit", ALL_THREADS),
    ("sync execution unit", ALL_THREADS),
    ("stall latch", ALL_THREADS),
    ("wait gate", ALL_THREADS),
    ("stalling thread", ALL_THREADS),
    ("blocking sync", ALL_THREADS),
    ("execution units of the thread", ALL_THREADS),
]
FIELD = re.compile(r"\b([A-Z][A-Za-z_0-9]{3,})\b")
# The architectures spell the same field three ways. Wormhole uses the cfg_defines name
# (ALU_ACC_CTRL_SFPU_Fp32_enabled); Blackhole and Quasar use Group::Field in CamelCase
# (AluAccCtrl::SFPU_Fp32_enabled). Normalise to the cfg_defines spelling or the table keys join
# to nothing -- which is how Blackhole silently produced zero fields from two real reader rows.
QUALIFIED = re.compile(r"\b([A-Z][A-Za-z0-9]+)::([A-Za-z_0-9]+)")


def camel_to_upper(name):
    # Split before capitals only, NOT before digits: AluFormatSpecReg0 -> ALU_FORMAT_SPEC_REG0,
    # matching the cfg_defines spelling. Splitting on digits yields ..._REG_0_... which joins
    # to nothing.
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper().replace("__", "_")


# A cfg_defines-style name: upper-case with at least one underscore (DISABLE_IMPLIED_SRCA_FMT_Base).
PLAIN = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Za-z0-9]+)+)\b")


def fields_in(res):
    """Field names in a resource string, normalised to cfg_defines spelling.

    A resource list can MIX both spellings, so collect qualified and plain names together --
    taking one or the other silently dropped entries like DISABLE_IMPLIED_SRCA_FMT_Base.
    Unknown names are filtered later against cfg_defines, so over-collecting here is safe.
    """
    out = {f"{camel_to_upper(g)}_{f}" for g, f in QUALIFIED.findall(res)}
    out |= set(PLAIN.findall(QUALIFIED.sub(" ", res)))
    return out or set(FIELD.findall(res))


def rows_of(doc):
    """Yield dicts with reader/resource, tolerating both database schemas."""
    stack = [doc]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if "reader" in n:
                yield n
            stack += list(n.values())
        elif isinstance(n, list):
            stack += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--defines", help="cfg_defines.h glob with @ARCH@ placeholder, for self-check"
    )
    a = ap.parse_args()
    import yaml

    try:
        commit = subprocess.run(
            ["git", "-C", a.db, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    out, stats = {}, {}
    for arch in ("wormhole_b0", "blackhole", "quasar"):
        # Walk EVERY file in the arch directory. Reader rows live in packer-config,
        # unpack-pack-config, dest, wait-conditions, addrmod-counters and others -- reading only
        # alu-control.yaml covered 22 of Wormhole's 157 reader rows and made the table look like
        # it only knew about ALU-space fields.
        import glob as _glob

        srcs = sorted(_glob.glob(os.path.join(a.db, arch, "*.yaml")))
        if not srcs:
            continue
        table, nrows = {}, 0
        rows = []
        for src in srcs:
            try:
                rows += list(rows_of(yaml.safe_load(open(src))))
            except Exception as e:
                print(
                    f"  WARNING {arch}/{os.path.basename(src)}: unparsed ({e.__class__.__name__})"
                )
        for r in rows:
            reader = str(r.get("reader") or "")
            # Wormhole rows use `resource:` (a string); Blackhole/Quasar use `resources:` (a list).
            raw = r.get("resource") or r.get("resources") or ""
            res = " ".join(map(str, raw)) if isinstance(raw, list) else str(raw)
            if not reader or reader.startswith("n/a") or reader == "none":
                continue
            low = reader.lower()
            threads = sorted({t for k, ts in READER_THREADS if k in low for t in ts})
            if not threads:
                continue
            nrows += 1
            for f in fields_in(res):
                table.setdefault(f, set()).update(threads)
        out[arch] = {k: sorted(v) for k, v in sorted(table.items())}
        stats[arch] = (nrows, len(table))

    with open(a.out, "w") as f:
        f.write(
            "# Config-field consumers: which Tensix thread's instructions READ each field.\n"
        )
        f.write(
            f"# GENERATED from the Tensix hazard database ({commit}) on {date.today()}.\n"
        )
        f.write("# Do not hand-edit -- rerun infra/gen_cfg_consumers.py.\n#\n")
        f.write(
            "# A field ABSENT here has no reader row recorded upstream: that is UNKNOWN, not\n"
        )
        f.write(
            "# 'unconsumed'. Coverage is very uneven between architectures -- see the counts below.\n#\n"
        )
        for arch, (nrows, nf) in stats.items():
            f.write(f"#   {arch:14s} {nrows:3d} reader rows -> {nf:3d} fields\n")
        f.write("\n")
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=True)
    for arch, (nrows, nf) in stats.items():
        print(f"{arch:14s} {nrows:3d} reader rows -> {nf:3d} fields")
    print(f"written: {a.out}")

    # Self-check: every generated key must exist in that architecture's cfg_defines.h, or the
    # normalisation is wrong and the table joins to nothing at check time.
    if a.defines:
        import glob

        for arch, table in out.items():
            # layouts differ: wormhole/wormhole_b0_defines/cfg_defines.h vs blackhole/cfg_defines.h
            cand = glob.glob(
                a.defines.replace("@ARCH@", arch.replace("_b0", "")), recursive=True
            )
            if not cand:
                print(
                    f"  {arch}: NOT VERIFIED (no cfg_defines.h found) -- keys are unchecked"
                )
                continue
            names = set()
            for line in open(cand[0], errors="ignore"):
                m = re.match(r"^#define\s+(\w+?)_ADDR32\s", line)
                if m:
                    names.add(m.group(1))
            # The database names resource GROUPS (ADDR_MOD_AB, PCK_EDGE_OFFSET_SEC0) where
            # cfg_defines names individual FIELDS (ADDR_MOD_AB_SEC0_SrcAIncr). Expand a group to
            # every field it prefixes; exact-match alone discarded them as unknown tokens.
            unknown, expanded = [], 0
            for k in sorted(table):
                if k in names:
                    continue
                # SEC-numbered groups: the database says ADDR_MOD_AB_SEC, the fields are
                # ADDR_MOD_AB_SEC0_*, so a "k + _" prefix misses them.
                kids = [n for n in names if n.startswith(k + "_")] or [
                    n for n in names if n.startswith(k)
                ]
                if kids:
                    for n in kids:  # values are lists at this point, not sets
                        table[n] = sorted(set(table.get(n, [])) | set(table[k]))
                    expanded += len(kids)
                else:
                    unknown.append(k)
                table.pop(k, None)
            print(
                f"  {arch}: {len(table)} fields ({expanded} via group expansion)"
                + (
                    f", {len(unknown)} non-field token(s) dropped: {unknown[:4]}"
                    if unknown
                    else ""
                )
            )
        # rewrite with the verified tables only
        with open(a.out) as fh:
            head = "".join(l for l in fh if l.startswith("#") or not l.strip())
        with open(a.out, "w") as fh:
            fh.write(head)
            yaml.safe_dump(out, fh, default_flow_style=False, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
