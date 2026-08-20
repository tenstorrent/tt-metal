#!/usr/bin/env python3
"""leg_store_gate.py — byte-gate a lane's EDIT leg against the shared
base-leg store WITHOUT recompiling the base leg (laneDA build-infra).

The pin-cycle byte-identity gate compares a lane compiler's corpus output
against the pinned base compiler's output.  The base half is IDENTICAL
for every lane (pinned cc1plus + flag set => byte-identical ELFs), so
lanes should never recompile it.  With a store entry published
(corpus_leg_store.py ensure, or leg_store_seed.py from a completed leg):

    # lane compiles ONLY its own edit leg, then:
    python3 tools/leg_store_gate.py --arch bh --flags "<base leg flags>" \
        --base-compiler <base riscv-tt-elf-g++> \
        --mine <edit leg's tt-llk-build tree>

Exit 0: byte-identical (SAME on every TU, no missing/extra).
Exit 2: differences found (changed/missing/extra listed — the lane's
        attribution work starts from that list).
Exit 1: REFUSED — no verified base entry.  Fail closed: recompile the
        base leg yourself (corpus_leg_store.py ensure publishes it for
        everyone else while you're at it).

Verification before ANY comparison (never trust a store path string):
  * the base cc1plus is resolved through --base-compiler and re-hashed;
    the entry's recorded sha must match (wrong-sha store => refusal);
  * text_hashes.tsv must hash to the value recorded in leg.json
    (tampered entry => refusal);
  * the entry's tt-metal head must equal the head of --tt-metal-home
    (the farm the EDIT leg compiled from — base and edit legs must be
    the same source tree or the diff means nothing);
  * the entry's farm realpath must equal --tt-metal-home's (LLK_PROFILER
    embeds a source-path hash in .text: cross-farm hash comparison is
    invalid and REFUSES here; --allow-cross-farm exists for eyeballing
    only and never exits 0).

`--list` walks the store and prints every entry's provenance summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import corpus_leg_store as store  # noqa: E402

RECOMPILE_HINT = (
    "fail closed: no trustworthy base entry — recompile the base leg "
    "yourself and publish it for the other lanes:\n"
    "    python3 corpus_leg_store.py ensure --arch {arch} --flags '{flags}' \\\n"
    "        --compiler {compiler}\n"
    "or seed it from an already-completed verified leg (tools/leg_store_seed.py)."
)


def refuse(args, msg):
    print(f"leg-store-gate: REFUSED: {msg}", file=sys.stderr)
    print(
        RECOMPILE_HINT.format(
            arch=args.arch, flags=args.flags, compiler=args.base_compiler
        ),
        file=sys.stderr,
    )
    sys.exit(1)


def load_manifest(path):
    rows = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rel, t, e = line.split("\t")
        rows[rel] = (t.removeprefix("text:"), e.removeprefix("elf:"))
    return rows


def hash_mine(paths, objcopy):
    roots = []
    for p in paths:
        p = pathlib.Path(p).resolve()
        if p.name == "tt-llk-build" and p.is_dir():
            roots.append(p)
        elif (p / "tt-llk-build").is_dir():
            roots.append(p / "tt-llk-build")
        else:
            sys.exit(
                f"leg-store-gate: --mine {p} is not a tt-llk-build tree "
                "(or a directory containing one)"
            )
    rows = {}
    for root in roots:
        for elf in sorted(root.rglob("*.elf")):
            if "shared" in elf.parts:
                continue
            rel = str(elf.relative_to(root))
            out = subprocess.run(
                [
                    str(objcopy),
                    "-O",
                    "binary",
                    "--only-section=.text",
                    str(elf),
                    "/dev/stdout",
                ],
                capture_output=True,
            )
            if out.returncode != 0:
                sys.exit(
                    f"leg-store-gate: objcopy failed on {elf}: "
                    f"{out.stderr.decode(errors='replace')[:200]}"
                )
            row = (
                hashlib.sha256(out.stdout).hexdigest(),
                store.sha256_file(elf),
            )
            if rel in rows and rows[rel] != row:
                sys.exit(
                    f"leg-store-gate: --mine roots disagree on {rel} — "
                    "these are not shards of one edit leg"
                )
            rows[rel] = row
    return rows


def cmd_list(store_root):
    root = pathlib.Path(store_root)
    n = 0
    for leg_path in sorted(root.glob("*/*/leg.json")):
        try:
            leg = json.loads(leg_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"{leg_path.parent}  UNREADABLE ({e})")
            continue
        n += 1
        print(
            f"{leg_path.parent}\n"
            f"    cc1plus {leg.get('cc1plus_sha256', '?')[:16]}…  arch {leg.get('arch')}"
            f"  producer {leg.get('producer')}  elfs {leg.get('elf_count')}"
            f"  finished {leg.get('finished')}\n"
            f"    farm {leg.get('tt_metal_home_realpath')} @ {str(leg.get('tt_metal_head'))[:12]}\n"
            f"    flags {str(leg.get('flags'))[:100]}"
            + ("…" if len(str(leg.get("flags"))) > 100 else "")
            + (
                f"\n    seeded_evidence {leg.get('seeded_evidence')}"
                if leg.get("producer") == "seeded"
                else ""
            )
        )
    print(f"{n} entries under {root}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every store entry's provenance summary",
    )
    ap.add_argument("--arch", choices=("bh", "wh"))
    ap.add_argument(
        "--flags", help="exact TT_LLK_EXTRA_COMPILER_OPTIONS of the BASE leg"
    )
    ap.add_argument(
        "--base-compiler",
        type=pathlib.Path,
        help="the BASE g++ driver (default: the installed pin, "
        "~/sfpi-uplift/sfpi/build/sfpi/compiler/bin/riscv-tt-elf-g++)",
    )
    ap.add_argument(
        "--base-cc1plus",
        type=pathlib.Path,
        help="the BASE cc1plus binary itself, for -B-selected bases where "
        "the driver resolves a different cc1plus (the binary is hashed "
        "now; mutually exclusive with --base-compiler)",
    )
    ap.add_argument(
        "--mine",
        action="append",
        help="the EDIT leg's compiled tt-llk-build tree " "(repeatable for shards)",
    )
    ap.add_argument(
        "--mine-manifest",
        type=pathlib.Path,
        help="alternative to --mine: a precomputed manifest in "
        "store format (rel\\ttext:sha\\telf:sha)",
    )
    ap.add_argument(
        "--objcopy",
        type=pathlib.Path,
        help="riscv-tt-elf-objcopy (default: next to " "--base-compiler)",
    )
    ap.add_argument(
        "--tt-metal-home",
        type=pathlib.Path,
        default=HERE.parents[4],
        help="the farm BOTH legs compiled from (default: the "
        "checkout containing this script)",
    )
    ap.add_argument("--store-root", type=pathlib.Path, default=store.DEFAULT_STORE)
    ap.add_argument(
        "--allow-cross-farm",
        action="store_true",
        help="EYEBALLING ONLY: proceed across farm paths; the "
        "verdict is forced to exit 2 (hash equality is not "
        "meaningful cross-farm)",
    )
    args = ap.parse_args(argv)

    if args.list:
        return cmd_list(args.store_root)
    for req in ("arch", "flags"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required (unless --list)")
    if not args.mine and not args.mine_manifest:
        ap.error("pass the edit leg: --mine <tt-llk-build> (or --mine-manifest)")
    if args.base_cc1plus is not None and args.base_compiler is not None:
        ap.error("--base-cc1plus and --base-compiler are mutually exclusive")
    if args.base_compiler is None:
        args.base_compiler = pathlib.Path.home() / (
            "sfpi-uplift/sfpi/build/sfpi/compiler/bin/riscv-tt-elf-g++"
        )
    if args.objcopy is None:
        # objcopy's .text extraction is toolchain-generic; the installed
        # pin's binutils serve every base.
        args.objcopy = args.base_compiler.with_name("riscv-tt-elf-objcopy")
        if not args.objcopy.is_file():
            args.objcopy = pathlib.Path.home() / (
                "sfpi-uplift/sfpi/build/sfpi/compiler/bin/riscv-tt-elf-objcopy"
            )

    # --- derive the base identity and verify the entry (fail closed) ---
    if args.base_cc1plus is not None:
        # -B-selected bases: the driver would resolve the WRONG cc1plus —
        # the caller names the exact binary its -B option selects, and the
        # sha is still derived by hashing that binary NOW (never typed in).
        if not args.base_cc1plus.is_file():
            refuse(args, f"--base-cc1plus {args.base_cc1plus} does not exist")
        cc1_sha = store.sha256_file(args.base_cc1plus)
    else:
        _, cc1_sha = store.resolve_cc1plus(args.base_compiler)
    entry = (
        pathlib.Path(args.store_root)
        / cc1_sha[:12]
        / store.flagset_key(args.arch, args.flags)[:12]
    )
    if not (entry / "leg.json").is_file():
        refuse(
            args,
            f"no base entry for cc1plus {cc1_sha[:12]} + this flag set "
            f"(looked at {entry})",
        )
    farm_real = str(pathlib.Path(args.tt_metal_home).resolve())
    want = {
        "cc1plus_sha256": cc1_sha,
        "flags": args.flags,
        "arch": args.arch,
        "tt_metal_head": store.tt_metal_head(args.tt_metal_home),
        "tt_metal_home_realpath": farm_real,
    }
    ok, detail = store.verify_entry(entry, want)
    if not ok:
        refuse(args, f"{entry}: {detail}")
    cross_farm = bool(detail)  # verify_entry's only warning is the farm path
    if cross_farm:
        if not args.allow_cross_farm:
            refuse(
                args,
                f"{entry}: {detail}\n(the base leg was compiled from a "
                "different farm path — its .text hashes cannot gate this "
                "edit leg)",
            )
        print(f"leg-store-gate: {detail}", file=sys.stderr)

    base = load_manifest(entry / "text_hashes.tsv")
    mine = (
        load_manifest(args.mine_manifest)
        if args.mine_manifest
        else hash_mine(args.mine, args.objcopy)
    )

    changed = sorted(
        rel for rel in base.keys() & mine.keys() if base[rel][0] != mine[rel][0]
    )
    missing = sorted(base.keys() - mine.keys())
    extra = sorted(mine.keys() - base.keys())
    same = len(base.keys() & mine.keys()) - len(changed)

    print(
        f"leg-store-gate: base {entry}\n"
        f"  SAME {same}  CHANGED {len(changed)}  "
        f"MISSING-IN-MINE {len(missing)}  EXTRA-IN-MINE {len(extra)}"
    )
    for rel in changed:
        print(
            f"  CHANGED {rel}\n    base text:{base[rel][0]}\n    mine text:{mine[rel][0]}"
        )
    for rel in missing:
        print(f"  MISSING {rel}")
    for rel in extra:
        print(f"  EXTRA   {rel}")
    if changed or missing or extra or cross_farm:
        print("leg-store-gate: VERDICT DIFF")
        return 2
    print(f"leg-store-gate: VERDICT BYTE-IDENTICAL ({same} ELFs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
