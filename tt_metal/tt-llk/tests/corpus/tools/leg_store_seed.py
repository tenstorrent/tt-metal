#!/usr/bin/env python3
"""leg_store_seed.py — populate the shared base-leg store from a COMPLETED
leg, without recompiling (laneDA build-infra).

corpus_leg_store.py's `ensure` publishes a base leg by compiling it.  But
at every pin cut at least one full base leg has ALREADY been compiled and
verified (the pin's union gates, a lane's base leg) — recompiling it once
more just to populate the store is exactly the waste the store exists to
kill.  This tool adopts such a completed leg into the store:

    python3 tools/leg_store_seed.py \
        --arch bh --flags "<exact TT_LLK_EXTRA_COMPILER_OPTIONS>" \
        --cc1plus <path> --recorded-cc1plus-sha <sha256 from the evidence> \
        --farm <tt-metal checkout the leg compiled from> \
        --build-root <...>/tt-llk-build [--build-root ...] \
        --evidence "<where this leg's provenance is recorded>"

PROVENANCE IS RECHECKED, NEVER TRUSTED (fail-closed):
  * the cc1plus binary at --cc1plus is re-hashed NOW and must equal
    --recorded-cc1plus-sha (the sha written in the leg's evidence).  A
    dead/absent/moved binary refuses — a seed whose toolchain can no
    longer be identified is not evidence, and the store must answer
    "recompile the base leg yourself" instead.
  * every ELF is RE-HASHED from the build tree bytes (objcopy .text +
    full-file sha256) — recorded manifests are never copied into the
    store as truth.
  * the farm head is derived from --farm via git (same +dirty rule as the
    store); --farm-head, when given, must agree or the seed refuses.
  * an existing entry is NEVER overwritten (a second seed of the same key
    must find byte-identical content or refuse loudly).
  * the cc1plus is re-hashed AGAIN after the ELF walk: a toolchain swap
    during the seed window discards the seed (same mid-leg rule as the
    store's compiling path).

The published entry is verify_entry-compatible with corpus_leg_store.py:
consumers (leg_store_gate.py, ensure, manifest, find_build) verify it
with the same fail-closed checks as a compiled entry.  leg.json marks
producer="seeded" with the build roots and the evidence pointer, so a
seeded entry is always distinguishable from a compiled one.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import corpus_leg_store as store  # noqa: E402

SEED_RESULTS_STUB = (
    "id\tstatus\treason\n"
    "__seeded__\tSEEDED_NO_RESULTS\tentry adopted from a completed leg; "
    "row results live in the evidence dir recorded in leg.json\n"
)


def refuse(msg):
    sys.exit(f"leg-store-seed: REFUSED: {msg}")


def find_build_roots(paths):
    """Each --build-root is a tt-llk-build tree, or a directory containing
    exactly one (RUNNER_TEMP layout).  Anything else refuses."""
    roots = []
    for p in paths:
        p = pathlib.Path(p).resolve()
        if not p.is_dir():
            refuse(f"build root {p} is not a directory")
        if p.name == "tt-llk-build":
            roots.append(p)
        elif (p / "tt-llk-build").is_dir():
            roots.append(p / "tt-llk-build")
        else:
            refuse(
                f"{p} is neither a tt-llk-build tree nor a directory "
                "containing one — pass the leg's compiled build tree"
            )
    return roots


def hash_roots(roots, objcopy):
    """Union .text/.elf manifest over the build roots, store hash_build
    format (rel\\ttext:sha\\telf:sha).  Overlapping rel paths must agree
    byte-for-byte (shards of one leg) — disagreement refuses: the roots
    are not shards of the SAME leg."""
    entries = {}
    for root in roots:
        for elf in sorted(root.rglob("*.elf")):
            if "shared" in elf.parts:
                continue  # brisc bootrom scaffolding, flag-independent
            rel = str(elf.relative_to(root))
            text = subprocess.run(
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
            if text.returncode != 0:
                refuse(
                    f"objcopy failed on {elf}: "
                    f"{text.stderr.decode(errors='replace')[:200]}"
                )
            row = (
                hashlib.sha256(text.stdout).hexdigest(),
                store.sha256_file(elf),
            )
            if rel in entries and entries[rel] != row:
                refuse(
                    f"build roots DISAGREE on {rel} — these are not shards "
                    "of one leg; seed each leg separately"
                )
            entries[rel] = row
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", required=True, choices=("bh", "wh"))
    ap.add_argument(
        "--flags", required=True, help="exact TT_LLK_EXTRA_COMPILER_OPTIONS of the leg"
    )
    ap.add_argument(
        "--cc1plus",
        required=True,
        type=pathlib.Path,
        help="the cc1plus binary that compiled the leg (must "
        "still exist; re-hashed and checked)",
    )
    ap.add_argument(
        "--recorded-cc1plus-sha",
        required=True,
        help="sha256 recorded in the leg's evidence — the "
        "provenance anchor this seed is checked against",
    )
    ap.add_argument(
        "--compiler",
        type=pathlib.Path,
        help="the leg's g++ driver (recorded; default: derived "
        "bin/riscv-tt-elf-g++ next to --cc1plus if present)",
    )
    ap.add_argument(
        "--objcopy",
        type=pathlib.Path,
        help="riscv-tt-elf-objcopy for the .text re-hash "
        "(default: next to --compiler)",
    )
    ap.add_argument(
        "--farm",
        required=True,
        type=pathlib.Path,
        help="tt-metal checkout the leg compiled from (.text "
        "hashes are farm-path-dependent)",
    )
    ap.add_argument(
        "--farm-head",
        help="expected farm head (refuses if the derived head "
        "differs); the derived head is recorded either way",
    )
    ap.add_argument(
        "--build-root",
        action="append",
        required=True,
        help="compiled tt-llk-build tree(s) of the leg "
        "(repeatable; shards are unioned, disagreement refuses)",
    )
    ap.add_argument(
        "--results-tsv",
        type=pathlib.Path,
        help="the leg's results.tsv (optional; a marked stub " "is stored otherwise)",
    )
    ap.add_argument(
        "--results-json", type=pathlib.Path, help="the leg's results.json (optional)"
    )
    ap.add_argument(
        "--evidence",
        required=True,
        help="where this leg's provenance is recorded (evidence "
        "dir / review record) — stored in leg.json",
    )
    ap.add_argument("--store-root", type=pathlib.Path, default=store.DEFAULT_STORE)
    args = ap.parse_args(argv)

    # --- provenance recheck (the trust anchor) ---
    if not args.cc1plus.is_file():
        refuse(
            f"cc1plus {args.cc1plus} does not exist — cannot recheck "
            "provenance.  Fail closed: recompile the base leg yourself "
            "(corpus_leg_store.py ensure)"
        )
    got = store.sha256_file(args.cc1plus)
    if got != args.recorded_cc1plus_sha:
        refuse(
            f"cc1plus at {args.cc1plus} hashes to {got} but the evidence "
            f"records {args.recorded_cc1plus_sha} — WRONG BINARY (rebuilt/"
            "swapped since the leg ran?).  Fail closed: recompile the base "
            "leg yourself (corpus_leg_store.py ensure)"
        )
    if args.compiler is None:
        cand = args.cc1plus.parent
        # libexec/gcc/riscv-tt-elf/15.1.0/cc1plus -> ../../../../bin, or a
        # build tree gcc/cc1plus -> xg++ beside it; else require explicit.
        for c in (
            (
                cand.parents[3] / "bin/riscv-tt-elf-g++"
                if len(cand.parents) >= 4
                else None
            ),
            cand / "xg++",
        ):
            if c is not None and c.is_file():
                args.compiler = c
                break
    if args.objcopy is None and args.compiler is not None:
        cand = args.compiler.with_name("riscv-tt-elf-objcopy")
        if cand.is_file():
            args.objcopy = cand
    if args.objcopy is None or not pathlib.Path(args.objcopy).is_file():
        refuse("no usable riscv-tt-elf-objcopy (pass --objcopy)")
    if not (args.farm / ".git").exists():
        refuse(f"--farm {args.farm} is not a git checkout")
    farm_head = store.tt_metal_head(args.farm)
    if args.farm_head and farm_head != args.farm_head:
        refuse(
            f"farm {args.farm} is at {farm_head} but the evidence records "
            f"{args.farm_head} — the checkout moved since the leg ran; "
            "hashes cannot be attributed to it"
        )

    # --- re-hash the completed leg ---
    roots = find_build_roots(args.build_root)
    entries = hash_roots(roots, args.objcopy)
    if not entries:
        refuse("build roots contain ZERO kernel ELFs — nothing to seed")

    # Post-walk recheck: same mid-leg-swap rule as the store's compile path.
    got2 = store.sha256_file(args.cc1plus)
    if got2 != args.recorded_cc1plus_sha:
        refuse(
            f"cc1plus changed DURING the seed ({got2} after the ELF walk) — "
            "discarding"
        )

    entry = (
        pathlib.Path(args.store_root)
        / args.recorded_cc1plus_sha[:12]
        / store.flagset_key(args.arch, args.flags)[:12]
    )
    entry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = entry.parent / (entry.name + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (entry / "leg.json").is_file():
            refuse(
                f"entry already exists: {entry} — never overwritten.  "
                "Compare with leg_store_gate.py; a conflicting seed means "
                "one of the two legs is mislabeled"
            )
        workdir = entry.parent / (entry.name + f".seed.{os.getpid()}")
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir()
        with open(workdir / "text_hashes.tsv", "w") as f:
            for rel in sorted(entries):
                t, e = entries[rel]
                f.write(f"{rel}\ttext:{t}\telf:{e}\n")
        if args.results_tsv:
            shutil.copy2(args.results_tsv, workdir / "results.tsv")
        else:
            (workdir / "results.tsv").write_text(SEED_RESULTS_STUB)
        if args.results_json:
            shutil.copy2(args.results_json, workdir / "results.json")
        else:
            (workdir / "results.json").write_text(
                json.dumps({"seeded": True, "results": "see leg.json seeded_evidence"})
                + "\n"
            )
        leg = {
            "schema": store.SCHEMA,
            "arch": args.arch,
            "flags": args.flags,
            "flagset_sha256": store.flagset_key(args.arch, args.flags),
            "cc1plus": str(args.cc1plus),
            "cc1plus_sha256": args.recorded_cc1plus_sha,
            "compiler": str(args.compiler) if args.compiler else None,
            "compiler_realpath": (
                str(args.compiler.resolve()) if args.compiler else None
            ),
            "compiler_sha256": (
                store.sha256_file(args.compiler) if args.compiler else None
            ),
            "tt_metal_home": str(args.farm),
            "tt_metal_home_realpath": str(args.farm.resolve()),
            "tt_metal_head": farm_head,
            "producer": "seeded",
            "seeded_from": [str(r) for r in roots],
            "seeded_evidence": args.evidence,
            "seeded_by": f"{getpass.getuser()}@{socket.gethostname()}",
            "has_build": False,
            "elf_count": len(entries),
            "text_hashes_sha256": store.sha256_file(workdir / "text_hashes.tsv"),
            "results_tsv_sha256": store.sha256_file(workdir / "results.tsv"),
            "started": None,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        (workdir / "leg.json").write_text(json.dumps(leg, indent=2) + "\n")
        os.rename(workdir, entry)  # atomic publish
    # Consume-path sanity: the entry must verify exactly like a compiled one.
    ok, detail = store.verify_entry(
        entry,
        {
            "cc1plus_sha256": args.recorded_cc1plus_sha,
            "flags": args.flags,
            "arch": args.arch,
            "tt_metal_head": farm_head,
            "tt_metal_home_realpath": str(args.farm.resolve()),
        },
    )
    if not ok:
        refuse(f"freshly seeded entry FAILS verification ({detail}) — bug")
    print(f"SEEDED {entry} ({len(entries)} ELFs, producer=seeded)")
    print(f"ENTRY {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
