#!/usr/bin/env python3
"""Self-test for leg_store_seed.py + leg_store_gate.py (laneDA).

Same fixture technique as selftest_corpus_leg_store.py: a fake toolchain
(driver answering -print-prog-name=cc1plus with a fixture binary, objcopy
that cats the ELF bytes as .text) and a fixture git repo as the farm — the
REAL CLIs run end to end (real hashing, real flock, real atomic publish).

Proves (store-integrity selftest of the deliverable):
  1. seed of a completed leg publishes a verify_entry-clean entry
     (producer=seeded, evidence recorded);
  2. gate on a byte-identical edit leg exits 0 (BYTE-IDENTICAL);
  3. gate on an edited leg exits 2 and names exactly the changed TU;
  4. TAMPERED ENTRY DETECTED: edit text_hashes.tsv after publish ->
     gate refuses (exit 1) naming TAMPER, with the recompile-yourself hint;
  5. WRONG-SHA STORE REFUSED: a base compiler whose cc1plus differs from
     the seeded sha finds no entry -> exit 1, fail-closed hint (and a
     hand-forged entry under the wrong key refuses on the sha mismatch);
  6. seed REFUSES a wrong recorded sha (provenance recheck);
  7. seed REFUSES to overwrite an existing entry;
  8. gate REFUSES cross-farm comparison (farm-path-dependent hashes);
  9. missing/extra ELFs in the edit leg are reported and exit 2.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SEED = HERE / "leg_store_seed.py"
GATE = HERE / "leg_store_gate.py"
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


def run(*argv, **kw):
    return subprocess.run(
        [sys.executable, *map(str, argv)], capture_output=True, text=True, **kw
    )


def fake_toolchain(td, name, cc1_bytes):
    bin_dir = td / name / "bin"
    libexec = td / name / "libexec/gcc/riscv-tt-elf/15.1.0"
    bin_dir.mkdir(parents=True)
    libexec.mkdir(parents=True)
    cc1plus = libexec / "cc1plus"
    cc1plus.write_bytes(cc1_bytes)
    gxx = bin_dir / "riscv-tt-elf-g++"
    gxx.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-print-prog-name=cc1plus" ]; then\n'
        f'  echo "{cc1plus}"\n  exit 0\nfi\nexit 0\n'
    )
    gxx.chmod(0o755)
    objcopy = bin_dir / "riscv-tt-elf-objcopy"
    objcopy.write_text(
        '#!/bin/sh\ncat "$4"\n'
    )  # -O binary --only-section=.text ELF /dev/stdout
    objcopy.chmod(0o755)
    return gxx, cc1plus


def fake_repo(td, name):
    repo = td / name
    (repo / "tt_metal/tt-llk").mkdir(parents=True)
    (repo / "tt_metal/tt-llk/kernel.h").write_text(f"// fixture {name}\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "f"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return repo


def make_leg(td, name, elves):
    build = td / name / "tt-llk-build"
    for rel, payload in elves.items():
        p = build / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
    return build


def sha256(b):
    import hashlib

    return hashlib.sha256(b).hexdigest()


def main():
    td = pathlib.Path(tempfile.mkdtemp(prefix="leg-store-tools-selftest."))
    try:
        gxx, cc1plus = fake_toolchain(td, "tc-base", b"BASE CC1PLUS v1\n")
        base_sha = sha256(b"BASE CC1PLUS v1\n")
        farm = fake_repo(td, "farm")
        store_root = td / "store"
        ELVES = {
            "sources/a.cpp/k1/elf/math.elf": b"AAA-math-text",
            "sources/b.cpp/k2/elf/math.elf": b"BBB-math-text",
            "shared/skip.elf": b"scaffolding",  # must be skipped
        }
        base_leg = make_leg(td, "completed-base-leg", ELVES)
        FLAGS = "-mno-x -mno-y"

        common = [
            "--arch",
            "bh",
            "--flags",
            FLAGS,
            "--cc1plus",
            cc1plus,
            "--recorded-cc1plus-sha",
            base_sha,
            "--compiler",
            gxx,
            "--farm",
            farm,
            "--build-root",
            base_leg.parent,
            "--evidence",
            "selftest-fixture",
            "--store-root",
            store_root,
        ]

        # 1. seed publishes a verified entry
        r = run(SEED, *common)
        check(
            "seed publishes",
            r.returncode == 0 and "SEEDED" in r.stdout,
            r.stdout + r.stderr,
        )
        entries = list(store_root.glob("*/*/leg.json"))
        check("one entry exists", len(entries) == 1, str(entries))
        leg = json.loads(entries[0].read_text()) if entries else {}
        check(
            "entry marked seeded",
            leg.get("producer") == "seeded"
            and leg.get("seeded_evidence") == "selftest-fixture"
            and leg.get("elf_count") == 2,
            str(leg)[:200],
        )

        gate_common = [
            "--arch",
            "bh",
            "--flags",
            FLAGS,
            "--base-compiler",
            gxx,
            "--tt-metal-home",
            farm,
            "--store-root",
            store_root,
        ]

        # 2. identical edit leg -> exit 0
        same_leg = make_leg(td, "edit-same", ELVES)
        r = run(GATE, *gate_common, "--mine", same_leg)
        check(
            "gate identical exits 0",
            r.returncode == 0 and "BYTE-IDENTICAL" in r.stdout,
            f"rc={r.returncode} {r.stdout}{r.stderr}",
        )

        # 3. one changed ELF -> exit 2, named
        diff = dict(ELVES)
        diff["sources/b.cpp/k2/elf/math.elf"] = b"BBB-CHANGED"
        diff_leg = make_leg(td, "edit-diff", diff)
        r = run(GATE, *gate_common, "--mine", diff_leg)
        check(
            "gate diff exits 2",
            r.returncode == 2
            and "CHANGED sources/b.cpp/k2/elf/math.elf" in r.stdout
            and "CHANGED 1" in r.stdout,
            f"rc={r.returncode} {r.stdout}",
        )

        # 9. missing + extra -> exit 2, reported
        me = dict(ELVES)
        del me["sources/a.cpp/k1/elf/math.elf"]
        me["sources/c.cpp/k3/elf/math.elf"] = b"CCC-new"
        me_leg = make_leg(td, "edit-missing-extra", me)
        r = run(GATE, *gate_common, "--mine", me_leg)
        check(
            "gate missing/extra exits 2",
            r.returncode == 2
            and "MISSING sources/a.cpp/k1/elf/math.elf" in r.stdout
            and "EXTRA   sources/c.cpp/k3/elf/math.elf" in r.stdout,
            f"rc={r.returncode} {r.stdout}",
        )

        # 4. tampered manifest -> refusal with hint
        manifest = entries[0].parent / "text_hashes.tsv"
        orig = manifest.read_text()
        manifest.write_text(
            orig.replace("AAA", "XXX") if "AAA" in orig else orig + "#t\n"
        )
        # (tamper the bytes regardless of content shape)
        manifest.write_text(orig + "tampered\ttext:00\telf:00\n")
        r = run(GATE, *gate_common, "--mine", same_leg)
        check(
            "tampered entry refused",
            r.returncode == 1
            and "TAMPER" in r.stderr
            and "recompile the base leg yourself" in r.stderr,
            f"rc={r.returncode} {r.stderr[:300]}",
        )
        manifest.write_text(orig)  # restore for later cases

        # 5. wrong-sha store refused (different base compiler -> no entry)
        gxx2, _ = fake_toolchain(td, "tc-other", b"OTHER CC1PLUS v2\n")
        r = run(
            GATE,
            "--arch",
            "bh",
            "--flags",
            FLAGS,
            "--base-compiler",
            gxx2,
            "--tt-metal-home",
            farm,
            "--store-root",
            store_root,
            "--mine",
            same_leg,
        )
        check(
            "wrong compiler finds no entry (fail closed)",
            r.returncode == 1
            and "no base entry" in r.stderr
            and "recompile the base leg yourself" in r.stderr,
            f"rc={r.returncode} {r.stderr[:300]}",
        )
        # ... and a FORGED entry under that compiler's key refuses on sha.
        other_sha = sha256(b"OTHER CC1PLUS v2\n")
        forged = store_root / other_sha[:12] / entries[0].parent.name
        shutil.copytree(entries[0].parent, forged)
        r = run(
            GATE,
            "--arch",
            "bh",
            "--flags",
            FLAGS,
            "--base-compiler",
            gxx2,
            "--tt-metal-home",
            farm,
            "--store-root",
            store_root,
            "--mine",
            same_leg,
        )
        check(
            "forged wrong-sha entry refused",
            r.returncode == 1 and "SHA/IDENTITY MISMATCH" in r.stderr,
            f"rc={r.returncode} {r.stderr[:300]}",
        )
        shutil.rmtree(forged)

        # 6. seed refuses a wrong recorded sha
        r = run(
            SEED,
            "--arch",
            "bh",
            "--flags=-other",
            "--cc1plus",
            cc1plus,
            "--recorded-cc1plus-sha",
            "0" * 64,
            "--compiler",
            gxx,
            "--farm",
            farm,
            "--build-root",
            base_leg.parent,
            "--evidence",
            "x",
            "--store-root",
            store_root,
        )
        check(
            "seed wrong recorded sha refused",
            r.returncode != 0 and "WRONG BINARY" in r.stderr,
            f"rc={r.returncode} {r.stderr[:300]}",
        )

        # 7. seed never overwrites
        r = run(SEED, *common)
        check(
            "seed refuses overwrite",
            r.returncode != 0 and "never overwritten" in r.stderr,
            f"rc={r.returncode} {r.stderr[:300]}",
        )

        # 8. cross-farm refused
        farm2 = fake_repo(td, "farm2")
        (farm2 / "tt_metal/tt-llk/kernel.h").write_text("// fixture farm\n")
        r = run(
            GATE,
            "--arch",
            "bh",
            "--flags",
            FLAGS,
            "--base-compiler",
            gxx,
            "--tt-metal-home",
            farm2,
            "--store-root",
            store_root,
            "--mine",
            same_leg,
        )
        check(
            "cross-farm refused",
            r.returncode == 1,
            f"rc={r.returncode} {r.stderr[:300]}",
        )

        # 10. text-only seed (evidence retains only .text payloads) + gate
        #     via --base-cc1plus (a -B-selected base has no resolving driver)
        text_root = td / "text-evidence/text"
        for rel, payload in ELVES.items():
            p = text_root / (rel + ".text.bin")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
        r = run(
            SEED,
            "--arch",
            "bh",
            "--flags",
            "-mno-a -Bsomewhere/",
            "--cc1plus",
            cc1plus,
            "--recorded-cc1plus-sha",
            base_sha,
            "--compiler",
            gxx,
            "--farm",
            farm,
            "--text-root",
            text_root,
            "--evidence",
            "selftest-text-fixture",
            "--store-root",
            store_root,
        )
        check(
            "text-only seed publishes",
            r.returncode == 0 and "SEEDED" in r.stdout,
            r.stdout + r.stderr,
        )
        r = run(
            GATE,
            "--arch",
            "bh",
            "--flags",
            "-mno-a -Bsomewhere/",
            "--base-cc1plus",
            cc1plus,
            "--objcopy",
            gxx.with_name("riscv-tt-elf-objcopy"),
            "--tt-metal-home",
            farm,
            "--store-root",
            store_root,
            "--mine",
            same_leg,
        )
        check(
            "gate via --base-cc1plus on text-only entry exits 0",
            r.returncode == 0 and "BYTE-IDENTICAL" in r.stdout,
            f"rc={r.returncode} {r.stdout}{r.stderr}",
        )

        # --list smoke
        r = run(GATE, "--list", "--store-root", store_root)
        check(
            "--list shows the seeded entry",
            r.returncode == 0
            and "producer seeded" in r.stdout
            and "selftest-fixture" in r.stdout,
            r.stdout[:300],
        )
    finally:
        shutil.rmtree(td, ignore_errors=True)

    if FAILS:
        print(f"\nleg-store-tools selftest: {len(FAILS)} FAILURE(S): {FAILS}")
        return 1
    print("\nleg-store-tools selftest: ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
