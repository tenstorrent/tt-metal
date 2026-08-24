#!/usr/bin/env python3
"""Self-test for corpus_leg_store.py (shared base-leg artifact store).

Drives the REAL CLI (subprocess, real flock, real publish path) with a
FAKE toolchain + FAKE producer — no compiler, no corpus compile:

  * fake riscv-tt-elf-g++ answers -print-prog-name=cc1plus with a fixture
    binary (so the real cc1plus-resolution and sha256 pinning code runs);
  * fake riscv-tt-elf-objcopy emits the ELF bytes as the .text section
    (so the real manifest hashing runs);
  * the producer is a shell stub (--producer-cmd, recorded as
    producer=custom) that sleeps, then populates $LEG_RUN_ROOT/results.*
    and $LEG_RUNNER_TEMP/tt-llk-build/... and appends to a compile counter.

Proves:
  1. ensure on an empty store compiles once and publishes a verifiable
     entry (leg.json + text_hashes.tsv + results.tsv, shas recorded);
  2. TWO CONCURRENT ensure calls produce EXACTLY ONE compile — the loser
     waits on the flock and consumes the winner's entry (both exit 0, same
     ENTRY path, compile counter == 1);
  3. a second ensure is a verified HIT (no compile);
  4. manifest prints the stored .text manifest after verification;
  5. INTEGRITY: a tampered leg.json cc1plus sha REFUSES (exit nonzero,
     mismatch named); a tampered text_hashes.tsv REFUSES (TAMPER);
  6. a DIFFERENT tt-metal head refuses the entry (consumers verify the
     source-tree identity, not just the toolchain);
  7. --no-store compiles into --run-root and leaves the store untouched.
  8. FZ-F1 (lane GF): the store key is derived from the compiler the leg
     will ACTUALLY invoke (resolved under the --compiler-derived
     GCC_EXEC_PREFIX build env, the exact env build_leg pins), and a
     DIVERGENCE between that resolution and the caller-env resolution
     REFUSES LOUDLY before any compile — the lane-FZ poisoned-key
     fail-open (hybrid shell GCC_EXEC_PREFIX + default --compiler stored
     PIN bytes under the HYBRID's key) is structurally impossible; the
     sound hybrid workflow (--compiler <hybrid driver>) publishes under
     the hybrid cc1plus key.

Run with the other gate self-tests; exit 0 all green.
"""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
STORE_PY = HERE / "corpus_leg_store.py"
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


def write_fake_toolchain(td):
    bin_dir = td / "toolchain/compiler/bin"
    libexec = td / "toolchain/compiler/libexec"
    bin_dir.mkdir(parents=True)
    libexec.mkdir(parents=True)
    cc1plus = libexec / "cc1plus"
    cc1plus.write_bytes(b"FIXTURE CC1PLUS BINARY v1\n")
    gxx = bin_dir / "riscv-tt-elf-g++"
    gxx.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-print-prog-name=cc1plus" ]; then\n'
        f'  echo "{cc1plus}"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    gxx.chmod(0o755)
    objcopy = bin_dir / "riscv-tt-elf-objcopy"
    # Real call shape: objcopy -O binary --only-section=.text <elf> /dev/stdout
    objcopy.write_text('#!/bin/sh\ncat "$4"\n')
    objcopy.chmod(0o755)
    return gxx, cc1plus


def write_fake_repo(td):
    repo = td / "repo"
    (repo / "tt_metal/tt-llk").mkdir(parents=True)
    (repo / "tt_metal/tt-llk/kernel.h").write_text("// fixture\n")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.email=selftest@fixture",
            "-c",
            "user.name=selftest",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


PRODUCER = r"""
sleep 1
mkdir -p "$LEG_RUN_ROOT" "$LEG_RUNNER_TEMP/tt-llk-build/sources/fix.cpp/aa/elf" \
         "$LEG_RUNNER_TEMP/tt-llk-build/shared/elf"
printf 'id\tstatus\nfixture_row\tPASS\n' > "$LEG_RUN_ROOT/results.tsv"
printf '{"rows": 1}\n' > "$LEG_RUN_ROOT/results.json"
printf 'FAKE MATH ELF CONTENT' > "$LEG_RUNNER_TEMP/tt-llk-build/sources/fix.cpp/aa/elf/math.elf"
printf 'SHARED SCAFFOLD' > "$LEG_RUNNER_TEMP/tt-llk-build/shared/elf/brisc.elf"
echo x >> "$COUNTER"
"""


def run_store(args, env):
    return subprocess.run(
        [sys.executable, str(STORE_PY), *args],
        capture_output=True,
        text=True,
        env=env,
    )


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    gxx, cc1plus = write_fake_toolchain(td)
    repo = write_fake_repo(td)
    store = td / "store"
    counter = td / "compiles.count"
    env = dict(os.environ, COUNTER=str(counter))
    base_args = [
        # NOTE the --flags=<value> form: a single-token value starting with
        # '-' needs it (argparse); multi-token real flag strings don't.
        "--flags=-mfixture-flag-set",
        "--arch",
        "bh",
        "--store-root",
        str(store),
        "--tt-metal-home",
        str(repo),
        "--compiler",
        str(gxx),
        "--producer-cmd",
        PRODUCER,
    ]

    # 2. two CONCURRENT ensure calls -> exactly one compile
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(run_store, ["ensure", *base_args], env) for _ in range(2)]
        r1, r2 = [f.result() for f in futs]
    check(
        "concurrent ensure: both exit 0",
        r1.returncode == 0 and r2.returncode == 0,
        f"rc1={r1.returncode} rc2={r2.returncode}\n{r1.stderr}\n{r2.stderr}",
    )
    e1 = [x for x in r1.stdout.splitlines() if x.startswith("ENTRY ")]
    e2 = [x for x in r2.stdout.splitlines() if x.startswith("ENTRY ")]
    check(
        "concurrent ensure: same ENTRY path",
        e1 and e2 and e1[-1] == e2[-1],
        f"{e1} vs {e2}",
    )
    compiles = len(counter.read_text().splitlines()) if counter.is_file() else 0
    check("concurrent ensure: EXACTLY ONE compile", compiles == 1, compiles)

    entry = pathlib.Path(e1[-1].split(" ", 1)[1])
    leg = json.loads((entry / "leg.json").read_text())
    check(
        "published entry is complete and self-describing",
        (entry / "text_hashes.tsv").is_file()
        and (entry / "results.tsv").is_file()
        and leg["producer"] == "custom"
        and leg["flags"] == "-mfixture-flag-set"
        and len(leg["cc1plus_sha256"]) == 64
        and leg["elf_count"] == 1,  # shared/ scaffolding excluded
        leg,
    )
    check(
        "manifest excludes shared/ scaffolding and hashes .text",
        "math.elf\ttext:" in (entry / "text_hashes.tsv").read_text()
        and "brisc" not in (entry / "text_hashes.tsv").read_text(),
    )

    # 3. re-ensure is a verified HIT: no new compile
    r = run_store(["ensure", *base_args], env)
    compiles = len(counter.read_text().splitlines())
    check(
        "second ensure is a verified HIT (no compile)",
        r.returncode == 0 and compiles == 1 and "HIT (verified)" in r.stderr,
        r.stderr,
    )

    # 4. manifest prints the stored manifest after verification
    r = run_store(["manifest", *base_args], env)
    check(
        "manifest prints the verified .text manifest",
        r.returncode == 0 and "math.elf\ttext:" in r.stdout,
        r.stdout + r.stderr,
    )

    # 5a. tampered leg.json cc1plus sha -> REFUSE
    orig = (entry / "leg.json").read_text()
    (entry / "leg.json").write_text(orig.replace(leg["cc1plus_sha256"], "ab" * 32))
    r = run_store(["ensure", *base_args], env)
    check(
        "tampered cc1plus sha REFUSES (sha mismatch names the key)",
        r.returncode != 0 and "MISMATCH" in (r.stderr + r.stdout),
        r.stderr,
    )
    (entry / "leg.json").write_text(orig)

    # 5b. tampered manifest content -> REFUSE (TAMPER)
    man_orig = (entry / "text_hashes.tsv").read_text()
    (entry / "text_hashes.tsv").write_text(man_orig + "evil\ttext:00\telf:00\n")
    r = run_store(["manifest", *base_args], env)
    check(
        "tampered text_hashes.tsv REFUSES (TAMPER)",
        r.returncode != 0 and "TAMPER" in (r.stderr + r.stdout),
        r.stderr,
    )
    (entry / "text_hashes.tsv").write_text(man_orig)

    # 6. a different tt-metal head refuses (source identity is verified)
    (repo / "tt_metal/tt-llk/kernel.h").write_text("// EDITED fixture\n")
    r = run_store(["ensure", *base_args], env)
    check(
        "changed tt-metal head REFUSES the stored entry",
        r.returncode != 0 and "tt_metal_head" in (r.stderr + r.stdout),
        r.stderr,
    )
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, check=True)

    # 7. --no-store compiles into --run-root, store untouched
    before = sorted(p.name for p in store.rglob("*"))
    rr = td / "isolated"
    r = run_store(["ensure", *base_args, "--no-store", "--run-root", str(rr)], env)
    after = sorted(p.name for p in store.rglob("*"))
    compiles = len(counter.read_text().splitlines())
    check(
        "--no-store compiles into --run-root and leaves the store untouched",
        r.returncode == 0
        and (rr / "leg/leg.json").is_file()
        and before == after
        and compiles == 2,
        f"rc={r.returncode} compiles={compiles}\n{r.stderr}",
    )


# ---------------- 8. FZ-F1 poisoned-key fail-open (lane GF) ----------------
def write_gep_toolchain(root, name, cc1_bytes):
    """A fake toolchain whose driver mimics the REAL one's GCC_EXEC_PREFIX
    semantics: with GCC_EXEC_PREFIX set and ${GCC_EXEC_PREFIX}cc1plus
    present it resolves THERE; otherwise it falls back to its own install's
    lib/gcc/cc1plus.  cc1plus staged at <root>/<name>/compiler/lib/gcc/
    (== build_gcc_exec_prefix of the driver realpath)."""
    comp = root / name / "compiler"
    (comp / "bin").mkdir(parents=True)
    libgcc = comp / "lib/gcc"
    libgcc.mkdir(parents=True)
    cc1 = libgcc / "cc1plus"
    cc1.write_bytes(cc1_bytes)
    gxx = comp / "bin/riscv-tt-elf-g++"
    gxx.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-print-prog-name=cc1plus" ]; then\n'
        '  if [ -n "${GCC_EXEC_PREFIX:-}" ] && [ -f "${GCC_EXEC_PREFIX}cc1plus" ]; then\n'
        '    echo "${GCC_EXEC_PREFIX}cc1plus"\n'
        "  else\n"
        f'    echo "{cc1}"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    gxx.chmod(0o755)
    objcopy = comp / "bin/riscv-tt-elf-objcopy"
    objcopy.write_text('#!/bin/sh\ncat "$4"\n')
    objcopy.chmod(0o755)
    return gxx, cc1


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    pin_gxx, pin_cc1 = write_gep_toolchain(td, "pin", b"PIN CC1PLUS BYTES v1\n")
    hyb_gxx, hyb_cc1 = write_gep_toolchain(td, "hybrid", b"HYBRID CC1PLUS BYTES v1\n")
    repo = write_fake_repo(td)
    store = td / "store"
    counter = td / "compiles.count"
    fz_args = [
        "--flags=-mfz-fixture-flags",
        "--arch",
        "bh",
        "--store-root",
        str(store),
        "--tt-metal-home",
        str(repo),
        "--producer-cmd",
        PRODUCER,
    ]

    # 8a. THE FZ INCIDENT SHAPE: caller shell exports the HYBRID's
    # GCC_EXEC_PREFIX but --compiler is the PIN driver (the checkout
    # default).  The caller-env resolution says hybrid cc1plus; the leg
    # would COMPILE with the pin (build_leg rewrites GCC_EXEC_PREFIX from
    # the --compiler realpath).  Must REFUSE loudly BEFORE compiling —
    # without the fix this published PIN bytes under the HYBRID key.
    env_poison = dict(
        os.environ,
        COUNTER=str(counter),
        GCC_EXEC_PREFIX=str(hyb_cc1.parent) + "/",
    )
    r = run_store(["ensure", *fz_args, "--compiler", str(pin_gxx)], env_poison)
    compiles = len(counter.read_text().splitlines()) if counter.is_file() else 0
    check(
        "FZ-F1: caller-env vs build-env cc1plus divergence REFUSES loudly "
        "before any compile (poisoned-key fail-open closed)",
        r.returncode != 0
        and "DIVERGENCE" in (r.stderr + r.stdout)
        and "--compiler" in (r.stderr + r.stdout)
        and compiles == 0
        and not (store / "leg.json").exists()
        and not any(store.rglob("leg.json")),
        f"rc={r.returncode} compiles={compiles}\n{r.stderr[:800]}",
    )

    # 8b. The SOUND hybrid workflow: --compiler <hybrid driver>.  Publishes
    # under the HYBRID cc1plus key (the compiler the leg actually invokes),
    # caller GCC_EXEC_PREFIX present or not.
    r = run_store(["ensure", *fz_args, "--compiler", str(hyb_gxx)], env_poison)
    entries = sorted(store.rglob("leg.json"))
    hyb_sha = __import__("hashlib").sha256(hyb_cc1.read_bytes()).hexdigest()
    leg = json.loads(entries[0].read_text()) if entries else {}
    check(
        "FZ-F1: hybrid leg WITH --compiler <hybrid driver> publishes under "
        "the hybrid cc1plus key (key == the compiler actually invoked)",
        r.returncode == 0
        and len(entries) == 1
        and entries[0].parent.parent.name == hyb_sha[:12]
        and leg.get("cc1plus_sha256") == hyb_sha,
        f"rc={r.returncode} entries={entries}\n{r.stderr[:800]}",
    )

    # 8c. No-GEP caller env with the default-style --compiler still works
    # (both resolutions agree on the pin cc1plus; key = pin sha).
    env_clean = dict(os.environ, COUNTER=str(counter))
    env_clean.pop("GCC_EXEC_PREFIX", None)
    r = run_store(["ensure", *fz_args, "--compiler", str(pin_gxx)], env_clean)
    pin_sha = __import__("hashlib").sha256(pin_cc1.read_bytes()).hexdigest()
    check(
        "FZ-F1: agreeing resolutions (no caller GEP) proceed and key by the "
        "pin cc1plus",
        r.returncode == 0 and (store / pin_sha[:12]).is_dir(),
        f"rc={r.returncode}\n{r.stderr[:800]}",
    )

if FAILS:
    print(f"corpus-leg-store self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "corpus-leg-store self-test: ALL GREEN (concurrent ensure -> one compile; "
    "verified hit; manifest; cc1plus-sha/manifest tamper refuse; head-change "
    "refuse; --no-store isolation)"
)
