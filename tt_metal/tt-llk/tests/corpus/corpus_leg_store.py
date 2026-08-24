#!/usr/bin/env python3
"""corpus_leg_store.py — shared, content-addressed BASE corpus-leg store.

laneBU (pin-cycle infrastructure): during a pin cycle every lane compiles
the SAME base corpus leg — the full mapped-corpus compile at the pinned
toolchain and a reviewed flag set — only to byte-compare its own edit leg
against it.  Nine lanes recompiling an identical ~15-minute leg is pure
wall-time waste.  This store publishes ONE completed base leg per
(toolchain, flag-set) key that every lane consumes read-only:

    <store-root>/<cc1plus-sha12>/<flagset-sha12>/
        leg.json          full provenance (see below) — the trust anchor
        text_hashes.tsv   .text/.elf sha256 manifest of every kernel ELF
        results.tsv       sfpu_corpus.py compile-leg row results
        results.json      full machine-readable leg results
        compile-tail.log  the last lines of the leg's compile log

  * cc1plus-sha12 = first 12 hex of the sha256 of the cc1plus binary that
    ACTUALLY compiles (resolved through the driver via
    `g++ -print-prog-name=cc1plus`, same primary-pin discipline as
    sweep_2x2.py).
  * flagset-sha12 = first 12 hex of sha256("arch=<arch>\\nflags=<exact
    TT_LLK_EXTRA_COMPILER_OPTIONS string>\\n") — the preimage is recorded in
    leg.json, so a key is always explainable.

INTEGRITY (verify-before-trust; a mismatch REFUSES, never degrades):
  * leg.json records the FULL cc1plus sha256, the exact flag string, the
    arch, and the tt-metal head (rev-parse HEAD + `+dirty.<sha>` suffix over
    tracked tt_metal/tt-llk edits — the sweep_2x2.py recipe).  Every consume
    path (ensure-hit and manifest) re-derives all three from the caller's
    own toolchain/checkout and refuses on any disagreement.
  * text_hashes.tsv and results.tsv are covered by sha256 values inside
    leg.json; tampering refuses.
  * the resolved cc1plus is re-hashed AFTER the compile finishes; a mid-leg
    toolchain flip (the tests/sfpi symlink is repointable) discards the leg
    instead of publishing poison.  The leg additionally pins its cc1plus via
    GCC_EXEC_PREFIX for the compile's duration, so a concurrent symlink flip
    cannot even change which cc1plus the driver launches.
  * .text hashes are FARM-PATH-DEPENDENT (LLK_PROFILER embeds a source-path
    hash as immediates): leg.json records tt_metal_home_realpath and a
    consumer from a different farm path gets a loud warning — hash-equality
    comparisons are only valid from the same farm path (cross-farm
    equivalence needs an instruction-stream diff).

CONCURRENCY: `ensure` takes an exclusive flock on <entry>.lock before
compiling and re-checks for a published entry after acquiring it, so two
lanes racing the same key produce exactly ONE compile — the loser waits on
the lock, then consumes the winner's published entry.  Publication is an
atomic directory rename (never a partially-written entry).  The flock is
kernel-owned: a crashed producer releases it automatically (no stale-lock
protocol needed).  Waiting on a leg from ANOTHER shell/lane goes through
corpus_watch.py (see README): watch <entry>/leg.json with the producer's
log under --max-age-min.

Commands:
  ensure    publish-if-absent, then print the verified entry path.
            `--no-store --run-root DIR` compiles into DIR without touching
            the store (isolation-critical runs).
  manifest  verify the entry, then print its text_hashes.tsv to stdout.

Typical lane use (BASE leg at the pinned toolchain + reviewed ON set):
  python3 corpus_leg_store.py ensure --arch bh \\
      --flags "$(python3 -c 'import sweep_2x2;print(sweep_2x2.ON_FLAGS)')"
Edit legs (lane compilers) stay per-lane: pass the lane's --compiler and a
lane-private --store-root, or --no-store.
"""
from __future__ import annotations

import argparse
import fcntl
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
DEFAULT_STORE = pathlib.Path.home() / "sfpi-uplift/corpus-legs"
SCHEMA = 1
LEG_FILES = ("leg.json", "text_hashes.tsv", "results.tsv", "results.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def flagset_key(arch, flags):
    """Content key for (arch, flag string); preimage recorded in leg.json."""
    return sha256_text(f"arch={arch}\nflags={flags}\n")


def build_gcc_exec_prefix(compiler_realpath):
    """The GCC_EXEC_PREFIX the leg compile pins (build_leg): derived from
    the --compiler realpath, single source for both the compile env and the
    key-resolution env (FZ-F1)."""
    return str(pathlib.Path(compiler_realpath).parent.parent / "lib/gcc") + "/"


def resolve_cc1plus(compiler, env=None):
    """cc1plus resolved through the driver (the binary that compiles) —
    same primary-pin discipline as sweep_2x2.py.  `env` selects WHOSE
    resolution this is: the store key must come from the BUILD env (the
    exact GCC_EXEC_PREFIX build_leg pins for the compile), never from the
    caller's shell env — see derive_identity for the FZ-F1 divergence
    refusal."""
    compiler = pathlib.Path(compiler)
    if not compiler.is_file():
        sys.exit(f"corpus-leg-store: missing compiler {compiler}")
    cc1 = subprocess.run(
        [str(compiler), "-print-prog-name=cc1plus"],
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    if not cc1 or not pathlib.Path(cc1).is_file():
        sys.exit(
            f"corpus-leg-store: cannot resolve cc1plus via {compiler} (got '{cc1}')"
        )
    return pathlib.Path(cc1), sha256_file(cc1)


def tt_metal_head(root):
    """rev-parse HEAD with the sweep_2x2.py +dirty.<sha> suffix over tracked
    tt_metal/tt-llk edits (untracked files excluded: tests/sfpi symlink and
    __pycache__ churn must not fork the key)."""
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--", "tt_metal/tt-llk"],
        capture_output=True,
    ).stdout
    if dirty.strip():
        head += "+dirty." + hashlib.sha256(dirty).hexdigest()[:16]
    return head


def hash_build(rt, objcopy, out_file):
    """Hash .text and full bytes of every kernel ELF under one RUNNER_TEMP —
    byte-compatible with sweep_2x2.py's hashes-<leg>.txt format."""
    entries = []
    build = pathlib.Path(rt) / "tt-llk-build"
    for elf in sorted(build.rglob("*.elf")):
        if "shared" in elf.parts:
            continue  # brisc bootrom is flag-independent scaffolding
        rel = elf.relative_to(build)
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
            sys.exit(
                f"corpus-leg-store: objcopy failed on {elf} (the manifest is "
                f"the store's product; it must not silently degrade): "
                f"{text.stderr.decode(errors='replace')[:300]}"
            )
        entries.append(
            (str(rel), hashlib.sha256(text.stdout).hexdigest(), sha256_file(elf))
        )
    with open(out_file, "w") as f:
        for rel, t, e in entries:
            f.write(f"{rel}\ttext:{t}\telf:{e}\n")
    return len(entries)


def verify_entry(entry, want):
    """Verify a published entry against the CONSUMER's own derived identity.

    `want` carries the consumer-side truth: cc1plus_sha256, flags, arch,
    tt_metal_head, tt_metal_home_realpath.  Returns (ok, detail); ok=False
    details name the exact disagreement (verify-before-trust: a consumer
    must never trust a store path string over the recorded provenance)."""
    entry = pathlib.Path(entry)
    leg_path = entry / "leg.json"
    if not leg_path.is_file():
        return False, f"no leg.json in {entry}"
    try:
        leg = json.loads(leg_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"unreadable leg.json in {entry}: {e}"
    for key in ("cc1plus_sha256", "flags", "arch", "tt_metal_head"):
        if key not in leg:
            return False, f"leg.json lacks '{key}' (pre-schema entry?) in {entry}"
        if want.get(key) is not None and leg[key] != want[key]:
            return (
                False,
                f"SHA/IDENTITY MISMATCH on '{key}': store recorded "
                f"{leg[key]!r}, this consumer derived {want[key]!r} — "
                "refusing to trust the entry (re-pin/re-derive through the "
                "store, never by hand-editing leg.json)",
            )
    for name, sha_key in (
        ("text_hashes.tsv", "text_hashes_sha256"),
        ("results.tsv", "results_tsv_sha256"),
    ):
        f = entry / name
        if not f.is_file():
            return False, f"entry is missing {name}: {entry}"
        got = sha256_file(f)
        if leg.get(sha_key) != got:
            return (
                False,
                f"TAMPER: {name} hashes to {got} but leg.json records "
                f"{leg.get(sha_key)} — refusing the entry",
            )
    warn = ""
    if (
        want.get("tt_metal_home_realpath")
        and leg.get("tt_metal_home_realpath") != want["tt_metal_home_realpath"]
    ):
        warn = (
            "WARNING: entry was compiled from farm path "
            f"{leg.get('tt_metal_home_realpath')} but this consumer is at "
            f"{want['tt_metal_home_realpath']} — .text hashes are FARM-PATH-"
            "DEPENDENT (LLK_PROFILER path-hash immediates): hash-equality "
            "comparison against this manifest is NOT valid cross-farm"
        )
    return True, warn


def default_producer(args, workdir, rt, env):
    """The real leg producer: sfpu_corpus.py compile mode of the target
    checkout (the script physically inside that checkout — it resolves
    __file__, symlinked copies escape the farm)."""
    corpus_py = (
        pathlib.Path(args.tt_metal_home) / "tt_metal/tt-llk/tests/corpus/sfpu_corpus.py"
    )
    if not corpus_py.is_file():
        sys.exit(f"corpus-leg-store: no sfpu_corpus.py at {corpus_py}")
    run_root = workdir / "run"
    cmd = [
        sys.executable,
        str(corpus_py),
        "--mode",
        "compile",
        "--arch",
        args.arch,
        "--execute",
        "--require-executed-mapped",
        "--run-root",
        str(run_root),
    ]
    log = workdir / "compile.log"
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, run_root, log, cmd


def run_producer(args, workdir, rt, env):
    if args.producer_cmd:
        # Test/expert escape: an arbitrary producer populates the leg.  The
        # command is RECORDED in leg.json (producer=custom) so consumers can
        # see the leg did not come from the standard sfpu_corpus pipeline.
        run_root = workdir / "run"
        log = workdir / "compile.log"
        env = dict(env, LEG_RUN_ROOT=str(run_root), LEG_RUNNER_TEMP=str(rt))
        with open(log, "w") as f:
            rc = subprocess.run(
                args.producer_cmd,
                shell=True,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode
        return rc, run_root, log, ["<custom>", args.producer_cmd]
    return default_producer(args, workdir, rt, env)


def build_leg(args, ident, dest_dir):
    """Compile one leg and assemble its publishable content into dest_dir.
    Returns the leg dict.  dest_dir must not exist."""
    dest_dir = pathlib.Path(dest_dir)
    workdir = dest_dir.parent / (dest_dir.name + f".build.{os.getpid()}")
    shutil.rmtree(workdir, ignore_errors=True)
    rt = workdir / "rt"
    rt.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        RUNNER_TEMP=str(rt),
        TT_LLK_EXTRA_COMPILER_OPTIONS=args.flags,
        CORPUS_TT_METAL_HOME=str(args.tt_metal_home),
        # Pin the cc1plus the driver launches for the leg's whole duration:
        # a concurrent tests/sfpi symlink flip changes the driver's default
        # search path, but GCC_EXEC_PREFIX wins (drivers are byte-identical
        # across cc1plus-only rebuilds, so pinning cc1plus is the pin that
        # matters).  Verified belt-and-braces by the post-compile re-hash.
        # Single source with derive_identity's key resolution (FZ-F1): the
        # entry key was resolved under exactly this GCC_EXEC_PREFIX.
        GCC_EXEC_PREFIX=build_gcc_exec_prefix(ident["compiler_realpath"]),
    )
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rc, run_root, log, cmd = run_producer(args, workdir, rt, env)
    if rc != 0:
        tail = ""
        if log.is_file():
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-15:])
        if not args.keep_failed:
            shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(
            f"corpus-leg-store: leg compile FAILED (rc={rc}); refusing to "
            f"publish.  Log: {log if args.keep_failed else '(removed; re-run with --keep-failed)'}\n{tail}"
        )
    # Post-compile identity re-check: a mid-leg toolchain flip publishes
    # poison labeled with the preflight sha — discard instead.  Resolved
    # under the SAME build env the compile pinned (FZ-F1): the caller-env
    # resolution would re-verify the wrong compiler.
    _, cc1_sha_after = resolve_cc1plus(
        args.compiler,
        env=dict(
            os.environ,
            GCC_EXEC_PREFIX=build_gcc_exec_prefix(ident["compiler_realpath"]),
        ),
    )
    if cc1_sha_after != ident["cc1plus_sha256"]:
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(
            "corpus-leg-store: TOOLCHAIN CHANGED MID-LEG: cc1plus was "
            f"{ident['cc1plus_sha256']} at preflight but {cc1_sha_after} "
            "after the compile — discarding the leg (never publishing "
            "evidence keyed to an identity that did not hold throughout)"
        )
    head_after = tt_metal_head(args.tt_metal_home)
    if head_after != ident["tt_metal_head"]:
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(
            "corpus-leg-store: SOURCE TREE CHANGED MID-LEG: tt-metal head "
            f"was {ident['tt_metal_head']} at preflight but {head_after} "
            "after the compile — discarding the leg"
        )
    pub = workdir / "publish"
    pub.mkdir()
    elf_count = hash_build(rt, ident["objcopy"], pub / "text_hashes.tsv")
    if args.keep_build:
        # Retain the compiled tree so batched-silicon consumers (sweep_2x2
        # group producers) can seed a group RUNNER_TEMP from the store
        # instead of recompiling.  Moved, not copied: the build is large.
        os.rename(rt / "tt-llk-build", pub / "build")
    if elf_count == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(
            "corpus-leg-store: leg produced ZERO kernel ELFs under "
            "RUNNER_TEMP/tt-llk-build — an empty manifest is not a leg; "
            "refusing to publish (harness ignored RUNNER_TEMP?)"
        )
    for name in ("results.tsv", "results.json"):
        src = run_root / name
        if not src.is_file():
            shutil.rmtree(workdir, ignore_errors=True)
            sys.exit(
                f"corpus-leg-store: producer left no {name} under {run_root} "
                "— refusing to publish a leg without row results"
            )
        shutil.copy2(src, pub / name)
    if log.is_file():
        (pub / "compile-tail.log").write_text(
            "".join(log.read_text(errors="replace").splitlines(True)[-200:])
        )
    leg = {
        "schema": SCHEMA,
        "arch": args.arch,
        "flags": args.flags,
        "flagset_sha256": flagset_key(args.arch, args.flags),
        "cc1plus": str(ident["cc1plus"]),
        "cc1plus_sha256": ident["cc1plus_sha256"],
        "compiler": str(args.compiler),
        "compiler_realpath": ident["compiler_realpath"],
        "compiler_sha256": ident["compiler_sha256"],
        "tt_metal_home": str(args.tt_metal_home),
        "tt_metal_home_realpath": str(pathlib.Path(args.tt_metal_home).resolve()),
        "tt_metal_head": ident["tt_metal_head"],
        "producer": "custom" if args.producer_cmd else "sfpu_corpus",
        "producer_cmd": cmd,
        "has_build": bool(args.keep_build),
        "elf_count": elf_count,
        "text_hashes_sha256": sha256_file(pub / "text_hashes.tsv"),
        "results_tsv_sha256": sha256_file(pub / "results.tsv"),
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    (pub / "leg.json").write_text(json.dumps(leg, indent=2) + "\n")
    if dest_dir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(
            f"corpus-leg-store: {dest_dir} appeared while we were building "
            "(lock protocol violated?) — refusing to overwrite"
        )
    os.rename(pub, dest_dir)  # atomic publish (same filesystem by layout)
    shutil.rmtree(workdir, ignore_errors=True)
    return leg


def derive_identity(args):
    """Store-key identity, resolved under the BUILD env (FZ-F1 fix).

    build_leg compiles with GCC_EXEC_PREFIX rewritten from the --compiler
    realpath, so the cc1plus that keys the entry MUST be resolved under
    that same env — the historical caller-env resolution let a hybrid lane
    (shell GCC_EXEC_PREFIX pointing at its own cc1plus) run `ensure`
    WITHOUT --compiler and silently store PINNED-toolchain bytes under the
    HYBRID's key (lane FZ, poisoned-key fail-open).  Both resolutions are
    computed and any divergence REFUSES LOUDLY: it means the compiler the
    caller's env selects is not the compiler the leg will actually invoke
    — the fix is to pass --compiler <the intended driver>."""
    compiler_realpath = str(pathlib.Path(args.compiler).resolve())
    build_env = dict(
        os.environ, GCC_EXEC_PREFIX=build_gcc_exec_prefix(compiler_realpath)
    )
    cc1, cc1_sha = resolve_cc1plus(args.compiler, env=build_env)
    caller_cc1, caller_sha = resolve_cc1plus(args.compiler)
    if caller_sha != cc1_sha:
        sys.exit(
            "corpus-leg-store: CC1PLUS RESOLUTION DIVERGENCE (FZ-F1 "
            "fail-closed): the caller environment resolves cc1plus to\n"
            f"  {caller_cc1} (sha {caller_sha})\n"
            "but the leg will COMPILE with the --compiler-derived "
            f"GCC_EXEC_PREFIX ({build_env['GCC_EXEC_PREFIX']}), which "
            "resolves to\n"
            f"  {cc1} (sha {cc1_sha})\n"
            "— storing under either key would poison the store (a leg "
            "keyed to one compiler, compiled by another).  Pass "
            "--compiler <the driver whose cc1plus you intend> (hybrid "
            "legs MUST pass their hybrid driver), or unset/realign "
            "GCC_EXEC_PREFIX in the calling environment."
        )
    return {
        "cc1plus": cc1,
        "cc1plus_sha256": cc1_sha,
        "compiler_realpath": compiler_realpath,
        "compiler_sha256": sha256_file(args.compiler),
        "objcopy": pathlib.Path(args.compiler).with_name("riscv-tt-elf-objcopy"),
        "tt_metal_head": tt_metal_head(args.tt_metal_home),
    }


def entry_path(args, ident):
    return (
        pathlib.Path(args.store_root)
        / ident["cc1plus_sha256"][:12]
        / flagset_key(args.arch, args.flags)[:12]
    )


def want_of(args, ident):
    return {
        "cc1plus_sha256": ident["cc1plus_sha256"],
        "flags": args.flags,
        "arch": args.arch,
        "tt_metal_head": ident["tt_metal_head"],
        "tt_metal_home_realpath": str(pathlib.Path(args.tt_metal_home).resolve()),
    }


def consume_or_none(entry, want):
    """Verify a published entry; ok -> print warnings and return the path,
    identity mismatch on an EXISTING entry -> hard refuse (exit 1),
    absent entry -> None."""
    if not pathlib.Path(entry, "leg.json").is_file():
        return None
    ok, detail = verify_entry(entry, want)
    if not ok:
        sys.exit(f"corpus-leg-store: REFUSED {entry}: {detail}")
    if detail:
        print(f"corpus-leg-store: {detail}", file=sys.stderr)
    return entry


def find_build(store_root, cc1plus_sha256, arch, flags, tt_metal_head, farm_realpath):
    """Verified prebuilt tree for (toolchain, flag-set), or None.

    Used by sweep_2x2.py's batched-silicon group producer: when a store
    entry exists for this exact cc1plus sha + arch + flag string, verifies
    (verify_entry: shas, manifest integrity), carries a retained build tree
    (`ensure --keep-build`), matches the caller's tt-metal head AND farm
    realpath (LLK_PROFILER .text hashes are farm-path-dependent — a build
    from another farm would fail every hash-match gate), the group compile
    can be seeded from it instead of recompiling.  ANY mismatch returns
    None (the caller falls back to a fresh compile — reuse is an
    optimization, never a trust decision)."""
    entry = (
        pathlib.Path(store_root) / cc1plus_sha256[:12] / flagset_key(arch, flags)[:12]
    )
    build = entry / "build"
    if not (entry / "leg.json").is_file() or not build.is_dir():
        return None
    ok, detail = verify_entry(
        entry,
        {
            "cc1plus_sha256": cc1plus_sha256,
            "flags": flags,
            "arch": arch,
            "tt_metal_head": tt_metal_head,
            "tt_metal_home_realpath": farm_realpath,
        },
    )
    if not ok or detail:  # a farm-path WARNING also disqualifies build reuse
        return None
    try:
        leg = json.loads((entry / "leg.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if leg.get("tt_metal_home_realpath") != farm_realpath or not leg.get("has_build"):
        return None
    return build


def cmd_ensure(args):
    ident = derive_identity(args)
    if args.no_store:
        if not args.run_root:
            sys.exit("corpus-leg-store: --no-store requires --run-root")
        dest = pathlib.Path(args.run_root).resolve() / "leg"
        if dest.exists():
            sys.exit(f"corpus-leg-store: --run-root leg already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        build_leg(args, ident, dest)
        print(f"ENTRY {dest}")
        return 0
    entry = entry_path(args, ident)
    want = want_of(args, ident)
    hit = consume_or_none(entry, want)
    if hit is not None:
        print(f"corpus-leg-store: HIT (verified) {entry}", file=sys.stderr)
        print(f"ENTRY {entry}")
        return 0
    entry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = entry.parent / (entry.name + ".lock")
    print(
        f"corpus-leg-store: MISS — acquiring {lock_path} (a concurrent lane "
        "compiling this key holds it; we wait instead of duplicating)",
        file=sys.stderr,
    )
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # kernel-owned; crash-safe
        # Re-check under the lock: the previous holder may have published.
        hit = consume_or_none(entry, want)
        if hit is not None:
            print(
                f"corpus-leg-store: published while we waited — consuming "
                f"(verified) {entry}",
                file=sys.stderr,
            )
            print(f"ENTRY {entry}")
            return 0
        print(f"corpus-leg-store: compiling leg -> {entry}", file=sys.stderr)
        build_leg(args, ident, entry)
        print(f"corpus-leg-store: PUBLISHED {entry}", file=sys.stderr)
    print(f"ENTRY {entry}")
    return 0


def cmd_manifest(args):
    ident = derive_identity(args)
    entry = entry_path(args, ident)
    want = want_of(args, ident)
    if not pathlib.Path(entry, "leg.json").is_file():
        sys.exit(f"corpus-leg-store: no entry at {entry} (run ensure first)")
    ok, detail = verify_entry(entry, want)
    if not ok:
        sys.exit(f"corpus-leg-store: REFUSED {entry}: {detail}")
    if detail:
        print(f"corpus-leg-store: {detail}", file=sys.stderr)
    sys.stdout.write((pathlib.Path(entry) / "text_hashes.tsv").read_text())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("ensure", "manifest"))
    ap.add_argument(
        "--flags",
        required=True,
        help="exact TT_LLK_EXTRA_COMPILER_OPTIONS flag string for the leg "
        "(part of the store key; byte-exact)",
    )
    ap.add_argument("--arch", default="bh", choices=("bh", "wh"))
    ap.add_argument(
        "--store-root",
        type=pathlib.Path,
        default=DEFAULT_STORE,
        help=f"store root (default {DEFAULT_STORE})",
    )
    ap.add_argument(
        "--tt-metal-home",
        type=pathlib.Path,
        # HERE is .../tt_metal/tt-llk/tests/corpus -> checkout root is 4 up.
        default=HERE.parents[3],
        help="tt-metal checkout whose harness compiles the leg (default: "
        "the checkout containing this script)",
    )
    ap.add_argument(
        "--compiler",
        type=pathlib.Path,
        help="riscv-tt-elf-g++ driver (default: the checkout's "
        "tests/sfpi/compiler/bin/riscv-tt-elf-g++)",
    )
    ap.add_argument(
        "--no-store",
        action="store_true",
        help="isolation escape: compile into --run-root without reading or "
        "writing the shared store",
    )
    ap.add_argument("--run-root", type=pathlib.Path, help="workdir for --no-store")
    ap.add_argument(
        "--keep-failed",
        action="store_true",
        help="keep the workdir of a failed leg for debugging",
    )
    ap.add_argument(
        "--keep-build",
        action="store_true",
        help="retain the compiled tt-llk-build tree in the entry (build/) so "
        "sweep_2x2.py batched-silicon group producers can seed a RUNNER_TEMP "
        "from the store (find_build); large — use for BASE legs on the "
        "shared farm",
    )
    ap.add_argument(
        "--producer-cmd",
        help="EXPERT/TEST: replace the sfpu_corpus.py producer with a shell "
        "command (gets LEG_RUN_ROOT/LEG_RUNNER_TEMP env; must populate "
        "$LEG_RUN_ROOT/results.{tsv,json} and $LEG_RUNNER_TEMP/tt-llk-build)."
        "  Recorded in leg.json as producer=custom.",
    )
    args = ap.parse_args(argv)
    if args.compiler is None:
        args.compiler = (
            pathlib.Path(args.tt_metal_home)
            / "tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-g++"
        )
    if args.command == "ensure":
        return cmd_ensure(args)
    return cmd_manifest(args)


if __name__ == "__main__":
    sys.exit(main())
