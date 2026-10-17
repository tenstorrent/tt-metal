#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Move one LLK perf shard's compiled ELFs from a CPU runner to a board runner.

The compile pass (--compile-producer) can run on a CPU-only runner; the board
runner then only measures (--compile-consumer). This tool carries the build
between them:

  pack       after the compile pass: archive the files the measuring pass reads,
             with a manifest and a checksum list (tar + zstd -3).
  unpack     on the board, before the measuring pass: check the manifest against
             this job, extract, and check every file against the checksum list.
  elf-check  on the board, after the measuring pass: rebuild a sample of the
             shard here and compare it, file by file, with the shipped ELFs.

The variant id ignores compile flags and the environment, so a build made with
other flags would be found and loaded without complaint. The manifest records
them, and unpack refuses a build that does not match this job.

Both runners must use the same build path ($RUNNER_TEMP/tt-llk-build): the ELFs
contain it. The LLK container gives /__w/_temp on every runner.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PYTHON_TESTS = TESTS_DIR / "python_tests"
GXX = TESTS_DIR / "sfpi/compiler/bin/riscv-tt-elf-g++"
OBJCOPY = TESTS_DIR / "sfpi/compiler/bin/riscv-tt-elf-objcopy"
META_DIR = "_offload"

# Environment the compiled ELFs depend on. The variant id does not see these.
BUILD_ENV = [
    "CHIP_ARCH",
    "SPEED_OF_LIGHT",
    "LLK_PERF_DEBUG_SYMBOLS",
    "TT_LLK_DISABLE_ASSERTS",
    "LLK_PERF_RUN_TYPES",
    "LLK_PERF_L1_MUX_GROUP",
    "LLK_PERF_L1_CLIENT_SEL",
    "LLK_TEST_SEED",
    "TT_METAL_DISABLE_SFPLOADMACRO",
]
# Manifest fields that must match between the compile job and the board job.
MUST_MATCH = ["commit", "compiler", "cmd", "group", "splits", "env"]


def build_dir() -> Path:
    runner_temp = os.environ.get("RUNNER_TEMP")
    if not runner_temp:
        sys.exit("RUNNER_TEMP is not set; the build path must match the CI one")
    return Path(runner_temp) / "tt-llk-build"


def summary(lines):
    text = "\n".join(lines) + "\n"
    print(text, end="")
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text)


def shard_of(cmd: str) -> tuple[int, int]:
    m = re.search(r"run_llk_perf_\w+\.sh\s+(\d+)\s+(\d+)", cmd)
    if not m:
        sys.exit(f"cannot find '<group> <n_groups>' in the matrix cmd: {cmd!r}")
    return int(m.group(1)), int(m.group(2))


def describe(cmd: str) -> dict:
    group, splits = shard_of(cmd)
    commit = subprocess.run(
        ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
        cwd=TESTS_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    compiler = subprocess.run(
        [str(GXX), "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    return {
        "commit": commit,
        "compiler": compiler,
        "cmd": cmd.strip(),
        "group": group,
        "splits": splits,
        "env": {k: os.environ.get(k) for k in BUILD_ENV},
    }


def shipped_files(root: Path) -> list[str]:
    """What the measuring pass reads: every */elf/*.elf and profiler_meta/**."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            dirnames[:] = [d for d in dirnames if d != META_DIR]
        in_meta = rel_dir == "profiler_meta" or rel_dir.startswith("profiler_meta/")
        is_elf_dir = os.path.basename(dirpath) == "elf"
        for name in filenames:
            if in_meta or (is_elf_dir and name.endswith(".elf")):
                out.append(os.path.normpath(os.path.join(rel_dir, name)))
    out.sort()
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_pack(args):
    root = build_dir()
    t0 = time.monotonic()
    files = shipped_files(root)
    if not files:
        sys.exit(f"no ELFs under {root}; did the compile pass run?")
    meta = root / META_DIR
    shutil.rmtree(meta, ignore_errors=True)
    meta.mkdir()
    total = 0
    with open(meta / "SHA256SUMS", "w") as f:
        for rel in files:
            p = root / rel
            total += p.stat().st_size
            f.write(f"{sha256(p)}  {rel}\n")
    manifest = describe(args.cmd)
    manifest.update(
        {
            "file_count": len(files),
            "total_bytes": total,
            "runner_name": os.environ.get("RUNNER_NAME"),
            "nproc": os.cpu_count(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    (meta / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    junits = []
    for j in args.junit:
        if Path(j).is_file():
            shutil.copy(j, meta / Path(j).name)
            junits.append(Path(j).name)
    t1 = time.monotonic()

    listing = "\0".join(files + [f"{META_DIR}/{n}" for n in os.listdir(meta)]) + "\0"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(
        ["tar", "-C", str(root), "--null", "-T", "-", "-cf", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    zstd = subprocess.Popen(
        ["zstd", "-3", "-T0", "-q", "-f", "-o", str(out)], stdin=tar.stdout
    )
    tar.stdout.close()
    tar.stdin.write(listing.encode())
    tar.stdin.close()
    if tar.wait() or zstd.wait():
        sys.exit("tar | zstd failed")
    t2 = time.monotonic()
    summary(
        [
            f"### Pack: {out.name}",
            f"- files: {len(files)} ({total / 1e6:.1f} MB), junit shipped: {', '.join(junits) or 'none'}",
            f"- archive: {out.stat().st_size / 1e6:.1f} MB",
            f"- list + checksum: {t1 - t0:.1f} s, tar + zstd: {t2 - t1:.1f} s",
        ]
    )


def cmd_unpack(args):
    found = sorted(Path(args.dir).rglob("*.tar.zst"))
    if len(found) != 1:
        sys.exit(f"expected one .tar.zst under {args.dir}, found {found}")
    archive = found[0]
    root = build_dir()
    # Start from an empty build directory so no stale ELF can be measured.
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    t0 = time.monotonic()
    zstd = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-C", str(root), "-xf", "-"], stdin=zstd.stdout)
    zstd.stdout.close()
    if tar.wait() or zstd.wait():
        sys.exit("zstd | tar failed")
    t1 = time.monotonic()

    meta = root / META_DIR
    shipped = json.loads((meta / "manifest.json").read_text())
    here = describe(args.cmd)
    bad = [k for k in MUST_MATCH if shipped.get(k) != here[k]]
    if bad:
        for k in bad:
            print(
                f"::error::manifest {k}: shipped {shipped.get(k)!r}, this job {here[k]!r}"
            )
        sys.exit("the shipped build does not match this job; not opening the device")

    files = shipped_files(root)
    if len(files) != shipped["file_count"]:
        sys.exit(
            f"archive has {len(files)} files, manifest says {shipped['file_count']}"
        )
    mismatched = []
    for line in (meta / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if sha256(root / rel) != digest:
            mismatched.append(rel)
    if mismatched:
        sys.exit(f"{len(mismatched)} files fail their checksum, e.g. {mismatched[:5]}")
    t2 = time.monotonic()

    for j in meta.glob("*.xml"):
        shutil.copy(j, PYTHON_TESTS / j.name)
    summary(
        [
            f"### Unpack: {archive.name}",
            f"- archive: {archive.stat().st_size / 1e6:.1f} MB, files: {len(files)}",
            f"- built on {shipped.get('runner_name')} ({shipped.get('nproc')} CPUs) at {shipped.get('created')}",
            f"- extract: {t1 - t0:.1f} s, verify: {t2 - t1:.1f} s",
        ]
    )


def same_loaded_code(a: Path, b: Path, tmp: Path) -> bool:
    """True when the two ELFs differ only in debug sections."""
    sa, sb = tmp / "a.elf", tmp / "b.elf"
    for src, dst in ((a, sa), (b, sb)):
        subprocess.run([str(OBJCOPY), "--strip-debug", str(src), str(dst)], check=True)
    return sa.read_bytes() == sb.read_bytes()


def cmd_elf_check(args):
    root = build_dir()
    shipped_root = root.with_name(root.name + ".offload")
    shipped = json.loads((root / META_DIR / "manifest.json").read_text())
    group, splits = shipped["group"], shipped["splits"]
    fine_splits = splits * args.factor
    fine_groups = [(group - 1) * args.factor + k for k in args.fine_groups]

    shutil.rmtree(shipped_root, ignore_errors=True)
    root.rename(shipped_root)

    pytest_args = ["-q", "--override-ini=log_cli=false", "--compile-producer"]
    pytest_args += [
        "-n",
        str(args.workers),
        "-m",
        "perf and not accuracy",
        "--timeout=60",
    ]
    if shipped["env"].get("SPEED_OF_LIGHT") == "true":
        pytest_args.append("--speed-of-light")
    if shipped["env"].get("LLK_PERF_DEBUG_SYMBOLS") == "false":
        pytest_args.append("--no-debug-symbols")

    same = stripped_same = differ = absent = 0
    examples = []
    tmp = shipped_root.with_name("elf-check-tmp")
    tmp.mkdir(exist_ok=True)
    t0 = time.monotonic()
    for fg in fine_groups:
        # The producer clears the build directory first, so each sample starts clean.
        subprocess.run(
            [
                "pytest",
                *pytest_args,
                "--perf-splits",
                str(fine_splits),
                "--perf-group",
                str(fg),
                ".",
            ],
            cwd=PYTHON_TESTS,
            check=False,
        )
        for rel in shipped_files(root):
            ours, theirs = root / rel, shipped_root / rel
            if not theirs.exists():
                absent += 1
                if len(examples) < 10:
                    examples.append(f"not shipped: {rel}")
            elif ours.read_bytes() == theirs.read_bytes():
                same += 1
            elif rel.endswith(".elf") and same_loaded_code(ours, theirs, tmp):
                stripped_same += 1
                if len(examples) < 10:
                    examples.append(f"debug sections differ: {rel}")
            else:
                differ += 1
                if len(examples) < 10:
                    examples.append(f"DIFFERENT: {rel}")
    t1 = time.monotonic()
    shutil.rmtree(tmp, ignore_errors=True)

    compared = same + stripped_same + differ
    summary(
        [
            f"### ELF identity: board rebuild vs shipped build ({group}/{splits}, fine groups {fine_groups} of {fine_splits})",
            f"- compared: {compared}, byte-identical: {same}, same after --strip-debug: {stripped_same}, "
            f"different: {differ}, rebuilt but not shipped: {absent}",
            f"- rebuild + compare: {t1 - t0:.0f} s",
            *[f"  - {e}" for e in examples],
        ]
    )
    if compared == 0 or differ or absent:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack")
    p.add_argument("--cmd", required=True, help="the matrix row's cmd")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--junit", nargs="*", default=[], help="compile JUnit report(s) to ship"
    )
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack")
    p.add_argument("--cmd", required=True, help="the matrix row's cmd")
    p.add_argument("--dir", required=True, help="where the archive was downloaded")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("elf-check")
    p.add_argument("--factor", type=int, default=100, help="fine shards per shard")
    p.add_argument("--fine-groups", type=int, nargs="+", default=[10, 60])
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_elf_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
