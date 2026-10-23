#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Move a perf shard's ELFs from a CPU runner to a board: pack, unpack, verify-tree.

The ELFs embed their build path, so both runners must build in $RUNNER_TEMP/tt-llk-build.
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
META_DIR = "_offload"

# The variant id ignores these, so the manifest must match on them.
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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        sums[rel] = digest
    return sums


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


def find_archive(directory: str) -> Path:
    found = sorted(Path(directory).rglob("*.tar.zst"))
    if len(found) != 1:
        sys.exit(f"expected one .tar.zst under {directory}, found {found}")
    return found[0]


def cmd_unpack(args):
    archive = find_archive(args.dir)
    root = build_dir()
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
    sums = read_sums(meta / "SHA256SUMS")
    mismatched = [
        rel
        for rel, digest in sums.items()
        if not (root / rel).is_file() or sha256(root / rel) != digest
    ]
    if mismatched:
        sys.exit(f"{len(mismatched)} files missing or bad, e.g. {mismatched[:5]}")
    t2 = time.monotonic()

    if not args.no_junit:
        for j in meta.glob("*.xml"):
            shutil.copy(j, PYTHON_TESTS / j.name)
    size = archive.stat().st_size
    if args.delete_archive:
        archive.unlink()
    summary(
        [
            f"### Unpack: {archive.name}",
            f"- archive: {size / 1e6:.1f} MB, files: {len(files)}",
            f"- built on {shipped.get('runner_name')} ({shipped.get('nproc')} CPUs) at {shipped.get('created')}",
            f"- extract: {t1 - t0:.1f} s, verify: {t2 - t1:.1f} s",
        ]
    )


def archive_sums(archive: Path) -> dict[str, str]:
    zstd = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    tar = subprocess.run(
        ["tar", "-xOf", "-", f"{META_DIR}/SHA256SUMS"],
        stdin=zstd.stdout,
        capture_output=True,
        text=True,
    )
    zstd.stdout.close()
    if tar.returncode or zstd.wait():
        sys.exit(f"cannot read {META_DIR}/SHA256SUMS from {archive}: {tar.stderr}")
    return dict(
        reversed(line.split("  ", 1)) for line in tar.stdout.splitlines() if line
    )


def cmd_verify_tree(args):
    tree = build_dir()
    t0 = time.monotonic()
    sums = archive_sums(find_archive(args.dir))
    present = set(shipped_files(tree)) if tree.exists() else set()
    identical, different, missing = 0, [], []
    for rel, digest in sums.items():
        if rel not in present:
            missing.append(rel)
        elif sha256(tree / rel) == digest:
            identical += 1
        else:
            different.append(rel)
    extra = sorted(present - sums.keys())
    result = {
        "tree": str(tree),
        "compared": len(sums),
        "identical": identical,
        "different": len(different),
        "missing_in_tree": len(missing),
        "extra_in_tree": len(extra),
        "examples": {
            "different": different[:20],
            "missing_in_tree": missing[:20],
            "extra_in_tree": extra[:20],
        },
        "seconds": round(time.monotonic() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    ok = not (different or missing or extra)
    summary(
        [
            f"### Build identity: {tree.name} vs the shipped build",
            f"- {'identical' if ok else 'NOT identical'}: {identical} of {len(sums)} files byte-identical, "
            f"{len(different)} different, {len(missing)} missing, {len(extra)} extra ({result['seconds']} s)",
            *[f"  - different: {r}" for r in different[:5]],
            *[f"  - missing: {r}" for r in missing[:5]],
            *[f"  - extra: {r}" for r in extra[:5]],
        ]
    )
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack")
    p.add_argument("--cmd", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--junit", nargs="*", default=[])
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("unpack")
    p.add_argument("--cmd", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--delete-archive", action="store_true")
    p.add_argument("--no-junit", action="store_true")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("verify-tree")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_verify_tree)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
