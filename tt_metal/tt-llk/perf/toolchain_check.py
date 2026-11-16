# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Nightly build and toolchain check.

    toolchain_check.py toolchain --llk DIR --out toolchain.json     (in each nightly leg)
    toolchain_check.py elfs --a TREE --b TREE --objdump BIN --out build_check.json   (CPU runner)
    toolchain_check.py report --repo R [--date D] --out nightly_checks/build.json   (health job)

toolchain records the installed SFPI, the pinned SFPI and pip freeze. elfs compares the
sentinel ELFs built in two build paths (they must match without their .debug sections) and
keeps per ELF the text size, the TILE_LOOP start address and the loop fingerprint. report
compares today's nightly artifacts with the last earlier ones.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

_LLK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLK / "perf"))
sys.path.append(str(_LLK / "tests/python_tests/helpers/perf"))
import window_hash  # noqa: E402
from nightly_health import _gh, scheduled_run  # noqa: E402

NIGHTLY = "llk-perf.yaml"
PATTERNS = ("llk-toolchain-*", "llk-build-check-*")
WATCHED = {"tt-umd", "pytest", "pytest-xdist", "numpy", "torch"}
LOOKBACK_DAYS = 7
EXAMPLES = 3
SHF_WRITE, SHF_ALLOC, SHT_NOBITS = 0x1, 0x2, 8


def sections(path):
    """{name: (type, flags, addr, bytes)} of an ELF32/64 little-endian file."""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"{path} is not an ELF")
    if data[4] == 2:
        (shoff,) = struct.unpack_from("<Q", data, 0x28)
        entsize, count, strndx = struct.unpack_from("<HHH", data, 0x3A)
        fmt = "<IIQQQQIIQQ"
    else:
        (shoff,) = struct.unpack_from("<I", data, 0x20)
        entsize, count, strndx = struct.unpack_from("<HHH", data, 0x2E)
        fmt = "<IIIIIIIIII"
    heads = [struct.unpack_from(fmt, data, shoff + i * entsize) for i in range(count)]
    names = heads[strndx][4]
    out = {}
    for name, kind, flags, addr, offset, size, *_ in heads[1:]:
        label = data[names + name : data.index(b"\0", names + name)].decode()
        out[label] = (
            kind,
            flags,
            addr,
            size if kind == SHT_NOBITS else data[offset : offset + size],
        )
    return out


def text_size(secs):
    return sum(
        len(body)
        for kind, flags, _, body in secs.values()
        if flags & SHF_ALLOC and not flags & SHF_WRITE and kind != SHT_NOBITS
    )


def differs(a, b):
    """Sections that differ between two builds, debug info (it holds the build path) left out."""
    names = {n for n in [*a, *b] if ".debug" not in n}
    return sorted(n for n in names if a.get(n) != b.get(n))


def loop_start(objdump, elf):
    dump = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(elf)], capture_output=True, text=True
    )
    funcs = window_hash._functions(dump.stdout)
    kernel = next((name for name in funcs if "run_kernel" in name), None)
    if kernel is None:
        return None
    region = window_hash._window(funcs[kernel], pathlib.Path(elf).read_bytes())
    return funcs[kernel][region[0]][0] if region else None


def elfs(tree_a, tree_b, objdump):
    a_root, b_root = pathlib.Path(tree_a), pathlib.Path(tree_b)
    paths = sorted(
        {str(p.relative_to(r)) for r in (a_root, b_root) for p in r.rglob("*.elf")}
    )

    def one(rel):
        a, b = a_root / rel, b_root / rel
        if not a.exists() or not b.exists():
            return {"reproducible": False, "differs": ["only in one build"]}
        secs = sections(a)
        diff = differs(secs, sections(b))
        return {
            "reproducible": not diff,
            "differs": diff,
            "text_size": text_size(secs),
            "loop": loop_start(objdump, a),
            "hash": window_hash.fingerprint(objdump, a),
        }

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        return dict(zip(paths, pool.map(one, paths)))


def toolchain(llk):
    llk = pathlib.Path(llk)
    pin = re.search(r"sfpi_version='([^']*)'", (llk / "tests/sfpi-version").read_text())
    installed = llk / "tests/sfpi/sfpi.version"
    gxx = llk / "tests/sfpi/compiler/bin/riscv-tt-elf-g++"
    compiler = (
        subprocess.run([gxx, "--version"], capture_output=True, text=True).stdout
        if gxx.exists()
        else ""
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    packages = {}
    for line in freeze.splitlines():
        name, _, version = (re.split(r"(==| @ )", line, maxsplit=1) + ["", ""])[:3]
        packages[name.strip().lower().replace("_", "-")] = version.strip()
    return {
        "pin": pin.group(1) if pin else "?",
        "installed": installed.read_text().strip() if installed.exists() else "none",
        "compiler": compiler.splitlines()[0] if compiler else "none",
        "packages": packages,
    }


def toolchain_report(today, earlier):
    problems, lines, changes = set(), set(), set()
    for name, t in today.items():
        if not name.startswith("llk-toolchain-"):
            continue
        if t["installed"] != t["pin"]:
            problems.add(f"SFPI {t['installed']} installed, {t['pin']} pinned")
        e = earlier.get(name)
        if not e:
            continue
        for key in ("pin", "compiler"):
            if t[key] != e[key]:
                lines.add(f"SFPI {key}: {e[key]} → {t[key]}")
        for pkg in set(t["packages"]) | set(e["packages"]):
            old, new = e["packages"].get(pkg, "none"), t["packages"].get(pkg, "none")
            if old != new:
                changes.add((pkg, old, new))
    for pkg, old, new in changes:
        (problems if pkg in WATCHED else lines).add(f"pip {pkg} {old} → {new}")
    return sorted(problems), sorted(lines)


def _kernel(rel):
    return rel.rsplit("/elf/", 1)[0]


def _move(old, new):
    if old["hash"] != new["hash"]:
        return "loop code changed"
    if old["loop"] != new["loop"]:
        return "same loop, moved"
    return "unchanged"


def build_report(today, earlier):
    problems, lines = [], []
    for name, t in sorted(today.items()):
        if not name.startswith("llk-build-check-"):
            continue
        arch = name.removeprefix("llk-build-check-")
        bad = [rel for rel, e in t.items() if not e["reproducible"]]
        if bad:
            problems.append(
                f"{arch}: {len(bad)} of {len(t)} sentinel ELFs not reproducible, "
                f"e.g. {bad[0]} ({', '.join(t[bad[0]]['differs'])})"
            )
        e = earlier.get(name)
        if e is None:
            lines.append(f"{arch}: {len(t)} sentinel ELFs, no earlier build to compare")
            continue
        kernels = {}
        for rel, new in t.items():
            old = e.get(rel)
            if "hash" not in new or not old or "hash" not in old:
                continue
            kernels.setdefault(_kernel(rel), []).append((rel, old, new))
        counts = {"loop code changed": [], "same loop, moved": [], "unchanged": []}
        for kernel, threads in sorted(kernels.items()):
            labels = [_move(old, new) for _, old, new in threads]
            label = min(labels, key=list(counts).index)
            counts[label].append(threads)
        line64 = sum(
            any(
                o["loop"] is not None
                and n["loop"] is not None
                and o["loop"] % 64 != n["loop"] % 64
                for _, o, n in threads
            )
            for threads in counts["same loop, moved"]
        )
        lines.append(
            f"{arch}: {len(t) - len(bad)}/{len(t)} ELFs reproducible; kernels: "
            + ", ".join(f"{len(v)} {k}" for k, v in counts.items())
            + f" ({line64} moved to another 64-byte offset)"
        )
        for label in ("loop code changed", "same loop, moved"):
            for threads in counts[label][:EXAMPLES]:
                lines.append(f"  {label}: " + "; ".join(_detail(*x) for x in threads))
    return problems, lines


def _where(addr):
    return (
        "?" if addr is None else f"{addr:#x} (mod 16 {addr % 16}, mod 64 {addr % 64})"
    )


def _detail(rel, old, new):
    text = f"{rel} TILE_LOOP {_where(old['loop'])}"
    if old["loop"] != new["loop"]:
        text += f" → {_where(new['loop'])}"
    if old["text_size"] != new["text_size"]:
        text += f", text {old['text_size']} → {new['text_size']} B"
    return text


def _load(root):
    out = {}
    for path in sorted(pathlib.Path(root).glob("*/*.json")):
        with open(path) as fh:
            out[path.parent.name] = json.load(fh)
    return out


def download(repo, run, dest):
    try:
        _gh(
            "run",
            "download",
            str(run["id"]),
            "-R",
            repo,
            *[x for p in PATTERNS for x in ("-p", p)],
            "-D",
            str(dest),
        )
    except subprocess.CalledProcessError as e:
        print(f"no artifacts in run {run['id']}: {e.stderr.strip()}", file=sys.stderr)
    return _load(dest)


def report(repo, day, workdir):
    out = {"name": "Build and toolchain", "problems": [], "lines": []}
    run = scheduled_run(repo, NIGHTLY, day.isoformat())
    today = download(repo, run, f"{workdir}/today") if run else {}
    if not today:
        out["lines"].append(f"no build or toolchain artifacts from the {day} nightly")
        return out
    earlier, when = {}, None
    for back in range(1, LOOKBACK_DAYS + 1):
        when = (day - dt.timedelta(days=back)).isoformat()
        old = scheduled_run(repo, NIGHTLY, when)
        earlier = download(repo, old, f"{workdir}/{when}") if old else {}
        if earlier:
            break
    tool_p, tool_l = toolchain_report(today, earlier)
    build_p, build_l = build_report(today, earlier)
    out["problems"] = tool_p + build_p
    out["lines"] = [
        f"compared with the {when} nightly" if earlier else "no earlier artifacts"
    ]
    out["lines"] += tool_l + build_l
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toolchain")
    t.add_argument("--llk", default=str(_LLK))
    e = sub.add_parser("elfs")
    e.add_argument("--a", required=True)
    e.add_argument("--b", required=True)
    e.add_argument("--objdump", required=True)
    r = sub.add_parser("report")
    r.add_argument("--repo", required=True)
    r.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    for p in (t, e, r):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "toolchain":
        result = toolchain(a.llk)
    elif a.cmd == "elfs":
        result = elfs(a.a, a.b, a.objdump)
    else:
        with tempfile.TemporaryDirectory() as workdir:
            result = report(a.repo, dt.date.fromisoformat(a.date), workdir)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    if a.cmd == "report":
        print("\n".join(result["problems"] + result["lines"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
