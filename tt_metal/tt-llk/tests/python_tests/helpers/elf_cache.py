# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Content-keyed cache of variant ELFs that stays valid across commits.

LLK_ELF_CACHE=<dir> turns it on; LLK_ELF_CACHE_READONLY=1 only reads it. A hit needs
the same argv (output path aside), generated source and compiler version, the same
content of every file the miss read (headers from -MD, linker inputs from --trace,
-T scripts, toolchain binaries), and every file that would shadow one of those
headers on the search path still absent. A miss whose headers outside the toolchain
probe for headers (__has_include, #include_next) is not stored. LLK_ELF_CACHE_STATS=<dir> writes one
JSON line per ELF; `python3 elf_cache.py <dir>` sums them up.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_HASHES: dict = {}
_EXISTS: dict = {}
_TOOLCHAIN: dict = {}
_NEG_OK: dict = {}
_PROBES = (b"__has_include", b"#include_next")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: str):
    """(sha256, probes for headers) of a file, or None if it is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        known = _HASHES.get(path)
    if known and known[0] == stamp:
        return known[1]
    data = Path(path).read_bytes()
    value = (_sha_bytes(data), any(p in data for p in _PROBES))
    with _LOCK:
        _HASHES[path] = (stamp, value)
    return value


def _exists(path: str) -> bool:
    with _LOCK:
        known = _EXISTS.get(path)
    if known is None:
        known = os.path.exists(path)
        with _LOCK:
            _EXISTS[path] = known
    return known


def _run(argv, cwd, stdin=None):
    result = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command:\n{' '.join(map(str, argv))}\n\nCommand's stderr:\n{result.stderr}"
        )
    return result


def _toolchain(gxx: str, cwd):
    """Version string and binaries, once per process."""
    with _LOCK:
        if gxx in _TOOLCHAIN:
            return _TOOLCHAIN[gxx]
    version = _run([gxx, "--version"], cwd).stdout
    exe = os.path.join(cwd, gxx) if "/" in gxx else shutil.which(gxx) or gxx
    binaries = [os.path.realpath(exe)]
    for prog in ("cc1plus", "as", "collect2", "ld"):
        found = _run([gxx, f"-print-prog-name={prog}"], cwd).stdout.strip()
        if os.path.isabs(found) and os.path.exists(found):
            binaries.append(os.path.realpath(found))
    value = (version, binaries)
    with _LOCK:
        _TOOLCHAIN[gxx] = value
    return value


def _builtin_dirs(head: list, cwd) -> list:
    """The compiler's own include dirs for these target flags."""
    argv = [head[0]] + [
        t for t in head[1:] if t.startswith(("-m", "-nostd", "--sysroot"))
    ]
    with _LOCK:
        if tuple(argv) in _TOOLCHAIN:
            return _TOOLCHAIN[tuple(argv)]
    verbose = _run(argv + ["-E", "-v", "-x", "c++", "-"], cwd, "").stderr.splitlines()
    dirs, inside = [], False
    for line in verbose:
        if line.startswith("#include"):
            inside = True
        elif line.startswith("End of search list"):
            inside = False
        elif inside and line.startswith(" "):
            dirs.append(os.path.normpath(line.split(" (")[0].strip()))
    with _LOCK:
        _TOOLCHAIN[tuple(argv)] = dirs
    return dirs


def _search_dirs(head: list, cwd: str) -> list:
    dirs, take = [], False
    for token in head:
        if take:
            dirs.append(token)
            take = False
        elif token in ("-I", "-iquote", "-isystem", "-idirafter"):
            take = True
        else:
            for flag in ("-I", "-iquote", "-isystem", "-idirafter"):
                if token.startswith(flag) and len(token) > len(flag):
                    dirs.append(token[len(flag) :])
                    break
    return [os.path.normpath(os.path.join(cwd, d)) for d in dirs]


def _parse_depfile(text: str, cwd) -> list:
    body = text.replace("\\\n", " ").split(":", 1)[1]
    return [
        os.path.normpath(os.path.join(cwd, token))
        for token in body.split()
        if token not in ("-", "<stdin>")
    ]


def _parse_trace(text: str, cwd) -> list:
    paths = []
    for line in text.splitlines():
        token = line.strip().split("(", 1)[0]
        if token and not token.startswith("-"):
            path = os.path.normpath(os.path.join(cwd, token))
            if os.path.isfile(path) and not path.startswith(("/tmp/", "/var/tmp/")):
                paths.append(path)
    return paths


def _shadows(headers: list, search: list) -> dict:
    """Where a new file would be found before one of the headers: every search dir
    (each header's own dir for #include "...") joined with every name the header can
    be included by. The ones present now are listed; the rest must stay absent."""
    dirs = list(dict.fromkeys([os.path.dirname(h) for h in headers] + search))
    rels = sorted(
        {
            os.path.relpath(h, d)
            for h in headers
            for d in dirs
            if h.startswith(d.rstrip("/") + "/")
        }
    )
    present = sorted(
        p for p in (os.path.join(d, r) for d in dirs for r in rels) if _exists(p)
    )
    return {"dirs": dirs, "rels": rels, "present": present}


def _shadows_absent(cache: Path, neg: str) -> bool:
    with _LOCK:
        known = _NEG_OK.get(neg)
    if known is None:
        try:
            spec = json.loads((cache / "n" / f"{neg}.json").read_text())
            present = set(spec["present"])
            known = not any(
                p not in present and _exists(p)
                for p in (
                    os.path.join(d, r) for d in spec["dirs"] for r in spec["rels"]
                )
            )
        except (OSError, ValueError, KeyError):
            known = False
        with _LOCK:
            _NEG_OK[neg] = known
    return known


def _stats(record: dict):
    out = os.environ.get("LLK_ELF_CACHE_STATS")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    with _LOCK:
        with open(os.path.join(out, f"elf.{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _entries(cache: Path, base: str) -> list:
    try:
        return json.loads((cache / "m" / base[:2] / f"{base}.json").read_text())
    except (OSError, ValueError):
        return []


def _lookup(cache: Path, base: str, out: str) -> bool:
    for entry in _entries(cache, base):
        if all(
            (_sha_file(p) or (None,))[0] == h for p, h in entry["deps"].items()
        ) and _shadows_absent(cache, entry["neg"]):
            blob = (cache / "o" / f"{entry['elf']}.elf").read_bytes()
            if _sha_bytes(blob) != entry["elf"]:
                return False
            Path(out).write_bytes(blob)
            return True
    return False


def _store(cache, base, compile_command, cwd, headers, trace, binaries) -> bool:
    head = compile_command[: compile_command.index("-x")]
    root = os.path.dirname(os.path.dirname(binaries[0]))
    scripts = [
        os.path.normpath(os.path.join(cwd, a[2:])) for a in head if a.startswith("-T")
    ]
    header_set = set(headers)
    deps = {p: _sha_file(p) for p in sorted(header_set.union(trace, scripts, binaries))}
    if any(
        v is None or (v[1] and p in header_set and not p.startswith(root + "/"))
        for p, v in deps.items()
    ):
        return False
    blob = Path(compile_command[-1]).read_bytes()
    elf = _sha_bytes(blob)
    if not (cache / "o" / f"{elf}.elf").exists():
        _atomic_write(cache / "o" / f"{elf}.elf", blob)
    search = _search_dirs(head, cwd) + _builtin_dirs(head, cwd)
    spec = json.dumps(_shadows(sorted(set(headers)), search))
    neg = _sha_bytes(spec.encode())
    if not (cache / "n" / f"{neg}.json").exists():
        _atomic_write(cache / "n" / f"{neg}.json", spec.encode())
    entry = {"deps": {p: v[0] for p, v in deps.items()}, "neg": neg, "elf": elf}
    entries = [entry] + _entries(cache, base)[:7]
    _atomic_write(cache / "m" / base[:2] / f"{base}.json", json.dumps(entries).encode())
    return True


def compile_elf(compile_command: list, source: str, cwd) -> None:
    """Compile and link one variant ELF, through the cache. A fault in the cache is a miss."""
    cwd = str(cwd)
    cache = Path(os.environ["LLK_ELF_CACHE"])
    out = compile_command[-1]
    try:
        version, binaries = _toolchain(compile_command[0], cwd)
        key_argv = json.dumps([str(t) for t in compile_command[:-1]])
        base = _sha_bytes(f"direct\0{version}\0{key_argv}\0{source}".encode())
        if _lookup(cache, base, out):
            _stats({"out": out, "result": "hit"})
            return
    except Exception as error:
        print(f"ELF cache lookup failed, compiling: {error}", file=sys.stderr)
        base = None

    if base is None or os.environ.get("LLK_ELF_CACHE_READONLY") == "1":
        _run(compile_command, cwd, source)
        _stats({"out": out, "result": "miss"})
        return

    depfile = f"{out}.d"
    split = compile_command.index("-x")
    argv = compile_command[:split] + ["-MD", "-MF", depfile] + compile_command[split:]
    result = _run(argv, cwd, source)
    try:
        headers = _parse_depfile(Path(depfile).read_text(), cwd)
        os.remove(depfile)
        stored = _store(
            cache,
            base,
            compile_command,
            cwd,
            headers,
            _parse_trace(result.stdout, cwd),
            binaries,
        )
    except Exception as error:
        print(f"ELF cache store failed: {error}", file=sys.stderr)
        stored = False
    _stats({"out": out, "result": "miss", "stored": stored})


if __name__ == "__main__":
    # A variant hits only when all of its ELFs hit: one miss and it waits for the compiler.
    records = [
        json.loads(line)
        for f in Path(sys.argv[1]).glob("elf.*.jsonl")
        for line in f.read_text().splitlines()
    ]
    variants: dict = {}
    for r in records:
        key = str(Path(r["out"]).parent.parent)
        variants[key] = variants.get(key, True) and r["result"] == "hit"
    hits = sum(r["result"] == "hit" for r in records)
    print(
        f"ELF cache: {hits} of {len(records)} ELFs hit, "
        f"{sum(variants.values())} of {len(variants)} variants hit all their ELFs"
    )
