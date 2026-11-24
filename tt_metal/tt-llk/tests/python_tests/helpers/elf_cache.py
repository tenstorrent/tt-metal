# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Content-keyed cache of variant ELFs that stays valid across commits.

LLK_ELF_CACHE=<dir> turns it on. A hit needs the same argv (output path aside), the
same generated source, the same compiler version string, and the same content of
every file the miss read: headers (gcc -MD), linker inputs (ld --trace), linker
scripts (-T) and the toolchain binaries. LLK_ELF_CACHE_MODE picks the key:
  direct  headers are entry dependencies (ccache direct mode; misses a new header
          that shadows an old one on the include path)
  pre     the g++ -E output is in the key (correct by construction, costs a -E)
  both    look up direct, store under both keys (to compare the modes)
LLK_ELF_CACHE_STATS=<dir> writes one JSON line per ELF (works with the cache off).
"""

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_HASHES: dict = {}
_TOOLCHAIN: dict = {}
_SEEN_DEPSETS: set = set()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        known = _HASHES.get(path)
    if known and known[0] == stamp:
        return known[1]
    digest = _sha_bytes(Path(path).read_bytes())
    with _LOCK:
        _HASHES[path] = (stamp, digest)
    return digest


def _run(argv, cwd, stdin=None):
    result = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command:\n{' '.join(map(str, argv))}\n\nCommand's stderr:\n{result.stderr}")
    return result


def _toolchain(gxx: str, cwd):
    """Version string (key) and binaries (dependencies), once per process."""
    with _LOCK:
        if gxx in _TOOLCHAIN:
            return _TOOLCHAIN[gxx]
    version = _run([gxx, "--version"], cwd).stdout
    binaries = [os.path.realpath(shutil.which(gxx) or gxx)]
    for prog in ("cc1plus", "as", "collect2", "ld"):
        found = _run([gxx, f"-print-prog-name={prog}"], cwd).stdout.strip()
        if os.path.isabs(found) and os.path.exists(found):
            binaries.append(os.path.realpath(found))
    value = (version, binaries)
    with _LOCK:
        _TOOLCHAIN[gxx] = value
    return value


def _parse_depfile(text: str, cwd) -> list:
    body = text.replace("\\\n", " ").split(":", 1)[1]
    paths = []
    for token in body.split():
        if token in ("-", "<stdin>"):
            continue
        paths.append(os.path.normpath(os.path.join(cwd, token)))
    return paths


def _parse_trace(text: str, cwd) -> list:
    paths = []
    for line in text.splitlines():
        token = line.strip().split("(", 1)[0]
        if token and not token.startswith("-"):
            path = os.path.normpath(os.path.join(cwd, token))
            if os.path.isfile(path) and not path.startswith(("/tmp/", "/var/tmp/")):
                paths.append(path)
    return paths


def _stats(record: dict, depset=None):
    out = os.environ.get("LLK_ELF_CACHE_STATS")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    line = json.dumps(record) + "\n"
    with _LOCK:
        with open(os.path.join(out, f"tu.{os.getpid()}.jsonl"), "a") as f:
            f.write(line)
        if depset and depset[0] not in _SEEN_DEPSETS:
            _SEEN_DEPSETS.add(depset[0])
            with open(os.path.join(out, f"depsets.{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps({"id": depset[0], "files": depset[1]}) + "\n")


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _lookup(cache: Path, base: str):
    try:
        entries = json.loads((cache / "m" / base[:2] / f"{base}.json").read_text())
    except (OSError, ValueError):
        return None, []
    for entry in entries:
        if all(_sha_file(p) == h for p, h in entry["deps"].items()):
            return entry, entries
    return None, entries


def compile_elf(compile_command: list, source: str, cwd) -> None:
    """Compile and link one variant ELF, through the cache when LLK_ELF_CACHE is set."""
    cwd = str(cwd)
    out = compile_command[-1]
    split = compile_command.index("-x")
    head = compile_command[:split]
    cache = os.environ.get("LLK_ELF_CACHE")
    mode = os.environ.get("LLK_ELF_CACHE_MODE", "direct")
    record = {"out": out, "result": "off"}
    t0 = time.monotonic()

    if not cache:
        _run(compile_command, cwd, source)
        record["compile_s"] = time.monotonic() - t0
        _stats(record)
        return

    cache = Path(cache)
    version, binaries = _toolchain(compile_command[0], cwd)
    key_argv = json.dumps([str(t) for t in compile_command[:-1]])
    base_direct = _sha_bytes(f"direct\0{version}\0{key_argv}\0{source}".encode())
    base_pre = None
    if mode in ("pre", "both"):
        t = time.monotonic()
        pre = _run(head + ["-E", "-x", "c++", "-", "-o", "-"], cwd, source).stdout
        record["pre_s"] = time.monotonic() - t
        base_pre = _sha_bytes(f"pre\0{version}\0{key_argv}\0{_sha_bytes(pre.encode())}".encode())
    lookup_base = base_pre if mode == "pre" else base_direct

    t = time.monotonic()
    entry, _ = _lookup(cache, lookup_base)
    record["lookup_s"] = time.monotonic() - t
    if entry is not None:
        blob = (cache / "o" / f"{entry['elf']}.elf").read_bytes()
        if _sha_bytes(blob) == entry["elf"]:
            Path(out).write_bytes(blob)
            record["result"] = "hit"
            record["total_s"] = time.monotonic() - t0
            _stats(record)
            return

    depfile = f"{out}.d"
    t = time.monotonic()
    argv = head + ["-MD", "-MF", depfile] + compile_command[split:]
    result = _run(argv, cwd, source)
    record["compile_s"] = time.monotonic() - t

    t = time.monotonic()
    headers = _parse_depfile(Path(depfile).read_text(), cwd)
    os.remove(depfile)
    scripts = [os.path.normpath(os.path.join(cwd, a[2:])) for a in head if a.startswith("-T")]
    link_inputs = sorted(set(_parse_trace(result.stdout, cwd) + scripts + binaries))
    blob = Path(out).read_bytes()
    elf = _sha_bytes(blob)
    obj = cache / "o" / f"{elf}.elf"
    if not obj.exists():
        _atomic_write(obj, blob)
    for base, dep_paths in (
        (base_direct, headers + link_inputs) if mode in ("direct", "both") else (None, None),
        (base_pre, link_inputs) if mode in ("pre", "both") else (None, None),
    ):
        if base is None:
            continue
        deps = {p: _sha_file(p) for p in sorted(set(dep_paths))}
        if None in deps.values():
            continue
        manifest = cache / "m" / base[:2] / f"{base}.json"
        _, entries = _lookup(cache, base)
        entries = [{"deps": deps, "elf": elf}] + entries[:7]
        _atomic_write(manifest, json.dumps(entries).encode())
    record["store_s"] = time.monotonic() - t
    record["result"] = "miss"
    depset = sorted(set(headers))
    depset_id = _sha_bytes("\n".join(depset).encode())[:16]
    record["depset"] = depset_id
    _stats(record, (depset_id, depset))
