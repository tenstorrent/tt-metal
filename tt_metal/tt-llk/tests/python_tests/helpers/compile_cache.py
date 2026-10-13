# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment hook: a content-addressed cache for the Wormhole perf builds' RISC-V compile step (``g++ -S``).

Enabled by ``LLK_EXP_COMPILE_CACHE=<dir>`` (test_config.py ``_build_kernel_part``); unset, nothing changes.

Key: sha256 of the compiler identity (driver and cc1plus contents), the compile flags without the preprocessor-only
ones (-D/-U/-I: their whole effect is in the preprocessed source), and the preprocessed source (``g++ -E`` with the
same flags). Value: the assembly. The assembly names the build's directories only in its debug strings (the line
table's directory and file names, the compile directory), so the variant directory and the tests directory are
replaced by tokens in the preprocessor's line markers (the key) and in the stored assembly, and put back on a hit:
legs built in different RUNNER_TEMPs or trees share entries. A path anywhere else in the source stays in the key.

Size cap ``LLK_EXP_COMPILE_CACHE_MAX_GB`` (default 5), least recently used entries go first.
``LLK_EXP_COMPILE_CACHE_LOG=<file>`` appends one line per compile: hit/miss, seconds, key, output path.
"""
import fcntl
import gzip
import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

VERSION = "llkcc-1"
_TOKEN = "@@LLKCC:{}@@"
_compiler_ids = {}
_stores = [0]


def _run(cmd, cwd, stdin_text):
    r = subprocess.run(cmd, cwd=cwd, input=stdin_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode, r.stdout, r.stderr


def _compiler_id(gxx):
    """sha256 of the driver and of the cc1plus it runs (content, once per process)"""
    if gxx not in _compiler_ids:
        h = hashlib.sha256()
        cc1 = subprocess.run([gxx, "-print-prog-name=cc1plus"], text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
        for f in (gxx, cc1):
            h.update(os.path.realpath(f).encode() + b"\0")
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        _compiler_ids[gxx] = h.hexdigest()
    return _compiler_ids[gxx]


def _key_flags(flags):
    """the flags without the preprocessor-only ones (-D, -U, -I and their separate-argument forms)"""
    out, skip = [], False
    for f in flags:
        if skip:
            skip = False
            continue
        if f in ("-D", "-U", "-I", "-isystem", "-iquote", "-idirafter"):
            skip = True
            continue
        if f.startswith(("-D", "-U", "-I", "-isystem", "-iquote", "-idirafter")):
            continue
        out.append(f)
    return out


def _paths(paths):
    """(literal, token) pairs, every spelling of each directory (as given and resolved), longest first"""
    pairs = {}
    for p, name in paths.items():
        for s in {str(p), os.path.realpath(str(p))}:
            pairs.setdefault(s.rstrip("/"), _TOKEN.format(name))
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def _normalize_markers(pre, pairs):
    """the preprocessed source with the directories replaced in line markers (# <line> "<file>" ...) only"""
    for lit, tok in pairs:
        pre = re.sub(r'^(# \d+ ")' + re.escape(lit) + r'(?=[/"])', lambda m: m.group(1) + tok, pre, flags=re.M)
    return pre


def _normalize(text, pairs):
    for lit, tok in pairs:
        text = text.replace(lit, tok)
    return text


def _denormalize(text, pairs):
    for lit, tok in pairs:
        text = text.replace(tok, lit)
    return text


def _log(line):
    f = os.environ.get("LLK_EXP_COMPILE_CACHE_LOG")
    if f:
        with open(f, "a") as fh:
            fh.write(line + "\n")


def _prune(root, cap):
    """delete least recently used entries until the cache is under 90 % of cap (one pruner at a time)"""
    with open(root / ".prune.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        entries, total = [], 0
        for sub in os.scandir(root):
            if not sub.is_dir() or len(sub.name) != 2:
                continue
            for e in os.scandir(sub.path):
                if e.name.endswith(".s.gz"):
                    st = e.stat()
                    entries.append((st.st_mtime, st.st_size, e.path))
                    total += st.st_size
        if total <= cap:
            return
        entries.sort()
        for _, size, path in entries:
            if total <= 0.9 * cap:
                break
            try:
                os.unlink(path)
                total -= size
            except OSError:
                pass


def compile_assembly(gxx, flags, cwd, source, assembly, paths):
    """the assembly text of ``gxx *flags -S -x c++ - -o assembly`` (source on stdin, run in cwd), from the cache when
    an entry for the same compiler, flags and preprocessed source exists. paths: {directory: token name} for the
    directories the build names (variant dir, tests dir). Returns the text; the assembly file is not kept on a hit."""
    root = Path(os.environ["LLK_EXP_COMPILE_CACHE"])
    t0 = time.perf_counter()
    compile_command = [gxx, *flags, "-S", "-x", "c++", "-", "-o", str(assembly)]
    pairs = _paths(paths)
    rc, pre, _ = _run([gxx, *flags, "-E", "-x", "c++", "-", "-o", "-"], cwd, source)
    t_pre = time.perf_counter() - t0
    key = None
    if rc == 0:
        h = hashlib.sha256()
        for part in (VERSION, _compiler_id(gxx), _normalize(str(cwd), pairs), os.path.basename(str(assembly)),
                     *_normalize("\0".join(_key_flags(flags)), pairs).split("\0"), _normalize_markers(pre, pairs)):
            h.update(part.encode() + b"\0\1")
        key = h.hexdigest()
        entry = root / key[:2] / f"{key}.s.gz"
        try:
            with gzip.open(entry, "rt") as f:
                text = _denormalize(f.read(), pairs)
            os.utime(entry)
            _log(f"hit {time.perf_counter() - t0:.3f} pre={t_pre:.3f} {key[:16]} {assembly}")
            return text
        except Exception:  # missing or unreadable entry (corrupt gzip raises zlib.error): compile
            pass
    r = subprocess.run(compile_command, cwd=cwd, input=source, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"Command:\n{' '.join(compile_command)}\n\nCommand's stderr:\n{r.stderr}")
    text = Path(assembly).read_text()
    t1 = time.perf_counter()
    if key is not None:
        norm = _normalize(text, pairs)
        # store only what the tokens give back exactly
        if "@@LLKCC:" not in text and _denormalize(norm, pairs) == text:
            entry.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=entry.parent, prefix=".tmp-")
            with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as f:
                f.write(norm.encode())
            os.replace(tmp, entry)
            _stores[0] += 1
            if _stores[0] % 100 == 1:
                _prune(root, float(os.environ.get("LLK_EXP_COMPILE_CACHE_MAX_GB", "5")) * (1 << 30))
    _log(f"miss {t1 - t0:.3f} pre={t_pre:.3f} {key[:16] if key else 'nokey'} {assembly}")
    return text


_layout_contexts = {}


def _file_digest(h, path):
    h.update(os.path.realpath(str(path)).encode() + b"\0")
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)


def layout_context(gxx, arch_flag, sources, words):
    """Experiment hook (``LLK_EXP_LAYOUT_CACHE=<dir>``): the subdirectory of the persistent layout cache for this
    build. perf/layout.py shares its maps and flow choices between variants by the code key of a thread's assembly;
    across builds the results also depend on what links that assembly and on the layout code itself, so they are keyed
    by: the source files given (layout.py, test_config.py, the linker scripts), python and numpy, the assembler,
    linker and libraries the driver uses, and the build's flags other than -D/-U/-I/-g (preprocessor only, or dropped
    from the link)."""
    key = (gxx, arch_flag, tuple(str(s) for s in sources), tuple(words))
    if key not in _layout_contexts:
        import sys

        import numpy

        h = hashlib.sha256(f"{VERSION} layout {sys.version} numpy {numpy.__version__}".encode())
        for s in sorted(str(s) for s in sources):
            h.update(os.path.basename(s).encode() + b"\0")
            with open(s, "rb") as fh:
                h.update(fh.read())
        for q in ("-print-prog-name=as", "-print-prog-name=ld", "-print-file-name=libc.a", "-print-file-name=libgcc.a"):
            f = subprocess.run([gxx, arch_flag, q], text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
            if not os.path.isabs(f):  # -print-prog-name gives a bare name when the program is found on PATH
                f = subprocess.run(["which", f], text=True, stdout=subprocess.PIPE).stdout.strip() or f
            _file_digest(h, f)
        h.update("\0".join(w for w in _key_flags(words) if w != "-g").encode())
        _layout_contexts[key] = h.hexdigest()[:24]
        _prune_layout(_layout_contexts[key])
    return _layout_contexts[key]


def _prune_layout(current):
    """keep the layout cache under LLK_EXP_LAYOUT_CACHE_MAX_GB (default 1): whole contexts other than this build's go,
    least recently started first (one pruner at a time)"""
    root = Path(os.environ["LLK_EXP_LAYOUT_CACHE"])
    (root / current).mkdir(parents=True, exist_ok=True)
    os.utime(root / current)
    cap = float(os.environ.get("LLK_EXP_LAYOUT_CACHE_MAX_GB", "1")) * (1 << 30)
    with open(root / ".prune.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        sizes = {}
        for d in os.scandir(root):
            if d.is_dir():
                sizes[d.name] = (d.stat().st_mtime, sum(f.stat().st_size for f in Path(d.path).rglob("*") if f.is_file()))
        total = sum(v[1] for v in sizes.values())
        for name, (_, size) in sorted(sizes.items(), key=lambda kv: kv[1][0]):
            if total <= cap:
                break
            if name != current:
                import shutil

                shutil.rmtree(root / name, ignore_errors=True)
                total -= size
