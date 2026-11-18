# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Continuation pre-warm, once per kernel cache (update 4 item B; bench/packaging/update4/UPDATE4_PLAN.md §B, addendum 1
§3.1, LEAD_DECISIONS_PKG P101/P102/P106). Package-only, pure Python (no ttnn import): the boot classes
(package_boot_pb4, package_boot_s3pb_ring) pass the warm itself and a device synchronize in.

A first-sight continuation bucket costs ~1 s of kernel JIT on a host whose kernel cache lacks it (P36; B0). The JIT
output persists in the kernel cache (TT_METAL_CACHE=/cache in the container, the host's tt-model cache), so the warm runs
when no valid marker exists for this profile, this code and this kernel cache, and writes one when it completes.

Marker: <TT_METAL_CACHE>/tt-metal-cache<build key>/gemma4_contwarm/<profile>-<code hash[:16]>.done, inside the build-key
dir the warm wrote to, so a deleted build-key dir takes the marker with it (critic 05fcb318 requirement 1). JSON: the
profile, the full code hash, up to JIT_SAMPLE kernel-cache paths the warm created (relative to the build-key dir), the
pass count, the new-file count and the seconds. GEMMA4_CONT_PREWARM:
  auto   (default) warm unless a marker for this profile and code hash parses, matches, and every recorded JIT path
         still exists (a truncated or unparsable marker, or a removed JIT output, means re-warm: N2);
  off    never warm;
  force  always warm (and rewrite the marker).
Where the marker goes after a completed warm: the single build-key dir whose kernels/ gained files; if none gained files
(everything was already compiled) and exactly one <root>/tt-metal-cache*/kernels exists, that one with an empty JIT
list (P102 N1); otherwise (two candidates) no marker, logged. A warm that raises writes nothing. Written atomically
(a temp file, then os.replace). The code hash covers the shipped sources that decide what the warm compiles: this
package's tt/ (.py, .cpp, .h, .hpp) and the profile's engine package tree."""

import glob
import hashlib
import json
import os
import time
from pathlib import Path

ENV = "GEMMA4_CONT_PREWARM"
MODES = ("auto", "off", "force")
JIT_SAMPLE = 32
SOURCE_SUFFIXES = (".py", ".cpp", ".h", ".hpp")
PKG_DIR = Path(__file__).resolve().parent.parent  # models/demos/gemma4_31b_qb2


def mode():
    m = os.environ.get(ENV, "auto")
    if m not in MODES:
        raise ValueError(f"{ENV}={m!r}: expected one of {MODES}")
    return m


def code_hash(engine_package, pkg_dir=PKG_DIR):
    """sha256 over (relative path, sha256) of every source file in tt/ and the engine package, in sorted order"""
    h = hashlib.sha256()
    for sub in ("tt", engine_package):
        root = Path(pkg_dir) / sub
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts:
                rel = p.relative_to(pkg_dir).as_posix()
                h.update(f"{rel}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def cache_root():
    root = os.environ.get("TT_METAL_CACHE")
    return Path(root) if root else None


def build_dirs(root):
    """The <root>/tt-metal-cache* dirs (build keys); those with a kernels/ subdir are kernel caches."""
    return sorted(Path(p) for p in glob.glob(str(Path(root) / "tt-metal-cache*")) if Path(p).is_dir())


def kernel_files(root):
    out = set()
    for bd in build_dirs(root):
        k = bd / "kernels"
        if k.is_dir():
            for dp, dns, fns in os.walk(k):
                for f in fns:
                    out.add(os.path.join(dp, f))
    return out


def marker_name(profile, chash):
    return f"{profile}-{chash[:16]}.done"


def find_valid_marker(root, profile, chash):
    """(path, None) for a valid marker, else (None, the reason to warm)"""
    reasons = []
    for bd in build_dirs(root):
        p = bd / "gemma4_contwarm" / marker_name(profile, chash)
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text())
            ok = m.get("profile") == profile and m.get("code_hash") == chash and isinstance(m.get("jit"), list)
        except (OSError, ValueError, AttributeError):
            ok, m = False, None
        if not ok:
            reasons.append(f"{p}: unparsable or mismatched")
            continue
        missing = [j for j in m["jit"] if not (bd / j).exists()]
        if missing:
            reasons.append(f"{p}: {len(missing)} recorded JIT outputs missing")
            continue
        return p, None
    return None, "; ".join(reasons) or "no marker"


def write_marker(build_dir, profile, chash, jit_rel, passes, new_files, seconds):
    d = Path(build_dir) / "gemma4_contwarm"
    d.mkdir(parents=True, exist_ok=True)
    final = d / marker_name(profile, chash)
    tmp = d / (final.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"profile": profile, "code_hash": chash, "jit": sorted(jit_rel)[:JIT_SAMPLE],
                               "passes": passes, "new_files": new_files, "seconds": round(seconds, 1),
                               "written": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    os.replace(tmp, final)
    return final


def place_marker(root, before, after):
    """(build dir or None, [JIT paths relative to it], reason): where a completed warm's marker goes"""
    new = sorted(after - before)
    owners = {}
    for f in new:
        rel = Path(f).relative_to(root)
        owners.setdefault(root / rel.parts[0], []).append(Path(f))
    if len(owners) == 1:
        bd, files = next(iter(owners.items()))
        return bd, [f.relative_to(bd).as_posix() for f in files], "the build dir the warm wrote to"
    if not owners:
        kdirs = [bd for bd in build_dirs(root) if (bd / "kernels").is_dir()]
        if len(kdirs) == 1:
            return kdirs[0], [], "no new files; the single kernel cache (N1)"
        return None, [], f"no new files and {len(kdirs)} kernel caches"
    return None, [], f"new files in {len(owners)} build dirs: {sorted(str(o) for o in owners)}"


def run(profile, engine_package, warm, synchronize, log=print, pkg_dir=PKG_DIR):
    """Warm if needed. warm() returns the pass count; synchronize() waits for the device (and so the JIT). Returns a
    dict describing what happened (the boot class logs it)."""
    t_check = time.perf_counter()
    m = mode()
    root = cache_root()
    if m == "off":
        return {"warmed": False, "why": f"{ENV}=off", "check_s": round(time.perf_counter() - t_check, 3)}
    chash = code_hash(engine_package, pkg_dir)
    if root is None:
        return {"warmed": False, "why": "TT_METAL_CACHE unset: no kernel cache to key a marker to"}
    if m == "auto":
        found, why = find_valid_marker(root, profile, chash)
        if found:  # a later boot: this check is the whole cost (the B gate: later boots add <= 10 s)
            res = {"warmed": False, "why": f"marker {found}", "check_s": round(time.perf_counter() - t_check, 3)}
            log(f"continuation pre-warm ({profile}): {res}")
            return res
    else:
        why = f"{ENV}=force"
    t0 = time.perf_counter()
    before = kernel_files(root)
    passes = warm()
    synchronize()
    after = kernel_files(root)
    seconds = time.perf_counter() - t0
    bd, jit_rel, where = place_marker(root, before, after)
    marker = write_marker(bd, profile, chash, jit_rel, passes, len(after - before), seconds) if bd else None
    res = {"warmed": True, "why": why, "passes": passes, "new_files": len(after - before), "seconds": round(seconds, 1),
           "marker": str(marker) if marker else None, "marker_place": where, "code_hash": chash[:16]}
    log(f"continuation pre-warm ({profile}): {res}")
    return res
