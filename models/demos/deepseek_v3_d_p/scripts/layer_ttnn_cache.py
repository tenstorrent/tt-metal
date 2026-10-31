#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the prefill TTNN weight cache for one decoder layer (or a range) and compare it with the Weka cache.

Runs on a mock 8x4 Blackhole Galaxy (no chips needed), writes only under --out-root (in $HOME), reads the
Weka cache and the HF weights read-only, prints a per-stage time profile and writes a JSON report.

    python models/demos/deepseek_v3_d_p/scripts/layer_ttnn_cache.py --model kimi_k2_7 --layers 5
    python models/demos/deepseek_v3_d_p/scripts/layer_ttnn_cache.py --model glm_5_3 --layers 0-10

Exit codes: 0 all PASS, 1 DIFF, 2 guard or usage error, 3 build FAIL, 4 Weka changed, 130 Ctrl-C.
"""

import argparse
import contextlib
import functools
import gc
import inspect
import json
import os
import re
import resource
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

WEKA_ROOTS = ("/mnt/weka",)
WEKA_FSTYPES = ("wekafs",)
REPO_ROOT = Path(__file__).resolve().parents[4]
MOCK_DESC = (
    "tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
)
SEQ_LEN = 1024
MESH_SHAPE = (8, 4)
NUM_DEVICES = 32
CHUNK = 64 << 20
GIB = 1 << 30

# Mirrors the adapter attributes; the adapter modules import ttnn, so the guards cannot read them.
# run_phase checks this map against the adapter after import.
MODELS = {
    "kimi_k2_7": {
        "num_layers": 61,
        "num_dense_layers": 1,
        "ttnn_cache_env": "TT_KIMI_PREFILL_TTNN_CACHE",
        "env_var": "KIMI_K2_7_HF_MODEL",
        "ttnn_cache_default": "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill",
        "hf_model_default": "/mnt/weka/model-weights/llm/moonshotai/Kimi-K2.7-Code-dequantized",
    },
    "glm_5_3": {
        "num_layers": 78,
        "num_dense_layers": 3,
        "ttnn_cache_env": "TT_GLM53_PREFILL_TTNN_CACHE",
        "env_var": "GLM53_HF_MODEL",
        "ttnn_cache_default": "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill",
        "hf_model_default": "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4",
    },
}

# Locations tt-metal, ttnn, transformers or Python write to.
ENV_WRITE_VARS = (
    "TT_METAL_CACHE",
    "TT_METAL_LOGS_PATH",
    "TT_METAL_HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTHONPYCACHEPREFIX",
)
TTNN_CONFIG_KEYS = ("cache_path", "model_cache_path", "tmp_dir", "root_report_path")

SUBMODULES = ("norms", "mla", "indexer", "moe_gate", "routed_experts", "shared_expert", "dense_ffn")
FILE_STAGES = (
    ("_norm_weight_", "norms"),
    (".mla.indexer_", "indexer"),
    (".mla.", "mla"),
    (".gate.", "moe_gate"),
    (".routed_expert.", "routed_experts"),
    (".shared_expert.", "shared_expert"),
    (".ffn.", "dense_ffn"),
)
PROFILE_ROWS = (
    "startup",
    "hf_read",
    "assemble",
    *SUBMODULES,
    "other",
    "build_total",
    "check_complete",
    "validate.fileset",
    "validate.content",
    "cleanup",
)
STATUSES = ("EQUAL", "EQUAL_DATA", "DATA_DIFF", "META_DIFF", "COMPARE_ERROR")
MATCH_STATUSES = ("EQUAL", "EQUAL_DATA")

AUDIT_PATH_EVENTS = {
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.truncate",
    "os.chmod",
    "os.chown",
    "os.utime",
    "os.link",
    "shutil.rmtree",
    "shutil.move",
}
OPEN_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


class GuardError(Exception):
    pass


class FileSet(NamedTuple):
    common: list
    missing_on_weka: list
    weka_only: list


class FileResult(NamedTuple):
    status: str
    detail: str


# ---------------------------------------------------------------- CLI


def parse_layers(spec: str, num_layers: int) -> list[int]:
    def as_int(s):
        if not (s.isascii() and s.isdigit()):
            raise ValueError(f"bad layer spec {spec!r}: {s!r} is not a layer number")
        return int(s)

    if not spec or not spec.strip():
        raise ValueError("empty layer spec")
    layers = set()
    for part in spec.strip().split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = as_int(a), as_int(b)
            if lo > hi:
                raise ValueError(f"bad layer range {part!r}: start > end")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(as_int(part))
    too_big = sorted(n for n in layers if n >= num_layers)
    if too_big:
        raise ValueError(f"layers {too_big} out of range: the model has {num_layers} layers")
    return sorted(layers)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=sorted(MODELS), help="Registry key of the model")
    p.add_argument("--layers", required=True, help="N, A-B (inclusive) or A,B,C")
    p.add_argument("--out-root", default="~/ttnn-layer-cache", help="Output root; must be under $HOME, not on Weka")
    p.add_argument("--weka-cache", default=None, help="Reference cache root (read only); default from the adapter")
    p.add_argument("--hf-model", default=None, help="HF checkpoint dir (read only); default from the adapter")
    p.add_argument("--keep", action="store_true", help="Keep the built files (default: delete after validation)")
    p.add_argument(
        "--min-free-gb", type=float, default=40, help="Free GiB that must remain on the output fs after a layer"
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------- path guards


def _unescape_mount(s: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def read_mounts(text: str) -> list[tuple[str, str]]:
    mounts = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((_unescape_mount(fields[1]), fields[2]))
    return mounts


def _nearest_existing(path: Path) -> Path:
    p = Path(os.path.abspath(path))
    while not p.exists() and p != p.parent:
        p = p.parent
    return p.resolve()


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def fstype_of(path: Path, mounts) -> str:
    p = _nearest_existing(path)
    best, best_len = "", -1
    for mountpoint, fstype in mounts:
        mp = Path(mountpoint)
        if _is_under(p, mp) and len(mountpoint) > best_len:
            best, best_len = fstype, len(mountpoint)
    return best


def is_weka(path: Path, mounts, roots=WEKA_ROOTS) -> bool:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    resolved = absolute.resolve()
    for root in roots:
        for r in {Path(root), Path(root).resolve()}:
            if _is_under(absolute, r) or _is_under(resolved, r):
                return True
    return fstype_of(resolved, mounts) in WEKA_FSTYPES


def check_out_root(out_root: str, home: Path, mounts, roots=WEKA_ROOTS) -> Path:
    out = Path(out_root).expanduser().resolve()
    home = Path(home).resolve()
    if is_weka(home, mounts, roots):
        raise GuardError(f"home {home} is on Weka")
    if is_weka(out, mounts, roots):
        raise GuardError(f"--out-root {out_root} resolves to {out}, which is on Weka")
    if home not in out.parents:
        raise GuardError(f"--out-root {out_root} resolves to {out}, which is not under {home}")
    return out


def check_env_paths(environ: Mapping, mounts, roots=WEKA_ROOTS) -> None:
    for var in ENV_WRITE_VARS:
        value = environ.get(var)
        if value and is_weka(Path(value), mounts, roots):
            raise GuardError(f"${var}={value} is on Weka")


def layer_files(directory: Path, n: int) -> dict[str, int]:
    prefix = f"layer_{n}."
    if not os.path.isdir(directory):
        return {}
    with os.scandir(directory) as it:
        return {e.name: e.stat().st_size for e in it if e.name.startswith(prefix) and e.is_file()}


# ---------------------------------------------------------------- runtime guards


def make_audit_hook(roots=WEKA_ROOTS):
    root_strs = tuple(sorted({os.path.abspath(r) for r in roots} | {os.path.realpath(r) for r in roots}))

    def on_weka(p) -> bool:
        if not isinstance(p, (str, bytes, os.PathLike)):
            return False
        a = os.path.abspath(os.fsdecode(p))
        for candidate in (a, os.path.realpath(a)):
            if any(candidate == r or candidate.startswith(r + os.sep) for r in root_strs):
                return True
        return False

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & OPEN_WRITE_FLAGS
            )
            if writes and on_weka(path):
                raise PermissionError(f"Weka is read-only: open({path!r}, {mode!r})")
        elif event == "os.symlink":
            if on_weka(args[1]):
                raise PermissionError(f"Weka is read-only: os.symlink(..., {args[1]!r})")
        elif event in AUDIT_PATH_EVENTS:
            for a in args:
                if on_weka(a):
                    raise PermissionError(f"Weka is read-only: {event}({a!r})")

    return hook


def make_dump_guard(orig, out: Path):
    out = Path(out).resolve()

    def guarded(file_name, *args, **kwargs):
        target = Path(os.fsdecode(file_name)).resolve()
        if out not in target.parents:
            raise PermissionError(f"tensor dump outside {out}: {file_name}")
        return orig(file_name, *args, **kwargs)

    return guarded


def check_ttnn_config(paths: Mapping, mounts, roots=WEKA_ROOTS) -> None:
    for key, value in paths.items():
        if value is None or str(value) == "":
            continue
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        if is_weka(p, mounts, roots):
            raise GuardError(f"ttnn.CONFIG.{key}={value} is on Weka")


# ---------------------------------------------------------------- file set and content compare


def split_fileset(built, weka) -> FileSet:
    b, w = set(built), set(weka)
    return FileSet(sorted(b & w), sorted(b - w), sorted(w - b))


def header_size(path) -> int:
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path}: shorter than the 8-byte header size")
    return struct.unpack("<Q", raw)[0]


def _data_offset(path) -> int:
    hs = header_size(path)
    size = os.path.getsize(path)
    if 8 + hs > size:
        raise ValueError(f"{path}: truncated header (header_size={hs}, file_size={size})")
    return 8 + hs


def _first_diff(fa, fb, length: int, chunk: int):
    pos = 0
    while pos < length:
        n = min(chunk, length - pos)
        a, b = fa.read(n), fb.read(n)
        if len(a) != n or len(b) != n:
            raise ValueError(f"short read at offset {pos}")
        if a != b:
            return pos + next(i for i in range(n) if a[i] != b[i])
        pos += n
    return None


def files_equal(a, b, chunk: int = CHUNK) -> bool:
    size = os.path.getsize(a)
    if size != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return _first_diff(fa, fb, size, chunk) is None


def first_data_diff(a, b, chunk: int = CHUNK):
    """None when the data regions match, else a description of the first difference."""
    off_a, off_b = _data_offset(a), _data_offset(b)
    len_a, len_b = os.path.getsize(a) - off_a, os.path.getsize(b) - off_b
    if len_a != len_b:
        return f"data size {len_a} != {len_b}"
    with open(a, "rb") as fa, open(b, "rb") as fb:
        fa.seek(off_a)
        fb.seek(off_b)
        pos = _first_diff(fa, fb, len_a, chunk)
    return None if pos is None else f"first data diff at data offset {pos}"


def data_equal(a, b, chunk: int = CHUNK) -> bool:
    return first_data_diff(a, b, chunk) is None


def compare_tensorbin(built, weka, load_meta, chunk: int = CHUNK) -> FileResult:
    try:
        if files_equal(built, weka, chunk):
            return FileResult("EQUAL", "")
        diff = first_data_diff(built, weka, chunk)
        if diff is not None:
            return FileResult("DATA_DIFF", diff)
        meta_b, meta_w = load_meta(built), load_meta(weka)
        meta_diffs = [
            f"{k}: {meta_b.get(k)} != {meta_w.get(k)}"
            for k in sorted(set(meta_b) | set(meta_w))
            if meta_b.get(k) != meta_w.get(k)
        ]
        if meta_diffs:
            return FileResult("META_DIFF", "; ".join(meta_diffs))
        return FileResult("EQUAL_DATA", f"header {header_size(built)} B vs {header_size(weka)} B")
    except Exception as e:
        return FileResult("COMPARE_ERROR", f"COMPARE_ERROR: {type(e).__name__}: {e}")


def tensor_meta(path) -> dict[str, str]:
    import ttnn

    t = ttnn._ttnn.tensor.load_tensor_flatbuffer(str(path), None)
    topology = t.tensor_topology()
    return {
        "shape": str(t.shape),
        "padded_shape": str(t.padded_shape),
        "dtype": str(t.dtype),
        "layout": str(t.layout),
        "tile": str(t.tile),
        "shard_count": str(len(topology.mesh_coords())),
        "tensor_topology": str(topology),
    }


# ---------------------------------------------------------------- verdict and Weka snapshot


def layer_verdict(build_error, complete_ok: bool, fileset: FileSet, results: Mapping) -> str:
    if build_error or not complete_ok:
        return "FAIL"
    if fileset.missing_on_weka or any(r.status not in MATCH_STATUSES for r in results.values()):
        return "DIFF"
    return "PASS"


def exit_code(verdicts, weka_changed, interrupted: bool) -> int:
    if weka_changed:
        return 4
    if interrupted:
        return 130
    if "FAIL" in verdicts:
        return 3
    if "DIFF" in verdicts:
        return 1
    return 0


def weka_snapshot(directory: Path, n: int) -> dict:
    prefix = f"layer_{n}."
    files = {}
    with os.scandir(directory) as it:
        for e in it:
            if e.name.startswith(prefix):
                st = e.stat()
                files[e.name] = [st.st_size, st.st_mtime_ns]
    return {"dir_mtime_ns": os.stat(directory).st_mtime_ns, "files": files}


def snapshot_diff(before: dict, after: dict) -> list[str]:
    lines = []
    if before["dir_mtime_ns"] != after["dir_mtime_ns"]:
        lines.append(f"dir mtime_ns {before['dir_mtime_ns']} -> {after['dir_mtime_ns']}")
    b, a = before["files"], after["files"]
    lines += [f"added {name}" for name in sorted(set(a) - set(b))]
    lines += [f"removed {name}" for name in sorted(set(b) - set(a))]
    for name in sorted(set(a) & set(b)):
        if a[name] != b[name]:
            lines.append(f"changed {name}: size {b[name][0]} -> {a[name][0]}, mtime_ns {b[name][1]} -> {a[name][1]}")
    return lines


# ---------------------------------------------------------------- time profile


def stage_for_file(name: str) -> str:
    for pattern, stage in FILE_STAGES:
        if pattern in name:
            return stage
    return "other"


class StageTimer:
    """Exclusive time per label for nested wrapped calls.

    TtFfn.build_ttnn_cache only delegates to TtSharedExpert.build_ttnn_cache, so a shared_expert call
    directly inside dense_ffn is charged to dense_ffn.
    """

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self._stack = []

    def wrap(self, fn, label: str):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            charged = label
            if label == "shared_expert" and self._stack and self._stack[-1][0] == "dense_ffn":
                charged = "dense_ffn"
            frame = [charged, self.clock(), 0.0]
            self._stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                self._stack.pop()
                elapsed = self.clock() - frame[1]
                self.totals[charged] += elapsed - frame[2]
                self.calls[charged] += 1
                if self._stack:
                    self._stack[-1][2] += elapsed

        return timed

    def exclusive(self) -> dict[str, float]:
        return dict(self.totals)


def install_timers(timer: StageTimer, targets) -> list:
    saved = []
    for cls, label in targets:
        own = cls.__dict__.get("build_ttnn_cache")
        raw = own if own is not None else inspect.getattr_static(cls, "build_ttnn_cache")
        if isinstance(raw, staticmethod):
            wrapped = staticmethod(timer.wrap(raw.__func__, label))
        elif isinstance(raw, classmethod):
            wrapped = classmethod(timer.wrap(raw.__func__, label))
        else:
            wrapped = timer.wrap(raw, label)
        saved.append((cls, own))
        setattr(cls, "build_ttnn_cache", wrapped)
    return saved


def restore_timers(saved) -> None:
    for cls, own in reversed(saved):
        if own is None:
            delattr(cls, "build_ttnn_cache")
        else:
            setattr(cls, "build_ttnn_cache", own)


def _fmt_time(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def _content_line(content: Mapping, header_only: int) -> str:
    return " ".join(f"{s}={content.get(s, 0)}" for s in STATUSES) + f" header_only_diffs={header_only}"


def format_profile(layer: dict) -> str:
    stages = layer["stages"]
    kind = layer["kind"] + (", indexer" if layer.get("has_indexer") else "")
    lines = [
        f"=== {layer['model']} layer {layer['layer']} ({kind}) -> {layer['out_dir']} ===",
        f"{'stage':<18}{'time_s':>8}{'files':>8}{'bytes':>16}",
    ]
    called = set(layer.get("called", ()))
    for row in PROFILE_ROWS:
        s = stages[row]
        t = _fmt_time(s["time_s"])
        if row in SUBMODULES or row == "other":
            if row not in called and not s["files"]:
                continue
            lines.append(f"{row:<18}{t:>8}{s['files']:>8}{s['bytes']:>16,}")
        elif row == "build_total":
            lines.append(f"{row:<18}{t:>8}{layer['files_built']:>8}{layer['bytes_built']:>16,}")
        elif row == "check_complete":
            lines.append(f"{row:<18}{t:>8}   {'ok' if layer['check_complete'] else 'INCOMPLETE'}")
        elif row == "validate.fileset":
            fs = layer["fileset"]
            lines.append(
                f"{row:<18}{t:>8}   common={fs['common']} missing_on_weka={len(fs['missing_on_weka'])} "
                f"weka_only={len(fs['weka_only'])}"
            )
        elif row == "validate.content":
            lines.append(f"{row:<18}{t:>8}   {_content_line(layer['content'], layer['header_only_diffs'])}")
        elif row == "cleanup":
            lines.append(f"{row:<18}{t:>8}   deleted {layer['deleted_files']} files")
        else:
            lines.append(f"{row:<18}{t:>8}{'-':>8}{'-':>16}")
    lines.append(f"{'peak_rss_gb':<18}{layer['peak_rss_gb']:>8.1f}")
    lines.append(f"{'verdict':<18}{layer['verdict']}")
    if layer.get("error"):
        lines.append("error: " + layer["error"].strip().splitlines()[-1])
    lines += [f"MISSING on weka: {name}" for name in layer["fileset"]["missing_on_weka"]]
    for name, r in sorted(layer["files"].items()):
        if r["status"] not in MATCH_STATUSES:
            lines.append(f"{r['status']}: {name}: {r['detail']}")
    lines += [f"INFO weka_only: {name}" for name in layer["fileset"]["weka_only"]]
    return "\n".join(lines)


def _content_short(content: Mapping) -> str:
    return " ".join(f"{content[s]} {s}" for s in STATUSES if content.get(s)) or "-"


def format_summary(reports: list, code: int) -> str:
    head = (
        f"{'layer':<6}{'kind':<7}{'build_s':>8}{'valid_s':>8}{'bytes_GB':>9}{'files':>6}{'miss':>5}"
        f"{'weka_only':>10}{'hdr_only':>9}{'routed_s':>9}{'mla_s':>7}  {'content':<20}  verdict"
    )
    lines = [head]
    for r in reports:
        lines.append(
            f"{r['layer']:<6}{r['kind']:<7}{r['build_s']:>8.1f}{r['valid_s']:>8.1f}{r['bytes_built'] / 1e9:>9.2f}"
            f"{r['files_built']:>6}{len(r['fileset']['missing_on_weka']):>5}{len(r['fileset']['weka_only']):>10}"
            f"{r['header_only_diffs']:>9}{r['stages']['routed_experts']['time_s']:>9.1f}"
            f"{r['stages']['mla']['time_s']:>7.1f}  {_content_short(r['content']):<20}  {r['verdict']}"
        )
    total_bytes = sum(r["bytes_built"] for r in reports)
    peak = max((r["bytes_built"] for r in reports), default=0)
    lines.append(
        f"total  {len(reports)} layers  build_s {sum(r['build_s'] for r in reports):.1f}  "
        f"valid_s {sum(r['valid_s'] for r in reports):.1f}  {total_bytes / 1e9:.2f} GB written "
        f"(peak layer {peak / 1e9:.2f} GB)  exit {code}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- run


@dataclass
class Ctx:
    args: argparse.Namespace
    model: str
    layers: list
    out: Path
    out_cache: Path
    weka_dir: Path
    hf_model: Path
    mounts: list
    weka_bytes: dict = field(default_factory=dict)


def _disk_check(path: Path, needed: int, min_free_gb: float) -> None:
    free = shutil.disk_usage(_nearest_existing(path)).free
    if free - needed < min_free_gb * GIB:
        raise GuardError(
            f"disk: {free / GIB:.1f} GiB free on {path}, need {needed / GIB:.1f} GiB "
            f"+ {min_free_gb:g} GiB reserve (--min-free-gb)"
        )


def guard_phase(args) -> Ctx:
    mounts = read_mounts(Path("/proc/self/mounts").read_text())
    out = check_out_root(args.out_root, Path.home().resolve(), mounts)
    check_env_paths(os.environ, mounts)

    m = MODELS[args.model]
    weka_root = args.weka_cache or os.environ.get(m["ttnn_cache_env"]) or m["ttnn_cache_default"]
    weka_dir = Path(weka_root).expanduser() / f"{args.model}_bh_32dev" / "8x4"
    if not weka_dir.is_dir():
        raise GuardError(f"Weka cache dir missing: {weka_dir}")
    hf_model = Path(args.hf_model or os.environ.get(m["env_var"]) or m["hf_model_default"]).expanduser()
    if not (hf_model / "model.safetensors.index.json").is_file():
        raise GuardError(f"no model.safetensors.index.json in {hf_model}")

    try:
        layers = parse_layers(args.layers, m["num_layers"])
    except ValueError as e:
        raise GuardError(str(e)) from None

    out_cache = out / args.model / f"{args.model}_bh_32dev" / "8x4"
    if out_cache.resolve() == weka_dir.resolve():
        raise GuardError(f"output dir and reference dir are the same: {out_cache}")
    weka_bytes = {}
    for n in layers:
        files = layer_files(weka_dir, n)
        if not files:
            raise GuardError(f"no layer_{n}. files in {weka_dir}")
        weka_bytes[n] = sum(files.values())
    dirty = [n for n in layers if layer_files(out_cache, n)]
    if dirty:
        raise GuardError(f"out not clean: {out_cache} already holds files for layers {dirty}")
    needed = sum(weka_bytes.values()) if args.keep else max(weka_bytes.values())
    _disk_check(out_cache, needed, args.min_free_gb)
    return Ctx(args, args.model, layers, out, out_cache, weka_dir, hf_model, mounts, weka_bytes)


def mock_desc_path() -> Path:
    return Path(os.environ.get("TT_METAL_HOME") or REPO_ROOT) / MOCK_DESC


def open_mock_mesh():
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    if mesh.get_num_devices() != NUM_DEVICES or mesh.arch() != ttnn.Arch.BLACKHOLE:
        n, arch = mesh.get_num_devices(), mesh.arch()
        ttnn.close_mesh_device(mesh)
        raise RuntimeError(f"mock mesh is {n} x {arch}, expected {NUM_DEVICES} x BLACKHOLE")
    return mesh


@dataclass
class Runtime:
    ttnn: object
    mesh: object
    config: object
    model_cfg: object
    lazy: object
    pfx: str
    startup_s: float
    weka_changed: list = field(default_factory=list)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _new_layer_report(ctx: Ctx, n: int, weka: dict) -> dict:
    return {
        "model": ctx.model,
        "out_dir": str(ctx.out_cache),
        "layer": n,
        "kind": "dense" if n < MODELS[ctx.model]["num_dense_layers"] else "moe",
        "has_indexer": any(".mla.indexer_" in name for name in weka),
        "verdict": "FAIL",
        "build_s": 0.0,
        "valid_s": 0.0,
        "stages": {row: {"time_s": 0.0, "files": 0, "bytes": 0} for row in PROFILE_ROWS},
        "called": [],
        "files_built": 0,
        "bytes_built": 0,
        "weka_bytes": sum(weka.values()),
        "check_complete": False,
        "fileset": {"common": 0, "missing_on_weka": [], "weka_only": []},
        "content": {s: 0 for s in STATUSES},
        "header_only_diffs": 0,
        "files": {},
        "peak_rss_gb": 0.0,
        "deleted_files": 0,
        "error": None,
    }


def _timed(stages: dict, label: str):
    @contextlib.contextmanager
    def cm():
        t0 = time.perf_counter()
        try:
            yield
        finally:
            stages[label]["time_s"] += time.perf_counter() - t0

    return cm()


def build_one_layer(ctx: Ctx, rt: Runtime, n: int, rep: dict) -> None:
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGatePrefill
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
    from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
    from models.demos.deepseek_v3_d_p.utils.test_utils import convert_state_dict
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import block_cache_state_dict

    ttnn = rt.ttnn
    stages = rep["stages"]
    before = weka_snapshot(ctx.weka_dir, n)
    layer_sd = layer_dequant = state_dict = None
    try:
        try:
            with _timed(stages, "hf_read"):
                layer_sd = sub_state_dict(rt.lazy, f"{rt.pfx}model.layers.{n}.")
                layer_dequant = convert_state_dict(layer_sd, rt.config)
            is_dense = n < rt.config.first_k_dense_replace
            if is_dense != (n < rt.model_cfg.NUM_DENSE_LAYERS):
                raise RuntimeError(
                    f"dense rule mismatch: first_k_dense_replace={rt.config.first_k_dense_replace}, "
                    f"NUM_DENSE_LAYERS={rt.model_cfg.NUM_DENSE_LAYERS}"
                )
            with _timed(stages, "assemble"):
                state_dict = block_cache_state_dict(layer_dequant, is_dense, rt.config.n_routed_experts)

            timer = StageTimer()
            targets = [
                (TtDistributedRmsNorm, "norms"),
                (ttMLA, "mla"),
                (TtIndexer, "indexer"),
                (TtMoEGatePrefill, "moe_gate"),
                (TtRoutedExpert, "routed_experts"),
                (TtSharedExpert, "shared_expert"),
                (TtFfn, "dense_ffn"),
            ]
            saved = install_timers(timer, targets)
            try:
                with _timed(stages, "build_total"):
                    TtPrefillBlock.build_ttnn_cache(
                        state_dict=state_dict,
                        layer_idx=n,
                        cache_path=ctx.out_cache,
                        mesh_device=rt.mesh,
                        config=rt.config,
                        model_cfg=rt.model_cfg,
                        seq_len=SEQ_LEN,
                    )
            finally:
                restore_timers(saved)
                excl = timer.exclusive()
                for label in SUBMODULES:
                    stages[label]["time_s"] = excl.get(label, 0.0)
                rep["called"] = sorted(label for label, c in timer.calls.items() if c)
                stages["other"]["time_s"] = max(0.0, stages["build_total"]["time_s"] - sum(excl.values()))
        except Exception:
            rep["error"] = traceback.format_exc()
            print(rep["error"], file=sys.stderr, flush=True)
        finally:
            state_dict = None

        with _timed(stages, "check_complete"):
            try:
                init_checker(ctx.out_cache)
                rep["check_complete"] = bool(
                    TtPrefillBlock.check_cache_complete(
                        ctx.out_cache,
                        n,
                        n < rt.model_cfg.NUM_DENSE_LAYERS,
                        experts_per_chip=rt.model_cfg.NUM_ROUTED_EXPERTS // rt.mesh.get_num_devices(),
                        model_cfg=rt.model_cfg,
                        routed_expert_weights_dtype=ttnn.bfloat4_b,
                    )
                )
            except Exception:
                rep["check_complete"] = False

        with _timed(stages, "validate.fileset"):
            built = layer_files(ctx.out_cache, n)
            weka = layer_files(ctx.weka_dir, n)
            fs = split_fileset(built, weka)
        rep["fileset"] = {"common": len(fs.common), "missing_on_weka": fs.missing_on_weka, "weka_only": fs.weka_only}
        rep["files_built"], rep["bytes_built"] = len(built), sum(built.values())
        for name, size in built.items():
            s = stages[stage_for_file(name)]
            s["files"] += 1
            s["bytes"] += size

        results = {}
        with _timed(stages, "validate.content"):
            for name in fs.common:
                results[name] = compare_tensorbin(ctx.out_cache / name, ctx.weka_dir / name, tensor_meta)
        rep["files"] = {name: {"status": r.status, "detail": r.detail} for name, r in results.items()}
        rep["content"] = {s: sum(r.status == s for r in results.values()) for s in STATUSES}
        rep["header_only_diffs"] = rep["content"]["EQUAL_DATA"]
        rep["verdict"] = layer_verdict(rep["error"], rep["check_complete"], fs, results)
    except Exception:
        rep["error"] = rep["error"] or traceback.format_exc()
        rep["verdict"] = "FAIL"
        print(rep["error"], file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        rep["error"] = rep["error"] or "KeyboardInterrupt"
        rep["verdict"] = "FAIL"
        raise
    finally:
        changed = snapshot_diff(before, weka_snapshot(ctx.weka_dir, n))
        if changed:
            rt.weka_changed += [f"layer_{n}: {line}" for line in changed]
            for line in changed:
                print(f"WEKA CHANGED: {line}", flush=True)
        with _timed(stages, "cleanup"):
            if not ctx.args.keep:
                for name in layer_files(ctx.out_cache, n):
                    os.remove(ctx.out_cache / name)
                    rep["deleted_files"] += 1
        if layer_sd is not None:
            for k in list(layer_sd.keys()):
                rt.lazy.evict(k)
        layer_sd = layer_dequant = None
        gc.collect()
        rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
        rep["build_s"] = sum(stages[k]["time_s"] for k in ("hf_read", "assemble", "build_total"))
        rep["valid_s"] = stages["validate.fileset"]["time_s"] + stages["validate.content"]["time_s"]


def _start_runtime(ctx: Ctx, t0: float) -> Runtime:
    import ttnn

    check_ttnn_config({k: getattr(ttnn.CONFIG, k, None) for k in TTNN_CONFIG_KEYS}, ctx.mounts)
    try:
        mesh = open_mock_mesh()
    except Exception as e:
        raise GuardError(f"mock mesh did not open ({mock_desc_path()}): {e}") from None
    try:
        from models.demos.common.prefill.adapter import get_adapter
        from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
        from models.demos.deepseek_v3_d_p.utils.test_utils import detect_language_model_prefix

        adapter = get_adapter(ctx.model)
        expected = MODELS[ctx.model]
        actual = {
            "num_layers": adapter.model_config.NUM_LAYERS,
            "num_dense_layers": adapter.model_config.NUM_DENSE_LAYERS,
            "ttnn_cache_env": adapter.ttnn_cache_env,
            "env_var": adapter.env_var,
            "ttnn_cache_default": adapter.ttnn_cache_default,
            "hf_model_default": adapter.hf_model_default,
        }
        mismatch = {k: (expected[k], actual[k]) for k in expected if str(expected[k]) != str(actual[k])}
        if mismatch:
            raise GuardError(f"script model map differs from the adapter: {mismatch}")
        config = adapter.load_hf_config()
        lazy = LazyStateDict(ctx.hf_model)
        pfx = detect_language_model_prefix(lazy)
    except BaseException:
        ttnn.close_mesh_device(mesh)
        raise
    return Runtime(ttnn, mesh, config, adapter.model_config, lazy, pfx, time.perf_counter() - t0)


def run_phase(ctx: Ctx) -> int:
    sys.addaudithook(make_audit_hook(WEKA_ROOTS))
    desc = mock_desc_path()
    if "TT_METAL_MOCK_CLUSTER_DESC_PATH" not in os.environ and not desc.is_file():
        print(f"GUARD: mock cluster descriptor missing: {desc}", file=sys.stderr)
        return 2
    os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", str(desc))
    os.environ.pop("TT_METAL_WATCHER", None)
    os.environ["PREFILL_HF_MODEL"] = str(ctx.hf_model)

    reports_dir = ctx.out / ctx.model / "reports"
    spec = ctx.args.layers.replace(",", "_")
    report_path = reports_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_L{spec}.json"
    report = {
        "model": ctx.model,
        "layers_spec": ctx.args.layers,
        "layers": ctx.layers,
        "argv": " ".join(sys.argv),
        "host": socket.gethostname(),
        "git_commit": _git_commit(),
        "started_utc": _utc_now(),
        "finished_utc": None,
        "out_dir": str(ctx.out_cache),
        "weka_dir": str(ctx.weka_dir),
        "hf_model": str(ctx.hf_model),
        "startup_s": None,
        "exit_code": None,
        "interrupted": False,
        "weka_changed": [],
        "per_layer": [],
    }

    def finish(code: int, interrupted: bool = False) -> int:
        report["exit_code"] = code
        report["interrupted"] = interrupted
        report["finished_utc"] = _utc_now()
        reports_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, report)
        if len(report["per_layer"]) > 1:
            print(format_summary(report["per_layer"], code), flush=True)
        if report["weka_changed"]:
            print("WEKA CHANGED (the script does not claim the cause):", flush=True)
            for line in report["weka_changed"]:
                print(f"  {line}", flush=True)
        print(f"report: {report_path}", flush=True)
        return code

    t0 = time.perf_counter()
    import ttnn

    orig_dump = ttnn._ttnn.tensor.dump_tensor_flatbuffer
    ttnn._ttnn.tensor.dump_tensor_flatbuffer = make_dump_guard(orig_dump, ctx.out_cache)
    try:
        try:
            rt = _start_runtime(ctx, t0)
        except GuardError as e:
            print(f"GUARD: {e}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return finish(130, interrupted=True)
        except Exception:
            traceback.print_exc()
            return finish(3)
        report["startup_s"] = rt.startup_s
        reports_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i, n in enumerate(ctx.layers):
                if not ctx.args.keep:
                    _disk_check(ctx.out_cache, ctx.weka_bytes[n], ctx.args.min_free_gb)
                rep = _new_layer_report(ctx, n, layer_files(ctx.weka_dir, n))
                rep["stages"]["startup"]["time_s"] = rt.startup_s if i == 0 else None
                report["per_layer"].append(rep)
                try:
                    build_one_layer(ctx, rt, n, rep)
                finally:
                    report["weka_changed"] = list(rt.weka_changed)
                    _write_json_atomic(report_path, report)
                print(format_profile(rep), flush=True)
        except KeyboardInterrupt:
            return finish(exit_code([], report["weka_changed"], True), interrupted=True)
        except GuardError as e:
            print(f"GUARD: {e}", file=sys.stderr)
            return finish(4 if report["weka_changed"] else 2)
        finally:
            rt.ttnn.close_mesh_device(rt.mesh)
            rt.lazy.close()
    finally:
        ttnn._ttnn.tensor.dump_tensor_flatbuffer = orig_dump

    verdicts = [r["verdict"] for r in report["per_layer"]]
    return finish(exit_code(verdicts, report["weka_changed"], False))


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        ctx = guard_phase(args)
    except GuardError as e:
        print(f"GUARD: {e}", file=sys.stderr)
        return 2
    try:
        return run_phase(ctx)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
