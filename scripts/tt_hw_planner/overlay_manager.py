from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_OVERLAYS_DIR = _THIS_DIR / "overlays"
_INDEX_NAME = "index.json"

_SHARED_PREFIXES = (
    "models/tt_transformers/",
    "models/common/",
    "models/perf/",
)

_DEMO_CATEGORY_DIRS = frozenset(
    {
        "multimodal",
        "vision",
        "speech_recognition",
        "audio",
        "diffusion",
    }
)


_REPO_OVERRIDE: Optional[Path] = None


@contextmanager
def using_repo(repo_dir: Path) -> Iterator[None]:
    global _REPO_OVERRIDE
    prev = _REPO_OVERRIDE
    _REPO_OVERRIDE = Path(repo_dir).resolve()
    try:
        yield
    finally:
        _REPO_OVERRIDE = prev


def _repo_root() -> Path:
    if _REPO_OVERRIDE is not None:
        return _REPO_OVERRIDE
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def classify_path(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/")
    if norm.startswith("scripts/tt_hw_planner/"):
        return "tool"
    if any(norm.startswith(p) for p in _SHARED_PREFIXES):
        return "shared"
    if norm.startswith("models/demos/"):
        parts = norm.split("/")
        for i in range(3, len(parts)):
            if parts[i] in ("tt", "tests", "demo"):
                between = parts[2:i]
                if len(between) <= 1:
                    return "shared"
                if len(between) == 2 and between[0] in _DEMO_CATEGORY_DIRS:
                    return "shared"
                return "model_local"
        return "other"
    return "other"


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _model_dir(model_id: str) -> Path:
    return _OVERLAYS_DIR / _slug(model_id)


def _index_path(model_id: str) -> Path:
    return _model_dir(model_id) / _INDEX_NAME


def _load_index(model_id: str) -> Dict[str, dict]:
    p = _index_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_index(model_id: str, idx: Dict[str, dict]) -> None:
    p = _index_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx, indent=2, sort_keys=True))


def _patch_filename(rel_path: str) -> str:
    safe = rel_path.replace("/", "__")
    return f"{safe}.patch"


@dataclass
class OverlayRecord:
    model_id: str
    rel_path: str
    patch_path: Path
    captured_ts: float
    sha256: str


def _git_diff_file(rel_path: str, *, against: str = "HEAD") -> str:
    out = subprocess.run(
        ["git", "diff", against, "--", rel_path],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def _git_apply(patch_text: str, *, reverse: bool = False, check_only: bool = False) -> Tuple[int, str]:
    args = ["git", "apply"]
    if reverse:
        args.append("--reverse")
    if check_only:
        args.append("--check")
    proc = subprocess.run(
        args,
        cwd=_repo_root(),
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stderr or proc.stdout)


def _git_checkout_file(rel_path: str) -> Tuple[int, str]:
    proc = subprocess.run(
        ["git", "checkout", "HEAD", "--", rel_path],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stderr


def capture(model_id: str, rel_path: str) -> Optional[OverlayRecord]:
    diff = _git_diff_file(rel_path)
    if not diff.strip():
        return None
    cls = classify_path(rel_path)
    if cls not in ("shared",):
        return None

    patch_name = _patch_filename(rel_path)
    md = _model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    patch_file = md / patch_name
    patch_file.write_text(diff)

    sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    idx = _load_index(model_id)
    idx[rel_path] = {
        "patch_file": patch_name,
        "captured_ts": time.time(),
        "sha256": sha,
        "line_count": diff.count("\n"),
        "classification": cls,
    }
    _save_index(model_id, idx)

    rc, err = _git_checkout_file(rel_path)
    if rc != 0:
        print(
            f"[overlay] WARNING captured {rel_path} for {model_id} "
            f"but `git checkout` failed (rc={rc}): {err.strip()}",
            file=sys.stderr,
        )

    return OverlayRecord(
        model_id=model_id,
        rel_path=rel_path,
        patch_path=patch_file,
        captured_ts=time.time(),
        sha256=sha,
    )


def list_overlays(model_id: Optional[str] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not _OVERLAYS_DIR.is_dir():
        return rows
    model_dirs = [_model_dir(model_id)] if model_id else sorted([p for p in _OVERLAYS_DIR.iterdir() if p.is_dir()])
    for md in model_dirs:
        if not md.is_dir():
            continue
        mid = md.name
        idx_path = md / _INDEX_NAME
        if not idx_path.is_file():
            continue
        try:
            idx = json.loads(idx_path.read_text())
        except Exception:
            continue
        for rel, meta in sorted(idx.items()):
            rows.append(
                {
                    "model_dir_slug": mid,
                    "rel_path": rel,
                    "patch_file": str(md / meta.get("patch_file", _patch_filename(rel))),
                    "captured_ts": meta.get("captured_ts"),
                    "line_count": meta.get("line_count"),
                    "classification": meta.get("classification"),
                }
            )
    return rows


def apply_for(model_id: str) -> Tuple[int, List[str]]:
    idx = _load_index(model_id)
    applied: List[str] = []
    if not idx:
        return 0, applied
    md = _model_dir(model_id)
    for rel, meta in idx.items():
        patch_text = (md / meta["patch_file"]).read_text()
        rc_check, _ = _git_apply(patch_text, check_only=True)
        if rc_check != 0:
            continue
        rc, err = _git_apply(patch_text)
        if rc == 0:
            applied.append(rel)
        else:
            print(
                f"[overlay] apply failed for {rel} (model={model_id}): {err.strip()}",
                file=sys.stderr,
            )
    return len(applied), applied


def revert_for(model_id: str) -> Tuple[int, List[str]]:
    idx = _load_index(model_id)
    reverted: List[str] = []
    if not idx:
        return 0, reverted
    md = _model_dir(model_id)
    for rel, meta in idx.items():
        patch_text = (md / meta["patch_file"]).read_text()
        rc, _ = _git_apply(patch_text, reverse=True, check_only=True)
        if rc != 0:
            continue
        rc, err = _git_apply(patch_text, reverse=True)
        if rc == 0:
            reverted.append(rel)
        else:
            print(
                f"[overlay] revert failed for {rel} (model={model_id}): {err.strip()}",
                file=sys.stderr,
            )
    return len(reverted), reverted


def drop(model_id: str, rel_path: str) -> bool:
    idx = _load_index(model_id)
    if rel_path not in idx:
        return False
    md = _model_dir(model_id)
    patch_file = md / idx[rel_path]["patch_file"]
    if patch_file.is_file():
        patch_file.unlink()
    del idx[rel_path]
    if idx:
        _save_index(model_id, idx)
    else:
        _index_path(model_id).unlink(missing_ok=True)
        try:
            md.rmdir()
        except OSError:
            pass
    return True


def promote(model_id: str, rel_path: str) -> Tuple[bool, str]:
    idx = _load_index(model_id)
    if rel_path not in idx:
        return False, f"no overlay for {rel_path} under {model_id}"
    md = _model_dir(model_id)
    patch_text = (md / idx[rel_path]["patch_file"]).read_text()
    rc_check, err_check = _git_apply(patch_text, check_only=True)
    if rc_check != 0:
        return False, f"patch would not apply cleanly: {err_check.strip()}"
    rc, err = _git_apply(patch_text)
    if rc != 0:
        return False, f"git apply failed: {err.strip()}"
    drop(model_id, rel_path)
    return True, f"applied {rel_path} to working tree; overlay removed"


def extract_from_working_tree(
    model_id: str,
    rel_path: str,
    *,
    hunks_matching: Optional[str] = None,
    intended_for_production: bool = False,
) -> Tuple[bool, str]:
    diff = _git_diff_file(rel_path)
    if not diff.strip():
        return False, f"no changes to {rel_path} in working tree"

    if hunks_matching:
        diff = _filter_hunks(diff, pattern=hunks_matching)
        if not diff.strip():
            return False, f"no hunks matched /{hunks_matching}/ in {rel_path}"

    rec = _store_extracted(
        model_id,
        rel_path,
        diff,
        intended_for_production=intended_for_production,
    )
    rc_rev, err = _git_apply(diff, reverse=True)
    if rc_rev != 0:
        return False, (
            f"stored overlay at {rec.patch_path.relative_to(_repo_root())} "
            f"but reverse-apply on working tree FAILED ({err.strip()}). "
            f"working tree unchanged; you may need to revert manually."
        )
    return True, (
        f"extracted to {rec.patch_path.relative_to(_repo_root())}; " f"shared file {rel_path} reverted in working tree"
    )


def _store_extracted(
    model_id: str,
    rel_path: str,
    diff: str,
    *,
    intended_for_production: bool = False,
) -> OverlayRecord:
    md = _model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    patch_file = md / _patch_filename(rel_path)
    patch_file.write_text(diff)
    sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    idx = _load_index(model_id)
    idx[rel_path] = {
        "patch_file": patch_file.name,
        "captured_ts": time.time(),
        "sha256": sha,
        "line_count": diff.count("\n"),
        "classification": classify_path(rel_path),
        "source": "extracted_from_working_tree",
        "intended_for_production": intended_for_production,
    }
    _save_index(model_id, idx)
    return OverlayRecord(
        model_id=model_id,
        rel_path=rel_path,
        patch_path=patch_file,
        captured_ts=time.time(),
        sha256=sha,
    )


def _filter_hunks(diff: str, *, pattern: str) -> str:
    rx = re.compile(pattern, re.IGNORECASE)
    lines = diff.splitlines(keepends=True)
    out: List[str] = []
    header: List[str] = []
    in_hunk = False
    hunk_buf: List[str] = []

    def flush_hunk():
        if hunk_buf and any(rx.search(ln) for ln in hunk_buf):
            out.extend(hunk_buf)

    for ln in lines:
        if ln.startswith("diff --git ") or ln.startswith("index ") or ln.startswith("--- ") or ln.startswith("+++ "):
            if in_hunk:
                flush_hunk()
                hunk_buf = []
                in_hunk = False
            header.append(ln)
            continue
        if ln.startswith("@@"):
            if in_hunk:
                flush_hunk()
                hunk_buf = []
            if header:
                out.extend(header)
                header = []
            in_hunk = True
            hunk_buf = [ln]
            continue
        if in_hunk:
            hunk_buf.append(ln)
    if in_hunk:
        flush_hunk()
    return "".join(out)
