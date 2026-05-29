from __future__ import annotations

from .discovery import safe_relative_to_root

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from .family_backends import FamilyBackend


_EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", "build", "dist", ".cache"}
_EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a"}
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".rst",
    ".txt",
    ".sh",
    ".cfg",
    ".ini",
    ".toml",
}


def _marker_for_category(category: str) -> str:
    """Return the terminal output marker the category's
    :mod:`correctness` comparator expects to see in pytest stdout.

    Single source of truth: :data:`correctness.engine._COMPLETED_DEMO_MARKERS`
    already enumerates these. We just pick the right one for the
    backend's category so the skeleton demo emits something the gate
    can recognise.
    """
    from .correctness.engine import _COMPLETED_DEMO_MARKERS

    cat_upper = (category or "").upper()
    if cat_upper.startswith("CNN/SEGMENT") or cat_upper == "SEGMENTATION":
        prefix = "==SEG 0 -"
    elif cat_upper.startswith("CNN/DETECT") or cat_upper == "DETECTION":
        prefix = "==DET 0 -"
    elif cat_upper.startswith("CNN") or cat_upper in {"IMAGE", "VIDEO"}:
        prefix = "==CLASS 0 -"
    elif cat_upper in {"STT", "TTS"}:
        prefix = "==ASR 0 -"
    elif cat_upper in {"EMBED", "EMBEDDING"}:
        prefix = "==EMBED 0 -"
    else:
        prefix = "==USER 0 -"
    for m in _COMPLETED_DEMO_MARKERS:
        if m.startswith(prefix):
            return m
    return "==USER 0 - OUTPUT"


_SKELETON_DEMO_TEMPLATE = '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCAFFOLD-TODO: auto-generated skeleton demo for `{model_id}`.

This file was emitted by ``scaffold_demo_folder`` because the backend
``{backend_name}`` has ``use_module_tree=True`` and no sibling template
existed at ``{backend_demo_path}``. It runs the HF reference once on
CPU (no tt-metal device traffic yet) and prints the category marker
(``{marker}``) so the planner's correctness gate can validate output
shape and the iterate loop has a runnable starting point.

To graduate this demo onto tt-metal, the per-component bring-up loop
(``up --auto``) will progressively replace the HF forward call with
TTNN components under ``_stubs/`` and ``tt/``. The marker emission
stays intact -- the correctness comparator reads it regardless of
whether the underlying compute is CPU or device.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import pytest


HF_MODEL_ID = "{model_id}"
CATEGORY = "{category}"
MARKER = "{marker}"


def _emit_marker(payload: Any) -> None:
    print(MARKER)
    if payload is None:
        print("(no payload)")
        return
    try:
        import numpy as np

        if hasattr(payload, "detach"):
            arr = payload.detach().to("cpu").to(dtype_=None) if False else payload.detach().cpu().float().numpy()
        else:
            arr = np.asarray(payload)
        buf = io.BytesIO()
        np.save(buf, arr)
        print(base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception as exc:
        print(f"(marker payload serialization failed: {{type(exc).__name__}}: {{exc}})")
        print(repr(payload)[:512])


@pytest.mark.parametrize("device_params", [{{}}], indirect=True)
def test_demo(device_params, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_MODEL", HF_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.eval()
    sentence = os.environ.get("TT_PLANNER_PROBE_INPUT", "The quick brown fox jumps over the lazy dog.")
    inputs = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    if MARKER.startswith("==EMBED"):
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and isinstance(out, (tuple, list)) and out:
            hidden = out[0]
        if hidden is not None:
            _emit_marker(hidden[0, -1, :])
        else:
            _emit_marker(None)
    elif MARKER.startswith("==CLASS"):
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out
        _emit_marker(logits[0])
    else:
        last = getattr(out, "last_hidden_state", None)
        _emit_marker(last[0, -1, :] if last is not None else None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__ + "::test_demo", "-svv"]))
'''


def _render_skeleton_demo_folder(
    *,
    backend: FamilyBackend,
    new_model_id: str,
) -> Tuple[List[Tuple[Path, bytes, Path]], List[str], List[str]]:
    new_tail = new_model_id.split("/")[-1]
    new_slug = _slug(new_tail)
    new_dir_rel = Path(backend.demo_path).parent / new_slug
    marker = _marker_for_category(backend.category)
    body = _SKELETON_DEMO_TEMPLATE.format(
        model_id=new_model_id,
        category=backend.category,
        backend_name=backend.name,
        backend_demo_path=backend.demo_path,
        marker=marker,
    )
    # Align to tt-metal repo standard: demo.py lives under <model>/demo/demo.py
    # (matches qwen3_vl/demo/demo.py, bert/demo/demo.py, etc.).
    creates: List[Tuple[Path, bytes, Path]] = [
        (
            new_dir_rel / "demo" / "demo.py",
            body.encode("utf-8"),
            Path("(skeleton — no sibling source)"),
        ),
        (
            new_dir_rel / "demo" / "__init__.py",
            b"# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.\n" b"# SPDX-License-Identifier: Apache-2.0\n",
            Path("(skeleton — no sibling source)"),
        ),
        (
            new_dir_rel / "__init__.py",
            b"# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.\n" b"# SPDX-License-Identifier: Apache-2.0\n",
            Path("(skeleton — no sibling source)"),
        ),
    ]
    warnings = [
        f"no sibling source at `{backend.demo_path}`; emitted skeleton "
        f"demo for category `{backend.category}` (marker={marker}). "
        f"Use the iterate loop to graduate components from CPU to TT."
    ]
    return creates, [], warnings


def _slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "model"


def _backend_template_slug(backend: FamilyBackend) -> Optional[str]:
    parts = Path(backend.demo_path).parts
    if not parts:
        return None
    return parts[-1].lower() or None


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


def _should_skip_dir(path: Path) -> bool:
    return path.name in _EXCLUDE_DIR_NAMES


def _should_skip_file(path: Path) -> bool:
    if path.suffix.lower() in _EXCLUDE_FILE_SUFFIXES:
        return True
    return False


def _rewrite_text(
    src_text: str,
    *,
    sibling_slug: str,
    new_slug: str,
    sibling_hf_id: Optional[str],
    new_hf_id: str,
    is_demo_entry: bool,
) -> str:
    if not sibling_slug:
        return src_text

    new_lower = new_slug.lower()
    new_upper = new_slug.upper()
    new_title = "".join(part.capitalize() for part in new_slug.split("_")) or new_slug

    sib_lower = sibling_slug.lower()
    sib_upper = sibling_slug.upper()
    sib_title = "".join(part.capitalize() for part in sibling_slug.split("_")) or sibling_slug

    out = src_text
    out = re.sub(rf"(?<![A-Z]){re.escape(sib_title)}(?![a-z])", new_title, out)
    out = re.sub(rf"(?<![A-Z0-9_]){re.escape(sib_upper)}(?![A-Z0-9_])", new_upper, out)
    out = re.sub(rf"(?<![a-z0-9_]){re.escape(sib_lower)}(?![a-z0-9])", new_lower, out)

    if sibling_hf_id:
        out = out.replace(sibling_hf_id, new_hf_id)

    if is_demo_entry:
        banner = (
            "# ============================================================================\n"
            f"# TODO: scaffold from `{sibling_slug}` for `{new_hf_id}` — adapt model-specific\n"
            "#       wiring (loader, preprocess, head, output decode) before this passes.\n"
            f"#       Track adaptation points by searching for `SCAFFOLD-TODO` in this folder.\n"
            "# ============================================================================\n"
        )
        out = banner + out
    return out


def _rewrite_relative_path(rel: Path, sibling_slug: str, new_slug: str) -> Path:
    new_parts: List[str] = []
    pattern = re.compile(rf"(?i)(?:^|(?<=[_./\-]))({re.escape(sibling_slug)})(?:$|(?=[_./\-]))")
    for part in rel.parts:
        renamed = pattern.sub(new_slug, part)
        new_parts.append(renamed)
    return Path(*new_parts)


def _demo_entry_files(rel: Path) -> bool:
    name = rel.name.lower()
    return (
        name.startswith("demo")
        or name.startswith("test_demo")
        or name == "model_config.py"
        or name == "common.py"
        or rel.parts[:1] == ("demo",)
    )


def collect_demo_folder_changes(
    *,
    backend: FamilyBackend,
    new_model_id: str,
    repo_root: Path,
) -> Tuple[List[Tuple[Path, bytes, Path]], List[str], List[str]]:
    src_dir = (repo_root / backend.demo_path).resolve()
    new_tail = new_model_id.split("/")[-1]
    new_slug = _slug(new_tail)
    new_dir_rel = Path(backend.demo_path).parent / new_slug
    target_dir = (repo_root / new_dir_rel).resolve()

    needs_skeleton = not src_dir.is_dir() or src_dir == target_dir
    if needs_skeleton:
        if bool(getattr(backend, "use_module_tree", False)):
            return _render_skeleton_demo_folder(
                backend=backend,
                new_model_id=new_model_id,
            )
        return ([], [f"backend demo path missing on disk: {backend.demo_path}"], [])

    sibling_slug = _backend_template_slug(backend) or ""

    skipped: List[str] = []
    warnings: List[str] = []

    # Skip if a demo.py already exists at EITHER the legacy path
    # (<model>/demo.py) OR the new standard (<model>/demo/demo.py).
    legacy_demo = target_dir / "demo.py"
    standard_demo = target_dir / "demo" / "demo.py"
    if legacy_demo.is_file():
        skipped.append(f"{new_dir_rel.as_posix()}/demo.py already present (legacy path) — leaving untouched")
        return ([], skipped, warnings)
    if standard_demo.is_file():
        skipped.append(f"{new_dir_rel.as_posix()}/demo/demo.py already present — leaving untouched")
        return ([], skipped, warnings)

    if not sibling_slug:
        warnings.append("could not derive template slug from backend.demo_path; identifier rewriting skipped")
    if backend.canonical_hf_id is None:
        warnings.append("backend has no canonical_hf_id; HF id rewriting skipped")

    creates: List[Tuple[Path, bytes, Path]] = []
    for src_path in src_dir.rglob("*"):
        if src_path.is_dir():
            if _should_skip_dir(src_path):
                continue
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in src_path.relative_to(src_dir).parts):
            continue
        if _should_skip_file(src_path):
            continue

        rel = src_path.relative_to(src_dir)
        renamed_rel = _rewrite_relative_path(rel, sibling_slug, new_slug) if sibling_slug else rel
        target_rel = new_dir_rel / renamed_rel

        if _is_text_file(src_path):
            try:
                text = src_path.read_text()
            except UnicodeDecodeError:
                text = None
            if text is not None:
                new_text = _rewrite_text(
                    text,
                    sibling_slug=sibling_slug,
                    new_slug=new_slug,
                    sibling_hf_id=backend.canonical_hf_id,
                    new_hf_id=new_model_id,
                    is_demo_entry=_demo_entry_files(rel),
                )
                creates.append((target_rel, new_text.encode("utf-8"), safe_relative_to_root(src_path)))
                continue

        creates.append((target_rel, src_path.read_bytes(), safe_relative_to_root(src_path)))

    return creates, skipped, warnings
