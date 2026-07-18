"""Registry drift detection for tt_hw_planner (fixes-plan Point 2a).

The Layer-2 mapping registries point at concrete tt-metal paths:

  * ``family_backends._BACKENDS`` -> ``demo_path`` / ``smoke_test_entry``
  * ``compatibility.BUILDING_BLOCKS`` -> ``tt_path`` / ``registry_tt_path``

When the repo moves or renames those files the registries silently go stale,
which is the root cause of wrong-sibling-template and unknown-arch findings.
:func:`check_registry_drift` verifies every registered path still exists in the
checkout and (reverse) flags reusable ``tt_transformers/tt`` modules that no
registry entry references, so ``sync-registry --check`` fails loudly instead of
the planner mis-pointing at a path that no longer exists.

This is the "validate now" half of Point 2a; the AST-declared auto-generation
of the registry ("generate later") builds on the same path model.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


UPSTREAM_URL = "https://github.com/tenstorrent/tt-metal.git"
UPSTREAM_BRANCH = "main"

_SYNCED_SUBTREES = (
    "models/tt_transformers",
    "models/tt_dit",
    "models/tt_cnn",
    "models/common",
)
_PROVIDES_MARKERS = ("TT_HW_PLANNER_PROVIDES", "TT_HW_PLANNER_FAMILY")
_ROOT_MARKER = ".tt_hw_planner_root"


@dataclass(frozen=True)
class Root:
    """A scanned reusable-source root (fixes-plan Point 10).

    ``kind`` is ``'component'`` (its modules become per-component REUSE/ADAPT
    targets) or ``'family'`` (its subdirs become rankable sibling families).
    ``default`` is the classification for a MANIFEST-declared entry; heuristic
    (non-manifest) derivations are always clamped to ADAPT (never confident
    REUSE) per the Point 10 caveats."""

    path: str
    kind: str
    default: str


REUSE_ROOTS: Tuple[Root, ...] = (
    Root("models/tt_transformers/tt", "component", "reuse"),
    Root("models/common/modules", "component", "adapt"),
    Root("models/tt_dit/blocks", "component", "adapt"),
    Root("models/tt_dit/layers", "component", "adapt"),
    Root("models/tt_dit/encoders", "component", "adapt"),
    Root("models/tt_dit/models", "component", "adapt"),
    Root("models/tt_dit/pipelines", "family", "adapt"),
)


def _cache_root() -> Path:
    base = os.environ.get("TT_HW_PLANNER_CACHE") or (Path.home() / ".cache" / "tt_hw_planner")
    return Path(base)


def _sources_path() -> Path:
    return _cache_root() / "registry_sources.json"


def load_extra_sources() -> List[Root]:
    """User-registered extra scanned roots (fixes-plan Point 10b), persisted by
    ``sync-registry --add-source`` so every subsequent up/auto-up includes them
    without a tool edit. [] if none / unreadable."""
    try:
        data = json.loads(_sources_path().read_text())
    except Exception:
        return []
    out: List[Root] = []
    for e in data if isinstance(data, list) else []:
        p = (e or {}).get("path") if isinstance(e, dict) else None
        if p:
            out.append(Root(str(p).strip("/"), (e.get("kind") or "component"), (e.get("default") or "adapt")))
    return out


def add_source(path: str, kind: str = "component", default: str = "adapt") -> List[Root]:
    """Register (and persist) an extra scanned root; dedup by path. Returns the
    full persisted source list. Subsequent up/auto-up fetch + scan it."""
    path = str(path).strip().strip("/")
    existing = [r for r in load_extra_sources() if r.path != path]
    existing.append(Root(path, kind, default))
    p = _sources_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"path": r.path, "kind": r.kind, "default": r.default} for r in existing], indent=2))
    return existing


def _configured_roots() -> List[Root]:
    return list(REUSE_ROOTS) + load_extra_sources()


def _discover_marker_roots(tree_root: Path) -> List[Root]:
    """Roots self-declared in the fetched tree by a ``.tt_hw_planner_root`` marker
    (fixes-plan Point 10b), so a new source becomes usable the moment it ships the
    marker — no flag, no tool edit. Scans only ``models/``. Never raises."""
    out: List[Root] = []
    try:
        base = tree_root / "models"
        if base.is_dir():
            for m in base.rglob(_ROOT_MARKER):
                rel = str(m.parent.relative_to(tree_root))
                out.append(Root(rel, "component", "adapt"))
    except Exception:
        return []
    return out


def _fetch_path_set() -> List[str]:
    """Sparse-checkout path set: the default synced subtrees plus the top-level of
    every registered extra source (Point 10b), so extra roots are fetched too."""
    paths = list(_SYNCED_SUBTREES)
    for r in load_extra_sources():
        top = "/".join(r.path.split("/")[:2]) if r.path.startswith("models/") else r.path
        if top and top not in paths:
            paths.append(top)
    return paths


@dataclass
class UpstreamTree:
    """The tree ``refresh_registry`` / drift-check run against for one run.

    ``source`` is ``'remote'`` when a pinned upstream snapshot was fetched, or
    ``'local'`` when we fell back to the checkout (offline / no network / fetch
    failure). ``stale`` is True whenever ``source == 'local'`` — the caller
    prints a loud warning so an air-gapped run never silently claims freshness.
    """

    root: Path
    sha: str
    source: str
    stale: bool


@dataclass
class DriftIssue:
    """A single registry/tree mismatch.

    kind == 'missing_path' : a registered path no longer exists (hard drift).
    kind == 'unmapped'     : a reusable tt module exists but nothing maps it (soft).
    """

    kind: str
    where: str
    path: str
    detail: str = ""


def _extract_path(raw: Optional[str]) -> Optional[str]:
    """Leading repo-relative path token of a registry path field, or None.

    Returns None for non-filesystem sentinels the registries also store in
    these fields — bare markers like ``"HF"`` and op references like
    ``"ttnn.argmax"`` (no path separator) are not checkable paths. Handles
    annotations (``"a/b.py (wraps c/d.py)"``) and trailing-slash directories.
    """
    if not raw:
        return None
    tok = raw.strip().split(" ")[0].strip().rstrip("/")
    if not tok or "/" not in tok:
        return None
    return tok


def _registered_paths() -> List[tuple]:
    """Every (where, path) the deterministic registries point at."""
    from .compatibility import BUILDING_BLOCKS
    from .family_backends import all_backends

    out: List[tuple] = []
    for b in all_backends():
        for fld in ("demo_path", "smoke_test_entry"):
            p = _extract_path(getattr(b, fld, None))
            if p:
                out.append((f"family_backends[{b.name}].{fld}", p))
    for bb in BUILDING_BLOCKS:
        for fld in ("tt_path", "registry_tt_path"):
            p = _extract_path(getattr(bb, fld, None))
            if p:
                out.append((f"building_blocks[{bb.name}].{fld}", p))
    try:
        from .reuse_registry import all_entries

        for e in all_entries():
            p = _extract_path(getattr(e, "tt_path", None))
            if p:
                out.append((f"reuse_registry[{e.concept}].tt_path", p))
    except Exception:
        pass
    return out


def check_registry_drift(repo_root, include_unmapped: bool = True, unmapped_root=None) -> List[DriftIssue]:
    """Return every registry/tree mismatch under ``repo_root``.

    Hard drift (``missing_path``) is a registered path that no longer exists in
    the local checkout (the paths the tool will actually load). Soft drift
    (``unmapped``) is a reusable module that no registry entry references — a
    hint a new building block needs mapping. When ``unmapped_root`` is given
    (e.g. a fetched upstream snapshot from :func:`fetch_upstream_models`), the
    unmapped scan runs against IT so new upstream modules surface even when the
    local checkout is stale. Never raises; a missing tree yields no hints.
    """
    root = Path(repo_root)
    issues: List[DriftIssue] = []
    referenced: set = set()

    for where, rel in _registered_paths():
        referenced.add(rel)
        if not (root / rel).exists():
            issues.append(DriftIssue("missing_path", where, rel, "registered path does not exist in the checkout"))

    if include_unmapped:
        scan_root = Path(unmapped_root) if unmapped_root else root
        for base, recursive in (
            ("models/tt_transformers/tt", False),
            ("models/tt_dit", True),
            ("models/tt_cnn", True),
        ):
            d = scan_root / base
            if not d.is_dir():
                continue
            files = d.rglob("*.py") if recursive else d.glob("*.py")
            for f in sorted(files):
                if f.name.startswith("_") or f.name == "__init__.py":
                    continue
                rel = str(f.relative_to(scan_root))
                if "/tests/" in rel or "/test/" in rel:
                    continue
                if not any(rel == r or rel.startswith(r + "/") or r.startswith(rel) for r in referenced):
                    issues.append(
                        DriftIssue(
                            "unmapped",
                            base.split("/", 1)[-1],
                            rel,
                            "reusable module not referenced by any registry entry",
                        )
                    )

    return issues


def format_drift(issues: List[DriftIssue]) -> str:
    """Human-readable drift report (grouped: hard drift first, then hints)."""
    if not issues:
        return "registry OK — every mapped path exists; no unmapped reusable modules."
    missing = [i for i in issues if i.kind == "missing_path"]
    unmapped = [i for i in issues if i.kind == "unmapped"]
    lines: List[str] = []
    if missing:
        lines.append(f"STALE registry paths ({len(missing)}) — these no longer exist in the checkout:")
        for i in missing:
            lines.append(f"  [MISSING] {i.where}\n            -> {i.path}")
    if unmapped:
        by_base: Dict[str, List[DriftIssue]] = {}
        for i in unmapped:
            by_base.setdefault(i.where, []).append(i)
        lines.append(f"Unmapped reusable modules ({len(unmapped)}) — present in the tree, no registry entry:")
        for base, items in sorted(by_base.items()):
            lines.append(f"  {base}: {len(items)} module(s) unmapped")
            for i in items[:5]:
                lines.append(f"    [unmapped] {i.path}")
            if len(items) > 5:
                lines.append(
                    f"    … and {len(items) - 5} more (a whole base like tt_dit unmapped = add its modules to the reuse map)"
                )
    return "\n".join(lines)


def has_hard_drift(issues: List[DriftIssue]) -> bool:
    """True if any registered path is missing (the loud-failure condition)."""
    return any(i.kind == "missing_path" for i in issues)


def _git(args: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def resolve_upstream_sha(timeout: int = 20) -> Optional[str]:
    """Resolve ``tenstorrent/tt-metal`` ``main`` to a concrete commit sha.

    A single cheap ``git ls-remote`` — this is the per-run pin that reconciles
    "always latest" with "deterministic" (same sha -> same registry). Returns
    None on any network/timeout failure so the caller falls back to local.
    """
    try:
        cp = _git(["ls-remote", UPSTREAM_URL, f"refs/heads/{UPSTREAM_BRANCH}"], timeout=timeout)
    except Exception:
        return None
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    return cp.stdout.split()[0].strip() or None


def _local_head_sha(repo_root: Path) -> str:
    try:
        cp = _git(["rev-parse", "HEAD"], cwd=repo_root, timeout=10)
        if cp.returncode == 0:
            return cp.stdout.strip()[:40] or "LOCAL"
    except Exception:
        pass
    return "LOCAL"


def fetch_upstream_models(repo_root, offline: bool = False, timeout: int = 90) -> UpstreamTree:
    """Fetch a pinned snapshot of upstream's reusable-module subtrees.

    Remote-first (fixes-plan Point 2a): resolve ``main`` to a sha, then
    sha-cached shallow+sparse fetch of the subtrees the registry actually reads
    (``tt_transformers``/``tt_dit``/``tt_cnn``/``common``). A sha already cached
    is reused with no network. ``offline=True`` (or ``TT_HW_PLANNER_OFFLINE``),
    no network, or any git failure falls back to the local checkout with
    ``stale=True``. NEVER raises — bring-up must not be blocked by a fetch.
    """
    repo_root = Path(repo_root)
    if offline or os.environ.get("TT_HW_PLANNER_OFFLINE"):
        return UpstreamTree(repo_root, _local_head_sha(repo_root), "local", True)

    sha = resolve_upstream_sha()
    if not sha:
        return UpstreamTree(repo_root, _local_head_sha(repo_root), "local", True)

    dest = _cache_root() / "upstream" / sha
    marker = dest / ".fetch_ok"
    if marker.exists():
        return UpstreamTree(dest, sha, "remote", False)

    try:
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / ".git").is_dir():
            if _git(["init", "-q"], cwd=dest, timeout=20).returncode != 0:
                raise RuntimeError("git init failed")
            _git(["remote", "add", "origin", UPSTREAM_URL], cwd=dest, timeout=20)
        _git(["config", "core.sparseCheckout", "true"], cwd=dest, timeout=20)
        sparse = dest / ".git" / "info" / "sparse-checkout"
        sparse.parent.mkdir(parents=True, exist_ok=True)
        sparse.write_text("\n".join(f"{s}/*" for s in _fetch_path_set()) + "\n")
        if _git(["fetch", "--depth", "1", "origin", sha], cwd=dest, timeout=timeout).returncode != 0:
            raise RuntimeError("git fetch failed")
        if _git(["checkout", "-q", "FETCH_HEAD"], cwd=dest, timeout=timeout).returncode != 0:
            raise RuntimeError("git checkout failed")
        marker.write_text(sha)
        return UpstreamTree(dest, sha, "remote", False)
    except Exception:
        return UpstreamTree(repo_root, _local_head_sha(repo_root), "local", True)


def hydrate_upstream_into_repo(repo_root, tree, *, overwrite: bool = True) -> List[str]:
    """Copy every synced upstream subtree (:func:`_fetch_path_set` — currently
    ``tt_transformers``/``tt_dit``/``tt_cnn``/``common`` plus any ``--add-source``
    roots) from the pinned fresh-main cache snapshot into ``repo_root``.

    The registry sync only ever wrote the fetched modules to the cache and derived
    a name/family overlay; the actual REUSE (import canonical-wrapper) and ADAPT
    (verbatim copy) porting resolves its source under ``repo_root``. So a sibling
    added to main after this checkout's fork point was discoverable-by-name but its
    code never reached the tree the bring-up ports against. Hydrating the fetched
    subtrees into ``repo_root`` makes those new siblings physically present, so the
    existing import/copy path picks them up with no other change.

    No-op unless a fresh remote snapshot was fetched (``tree.source == 'remote'``);
    on an offline/local fallback the tree IS ``repo_root`` and there is nothing to
    copy. Intended for a disposable isolation worktree — callers must not point it
    at a user's real checkout. ``overwrite`` also refreshes drifted existing files
    (use latest); set False for add-only. Returns the repo-relative files hydrated;
    never raises.
    """
    hydrated: List[str] = []
    try:
        if getattr(tree, "source", "") != "remote":
            return hydrated
        src_root = Path(getattr(tree, "root", ""))
        dst_root = Path(repo_root)
        if not src_root.is_dir() or src_root.resolve() == dst_root.resolve():
            return hydrated
        for sub in _fetch_path_set():
            src_sub = src_root / sub
            if not src_sub.is_dir():
                continue
            for f in src_sub.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(src_root)
                out = dst_root / rel
                if out.exists() and not overwrite:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                hydrated.append(str(rel))
    except Exception:
        pass
    return hydrated


def _read_provides(py: Path) -> List[dict]:
    """AST-read module-level ``TT_HW_PLANNER_PROVIDES`` / ``_FAMILY`` dict literals.

    Reusable modules / demo families self-declare their concept, category,
    HF class patterns and pipeline tags in one of these markers; ``refresh_registry``
    collects them into the generated overlay. AST-only — no import, no code exec.
    """
    out: List[dict] = []
    try:
        tree = ast.parse(py.read_text(errors="replace"))
    except Exception:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in _PROVIDES_MARKERS:
                try:
                    val = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(val, dict):
                    out.append(val)
                elif isinstance(val, (list, tuple)):
                    out.extend(d for d in val if isinstance(d, dict))
    return out


def _module_top_classes(py: Path) -> List[str]:
    """Top-level class names in a module (AST, no import/exec). [] on error."""
    try:
        tree = ast.parse(py.read_text(errors="replace"))
    except Exception:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef)]


def _derive_reuse_concepts(root: Path) -> List[dict]:
    """Derive per-module REUSE/ADAPT targets from the fetched reusable-module
    tree so a component's bring-up can wrap an already-implemented upstream
    module instead of writing it from scratch (fixes-plan Points 2a/10).

    Scans the ``component`` roots in :data:`REUSE_ROOTS` (the LLM stack
    ``tt_transformers/tt`` PLUS the v2 sources ``common/modules`` and
    ``tt_dit/{blocks,layers,encoders,models}``), any ``--add-source`` roots, and
    ``.tt_hw_planner_root``-marked dirs. Each heuristic candidate is status ADAPT
    (never a trusted REUSE — Point 10 caveat): the loop wraps it, emits a PCC
    test, and the LLM refines on PCC failure, so a wrong auto-match degrades to
    NEW rather than silently claiming a wrong module. Keyed by file basename as
    the concept; consumers match by concept overlap.
    """
    out: List[dict] = []
    roots = [r for r in _configured_roots() if r.kind == "component"] + _discover_marker_roots(root)
    for r in roots:
        d = root / r.path
        if not d.is_dir():
            continue
        for py in sorted(d.rglob("*.py")):
            rel = str(py.relative_to(root))
            if py.name.startswith("_") or py.name == "__init__.py":
                continue
            if "/tests/" in rel or "/test/" in rel or "/experimental/" in rel:
                continue
            classes = _module_top_classes(py)
            if not classes:
                continue
            out.append(
                {
                    "concept": py.stem.lower().strip("_"),
                    "tt_path": rel,
                    "tt_class": max(classes, key=len),
                    "status": "ADAPT",
                    "default_status": r.default,
                    "source": "auto-derived-upstream",
                }
            )
    return out


_PIPELINE_CATEGORY_HINTS = (
    ("wan", "Video"),
    ("mochi", "Video"),
    ("ltx", "Video"),
    ("video", "Video"),
    ("audio", "TTS"),
    ("music", "TTS"),
    ("flux", "Image"),
    ("stable_diffusion", "Image"),
    ("sd3", "Image"),
    ("qwenimage", "Image"),
    ("qwen_image", "Image"),
    ("sana", "Image"),
)

_PIPELINE_TAGS = {"Image": ["text-to-image"], "Video": ["text-to-video"], "TTS": ["text-to-audio"]}


def _derive_family_roots_candidates(root: Path) -> List[dict]:
    """Derive rankable sibling families from ``family`` roots — i.e. each subdir
    of ``tt_dit/pipelines`` (flux1, wan, mochi, ltx, sd35, qwenimage, …) becomes a
    candidate so scaffold picks a real diffusion/video/audio DiT sibling instead
    of SD1.4/hf_eager (fixes-plan Point 10a). Category + pipeline_tag inferred
    from the pipeline name (default Image for tt_dit); model_type_keys = dir name.
    Low-confidence: ranked below curated, dropped on name/model_type collision."""
    out: List[dict] = []
    for r in [x for x in _configured_roots() if x.kind == "family"]:
        d = root / r.path
        if not d.is_dir():
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            name = sub.name
            if name.startswith("_") or name in ("tests", "test"):
                continue
            low = name.lower()
            cat = next((c for k, c in _PIPELINE_CATEGORY_HINTS if k in low), "Image")
            out.append(
                {
                    "kind": "family",
                    "name": f"tt_dit/{name} (auto-upstream)",
                    "category": cat,
                    "demo_path": f"{r.path}/{name}",
                    "routing_mode": "template",
                    "canonical_hf_id": None,
                    "model_type_keys": [low],
                    "pipeline_tags": _PIPELINE_TAGS.get(cat, []),
                    "notes": "auto-derived tt_dit pipeline (fixes-plan Point 10); low-confidence sibling for diffusion/video/audio DiT, ranked below curated.",
                    "source": "auto-derived-upstream",
                }
            )
    return out


_DEMO_CATEGORY_HINTS = (
    ("whisper", "STT"),
    ("speech", "STT"),
    ("wav2vec", "STT"),
    ("tts", "TTS"),
    ("xtts", "TTS"),
    ("vocoder", "TTS"),
    ("stable_diffusion", "Image"),
    ("diffusion", "Image"),
    ("flux", "Image"),
    ("sana", "Image"),
    ("video", "Video"),
    ("segformer", "CNN"),
    ("yolo", "CNN"),
    ("resnet", "CNN"),
    ("vgg", "CNN"),
    ("unet", "CNN"),
    ("mobilenet", "CNN"),
    ("vit", "CNN"),
    ("sam", "CNN"),
    ("bert", "NLP"),
    ("roberta", "NLP"),
    ("llama", "LLM"),
    ("qwen", "LLM"),
    ("mistral", "LLM"),
    ("mixtral", "LLM"),
    ("gemma", "LLM"),
    ("phi", "LLM"),
    ("falcon", "LLM"),
    ("mamba", "LLM"),
)


def _list_upstream_demos(git_root: Path) -> List[str]:
    """Top-level demo directory names under ``models/demos`` at the fetched sha,
    via ``git ls-tree`` (names only — no content checkout). [] if unavailable."""
    try:
        cp = _git(["ls-tree", "-d", "--name-only", "HEAD", "models/demos/"], cwd=git_root, timeout=20)
    except Exception:
        return []
    if cp.returncode != 0:
        return []
    names: List[str] = []
    for line in cp.stdout.splitlines():
        line = line.strip().rstrip("/")
        if line:
            names.append(Path(line).name)
    return names


def _derive_demo_families(demo_names: List[str]) -> List[dict]:
    """Derive low-confidence sibling family candidates from fetched demo dir
    names (fixes-plan Point 2a) so a brand-new upstream demo is auto-proposed to
    ``rank_backends`` without a hand-edit. Conservative: only when a category can
    be inferred; ``model_type_keys`` = the dir name, so an exact model_type match
    surfaces the demo, and it never overrides a curated entry (dropped on name/
    model_type collision by the family overlay loader)."""
    out: List[dict] = []
    for name in demo_names:
        low = name.lower()
        cat = next((c for k, c in _DEMO_CATEGORY_HINTS if k in low), None)
        if not cat:
            continue
        out.append(
            {
                "kind": "family",
                "name": f"{name} (auto-upstream)",
                "category": cat,
                "demo_path": f"models/demos/{name}",
                "routing_mode": "template",
                "canonical_hf_id": None,
                "model_type_keys": [low],
                "pipeline_tags": [],
                "notes": "auto-derived upstream demo (fixes-plan Point 2a); low-confidence sibling candidate ranked below curated entries.",
                "source": "auto-derived-upstream",
            }
        )
    return out


def _overlay_path() -> Path:
    return _cache_root() / "registry_overlay.json"


def refresh_registry(tree_root, sha: str = "LOCAL") -> dict:
    """Regenerate the overlay registry by walking the (fetched) tree.

    Collects self-declared ``TT_HW_PLANNER_PROVIDES``/``_FAMILY`` manifests from
    the synced subtrees and writes ``registry_overlay.json``. The overlay is a
    pure SUPPLEMENT — loaders add only entries a static list doesn't already
    cover, so an empty overlay (today: no module declares a marker yet) is a
    no-op. Records the pinned ``sha`` so a run is reproducible. Never raises.
    """
    root = Path(tree_root)
    families: List[dict] = []
    concepts: List[dict] = []
    try:
        for sub in _SYNCED_SUBTREES:
            d = root / sub
            if not d.is_dir():
                continue
            for py in sorted(d.rglob("*.py")):
                rel = str(py.relative_to(root))
                if "/tests/" in rel or "/test/" in rel:
                    continue
                for m in _read_provides(py):
                    m = dict(m)
                    m.setdefault("tt_path", rel)
                    (families if m.get("kind") == "family" else concepts).append(m)
    except Exception:
        pass

    declared_concept_paths = {c.get("tt_path") for c in concepts}
    declared_family_names = {f.get("name") for f in families}
    try:
        for c in _derive_reuse_concepts(root):
            if c["tt_path"] not in declared_concept_paths:
                concepts.append(c)
    except Exception:
        pass
    try:
        for f in _derive_demo_families(_list_upstream_demos(root)):
            if f["name"] not in declared_family_names:
                families.append(f)
    except Exception:
        pass
    try:
        have = {f.get("name") for f in families}
        for f in _derive_family_roots_candidates(root):
            if f["name"] not in have:
                families.append(f)
    except Exception:
        pass

    overlay = {"sha": sha, "families": families, "concepts": concepts}
    try:
        p = _overlay_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(overlay, indent=2, sort_keys=True))
    except Exception:
        pass
    return overlay


def load_generated_overlay() -> dict:
    """Read the last-written overlay (supplement layer). ``{}`` if none/unreadable."""
    try:
        return json.loads(_overlay_path().read_text())
    except Exception:
        return {}
