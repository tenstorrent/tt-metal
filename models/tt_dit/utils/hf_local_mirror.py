# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Populate the HuggingFace cache from a local mirror of HF repos, so loads need no hub access.

The mirror holds repo snapshots at `<root>/<org>/<repo>`, as `hf download --local-dir` writes them:
plain files plus `.cache/huggingface/download/<file>.metadata`, whose first two lines are the commit
and the file's etag. Those are exactly the keys of the HF cache layout, so `ensure_cached` builds a
cache entry that links into the mirror instead of downloading:

    <hub cache>/models--<org>--<repo>/blobs/<etag>                -> <mirror>/<file>
    <hub cache>/models--<org>--<repo>/snapshots/<commit>/<file>   -> ../../blobs/<etag>
    <hub cache>/models--<org>--<repo>/refs/main                   =  <commit>

Callers keep passing the repo id to `from_pretrained`, which then finds every file in the cache, so
the tt_dit cache namespace and `return_commit_hash` behave as they do for a normal download. With
`HF_HUB_OFFLINE=1` nothing touches the hub; without it, a repo absent from the mirror downloads as usual.

The mirror is opt-in: it is only consulted when `$TT_DIT_LOCAL_WEIGHTS_ROOT` names its root.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from loguru import logger

WEIGHTS_ROOT_ENV = "TT_DIT_LOCAL_WEIGHTS_ROOT"

_METADATA_DIR = Path(".cache") / "huggingface" / "download"


class CheckpointCacheError(RuntimeError):
    """The mirror cannot populate the HF cache: it is malformed, or the cache holds a different commit."""


def weights_root() -> Path | None:
    """The mirror root from `$TT_DIT_LOCAL_WEIGHTS_ROOT`, or None (mirror disabled) when it is unset or empty."""
    root = os.environ.get(WEIGHTS_ROOT_ENV)
    return Path(root) if root else None


def hf_hub_cache_dir() -> Path:
    """The HF hub cache, resolved from the environment the way `huggingface_hub.constants` does.

    Read per call rather than from `huggingface_hub.constants`, which freezes it at import.
    """
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache") / "huggingface"
    return Path(hf_home).expanduser() / "hub"


def _is_repo_id(name: str) -> bool:
    org, _, repo = name.partition("/")
    return bool(org) and bool(repo) and "/" not in repo and not org.startswith(".")


def local_checkpoint_dir(repo_id: str) -> Path | None:
    """The mirror directory for `repo_id`, or None when the mirror is disabled or does not hold it."""
    root = weights_root()
    if root is None or not _is_repo_id(repo_id):
        return None
    directory = root / repo_id
    return directory if directory.is_dir() else None


def _mirror_manifest(mirror: Path) -> tuple[str, dict[str, str]]:
    """The mirror's commit and `{relative path: etag}` for every file in it.

    The file list comes from the directory, not the metadata: a mirror can keep metadata for files
    that were downloaded and later deleted.
    """
    files = sorted(
        p.relative_to(mirror).as_posix()
        for p in mirror.rglob("*")
        if p.is_file() and not p.relative_to(mirror).as_posix().startswith(".cache/")
    )
    commits: set[str] = set()
    etags: dict[str, str] = {}
    missing = []
    for rel in files:
        metadata = mirror / _METADATA_DIR / f"{rel}.metadata"
        lines = metadata.read_text().split() if metadata.is_file() else []
        if len(lines) < 2:
            missing.append(rel)
            continue
        commits.add(lines[0])
        etags[rel] = lines[1]
    if missing:
        raise CheckpointCacheError(
            f"Local mirror {mirror} has no download metadata for {missing[:5]}{'...' if len(missing) > 5 else ''}; "
            "repopulate it with `hf download <repo> --local-dir <dir>`"
        )
    if len(commits) != 1:
        raise CheckpointCacheError(f"Local mirror {mirror} mixes commits {sorted(commits)}")
    return commits.pop(), etags


def _replace_with_symlink(target: str | Path, link: Path) -> None:
    """Point `link` at `target` atomically, so concurrent ranks never see a partial link."""
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
    os.symlink(target, tmp)
    os.replace(tmp, link)


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def ensure_cached(repo_id: str) -> str:
    """Make `repo_id` loadable from the HF cache, linking it in from the local mirror if needed.

    Returns `repo_id` unchanged, for the caller to pass to `from_pretrained`. A complete cache entry at
    the mirror's commit is left alone; a missing or partial one is filled from the mirror. Raises
    `CheckpointCacheError` when the cache's `refs/main` names a different commit than the mirror, or the
    mirror lacks download metadata. Without a mirror copy of `repo_id` this does nothing, and
    `from_pretrained` downloads (or, offline, fails) as it normally would.
    """
    mirror = local_checkpoint_dir(repo_id)
    if mirror is None:
        return repo_id

    commit, etags = _mirror_manifest(mirror)
    repo_cache = hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    ref = repo_cache / "refs" / "main"
    cached_commit = ref.read_text().strip() if ref.is_file() else None
    if cached_commit is not None and cached_commit != commit:
        raise CheckpointCacheError(
            f"HF cache {repo_cache} is at commit {cached_commit}, but local mirror {mirror} is at {commit}. "
            "Update the mirror or remove the cache entry."
        )

    snapshot = repo_cache / "snapshots" / commit
    missing = [rel for rel in etags if not (snapshot / rel).exists()]
    if cached_commit == commit and not missing:
        return repo_id

    logger.info(f"Linking {len(missing)} files of {repo_id}@{commit[:8]} from local mirror {mirror} into {repo_cache}")
    for rel in missing:
        blob = repo_cache / "blobs" / etags[rel]
        if not blob.exists():
            _replace_with_symlink((mirror / rel).resolve(), blob)
        depth = len(Path(rel).parts) + 1  # up through the file's parents, <commit>, and snapshots/
        _replace_with_symlink(Path(*[".."] * depth) / "blobs" / etags[rel], snapshot / rel)
    _write_atomically(ref, commit)  # last: a ref only ever names a complete snapshot
    return repo_id
