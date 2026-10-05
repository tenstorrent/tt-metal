#!/usr/bin/env python3
"""Fail-closed resume records shared by the exhaustive streamers."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import sys


SCHEMA = 3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def python_tree_sha256(root_text: str) -> str:
    """Fingerprint the Python harness which interprets a streamed result."""
    root = Path(root_text)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise RuntimeError(f"no Python provenance files under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_record(
    script: Path,
    args,
    node: str,
    start: int,
    count: int,
    leg: str,
    *,
    compiler_options: str | None = None,
    golden: str | None = None,
) -> dict:
    idmap = Path(args.idmap).resolve() if args.idmap else None
    farm = Path(args.farm).resolve()
    tools = script.resolve().parent
    venv = Path(args.venv).resolve()
    return {
        "schema": SCHEMA,
        "streamer_sha256": sha256_file(script.resolve()),
        "farm_python_root": str(farm),
        "farm_python_sha256": python_tree_sha256(str(farm)),
        "tools_python_sha256": python_tree_sha256(str(tools)),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "pytest_python": str(venv),
        "pytest_python_sha256": sha256_file(venv),
        "runner_temp": str(Path(args.runner_temp).resolve()),
        "llk_home": str(Path(args.llk_home).resolve()),
        "chip": str(args.chip),
        "op": args.op,
        "node": node,
        "arm_nodes": {
            key: getattr(args, key)
            for key in (
                "sem_node", "hand_node", "selected_sem_node",
                "baseline_sem_node", "baseline_hand_node",
            )
            if getattr(args, key, None) is not None
        },
        "compiler_options": (
            os.environ.get("TT_LLK_EXTRA_COMPILER_OPTIONS", "")
            if compiler_options is None else compiler_options
        ),
        "start": start,
        "count": count,
        "leg": leg,
        "golden": args.golden if golden is None else golden,
        "tile_dim": getattr(args, "tile_dim", None),
        "band_bits": args.band_bits,
        "idmap_source": getattr(args, "idmap_source", None),
        "idmap_sha256": sha256_file(idmap) if idmap else None,
    }


def require_matching_cache(output: Path, metadata: Path, expected: dict) -> bool:
    """Return False for no cache; reject legacy, unpinned, or mismatched cache."""
    if not output.exists():
        return False
    if expected["idmap_sha256"] is None:
        raise RuntimeError("refusing resume without an identity-map provenance hash")
    if not metadata.is_file():
        raise RuntimeError(f"refusing legacy cache without provenance: {output}")
    try:
        actual = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid cache provenance {metadata}: {error}") from error
    if not isinstance(actual, dict) or actual.get("request") != expected:
        raise RuntimeError(f"cache provenance mismatch: {output}")
    artifacts = actual.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"cache artifact custody is missing: {output}")
    if artifacts.get("output_sha256") != sha256_file(output):
        raise RuntimeError(f"cached output digest mismatch: {output}")
    corr = Path(str(output) + ".corr")
    if expected["golden"]:
        if not corr.is_file() or artifacts.get("corr_sha256") != sha256_file(corr):
            raise RuntimeError(f"cached correctness digest mismatch: {corr}")
    elif artifacts.get("corr_sha256") is not None:
        raise RuntimeError(f"unexpected cached correctness artifact: {corr}")
    return True


def write_cache_record(metadata: Path, record: dict, output: Path) -> None:
    corr = Path(str(output) + ".corr")
    payload = {
        "request": record,
        "artifacts": {
            "output_sha256": sha256_file(output),
            "corr_sha256": sha256_file(corr) if corr.is_file() else None,
        },
    }
    temporary = metadata.with_suffix(metadata.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata)
