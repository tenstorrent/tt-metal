# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic smoke / train / holdout splits for model-traced matmul vectors.

Split rules (Milestone 1):
  1. Stratum key = (traced_source, input_a_shape, input_b_shape) with stable string forms.
  2. Within each stratum, vectors are ordered by input_hash (lexicographic).
  3. Strata are ordered by stratum key.
  4. Round-robin interleave across strata to form sequence L (diversity-preserving).
  5. L[0:smoke_max] -> smoke; remainder split by train_fraction_of_remainder -> train / holdout.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROTOCOL_VERSION = "1"
DEFAULT_MODULE_STEM = "model_traced.matmul_model_traced"
DEFAULT_SUITE_NAME = "model_traced"


@dataclass(frozen=True)
class PartitionParams:
    smoke_max: int = 16
    train_fraction_of_remainder: float = 0.58


def _normalize_traced_source(raw: Any) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        parts = [_normalize_traced_source(x) for x in raw]
        return ",".join(parts)
    return str(raw)


def _normalize_shape(val: Any) -> str:
    if val is None:
        return "none"
    if isinstance(val, (list, tuple)):
        return json.dumps(list(val), separators=(",", ":"), sort_keys=False)
    return str(val)


def stratum_key(vector: dict[str, Any]) -> str:
    ts = _normalize_traced_source(vector.get("traced_source"))
    sa = _normalize_shape(vector.get("input_a_shape"))
    sb = _normalize_shape(vector.get("input_b_shape"))
    return f"{ts}\x1f{sa}\x1f{sb}"


def find_matmul_vector_files(vectors_export_dir: Path, module_stem: str = DEFAULT_MODULE_STEM) -> list[Path]:
    """Match sweeps vector export naming (base + optional mesh suffix)."""
    out: list[Path] = []
    exact = vectors_export_dir / f"{module_stem}.json"
    if exact.exists():
        out.append(exact)
    out.extend(sorted(vectors_export_dir.glob(f"{module_stem}__mesh_*.json")))
    return sorted(set(out))


def load_model_traced_suite(
    vectors_export_dir: Path,
    module_stem: str = DEFAULT_MODULE_STEM,
    suite_name: str = DEFAULT_SUITE_NAME,
) -> dict[str, dict[str, Any]]:
    """Merge suite maps from all export files for this module (first hash wins)."""
    merged: dict[str, dict[str, Any]] = {}
    files = find_matmul_vector_files(vectors_export_dir, module_stem)
    if not files:
        raise FileNotFoundError(
            f"No vector JSON for '{module_stem}' under {vectors_export_dir}. "
            f"Generate with: python tests/sweep_framework/sweeps_parameter_generator.py "
            f"--module-name {module_stem} --suite-name {suite_name} --model-traced all"
        )
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if suite_name not in data or not isinstance(data[suite_name], dict):
            continue
        for input_hash, vec in data[suite_name].items():
            if input_hash not in merged:
                merged[input_hash] = dict(vec)
    if not merged:
        raise ValueError(f"Suite '{suite_name}' missing or empty in {files}")
    return merged


def _round_robin_interleave(strata: dict[str, list[str]]) -> list[str]:
    keys = sorted(strata.keys())
    queues = {k: list(v) for k, v in strata.items()}
    out: list[str] = []
    while True:
        moved = False
        for k in keys:
            q = queues.get(k, [])
            if q:
                out.append(q.pop(0))
                moved = True
        if not moved:
            break
    return out


def partition_hashes(
    suite_vectors: dict[str, dict[str, Any]],
    params: PartitionParams | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """
    Returns (smoke, train, holdout, stats).
    """
    p = params or PartitionParams()
    strata: dict[str, list[str]] = {}
    for input_hash, vec in suite_vectors.items():
        sk = stratum_key(vec)
        strata.setdefault(sk, []).append(input_hash)
    for sk in strata:
        strata[sk].sort()

    ordered = _round_robin_interleave(strata)
    n = len(ordered)
    smoke_n = min(p.smoke_max, n)
    smoke = ordered[:smoke_n]
    rest = ordered[smoke_n:]
    if not rest:
        return (
            smoke,
            [],
            [],
            {
                "total_vectors": n,
                "strata_count": len(strata),
                "smoke_count": len(smoke),
                "train_count": 0,
                "holdout_count": 0,
            },
        )
    train_n = int(math.floor(p.train_fraction_of_remainder * len(rest)))
    train = rest[:train_n]
    holdout = rest[train_n:]
    stats = {
        "total_vectors": n,
        "strata_count": len(strata),
        "smoke_count": len(smoke),
        "train_count": len(train),
        "holdout_count": len(holdout),
    }
    return smoke, train, holdout, stats


def build_manifest(
    smoke: list[str],
    train: list[str],
    holdout: list[str],
    stats: dict[str, Any],
    *,
    module_name: str,
    suite_name: str,
    params: PartitionParams,
    vectors_export_dir: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "module_name": module_name,
        "suite_name": suite_name,
        "vectors_export_dir": vectors_export_dir,
        "partition_params": {
            "smoke_max": params.smoke_max,
            "train_fraction_of_remainder": params.train_fraction_of_remainder,
        },
        "smoke": smoke,
        "train": train,
        "holdout": holdout,
        "stats": stats,
    }


def write_suite_subset_json(
    suite_vectors: dict[str, dict[str, Any]],
    hashes: list[str],
    out_path: Path,
    suite_name: str = DEFAULT_SUITE_NAME,
) -> None:
    subset = {}
    missing = []
    for h in hashes:
        if h in suite_vectors:
            subset[h] = suite_vectors[h]
        else:
            missing.append(h)
    if missing:
        raise ValueError(f"Missing {len(missing)} vector(s) for subset (e.g. {missing[:3]})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({suite_name: subset}, f, indent=2)
        f.flush()
    tmp.replace(out_path)


def iter_manifest_hashes(manifest: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield (partition_name, input_hash) for all protocol vectors."""
    for name in ("smoke", "train", "holdout"):
        for h in manifest.get(name, []):
            yield name, h


def partition_name_for_hash(manifest: dict[str, Any], input_hash: str) -> str | None:
    for name in ("smoke", "train", "holdout"):
        if input_hash in manifest.get(name, []):
            return name
    return None
