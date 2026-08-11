# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the Muse-Glimmer functional-decoder tests.

A module-scoped 1x1 mesh device and a cache of built layers keep the suite affordable:
uploading one decoder layer is ~1 GB of weights, so it is done once per
(layer kind, block size, cache dtype) instead of once per test. K/V caches and page
tables are always per test.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"
WEIGHT_STATS_PATH = ARTIFACT_DIR / "weight_stats.json"
SYNTHETIC_SEED = 7


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: very long sequence tests (minutes to hours)")
    config.addinivalue_line("markers", "real_weights: needs the real HF checkpoint on disk")


@pytest.fixture(scope="session", autouse=True)
def _host_threads():
    R.configure_host_threads()


@pytest.fixture(scope="session")
def text_config():
    return R.load_text_config()


@pytest.fixture(scope="session")
def kinds(text_config):
    return R.layer_kinds(text_config)


@pytest.fixture(scope="session")
def weight_stats():
    """Real-checkpoint per-tensor stats (committed artifact, so CI needs no weights)."""
    if WEIGHT_STATS_PATH.is_file():
        return json.loads(WEIGHT_STATS_PATH.read_text())
    pytest.fail(f"missing {WEIGHT_STATS_PATH}; regenerate it with scripts/dump_weight_stats.py")


@pytest.fixture(scope="session")
def synthetic_state_dicts(weight_stats):
    """Deterministic synthetic weights per layer index, with the real shapes and scales."""
    out = {}
    for layer_idx, entry in weight_stats["layers"].items():
        out[int(layer_idx)] = R.synthetic_layer_state_dict(entry["tensors"], seed=SYNTHETIC_SEED)
    return out


@pytest.fixture(scope="module")
def mg_mesh_device():
    """1x1 mesh (functional bringup is single-device; multichip is a later stage).

    ``trace_region_size=0`` lets TTNN size the trace region automatically.
    """
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def decoder_cache(mg_mesh_device):
    cache: dict = {}
    yield cache
    cache.clear()


@pytest.fixture
def build_cached_decoder(mg_mesh_device, text_config, decoder_cache):
    """``(layer_idx, state_dict, block_size=..., ...) -> FunctionalDecoder`` with caching.

    ``state_dict`` is keyed by ``tag`` so real- and synthetic-weight layers never alias.
    """

    def _build(
        layer_idx, state_dict, *, tag="synthetic", block_size=64, prefill_chunk_size=8192, cache_dtype=ttnn.bfloat16
    ):
        key = (layer_idx, tag, block_size, prefill_chunk_size, str(cache_dtype))
        if key not in decoder_cache:
            decoder_cache[key] = U.build_decoder(
                mesh_device=mg_mesh_device,
                hf_config=text_config,
                layer_idx=layer_idx,
                state_dict=state_dict,
                block_size=block_size,
                prefill_chunk_size=prefill_chunk_size,
                cache_dtype=cache_dtype,
            )
        return decoder_cache[key]

    return _build


@pytest.fixture
def reference_cache():
    """Per-test cache of host reference layers (they are cheap to keep, costly to rebuild)."""
    return {}


@pytest.fixture
def build_reference(text_config, reference_cache):
    def _build(layer_idx, state_dict, *, tag="synthetic"):
        key = (layer_idx, tag)
        if key not in reference_cache:
            reference_cache[key] = R.ReferenceDecoderLayer(text_config, layer_idx, state_dict)
        return reference_cache[key]

    return _build


def _code_fingerprint() -> dict:
    """Provenance for a PCC record: what code produced it, and when.

    Records accumulate across runs (a fast-suite run must not drop the long-context records),
    so each one carries the git HEAD and a hash of the files that decide the numbers. Without
    this a record from an older revision is indistinguishable from a fresh one — which a stage
    review flagged. ``scripts/render_evidence.py`` reports any record whose fingerprint is not
    the current one.
    """
    repo_root = Path(__file__).resolve().parents[4]
    tracked = [
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/tt/functional_decoder.py",
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/reference/hf_reference.py",
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py",
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder_perf.py",
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/tests/decoder_test_utils.py",
        repo_root / "models/autoports/meta_models_muse_glimmer_30b/tests/conftest.py",
    ]
    digest = hashlib.sha256()
    for path in tracked:
        digest.update(path.read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # pragma: no cover - git absent
        commit = ""
    return {
        "code_sha256": digest.hexdigest()[:16],
        "git_head": commit,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@pytest.fixture(scope="session")
def pcc_record():
    """Collects every measured PCC and dumps it as a stage artifact."""
    records: list[dict] = []
    yield records
    if not records:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "pcc" / "pcc_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text())["records"] if path.is_file() else []
    by_key = {(r["test"], r["metric"]): r for r in existing}
    for record in records:
        by_key[(record["test"], record["metric"])] = record
    payload = {
        "current_code_sha256": _code_fingerprint()["code_sha256"],
        "records": sorted(by_key.values(), key=lambda r: (r["test"], r["metric"])),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture(scope="session")
def code_fingerprint():
    return _code_fingerprint()


@pytest.fixture
def record_pcc(pcc_record, request, code_fingerprint):
    def _record(metric: str, value: float, *, threshold: float = 0.995, **extra):
        entry = {"test": request.node.name, "metric": metric, "pcc": float(value), "threshold": threshold}
        entry.update(extra)
        entry.update(code_fingerprint)
        pcc_record.append(entry)
        return value

    return _record
