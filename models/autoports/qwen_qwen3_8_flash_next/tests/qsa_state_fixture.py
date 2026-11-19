# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-boundary serialization of TT-produced QSA state, never HF state."""

import hashlib
import os
from pathlib import Path

import torch

STATE_TENSORS = (
    "position",
    "page_table",
    "k_cache",
    "v_cache",
    "raw_history",
    "compressed_keys",
    "block_numbers",
    "token_numbers",
)


def runtime_source_hashes():
    root = Path(__file__).resolve().parents[1] / "tt"
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.suffix in (".py", ".cpp", ".h", ".hpp")
    }


def export_prefix(state, path, *, prefix, fixture_cache, provenance, prefix_metric):
    """Setup/test boundary after a passing, from-empty TT prefix forward."""
    import ttnn

    if not prefix_metric["passed"]:
        raise ValueError("Refusing to export a prefix whose all-output gate failed")
    tensors = {name: ttnn.to_torch(getattr(state, name)).clone() for name in STATE_TENSORS}
    assert int(tensors["position"].reshape(-1)[0]) == prefix
    payload = {
        "version": 1,
        "origin": "TT QSA prefill from initially empty state; exported before final decode",
        "prefix": prefix,
        "capacity": state.capacity,
        "page_size": state.page_size,
        "source_fixture_key": fixture_cache.get("source_key", fixture_cache["key"]),
        "source_fixture_path": fixture_cache["path"],
        "provenance": provenance,
        "runtime_source_sha256": runtime_source_hashes(),
        "prefix_metric": prefix_metric,
        "tensors": tensors,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "prefix": prefix,
        "capacity": state.capacity,
        "runtime_source_sha256": payload["runtime_source_sha256"],
        "origin": payload["origin"],
    }


def load_prefix(path, *, source_key, prefix, capacity):
    """Validate provenance before allowing any saved TT buffer onto a device."""
    payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    assert payload["version"] == 1
    assert payload["origin"] == "TT QSA prefill from initially empty state; exported before final decode"
    assert payload["prefix"] == prefix and payload["capacity"] == capacity and payload["page_size"] == 32
    assert payload["source_fixture_key"] == source_key
    assert payload["runtime_source_sha256"] == runtime_source_hashes(), "Runtime source changed after TT prefix export"
    assert payload["prefix_metric"]["passed"]
    assert set(payload["tensors"]) == set(STATE_TENSORS)
    assert int(payload["tensors"]["position"].reshape(-1)[0]) == prefix
    return payload
