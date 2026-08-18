# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight (CPU-only, no device, no model download) tests for the warm ttnn weight-cache
detector generalized into tt_transformers (issue #45400, generalizes GPT-OSS PR #48531).

These exercise the *real* ModelArgs.weight_cache_is_complete / mark_weight_cache_complete /
placeholder_state_dict logic by binding the unbound methods to a tiny stub whose
weight_cache_path points at a tmp dir -- so we validate marker round-trip, the shape/dtype
manifest, staleness rejection, the force-load override, the .tensorbin belt-and-suspenders
check, and the dataless placeholder state_dict without constructing a full ModelArgs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.tt_transformers.tt.model_config import ModelArgs

DTYPE = "bfp8"  # opaque here: the stub's weight_cache_path ignores it; marker stores str(dtype)

SAMPLE_SD = {
    "tok_embeddings.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
    "layers.0.attention.wo.weight": torch.zeros(8, 8, dtype=torch.float32),
}


class _FakeArgs:
    """Minimal stand-in exposing exactly what the marker methods touch, with the real methods
    bound so the production logic is under test."""

    WEIGHT_CACHE_MARKER = ModelArgs.WEIGHT_CACHE_MARKER
    WEIGHT_CACHE_FORMAT_VERSION = ModelArgs.WEIGHT_CACHE_FORMAT_VERSION
    _weight_cache_identity = ModelArgs._weight_cache_identity
    weight_cache_is_complete = ModelArgs.weight_cache_is_complete
    mark_weight_cache_complete = ModelArgs.mark_weight_cache_complete
    placeholder_state_dict = ModelArgs.placeholder_state_dict

    def __init__(self, cache_dir, model_name="Test-Model-8B", n_layers=32, mesh_shape=(1, 8)):
        self._cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.n_layers = n_layers
        self.dummy_weights = False
        self.is_mixture_of_experts = False
        self.mesh_device = SimpleNamespace(shape=mesh_shape)

    def weight_cache_path(self, dtype):
        return self._cache_dir


def _touch_tensorbin(cache_dir):
    (Path(cache_dir) / "some.weight.tensorbin").write_bytes(b"\x00")


@pytest.fixture(autouse=True)
def _clear_force_env(monkeypatch):
    monkeypatch.delenv("TT_TRANSFORMERS_FORCE_MODEL_LOAD", raising=False)


def test_cold_cache_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    assert args.weight_cache_is_complete(DTYPE) is False


def test_mark_then_complete_roundtrip(tmp_path):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)  # a real build writes tensor files alongside the marker
    args.mark_weight_cache_complete(DTYPE, SAMPLE_SD)
    assert args.weight_cache_is_complete(DTYPE) is True

    # Marker payload includes the shape/dtype manifest.
    meta = json.loads((tmp_path / ModelArgs.WEIGHT_CACHE_MARKER).read_text())
    assert meta["model_name"] == "Test-Model-8B"
    assert meta["n_layers"] == 32
    assert meta["mesh_shape"] == "(1, 8)"
    assert meta["format_version"] == ModelArgs.WEIGHT_CACHE_FORMAT_VERSION
    assert meta["weights"]["tok_embeddings.weight"] == [[4, 8], "torch.bfloat16"]


def test_marker_without_manifest_is_incomplete(tmp_path):
    # A marker with no weight manifest (e.g. an old v1-style write) can't back a warm build.
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    args.mark_weight_cache_complete(DTYPE)  # no state_dict -> weights == {}
    assert args.weight_cache_is_complete(DTYPE) is False


def test_marker_without_tensorbin_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    args.mark_weight_cache_complete(DTYPE, SAMPLE_SD)
    # No .tensorbin present -> belt-and-suspenders check fails.
    assert args.weight_cache_is_complete(DTYPE) is False


def test_force_env_disables_skip(tmp_path, monkeypatch):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    args.mark_weight_cache_complete(DTYPE, SAMPLE_SD)
    assert args.weight_cache_is_complete(DTYPE) is True  # warm...
    monkeypatch.setenv("TT_TRANSFORMERS_FORCE_MODEL_LOAD", "1")
    assert args.weight_cache_is_complete(DTYPE) is False  # ...but forced to cold-load


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param({"format_version": 999}, id="stale-format"),
        pytest.param({"model_name": "Other-Model"}, id="wrong-model"),
        pytest.param({"n_layers": 1}, id="partial-build"),
        pytest.param({"mesh_shape": "(2, 4)"}, id="wrong-mesh"),
        pytest.param({"weights": {}}, id="empty-manifest"),
    ],
)
def test_stale_marker_rejected(tmp_path, mutate):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    args.mark_weight_cache_complete(DTYPE, SAMPLE_SD)
    marker = tmp_path / ModelArgs.WEIGHT_CACHE_MARKER
    meta = json.loads(marker.read_text())
    meta.update(mutate)
    marker.write_text(json.dumps(meta))
    assert args.weight_cache_is_complete(DTYPE) is False


def test_corrupt_marker_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    (tmp_path / ModelArgs.WEIGHT_CACHE_MARKER).write_text("{ not json")
    assert args.weight_cache_is_complete(DTYPE) is False


def test_placeholder_state_dict_is_dataless_and_falsy(tmp_path):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    args.mark_weight_cache_complete(DTYPE, SAMPLE_SD)

    sd = args.placeholder_state_dict(DTYPE)
    # Falsy so reference-building callers (`if not state_dict`) load real weights instead.
    assert not sd
    assert len(sd) == 2
    assert set(sd.keys()) == set(SAMPLE_SD.keys())
    # Reconstructs correct shape/dtype without any real data.
    emb = sd["tok_embeddings.weight"]
    assert tuple(emb.shape) == (4, 8)
    assert emb.dtype == torch.bfloat16
    assert sd["layers.0.attention.wo.weight"].dtype == torch.float32


# ---------------------------------------------------------------------------
# models/common/weight_cache.py -- the shared helper used by the forked loaders.
# The tests above bind the tt_transformers ModelArgs methods; these cover the shared
# module's own behaviour: sidecar capture/rejection, per-file completeness, component
# matching, atomic publish, and the CachedStateDict contract. (#45400 review)
# ---------------------------------------------------------------------------

from models.common.weight_cache import (  # noqa: E402
    HOST_WEIGHTS_SIDECAR,
    WEIGHT_CACHE_MARKER,
    CachedStateDict,
    build_cached_state_dict,
    mark_weight_cache_complete,
    normalize_mesh_shape,
    weight_cache_is_complete,
)

SHARED_ID = dict(model_name="unit/test-model", n_layers=2, mesh_shape=(1, 8))


def _seed(tmp_path, *, components=None, is_host_weight=None, sd=None):
    """Write a tensorbin then mark the cache complete, mimicking a real cold build."""
    _touch_tensorbin(tmp_path)
    mark_weight_cache_complete(
        tmp_path, sd if sd is not None else SAMPLE_SD, components=components, is_host_weight=is_host_weight, **SHARED_ID
    )


def test_shared_marker_roundtrip(tmp_path):
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is False
    _seed(tmp_path)
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is True


def test_shared_marker_rejects_missing_tensorbin(tmp_path):
    """A recorded tensorbin that later disappears must force a cold load -- otherwise as_tensor
    regenerates it from the placeholder and writes garbage into the cache permanently."""
    _seed(tmp_path)
    for f in tmp_path.glob("*.tensorbin"):
        f.unlink()
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is False


def test_shared_marker_finds_tensorbins_in_subdirs(tmp_path):
    """Forked loaders nest per-layer weights (qwen36 layers.N/, gemma4 layer_N/)."""
    sub = tmp_path / "layers.0"
    sub.mkdir()
    (sub / "wq_dtype_BFLOAT8_B_layout_TILE.tensorbin").write_bytes(b"x")
    mark_weight_cache_complete(tmp_path, SAMPLE_SD, **SHARED_ID)
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is True
    (sub / "wq_dtype_BFLOAT8_B_layout_TILE.tensorbin").unlink()
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is False


def test_components_subset_matching(tmp_path):
    """A text-only seed must not certify a build that also needs the vision tower; the reverse
    (vision seed satisfying a text-only build) is fine."""
    _seed(tmp_path, components=["text"])
    assert weight_cache_is_complete(tmp_path, components=["text"], **SHARED_ID) is True
    assert weight_cache_is_complete(tmp_path, components=["text", "vision"], **SHARED_ID) is False

    _seed(tmp_path, components=["text", "vision"])
    assert weight_cache_is_complete(tmp_path, components=["text"], **SHARED_ID) is True


def test_sidecar_capture_and_corruption_rejected(tmp_path):
    _seed(tmp_path, is_host_weight=lambda k: k == "tok_embeddings.weight")
    assert (tmp_path / HOST_WEIGHTS_SIDECAR).is_file()
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is True

    sd = build_cached_state_dict(tmp_path)
    # The captured host weight is served REAL; everything else is a dataless placeholder.
    assert torch.equal(sd["tok_embeddings.weight"], SAMPLE_SD["tok_embeddings.weight"])

    # A torn sidecar must degrade to a cold load, not crash every later run.
    (tmp_path / HOST_WEIGHTS_SIDECAR).write_bytes(b"not a torch file")
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is False


def test_no_temp_files_left_behind(tmp_path):
    _seed(tmp_path, is_host_weight=lambda k: k == "tok_embeddings.weight")
    assert not list(tmp_path.glob("*.tmp*")), "atomic publish must not leave temp files"


def test_mesh_shape_encoding_is_writer_agnostic(tmp_path):
    """Both marker writers must encode the mesh identically or each rejects the other's marker."""

    class _MeshShape:
        def __init__(self, dims):
            self._dims = dims

        def __iter__(self):
            return iter(self._dims)

        def __str__(self):
            return f"MeshShape({list(self._dims)})"

    assert normalize_mesh_shape(_MeshShape((1, 8))) == normalize_mesh_shape((1, 8))
    _seed(tmp_path)
    identity = dict(SHARED_ID, mesh_shape=_MeshShape((1, 8)))
    assert weight_cache_is_complete(tmp_path, **identity) is True


def test_cached_state_dict_contract():
    manifest = {k: [list(v.shape), str(v.dtype)] for k, v in SAMPLE_SD.items()}
    real = SAMPLE_SD["tok_embeddings.weight"]
    sd = CachedStateDict(manifest, {"tok_embeddings.weight": real})

    # Truthy and flagged (the tt_transformers placeholder is falsy -- callers must branch on the
    # attribute, not truthiness).
    assert sd
    assert sd.is_placeholder is True

    # Membership must not materialize a tensor.
    assert "layers.0.attention.wo.weight" in sd
    assert "nope" not in sd
    assert sd.get("nope") is None

    # Mutable: loaders setdefault KV-shared weights.
    sd["extra"] = torch.zeros(2)
    assert "extra" in sd
    del sd["extra"]
    assert "extra" not in sd

    del sd["tok_embeddings.weight"]
    assert "tok_embeddings.weight" not in sd
    try:
        sd["tok_embeddings.weight"]
        raise AssertionError("a deleted key must raise KeyError, even when the sidecar still has it")
    except KeyError:
        pass


def test_build_variant_must_match_exactly(tmp_path):
    """Build options that change an as_tensor cache FILENAME (prefetcher, precision) are matched
    exactly, not as a superset: a different variant needs DIFFERENT files, and any it is missing
    would be regenerated from the placeholder rather than cold-loaded."""
    perf = {"prefetcher": False, "precision": "aaaaaaaaaaaa"}
    _touch_tensorbin(tmp_path)
    mark_weight_cache_complete(tmp_path, SAMPLE_SD, build_variant=perf, **SHARED_ID)

    assert weight_cache_is_complete(tmp_path, build_variant=perf, **SHARED_ID) is True
    # different precision config
    assert (
        weight_cache_is_complete(tmp_path, build_variant={"prefetcher": False, "precision": "bbbb"}, **SHARED_ID)
        is False
    )
    # prefetcher flips the dtypes and adds ring-matmul splits
    assert (
        weight_cache_is_complete(tmp_path, build_variant={"prefetcher": True, "precision": "aaaaaaaaaaaa"}, **SHARED_ID)
        is False
    )
    # a caller that records no variant must not be satisfied by one that did
    assert weight_cache_is_complete(tmp_path, **SHARED_ID) is False
