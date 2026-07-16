# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The on-disk weight cache must never serve weights that aren't the ones its key names.

A cache whose key encoded only shape/mesh/dtype-label once served LTX connector tensorbins that
did not hold the checkpoint's weights at all. They deserialized as finite, plausibly-scaled
floats, so nothing raised: the aggregate projection overflowed, the embedding collapsed to zero,
and zero is the DiT's unconditional input — every prompt rendered the same clip, silently.

These are host-only: they drive the cache key and manifest directly, with a stub module, and
never open a device.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...utils import cache


class _StubParameter:
    """Duck-types the fields of `Parameter` that decide a tensorbin's bytes."""

    def __init__(self, shape=(32, 32), dtype="DataType.BFLOAT16", layout="Layout.TILE", mesh_axes=(None, None)):
        self.total_shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.mesh_axes = tuple(mesh_axes)


class _StubModule:
    """Duck-types the `Module` surface the cache uses: children, parameters, and `save`."""

    def __init__(self, parameters: dict, children: dict | None = None):
        self._parameters = parameters
        self._children = children or {}

    def named_children(self):
        return self._children.items()

    def named_parameters(self):
        return self._parameters.items()

    def save(self, directory, /, *, prefix: str = ""):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, child in self.named_children():
            child.save(directory, prefix=f"{prefix}{name}.")
        for name, _ in self.named_parameters():
            (directory / f"{prefix}{name}.tensorbin").write_bytes(b"\x00" * 64)


def _module():
    return _StubModule({"weight": _StubParameter(), "bias": _StubParameter(shape=(32,), mesh_axes=(None,))})


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original weights")
    return path


def _key(module, sources):
    return cache.model_cache_dir(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        content=cache.content_key(module, sources),
    )


def test_key_is_stable_across_runs(checkpoint, monkeypatch, tmp_path):
    """The whole point of a cache: unchanged inputs must land on the same path every process
    start, or every run re-materializes the checkpoint."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    assert _key(_module(), [checkpoint]) == _key(_module(), [checkpoint])


def test_key_tracks_the_source_weights(checkpoint, monkeypatch, tmp_path):
    """Rewriting the checkpoint under a fixed filename must not keep serving the old artifact."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    before = _key(_module(), [checkpoint])

    checkpoint.write_bytes(b"different weights, same name")
    assert _key(_module(), [checkpoint]) != before


def test_key_tracks_the_weight_dtype_not_the_preset_name(checkpoint, monkeypatch, tmp_path):
    """A quant preset that only retunes compute leaves the weights byte-identical, so it must keep
    the cache — re-materializing 22B to change a fidelity flag is the bug this replaces. A preset
    that restages a weight must not."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    baseline = _key(_module(), [checkpoint])

    same_weights_new_compute = _module()  # e.g. all_bf8_lofi -> all_bf8_lofi_sdpa_bf8
    assert _key(same_weights_new_compute, [checkpoint]) == baseline

    quantized = _StubModule(
        {"weight": _StubParameter(dtype="DataType.BFLOAT8_B"), "bias": _StubParameter(shape=(32,), mesh_axes=(None,))}
    )
    assert _key(quantized, [checkpoint]) != baseline


def test_key_tracks_the_prep_code_version(checkpoint, monkeypatch, tmp_path):
    """A prep change that permutes values under a fixed shape and dtype is invisible to the
    module signature, so `CACHE_VERSION` is the only thing that can evict its artifacts."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    before = _key(_module(), [checkpoint])

    monkeypatch.setattr(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1)
    assert _key(_module(), [checkpoint]) != before


def test_published_cache_hits(checkpoint, monkeypatch, tmp_path):
    """A cache the current code wrote for the current inputs must load, or the key is useless."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])

    cache._publish_cache(module, cache_dir, key)

    assert cache._cache_is_complete(cache_dir, module, key)


def test_publish_is_all_or_nothing(checkpoint, monkeypatch, tmp_path, expect_error):
    """A build that dies partway must leave nothing behind that a later run reads as complete."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    key = cache.content_key(_module(), [checkpoint])
    cache_dir = _key(_module(), [checkpoint])

    exploding = _module()
    exploding.save = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("device fell over mid-save"))
    with expect_error(RuntimeError, "device fell over mid-save"):
        cache._publish_cache(exploding, cache_dir, key)

    assert not cache_dir.exists()
    assert not list(cache_dir.parent.glob("*.staging-*")), "staging directory leaked"


def test_the_superseded_directory_is_reclaimed(checkpoint, monkeypatch, tmp_path):
    """Once a module keys on content, its content-blind directory is unreachable — and for the 22B
    transformer that is 37GB. The migration has to free it, not ask the disk for room for both."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    legacy = cache.model_cache_dir(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        content=None,
    )
    _module().save(legacy)
    assert legacy.is_dir()

    cache._reclaim_superseded(legacy)

    assert not legacy.exists()
    assert legacy != _key(_module(), [checkpoint]), "the content-keyed cache must not be the one reclaimed"


def test_a_cache_with_no_manifest_misses(checkpoint, monkeypatch, tmp_path):
    """Every artifact written before this manifest existed is unverifiable, so it must not be
    served — including the poisoned one. The empty `cache_dict.json` of the old format is exactly
    this case."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])

    module.save(cache_dir)
    (cache_dir / cache.CACHE_DICT_FILE).touch()  # the old completion marker: an empty file

    assert not cache._cache_is_complete(cache_dir, module, key)


def test_a_truncated_tensorbin_misses(checkpoint, monkeypatch, tmp_path):
    """A manifest that agrees with the module is not enough; the bytes have to still be there."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])
    cache._publish_cache(module, cache_dir, key)

    (cache_dir / "weight.tensorbin").write_bytes(b"\x00" * 8)

    assert not cache._cache_is_complete(cache_dir, module, key)


def test_a_manifest_for_other_weights_misses(checkpoint, monkeypatch, tmp_path):
    """The artifact and its key must be checked against each other, not just present."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])
    cache._publish_cache(module, cache_dir, key)

    marker = cache_dir / cache.CACHE_DICT_FILE
    manifest = json.loads(marker.read_text())
    manifest["content_key"] = "0" * 12
    marker.write_text(json.dumps(manifest))

    assert not cache._cache_is_complete(cache_dir, module, key)


# The tensorbins that shipped the silent-unconditional bug, kept as the one reproduction of a
# poisoned artifact. Their payloads are densely-packed bfloat16 inside a buffer the header
# declares float32, so they load as finite garbage of the right shape and size.
POISONED = Path("~/.cache/tt-dit-ltxrt/ltx-2.3-22b-distilled-1.1").expanduser()


@pytest.mark.parametrize("subfolder", ["feature_extractor", "video_connector", "audio_connector"])
def test_the_poisoned_cache_is_not_servable(subfolder, checkpoint, monkeypatch, tmp_path):
    """The real artifact, against the real loader. Size and shape agree with the module, so no
    structural check can reject it — it is rejected because it carries no manifest binding it to
    the weights it claims, which is true of every artifact written before this key existed."""
    poisoned = next(POISONED.glob(f"{subfolder}.stale-jul7/*/"), None)
    if poisoned is None:
        pytest.skip(f"quarantined {subfolder} not on this box")

    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])

    # Negative control: the content-blind key this replaces reads the poison as a complete cache
    # and serves it. That is the bug, reproduced.
    assert cache._cache_is_complete(poisoned, module, None)

    assert not cache._cache_is_complete(poisoned, module, key)
