# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""checkpoint_cache_key() must change whenever the checkpoint at a path changes, stay stable when it
does not, and stay cheap (no full read) by default.

ttnn.as_tensor serves a cached tensorbin without looking at the torch weight, and the tensorbin name
carries only tensor name/dtype/layout. Without a content key in the cache directory, replacing or
modifying the checkpoint at the same HF_MODEL path silently loads the previous checkpoint's weights.
CPU only: builds tiny safetensors-format files by hand.
"""

import json
import os
import shutil
import struct

import pytest

from models.common import weight_cache
from models.common.weight_cache import CHECKPOINT_KEY_ENV, checkpoint_cache_key

DATA_BYTES = 64 << 10


def _write_safetensors(path, data, dtype="BF16"):
    header = json.dumps({"w": {"dtype": dtype, "shape": [len(data) // 2], "data_offsets": [0, len(data)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _make_ckpt(d, data=None, config=None):
    d.mkdir(parents=True, exist_ok=True)
    _write_safetensors(d / "model.safetensors", data if data is not None else bytes(DATA_BYTES))
    (d / "config.json").write_text(json.dumps(config or {"hidden_size": 8}))
    (d / "tokenizer.model").write_bytes(b"tok")
    return d


def _rewrite_keep_stat(path, offset, payload):
    """Modify bytes in place while keeping size AND mtime -- the worst case for a stat-based key
    (coarse-mtime filesystems, mtime-preserving copies)."""
    st = path.stat()
    with path.open("r+b") as f:
        f.seek(offset)
        f.write(payload)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


@pytest.fixture(autouse=True)
def _opt_in(monkeypatch):
    # The key is opt-in; these tests exercise the opted-in default mode.
    monkeypatch.setenv(CHECKPOINT_KEY_ENV, "fast")


def test_default_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv(CHECKPOINT_KEY_ENV)
    assert checkpoint_cache_key(_make_ckpt(tmp_path / "ckpt")) is None


def test_not_a_local_dir_is_unkeyed(tmp_path):
    assert checkpoint_cache_key("meta-llama/Llama-3.1-8B-Instruct") is None
    assert checkpoint_cache_key(tmp_path / "missing") is None
    assert checkpoint_cache_key(None) is None


def test_off_restores_legacy(tmp_path, monkeypatch):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    assert checkpoint_cache_key(ckpt, mode="off") is None
    monkeypatch.setenv(CHECKPOINT_KEY_ENV, "off")
    assert checkpoint_cache_key(ckpt) is None


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        checkpoint_cache_key(_make_ckpt(tmp_path / "ckpt"), mode="sometimes")


def test_stable_for_unchanged_and_mtime_preserving_copy(tmp_path):
    a = _make_ckpt(tmp_path / "a")
    key = checkpoint_cache_key(a)
    assert key and len(key) == 16
    assert checkpoint_cache_key(a) == key
    b = tmp_path / "b"
    shutil.copytree(a, b)  # copy2: preserves mtime
    assert checkpoint_cache_key(b) == key


def test_replaced_checkpoint_at_same_path(tmp_path):
    """The repro: checkpoint A, then a different same-shape checkpoint B at the same path."""
    ckpt = _make_ckpt(tmp_path / "ckpt", data=bytes(DATA_BYTES))
    key_a = checkpoint_cache_key(ckpt)
    shutil.rmtree(ckpt)
    _make_ckpt(tmp_path / "ckpt", data=b"\x01" * DATA_BYTES)
    assert checkpoint_cache_key(ckpt) != key_a


@pytest.mark.parametrize("mode", ["fast", "full"])
def test_in_place_same_size_same_mtime_edit(tmp_path, mode):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    key = checkpoint_cache_key(ckpt, mode=mode)
    # Lands inside one of the fast mode's evenly spaced samples.
    _rewrite_keep_stat(ckpt / "model.safetensors", (ckpt / "model.safetensors").stat().st_size // 2, b"\xff" * 16)
    assert checkpoint_cache_key(ckpt, mode=mode) != key


def test_full_mode_sees_every_byte_and_ignores_mtime(tmp_path):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    w = ckpt / "model.safetensors"
    key = checkpoint_cache_key(ckpt, mode="full")
    os.utime(w, ns=(1, 1))  # touch only: same content
    assert checkpoint_cache_key(ckpt, mode="full") == key
    fast = checkpoint_cache_key(ckpt, mode="fast")
    # An edit between fast-mode samples that keeps size, header AND mtime is fast mode's documented
    # blind spot (that is the price of not reading 100+ GB); full mode is the escape hatch.
    # size//8 - 64 sits after the first 4 KiB sample and before the second (size//8 > 8 KiB here).
    _rewrite_keep_stat(w, w.stat().st_size // 8 - 64, b"\xff")
    assert checkpoint_cache_key(ckpt, mode="fast") == fast
    assert checkpoint_cache_key(ckpt, mode="full") != key


def test_rewrite_changes_fast_key_via_mtime(tmp_path):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    w = ckpt / "model.safetensors"
    key = checkpoint_cache_key(ckpt)
    st = w.stat()
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert checkpoint_cache_key(ckpt) != key


def test_header_and_config_changes(tmp_path):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    key = checkpoint_cache_key(ckpt)
    (ckpt / "config.json").write_text(json.dumps({"hidden_size": 16}))
    key_cfg = checkpoint_cache_key(ckpt)
    assert key_cfg != key
    w = ckpt / "model.safetensors"
    st = w.stat()
    _write_safetensors(w, bytes(DATA_BYTES), dtype="F16")  # same size, different dtype in header
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert checkpoint_cache_key(ckpt) != key_cfg


def test_non_weight_files_ignored(tmp_path):
    ckpt = _make_ckpt(tmp_path / "ckpt")
    key = checkpoint_cache_key(ckpt)
    (ckpt / "tokenizer.model").write_bytes(b"other tokenizer")
    (ckpt / "README.md").write_text("notes")
    (ckpt / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}))
    (ckpt / "generation_config.json").write_text(json.dumps({"temperature": 0.6}))
    assert checkpoint_cache_key(ckpt) == key


def test_producer_manifest_folds_in(tmp_path):
    """Small top-level JSON (index, producer manifests carrying a content hash) is part of the key."""
    ckpt = _make_ckpt(tmp_path / "ckpt")
    key = checkpoint_cache_key(ckpt)
    (ckpt / "producer_manifest.json").write_text(json.dumps({"content_hash": "abc"}))
    key_m = checkpoint_cache_key(ckpt)
    assert key_m != key
    (ckpt / "producer_manifest.json").write_text(json.dumps({"content_hash": "def"}))
    assert checkpoint_cache_key(ckpt) != key_m


def test_full_mode_reads_each_file_once_per_process(tmp_path, monkeypatch):
    """Data-parallel runs build one ModelArgs per replica; `full` must not re-read the checkpoint."""
    ckpt = _make_ckpt(tmp_path / "ckpt")
    key = checkpoint_cache_key(ckpt, mode="full")
    opened = []
    real_open = weight_cache.Path.open
    monkeypatch.setattr(
        weight_cache.Path, "open", lambda self, *a, **k: opened.append(self.name) or real_open(self, *a, **k)
    )
    assert checkpoint_cache_key(ckpt, mode="full") == key
    assert "model.safetensors" not in opened
    # Any write bumps ctime, so the memo cannot serve a stale digest (also covered above with mtime kept).
    _rewrite_keep_stat(ckpt / "model.safetensors", 0, b"\x01")
    assert checkpoint_cache_key(ckpt, mode="full") != key
    assert "model.safetensors" in opened
