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
import os
import shutil
import stat
import subprocess
import threading
import time
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
        # No device: the ownership suffix is empty below world size 2, so mesh_device is never read.
        mesh_device=None,
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


def _legacy_dir():
    return cache.model_cache_dir(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
        content=None,
    )


class _LoadableStub(_StubModule):
    """A stub `load_model` can fill: it records loads and builds instead of touching a device."""

    def __init__(self):
        super().__init__(_module()._parameters)
        self.loaded_from = []

    def is_loaded(self):
        return False

    def load_torch_state_dict(self, state_dict):
        self.loaded_from.append("torch")

    def load(self, directory):
        self.loaded_from.append(str(directory))

    def deallocate_weights(self):
        pass


def _fill(module, sources, monkeypatch):
    monkeypatch.setattr(cache.ttnn, "distributed_context_barrier", lambda: None, raising=False)
    cache.load_model(
        module,
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
        sources=sources,
        get_torch_state_dict=dict,
    )


def test_reclaim_keeps_served(checkpoint, monkeypatch, tmp_path):
    """Every tree on a box shares one cache root: the tree a rollout prepares beside the served one
    builds its content-keyed leaf while the served tree still reads the content-blind one. That
    fill must leave the served tree's leaf whole, or the served tree's next load is cold."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TT_DIT_CACHE_RECLAIM_SUPERSEDED", raising=False)
    served = _legacy_dir()
    _module().save(served)
    (served / cache.CACHE_DICT_FILE).touch()
    before = sorted(f.name for f in served.iterdir())

    module = _LoadableStub()
    _fill(module, [checkpoint], monkeypatch)

    assert module.loaded_from[0] == "torch", "the stub should have missed and built"
    assert _key(_module(), [checkpoint]).is_dir(), "the new tree's keyed leaf was published"
    assert sorted(f.name for f in served.iterdir()) == before, "the served tree's leaf was touched"


def test_an_opted_in_reclaim_still_keeps_a_leaf_in_use(monkeypatch, tmp_path):
    """Even a single-tree checkout that opts in (TT_DIT_CACHE_RECLAIM_SUPERSEDED=1) keeps a leaf
    some process read or wrote within RECLAIM_IDLE_HOURS: another tree may still serve it."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    legacy = _legacy_dir()
    _module().save(legacy)

    cache._reclaim_superseded(legacy)
    assert legacy.is_dir()

    old = time.time() - (cache.RECLAIM_IDLE_HOURS + 1) * 3600
    for f in [*legacy.iterdir(), legacy]:
        os.utime(f, (old, old))
    cache._reclaim_superseded(legacy)
    assert not legacy.exists(), "an idle superseded leaf is reclaimed when opted in"


def test_ltx_rename_hits(checkpoint, monkeypatch, tmp_path):
    """ltx-server's model_rename moves a checkpoint to its code name in place:
    same bytes, same size and mtime, new name and ctime. Its cache key must not move, or every
    rename rebuilds a 37GB DiT cache for weights that did not change."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    before = cache.content_key(_module(), [checkpoint])

    time.sleep(0.01)  # a rename after the build moves the ctime on
    renamed = checkpoint.with_name("q3a.safetensors")
    os.rename(checkpoint, renamed)

    assert cache.content_key(_module(), [renamed]) == before
    assert cache.source_id(renamed) == cache.source_id(renamed.resolve())


def test_a_cache_built_under_the_old_source_key_is_adopted_not_rebuilt(checkpoint, monkeypatch, tmp_path):
    """The rename-stable key spells the source differently, so every leaf the previous trees built
    sits under the old key. A load must adopt it (hard links, new manifest) rather than rebuild,
    and leave the old leaf, which a tree still on the old key serves, as it was."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    old_key = cache.content_key(_module(), [checkpoint], _source_id=_old_source_id)
    old_dir = cache.model_cache_dir(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
        content=old_key,
    )
    cache._publish_cache(_module(), old_dir, old_key)
    old_manifest = (old_dir / cache.CACHE_DICT_FILE).read_text()

    module = _LoadableStub()
    _fill(module, [checkpoint], monkeypatch)

    new_dir = _key(_module(), [checkpoint])
    assert new_dir != old_dir
    assert module.loaded_from == [str(new_dir)], "a cache hit, with no weight rebuilt"
    assert (old_dir / cache.CACHE_DICT_FILE).read_text() == old_manifest
    assert (new_dir / "weight.tensorbin").stat().st_ino == (old_dir / "weight.tensorbin").stat().st_ino


def _old_source_id(path):
    """`source_id` as trees before the rename-stable key spelled it (size, mtime, ctime, inode)."""
    st = Path(path).resolve().stat()
    return f"stat:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}"


def _old_leaf(checkpoint):
    old_key = cache.content_key(_module(), [checkpoint], _source_id=_old_source_id)
    old_dir = cache.model_cache_dir(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
        content=old_key,
    )
    cache._publish_cache(_module(), old_dir, old_key)
    return old_dir


def test_the_old_key_keeps_a_recorded_sha256(checkpoint, monkeypatch):
    """A server that wraps `source_id` with a recorded content hash got that hash under the old
    key too, so the old spelling of such a source is the wrapper's answer, not a stat key."""
    recorded = "sha256:" + "ab" * 32
    monkeypatch.setattr(cache, "source_id", lambda path: recorded)
    assert cache.legacy_source_id(checkpoint) == recorded


def test_adoption_does_not_report_the_old_leaf_as_loaded(checkpoint, monkeypatch, tmp_path):
    """A server wraps `model_cache_dir` to log every leaf a load resolves to (its prune keeps those).
    Looking for an old-key leaf to adopt is not a load of it: only the new leaf is reported."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    old_dir = _old_leaf(checkpoint)
    reported = []
    wrapped = cache.model_cache_dir

    def logging_model_cache_dir(**kw):
        path = wrapped(**kw)
        if kw.get("content") is not None:
            reported.append(path)
        return path

    monkeypatch.setattr(cache, "model_cache_dir", logging_model_cache_dir)
    _fill(_LoadableStub(), [checkpoint], monkeypatch)
    assert old_dir not in reported
    assert set(reported) == {_key(_module(), [checkpoint])}


def test_a_refused_link_adopts_nothing(checkpoint, monkeypatch, tmp_path):
    """Across filesystems a link is refused; a copy would need a second full cache with no room
    check, so the load builds the leaf instead and leaves no partial one behind."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    _old_leaf(checkpoint)

    def refuse(src, dst, **kw):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(cache.os, "link", refuse)
    module = _LoadableStub()
    _fill(module, [checkpoint], monkeypatch)
    assert module.loaded_from[0] == "torch", "built, not adopted"
    assert not list(_key(_module(), [checkpoint]).parent.glob(".*staging-*")), "staging directory leaked"


def _b_publishes_when_a_looks(monkeypatch, new_dir, key):
    """Interleave process B into process A: B publishes `new_dir` complete right after A first reads it
    (as incomplete) or right before A first removes or moves into it, whichever A does first. Returns
    [inode of B's weight.tensorbin] once B has published."""
    published = []

    def b_publishes():
        if not published:
            published.append(None)
            cache._publish_cache(_module(), new_dir, key)
            published[0] = (new_dir / "weight.tensorbin").stat().st_ino

    def after_a_checks(real):
        def call(path, *a, **kw):
            result = real(path, *a, **kw)
            if Path(path) == new_dir:
                b_publishes()
            return result

        return call

    def before_a_moves(real):
        def call(src, *a, **kw):
            if new_dir in (Path(src), Path(a[0]) if a and not isinstance(a[0], bool) else None):
                b_publishes()
            return real(src, *a, **kw)

        return call

    monkeypatch.setattr(cache, "_cache_is_complete", after_a_checks(cache._cache_is_complete))
    for name, mod in (("rename", cache.os), ("replace", cache.os), ("rmtree", cache.shutil)):
        monkeypatch.setattr(mod, name, before_a_moves(getattr(mod, name)))
    return published


def test_adoption_never_deletes_a_leaf_another_process_published_meanwhile(checkpoint, monkeypatch, tmp_path):
    """Process A finds the new-key leaf incomplete; process B then publishes it complete and starts
    loading from it. A must keep B's leaf and report it complete, never delete it to put its own
    links in place."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    _old_leaf(checkpoint)
    key = cache.content_key(_module(), [checkpoint])
    new_dir = _key(_module(), [checkpoint])
    key_args = dict(
        model_name="m",
        subfolder="connector",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
    )
    published = _b_publishes_when_a_looks(monkeypatch, new_dir, key)
    assert cache._adopt_legacy_keyed(_module(), [checkpoint], key_args, new_dir, key)
    assert published and (new_dir / "weight.tensorbin").stat().st_ino == published[0], "B's leaf was replaced"
    assert not list(new_dir.parent.glob(".*staging-*")), "staging directory leaked"


def test_a_build_never_deletes_a_leaf_another_process_published_meanwhile(checkpoint, monkeypatch, tmp_path):
    """The same interleaving for a build: A's publish keeps B's complete leaf rather than replace it."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    key = cache.content_key(_module(), [checkpoint])
    new_dir = _key(_module(), [checkpoint])
    published = _b_publishes_when_a_looks(monkeypatch, new_dir, key)
    cache._publish_cache(_module(), new_dir, key)
    assert published and (new_dir / "weight.tensorbin").stat().st_ino == published[0], "B's leaf was replaced"
    assert not list(new_dir.parent.glob(".*staging-*")), "staging directory leaked"


def test_two_builders_of_one_stale_leaf_never_delete_a_complete_one(checkpoint, monkeypatch, tmp_path):
    """Both builders find the same stale, incomplete leaf. A has decided to delete it when B deletes
    it too and publishes its complete leaf; A must not then delete B's leaf. The stale leaf is still
    cleaned up, and the key ends complete."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.tensorbin").write_bytes(b"half a build")

    a_at_the_delete, b_published = threading.Event(), threading.Event()
    complete_when_deleted, errors = [], []
    real_swap = cache._swap_into_place

    def swap(staging, target):
        if Path(target) == cache_dir:
            if threading.current_thread().name == "A":
                a_at_the_delete.set()
                b_published.wait(1.0)  # B never gets here while A holds the leaf's lock
            complete_when_deleted.append(cache._cache_is_complete(cache_dir, module, key))
        return real_swap(staging, target)

    monkeypatch.setattr(cache, "_swap_into_place", swap)

    def build():
        try:
            cache._publish_cache(_module(), cache_dir, key)
        except BaseException as err:  # reported by the assert below
            errors.append(err)

    a = threading.Thread(target=build, name="A")
    a.start()
    assert a_at_the_delete.wait(5), "A never reached the delete"
    build()  # B, on this thread
    b_published.set()
    a.join(5)

    assert not errors, errors
    assert complete_when_deleted and not any(complete_when_deleted), "a complete leaf was deleted"
    assert cache._cache_is_complete(cache_dir, module, key)
    assert not (cache_dir / "stale.tensorbin").exists(), "the stale leaf was not cleaned up"
    assert not list(cache_dir.parent.glob(".*staging-*")), "staging directory leaked"


def _killed_publish(module, cache_dir, key, step):
    """Run `_publish_cache` in a forked child that dies (os._exit, as SIGKILL would) at `step`."""
    pid = os.fork()
    if pid == 0:  # child: never returns into pytest
        try:
            if step == "save":
                real_save = module.save

                def save(directory, **kw):
                    real_save(directory, **kw)
                    os._exit(9)

                module.save = save
            real_swap = cache._swap_into_place

            def swap(staging, target):
                if step == "before_swap":
                    os._exit(9)
                real_swap(staging, target)
                os._exit(9)  # after the swap, before the swapped-out leaf is removed

            cache._swap_into_place = swap
            cache._publish_cache(module, cache_dir, key)
        finally:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


@pytest.mark.parametrize("step", ["save", "before_swap", "after_swap"])
def test_a_killed_publisher_never_removes_a_complete_leaf(step, checkpoint, monkeypatch, tmp_path):
    """A worker preempted, SIGTERMed or reset anywhere in a publish leaves the complete leaf it
    found exactly as it was (a reader may be loading it)."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])
    cache._publish_cache(_module(), cache_dir, key)
    inode = (cache_dir / "weight.tensorbin").stat().st_ino

    _killed_publish(_module(), cache_dir, key, step)

    assert cache._cache_is_complete(cache_dir, module, key)
    assert (cache_dir / "weight.tensorbin").stat().st_ino == inode, "the complete leaf was replaced"


@pytest.mark.parametrize("step", ["save", "before_swap", "after_swap"])
def test_a_publisher_killed_while_replacing_a_stale_leaf_never_leaves_none(step, checkpoint, monkeypatch, tmp_path):
    """Replacing an incomplete leaf is one atomic swap: killed before it, the old leaf is still
    there (and still misses); killed after it, the new complete leaf is. Never no leaf at all,
    which a delete-then-move would leave between its two steps."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.tensorbin").write_bytes(b"half a build")

    assert _killed_publish(_module(), cache_dir, key, step) == 9, "the child did not die at the step"

    assert cache_dir.is_dir(), "a killed publisher left no leaf"
    if step == "after_swap":
        assert cache._cache_is_complete(cache_dir, module, key)
    else:
        assert (cache_dir / "stale.tensorbin").exists()
    # The next builder finishes the job and leaves nothing of the killed one behind.
    cache._publish_cache(_module(), cache_dir, key)
    assert cache._cache_is_complete(cache_dir, module, key)


def test_swap_into_place_keeps_the_old_leaf_at_the_staging_path(tmp_path):
    staging, leaf = tmp_path / ".leaf.staging-1-ab", tmp_path / "leaf"
    staging.mkdir()
    (staging / "new").touch()
    leaf.mkdir()
    (leaf / "old").touch()
    cache._swap_into_place(staging, leaf)
    assert (leaf / "new").exists() and (staging / "old").exists()


def test_swap_into_place_without_rename_exchange(tmp_path, monkeypatch):
    """Kernels or filesystems without RENAME_EXCHANGE get the rename-aside fallback."""
    monkeypatch.setattr(cache, "_rename_exchange", lambda a, b: False)
    test_swap_into_place_keeps_the_old_leaf_at_the_staging_path(tmp_path)
    assert not list(tmp_path.glob("*.old")), "the aside dir leaked"


def test_reclaim_never_deletes_through_a_linked_leaf(checkpoint, monkeypatch, tmp_path):
    """ltx-server links the base H3 root's shared VAE-decoder leaf into the 10Eros root. A tree that
    opts in to reclaim and misses its keyed leaf in the 10Eros root must not delete the base
    root's content-blind leaf through that link: the served tree reads it (blx04, 11-26: three
    10-minute cold loads after a test tree's reclaim emptied it)."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(shared))
    served = _legacy_dir()
    _module().save(served)
    (served / cache.CACHE_DICT_FILE).touch()
    old = time.time() - (cache.RECLAIM_IDLE_HOURS + 1) * 3600
    for f in [*served.iterdir(), served]:
        os.utime(f, (old, old))

    variant = tmp_path / "variant"
    linked = variant / served.parent.relative_to(shared)
    linked.parent.mkdir(parents=True)
    linked.symlink_to(served.parent, target_is_directory=True)
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(variant))
    monkeypatch.setenv("TT_DIT_CACHE_RECLAIM_SUPERSEDED", "1")
    assert _legacy_dir() == linked / served.name

    module = _LoadableStub()
    _fill(module, [checkpoint], monkeypatch)

    assert module.loaded_from[0] == "torch", "the stub should have missed and built"
    assert (served / cache.CACHE_DICT_FILE).exists(), "the linked-in leaf was deleted"


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


def test_a_capture_pass_miss_names_the_capture_pass(checkpoint, monkeypatch, tmp_path):
    """The prewarm capture pass misses by design and never writes. Its anomaly must say so, or it
    reads as an unstable key and sends the next reader chasing one."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TT_METAL_KERNEL_CAPTURE_ONLY", "1")
    cache_dir = _key(_module(), [checkpoint])

    assert "capture-only" in cache.miss_reason(cache_dir)


def test_a_miss_says_whether_the_cache_was_absent_or_rejected(checkpoint, monkeypatch, tmp_path):
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TT_METAL_KERNEL_CAPTURE_ONLY", raising=False)
    module, key = _module(), cache.content_key(_module(), [checkpoint])
    cache_dir = _key(module, [checkpoint])

    assert "no cache" in cache.miss_reason(cache_dir)

    cache._publish_cache(module, cache_dir, key)
    (cache_dir / "weight.tensorbin").write_bytes(b"\x00" * 8)

    assert "rejected" in cache.miss_reason(cache_dir)


def test_a_miss_under_tmp_warns_that_reboots_wipe_it(monkeypatch):
    monkeypatch.delenv("TT_METAL_KERNEL_CAPTURE_ONLY", raising=False)

    assert "wiped at every boot" in cache.miss_reason("/tmp/tt-dit-cache-test/absent/key")
    assert "wiped" not in cache.miss_reason("/var/tmp/tt-dit-cache-test/absent/key")


class _LoadingModule(_StubModule):
    """A stub module that records the load_model calls that place its weights on the device."""

    def __init__(self, log):
        super().__init__({"weight": _StubParameter(), "bias": _StubParameter(shape=(32,), mesh_axes=(None,))})
        self.log, self.loaded = log, False

    def is_loaded(self):
        return self.loaded

    def load_torch_state_dict(self, state):
        self.log.append("state dict")
        self.loaded = True

    def save(self, directory, /, *, prefix=""):
        self.log.append("save")
        super().save(directory, prefix=prefix)

    def deallocate_weights(self):
        self.log.append("deallocate")
        self.loaded = False

    def load(self, directory, /):
        self.log.append(f"load {Path(directory).name}")
        self.loaded = True


def _load(module, checkpoint, hook):
    cache.load_model(
        module,
        model_name="m",
        subfolder="transformer",
        parallel_config=SimpleNamespace(_asdict=dict),
        mesh_shape=(4, 8),
        mesh_device=None,
        sources=[checkpoint],
        get_torch_state_dict=dict,
        post_load_hook=hook,
    )


def test_a_miss_places_the_weights_as_a_cache_hit_does(checkpoint, monkeypatch, tmp_path):
    """A trace captured on weights built from the state dict, then replayed after a dynamic-load mesh
    reloaded them from the cache, rendered pure noise (f07cs04 jobs 2749-2750 and 2753-2754): the two
    paths put the weights in different places. So a miss reloads from the cache it just published, and
    every load leaves the weights where a cache hit puts them."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TT_METAL_KERNEL_CAPTURE_ONLY", raising=False)
    monkeypatch.setattr(cache.ttnn, "distributed_context_barrier", lambda: None)
    log = []
    module = _LoadingModule(log)
    leaf = _key(module, [checkpoint]).name

    _load(module, checkpoint, lambda m: log.append("hook"))

    assert log == ["state dict", "hook", "save", "deallocate", f"load {leaf}", "hook"]
    log.clear()
    module.deallocate_weights()
    _load(module, checkpoint, lambda m: log.append("hook"))  # the next reload: a plain hit, placed the same way
    assert log == ["deallocate", f"load {leaf}", "hook"]


def test_a_miss_that_publishes_nothing_keeps_the_weights_it_built(checkpoint, monkeypatch, tmp_path):
    """The capture-only pass writes no cache, so there is nothing to reload from."""
    monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TT_METAL_KERNEL_CAPTURE_ONLY", "1")
    monkeypatch.setattr(cache.ttnn, "distributed_context_barrier", lambda: None)
    log = []
    _load(_LoadingModule(log), checkpoint, lambda m: log.append("hook"))
    assert log == ["state dict", "hook"]


# A published cache directory must be readable by the other users of a shared cache root:
# tempfile.mkdtemp staged it 0700 and the rename published that mode; under a default ACL that also
# set the leaf mask to ---, so the serving account could not read a cache another user had built.
class _SavesOneFile:
    def save(self, path) -> None:
        (path / "w.tensorbin").write_bytes(b"x")


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_published_cache_dir_gets_the_umask_mode_not_0700(tmp_path, umask_022) -> None:
    leaf = tmp_path / "model" / "transformer" / "leaf"
    cache._publish_cache(_SavesOneFile(), leaf, None)
    assert (leaf / "w.tensorbin").read_bytes() == b"x"
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o755
    assert not [p for p in leaf.parent.iterdir() if ".staging-" in p.name]


def test_published_cache_dir_keeps_the_parents_setgid_group(tmp_path, umask_022) -> None:
    parent = tmp_path / "root"
    parent.mkdir()
    parent.chmod(0o2775)
    cache._publish_cache(_SavesOneFile(), parent / "leaf", None)
    assert (parent / "leaf").stat().st_mode & stat.S_ISGID


@pytest.mark.skipif(not (shutil.which("setfacl") and shutil.which("getfacl")), reason="needs setfacl/getfacl")
def test_published_cache_dir_has_an_open_acl_mask_under_a_default_acl(tmp_path, umask_022) -> None:
    """The shared cache root grants the serving account a named default ACL entry, which brings a
    mask; a 0700 staging dir published mask ---, denying that account."""
    parent = tmp_path / "root"
    parent.mkdir()
    spec = f"u::rwx,u:{os.getuid()}:rwx,g::rwx,o::rx"
    if subprocess.run(["setfacl", "-d", "-m", spec, str(parent)], capture_output=True).returncode:
        pytest.skip("filesystem without ACL support")
    cache._publish_cache(_SavesOneFile(), parent / "leaf", None)
    acl = subprocess.run(["getfacl", "-p", str(parent / "leaf")], capture_output=True, text=True, check=True).stdout
    access = [line for line in acl.splitlines() if line and not line.startswith(("#", "default:"))]
    assert "mask::rwx" in access, acl
