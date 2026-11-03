# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MooncakeEmbeddingStore's read logic against a fake Mooncake client: no Mooncake needed. The same paths against a
real local master are in test_multimedia_store_mooncake.py."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from models.common.multimedia_store.mooncake_store import (
    OBJECT_NOT_FOUND,
    MediaStoreConfig,
    MooncakeEmbeddingStore,
    parse_bytes,
    store_key,
)
from models.common.multimedia_store.reader import Missing, StoreUnavailable

ME, OTHER = "10.0.0.1:17800", "10.0.0.2:17800"


def _replica(segment: str, data: np.ndarray, status: str = "COMPLETE"):
    desc = SimpleNamespace(transport_endpoint=segment, buffer_address=data.ctypes.data, size=data.nbytes)
    return SimpleNamespace(
        is_memory_replica=lambda: True,
        status=SimpleNamespace(name=status),
        get_memory_descriptor=lambda: SimpleNamespace(buffer_descriptor=desc),
    )


def _write(ptr: int, data: np.ndarray) -> None:
    np.ctypeslib.as_array((ctypes.c_uint8 * data.nbytes).from_address(ptr))[:] = data


class FakeClient:
    """Objects keyed like Mooncake: {key: (segment, bytes, status)}. Local objects live in this process. Counts the
    calls that go to the master."""

    def __init__(self):
        self.objects = {}
        self.master_up = True
        self.lookups = 0

    def register_buffer(self, ptr, size):
        return 0

    def health_check(self):
        return 0 if self.master_up else 2

    def get_replica_desc(self, key):
        self.lookups += 1
        if not self.master_up or key not in self.objects:
            return []
        segment, data, status = self.objects[key]
        return [_replica(segment, data, status)]

    def get_into(self, key, ptr, size):
        self.lookups += 1
        if key not in self.objects:
            return OBJECT_NOT_FOUND
        _write(ptr, self.objects[key][1][:size])
        return size

    def get_into_ranges(self, ptrs, keys, dst_offsets, src_offsets, sizes):
        self.lookups += 2
        (ptr,), ((key,),) = ptrs, keys
        if key not in self.objects:
            return [[[OBJECT_NOT_FOUND] * len(sizes[0][0])]]
        src = self.objects[key][1]
        for d, s, n in zip(dst_offsets[0][0], src_offsets[0][0], sizes[0][0]):
            _write(ptr + d, src[s : s + n])
        return [list(sizes[0])]

    def close(self):
        return 0


def _key(i: int) -> bytes:
    return bytes([i]) * 32


@pytest.fixture
def setup():
    client = FakeClient()
    store = MooncakeEmbeddingStore(client, ME, read_buffer_bytes=1 << 20)
    rng = np.random.default_rng(0)
    local = rng.integers(0, 256, 300_000, dtype=np.uint8)
    remote = rng.integers(0, 256, 200_000, dtype=np.uint8)
    client.objects[store_key(_key(1))] = (ME, local, "COMPLETE")
    client.objects[store_key(_key(2))] = (OTHER, remote, "COMPLETE")
    return client, store, local, remote


def test_store_key_is_hex_of_32_bytes():
    assert store_key(bytes(range(32))) == bytes(range(32)).hex()
    with pytest.raises(ValueError):
        store_key(b"short")


@pytest.mark.parametrize(
    "text, want", [("50G", 50 << 30), ("50GiB", 50 << 30), ("512M", 512 << 20), ("4096", 4096), ("1t", 1 << 40)]
)
def test_parse_bytes(text, want):
    assert parse_bytes(text) == want


def test_config_off_without_master():
    assert MediaStoreConfig.from_env({}) is None


def test_config_from_env():
    env = {
        "PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051",
        "PREFILL_MEDIA_STORE_SEGMENT": ME,
        "PREFILL_MEDIA_STORE_BYTES": "50G",
        "PREFILL_MEDIA_STORE_DEVICE": "rocep201s0f0",
    }
    c = MediaStoreConfig.from_env(env)
    assert (c.master, c.segment, c.segment_bytes, c.device) == ("10.0.0.9:50051", ME, 50 << 30, "rocep201s0f0")
    assert (c.metadata, c.protocol) == ("http://10.0.0.9:8080/metadata", "rdma")


def test_config_needs_segment_and_size():
    with pytest.raises(ValueError, match="SEGMENT and PREFILL_MEDIA_STORE_BYTES"):
        MediaStoreConfig.from_env({"PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051"})


def test_local_value_is_read_in_place_with_one_lookup(setup):
    client, store, local, _ = setup
    view = store.fetch(_key(1))
    assert bytes(view) == local.tobytes()
    local[0] ^= 0xFF  # the view aliases the segment
    assert view[0] == local[0]
    assert view.readonly
    assert client.lookups == 1
    assert store.size(_key(1)) == local.nbytes
    s = store.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["bytes_read"]) == (1, 0, local.nbytes)


def test_remote_value_is_copied(setup):
    client, store, _, remote = setup
    view = store.fetch(_key(2))
    assert bytes(view) == remote.tobytes()
    assert client.lookups == 2  # the lookup, then get_into's own
    s = store.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["bytes_read"]) == (0, 1, remote.nbytes)


@pytest.mark.parametrize("key", [1, 2], ids=["local", "remote"])
def test_ranges(setup, key):
    _, store, local, remote = setup
    src = local if key == 1 else remote
    ranges = [(0, 64), (1000, 5000), (src.nbytes - 7, 7), (10, 0), (src.nbytes - 100, -1)]
    want = [src[0:64], src[1000:6000], src[-7:], src[10:10], src[-100:]]
    got = store.fetch_ranges(_key(key), ranges)
    assert [bytes(v) for v in got] == [w.tobytes() for w in want]
    assert store.stats.snapshot()["bytes_read"] == sum(w.nbytes for w in want)


def test_closed_store_is_unavailable(setup):
    _, store, _, _ = setup
    store.close()
    for read in (store.fetch, store.size):
        with pytest.raises(StoreUnavailable, match="closed"):
            read(_key(1))


def test_range_outside_value(setup):
    _, store, local, _ = setup
    with pytest.raises(ValueError):
        store.fetch_ranges(_key(1), [(local.nbytes - 1, 2)])


def test_remote_read_larger_than_read_buffer(setup):
    client, store, _, _ = setup
    big = np.zeros(2 << 20, dtype=np.uint8)
    client.objects[store_key(_key(3))] = (OTHER, big, "COMPLETE")
    for read in (store.fetch, lambda k: store.fetch_ranges(k, [(0, big.nbytes - 1)])):
        with pytest.raises(StoreUnavailable, match="read buffer"):
            read(_key(3))


def test_missing_key_costs_one_lookup(setup):
    client, store, _, _ = setup
    for read in (store.fetch, store.size, lambda k: store.fetch_ranges(k, [(0, 1)])):
        with pytest.raises(Missing):
            read(_key(9))
    assert client.lookups == 3
    s = store.stats.snapshot()
    assert (s["misses"], s["hits"], s["errors"]) == (2, 0, 0)  # size() is a lookup, not a fetch


def test_missing_is_a_key_error_not_a_store_failure(setup):
    _, store, _, _ = setup
    with pytest.raises(KeyError):
        store.fetch(_key(9))


def test_unfinished_put_reads_as_missing(setup):
    client, store, local, _ = setup
    client.objects[store_key(_key(4))] = (ME, local, "PROCESSING")
    with pytest.raises(Missing):
        store.fetch(_key(4))


def test_unreachable_master_is_unavailable(setup):
    client, store, _, _ = setup
    client.master_up = False
    with pytest.raises(StoreUnavailable):
        store.fetch(_key(1))
    assert store.stats.snapshot()["errors"] == 1


def test_latency_stats(setup):
    _, store, _, _ = setup
    for _ in range(3):
        store.fetch(_key(1))
    s = store.stats.snapshot()
    assert s["fetches"] == 3 and 0 <= s["fetch_ms_mean"] <= s["fetch_ms_max"]
