# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MooncakeEmbeddingReader's read logic against a fake Mooncake client: no Mooncake needed. The same paths against a
real local master are in test_multimedia_store_mooncake.py."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from models.common.multimedia_store.mooncake_store import (
    OBJECT_NOT_FOUND,
    MediaEmbeddingStoreConfig,
    MooncakeEmbeddingReader,
    parse_bytes,
)
from models.common.multimedia_store.reader import FetchRequest, Missing, StoreUnavailable

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
    """Objects keyed like Mooncake: {key: (segment, bytes, status, replica count)}. Local objects live in this process.
    Counts the round trips to the master and the registered buffers."""

    def __init__(self):
        self.objects = {}
        self.master_up = True
        self.lookups = 0
        self.registered = {}

    def put(self, key: str, segment: str, data: np.ndarray, status: str = "COMPLETE", replicas: int = 1):
        self.objects[key] = (segment, data, status, replicas)

    def register_buffer(self, ptr, size):
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr):
        del self.registered[ptr]
        return 0

    def health_check(self):
        return 0 if self.master_up else 2

    def batch_get_replica_desc(self, keys):
        self.lookups += 1
        if not self.master_up:
            return {}
        return {k: [_replica(seg, data, status)] * n for k, (seg, data, status, n) in self.objects.items() if k in keys}

    def get_into_ranges(self, ptrs, keys, dst_offsets, src_offsets, sizes):
        self.lookups += 1 + len(keys[0])  # Mooncake looks the keys up again: one batched lookup plus one per key
        ((ptr,), (keys,), (dsts,), (srcs,), (lens,)) = (ptrs, keys, dst_offsets, src_offsets, sizes)
        assert ptr in self.registered
        out = []
        for key, d_list, s_list, n_list in zip(keys, dsts, srcs, lens):
            if key not in self.objects:
                out.append([OBJECT_NOT_FOUND] * len(n_list))
                continue
            src = self.objects[key][1]
            for d, s, n in zip(d_list, s_list, n_list):
                assert n > 0 and d + n <= self.registered[ptr], "Mooncake rejects empty or out-of-buffer ranges"
                _write(ptr + d, src[s : s + n])
            out.append(list(n_list))
        return [out]

    def close(self):
        return 0


def _key(i: int) -> bytes:
    return f"{i:032d}".encode()


@pytest.fixture
def setup():
    client = FakeClient()
    reader = MooncakeEmbeddingReader(client, ME, read_buffer_bytes=1 << 20)
    rng = np.random.default_rng(0)
    local = rng.integers(0, 256, 300_000, dtype=np.uint8)
    remote = rng.integers(0, 256, 200_000, dtype=np.uint8)
    client.put(_key(1).decode(), ME, local)
    client.put(_key(2).decode(), OTHER, remote)
    return client, reader, local, remote


def test_key_must_be_32_ascii_bytes(setup):
    _, reader, _, _ = setup
    for bad in (b"short", b"\xff" * 32):
        with pytest.raises(ValueError):
            reader.fetch(bad)


@pytest.mark.parametrize(
    "text, want", [("50G", 50 << 30), ("50GiB", 50 << 30), ("512M", 512 << 20), ("4096", 4096), ("1t", 1 << 40)]
)
def test_parse_bytes(text, want):
    assert parse_bytes(text) == want


def test_config_off_without_master():
    assert MediaEmbeddingStoreConfig.from_env({}) is None


def test_config_from_env():
    env = {
        "PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051",
        "PREFILL_MEDIA_STORE_SEGMENT": ME,
        "PREFILL_MEDIA_STORE_BYTES": "50G",
        "PREFILL_MEDIA_STORE_DEVICE": "rocep201s0f0",
    }
    c = MediaEmbeddingStoreConfig.from_env(env)
    assert (c.master, c.segment, c.segment_bytes, c.device) == ("10.0.0.9:50051", ME, 50 << 30, "rocep201s0f0")
    assert (c.metadata, c.protocol, c.read_buffer_bytes) == ("http://10.0.0.9:8080/metadata", "rdma", 10 << 20)


def test_config_needs_segment_and_size():
    with pytest.raises(ValueError, match="SEGMENT and PREFILL_MEDIA_STORE_BYTES"):
        MediaEmbeddingStoreConfig.from_env({"PREFILL_MEDIA_STORE_MASTER": "10.0.0.9:50051"})


def test_local_value_is_read_in_place_with_one_lookup(setup):
    client, reader, local, _ = setup
    view = reader.fetch(_key(1))
    assert bytes(view) == local.tobytes()
    local[0] ^= 0xFF  # the view aliases the segment
    assert view[0] == local[0] and view.readonly
    assert client.lookups == 1
    assert reader.size(_key(1)) == local.nbytes
    s = reader.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["bytes_read"]) == (1, 0, local.nbytes)


def test_remote_value_is_copied(setup):
    client, reader, _, remote = setup
    assert bytes(reader.fetch(_key(2))) == remote.tobytes()
    assert client.lookups == 3  # the batched lookup, then get_into_ranges' own two
    s = reader.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["bytes_read"]) == (0, 1, remote.nbytes)


def test_batch_mixes_local_remote_and_missing_in_one_lookup(setup):
    client, reader, local, remote = setup
    requests = [
        FetchRequest(_key(1), 1000, 5000),
        FetchRequest(_key(2), 64, 2000),
        FetchRequest(_key(9)),
        FetchRequest(_key(2), remote.nbytes - 100),
        FetchRequest(_key(1), 10, 0),
        FetchRequest(_key(2), 10, 0),
        FetchRequest(_key(1)),
    ]
    got = reader.fetch_batch(requests)
    assert bytes(got[0]) == local[1000:6000].tobytes()
    assert bytes(got[1]) == remote[64:2064].tobytes()
    assert isinstance(got[2], Missing)
    assert bytes(got[3]) == remote[-100:].tobytes()
    assert bytes(got[4]) == bytes(got[5]) == b""
    assert bytes(got[6]) == local.tobytes()
    assert client.lookups == 3  # one batched lookup for all keys, then one remote read of one key
    s = reader.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["misses"], s["fetch_calls"]) == (3, 3, 1, 1)


def test_range_outside_value(setup):
    _, reader, local, _ = setup
    with pytest.raises(ValueError):
        reader.fetch_batch([FetchRequest(_key(1), local.nbytes - 1, 2)])


def test_read_buffer_grows_on_demand(setup):
    client, reader, _, _ = setup
    big = np.random.default_rng(1).integers(0, 256, 3 << 20, dtype=np.uint8)
    client.put(_key(3).decode(), OTHER, big)
    assert bytes(reader.fetch(_key(3))) == big.tobytes()
    grown = reader.stats.snapshot()["read_buffer_bytes"]
    assert grown >= big.nbytes and list(client.registered.values()) == [grown]  # the old buffer was unregistered


def test_missing_key(setup):
    _, reader, _, _ = setup
    for read in (reader.fetch, reader.size):
        with pytest.raises(Missing):
            read(_key(9))
    with pytest.raises(KeyError):  # Missing is a KeyError, not a store failure
        reader.fetch(_key(9))
    s = reader.stats.snapshot()
    assert (s["misses"], s["hits"], s["errors"]) == (2, 0, 0)  # size() is a lookup, not a fetch


def test_unfinished_put_reads_as_missing(setup):
    client, reader, local, _ = setup
    client.put(_key(4).decode(), ME, local, status="PROCESSING")
    with pytest.raises(Missing):
        reader.fetch(_key(4))


def test_replicated_value_is_unavailable(setup):
    client, reader, local, _ = setup
    client.put(_key(5).decode(), ME, local, replicas=2)
    got = reader.fetch_batch([FetchRequest(_key(5)), FetchRequest(_key(1), 0, 8)])
    assert isinstance(got[0], StoreUnavailable) and bytes(got[1]) == local[:8].tobytes()


def test_unreachable_master_fails_every_request(setup):
    client, reader, _, _ = setup
    client.master_up = False
    got = reader.fetch_batch([FetchRequest(_key(1)), FetchRequest(_key(2))])
    assert all(isinstance(r, StoreUnavailable) for r in got)
    assert reader.stats.snapshot()["errors"] == 2


def test_closed_reader_is_unavailable(setup):
    _, reader, _, _ = setup
    reader.close()
    for read in (reader.fetch, reader.size):
        with pytest.raises(StoreUnavailable, match="closed"):
            read(_key(1))


def test_latency_stats(setup):
    _, reader, _, _ = setup
    for _ in range(3):
        reader.fetch(_key(1))
    s = reader.stats.snapshot()
    assert s["fetch_calls"] == 3 and 0 <= s["fetch_ms_mean"] <= s["fetch_ms_max"]
