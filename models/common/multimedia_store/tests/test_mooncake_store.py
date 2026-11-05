# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MooncakeEmbeddingReader's read logic against a fake Mooncake client. The module imports Mooncake, so this needs the
wheel (models/common/multimedia_store/requirements.txt); the same paths against a real local master are in
test_mooncake_store_integration.py."""

import ctypes
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from models.common.multimedia_store.reader import FetchRequest, Missing, StoreUnavailable

if not sys.platform.startswith("linux"):
    pytest.skip("Mooncake publishes Linux wheels only", allow_module_level=True)
try:
    from models.common.multimedia_store.mooncake_store import OBJECT_NOT_FOUND, MooncakeEmbeddingReader
except ImportError as e:  # a broken env, so it fails rather than skips
    pytest.fail(f"no Mooncake client ({e}): uv pip install -r models/common/multimedia_store/requirements.txt", False)

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
        self.registrable = 1 << 40  # bytes register_buffer accepts in one buffer

    def put(self, key, segment: str, data: np.ndarray, status: str = "COMPLETE", replicas: int = 1):
        """Stores under the store key; a bytes media key is written under its lowercase hex, as the encoder does."""
        self.objects[key.hex() if isinstance(key, bytes) else key] = (segment, data, status, replicas)

    def register_buffer(self, ptr, size):
        if size > self.registrable:
            return -1
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

    def prepare_get_into_ranges_snapshot(self, keys):
        self.lookups += 1  # one batched lookup
        return frozenset(keys)

    def get_into_ranges_from_snapshot(self, snapshot, ptrs, keys, dst_offsets, src_offsets, sizes):
        ((ptr,), (keys,), (dsts,), (srcs,), (lens,)) = (ptrs, keys, dst_offsets, src_offsets, sizes)
        assert set(keys) <= snapshot and ptr in self.registered
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
    return hashlib.sha256(f"media-{i}".encode()).digest()


@pytest.fixture
def setup():
    client = FakeClient()
    reader = MooncakeEmbeddingReader(client, ME)
    rng = np.random.default_rng(0)
    local = rng.integers(0, 256, 300_000, dtype=np.uint8)
    remote = rng.integers(0, 256, 200_000, dtype=np.uint8)
    client.put(_key(1), ME, local)
    client.put(_key(2), OTHER, remote)
    return client, reader, local, remote


def test_media_keys_are_read_under_their_hex(setup):
    """By default the caller passes the encoder's raw 32-byte media key, and the reader looks up its lowercase hex, the
    key the encoder wrote under."""
    client, reader, local, remote = setup
    assert reader.key_transform == "hex" and _key(1).hex() in client.objects  # setup's put, as the encoder writes
    assert bytes(reader.fetch(_key(1))) == local.tobytes() and reader.size(_key(2)) == remote.nbytes


def test_untransformed_keys_are_read_as_they_are(setup, expect_error):
    """With MEDIA_STORE_KEY_TRANSFORM=none a media key is its own store key, which Mooncake needs to be valid UTF-8."""
    client, _, local, _ = setup
    reader = MooncakeEmbeddingReader(client, ME, key_transform="none")
    client.put("media-\u00e9", ME, local)  # non-ASCII text is valid UTF-8, so the batched lookup returns it
    assert bytes(reader.fetch("media-\u00e9".encode())) == local.tobytes()
    with expect_error(Missing, "media-\u00e9-absent"):
        reader.fetch("media-\u00e9-absent".encode())
    with expect_error(ValueError, "is not valid UTF-8"):
        reader.fetch(_key(1))  # a raw digest needs the default hex transform


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


def test_remote_value_is_a_view_into_the_read_buffer(setup):
    client, reader, _, remote = setup
    view = reader.fetch(_key(2))
    assert bytes(view) == remote.tobytes() and view.readonly
    assert np.frombuffer(view, np.uint8).ctypes.data in client.registered  # no copy out of the read buffer
    assert client.lookups == 2  # the reader's batched lookup, then the snapshot's
    s = reader.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["bytes_read"]) == (0, 1, remote.nbytes)


def test_batch_mixes_local_and_remote_in_one_lookup(setup):
    client, reader, local, remote = setup
    requests = [
        FetchRequest(_key(1), 1000, 5000),
        FetchRequest(_key(2), 64, 2000),
        FetchRequest(_key(2), remote.nbytes - 100),
        FetchRequest(_key(1), 10, 0),
        FetchRequest(_key(2), 10, 0),
        FetchRequest(_key(1)),
    ]
    got = reader.fetch_batch(requests)
    assert bytes(got[0]) == local[1000:6000].tobytes()
    assert bytes(got[1]) == remote[64:2064].tobytes()
    assert bytes(got[2]) == remote[-100:].tobytes()
    assert bytes(got[3]) == bytes(got[4]) == b""
    assert bytes(got[5]) == local.tobytes()
    assert client.lookups == 2  # one batched lookup for all keys, then the snapshot of the one remote key
    s = reader.stats.snapshot()
    assert (s["local_hits"], s["remote_hits"], s["fetch_calls"]) == (3, 3, 1)


def test_one_missing_key_fails_the_batch(setup, expect_error):
    _, reader, _, _ = setup
    with expect_error(Missing, _key(9).hex()):
        reader.fetch_batch([FetchRequest(_key(1)), FetchRequest(_key(9)), FetchRequest(_key(2))])
    s = reader.stats.snapshot()
    assert (s["misses"], s["hits"]) == (1, 0)


def test_range_outside_value(setup, expect_error):
    _, reader, local, _ = setup
    with expect_error(ValueError, "outside the"):
        reader.fetch_batch([FetchRequest(_key(1), local.nbytes - 1, 2)])


def test_read_buffer_starts_empty_and_grows_on_demand(setup):
    client, reader, _, remote = setup
    assert reader.stats.snapshot()["read_buffer_bytes"] == 0 and not client.registered
    reader.fetch(_key(1))  # in place: no buffer needed
    assert not client.registered
    reader.fetch(_key(2))
    assert list(client.registered.values()) == [remote.nbytes]
    big = np.random.default_rng(1).integers(0, 256, 3 << 20, dtype=np.uint8)
    client.put(_key(3), OTHER, big)
    assert bytes(reader.fetch(_key(3))) == big.tobytes()
    grown = reader.stats.snapshot()["read_buffer_bytes"]
    assert grown >= big.nbytes and list(client.registered.values()) == [grown]  # the old buffer was unregistered


def test_failed_buffer_registration_is_unavailable(setup, expect_error):
    """A read buffer Mooncake refuses to register (e.g. over `ulimit -l` with RDMA) fails the read, and the next read
    that fits registers one."""
    client, reader, local, remote = setup
    client.registrable = remote.nbytes - 1
    with expect_error(StoreUnavailable, "registering a"):
        reader.fetch_batch([FetchRequest(_key(1)), FetchRequest(_key(2))])
    assert not client.registered and reader.stats.snapshot()["errors"] == 1
    assert bytes(reader.fetch(_key(1))) == local.tobytes()  # in place: needs no buffer
    (rows,) = reader.fetch_batch([FetchRequest(_key(2), 0, 4096)])
    assert bytes(rows) == remote[:4096].tobytes() and list(client.registered.values()) == [4096]


def test_missing_key(setup, expect_error):
    _, reader, _, _ = setup
    for read in (reader.fetch, reader.size):
        with expect_error(Missing, _key(9).hex()):
            read(_key(9))
    with expect_error(KeyError, _key(9).hex()):  # Missing is a KeyError, not a store failure
        reader.fetch(_key(9))
    s = reader.stats.snapshot()
    assert (s["misses"], s["hits"], s["errors"]) == (2, 0, 0)  # size() is a lookup, not a fetch


def test_unfinished_put_reads_as_missing(setup, expect_error):
    client, reader, local, _ = setup
    client.put(_key(4), ME, local, status="PROCESSING")
    with expect_error(Missing, "not a complete memory replica"):
        reader.fetch(_key(4))


def test_replicated_value_is_unavailable(setup, expect_error):
    client, reader, local, _ = setup
    client.put(_key(5), ME, local, replicas=2)
    with expect_error(StoreUnavailable, "2 replicas"):
        reader.fetch(_key(5))


def test_unreachable_master_is_unavailable(setup, expect_error):
    client, reader, _, _ = setup
    client.master_up = False
    with expect_error(StoreUnavailable, "unreachable"):
        reader.fetch_batch([FetchRequest(_key(1)), FetchRequest(_key(2))])
    assert reader.stats.snapshot()["errors"] == 1


def test_closed_reader_is_unavailable(setup, expect_error):
    _, reader, _, _ = setup
    reader.close()
    for read in (reader.fetch, reader.size):
        with expect_error(StoreUnavailable, "closed"):
            read(_key(1))


def test_latency_stats(setup):
    _, reader, _, _ = setup
    for _ in range(3):
        reader.fetch(_key(1))
    s = reader.stats.snapshot()
    assert s["fetch_calls"] == 3 and 0 <= s["fetch_ms_mean"] <= s["fetch_ms_max"]


def test_plain_tuples_and_an_empty_batch(setup):
    client, reader, local, _ = setup
    (got,) = reader.fetch_batch([(_key(1), 0, 16)])
    assert bytes(got) == local[:16].tobytes()
    lookups = client.lookups
    assert reader.fetch_batch([]) == [] and client.lookups == lookups


@pytest.mark.parametrize("rc", [-800, -900], ids=["peer_gone", "master_gone"])
def test_failed_remote_read_is_unavailable(setup, rc, expect_error):
    """The codes Mooncake returns when the host holding the value, or the master, has died."""
    client, reader, _, _ = setup
    client.get_into_ranges_from_snapshot = lambda *args: [[[rc]]]
    with expect_error(StoreUnavailable, f"read {rc} of"):
        reader.fetch_batch([FetchRequest(_key(1), 0, 8), FetchRequest(_key(2), 0, 8)])
