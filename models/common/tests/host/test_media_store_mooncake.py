# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runner's embedding store against a real Mooncake master on this host, over TCP.

Three clients join a local master started the way the encoder service starts its own (HTTP metadata server,
allow_evict_soft_pinned_objects=false):

- ``store``: this runner's store (MooncakeEmbeddingStore.open, segment 127.0.0.1:<port>);
- ``other``: another prefill host's store (segment 127.0.0.2:<port>, a second loopback address);
- ``writer``: the encoder service stand-in, with no segment, writing with put_from, a soft pin and preferred_segment.

Needs the Mooncake wheel (pip install mooncake-transfer-engine-non-cuda==0.3.13.post1) and Linux (127.0.0.2);
skipped otherwise.
"""

import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

mooncake_store = pytest.importorskip("mooncake.store")

from models.common.media_store.mooncake_store import MediaStoreConfig, MooncakeEmbeddingStore, store_key
from models.common.media_store.reader import Missing

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs 127.0.0.2 (Linux loopback)")

# One MiniMax-M3 image: a 64-byte header and 576 bf16 rows of width 6144.
IMAGE_BYTES = 64 + 576 * 6144 * 2


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _master_binary() -> str:
    found = shutil.which("mooncake_master")
    if found:
        return found
    bundled = os.path.join(os.path.dirname(mooncake_store.__file__), "mooncake_master")
    if not os.access(bundled, os.X_OK):
        pytest.skip("mooncake_master not found")
    return bundled


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    rpc, http = _free_port(), _free_port()
    log = tmp_path_factory.mktemp("mooncake") / "master.log"
    master = subprocess.Popen(
        [
            _master_binary(),
            f"--rpc_port={rpc}",
            "--enable_http_metadata_server=true",
            f"--http_metadata_server_port={http}",
            "--allow_evict_soft_pinned_objects=false",
            f"--metrics_port={_free_port()}",
        ],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30
    while True:  # wait for the master's RPC port
        try:
            socket.create_connection(("127.0.0.1", rpc), timeout=1).close()
            break
        except OSError:
            if master.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"mooncake_master did not start; see {log}")
            time.sleep(0.2)

    def config(host: str, segment_bytes: int) -> MediaStoreConfig:
        return MediaStoreConfig(
            master=f"127.0.0.1:{rpc}",
            segment=f"{host}:{_free_port(host)}",
            segment_bytes=segment_bytes,
            metadata=f"http://127.0.0.1:{http}/metadata",
            protocol="tcp",
            read_buffer_bytes=32 << 20,
        )

    store = MooncakeEmbeddingStore.open(config("127.0.0.1", 64 << 20))
    other = MooncakeEmbeddingStore.open(config("127.0.0.2", 64 << 20))
    w = config("127.0.0.3", 0)
    writer = mooncake_store.MooncakeDistributedStore()
    assert writer.setup(w.segment, w.metadata, 0, 16 << 20, w.protocol, "", w.master) == 0
    try:
        yield SimpleCluster(store, other, writer)
    finally:
        writer.close()
        other.close()
        store.close()
        master.terminate()
        master.wait(timeout=30)


class SimpleCluster:
    def __init__(self, store, other, writer):
        self.store, self.other, self.writer = store, other, writer
        self._buffers = []
        self._n = 0

    def put(self, segment: str, nbytes: int = IMAGE_BYTES) -> tuple[bytes, np.ndarray]:
        """The encoder service's write: put_from a registered buffer into segment, soft-pinned. Returns (key, value)."""
        self._n += 1
        media_key = self._n.to_bytes(32, "big")
        value = np.random.default_rng(self._n).integers(0, 256, nbytes, dtype=np.uint8)
        assert self.writer.register_buffer(value.ctypes.data, nbytes) == 0
        self._buffers.append(value)
        config = mooncake_store.ReplicateConfig()
        config.preferred_segment = segment
        config.soft_pin_action = mooncake_store.SoftPinAction.ENABLE
        config.soft_pin_ttl_ms = 60_000
        assert self.writer.put_from(store_key(media_key), value.ctypes.data, nbytes, config) == 0
        return media_key, value

    def _descriptor(self, media_key: bytes):
        (replica,) = self.writer.get_replica_desc(store_key(media_key))
        return replica.get_memory_descriptor().buffer_descriptor

    def placed_on(self, media_key: bytes) -> str:
        return self._descriptor(media_key).transport_endpoint

    def address(self, media_key: bytes) -> int:
        return self._descriptor(media_key).buffer_address


def test_segment_is_named_as_configured(cluster):
    assert cluster.store.segment.startswith("127.0.0.1:") and cluster.store.segment != cluster.other.segment


def test_round_trip_from_this_hosts_store(cluster):
    media_key, value = cluster.put(cluster.store.segment)
    assert cluster.placed_on(media_key) == cluster.store.segment
    before = cluster.store.stats.snapshot()
    got = cluster.store.fetch(media_key)
    assert np.frombuffer(got, np.uint8).ctypes.data == cluster.address(media_key)  # read in place, no copy
    assert got.nbytes == value.nbytes == cluster.store.size(media_key)
    assert bytes(got) == value.tobytes()
    after = cluster.store.stats.snapshot()
    assert after["local_hits"] == before["local_hits"] + 1
    assert after["bytes_read"] == before["bytes_read"] + IMAGE_BYTES


def test_round_trip_from_another_hosts_store(cluster):
    media_key, value = cluster.put(cluster.other.segment)
    assert cluster.placed_on(media_key) == cluster.other.segment
    before = cluster.store.stats.snapshot()
    assert bytes(cluster.store.fetch(media_key)) == value.tobytes()
    assert cluster.store.stats.snapshot()["remote_hits"] == before["remote_hits"] + 1


@pytest.mark.parametrize("where", ["this_host", "other_host"])
def test_range_read(cluster, where):
    segment = cluster.store.segment if where == "this_host" else cluster.other.segment
    media_key, value = cluster.put(segment)
    row = 6144 * 2
    ranges = [(0, 64), (64 + 100 * row, 37 * row), (IMAGE_BYTES - row, row)]  # header, rows 100-136, the last row
    got = cluster.store.fetch_ranges(media_key, ranges)
    assert [bytes(v) for v in got] == [value[o : o + n].tobytes() for o, n in ranges]


def test_missing_key(cluster):
    before = cluster.store.stats.snapshot()
    with pytest.raises(Missing):
        cluster.store.fetch(b"\xee" * 32)
    with pytest.raises(Missing):
        cluster.store.fetch_ranges(b"\xee" * 32, [(0, 64)])
    with pytest.raises(Missing):
        cluster.store.size(b"\xee" * 32)
    after = cluster.store.stats.snapshot()
    assert after["misses"] == before["misses"] + 2 and after["errors"] == before["errors"]


def test_many_values_from_both_stores(cluster):
    """Values read in place and transferred stay intact while more are written and read."""
    written = [cluster.put(seg, 1 << 20) for seg in [cluster.store.segment, cluster.other.segment] * 4]
    views = [cluster.store.fetch(k) for k, _ in written]
    for view, (_, value) in zip(views, written):
        assert bytes(view) == value.tobytes()
