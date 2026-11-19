# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runner's media embedding reader against a real Mooncake master on this host, over TCP.

Three clients join a local master started the way the encoder service starts its own (HTTP metadata server,
allow_evict_soft_pinned_objects=false):

- ``store``: this runner's store (MooncakeEmbeddingReader.open, segment 127.0.0.1:<port>);
- ``other``: another prefill host's store (segment 127.0.0.2:<port>, a second loopback address);
- ``puller``: a host in pull mode, with no store (segment 127.0.0.4:<port>);
- ``plain_reader``: a pull-mode host whose media keys are their own store keys (MEDIA_STORE_KEY_TRANSFORM=none);
- ``writer``: the encoder service stand-in, with no segment, writing with put_from, a soft pin and preferred_segment.

Runs on Linux, after ``uv pip install -r models/common/multimedia_store/requirements.txt`` (the CI job does this);
127.0.0.2 is a Linux loopback address. There a missing Mooncake client fails the test instead of skipping it, so an
env that cannot start the store shows up. Skipped on other platforms. Kept out of models/common/tests, whose
broad CI job does not install Mooncake.
"""

import dataclasses
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from models.common.multimedia_store.config import MediaEmbeddingStoreConfig
from models.common.multimedia_store.reader import FetchRequest, Missing

if not sys.platform.startswith("linux"):
    pytest.skip("Mooncake and the 127.0.0.2 loopback address are Linux only", allow_module_level=True)
try:
    import mooncake
    import mooncake.store as mooncake_store

    from models.common.multimedia_store.mooncake_store import MooncakeEmbeddingReader
except ImportError as e:  # a broken env, so it fails rather than skips
    pytest.fail(f"no Mooncake client ({e}): uv pip install -r models/common/multimedia_store/requirements.txt", False)

# One MiniMax-M3 image: a 64-byte header and 576 bf16 rows of width 6144.
IMAGE_BYTES = 64 + 576 * 6144 * 2


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _master_binary() -> str:
    """mooncake_master from PATH, the venv's bin, or the wheel's package directory."""
    candidates = [
        shutil.which("mooncake_master"),
        os.path.join(os.path.dirname(sys.executable), "mooncake_master"),
        os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master"),
    ]
    for path in candidates:
        if path and os.access(path, os.X_OK):
            return path
    pytest.fail(f"mooncake_master not found in {candidates}")


def _wait_until_healthy(master: subprocess.Popen, metrics_port: int, log) -> None:
    """Mooncake has no readiness callback; its master serves GET /health on the metrics port once it is up. Wait for
    that, failing at once if the master exits; the deadline only turns a master that never starts into a failure."""
    deadline = time.monotonic() + 60
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{metrics_port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if master.poll() is not None:
            pytest.fail(f"mooncake_master exited with {master.returncode}; see {log}")
        if time.monotonic() > deadline:
            pytest.fail(f"mooncake_master is not healthy after 60 s; see {log}")
        time.sleep(0.1)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    rpc, http, metrics = _free_port(), _free_port(), _free_port()
    log = tmp_path_factory.mktemp("mooncake") / "master.log"
    log_file = open(log, "w")
    master = subprocess.Popen(
        [
            _master_binary(),
            f"--rpc_port={rpc}",
            "--enable_http_metadata_server=true",
            f"--http_metadata_server_port={http}",
            "--allow_evict_soft_pinned_objects=false",
            f"--metrics_port={metrics}",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _wait_until_healthy(master, metrics, log)

    def config(host: str, segment_bytes: int) -> MediaEmbeddingStoreConfig:
        return MediaEmbeddingStoreConfig(
            master=f"127.0.0.1:{rpc}",
            segment=f"{host}:{_free_port(host)}",
            segment_bytes=segment_bytes,
            metadata=f"http://127.0.0.1:{http}/metadata",
            protocol="tcp",
        )

    store = MooncakeEmbeddingReader.open(config("127.0.0.1", 64 << 20))
    other = MooncakeEmbeddingReader.open(config("127.0.0.2", 64 << 20))
    puller = MooncakeEmbeddingReader.open(config("127.0.0.4", 0))  # pull mode: no store on this host
    plain_reader = MooncakeEmbeddingReader.open(dataclasses.replace(config("127.0.0.5", 0), key_transform="none"))
    w = config("127.0.0.3", 0)
    writer = mooncake_store.MooncakeDistributedStore()
    assert writer.setup(w.segment, w.metadata, 0, 16 << 20, w.protocol, "", w.master) == 0
    try:
        yield SimpleCluster(store, other, writer, metrics, puller, plain_reader)
    finally:
        writer.close()
        plain_reader.close()
        puller.close()
        other.close()
        store.close()
        master.terminate()
        master.wait(timeout=30)
        log_file.close()


class SimpleCluster:
    def __init__(self, store, other, writer, metrics_port, puller, plain_reader):
        self.store, self.other, self.writer, self.puller, self.plain_reader = store, other, writer, puller, plain_reader
        self.metrics_port = metrics_port
        self._buffers = []
        self._n = 0

    def put(
        self, segment: str, nbytes: int = IMAGE_BYTES, pin: bool = True, store_key: str = ""
    ) -> tuple[bytes, np.ndarray]:
        """The encoder service's write: put_from a registered buffer into segment, soft-pinned. Returns (media key,
        value): by default a fresh 32-byte media key, stored under its lowercase hex as the encoder does; with
        store_key, stored under that text, whose UTF-8 bytes are the media key."""
        self._n += 1
        media_key = store_key.encode() if store_key else hashlib.sha256(f"test-media-key-{self._n}".encode()).digest()
        store_key = store_key or media_key.hex()
        value = np.random.default_rng(self._n).integers(0, 256, nbytes, dtype=np.uint8)
        assert self.writer.register_buffer(value.ctypes.data, nbytes) == 0
        self._buffers.append(value)
        config = mooncake_store.ReplicateConfig()
        config.preferred_segment = segment
        if pin:
            config.soft_pin_action = mooncake_store.SoftPinAction.ENABLE
            config.soft_pin_ttl_ms = 60_000
        assert self.writer.put_from(store_key, value.ctypes.data, nbytes, config) == 0
        return media_key, value

    def _descriptor(self, media_key: bytes):
        (replica,) = self.writer.get_replica_desc(media_key.hex())
        return replica.get_memory_descriptor().buffer_descriptor

    def placed_on(self, media_key: bytes) -> str:
        """Where the value under media_key (stored under its hex) lives."""
        return self._descriptor(media_key).transport_endpoint

    def address(self, media_key: bytes) -> int:
        return self._descriptor(media_key).buffer_address

    def master_lookups(self) -> int:
        """Lookups the master has served, single and batched, from its own request counters."""
        with urllib.request.urlopen(f"http://127.0.0.1:{self.metrics_port}/metrics", timeout=5) as r:
            text = r.read().decode()
        names = ("master_get_replica_list_requests_total", "master_batch_get_replica_list_requests_total")
        return sum(
            int(float(m.group(2))) for m in re.finditer(r"^(\w+) ([0-9.e+]+)$", text, re.M) if m.group(1) in names
        )


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
    after = cluster.store.stats.snapshot()
    assert after["remote_hits"] == before["remote_hits"] + 1
    assert after["read_buffer_bytes"] >= IMAGE_BYTES  # grew from empty to hold the value


@pytest.mark.parametrize("where", ["this_host", "other_host"])
def test_range_read(cluster, where):
    segment = cluster.store.segment if where == "this_host" else cluster.other.segment
    media_key, value = cluster.put(segment)
    row = 6144 * 2
    # The header, rows 100-136, an empty range, and the last two rows given as "to the end".
    ranges = [(0, 64), (64 + 100 * row, 37 * row), (64, 0), (IMAGE_BYTES - 2 * row, -1)]
    want = [value[:64], value[64 + 100 * row : 64 + 137 * row], value[64:64], value[IMAGE_BYTES - 2 * row :]]
    got = cluster.store.fetch_batch([FetchRequest(media_key, o, n) for o, n in ranges])
    assert [bytes(v) for v in got] == [w.tobytes() for w in want]


def test_missing_key(cluster, expect_error):
    absent = hashlib.sha256(b"no-such-media-key").digest()
    before = cluster.store.stats.snapshot()
    with expect_error(Missing, absent.hex()):
        cluster.store.fetch(absent)
    with expect_error(Missing, absent.hex()):
        cluster.store.fetch_batch([FetchRequest(absent, 0, 64)])
    with expect_error(Missing, absent.hex()):
        cluster.store.size(absent)
    after = cluster.store.stats.snapshot()
    assert after["misses"] == before["misses"] + 2 and after["errors"] == before["errors"]


def test_batch_of_several_requests(cluster, expect_error):
    """A forward's batch: ranges of values on this host and another in one lookup at the master and one batched read
    from the other host. One missing key fails the whole batch."""
    here = [cluster.put(cluster.store.segment) for _ in range(3)]
    there = [cluster.put(cluster.other.segment) for _ in range(2)]
    requests = [FetchRequest(k, 64 + 1000 * i, 4096) for i, (k, _) in enumerate(here + there)]
    lookups = cluster.master_lookups()
    got = cluster.store.fetch_batch(requests)
    # One batched lookup for all five keys, then one for the snapshot the other host's keys are read from; no lookup
    # per key.
    assert cluster.master_lookups() - lookups == 1 + 1
    for i, ((_, value), result) in enumerate(zip(here + there, got)):
        assert bytes(result) == value[64 + 1000 * i : 64 + 1000 * i + 4096].tobytes()
    lookups = cluster.master_lookups()
    cluster.store.fetch_batch([FetchRequest(k) for k, _ in here])  # all on this host: in place
    assert cluster.master_lookups() - lookups == 1
    absent = hashlib.sha256(b"absent-media-key").digest()
    with expect_error(Missing, absent.hex()):
        cluster.store.fetch_batch(requests + [FetchRequest(absent)])


def test_many_values_from_both_stores(cluster):
    """Values read in place stay intact while more are written and read. A value from the other host is read before
    the next fetch, which reuses the read buffer."""
    written = [cluster.put(seg, 1 << 20) for seg in [cluster.store.segment, cluster.other.segment] * 4]
    local_views = []
    for (key, value), seg in zip(written, [cluster.store.segment, cluster.other.segment] * 4):
        view = cluster.store.fetch(key)
        assert bytes(view) == value.tobytes()
        if seg == cluster.store.segment:
            local_views.append((view, value))
    for view, value in local_views:
        assert bytes(view) == value.tobytes()


def test_value_read_in_place_is_leased(cluster):
    """fetch's lookup leases the object, so the master refuses to remove it while the caller reads it in place."""
    unread, _ = cluster.put(cluster.store.segment, 1 << 20, pin=False)
    read, value = cluster.put(cluster.store.segment, 1 << 20, pin=False)
    view = cluster.store.fetch(read)
    assert cluster.writer.remove(read.hex()) != 0
    assert bytes(view) == value.tobytes()
    assert cluster.writer.remove(unread.hex()) == 0  # nothing looked it up, so nothing holds it


def test_pull_mode_reads_everything_from_the_host_that_holds_it(cluster):
    """A host with no store (PREFILL_MEDIA_STORE_BYTES=0) reads each value, and its ranges, from the host holding it,
    here standing in for the encoder's store."""
    media_key, value = cluster.put(cluster.other.segment)
    before = cluster.puller.stats.snapshot()
    assert bytes(cluster.puller.fetch(media_key)) == value.tobytes()
    (rows,) = cluster.puller.fetch_batch([FetchRequest(media_key, 64, 4096)])
    assert bytes(rows) == value[64 : 64 + 4096].tobytes()
    after = cluster.puller.stats.snapshot()
    assert after["remote_hits"] == before["remote_hits"] + 2 and after["local_hits"] == before["local_hits"]


def test_untransformed_non_ascii_media_key(cluster):
    """With MEDIA_STORE_KEY_TRANSFORM=none a media key is its own store key, and only has to be valid UTF-8, which
    Mooncake's batched lookup decodes."""
    media_key, value = cluster.put(cluster.other.segment, store_key="test-media-é-中")
    assert media_key == "test-media-é-中".encode()
    assert bytes(cluster.plain_reader.fetch(media_key)) == value.tobytes()
