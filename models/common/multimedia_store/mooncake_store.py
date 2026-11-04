# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reads media embeddings from this host's Mooncake store, for the prefill runner.

``MooncakeEmbeddingReader`` is an ``EmbeddingReader`` (reader.py):

- ``fetch_batch(requests)`` reads one byte range from each of several values in one go: one batched lookup at the
  master for the whole batch, plus one batched transfer for whatever lives on other hosts (inside which Mooncake
  looks those keys up again). ``fetch(media_key)`` is a batch of one whole value; ``size(media_key)`` returns a
  value's length.
- A value in this host's segment comes back in place, read-only. A value in another host's segment is read into this
  reader's registered read buffer, which grows on demand, and comes back as a read-only view into it.
- A missing key raises ``Missing`` and any other store failure ``StoreUnavailable``; in a batch, the first failing
  request fails the whole batch. A malformed call (a key that is not ASCII, a range outside the value) raises
  ``ValueError``.
- ``stats`` counts hits, misses, errors, bytes read, fetch latency and the read buffer's size.

Media keys are ASCII strings, stored as they are. ``open_media_embedding_reader_from_env`` starts the store from the
``PREFILL_MEDIA_STORE_*`` variables that ``MediaEmbeddingStoreConfig.from_env`` lists.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from loguru import logger

from models.common.multimedia_store.reader import FetchRequest, Missing, StoreUnavailable

# Mooncake ErrorCode the reader tells apart from other failures (mooncake-store/include/types.h).
OBJECT_NOT_FOUND = -704

_ENV_PREFIX = "PREFILL_MEDIA_STORE_"
_DEFAULT_HTTP_METADATA_PORT = 8080  # mooncake_master's --http_metadata_server_port default
_REQUIREMENTS = "models/common/multimedia_store/requirements.txt"


def _check_key(media_key: str) -> str:
    # A str, since Mooncake returns lookups keyed by str: a bytes key would never match and always read as Missing.
    if not isinstance(media_key, str) or not media_key.isascii():
        raise ValueError(f"media key {media_key!r} is not an ASCII str")
    return media_key


def parse_bytes(text: str) -> int:
    """'50G', '50GiB', '512M', '1048576' -> bytes. Suffixes are binary (G = GiB)."""
    m = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)(?:i?B)?\s*", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a byte size: {text!r} (examples: 50G, 512M, 1048576)")
    return int(m.group(1)) << {"": 0, "K": 10, "M": 20, "G": 30, "T": 40}[m.group(2).upper()]


@dataclass(frozen=True)
class MediaEmbeddingStoreConfig:
    master: str
    """Mooncake master address, host:port."""
    segment: str
    """This host's segment name, 'host:port', which writers pass as preferred_segment. A bare 'host' gets a port
    Mooncake picks, so the full name is known only once the store starts (MooncakeEmbeddingReader.segment)."""
    segment_bytes: int
    """Host memory this host gives the store."""
    metadata: str
    """Mooncake metadata server."""
    protocol: str = "rdma"
    device: str = ""
    """RDMA device(s), e.g. 'rocep201s0f0'; empty lets Mooncake pick."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["MediaEmbeddingStoreConfig"]:
        """The store's config from PREFILL_MEDIA_STORE_*, or None when PREFILL_MEDIA_STORE_MASTER is unset.

        PREFILL_MEDIA_STORE_MASTER    master address, host:port (turns the store on)
        PREFILL_MEDIA_STORE_SEGMENT   this host's address, host:port or host (required)
        PREFILL_MEDIA_STORE_BYTES     segment size, e.g. 50G (required)
        PREFILL_MEDIA_STORE_METADATA  metadata server (default http://<master host>:8080/metadata)
        PREFILL_MEDIA_STORE_PROTOCOL  rdma (default) or tcp
        PREFILL_MEDIA_STORE_DEVICE    RDMA device(s) (default: Mooncake's choice)
        """

        def get(name: str, default: Optional[str] = None) -> Optional[str]:
            value = environ.get(_ENV_PREFIX + name, "").strip()
            return value or default

        master = get("MASTER")
        if master is None:
            return None
        if ":" not in master:
            raise ValueError(f"{_ENV_PREFIX}MASTER={master!r}: want host:port")
        missing = [_ENV_PREFIX + n for n in ("SEGMENT", "BYTES") if get(n) is None]
        if missing:
            raise ValueError(f"{_ENV_PREFIX}MASTER is set, so {' and '.join(missing)} must be set too")
        master_host = master.rsplit(":", 1)[0]
        return cls(
            master=master,
            segment=get("SEGMENT"),
            segment_bytes=parse_bytes(get("BYTES")),
            metadata=get("METADATA", f"http://{master_host}:{_DEFAULT_HTTP_METADATA_PORT}/metadata"),
            protocol=get("PROTOCOL", "rdma"),
            device=get("DEVICE", ""),
        )


class MediaEmbeddingReaderStats:
    """Reader counters. Hits (in this host's segment or another's) and bytes read per request; misses, errors and
    fetch latency per call. Also the read buffer's current size."""

    OUTCOMES = ("local_hits", "remote_hits", "misses", "errors")

    def __init__(self, log_interval_s: float = 60.0):
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(self.OUTCOMES, 0)
        self._bytes_read = 0
        self._calls = 0
        self._latency_ms_total = 0.0
        self._latency_ms_max = 0.0
        self.read_buffer_bytes = 0
        self._log_interval_s = log_interval_s
        self._last_log = time.monotonic()

    def record(self, counts: Mapping[str, int], nbytes: int, ms: float) -> None:
        with self._lock:
            for outcome, n in counts.items():
                self._counts[outcome] += n
            self._bytes_read += nbytes
            self._calls += 1
            self._latency_ms_total += ms
            self._latency_ms_max = max(self._latency_ms_max, ms)
            now = time.monotonic()
            due = now - self._last_log >= self._log_interval_s
            if due:
                self._last_log = now
        if due:
            logger.info(f"[media_embedding_reader] {self.snapshot()}")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "hits": self._counts["local_hits"] + self._counts["remote_hits"],
                **self._counts,
                "bytes_read": self._bytes_read,
                "fetch_calls": self._calls,
                "fetch_ms_mean": round(self._latency_ms_total / self._calls, 3) if self._calls else None,
                "fetch_ms_max": round(self._latency_ms_max, 3) if self._calls else None,
                "read_buffer_bytes": self.read_buffer_bytes,
            }


@dataclass(frozen=True)
class _Value:
    size: int
    local_address: Optional[int]
    """The value's address in this process when it is in this host's segment, else None."""


def _view_at(address: int, nbytes: int) -> memoryview:
    """A read-only memoryview of nbytes at address in this process, with no copy."""
    return memoryview((ctypes.c_uint8 * nbytes).from_address(address)).cast("B").toreadonly()


class MooncakeEmbeddingReader:
    """EmbeddingReader (reader.py) over this host's Mooncake store.

    ``open(config)`` starts the client and mounts the segment; tests may pass an already set up client to the
    constructor. Not tied to a model: values are opaque bytes."""

    def __init__(self, client, segment: str):
        self._client = client
        self.segment = segment
        """This host's segment name, the preferred_segment writers use."""
        self.stats = MediaEmbeddingReaderStats()
        self._read_lock = threading.Lock()
        self._read_buffer = np.empty(0, dtype=np.uint8)  # registered on the first read from another host
        self._closed = False

    @classmethod
    def open(cls, config: MediaEmbeddingStoreConfig) -> "MooncakeEmbeddingReader":
        """Start this host's store: join the master and mount config.segment_bytes of host memory as its segment."""
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise StoreUnavailable(
                f"{_ENV_PREFIX}MASTER is set but this Python env has no Mooncake client: "
                f"uv pip install -r {_REQUIREMENTS}"
            ) from e
        client = MooncakeDistributedStore()
        t0 = time.perf_counter()
        # local_buffer_size 0: reads from other hosts land in this reader's own registered read buffer.
        rc = client.setup(
            config.segment, config.metadata, config.segment_bytes, 0, config.protocol, config.device, config.master
        )
        if rc != 0:
            raise StoreUnavailable(
                f"Mooncake setup failed ({rc}): segment {config.segment}, master {config.master}, "
                f"metadata {config.metadata}, {config.segment_bytes} B over {config.protocol}. RDMA registration locks "
                "the whole segment, so it must fit under `ulimit -l`."
            )
        if client.health_check() != 0:
            client.close()
            raise StoreUnavailable(f"Mooncake master {config.master} is unreachable")
        segment = client.get_hostname()
        reader = cls(client, segment)
        atexit.register(reader.close)
        logger.info(
            f"[media_embedding_reader] segment={segment} master={config.master} bytes={config.segment_bytes} "
            f"protocol={config.protocol} ready in {time.perf_counter() - t0:.1f} s"
        )
        return reader

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info(f"[media_embedding_reader] closing segment {self.segment}: {self.stats.snapshot()}")
        self._client.close()

    def fetch(self, media_key: str) -> memoryview:
        """The whole value stored under media_key, as fetch_batch returns it. Raises Missing or StoreUnavailable."""
        return self.fetch_batch([FetchRequest(media_key)])[0]

    def size(self, media_key: str) -> int:
        """Bytes stored under media_key. Raises Missing or StoreUnavailable."""
        return self._locate([_check_key(media_key)])[media_key].size

    def fetch_batch(self, requests: Sequence[FetchRequest]) -> list[memoryview]:
        """The bytes of each (media_key, offset, length) request, in order. Raises Missing or StoreUnavailable for the
        first request that fails, failing the whole batch, and ValueError for a range outside its value.

        A value in this host's segment comes back in place. The lookup gives this client a lease on the object: for
        the master's --default_kv_lease_ttl (10 s by default) after the call, the master refuses to remove or evict
        it. After the lease it stays only while the writer's soft pin holds it (with allow_evict_soft_pinned_objects
        =false, the master does not evict it) and nothing removes it, so copy what you need before the lease ends.

        A value in another host's segment comes back as a view into this reader's read buffer, valid until the next
        fetch on this reader overwrites it. Callers copy the rows out (to device staging) before fetching again."""
        if not requests:
            return []
        t0 = time.perf_counter()
        requests = [FetchRequest(*r) for r in requests]
        try:
            located = self._locate(list(dict.fromkeys(_check_key(r.media_key) for r in requests)))
            results: list[Optional[memoryview]] = [None] * len(requests)
            remote = []  # (request index, key, offset, length)
            for i, r in enumerate(requests):
                value = located[r.media_key]
                length = value.size - r.offset if r.length == -1 else r.length
                if r.offset < 0 or length < 0 or r.offset + length > value.size:
                    raise ValueError(f"range ({r.offset}, {r.length}) outside the {value.size} B value {r.media_key!r}")
                if value.local_address is not None:
                    results[i] = _view_at(value.local_address + r.offset, length)
                else:
                    remote.append((i, r.media_key, r.offset, length))
            if remote:
                self._read_remote(remote, results)
        except (Missing, StoreUnavailable) as e:
            outcome = "misses" if isinstance(e, Missing) else "errors"
            self.stats.record({outcome: 1}, 0, (time.perf_counter() - t0) * 1e3)
            raise
        n_remote = len(remote)
        counts = {"local_hits": len(requests) - n_remote, "remote_hits": n_remote}
        self.stats.record(counts, sum(v.nbytes for v in results), (time.perf_counter() - t0) * 1e3)
        return results

    def _locate(self, keys: list[str]) -> dict[str, _Value]:
        """Where each value lives, from one batched lookup at the master. Raises Missing or StoreUnavailable for the
        first key that has no readable value."""
        if self._closed:
            raise StoreUnavailable(f"the store on segment {self.segment} is closed")
        # Mooncake reports failures in its results, not by raising: an unreachable master leaves keys out of the
        # lookup and gives transfers a negative code.
        found = self._client.batch_get_replica_desc(keys)
        out = {}
        for key in keys:
            replicas = found.get(key, [])
            if not replicas:
                # A key the lookup leaves out is missing, or the master is unreachable. health_check() reads the
                # client's own connection state, so telling them apart costs no second round trip.
                if self._client.health_check() == 0:
                    raise Missing(key)
                raise StoreUnavailable(f"{key}: Mooncake master is unreachable")
            if len(replicas) != 1:  # writers store one replica; replicated values are not supported yet
                raise StoreUnavailable(f"{key}: {len(replicas)} replicas, expected 1")
            (replica,) = replicas
            if not replica.is_memory_replica() or replica.status.name != "COMPLETE":
                raise Missing(f"{key}: not a complete memory replica ({replica.status.name})")
            d = replica.get_memory_descriptor().buffer_descriptor
            out[key] = _Value(d.size, d.buffer_address if d.transport_endpoint == self.segment else None)
        return out

    def _grow_read_buffer(self, nbytes: int) -> None:
        """Make the read buffer at least nbytes, doubling it; it never shrinks, so a long run registers a new buffer
        only a handful of times. Called with _read_lock held."""
        if nbytes <= self._read_buffer.nbytes:
            return
        size = max(nbytes, 2 * self._read_buffer.nbytes)
        if self._read_buffer.nbytes:  # views a caller still holds keep the old buffer alive; it is only unregistered
            self._client.unregister_buffer(self._read_buffer.ctypes.data)
            self._read_buffer = np.empty(0, dtype=np.uint8)
            self.stats.read_buffer_bytes = 0
        buffer = np.empty(size, dtype=np.uint8)
        rc = self._client.register_buffer(buffer.ctypes.data, size)
        if rc != 0:
            raise StoreUnavailable(f"registering a {size} B read buffer failed: {rc}")
        self._read_buffer = buffer
        self.stats.read_buffer_bytes = size
        logger.info(f"[media_embedding_reader] read buffer is now {size} B")

    def _read_remote(self, remote: list[tuple[int, str, int, int]], results: list) -> None:
        """Reads the remote requests' ranges in one batched transfer into the read buffer and stores a view of each in
        results. Mooncake rejects empty ranges, so those are not read and come back empty."""
        by_key: dict[str, list[tuple[int, int, int, int]]] = {}  # key -> [(request index, dst, offset, length)]
        dst = 0
        for i, key, offset, length in remote:
            if length == 0:
                results[i] = memoryview(b"")
                continue
            by_key.setdefault(key, []).append((i, dst, offset, length))
            dst += length
        if not by_key:
            return
        with self._read_lock:
            self._grow_read_buffer(dst)
            keys = list(by_key)
            got = self._client.get_into_ranges(
                [self._read_buffer.ctypes.data],
                [keys],
                [[[d for _, d, _, _ in by_key[k]] for k in keys]],
                [[[o for _, _, o, _ in by_key[k]] for k in keys]],
                [[[n for _, _, _, n in by_key[k]] for k in keys]],
            )[0]
            if len(got) != len(keys) or any(len(rcs) != len(by_key[k]) for k, rcs in zip(keys, got)):
                raise StoreUnavailable(f"Mooncake returned an incomplete result for {len(keys)} values")
            for key, rcs in zip(keys, got):
                for rc, (i, d, offset, length) in zip(rcs, by_key[key]):
                    if rc == OBJECT_NOT_FOUND:
                        raise Missing(key)
                    if rc != length:
                        raise StoreUnavailable(f"{key}: read {rc} of {length} B at {offset}")
                    results[i] = memoryview(self._read_buffer[d : d + length]).toreadonly()


def open_media_embedding_reader_from_env(
    environ: Mapping[str, str] = os.environ,
) -> Optional[MooncakeEmbeddingReader]:
    """This host's media embedding reader when PREFILL_MEDIA_STORE_MASTER is set, else None."""
    config = MediaEmbeddingStoreConfig.from_env(environ)
    return MooncakeEmbeddingReader.open(config) if config is not None else None
