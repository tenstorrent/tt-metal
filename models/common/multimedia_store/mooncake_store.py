# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reads media embeddings from this host's Mooncake store, for the prefill runner.

``MooncakeEmbeddingStore`` is an ``EmbeddingReader`` (reader.py):

- ``fetch(media_key)`` returns the whole stored value, ``fetch_ranges(media_key, ranges)`` only the given byte ranges,
  and ``size(media_key)`` the value's length.
- A value in this host's segment comes back in place, as a view into the segment. A value in another host's segment is
  read into this client's read buffer and copied out.
- A missing key raises ``Missing`` and any other store failure ``StoreUnavailable``. A malformed call (a key that is
  not 32 bytes, a range outside the value) raises ``ValueError``.
- ``stats`` counts hits, misses, errors, bytes read and fetch latency.

Values are stored under ``store_key(media_key)``, which writers must use too. ``open_media_store_from_env`` starts the
store from the ``PREFILL_MEDIA_STORE_*`` variables that ``MediaStoreConfig.from_env`` lists.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Mapping, NoReturn, Optional, Sequence

import numpy as np
from loguru import logger

from models.common.multimedia_store.reader import Missing, StoreUnavailable

MEDIA_KEY_BYTES = 32

# Mooncake ErrorCode the reader tells apart from other failures (mooncake-store/include/types.h).
OBJECT_NOT_FOUND = -704

_ENV_PREFIX = "PREFILL_MEDIA_STORE_"
_DEFAULT_HTTP_METADATA_PORT = 8080  # mooncake_master's --http_metadata_server_port default
# The read buffer holds what one call reads from another host's segment. Size it as the largest embedding (a
# MiniMax-M3 image is ~7.1 MB) times the images read per call (the batch size); 64 MB is a sane default.
_DEFAULT_READ_BUFFER_BYTES = 64 << 20


def store_key(media_key: bytes) -> str:
    """The Mooncake key a media key is stored under: its lowercase hex string."""
    if len(media_key) != MEDIA_KEY_BYTES:
        raise ValueError(f"media key of {len(media_key)} B, want {MEDIA_KEY_BYTES}")
    return media_key.hex()


def parse_bytes(text: str) -> int:
    """'50G', '50GiB', '512M', '1048576' -> bytes. Suffixes are binary (G = GiB)."""
    m = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)(?:i?B)?\s*", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a byte size: {text!r} (examples: 50G, 512M, 1048576)")
    return int(m.group(1)) << {"": 0, "K": 10, "M": 20, "G": 30, "T": 40}[m.group(2).upper()]


@dataclass(frozen=True)
class MediaStoreConfig:
    master: str
    """Mooncake master address, host:port."""
    segment: str
    """This host's segment name, 'host:port', which writers pass as preferred_segment. A bare 'host' gets a port
    Mooncake picks, so the full name is known only once the store starts (MooncakeEmbeddingStore.segment)."""
    segment_bytes: int
    """Host memory this host gives the store."""
    metadata: str
    """Mooncake metadata server."""
    protocol: str = "rdma"
    device: str = ""
    """RDMA device(s), e.g. 'rocep201s0f0'; empty lets Mooncake pick."""
    read_buffer_bytes: int = _DEFAULT_READ_BUFFER_BYTES
    """Largest read from another host's segment in one call."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["MediaStoreConfig"]:
        """The store's config from PREFILL_MEDIA_STORE_*, or None when PREFILL_MEDIA_STORE_MASTER is unset.

        PREFILL_MEDIA_STORE_MASTER       master address, host:port (turns the store on)
        PREFILL_MEDIA_STORE_SEGMENT      this host's address, host:port or host (required)
        PREFILL_MEDIA_STORE_BYTES        segment size, e.g. 50G (required)
        PREFILL_MEDIA_STORE_METADATA     metadata server (default http://<master host>:8080/metadata)
        PREFILL_MEDIA_STORE_PROTOCOL     rdma (default) or tcp
        PREFILL_MEDIA_STORE_DEVICE       RDMA device(s) (default: Mooncake's choice)
        PREFILL_MEDIA_STORE_READ_BUFFER  largest read from another host in one call (default 64M)
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
            read_buffer_bytes=parse_bytes(get("READ_BUFFER", str(_DEFAULT_READ_BUFFER_BYTES))),
        )


class MediaStoreStats:
    """Client counters: hits (in this host's segment or another's), misses, errors, bytes read and fetch latency."""

    OUTCOMES = ("local_hits", "remote_hits", "misses", "errors")

    def __init__(self, log_interval_s: float = 60.0):
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(self.OUTCOMES, 0)
        self._bytes_read = 0
        self._latency_ms_total = 0.0
        self._latency_ms_max = 0.0
        self._log_interval_s = log_interval_s
        self._last_log = time.monotonic()

    def record(self, outcome: str, nbytes: int, ms: float) -> None:
        with self._lock:
            self._counts[outcome] += 1
            self._bytes_read += nbytes
            self._latency_ms_total += ms
            self._latency_ms_max = max(self._latency_ms_max, ms)
            now = time.monotonic()
            due = now - self._last_log >= self._log_interval_s
            if due:
                self._last_log = now
        if due:
            logger.info(f"[media_store] {self.snapshot()}")

    def snapshot(self) -> dict:
        with self._lock:
            fetches = sum(self._counts.values())
            return {
                "hits": self._counts["local_hits"] + self._counts["remote_hits"],
                **self._counts,
                "bytes_read": self._bytes_read,
                "fetches": fetches,
                "fetch_ms_mean": round(self._latency_ms_total / fetches, 3) if fetches else None,
                "fetch_ms_max": round(self._latency_ms_max, 3) if fetches else None,
            }


@dataclass(frozen=True)
class _Value:
    size: int
    local_address: Optional[int]
    """The value's address in this process when it is in this host's segment, else None."""


def _view_at(address: int, nbytes: int) -> memoryview:
    """A read-only memoryview of nbytes at address in this process, with no copy."""
    return memoryview((ctypes.c_uint8 * nbytes).from_address(address)).cast("B").toreadonly()


class MooncakeEmbeddingStore:
    """EmbeddingReader (reader.py) over this host's Mooncake store.

    ``open(config)`` starts the client and mounts the segment; tests may pass an already set up client to the
    constructor. Not tied to a model: values are opaque bytes."""

    def __init__(self, client, segment: str, read_buffer_bytes: int = _DEFAULT_READ_BUFFER_BYTES):
        self._client = client
        self.segment = segment
        """This host's segment name, the preferred_segment writers use."""
        self.stats = MediaStoreStats()
        self._read_buffer = np.empty(read_buffer_bytes, dtype=np.uint8)
        self._read_lock = threading.Lock()
        rc = client.register_buffer(self._read_buffer.ctypes.data, self._read_buffer.nbytes)
        if rc != 0:
            raise StoreUnavailable(f"registering the {read_buffer_bytes} B read buffer failed: {rc}")
        self._closed = False

    @classmethod
    def open(cls, config: MediaStoreConfig) -> "MooncakeEmbeddingStore":
        """Start this host's store: join the master and mount config.segment_bytes of host memory as its segment."""
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise StoreUnavailable(
                f"{_ENV_PREFIX}MASTER is set but this Python env has no Mooncake client "
                "(mooncake-transfer-engine-non-cuda, pinned in tt_metal/python_env/requirements-dev.txt): "
                "recreate python_env with ./create_venv.sh"
            ) from e
        client = MooncakeDistributedStore()
        t0 = time.perf_counter()
        # local_buffer_size 0: reads from other hosts land in this store's own registered read buffer.
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
        store = cls(client, segment, config.read_buffer_bytes)
        atexit.register(store.close)
        logger.info(
            f"[media_store] segment={segment} master={config.master} bytes={config.segment_bytes} "
            f"protocol={config.protocol} ready in {time.perf_counter() - t0:.1f} s"
        )
        return store

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.info(f"[media_store] closing segment {self.segment}: {self.stats.snapshot()}")
        self._client.close()

    def fetch(self, media_key: bytes) -> memoryview:
        """The whole value stored under media_key, as fetch_ranges returns a range. Raises Missing or
        StoreUnavailable."""
        return self.fetch_ranges(media_key, [(0, -1)])[0]

    def size(self, media_key: bytes) -> int:
        """Bytes stored under media_key. Raises Missing or StoreUnavailable."""
        if self._closed:
            raise StoreUnavailable(f"the store on segment {self.segment} is closed")
        return self._locate(store_key(media_key)).size

    def fetch_ranges(self, media_key: bytes, ranges: Sequence[tuple[int, int]]) -> list[memoryview]:
        """The (offset, length) byte ranges of the value under media_key, one memoryview each; length -1 runs to the
        end of the value. Raises Missing, StoreUnavailable, or ValueError for a range outside the value.

        A value in this host's segment comes back in place, read-only. The lookup gives this client a lease on the object: for
        the master's --default_kv_lease_ttl (10 s by default) after the call, the master refuses to remove or evict
        it. After the lease it stays only while the writer's soft pin holds it (with allow_evict_soft_pinned_objects
        =false, the master does not evict it) and nothing removes it, so copy what you need before the lease ends. A
        value in another host's segment comes back as a copy that the caller owns."""
        if self._closed:
            raise StoreUnavailable(f"the store on segment {self.segment} is closed")
        t0 = time.perf_counter()
        key = store_key(media_key)
        try:
            value = self._locate(key)
            spans = []
            for offset, length in ranges:
                if length == -1:
                    length = value.size - offset
                if offset < 0 or length < 0 or offset + length > value.size:
                    raise ValueError(f"range ({offset}, {length}) outside the {value.size} B value under {key}")
                spans.append((offset, length))
            if value.local_address is not None:
                out = [_view_at(value.local_address + offset, length) for offset, length in spans]
            else:
                out = self._read_remote(key, value.size, spans)
        except Missing:
            self.stats.record("misses", 0, (time.perf_counter() - t0) * 1e3)
            raise
        except StoreUnavailable:
            self.stats.record("errors", 0, (time.perf_counter() - t0) * 1e3)
            raise
        hit = "local_hits" if value.local_address is not None else "remote_hits"
        self.stats.record(hit, sum(v.nbytes for v in out), (time.perf_counter() - t0) * 1e3)
        return out

    def _locate(self, key: str) -> _Value:
        """Where the value under key lives: one lookup at the master."""
        try:
            replicas = self._client.get_replica_desc(key)
        except Exception as e:  # the binding raises on RPC failures in some versions
            raise StoreUnavailable(f"looking up {key}: {e}") from e
        complete = [r for r in replicas if r.is_memory_replica() and r.status.name == "COMPLETE"]
        if not complete:
            if replicas:  # written but not finished, or being removed
                raise Missing(f"{key}: no complete replica")
            # An empty lookup is a missing key or an unreachable master. health_check() reads the client's own
            # connection state, so telling them apart costs no second round trip.
            if self._client.health_check() == 0:
                raise Missing(key)
            raise StoreUnavailable(f"{key}: Mooncake master is unreachable")
        descs = [r.get_memory_descriptor().buffer_descriptor for r in complete]
        for d in descs:
            if d.transport_endpoint == self.segment:
                return _Value(d.size, d.buffer_address)
        return _Value(descs[0].size, None)

    @staticmethod
    def _raise_for(key: str, rc: int, want: int) -> NoReturn:
        if rc == OBJECT_NOT_FOUND:
            raise Missing(key)
        raise StoreUnavailable(f"{key}: read {rc} of {want} B")

    def _read_remote(self, key: str, size: int, spans: list[tuple[int, int]]) -> list[memoryview]:
        """The spans of a value in another host's segment, copied out of the read buffer. A whole-value read costs
        one more lookup at the master; a partial one costs two, but moves only the requested bytes."""
        total = sum(length for _, length in spans)
        if total > self._read_buffer.nbytes:
            raise StoreUnavailable(
                f"{key}: reading {total} B exceeds the {self._read_buffer.nbytes} B read buffer "
                f"({_ENV_PREFIX}READ_BUFFER)"
            )
        dst = np.cumsum([0] + [length for _, length in spans[:-1]]).tolist()
        # Mooncake rejects empty ranges, so only the non-empty ones are read; empty ones come back empty.
        reads = [(d, o, n) for d, (o, n) in zip(dst, spans) if n]
        with self._read_lock:
            ptr = self._read_buffer.ctypes.data
            if spans == [(0, size)]:
                rc = self._client.get_into(key, ptr, size)
                if rc != size:
                    self._raise_for(key, rc, size)
            elif reads:
                got = self._client.get_into_ranges(
                    [ptr],
                    [[key]],
                    [[[d for d, _, _ in reads]]],
                    [[[o for _, o, _ in reads]]],
                    [[[n for _, _, n in reads]]],
                )[0][0]
                for rc, (_, _, length) in zip(got, reads):
                    if rc != length:
                        self._raise_for(key, rc, length)
            return [memoryview(self._read_buffer[d : d + n].copy()) for d, (_, n) in zip(dst, spans)]


def open_media_store_from_env(environ: Mapping[str, str] = os.environ) -> Optional[MooncakeEmbeddingStore]:
    """This host's embedding store when PREFILL_MEDIA_STORE_MASTER is set, else None."""
    config = MediaStoreConfig.from_env(environ)
    return MooncakeEmbeddingStore.open(config) if config is not None else None
