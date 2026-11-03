# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runner's embedding store: a Mooncake store on this host, joined to the encoder service's master.

The encoder service keeps one Mooncake master for media embeddings, separate from the KV Manager's. Each prefill
runner holds part of that store: on start it mounts a segment of this host's memory there. For each image the encoder
writes one value with ``put_from``, naming this runner's segment as ``preferred_segment``, under the image's media key
(32 bytes, stored as the 64-character lowercase hex string, see ``store_key``).

Reads (``MooncakeEmbeddingStore``, an ``EmbeddingReader``):

- A value in this host's segment is read in place: the returned memoryview points into the segment, with no copy and no
  transfer. It stays valid while the encoder's soft pin holds the object (the master runs with
  ``allow_evict_soft_pinned_objects=false``) and at least for the master's KV lease after the read.
- A value in another host's segment (Mooncake places it elsewhere when this segment is full or the preferred segment
  name does not match) is transferred: whole values into a buffer from this client's local buffer, owned by the
  returned memoryview; byte ranges into a registered scratch buffer, then copied out.
- A missing key raises ``Missing``; any other store failure raises ``StoreUnavailable``, so the runner can fail just
  that request. Only a malformed call (a key of the wrong length, a range outside the value) raises ValueError.

Every read updates ``stats`` (hits in this host's segment or another's, misses, errors, bytes read, fetch latency),
logged every minute while reads happen and when the store closes.

Configuration comes from ``PREFILL_MEDIA_STORE_*`` environment variables (``MediaStoreConfig.from_env``); the store is
on only when ``PREFILL_MEDIA_STORE_MASTER`` is set. Needs the Mooncake wheel
(``pip install mooncake-transfer-engine-non-cuda==0.3.13.post1``), imported only when the store is opened.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, NoReturn, Optional, Sequence

import numpy as np
from loguru import logger

from models.common.media_store.reader import Missing, StoreUnavailable

MEDIA_KEY_BYTES = 32

# Mooncake ErrorCode values the reader tells apart (mooncake-store/include/types.h).
OBJECT_NOT_FOUND = -704

_ENV_PREFIX = "PREFILL_MEDIA_STORE_"
_DEFAULT_HTTP_METADATA_PORT = 8080  # mooncake_master's --http_metadata_server_port default
_DEFAULT_READ_BUFFER_BYTES = 64 << 20  # holds a few whole MiniMax-M3 images (~7.1 MB each) read from other hosts
_LATENCY_WINDOW = 4096  # fetch latencies kept for the percentiles in stats()


def store_key(media_key: bytes) -> str:
    """The Mooncake key a media key is stored under. The encoder service must use the same."""
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
    """The encoder service's Mooncake master, host:port."""
    segment: str
    """This host's segment name on the RDMA network: 'host:port' names it exactly that, which is what the encoder must
    pass as preferred_segment. A bare 'host' lets Mooncake pick the port; the full name is then only known at start
    (MooncakeEmbeddingStore.segment, also logged), since a bare host does not match preferred_segment."""
    segment_bytes: int
    """Host memory this runner gives the store."""
    metadata: str
    """Mooncake metadata server: the master's HTTP metadata server unless set."""
    protocol: str = "rdma"
    device: str = ""
    """RDMA device(s), e.g. 'rocep201s0f0'; empty lets Mooncake pick."""
    read_buffer_bytes: int = _DEFAULT_READ_BUFFER_BYTES
    """Client buffer for values and ranges read from other hosts; also the largest single remote read."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["MediaStoreConfig"]:
        """The store's config from PREFILL_MEDIA_STORE_*, or None when PREFILL_MEDIA_STORE_MASTER is unset.

        PREFILL_MEDIA_STORE_MASTER       master address, host:port (turns the store on)
        PREFILL_MEDIA_STORE_SEGMENT      this host's address, host:port or host (required)
        PREFILL_MEDIA_STORE_BYTES        segment size, e.g. 50G (required)
        PREFILL_MEDIA_STORE_METADATA     metadata server (default http://<master host>:8080/metadata)
        PREFILL_MEDIA_STORE_PROTOCOL     rdma (default) or tcp
        PREFILL_MEDIA_STORE_DEVICE       RDMA device(s) (default: Mooncake's choice)
        PREFILL_MEDIA_STORE_READ_BUFFER  client buffer for remote reads (default 64M)
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

    def __init__(self, log_interval_s: float = 60.0):
        self._lock = threading.Lock()
        self._log_interval_s = log_interval_s
        self._last_log = time.monotonic()
        self.local_hits = 0
        self.remote_hits = 0
        self.misses = 0
        self.errors = 0
        self.bytes_read = 0
        self._latency_ms = deque(maxlen=_LATENCY_WINDOW)
        self._latency_ms_total = 0.0
        self._latency_ms_max = 0.0

    def record(self, outcome: str, nbytes: int, ms: float) -> None:
        with self._lock:
            setattr(self, outcome, getattr(self, outcome) + 1)
            self.bytes_read += nbytes
            self._latency_ms.append(ms)
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
            fetches = self.local_hits + self.remote_hits + self.misses + self.errors
            recent = sorted(self._latency_ms)

            def pct(q: float) -> Optional[float]:
                return round(recent[min(len(recent) - 1, int(q * len(recent)))], 3) if recent else None

            return {
                "hits": self.local_hits + self.remote_hits,
                "local_hits": self.local_hits,
                "remote_hits": self.remote_hits,
                "misses": self.misses,
                "errors": self.errors,
                "bytes_read": self.bytes_read,
                "fetches": fetches,
                "fetch_ms_mean": round(self._latency_ms_total / fetches, 3) if fetches else None,
                "fetch_ms_p50": pct(0.5),
                "fetch_ms_p99": pct(0.99),
                "fetch_ms_max": round(self._latency_ms_max, 3) if fetches else None,
            }


class _Value:
    """Where a stored value lives: its address in this process when it is in this host's segment, else None."""

    __slots__ = ("size", "local_address")

    def __init__(self, size: int, local_address: Optional[int]):
        self.size = size
        self.local_address = local_address


def _view_at(address: int, nbytes: int) -> memoryview:
    """A writable memoryview of nbytes at address in this process, with no copy. The memory must outlive it."""
    return memoryview((ctypes.c_uint8 * nbytes).from_address(address)).cast("B")


class MooncakeEmbeddingStore:
    """EmbeddingReader (models/common/media_store/reader.py) over this host's Mooncake store.

    ``open(config)`` starts the client and mounts the segment; tests may pass an already set up client to the
    constructor. Not tied to a model: values are opaque bytes (MiniMax-M3's layout is vision/format.py)."""

    def __init__(self, client, segment: str, read_buffer_bytes: int = _DEFAULT_READ_BUFFER_BYTES):
        self._client = client
        self.segment = segment
        """This runner's segment name, the preferred_segment the encoder must write to."""
        self.stats = MediaStoreStats()
        self._scratch = np.empty(read_buffer_bytes, dtype=np.uint8)
        self._scratch_lock = threading.Lock()
        rc = client.register_buffer(self._scratch.ctypes.data, self._scratch.nbytes)
        if rc != 0:
            raise StoreUnavailable(f"registering the {read_buffer_bytes} B range-read buffer failed: {rc}")
        self._closed = False

    @classmethod
    def open(cls, config: MediaStoreConfig) -> "MooncakeEmbeddingStore":
        """Start this host's store: join the master and mount config.segment_bytes of host memory as its segment."""
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise StoreUnavailable(
                f"{_ENV_PREFIX}MASTER is set but the Mooncake wheel is missing: "
                "pip install mooncake-transfer-engine-non-cuda==0.3.13.post1"
            ) from e
        client = MooncakeDistributedStore()
        t0 = time.perf_counter()
        rc = client.setup(
            config.segment,
            config.metadata,
            config.segment_bytes,
            config.read_buffer_bytes,
            config.protocol,
            config.device,
            config.master,
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

    # -- EmbeddingReader --------------------------------------------------------------------------------------------

    def fetch(self, media_key: bytes) -> memoryview:
        """The whole value stored under media_key: in place when it is in this host's segment, else read into this
        client's buffer (the memoryview owns it). Raises Missing or StoreUnavailable."""
        t0 = time.perf_counter()
        key = store_key(media_key)
        try:
            value = self._locate(key)
            if value.local_address is not None:
                out = _view_at(value.local_address, value.size)
            else:
                out = self._read_remote(key, value.size)
        except Missing:
            self.stats.record("misses", 0, (time.perf_counter() - t0) * 1e3)
            raise
        except StoreUnavailable:
            self.stats.record("errors", 0, (time.perf_counter() - t0) * 1e3)
            raise
        hit = "local_hits" if value.local_address is not None else "remote_hits"
        self.stats.record(hit, out.nbytes, (time.perf_counter() - t0) * 1e3)
        return out

    # -- beyond the protocol ----------------------------------------------------------------------------------------

    def size(self, media_key: bytes) -> int:
        """Bytes stored under media_key. Raises Missing or StoreUnavailable."""
        return self._locate(store_key(media_key)).size

    def fetch_ranges(self, media_key: bytes, ranges: Sequence[tuple[int, int]]) -> list[memoryview]:
        """Only the (offset, length) byte ranges of the value under media_key, one memoryview each: in place when the
        value is in this host's segment, else just those bytes transferred and copied out. Raises Missing,
        StoreUnavailable, or ValueError for a range outside the value."""
        t0 = time.perf_counter()
        key = store_key(media_key)
        try:
            value = self._locate(key)
            for offset, length in ranges:
                if offset < 0 or length < 0 or offset + length > value.size:
                    raise ValueError(f"range ({offset}, {length}) outside the {value.size} B value under {key}")
            if value.local_address is not None:
                out = [_view_at(value.local_address + offset, length) for offset, length in ranges]
            else:
                out = self._read_remote_ranges(key, ranges)
        except Missing:
            self.stats.record("misses", 0, (time.perf_counter() - t0) * 1e3)
            raise
        except StoreUnavailable:
            self.stats.record("errors", 0, (time.perf_counter() - t0) * 1e3)
            raise
        hit = "local_hits" if value.local_address is not None else "remote_hits"
        self.stats.record(hit, sum(v.nbytes for v in out), (time.perf_counter() - t0) * 1e3)
        return out

    # -- internals --------------------------------------------------------------------------------------------------

    def _locate(self, key: str) -> _Value:
        try:
            replicas = self._client.get_replica_desc(key)
        except Exception as e:  # the binding raises on RPC failures in some versions
            raise StoreUnavailable(f"looking up {key}: {e}") from e
        complete = [r for r in replicas if r.is_memory_replica() and r.status.name == "COMPLETE"]
        if not complete:
            if replicas:  # written but not finished (or being removed): the encoder replies only after put completes
                raise Missing(f"{key}: no complete replica")
            self._raise_missing_or_unavailable(key, self._get_size(key))
        descs = [r.get_memory_descriptor().buffer_descriptor for r in complete]
        for d in descs:
            if d.transport_endpoint == self.segment:
                return _Value(d.size, d.buffer_address)
        return _Value(descs[0].size, None)

    def _get_size(self, key: str) -> int:
        try:
            return self._client.get_size(key)
        except Exception as e:
            raise StoreUnavailable(f"looking up the size of {key}: {e}") from e

    @staticmethod
    def _raise_missing_or_unavailable(key: str, rc: int) -> NoReturn:
        if rc == OBJECT_NOT_FOUND:
            raise Missing(key)
        if rc < 0:
            raise StoreUnavailable(f"{key}: Mooncake error {rc}")
        raise Missing(f"{key}: listed with {rc} B but no memory replica")

    def _read_remote(self, key: str, size: int) -> memoryview:
        handle = self._client.get_buffer(key)
        if handle is None:  # removed since _locate, or the value does not fit the client buffer
            rc = self._get_size(key)
            if rc == OBJECT_NOT_FOUND:
                raise Missing(key)
            raise StoreUnavailable(
                f"{key}: could not read the {size} B value into the client buffer ({_ENV_PREFIX}READ_BUFFER); "
                f"Mooncake returned {rc}"
            )
        if handle.size() != size:
            raise StoreUnavailable(f"{key}: read {handle.size()} B of {size}")
        return memoryview(handle)  # the BufferHandle frees its client-buffer slot when the last view goes

    def _read_remote_ranges(self, key: str, ranges: Sequence[tuple[int, int]]) -> list[memoryview]:
        if not ranges:
            return []
        total = sum(length for _, length in ranges)
        if total > self._scratch.nbytes:
            raise StoreUnavailable(
                f"{key}: {total} B of ranges exceed the {self._scratch.nbytes} B read buffer "
                f"({_ENV_PREFIX}READ_BUFFER)"
            )
        dst_offsets = np.cumsum([0] + [length for _, length in ranges[:-1]]).tolist()
        with self._scratch_lock:
            got = self._client.get_into_ranges(
                [self._scratch.ctypes.data],
                [[key]],
                [[dst_offsets]],
                [[[offset for offset, _ in ranges]]],
                [[[length for _, length in ranges]]],
            )[0][0]
            for rc, (offset, length) in zip(got, ranges):
                if rc < 0:
                    self._raise_missing_or_unavailable(key, rc)
                if rc != length:
                    raise StoreUnavailable(f"{key}: read {rc} B of the {length} B range at {offset}")
            return [memoryview(self._scratch[o : o + n].copy()) for o, (_, n) in zip(dst_offsets, ranges)]


def open_media_store_from_env(environ: Mapping[str, str] = os.environ) -> Optional[MooncakeEmbeddingStore]:
    """This host's embedding store when PREFILL_MEDIA_STORE_MASTER is set, else None."""
    config = MediaStoreConfig.from_env(environ)
    return MooncakeEmbeddingStore.open(config) if config is not None else None
