# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental hybrid-prefix transfer contract, independent of TT/vLLM imports.

This is NOT connected to the serving capability flag. The scheduler remains
responsible for prefix matching, KV block ownership and request admission.
Device adapters must implement the leases below before this can serve a hit.
The codec preserves opaque TT bytes; it never dequantizes/requantizes BFP8.
"""

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import BinaryIO, ContextManager, Protocol

MAGIC = b"QWEN38-HYBRID-PREFIX\x01"
COMPLETE = b"COMPLETE\x01"
MAX_CHUNK_BYTES = 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def positive(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class Identity:
    model_revision: str
    implementation_revision: str
    execution_fingerprint: str
    namespace: str  # Tenant/cache salt; never omit in a multi-tenant deployment.

    def __post_init__(self):
        for value in asdict(self).values():
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError("Cache identity requires bounded nonempty strings")


@dataclass(frozen=True)
class Layout:
    layer_types: tuple[str, ...]
    tp_size: int
    page_tokens: int
    kv_page_bytes: int
    recurrent_bytes: int
    conv_bytes: int
    abi: str

    def __post_init__(self):
        if (
            type(self.layer_types) is not tuple
            or not self.layer_types
            or len(self.layer_types) > 1024
            or set(self.layer_types) != {"full_attention", "linear_attention"}
            or not all(
                positive(v)
                for v in (self.tp_size, self.page_tokens, self.kv_page_bytes, self.recurrent_bytes, self.conv_bytes)
            )
            or not isinstance(self.abi, str)
            or not self.abi
            or len(self.abi) > 1024
        ):
            raise ValueError("Require an explicit hybrid layout and positive byte geometry")

    @classmethod
    def qwen_tp4_bfp8(cls, layer_types):
        kinds = tuple(layer_types)
        if len(kinds) != 64 or kinds.count("full_attention") != 16 or kinds.count("linear_attention") != 48:
            raise ValueError("Require all 64 Qwen layers in model order")
        return cls(
            kinds,
            tp_size=4,
            page_tokens=32,
            kv_page_bytes=8 * 1088,
            recurrent_bytes=12 * 128 * 128 * 4,
            conv_bytes=3 * (2 * 4 * 128 + 12 * 128) * 2,
            abi="tt-bh-tp4-bfp8-tile32-kv-fp32-tiled-gdn-bf16-row-conv-v1",
        )


@dataclass(frozen=True)
class Segment:
    rank: int
    layer: int
    kind: str
    size: int


@dataclass(frozen=True)
class Checkpoint:
    identity: Identity
    layout: Layout
    consumed: int
    token_sha256: str

    def __post_init__(self):
        if not isinstance(self.identity, Identity) or not isinstance(self.layout, Layout):
            raise ValueError("Checkpoint needs typed identity and layout")
        if not positive(self.consumed) or self.consumed % self.layout.page_tokens:
            raise ValueError("Consumed frontier must be a positive complete page boundary")
        if (
            not isinstance(self.token_sha256, str)
            or len(self.token_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.token_sha256)
        ):
            raise ValueError("Require a SHA256 token digest")

    @classmethod
    def for_tokens(cls, identity, layout, tokens, consumed):
        if not positive(consumed) or consumed > len(tokens):
            raise ValueError("Cannot checkpoint an unconsumed token")
        hasher = hashlib.sha256()
        for token in tokens[:consumed]:
            if type(token) is not int or not 0 <= token < 2**32:
                raise ValueError("Token IDs must be uint32 integers")
            hasher.update(struct.pack("<I", token))
        return cls(identity, layout, consumed, hasher.hexdigest())

    @property
    def key(self):
        return digest(asdict(self))

    @property
    def header(self):
        return canonical({"schema": 1, "checkpoint": asdict(self)})

    def segments(self):
        """KV bytes follow logical page order, never old physical page IDs.

        Recurrent/conv segments contain only this request's consumed frontier,
        with the per-rank encoding identified by layout. No sampler/RNG state:
        this format reuses prefixes; it does not resume a paused generation.
        """
        for rank in range(self.layout.tp_size):
            for layer, kind in enumerate(self.layout.layer_types):
                sizes = (
                    (("key", self.kv_bytes), ("value", self.kv_bytes))
                    if kind == "full_attention"
                    else (("recurrent", self.layout.recurrent_bytes), ("conv", self.layout.conv_bytes))
                )
                for name, size in sizes:
                    yield Segment(rank, layer, name, size)

    @property
    def kv_bytes(self):
        return self.consumed // self.layout.page_tokens * self.layout.kv_page_bytes

    @property
    def payload_bytes(self):
        return sum(segment.size for segment in self.segments())

    @property
    def encoded_bytes(self):
        return len(MAGIC) + 4 + len(self.header) + sum(s.size + 32 for s in self.segments()) + len(COMPLETE)


class Store(Protocol):
    """Opaque immutable blobs; existing storage backends can implement this API."""

    def write(self, key: str, size: int) -> ContextManager[BinaryIO]:
        ...

    def read(self, key: str) -> ContextManager[BinaryIO]:
        ...

    def contains(self, key: str) -> bool:
        ...


class FrozenSource(Protocol):
    def read(self, segment: Segment, offset: int, size: int) -> bytes:
        ...


class CaptureSource(Protocol):
    def freeze(self, checkpoint: Checkpoint) -> ContextManager[FrozenSource]:
        """Fence in-flight work, verify the consumed tokens, and pin ALL ranks.

        Before returning, publish any resident decode-bucket recurrent state
        back to its canonical slot. Hold pages and the immutable recurrent
        frontier until exit. Freeze failure must not create a visible entry.
        """
        ...


class RestoreTransaction(Protocol):
    def write(self, segment: Segment, offset: int, data: bytes) -> None:
        ...

    def commit(self, consumed: int) -> None:
        """Fence ALL rank writes, verify request generation/cancellation, publish.

        Restore into the stable recurrent slot and exclusively owned KV pages;
        remap logical pages through the destination allocation. No request may
        decode from the slot before commit. It starts with its own RNG/logits.
        """
        ...


class RestoreTarget(Protocol):
    def begin(self, checkpoint: Checkpoint) -> ContextManager[RestoreTransaction]:
        """Lease an unobservable destination. Abort/reset it on any exception.

        Cancellation or checksum failure must discard partial writes, release
        private pages, and force cold prefill. Never mutate another live slot,
        immutable shared prefix, or captured tensor address.
        """
        ...


def chunks(segment, chunk_bytes):
    for offset in range(0, segment.size, chunk_bytes):
        yield offset, min(chunk_bytes, segment.size - offset)


def validate_chunk_size(chunk_bytes):
    if not positive(chunk_bytes) or chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError("Transfer staging must be 1 byte through 1 MiB")


def capture(store, checkpoint, source, *, chunk_bytes=MAX_CHUNK_BYTES):
    """Stream one checkpoint with <=chunk_bytes codec staging (adapter excluded)."""
    validate_chunk_size(chunk_bytes)
    # Publication must occur AFTER source lease exit, including its final fence.
    with store.write(checkpoint.key, checkpoint.encoded_bytes) as output:
        with source.freeze(checkpoint) as frozen:
            output.write(MAGIC + struct.pack("<I", len(checkpoint.header)) + checkpoint.header)
            for segment in checkpoint.segments():
                hasher = hashlib.sha256()
                for offset, size in chunks(segment, chunk_bytes):
                    data = frozen.read(segment, offset, size)
                    if not isinstance(data, bytes) or len(data) != size:
                        raise ValueError("Source returned an incomplete or non-byte segment")
                    hasher.update(data)
                    output.write(data)
                output.write(hasher.digest())
            output.write(COMPLETE)


def restore(store, checkpoint, target, *, chunk_bytes=MAX_CHUNK_BYTES):
    """Validate while streaming to a private destination, then commit atomically.

    Callers treat missing, incompatible or corrupt entries as cache misses and
    cold-prefill only after the target lease has aborted. Never swallow a
    failed abort: the adapter must quarantine that destination.
    """
    validate_chunk_size(chunk_bytes)
    expected = MAGIC + struct.pack("<I", len(checkpoint.header)) + checkpoint.header
    with store.read(checkpoint.key) as stream:
        if stream.read(len(expected)) != expected:
            raise ValueError("Incompatible or incomplete checkpoint header")
        with target.begin(checkpoint) as transaction:
            for segment in checkpoint.segments():
                hasher = hashlib.sha256()
                for offset, size in chunks(segment, chunk_bytes):
                    data = stream.read(size)
                    if len(data) != size:
                        raise ValueError("Incomplete checkpoint payload")
                    hasher.update(data)
                    transaction.write(segment, offset, data)
                if stream.read(32) != hasher.digest():
                    raise ValueError("Checkpoint payload checksum mismatch")
            if stream.read(len(COMPLETE)) != COMPLETE or stream.read(1):
                raise ValueError("Checkpoint completion record missing or trailing bytes present")
            transaction.commit(checkpoint.consumed)


def find_prefix(store, identity, layout, tokens, frontiers):
    """Match only caller/scheduler-selected frontiers; leave a token for logits.

    No token or recurrent snapshot is fabricated for a longer matching KV hit.
    At an exact-prefix hit without cached logits, use an earlier checkpoint.
    This helper is a reference policy; production block matching stays in vLLM.
    """
    candidates = []
    for frontier in sorted(set(frontiers)):
        if not positive(frontier) or frontier % layout.page_tokens:
            raise ValueError("Candidate frontiers must be positive complete page boundaries")
        if frontier >= len(tokens):
            continue
        candidates.append(frontier)
    if not candidates:
        return None
    # A 256K miss across 4K checkpoints must not hash the same token history
    # 64 times. Preserve the exact uint32 digest while streaming it once.
    wanted, checkpoints, hasher = set(candidates), {}, hashlib.sha256()
    for index in range(candidates[-1]):
        token = tokens[index]
        if type(token) is not int or not 0 <= token < 2**32:
            raise ValueError("Token IDs must be uint32 integers")
        hasher.update(struct.pack("<I", token))
        consumed = index + 1
        if consumed in wanted:
            checkpoints[consumed] = Checkpoint(identity, layout, consumed, hasher.hexdigest())
    for frontier in reversed(candidates):
        checkpoint = checkpoints[frontier]
        if store.contains(checkpoint.key):
            return checkpoint
    return None
