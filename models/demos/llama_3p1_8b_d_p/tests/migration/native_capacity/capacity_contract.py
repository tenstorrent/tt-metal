"""Import-free geometry for future capacity owners; no table allocation or native imports."""

import json
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Geometry:
    capacity: int = 2048
    slots: ClassVar[int] = 2
    layers: ClassVar[int] = 32
    chunk_tokens: ClassVar[int] = 1024
    page_tokens: ClassVar[int] = 32
    page_bytes: ClassVar[int] = 4352
    configs: ClassVar[int] = 16

    def __post_init__(self):
        if type(self.capacity) is not int or not 2048 <= self.capacity <= 131072 or self.capacity % 1024:
            raise ValueError("capacity must be integer2048..131072 aligned to1024")

    @property
    def chunks_per_slot(self):
        return self.capacity // self.chunk_tokens

    @property
    def table_entries(self):
        return self.configs * self.slots * self.layers * (self.capacity // self.page_tokens)

    @property
    def packed_bytes(self):
        return self.table_entries * self.page_bytes

    @property
    def full_acks(self):
        return self.slots * self.chunks_per_slot * self.layers

    @property
    def full_audited_calls(self):
        return self.full_acks + self.slots * 4

    @property
    def command_buffer_bytes(self):
        return max(1 << 20, self.capacity * 11 + 4096)


def parse_geometry(text):
    def object_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate field")
            out[key] = value
        return out

    def nonfinite(value):
        raise ValueError("nonfinite JSON")

    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    if set(value) - {"capacity"}:
        raise ValueError("unknown geometry field")
    return Geometry(value.get("capacity", 2048))


def require_same_capacity(plan, source, passive):
    expected = Geometry(plan)
    if Geometry(source) != expected or Geometry(passive) != expected:
        raise ValueError("endpoint capacity mismatch")
    return expected


def require_snapshot_scope(capacity, snapshots):
    geometry = Geometry(capacity)
    if snapshots and geometry.capacity != 2048:
        raise ValueError("large-capacity full snapshots are outside this increment")
