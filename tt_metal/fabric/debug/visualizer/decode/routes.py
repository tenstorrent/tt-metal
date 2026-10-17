# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reads fabric routes from raw bytes, using only the values the fabric manifest carries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Route words are uint32, little-endian as on the device. The loader checks that the manifest's 1D hops fill one.
WORD_BYTES = 4

# The 1D hop names the rules know, each mapped to (writes, forwards).
HOP_ACTIONS = {
    "noop": (False, False),
    "write_only": (True, False),
    "forward_only": (False, True),
    "write_and_forward": (True, True),
}


class EncodingsError(ValueError):
    """The manifest lacks a value the route rules need, or holds one they can't use."""


@dataclass(frozen=True)
class Hop:
    """What a 1D router does at one hop: the hop field's name in routing_encodings.1d.hop, and its actions."""

    name: str
    writes: bool
    forwards: bool


@dataclass(frozen=True)
class Encodings1D:
    bits_per_hop: int
    hops_per_word: int
    hops: tuple[Hop, ...]  # indexed by a hop field's value


@dataclass(frozen=True)
class Encodings:
    """The manifest's routing_encodings, checked."""

    one_d: Encodings1D


def _lookup(manifest: dict, path: str):
    """The manifest's value at a dotted path, such as "routing_encodings.1d.hop"."""
    value = manifest
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise EncodingsError(f"manifest has no {path}")
        value = value[key]
    return value


def _encodings_1d(manifest: dict) -> Encodings1D:
    bits_per_hop = _lookup(manifest, "routing_encodings.1d.bits_per_hop")
    hops_per_word = _lookup(manifest, "routing_encodings.1d.hops_per_word")
    hop = _lookup(manifest, "routing_encodings.1d.hop")
    if bits_per_hop * hops_per_word != 8 * WORD_BYTES:
        raise EncodingsError(
            f"routing_encodings.1d: {hops_per_word} hops of {bits_per_hop} bits don't fill a {WORD_BYTES}-byte word"
        )
    if set(hop) != set(HOP_ACTIONS):
        raise EncodingsError(f"routing_encodings.1d.hop names {sorted(hop)}; the 1D rules know {sorted(HOP_ACTIONS)}")
    if sorted(hop.values()) != list(range(1 << bits_per_hop)):
        raise EncodingsError(
            f"routing_encodings.1d.hop values {sorted(hop.values())} aren't each {bits_per_hop}-bit value once"
        )
    by_value = sorted(hop.items(), key=lambda item: item[1])
    return Encodings1D(bits_per_hop, hops_per_word, tuple(Hop(name, *HOP_ACTIONS[name]) for name, _ in by_value))


def get_encodings_from_manifest(manifest: dict) -> Encodings:
    """Reads the manifest's routing_encodings. Raises EncodingsError naming the first value it can't use."""
    return Encodings(one_d=_encodings_1d(manifest))


def read_packed(data: bytes, bits_per_entry: int, num_entries: int) -> list[int]:
    """`num_entries` entries of `bits_per_entry` bits, read lowest bits first from one run of bits across the bytes."""
    if bits_per_entry * num_entries > 8 * len(data):
        raise ValueError(
            f"{num_entries} entries of {bits_per_entry} bits need {bits_per_entry * num_entries} bits; "
            f"got {8 * len(data)}"
        )
    run = int.from_bytes(data, "little")
    mask = (1 << bits_per_entry) - 1
    return [(run >> (bits_per_entry * index)) & mask for index in range(num_entries)]


def walk_1d(words: Sequence[int], enc: Encodings1D) -> list[Hop]:
    """
    Walks the 1D route.
    `words` are the route words, word 0 first: a header's routing_fields.value, then its route_buffer. Hop 0 is what
    the next router to read the route does (this router's hop has already been consumed), hop 1 the router one chip
    further on, and so on. The last hop is the first one that doesn't forward.
    """
    live = list(words)
    mask = (1 << enc.bits_per_hop) - 1
    hops = []
    # Each forward consumes a field, so once every field is used the next read is of an all-zero word.
    for _ in range(enc.hops_per_word * len(live) + 1):
        # Add the next hop
        hop = enc.hops[live[0] & mask]
        hops.append(hop)

        # Terminal hop, so we can early return
        if not hop.forwards:
            return hops

        # Go to the next hop, including refilling the first word if needed with an extension word
        live[0] >>= enc.bits_per_hop
        if live[0] == 0:
            live = live[1:] + [0]
    raise ValueError(f"1D route forwards past its {len(words)} route words / has no terminal hop")


def get_path_table_entry_1d(paths: bytes, hops: int, extension_words: int) -> list[int]:
    """The route words the 1D path table holds for a destination `hops` hops away, word 0 first."""
    entry_bytes = (1 + extension_words) * WORD_BYTES
    start = hops * entry_bytes
    if start + entry_bytes > len(paths):
        raise ValueError(f"entry {hops} ends at byte {start + entry_bytes}, past the {len(paths)}-byte table")
    offsets = range(start, start + entry_bytes, WORD_BYTES)
    return [int.from_bytes(paths[at : at + WORD_BYTES], "little") for at in offsets]
