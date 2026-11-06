# SPDX-License-Identifier: Apache-2.0
"""Host-only immutable checkpoint index; does not import torch or TTNN."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    tokens: bytes
    end: int
    parent: object
    payload: object
    owned_bytes: int


class CheckpointIndex:
    def __init__(self, namespace, byte_limit, node_limit=128, alignment=4096):
        if byte_limit < 0 or node_limit < 1 or alignment < 1:
            raise ValueError("Invalid cache bounds")
        self.namespace = namespace
        self.byte_limit, self.node_limit, self.alignment = byte_limit, node_limit, alignment
        self.nodes, self.owned_bytes = {}, 0

    def _key(self, tokens):
        return self.namespace, len(tokens), hashlib.sha256(tokens).digest()

    def lookup(self, tokens):
        if len(tokens) % (8 * self.alignment):
            return None
        node = self.nodes.get(self._key(tokens))
        return node if node is not None and node.tokens == tokens else None

    def parent(self, tokens):
        for node in sorted(self.nodes.values(), key=lambda value: value.end, reverse=True):
            if len(node.tokens) < len(tokens) and tokens.startswith(node.tokens):
                return node
        return None

    def can_admit(self, tokens, owned_bytes):
        return (
            bool(tokens)
            and len(tokens) % (8 * self.alignment) == 0
            and owned_bytes >= 0
            and (
                self._key(tokens) not in self.nodes
                and len(self.nodes) < self.node_limit
                and self.owned_bytes + owned_bytes <= self.byte_limit
            )
        )

    def insert(self, tokens, payload, owned_bytes, parent):
        if not self.can_admit(tokens, owned_bytes):
            return None
        if parent is not None and (
            self.lookup(parent.tokens) is not parent
            or not tokens.startswith(parent.tokens)
            or len(parent.tokens) >= len(tokens)
        ):
            raise ValueError("Ancestor must remain owned and be a strict exact prefix")
        node = Checkpoint(tokens, len(tokens) // 8, parent, payload, owned_bytes)
        self.nodes[self._key(tokens)] = node
        self.owned_bytes += owned_bytes
        return node

    def clear_after_device_fence(self):
        self.nodes.clear()
        self.owned_bytes = 0


def bank_reserved_bytes(pages, page_bytes, banks, alignment):
    """Interleaved allocator bank spread, excluding separately budgeted scratch."""
    if pages < 0 or page_bytes < 1 or banks < 1 or alignment < 1:
        raise ValueError("Invalid allocator geometry")
    return ((pages + banks - 1) // banks) * ((page_bytes + alignment - 1) // alignment) * alignment * banks
