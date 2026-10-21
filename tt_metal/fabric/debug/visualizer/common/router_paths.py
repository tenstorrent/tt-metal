# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
The path rule: how capture and decode name what a manifest router locates.

A router's path is M<mesh>/C<chip>/<router key>, such as M0/C0/E1. Inside a router, a node's path is the keys you follow
from the router to reach it, joined with "/", with array indices written as numbers. For example,
channels/senders/vc0/ch0/ring_buffer is router["channels"]["senders"]["vc0"]["ch0"]["ring_buffer"].
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup


class LeafKind(str, Enum):
    """What a node under a router locates, told by its keys."""

    REGION = "region"  # L1 bytes: it has an address
    STREAM = "stream"  # one stream register: it has a stream_id
    ELEMENT = "element"  # one element of a region: it has an array and an index


@dataclass(frozen=True)
class Leaf:
    path: str  # inside the router
    where: str  # the node's dotted path in the manifest, for messages
    kind: LeafKind
    node: dict


def _kind_of(node: dict) -> LeafKind | None:
    if "address" in node:
        return LeafKind.REGION
    if "stream_id" in node:
        return LeafKind.STREAM
    if "array" in node and "index" in node:
        return LeafKind.ELEMENT
    return None


def _walk(node, path: str, where: str) -> Iterator[Leaf]:
    if isinstance(node, dict):
        kind = _kind_of(node)
        if kind is not None:
            yield Leaf(path, where, kind, node)
            return
        children = [(key, f"{where}.{key}", child) for key, child in node.items()]
    elif isinstance(node, list):
        children = [(index, f"{where}[{index}]", child) for index, child in enumerate(node)]
    else:
        return
    for key, child_where, child in children:
        yield from _walk(child, f"{path}/{key}" if path else str(key), child_where)


def walk_router(router: dict, where: str) -> Iterator[Leaf]:
    """
    Every region, stream reference and element reference under a manifest router, in manifest order. `where` is the
    router's dotted path in the manifest. A leaf's own keys aren't walked, and null nodes, which the builder didn't
    allocate, are skipped.
    """
    return _walk(router, "", where)


def get_router_path(mesh_key: str, chip_key: str, router_key: str) -> str:
    return f"{mesh_key}/{chip_key}/{router_key}"


def get_router(manifest: dict, path: str) -> tuple[dict, str]:
    """The manifest router at a router path such as M0/C0/E1, and its dotted path in the manifest."""
    keys = path.split("/")
    if len(keys) != 3:
        raise ManifestError(f"{path!r} isn't a router path, M<mesh>/C<chip>/<router>")
    mesh_key, chip_key, router_key = keys
    where = f"meshes.{mesh_key}.chips.{chip_key}.routers.{router_key}"
    return lookup(manifest, where, "object"), where


def get_node(router: dict, path: str, where: str):
    """The node at `path` inside a manifest router whose dotted path is `where`."""
    node, node_where = router, where
    for key in path.split("/"):
        if isinstance(node, dict) and key in node:
            node, node_where = node[key], f"{node_where}.{key}"
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node, node_where = node[int(key)], f"{node_where}[{key}]"
        else:
            raise ManifestError(f"{where} has nothing at {path!r}: {node_where} has no {key!r}")
    return node
