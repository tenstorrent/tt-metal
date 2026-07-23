#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve the physical ring order of hosts from descriptor files.

Reads cabling+deployment descriptors (or a Factory System Descriptor) to derive
which hosts are directly connected, then walks the adjacency graph to produce
the ring-ordered host list required by fabric tests.

Textproto files are parsed with a lightweight regex-based parser so this script
has zero dependencies beyond the Python 3 standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Safe descriptor path I/O (OWASP path-traversal guidance)
# ---------------------------------------------------------------------------

# Absolute CLI paths must resolve under one of these roots (plus cwd, the
# tt-metal checkout that contains this script, and tempfile.gettempdir()).
# Extend at runtime with EXABOX_DESCRIPTOR_ALLOWED_ROOTS (os.pathsep-separated).
_DEFAULT_DESCRIPTOR_ALLOWED_ROOTS: tuple[str, ...] = (
    "/data/scaleout_configs",
    "/tmp",
    "/var/tmp",
)


def _descriptor_allowed_roots() -> list[str]:
    """Return absolute directory roots under which descriptor files may be read."""
    roots = [os.path.abspath(r) for r in _DEFAULT_DESCRIPTOR_ALLOWED_ROOTS]
    roots.append(os.path.abspath(os.getcwd()))
    roots.append(os.path.abspath(tempfile.gettempdir()))
    # tools/scaleout/exabox/<this file> -> tt-metal repo root
    roots.append(os.path.abspath(str(Path(__file__).resolve().parents[3])))
    extra = os.environ.get("EXABOX_DESCRIPTOR_ALLOWED_ROOTS", "")
    for part in extra.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(os.path.abspath(part))
    return roots


def _safe_read_text(path_arg: str) -> str:
    """Read a descriptor file after validating the path stays in an allowed root.

    Uses the OWASP-recommended absolute-path safelist check so dynamic CLI input
    cannot be used to read files outside the intended scope (path traversal).
    """
    if not path_arg or "\x00" in path_arg:
        raise ValueError(f"invalid descriptor path: {path_arg!r}")

    raw = Path(path_arg)
    if raw.is_symlink():
        raise ValueError(f"refusing to read symlink: {path_arg}")

    # Normalize .. / . components before the containment check.
    abs_path = os.path.abspath(path_arg)
    if not any(abs_path == root or abs_path.startswith(root + os.sep) for root in _descriptor_allowed_roots()):
        raise ValueError(f"refusing path outside allowed descriptor roots: {abs_path}")

    path = Path(abs_path)
    if not path.is_file():
        raise FileNotFoundError(f"descriptor is not a regular file: {abs_path}")

    return path.read_text()


# ---------------------------------------------------------------------------
# Lightweight textproto parser
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split textproto into tokens: braces, colons, strings, and barewords."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in " \t\r\n,;":
            i += 1
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
        elif c in "{}:[]":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            tokens.append(text[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < len(text) and text[j] not in ' \t\r\n,;{}:#[]"':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_value(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Parse a single textproto value starting at *pos*."""
    if tokens[pos] == "{":
        return _parse_message(tokens, pos + 1)
    if tokens[pos] == "[":
        items: list[Any] = []
        pos += 1
        while tokens[pos] != "]":
            val, pos = _parse_value(tokens, pos)
            items.append(val)
        return items, pos + 1
    return tokens[pos], pos + 1


def _parse_message(tokens: list[str], pos: int) -> tuple[dict, int]:
    """Parse a textproto message body (everything between braces)."""
    msg: dict[str, Any] = {}
    while pos < len(tokens) and tokens[pos] != "}":
        key = tokens[pos]
        pos += 1
        if pos < len(tokens) and tokens[pos] == ":":
            pos += 1
        val, pos = _parse_value(tokens, pos)
        if key in msg:
            existing = msg[key]
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(val)
            msg[key] = existing
        else:
            msg[key] = val
    if pos < len(tokens) and tokens[pos] == "}":
        pos += 1
    return msg, pos


def parse_textproto(text: str) -> dict:
    """Parse a textproto file into nested dicts/lists."""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    msg, _ = _parse_message(tokens, 0)
    return msg


def _normalize_map(value: Any) -> dict[str, Any]:
    """Convert protobuf map fields (repeated ``{key value}`` messages) into a dict.

    Textproto maps like ``child_mappings { key: "n1" value { ... } }`` are
    parsed as either a single dict or a list of dicts.  This normalises both
    forms into ``{"n1": {...}, ...}``.
    """
    if isinstance(value, dict) and "key" in value and "value" in value:
        return {value["key"]: value["value"]}
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[item["key"]] = item["value"]
        if result:
            return result
    if isinstance(value, dict):
        return value
    return {}


# ---------------------------------------------------------------------------
# FSD mode: build adjacency from Factory System Descriptor
# ---------------------------------------------------------------------------


def _build_adjacency_from_fsd(fsd: dict) -> tuple[dict[int, str], dict[int, set[int]]]:
    """Return (host_id_to_name, adjacency) from a parsed FSD textproto."""
    hosts_raw = fsd.get("hosts", [])
    if not isinstance(hosts_raw, list):
        hosts_raw = [hosts_raw]

    host_id_to_name: dict[int, str] = {}
    for idx, h in enumerate(hosts_raw):
        host_id_to_name[idx] = h.get("hostname", h.get("host", f"host_{idx}"))

    adjacency: dict[int, set[int]] = defaultdict(set)
    eth_conns = fsd.get("eth_connections", {})
    connections = eth_conns.get("connection", [])
    if not isinstance(connections, list):
        connections = [connections]

    for conn in connections:
        ep_a = conn.get("endpoint_a", {})
        ep_b = conn.get("endpoint_b", {})
        hid_a = int(ep_a.get("host_id", -1))
        hid_b = int(ep_b.get("host_id", -1))
        if hid_a != hid_b and hid_a >= 0 and hid_b >= 0:
            adjacency[hid_a].add(hid_b)
            adjacency[hid_b].add(hid_a)

    return host_id_to_name, dict(adjacency)


# ---------------------------------------------------------------------------
# Cabling + deployment mode
# ---------------------------------------------------------------------------


def _resolve_leaf_host_ids(
    instance: dict,
    templates: dict[str, dict],
) -> dict[str, int]:
    """Walk a root/sub GraphInstance to map leaf node names to host_id.

    Returns a dict mapping the *full path element* (the child key) to its
    ``host_id``.  For nested graphs, paths are joined with ``/``.
    """
    child_mappings = _normalize_map(instance.get("child_mappings", {}))

    result: dict[str, int] = {}
    for key, mapping in child_mappings.items():
        if isinstance(mapping, dict):
            if "host_id" in mapping:
                result[key] = int(mapping["host_id"])
            elif "sub_instance" in mapping:
                sub = mapping["sub_instance"]
                sub_map = _resolve_leaf_host_ids(sub, templates)
                for sub_key, hid in sub_map.items():
                    result[f"{key}/{sub_key}"] = hid
        else:
            try:
                result[key] = int(mapping)
            except (ValueError, TypeError):
                pass
    return result


def _host_id_for_path(path: list[str], path_to_hid: dict[str, int]) -> int | None:
    """Resolve a connection port path to a host_id.

    Paths in cabling descriptors are like ``["node1"]`` or
    ``["superpod1", "node3"]``.  The leaf element is a node whose host_id we
    stored.  We try the full ``/``-joined path first, then successively longer
    prefixes that include the leaf.
    """
    joined = "/".join(path)
    if joined in path_to_hid:
        return path_to_hid[joined]
    if len(path) == 1 and path[0] in path_to_hid:
        return path_to_hid[path[0]]
    return None


def _collect_connections_recursive(
    template_name: str,
    templates: dict[str, dict],
    path_to_hid: dict[str, int],
    prefix: list[str],
    adjacency: dict[int, set[int]],
) -> None:
    """Recursively walk graph templates and collect cross-host edges."""
    template = templates.get(template_name, {})
    internal_connections = _normalize_map(template.get("internal_connections", {}))

    for _port_type, port_conns in internal_connections.items():
        conns = port_conns if isinstance(port_conns, dict) else {}
        conn_list = conns.get("connections", [])
        if not isinstance(conn_list, list):
            conn_list = [conn_list]

        for conn in conn_list:
            pa = conn.get("port_a", {})
            pb = conn.get("port_b", {})
            path_a = pa.get("path", [])
            path_b = pb.get("path", [])
            if not isinstance(path_a, list):
                path_a = [path_a]
            if not isinstance(path_b, list):
                path_b = [path_b]

            full_a = prefix + path_a
            full_b = prefix + path_b

            hid_a = _host_id_for_path(full_a, path_to_hid)
            hid_b = _host_id_for_path(full_b, path_to_hid)
            if hid_a is not None and hid_b is not None and hid_a != hid_b:
                adjacency[hid_a].add(hid_b)
                adjacency[hid_b].add(hid_a)

    children = template.get("children", [])
    if not isinstance(children, list):
        children = [children]

    for child in children:
        name = child.get("name", "")
        graph_ref = child.get("graph_ref", {})
        if isinstance(graph_ref, dict) and "graph_template" in graph_ref:
            sub_template = graph_ref["graph_template"]
            _collect_connections_recursive(sub_template, templates, path_to_hid, prefix + [name], adjacency)


def _build_adjacency_from_cabling(cabling: dict, deployment: dict) -> tuple[dict[int, str], dict[int, set[int]]]:
    """Return (host_id_to_name, adjacency) from cabling + deployment."""
    hosts_raw = deployment.get("hosts", [])
    if not isinstance(hosts_raw, list):
        hosts_raw = [hosts_raw]

    host_id_to_name: dict[int, str] = {}
    for idx, h in enumerate(hosts_raw):
        host_id_to_name[idx] = h.get("host", f"host_{idx}")

    templates = _normalize_map(cabling.get("graph_templates", {}))

    root_instance = cabling.get("root_instance", {})
    path_to_hid = _resolve_leaf_host_ids(root_instance, templates)

    adjacency: dict[int, set[int]] = defaultdict(set)
    root_template = root_instance.get("template_name", "")
    _collect_connections_recursive(root_template, templates, path_to_hid, [], adjacency)

    return host_id_to_name, dict(adjacency)


# ---------------------------------------------------------------------------
# Ring walk
# ---------------------------------------------------------------------------


def _short_name(fqdn: str) -> str:
    return fqdn.split(".")[0]


def _walk_ring(
    requested_hosts: list[str],
    host_id_to_name: dict[int, str],
    adjacency: dict[int, set[int]],
) -> list[str]:
    """Walk the host adjacency subgraph as a ring and return ordered hosts.

    Raises ``ValueError`` when the subgraph is not a single ring.
    """
    name_to_hid: dict[str, int] = {}
    for hid, name in host_id_to_name.items():
        name_to_hid[_short_name(name)] = hid
        name_to_hid[name] = hid

    input_short_to_fqdn: dict[str, str] = {}
    for h in requested_hosts:
        input_short_to_fqdn[_short_name(h)] = h

    requested_hids: set[int] = set()
    for h in requested_hosts:
        hid = name_to_hid.get(h) or name_to_hid.get(_short_name(h))
        if hid is None:
            raise ValueError(f"Host '{h}' not found in descriptor (known: {list(host_id_to_name.values())})")
        requested_hids.add(hid)

    if len(requested_hids) == 1:
        hid = next(iter(requested_hids))
        name = host_id_to_name[hid]
        return [input_short_to_fqdn.get(_short_name(name), name)]

    sub_adj: dict[int, set[int]] = {}
    for hid in requested_hids:
        neighbors = adjacency.get(hid, set()) & requested_hids
        sub_adj[hid] = neighbors

    for hid, neighbors in sub_adj.items():
        if len(neighbors) != 2:
            name = host_id_to_name.get(hid, str(hid))
            neighbor_names = [host_id_to_name.get(n, str(n)) for n in neighbors]
            raise ValueError(
                f"Host '{name}' has {len(neighbors)} neighbor(s) {neighbor_names} "
                f"in the requested subset; expected exactly 2 for a ring"
            )

    start = next(iter(requested_hids))
    visited = [start]
    visited_set = {start}
    current = start
    while len(visited) < len(requested_hids):
        neighbors = sub_adj[current]
        unvisited = [n for n in neighbors if n not in visited_set]
        if not unvisited:
            break
        nxt = unvisited[0]
        visited.append(nxt)
        visited_set.add(nxt)
        current = nxt

    if len(visited) != len(requested_hids):
        raise ValueError(f"Could not walk a complete ring: visited {len(visited)} of {len(requested_hids)} hosts")

    last_neighbors = sub_adj[visited[-1]]
    if visited[0] not in last_neighbors:
        raise ValueError("Hosts do not form a ring: last host is not connected back to the first")

    ordered: list[str] = []
    for hid in visited:
        name = host_id_to_name[hid]
        ordered.append(input_short_to_fqdn.get(_short_name(name), name))
    return ordered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve physical ring order of hosts from descriptor files.")
    parser.add_argument(
        "--hosts",
        required=True,
        help="Comma-separated hostnames/FQDNs to order",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fsd",
        help=(
            "Path to Factory System Descriptor (.textproto). "
            "Must resolve under an allowed root (see EXABOX_DESCRIPTOR_ALLOWED_ROOTS)."
        ),
    )
    group.add_argument(
        "--cabling",
        help=(
            "Path to cabling descriptor (.textproto); requires --deployment. "
            "Must resolve under an allowed root (see EXABOX_DESCRIPTOR_ALLOWED_ROOTS)."
        ),
    )

    parser.add_argument(
        "--deployment",
        help=(
            "Path to deployment descriptor (.textproto); required with --cabling. "
            "Must resolve under an allowed root (see EXABOX_DESCRIPTOR_ALLOWED_ROOTS)."
        ),
    )

    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.cabling and not args.deployment:
        _emit_error("precondition_failed", "--deployment is required when --cabling is used", timestamp)
        return 1

    requested_hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not requested_hosts:
        _emit_error("precondition_failed", "No hosts provided", timestamp)
        return 1

    if len(requested_hosts) <= 1:
        _emit_success(requested_hosts, timestamp)
        return 0

    try:
        if args.fsd:
            text = _safe_read_text(args.fsd)
            fsd = parse_textproto(text)
            host_id_to_name, adjacency = _build_adjacency_from_fsd(fsd)
        else:
            cabling_text = _safe_read_text(args.cabling)
            deployment_text = _safe_read_text(args.deployment)
            cabling = parse_textproto(cabling_text)
            deployment = parse_textproto(deployment_text)
            host_id_to_name, adjacency = _build_adjacency_from_cabling(cabling, deployment)

        ordered = _walk_ring(requested_hosts, host_id_to_name, adjacency)
    except Exception as exc:
        _emit_error("ring_order_failed", str(exc), timestamp)
        return 1

    _emit_success(ordered, timestamp)
    return 0


def _emit_success(ordered: list[str], timestamp: str) -> None:
    print(
        json.dumps(
            {
                "status": "success",
                "message": "Host ring order resolved successfully",
                "ordered_hosts": ",".join(ordered),
                "checked_at": timestamp,
            }
        )
    )


def _emit_error(error_type: str, message: str, timestamp: str) -> None:
    print(
        json.dumps(
            {
                "status": "error",
                "error_type": error_type,
                "message": message,
                "checked_at": timestamp,
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
