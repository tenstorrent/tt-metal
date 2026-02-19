#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Combine per-host rank binding YAMLs and their referenced mesh graph descriptors
into a single combined rank_binding.yaml and mesh_graph_descriptor.textproto.

Hostnames are supplied as a comma-separated list; their order determines the
rank and mesh_id offset assigned to each host:

  host[0]: offset = 0
  host[1]: offset = ranks_per_host
  host[N]: offset = N * ranks_per_host

Each host's rank_bindings and mesh_ids are shifted by its offset.
TT_VISIBLE_DEVICES is preserved verbatim since it is already host-specific.

The combined textproto replicates the source graph's instances and connections
once per host with all mesh_ids shifted by the host's offset.  Mesh descriptor
definitions are emitted once (dedup by name).

Must be run from the repo root so that mesh_graph_desc_path values (which are
repo-root-relative) resolve correctly.

Usage:
    python combine_rank_bindings.py HOST1,HOST2[,...] [options]
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

_BINDING_FILE_DEFAULT = "bh_galaxy_split_4x2_multi_mesh_rank_binding.yaml"
_OUT_BINDING = "combined_rank_binding.yaml"
_OUT_MGD = "combined_mesh_graph_descriptor.textproto"


# ---------------------------------------------------------------------------
# Textproto helpers
# ---------------------------------------------------------------------------


def _extract_block(content, brace_pos):
    """Return (inner_content, end_pos) for the {} block whose '{' is at brace_pos."""
    assert content[brace_pos] == "{"
    depth, j = 0, brace_pos
    while j < len(content):
        if content[j] == "{":
            depth += 1
        elif content[j] == "}":
            depth -= 1
            if depth == 0:
                return content[brace_pos + 1 : j], j
        j += 1
    raise ValueError(f"Unmatched '{{' at offset {brace_pos}")


def _find_blocks(content, keyword):
    """Return inner-content strings for every `keyword { ... }` block."""
    blocks = []
    for m in re.finditer(r"\b" + re.escape(keyword) + r"\s*\{", content):
        inner, _ = _extract_block(content, m.end() - 1)
        blocks.append(inner)
    return blocks


def _str_field(content, name):
    m = re.search(r"\b" + re.escape(name) + r'\s*:\s*"([^"]*)"', content)
    return m.group(1) if m else None


def _int_field(content, name, default=0):
    m = re.search(r"\b" + re.escape(name) + r"\s*:\s*(\d+)", content)
    return int(m.group(1)) if m else default


# ---------------------------------------------------------------------------
# Mesh graph descriptor parsing
# ---------------------------------------------------------------------------


def parse_mesh_graph_descriptor(path):
    """
    Parse a mesh graph descriptor textproto.

    Returns a tuple:
      mesh_desc_blocks  list of (inner_raw, name) – one per mesh_descriptors block
      instances         list of (mesh_descriptor_name, mesh_id)
      connections       list of (node_a, node_b, channels_inner, extras)
                          node_a/b  = (mesh_descriptor_name, mesh_id)
                          channels_inner = raw text inside channels { }
                          extras         = dict of additional scalar fields
      graph_name        str
      graph_type        str
      top_level_raw     str – verbatim top_level_instance block
    """
    content = Path(path).read_text()

    # mesh_descriptor blocks – preserve raw so we can emit them unchanged
    mesh_desc_blocks = []
    for m in re.finditer(r"\bmesh_descriptors\s*\{", content):
        inner, _ = _extract_block(content, m.end() - 1)
        mesh_desc_blocks.append((inner, _str_field(inner, "name")))

    # graph_descriptor (expect exactly one)
    graph_inners = _find_blocks(content, "graph_descriptors")
    if len(graph_inners) != 1:
        raise ValueError(f"{path}: expected 1 graph_descriptors block, found {len(graph_inners)}")
    g = graph_inners[0]

    graph_name = _str_field(g, "name") or "G0"
    graph_type = _str_field(g, "type") or "FABRIC"

    instances = []
    for inst in _find_blocks(g, "instances"):
        for mesh in _find_blocks(inst, "mesh"):
            instances.append((_str_field(mesh, "mesh_descriptor") or "", _int_field(mesh, "mesh_id")))

    connections = []
    for conn in _find_blocks(g, "connections"):
        node_inners = _find_blocks(conn, "nodes")
        if len(node_inners) != 2:
            continue
        nodes = []
        for node in node_inners:
            for mesh in _find_blocks(node, "mesh"):
                nodes.append((_str_field(mesh, "mesh_descriptor") or "", _int_field(mesh, "mesh_id")))
        ch_blocks = _find_blocks(conn, "channels")
        channels_inner = ch_blocks[0].strip() if ch_blocks else "count: 0"
        extras = {}
        for field in ("directional", "assign_z_direction"):
            fm = re.search(r"\b" + field + r"\s*:\s*(\w+)", conn)
            if fm:
                extras[field] = fm.group(1)
        if len(nodes) == 2:
            connections.append((nodes[0], nodes[1], channels_inner, extras))

    # top_level_instance – reconstruct the block verbatim
    top_inners = _find_blocks(content, "top_level_instance")
    top_level_raw = (
        "top_level_instance {" + top_inners[0] + "}"
        if top_inners
        else 'top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }'
    )

    return mesh_desc_blocks, instances, connections, graph_name, graph_type, top_level_raw


# ---------------------------------------------------------------------------
# Combined textproto generation
# ---------------------------------------------------------------------------


def generate_combined_descriptor(
    mesh_desc_blocks, all_instances, all_connections, graph_name, graph_type, top_level_raw
):
    lines = []

    # Mesh descriptors – emit each unique name once, preserving raw content
    seen = set()
    for inner, name in mesh_desc_blocks:
        if name not in seen:
            lines += ["mesh_descriptors {", inner.rstrip(), "}", ""]
            seen.add(name)

    # Graph descriptor
    lines += [
        "graph_descriptors {",
        f'  name: "{graph_name}"',
        f'  type: "{graph_type}"',
    ]
    for desc_name, mesh_id in all_instances:
        lines.append(f'  instances {{ mesh {{ mesh_descriptor: "{desc_name}" mesh_id: {mesh_id} }} }}')
    for (desc_a, id_a), (desc_b, id_b), channels_inner, extras in all_connections:
        lines.append("  connections {")
        lines.append(f'    nodes {{ mesh {{ mesh_descriptor: "{desc_a}" mesh_id: {id_a} }} }}')
        lines.append(f'    nodes {{ mesh {{ mesh_descriptor: "{desc_b}" mesh_id: {id_b} }} }}')
        lines.append(f"    channels {{ {channels_inner} }}")
        for k, v in extras.items():
            lines.append(f"    {k}: {v}")
        lines.append("  }")
    lines += ["}", "", top_level_raw]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def combine(hostnames, binding_filename, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load each host's rank binding YAML
    per_host = []
    for hostname in hostnames:
        path = Path(hostname) / binding_filename
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            per_host.append((hostname, yaml.safe_load(f)))

    # Validate consistent rank count across hosts
    ranks_counts = {hostname: len(data["rank_bindings"]) for hostname, data in per_host}
    if len(set(ranks_counts.values())) != 1:
        print(f"error: hosts have different rank counts: {ranks_counts}", file=sys.stderr)
        sys.exit(1)
    ranks_per_host = next(iter(ranks_counts.values()))

    # Validate (warn) consistent mesh graph descriptor path
    mgd_paths = {data["mesh_graph_desc_path"] for _, data in per_host}
    if len(mgd_paths) != 1:
        print(f"warning: hosts reference different mesh graph descriptors: {mgd_paths}", file=sys.stderr)
    source_mgd = next(iter(mgd_paths))
    if not Path(source_mgd).exists():
        print(f"error: mesh graph descriptor not found: {source_mgd}", file=sys.stderr)
        sys.exit(1)

    # Parse source descriptor once
    mesh_desc_blocks, instances, connections, graph_name, graph_type, top_level_raw = parse_mesh_graph_descriptor(
        source_mgd
    )
    meshes_per_host = len(instances)

    # Build combined instances and connections with per-host mesh_id offset
    all_instances = []
    all_connections = []
    for i, (hostname, _) in enumerate(per_host):
        offset = i * meshes_per_host
        for desc_name, mesh_id in instances:
            all_instances.append((desc_name, mesh_id + offset))
        for node_a, node_b, channels_inner, extras in connections:
            all_connections.append(
                (
                    (node_a[0], node_a[1] + offset),
                    (node_b[0], node_b[1] + offset),
                    channels_inner,
                    extras,
                )
            )

    # Write combined textproto
    mgd_out = output_dir / _OUT_MGD
    mgd_out.write_text(
        generate_combined_descriptor(
            mesh_desc_blocks, all_instances, all_connections, graph_name, graph_type, top_level_raw
        )
    )
    print(f"written: {mgd_out}")

    # Build combined rank bindings with per-host rank and mesh_id offsets
    combined_bindings = []
    for i, (hostname, data) in enumerate(per_host):
        rank_offset = i * ranks_per_host
        mesh_offset = i * meshes_per_host
        for binding in data["rank_bindings"]:
            new_binding = dict(binding)
            new_binding["rank"] = binding["rank"] + rank_offset
            new_binding["mesh_id"] = binding["mesh_id"] + mesh_offset
            # env_overrides (TT_VISIBLE_DEVICES) is host-specific; preserve as-is
            combined_bindings.append(new_binding)

    # Store the MGD path relative to CWD (repo root convention) where possible
    try:
        mgd_path_str = str(mgd_out.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        mgd_path_str = str(mgd_out.resolve())

    binding_out = output_dir / _OUT_BINDING
    with open(binding_out, "w") as f:
        yaml.dump(
            {"rank_bindings": combined_bindings, "mesh_graph_desc_path": mgd_path_str},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"written: {binding_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-host rank binding YAMLs into a single rank_binding.yaml and mesh_graph_descriptor.textproto."
    )
    parser.add_argument(
        "hosts",
        help="Comma-separated hostnames in offset order (e.g. host0,host1,host2)",
    )
    parser.add_argument(
        "--binding-file",
        default=_BINDING_FILE_DEFAULT,
        metavar="FILE",
        help=f"Rank binding filename within each hostname directory (default: {_BINDING_FILE_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        metavar="DIR",
        help="Directory to write combined files into (default: current directory)",
    )
    args = parser.parse_args()

    hostnames = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hostnames:
        parser.error("at least one hostname is required")

    combine(hostnames, args.binding_file, args.output_dir)


if __name__ == "__main__":
    main()
