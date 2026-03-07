#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Converts a cabling guide CSV (cutsheet) into cabling and deployment
descriptor textproto files.  No dependencies beyond Python 3 stdlib.

Handles both 20-column (location info) and 8-column (hostname-only)
CSV formats emitted by run_cabling_generator.

See --help for usage.
"""

import argparse
import sys
from pathlib import Path

# Maps alternative / lowercase node type strings coming from CSVs to the
# canonical names expected by the cabling generator protobuf schema.
NODE_TYPE_ALIASES = {
    "blackhole": "BH_GALAXY",
    "bh_galaxy": "BH_GALAXY",
    "wh_galaxy": "WH_GALAXY",
    "n300_lb": "N300_LB",
    "n300_qb": "N300_QB",
    "p150_lb": "P150_LB",
    "p150_qb_ae": "P150_QB_AE",
    "p150_qb_global": "P150_QB_GLOBAL",
    "p150_qb_america": "P150_QB_AMERICA",
    "p300_qb_ge": "P300_QB_GE",
}

# Topology-variant suffixes that don't affect the base node type
STRIP_SUFFIXES = ("_xy_torus", "_x_torus", "_y_torus", "_default")

# Header synonyms we accept when parsing CSVs (lowercase).
# Cutsheets from different sources may use slightly different naming.
FIELD_SYNONYMS = {
    "hostname": {"hostname", "host", "node"},
    "hall": {"hall", "building", "facility", "data hall"},
    "aisle": {"aisle", "row", "corridor"},
    "rack": {"rack", "rack_num", "rack_number"},
    "shelf_u": {"shelf u", "shelf_u", "shelf", "u", "unit"},
    "tray": {"tray", "tray_num", "tray_number"},
    "port": {"port", "port_num", "port_number"},
    "label": {"label", "id", "identifier"},
    "node_type": {"node type", "node_type", "type", "model"},
}


def normalize_node_type(raw):
    if not raw:
        return "BH_GALAXY"
    s = raw.strip().lower()
    for suffix in STRIP_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return NODE_TYPE_ALIASES.get(s, s.upper())


def safe_int(value, default=0):
    v = value.strip()
    return int(v) if v.isdigit() else default


def parse_shelf_u(raw):
    """Strip the leading 'U' prefix that some cutsheets include, e.g. 'U02' -> 2."""
    v = raw.strip()
    if v.upper().startswith("U"):
        v = v[1:]
    return safe_int(v)


def map_headers(headers):
    """Return {field_name: [col_indices]} by matching each header against FIELD_SYNONYMS."""
    positions = {}
    for i, h in enumerate(headers):
        h_low = h.strip().lower()
        for field_name, synonyms in FIELD_SYNONYMS.items():
            if h_low in synonyms:
                positions.setdefault(field_name, []).append(i)
                break
    return positions


def split_source_dest(positions):
    """
    Cutsheet CSVs have source columns then destination columns with the same
    header names repeated.  Figure out where the split is and return two
    {field_name: col_index} dicts.
    """
    dest_start = None
    for poslist in positions.values():
        if len(poslist) > 1:
            dest_start = poslist[1]
            break
    if dest_start is None:
        raise ValueError("CSV does not appear to have Source/Destination column pairs")

    src, dst = {}, {}
    for fname, poslist in positions.items():
        for p in poslist:
            if p < dest_start:
                src.setdefault(fname, p)
            else:
                dst.setdefault(fname, p)
    return src, dst


def read_field(row, field_map, name):
    idx = field_map.get(name, -1)
    if 0 <= idx < len(row):
        return row[idx].strip()
    return ""


def parse_endpoint(row, field_map):
    return {
        "hostname": read_field(row, field_map, "hostname"),
        "hall": read_field(row, field_map, "hall"),
        "aisle": read_field(row, field_map, "aisle"),
        "rack": safe_int(read_field(row, field_map, "rack")),
        "shelf_u": parse_shelf_u(read_field(row, field_map, "shelf_u")),
        "tray": safe_int(read_field(row, field_map, "tray")),
        "port": safe_int(read_field(row, field_map, "port")),
        "node_type": normalize_node_type(read_field(row, field_map, "node_type")),
    }


def parse_csv(csv_path):
    """
    Parse a cabling guide CSV.  Returns (connections, hosts) where:
      connections: list of (source_endpoint, dest_endpoint) dicts
      hosts: sorted list of unique host info dicts (determines host_id assignment)
    """
    lines = Path(csv_path).read_text().splitlines()

    # The first header row is the grouping line ("Source,,,...,Destination,,,...").
    # The actual column names are on the line right after it.
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "source" in low and "destination" in low:
            header_idx = i + 1
            break
    if header_idx is None or header_idx >= len(lines):
        header_idx = 1  # fallback

    if header_idx >= len(lines):
        raise ValueError("CSV has no column-header line")

    positions = map_headers(lines[header_idx].split(","))
    src_fields, dst_fields = split_source_dest(positions)
    data_start = header_idx + 1

    if data_start >= len(lines):
        raise ValueError("CSV has headers but no data rows")

    connections = []
    seen = set()

    for line_no, line in enumerate(lines[data_start:], start=data_start + 1):
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("Source") or raw.startswith("Hostname") or raw.startswith("Data Hall"):
            continue

        row = raw.split(",")
        src = parse_endpoint(row, src_fields)
        dst = parse_endpoint(row, dst_fields)

        if not all([src["tray"], src["port"], dst["tray"], dst["port"]]):
            print(f"Line {line_no}: skipping row with missing tray/port", file=sys.stderr)
            continue

        # Bidirectional dedup: (A->B) and (B->A) are the same cable
        key_a = (src["hostname"], src["tray"], src["port"], dst["hostname"], dst["tray"], dst["port"])
        key_b = (dst["hostname"], dst["tray"], dst["port"], src["hostname"], src["tray"], src["port"])
        norm_key = min(key_a, key_b)
        if norm_key in seen:
            continue
        seen.add(norm_key)

        connections.append((src, dst))

    if not connections:
        raise ValueError("No valid connections found in CSV")

    # Build host list - the index becomes the host_id.
    # Sort order must match tt-CableGen web UI, which runs a DFS-based
    # recalculateHostIndices() in JavaScript after CSV import.  For CSV
    # imports (location mode, shelves nested under racks) the DFS reduces
    # to a flat sort:
    #   20-col (physical): hall asc, aisle asc, rack_num ASC, shelf_u ASC
    #   8-col  (hostname-only): hostname alphabetically ascending
    # Hostname is the final tiebreaker (drives order when location is absent).
    host_map = {}
    for src, dst in connections:
        for ep in (src, dst):
            name = ep["hostname"]
            if name and name not in host_map:
                host_map[name] = {
                    "hostname": name,
                    "node_type": ep["node_type"],
                    "hall": ep["hall"],
                    "aisle": ep["aisle"],
                    "rack": ep["rack"],
                    "shelf_u": ep["shelf_u"],
                }
    hosts = sorted(
        host_map.values(),
        key=lambda h: (h["hall"], h["aisle"], h["rack"], h["shelf_u"], h["hostname"]),
    )

    return connections, hosts


# -- textproto emitters -------------------------------------------------------

def emit_cabling_descriptor(connections, hosts):
    out = []
    out.append('graph_templates {')
    out.append('  key: "extracted_topology"')
    out.append('  value {')

    for h in hosts:
        out.append('    children {')
        out.append(f'      name: "{h["hostname"]}"')
        out.append(f'      node_ref {{ node_descriptor: "{h["node_type"]}" }}')
        out.append('    }')

    out.append('    internal_connections {')
    out.append('      key: "QSFP_DD"')
    out.append('      value {')
    for src, dst in connections:
        out.append(
            f'        connections {{ '
            f'port_a {{ path: "{src["hostname"]}" tray_id: {src["tray"]} port_id: {src["port"]} }} '
            f'port_b {{ path: "{dst["hostname"]}" tray_id: {dst["tray"]} port_id: {dst["port"]} }}'
            f' }}'
        )
    out.append('      }')
    out.append('    }')
    out.append('  }')
    out.append('}')

    out.append('root_instance {')
    out.append('  template_name: "extracted_topology"')
    for i, h in enumerate(hosts):
        out.append(f'  child_mappings {{ key: "{h["hostname"]}" value {{ host_id: {i} }} }}')
    out.append('}')

    return "\n".join(out) + "\n"


def emit_deployment_descriptor(hosts):
    out = []
    for h in hosts:
        out.append("hosts {")
        if h["hall"]:
            out.append(f'  hall: "{h["hall"]}"')
        if h["aisle"]:
            out.append(f'  aisle: "{h["aisle"]}"')
        if h["rack"]:
            out.append(f'  rack: {h["rack"]}')
        if h["shelf_u"]:
            out.append(f'  shelf_u: {h["shelf_u"]}')
        out.append(f'  node_type: "{h["node_type"]}"')
        out.append(f'  host: "{h["hostname"]}"')
        out.append("}")
    return "\n".join(out) + "\n"


# -- cli ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert cabling guide CSV (cutsheet) to cabling and/or deployment descriptor textproto.",
    )
    p.add_argument("csv_file", type=Path, help="Input cabling guide CSV")
    p.add_argument("--cabling-out", type=Path, default=None, help="Write cabling descriptor to file")
    p.add_argument("--deployment-out", type=Path, default=None, help="Write deployment descriptor to file")
    p.add_argument("--cabling-only", action="store_true", help="Only produce cabling descriptor")
    p.add_argument("--deployment-only", action="store_true", help="Only produce deployment descriptor")
    args = p.parse_args()

    if not args.csv_file.is_file():
        print(f"Error: file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    try:
        connections, hosts = parse_csv(args.csv_file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(connections)} connections across {len(hosts)} hosts", file=sys.stderr)

    do_cabling = not args.deployment_only
    do_deployment = not args.cabling_only

    if do_cabling:
        text = emit_cabling_descriptor(connections, hosts)
        if args.cabling_out:
            args.cabling_out.write_text(text)
            print(f"Wrote cabling descriptor to {args.cabling_out}", file=sys.stderr)
        elif not args.deployment_out and not args.deployment_only:
            print(text)

    if do_deployment:
        if any(h["hostname"] for h in hosts):
            text = emit_deployment_descriptor(hosts)
            if args.deployment_out:
                args.deployment_out.write_text(text)
                print(f"Wrote deployment descriptor to {args.deployment_out}", file=sys.stderr)
            elif args.deployment_only and not args.cabling_out:
                print(text)
        else:
            print("Deployment descriptor skipped: no hostnames in CSV", file=sys.stderr)


if __name__ == "__main__":
    main()
