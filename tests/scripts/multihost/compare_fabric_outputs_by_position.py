#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare two control plane generated/fabric directories by physical position.

For runs of the same mesh graph on cluster descriptors that describe the same machine but number its ASICs
differently, e.g. captured descriptors vs ones generated from a factory system descriptor (tt-run --offline).
asic_id and UMD chip ids are numbering, so they are not compared:
  - asic_to_fabric_node_mapping_rank_<r>_of_<n>: per rank, (hostname, tray_id, asic_location) -> fabric node.
  - physical_chip_mesh_coordinate_mapping_<r>_of_<n>: per rank, the set of mesh coordinates. Each one is
    chip_to_coordinate(fabric node), keyed by a process-local UMD chip id.
  - intermesh_port_assignment_rank_<r>_of_<n>: compared as text.

usage: compare_fabric_outputs_by_position.py <generated/fabric a> <generated/fabric b>
"""

import argparse
import pathlib
import re
import sys

import yaml

RANK_FILE = re.compile(r"_(\d+)_of_(\d+)\.yaml$")
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def per_rank_files(directory, prefix):
    return {
        RANK_FILE.search(f.name).groups(): f
        for f in pathlib.Path(directory).glob(f"{prefix}*_of_*.yaml")
        if RANK_FILE.search(f.name)
    }


def placement(path):
    out = {}
    for host in yaml.load(path.read_text(), Loader=LOADER)["asic_to_fabric_node_mapping"]["hostnames"]:
        for entry in host["mesh"]:
            for chip in entry.get("chips", []):
                position = (host["hostname"], chip["asic_position"]["tray_id"], chip["asic_position"]["asic_location"])
                out[position] = (chip["fabric_node_id"]["mesh_id"], chip["fabric_node_id"]["chip_id"])
    return out


def coordinates(path):
    return sorted(tuple(c) for c in (yaml.load(path.read_text(), Loader=LOADER)["chips"] or {}).values())


def compare(name, a_files, b_files, load):
    if a_files.keys() != b_files.keys():
        print(f"{name}: rank files differ: {sorted(a_files)} vs {sorted(b_files)}")
        return False
    bad = [rank for rank in sorted(a_files) if load(a_files[rank]) != load(b_files[rank])]
    for rank in bad[:5]:
        print(f"{name}: rank {rank[0]} of {rank[1]} differs ({a_files[rank].name})")
    if not a_files:
        print(f"{name}: no files")
    return bool(a_files) and not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=pathlib.Path)
    parser.add_argument("b", type=pathlib.Path)
    args = parser.parse_args()

    ok = compare(
        "asic_to_fabric_node_mapping",
        per_rank_files(args.a, "asic_to_fabric_node_mapping_rank_"),
        per_rank_files(args.b, "asic_to_fabric_node_mapping_rank_"),
        placement,
    )
    ok &= compare(
        "physical_chip_mesh_coordinate_mapping",
        per_rank_files(args.a, "physical_chip_mesh_coordinate_mapping_"),
        per_rank_files(args.b, "physical_chip_mesh_coordinate_mapping_"),
        coordinates,
    )
    ok &= compare(
        "intermesh_port_assignment",
        per_rank_files(args.a, "intermesh_port_assignment_rank_"),
        per_rank_files(args.b, "intermesh_port_assignment_rank_"),
        lambda path: path.read_text(),
    )
    print(f"{args.a} vs {args.b}: {'match by physical position' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
