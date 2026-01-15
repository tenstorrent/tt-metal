# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import yaml
import argparse
import mesh_graph_descriptor_pb2
from google.protobuf import text_format
from enum import Enum


# Given .textproto from MGD or FSD

# Parse for the topology, board etc.

# To start, only perform neighboring latency checks


class TorusTopology(Enum):
    INVALID_TYPE = 0
    LINE = 1
    RING = 2


class Architecture(Enum):
    INVALID_TYPE = 0
    WORMHOLE = 1
    BLACKHOLE = 2


def create_test(
    name: str,
    topology: str,
    dir: int,
    idx: int,
    src_idx: int,
    dst_idx: int,
    src_core: list[int],
    dst_core: list[int],
    ntype: str,
    mode: str,
    num_links: int,
):
    """Create a test configuration.

    Args:
        name: Test name
        topology: 'Linear' or 'Ring'
        dir: 0 for x-direction, 1 for y-direction
        idx: Fixed index (row or column)
        src_idx: Source device index in the varying dimension
        dst_idx: Destination device index in the varying dimension
        src_core: Source core coordinates
        dst_core: Destination core coordinates
        ntype: Network type (unicast_write, atomic_inc, etc.)
        mode: 'latency' or 'bandwidth'
        num_links: Number of links (1-4)
    """
    return {
        "name": f"{name}_{mode}_{num_links}links",
        "latency_test_mode": mode == "latency",
        "benchmark_mode": mode == "bandwidth",
        "fabric_setup": {"topology": topology, "num_links": num_links},
        "defaults": {
            "size": 1024,
            "num_packets": 10,
            "ntype": ntype,
            "ftype": "unicast",
        },
        "senders": [
            {
                "device": [0, [idx, src_idx]] if dir == 0 else [0, [src_idx, idx]],
                "core": list(src_core),
                "patterns": [
                    {
                        "destination": {
                            "device": [0, [idx, dst_idx]] if dir == 0 else [0, [dst_idx, idx]],
                            "core": list(dst_core),
                        }
                    }
                ],
            }
        ],
    }


def line_test(
    dir: int,
    idx: int,
    dim: int,
    src_core: list[int],
    dst_core: list[int],
    ntype: str,
    test_modes: list[str],
    num_links: list[int],
):
    tests = []
    for i in range(dim - 1):
        base_name = f"x_{idx}_{i}_{i+1}" if dir == 0 else f"y_{idx}_{i}_{i+1}"
        # Generate test for each mode
        for mode in test_modes:
            # Latency tests always use num_link=1, bandwidth tests parametrize across num_links
            links_to_test = [1] if mode == "latency" else num_links
            for num_link in links_to_test:
                tests.append(
                    create_test(base_name, "Linear", dir, idx, i, i + 1, src_core, dst_core, ntype, mode, num_link)
                )
    return tests


def ring_test(
    dir: int,
    idx: int,
    dim: int,
    src_core: list[int],
    dst_core: list[int],
    ntype: str,
    test_modes: list[str],
    num_links: list[int],
):
    tests = []
    for i in range(dim):
        base_name = f"x_{idx}_{i}_{(i+1) % dim}" if dir == 0 else f"y_{idx}_{i}_{(i+1) % dim}"
        # Generate test for each mode
        for mode in test_modes:
            # Latency tests always use num_link=1, bandwidth tests parametrize across num_links
            links_to_test = [1] if mode == "latency" else num_links
            for num_link in links_to_test:
                tests.append(
                    create_test(
                        base_name, "Ring", dir, idx, i, (i + 1) % dim, src_core, dst_core, ntype, mode, num_link
                    )
                )
    return tests


def gen_tests(dev_dims: list[int], dim_types: list[int], architecture: str, args):
    # Assume 2D for now
    if not dim_types:
        # For empty dim_types assume LINE for both
        dim_types = [TorusTopology.LINE.value, TorusTopology.LINE.value]
    assert len(dim_types) == 2, f"Expected 2D topology (2 dim_types), got {len(dim_types)}: {dim_types}"
    assert len(dev_dims) == 2, f"Expected 2 device dimensions, got {len(dev_dims)}: {dev_dims}"

    tests = []

    ROWS, COLS = dev_dims[0], dev_dims[1]

    # This is a lot of tests to generate, double check this
    if architecture == Architecture.WORMHOLE:
        pass
    elif architecture == Architecture.BLACKHOLE:
        pass

    # iterate rows
    for row in range(ROWS):
        if dim_types[0] == TorusTopology.LINE.value:
            tests.extend(
                line_test(
                    0, row, COLS, list(args.src_core), list(args.dst_core), args.ntype, args.test_modes, args.num_links
                )
            )
        elif dim_types[0] == TorusTopology.RING.value:
            tests.extend(
                ring_test(
                    0, row, COLS, list(args.src_core), list(args.dst_core), args.ntype, args.test_modes, args.num_links
                )
            )
        else:
            raise ValueError("Bad Topology")

    # iterate cols
    for col in range(COLS):
        if dim_types[1] == TorusTopology.LINE.value:
            tests.extend(
                line_test(
                    1, col, ROWS, list(args.src_core), list(args.dst_core), args.ntype, args.test_modes, args.num_links
                )
            )
        elif dim_types[1] == TorusTopology.RING.value:
            tests.extend(
                ring_test(
                    1, col, ROWS, list(args.src_core), list(args.dst_core), args.ntype, args.test_modes, args.num_links
                )
            )
        else:
            raise ValueError("Bad Topology")

    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True)
    parser.add_argument("-o", "--output", default="test_fabric_parametrized.yaml")
    parser.add_argument("--src_core", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--dst_core", type=int, nargs=2, default=[0, 0])
    parser.add_argument(
        "--ntype", type=str, choices=["unicast_write", "atomic_inc", "fused_atomic_inc"], default="unicast_write"
    )
    parser.add_argument(
        "--test_modes",
        type=str,
        nargs="+",
        choices=["latency", "bandwidth"],
        default=["latency", "bandwidth"],
        help="Test modes to generate (latency, bandwidth, or both)",
    )
    parser.add_argument(
        "--num_links",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Number of links to test (e.g., 1 2 3 4). Latency tests ignore this option (1 link)",
    )
    args = parser.parse_args()

    with open(args.filename, "r") as f:
        textproto_content = f.read()

    cfg = mesh_graph_descriptor_pb2.MeshGraphDescriptor()
    text_format.Parse(textproto_content, cfg)

    tests = []
    for m in cfg.mesh_descriptors:
        tests.extend(gen_tests(m.device_topology.dims, m.device_topology.dim_types, m.arch, args))

    out = {"Tests": tests}
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
