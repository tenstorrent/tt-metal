import yaml
import argparse
import mesh_graph_descriptor_pb2
from google.protobuf import runtime_version
from google.protobuf import text_format
from enum import Enum


# Given .textproto from MGD or FSD

# Parse for the topology, board etc.

# To start, only perform neighboring latency checks


class TorusTopology(Enum):
    INVALID_TYPE = 0
    LINE = 1
    RING = 2


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
    """
    return {
        "name": f"{name}_{mode}",
        "latency_test_mode": mode == "latency",
        "benchmark_mode": mode == "bandwidth",
        "fabric_setup": {"topology": topology},
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


def line_test(dir: int, idx: int, dim: int, src_core: list[int], dst_core: list[int], ntype: str):
    tests = []
    for i in range(dim - 1):
        base_name = f"x_{idx}_{i}_{i+1}" if dir == 0 else f"y_{idx}_{i}_{i+1}"
        # Generate both latency and bandwidth tests
        tests.append(create_test(base_name, "Linear", dir, idx, i, i + 1, src_core, dst_core, ntype, "latency"))
        tests.append(create_test(base_name, "Linear", dir, idx, i, i + 1, src_core, dst_core, ntype, "bandwidth"))
    return tests


def ring_test(dir: int, idx: int, dim: int, src_core: list[int], dst_core: list[int], ntype: str):
    tests = []
    for i in range(dim):
        base_name = f"x_{idx}_{i}_{(i+1) % dim}" if dir == 0 else f"y_{idx}_{i}_{(i+1) % dim}"
        # Generate both latency and bandwidth tests
        tests.append(create_test(base_name, "Ring", dir, idx, i, (i + 1) % dim, src_core, dst_core, ntype, "latency"))
        tests.append(create_test(base_name, "Ring", dir, idx, i, (i + 1) % dim, src_core, dst_core, ntype, "bandwidth"))
    return tests


def gen_tests(dev_dims: list[int], dim_types: list[int]):
    # Assume 2D for now
    if not dim_types:
        # For empty dim_types assume LINE for both
        dim_types = [TorusTopology.LINE.value, TorusTopology.LINE.value]
    assert len(dim_types) == 2
    assert len(dev_dims) == 2

    tests = []

    ROWS, COLS = dev_dims[0], dev_dims[1]

    # iterate rows
    for row in range(dev_dims[0]):
        if dim_types[0] == TorusTopology.LINE.value:
            tests.extend(line_test(0, row, COLS, list(args.src_core), list(args.dst_core), args.ntype))
        elif dim_types[0] == TorusTopology.RING.value:
            tests.extend(ring_test(0, row, COLS, list(args.src_core), list(args.dst_core), args.ntype))
        else:
            raise ValueError("Bad Topology")

    # iterate cols
    for col in range(dev_dims[1]):
        if dim_types[1] == TorusTopology.LINE.value:
            tests.extend(line_test(1, col, ROWS, list(args.src_core), list(args.dst_core), args.ntype))
        elif dim_types[1] == TorusTopology.RING.value:
            tests.extend(ring_test(1, col, ROWS, list(args.src_core), list(args.dst_core), args.ntype))
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
    args = parser.parse_args()

    with open(args.filename, "r") as f:
        textproto_content = f.read()

    cfg = mesh_graph_descriptor_pb2.MeshGraphDescriptor()
    text_format.Parse(textproto_content, cfg)

    tests = []
    for m in cfg.mesh_descriptors:
        tests.extend(gen_tests(m.device_topology.dims, m.device_topology.dim_types))

    out = {"Tests": tests}
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
