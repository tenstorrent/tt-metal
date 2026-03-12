# Quick debug script to isolate where g.build() fails
import math
import sys
import traceback

sys.path.insert(0, "/localdev/rmiller/tt-metal")

import ttnn
from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def debug_build():
    width = 7168
    epsilon = 1e-6
    shape = (1, width)
    tile = ttnn.Tile([1, 32])
    FULL_32x32_TILE = ttnn.Tile((32, 32))
    HALF_16x32_TILE = ttnn.Tile((16, 32))
    is_16x32 = (width // 32) % 32 != 0
    interpreted_tile = HALF_16x32_TILE if is_16x32 else FULL_32x32_TILE
    tile_height, tile_width = interpreted_tile.tile_shape
    num_tiles = (shape[0] * shape[1]) // (tile_height * tile_width)
    numel = shape[0] * shape[1]

    print(f"width={width}, num_tiles={num_tiles}, is_16x32={is_16x32}")
    print(f"tile_height={tile_height}, tile_width={tile_width}")

    # Step 1: Build FusionGraph and compile (no device needed)
    print("\n=== Step 1: FusionGraph.compile() ===")
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    g = FusionGraph()
    g.add(
        "rmsnorm",
        RMSNORM,
        cores=core_grid,
        ct_args={
            "fp32_acc": 0,
            "num_tiles": num_tiles,
            "rsqrt_fast_approx": 0,
            "input_num_pages": num_tiles,
            "gamma_num_pages": num_tiles,
            "epsilon": float_to_uint32(epsilon),
            "scalar": float_to_uint32(1.0 / math.sqrt(float(numel))),
        },
    )

    try:
        external_ports = {("rmsnorm", "input"), ("rmsnorm", "gamma"), ("rmsnorm", "output")}
        source, schedule, allocator = g.compile(external_ports=external_ports)
        print(f"Schedule: {schedule}")
        print(f"Allocations:")
        print(allocator.get_liveness_summary())
        print(f"\nGenerated kernel source ({len(source)} chars):")
        print("=" * 60)
        print(source)
        print("=" * 60)
    except Exception as e:
        traceback.print_exc()
        return

    # Step 2: Check compile-time args
    print("\n=== Step 2: Check what host_gen would produce ===")
    from models.demos.deepseek_v3_b1.auto_fusion.host_gen import HostGenerator

    # Create fake io_tensors to test without device
    # We just want to trace through the CT arg generation
    nodes = {n.id: n for n in g.nodes}
    for nid in schedule:
        node = nodes[nid]
        print(f"\nNode {nid} CB bindings: {node.cb_bindings}")
        print(f"Node {nid} CT args: {node.ct_args}")

    print("\n=== Step 3: Build CT args (dry run) ===")
    try:
        host_gen = HostGenerator.__new__(HostGenerator)
        host_gen._graph = g
        host_gen._schedule = schedule
        host_gen._allocator = allocator
        host_gen._cb_allocs = allocator._allocations
        host_gen._cb_configs = {}
        host_gen._nodes = nodes
        host_gen._pool_tensors = []

        ncrisc_ct, brisc_ct, trisc_ct = host_gen._build_compile_time_args()
        print(f"NCRISC CT args: {ncrisc_ct}")
        print(f"BRISC CT args: {brisc_ct}")
        print(f"TRISC CT args: {trisc_ct}")

        trisc_common_rt = host_gen._build_trisc_common_runtime_args()
        print(f"TRISC common RT args: {trisc_common_rt}")

        core_descs = host_gen._build_core_descriptors()
        print(f"Core descriptors: {len(core_descs)} entries")

        compute_config = host_gen._build_compute_config()
        print(f"Compute config: {compute_config}")
    except Exception as e:
        traceback.print_exc()
        return

    print("\n=== DONE (no device needed for these checks) ===")


if __name__ == "__main__":
    debug_build()
