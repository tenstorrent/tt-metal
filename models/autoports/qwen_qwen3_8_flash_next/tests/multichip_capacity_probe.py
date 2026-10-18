"""Allocate the planned TP4 full-stack payload with an explicit reserve.

This tests addressable capacity, not full-model execution or tensor correctness.
"""

import json, math, os
from pathlib import Path
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_memory_plan import plan


def view(mesh, buffer_type=ttnn.BufferType.DRAM):
    v = ttnn.get_memory_view(mesh, buffer_type)
    return {
        k: getattr(v, k)
        for k in (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "total_bytes_free_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
    }


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    result = dict(plan=plan(), initial=view(m), records=[])
    held = []
    try:
        groups = {
            **result["plan"]["weight_breakdown"],
            **result["plan"]["cache_breakdown"],
            "activation_trace_reserve": result["plan"]["reserved_trace_activation_bytes_per_device"],
        }
        for name, count in groups.items():
            # Small allocations bound the largest single row and count bank
            # padding conservatively. No payload initialization or weight loads.
            remaining = count
            while remaining:
                size = min(remaining, 256 * 1024**2)
                held.append(
                    ttnn.empty(
                        (1, 1, math.ceil(size / 2048), 1024),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=m,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                remaining -= size
            result["records"].append(dict(name=name, bytes=count, memory=view(m)))
        ttnn.synchronize_device(m)
        result["final"] = view(m)
        result["status"] = "capacity_pass"
        print(json.dumps(result), flush=True)
        (Path("models/autoports/qwen_qwen3_8_flash_next") / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder") / "capacity_probe.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    finally:
        held.clear()
        ttnn.close_mesh_device(m)


if __name__ == "__main__":
    main()
