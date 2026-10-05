"""Estimate, or explicitly allocate, the TP4 full-stack capacity envelope.

The default mode imports only the standard library and never touches devices.
--allocate reserves empty buffers on TP4; it does not instantiate/run 36 layers,
initialize KV, validate full-model accuracy, or prove runtime peak memory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def tile_bytes(shape, dtype):
    sizes = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}
    padded = list(shape)
    padded[-2:] = [math.ceil(n / 32) * 32 for n in padded[-2:]]
    return math.prod(padded) // 1024 * sizes[dtype]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="HF config.json; supplies real vocabulary and dimensions")
    parser.add_argument("--vocab-size", type=int, help="Required if --config is omitted")
    parser.add_argument("--allocate", action="store_true", help="Open TP4 and reserve all buffers concurrently")
    parser.add_argument("--workspace-gib", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    vocab = config.get("vocab_size", args.vocab_size)
    if not vocab or vocab < 1:
        parser.error("Provide the checkpoint's --config or --vocab-size")
    layers = config.get("num_hidden_layers", 36)
    context = config.get("max_position_embeddings", 524288)
    hidden = config.get("hidden_size", 4096)
    if (layers, context, hidden) != (36, 524288, 4096):
        parser.error("This probe expects the real 36-layer, 524288-context, hidden4096 model")
    source = json.loads(args.weights_json.read_text())
    weights = source["measurements"]["multichip"]["weight_allocations"]
    allocations = []
    for layer in range(layers):
        for index, (role, mode, shape, dtype) in enumerate(weights):
            allocations.append(
                {
                    "name": f"layer{layer}.{role}.{mode}.{index}",
                    "shape": shape,
                    "dtype": dtype,
                    "dram_sharded": mode == "decode",
                }
            )
        for kind in ("k", "v"):
            allocations.append(
                {
                    "name": f"layer{layer}.{kind}",
                    "shape": [context // 32, 2, 32, 128],
                    "dtype": "bfloat8_b",
                    "dram_sharded": False,
                }
            )
    # Conservative replicated BF16 embedding AND head, even for a tied/TP head.
    for name in ("embedding_allowance", "lm_head_allowance"):
        allocations.append({"name": name, "shape": [vocab, hidden], "dtype": "bfloat16", "dram_sharded": False})
    allocations.append(
        {"name": "final_norm_allowance", "shape": [32, hidden], "dtype": "bfloat16", "dram_sharded": False}
    )
    # Half-GiB allocations avoid requiring a single huge addressable buffer.
    for index in range(args.workspace_gib * 2):
        allocations.append(
            {"name": f"workspace_allowance{index}", "shape": [8192, 32768], "dtype": "bfloat16", "dram_sharded": False}
        )
    for allocation in allocations:
        allocation["bytes_per_device"] = tile_bytes(allocation["shape"], allocation["dtype"])
    evidence = {
        "scope": "allocation envelope only; no model execution or initialized-cache claim",
        "completed": False,
        "allocated_on_device": False,
        "layers": layers,
        "context": context,
        "vocab_size": vocab,
        "embedding_head_policy": "two separate replicated BF16 matrices",
        "workspace_bytes_per_device": args.workspace_gib * 2**30,
        "weight_bytes_per_layer_per_device": sum(tile_bytes(shape, dtype) for _, _, shape, dtype in weights),
        "expected_tensor_bytes_per_device": sum(a["bytes_per_device"] for a in allocations),
        "page_table_bytes_per_device": context // 32 * 4,
        "expected_total_bytes_per_device": sum(a["bytes_per_device"] for a in allocations) + context // 32 * 4,
        "weights_evidence_sha256": hashlib.sha256(args.weights_json.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "allocations": allocations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.allocate:
        import ttnn

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        held = []
        try:
            banks = mesh.dram_grid_size()
            if banks.x != 8 or banks.y != 1:
                raise ValueError(f"Expected the saved eight-bank geometry, got {banks}")
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
            for allocation in allocations:
                memory = ttnn.DRAM_MEMORY_CONFIG
                if allocation["dram_sharded"]:
                    k, n = allocation["shape"]
                    memory = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(grid, (k, n // 8), ttnn.ShardOrientation.ROW_MAJOR),
                    )
                held.append(
                    ttnn.empty(
                        allocation["shape"],
                        dtype=getattr(ttnn, allocation["dtype"]),
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=memory,
                    )
                )
                evidence["last_allocated"] = allocation["name"]
            held.append(
                ttnn.empty(
                    [1, context // 32],
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            ttnn.synchronize_device(mesh)
            evidence.update(
                completed=True,
                allocated_on_device=True,
                mesh_ids=mesh.get_device_ids(),
                concurrently_held_buffers=len(held),
            )
        finally:
            args.output.write_text(json.dumps(evidence, indent=2) + "\n")
            held.clear()
            ttnn.close_mesh_device(mesh)
    else:
        evidence["completed"] = True
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "allocations"}, indent=2))


if __name__ == "__main__":
    main()
