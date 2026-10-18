# SPDX-License-Identifier: Apache-2.0
"""Real-weight native expert fidelity/cache control, not a model-accuracy candidate.

Run as a separate serialized device job, for example:
  timeout 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.datatype_expert_control

Only layer-0 experts 0, 1, 2 from each 128-expert chip partition are loaded.
Inputs are deterministic synthetic activations. Output metrics are API/cache
controls and must not be used to select precision or infer full-model accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/datatype_sweep/expert_device_control.json")
    args = parser.parse_args()
    # Import only when explicitly executing the device control, never on module import.
    import torch
    import ttnn
    from safetensors import safe_open

    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    checkpoint = Path(args.checkpoint).resolve()
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = "model.language_model.layers.0.mlp.experts."
    keys = {"gate_up": prefix + "gate_up_proj", "down": prefix + "down_proj"}
    report = {
        "status": "running",
        "command": shlex.join([sys.executable, "-m", "models.autoports.qwen_qwen3_8_flash_next.tests.datatype_expert_control", *sys.argv[1:]]),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hardware": "4 Blackhole p300c",
        "mesh": [1, 4],
        "regime": "synchronized native API/cache regression, eager, no performance/accuracy ranking",
        "input": "deterministic torch.randn(seed=3816+capacity), BF16 row-major; same immutable input for all arms",
        "weight_dtype": "bfloat4_b",
        "geometry": [2560, 640, 2560],
        "checkpoint": str(checkpoint),
        "checkpoint_index_sha256": sha256(index_path),
        "weight_keys": keys,
        "local_expert_ids": [0, 1, 2],
        "checkpoint_expert_ids_by_mesh_rank": [[rank * 128 + e for e in range(3)] for rank in range(4)],
        "weight_provenance": [],
        "cases": [],
        "inactive_expert": "Local expert 2 has count zero. Its output is unspecified/uninitialized and is excluded from comparisons.",
        "comparison_scope": "The common fidelity also affects the gate-product multiply, not only the three matmuls.",
        "source_sha256": sha256(__file__),
        "binary_sha256": {str(path): sha256(path) for path in (Path("build/lib/_ttnncpp.so"), Path("ttnn/ttnn/_ttnn.so"))},
        "environment": {name: os.environ.get(name) for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def tensor_hash(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    save()
    mesh = None
    try:
        # Safetensors slicing avoids materializing all 512 experts in host memory.
        host = {"gate": [], "up": [], "down": []}
        for local_id in range(3):
            expert = {"gate": [], "up": [], "down": []}
            for rank in range(4):
                global_id = rank * 128 + local_id
                values = {}
                for group, key in keys.items():
                    shard = checkpoint / index[key]
                    with safe_open(shard, framework="pt", device="cpu") as handle:
                        values[group] = handle.get_slice(key)[global_id : global_id + 1].squeeze(0).contiguous()
                    report["weight_provenance"].append({
                        "mesh_rank": rank, "local_expert": local_id, "checkpoint_expert": global_id,
                        "key": key, "file": str(shard), "file_bytes": shard.stat().st_size,
                        "slice_shape": list(values[group].shape), "source_dtype": str(values[group].dtype),
                        "source_slice_sha256": tensor_hash(values[group]),
                    })
                assert tuple(values["gate_up"].shape) == (1280, 2560)
                assert tuple(values["down"].shape) == (2560, 640)
                expert["gate"].append(values["gate_up"][:640].T.contiguous())
                expert["up"].append(values["gate_up"][640:].T.contiguous())
                expert["down"].append(values["down"].T.contiguous())
            for group in host:
                host[group].append(torch.cat(expert[group], dim=0))
        save()
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        mesh.enable_program_cache()
        report["arch"] = str(mesh.arch())
        report["device_ids"] = list(mesh.get_device_ids())
        weights = {
            group: [ttnn.from_torch(value, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)) for value in values]
            for group, values in host.items()
        }
        del host

        def upload(value, dtype=ttnn.uint32):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        configs = {
            name: ttnn.init_device_compute_kernel_config(mesh.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, name), math_approx_mode=False,
                fp32_dest_acc_en=False, packer_l1_acc=True, dst_full_sync_en=False)
            for name in ("LoFi", "HiFi2")
        }
        report["constructed_compute_configs"] = {
            name: {field: (str(getattr(config, field)) if field in ("math_fidelity", "throttle_level")
                           else bool(getattr(config, field)))
                   for field in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc",
                                 "dst_full_sync_en", "throttle_level")}
            for name, config in configs.items()
        }
        report["runtime_weight_dtypes"] = {group: [str(value.dtype) for value in values] for group, values in weights.items()}
        ids = upload(torch.arange(3, dtype=torch.int32))
        for capacity, counts_to_test in ((32, (1, 31, 32)), (128, (1, 31, 32, 33, 127, 128))):
            # Leave the first region unused so every exercised expert has a nonzero offset.
            offsets_host = torch.tensor([capacity, 2 * capacity, 3 * capacity], dtype=torch.int32)
            offsets = upload(offsets_host)
            generator = torch.Generator().manual_seed(3816 + capacity)
            input_host = (torch.randn((1, 1, 4 * capacity, 2560), generator=generator) * 0.1).bfloat16()
            input_tt = upload(input_host, ttnn.bfloat16)
            for case_index, count in enumerate(counts_to_test):
                active_counts = [count, max(1, count // 2), 0]
                counts = upload(torch.tensor(active_counts, dtype=torch.int32))
                ttnn.synchronize_device(mesh)
                case = {"capacity": capacity, "counts": active_counts, "offsets": offsets_host.tolist(),
                        "input_sha256": tensor_hash(input_host), "arms": [], "status": "running"}
                report["cases"].append(case)
                save()
                outputs = []
                for arm_index, name in enumerate(("LoFi", "HiFi2", "LoFi")):
                    before = mesh.num_program_cache_entries()
                    tick = time.perf_counter()
                    result = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                        input_tt, offsets, counts, ids, weights["gate"], weights["up"], weights["down"], capacity,
                        compute_kernel_config=configs[name])
                    ttnn.synchronize_device(mesh)
                    after = mesh.num_program_cache_entries()
                    # Read and compare only valid rows; unused output memory is not initialized by the native op.
                    active = []
                    for shard in ttnn.get_device_tensors(result):
                        rows = ttnn.to_torch(shard).reshape(-1, 2560)
                        active.append(torch.cat([rows[int(offsets_host[e]) : int(offsets_host[e]) + n]
                                                 for e, n in enumerate(active_counts) if n], dim=0).clone())
                    ttnn.synchronize_device(mesh)
                    expected_delta = 1 if case_index == 0 and arm_index < 2 else 0
                    arm = {"fidelity": name, "cache_entries_before": before, "cache_entries_after_sync": after,
                           "cache_entry_delta": after - before, "expected_cache_entry_delta": expected_delta,
                           "cache_entries_after_readback": mesh.num_program_cache_entries(),
                           "completion": "synchronize, read all four device outputs, synchronize",
                           "completed_wall_seconds_including_compile_and_readback": time.perf_counter() - tick,
                           "active_output_sha256_by_mesh_rank": [tensor_hash(value) for value in active],
                           "finite_active_outputs": all(bool(torch.isfinite(value).all()) for value in active)}
                    case["arms"].append(arm)
                    save()
                    assert after - before == expected_delta, arm
                    assert arm["finite_active_outputs"], arm
                    outputs.append(active)
                    ttnn.deallocate(result)
                case["repeat_lofi_exact_by_mesh_rank"] = [torch.equal(a, b) for a, b in zip(outputs[0], outputs[2])]
                case["lofi_vs_hifi2_max_abs_by_mesh_rank"] = [float((a.float() - b.float()).abs().max())
                                                            for a, b in zip(outputs[0], outputs[1])]
                case["lofi_vs_hifi2_exact_by_mesh_rank"] = [torch.equal(a, b) for a, b in zip(outputs[0], outputs[1])]
                assert all(case["repeat_lofi_exact_by_mesh_rank"]), case
                case["status"] = "pass"
                save()
                ttnn.deallocate(counts)
            # ROW_MAJOR input must stay immutable across every arm and count at this capacity.
            ttnn.synchronize_device(mesh)
            input_matches = [torch.equal(ttnn.to_torch(shard), input_host) for shard in ttnn.get_device_tensors(input_tt)]
            report.setdefault("input_unchanged_by_capacity", {})[str(capacity)] = input_matches
            assert all(input_matches), input_matches
            ttnn.deallocate(input_tt)
            ttnn.deallocate(offsets)
        report["final_program_cache_entries"] = mesh.num_program_cache_entries()
        report["status"] = "pass"
        save()
        print("EXPERT_DEVICE_CONTROL_PASS", args.output, flush=True)
    except Exception as error:
        report.update(status="fail", error=repr(error))
        save()
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
