"""Integrate precision-locked working-shard winners into the actual36-layer chain."""

import argparse
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import torch

import ttnn

from ..tt.full_model_policy import stage6_precision_policy
from ..tt.generator import K2Generator

OUT = Path("models/autoports/ifm_k2_horizon_7b/doc/optimized_full_model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["roles", "final"], default="roles")
    args = parser.parse_args()
    output = OUT / ("geometry_full36.json" if args.phase == "roles" else "geometry_full36_finalists.json")
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, decoder_policies={i: stage6_precision_policy(i) for i in range(36)})
        base_policies = [layer.policy for layer in gen.model.layers]
        sources = {
            dtype: json.loads((OUT / filename).read_text())["records"]
            for dtype, filename in [
                ("bfloat8_b", "geometry_bfp8_hifi2_fp32.json"),
                ("bfloat4_b", "geometry_bfp4_lofi_fp32.json"),
            ]
        }
        targets = {}
        for dtype, roles in [("bfloat8_b", ["qkv", "o", "gate", "down"]), ("bfloat4_b", ["gate"])]:
            for role in roles:
                # These choices preserve the already-loaded immutable physical
                # weights. Inertly padded candidates are recorded by the micro
                # sweep but need separate weight owners before integration.
                n = {"qkv": 1536, "o": 4096, "gate": 3072, "down": 4096}[role]
                rows = [
                    r
                    for r in sources[dtype]
                    if r["role"] == role and r.get("passed") and r["nphysical"] == n and r["fp32"]
                ]
                targets[dtype + "/" + role] = min(rows, key=lambda r: r["traced_host_us"])
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 20)[:128]
        result = {
            "layers": 36,
            "targets": targets,
            "records": [],
            "order": (
                ["baseline", "qkv", "o", "mlp", "down", "combined", "baseline", "combined"]
                if args.phase == "roles"
                else [
                    "baseline",
                    "mlp_b8",
                    "winning_k2",
                    "winning_k4",
                    "coherent8_k4",
                    "coherent8_k4",
                    "combined",
                    "baseline",
                ]
            ),
        }
        for variant in result["order"]:
            gen._release_traces()
            gen.prefill_state = None
            gen.prepared_prefills.clear()
            gen.model.pool.tensors.clear()
            for layer, base in zip(gen.model.layers, base_policies):
                updates = {}
                for role, weights, dtype in [
                    ("qkv", [layer.wqkv], base.attention),
                    ("o", [layer.wo], base.attention),
                    ("mlp", [layer.wgate, layer.wup], base.mlp),
                    ("down", [layer.wdown], base.down),
                ]:
                    geometry = getattr(base, role + "_geometry")
                    active = (
                        variant in (role, "combined")
                        or variant == "mlp_b8"
                        and role == "mlp"
                        and dtype == "bfloat8_b"
                        or variant.startswith("winning_")
                        and (role != "mlp" or dtype == "bfloat8_b")
                        or variant.startswith("coherent8")
                    )
                    if active:
                        target = targets[dtype + "/" + ("gate" if role == "mlp" else role)]
                        if variant.startswith("coherent8") and role == "mlp" and dtype == "bfloat8_b":
                            target = min(
                                (
                                    r
                                    for r in sources[dtype]
                                    if r["role"] == "gate"
                                    and r.get("passed")
                                    and r["nphysical"] == 3072
                                    and r["cores"] == 8
                                ),
                                key=lambda r: r["traced_host_us"],
                            )
                        geometry = replace(
                            geometry,
                            cores=target["cores"],
                            block_w=4 if role == "down" and variant.endswith("_k4") else target["block_w"],
                            readers=target["readers"],
                        )
                    updates[role + "_geometry"] = geometry
                    for weight in weights:
                        key = id(weight)
                        # Use logical dimensions of the selected immutable role.
                        k = {"qkv": 4096, "o": 1024, "mlp": 4096, "down": 3072}[role]
                        n = {"qkv": 1536, "o": 4096, "mlp": 3072, "down": 4096}[role]
                        layer.decode_inputs[key] = ttnn.create_sharded_memory_config(
                            (32, k // geometry.cores),
                            core_grid=ttnn.num_cores_to_corerangeset(
                                geometry.cores, mesh.compute_with_storage_grid_size(), True
                            ),
                            strategy=ttnn.ShardStrategy.WIDTH,
                            use_height_and_width_as_shard_shape=True,
                        )
                        layer.decode_programs[key] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=geometry.block_w,
                            per_core_M=1,
                            per_core_N=n // 32 // geometry.cores,
                            num_workers_per_dram_bank=geometry.readers,
                        )
                layer.policy = replace(base, **updates)
            gen.generate(prompt, 128)
            runs = []
            for _ in range(3):
                tokens = gen.generate(prompt, 128)
                runs.append(gen.last_perf.copy())
            result["records"].append(
                {
                    "name": variant,
                    "runs": runs,
                    "tokens": tokens,
                    "median_decode_ms": statistics.median(r["decode_seconds"] * 1000 / 127 for r in runs),
                    "median_ttft_ms": statistics.median(r["ttft_seconds"] * 1000 for r in runs),
                    "policies": [asdict(layer.policy) for layer in gen.model.layers],
                    "greedy_tokens_equal_baseline": not result["records"] or tokens == result["records"][0]["tokens"],
                }
            )
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("GEOMETRY", variant, result["records"][-1]["median_decode_ms"], flush=True)
        result["complete"] = True
        output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
