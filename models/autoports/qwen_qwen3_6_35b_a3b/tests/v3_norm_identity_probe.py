# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: frozen v3 actual norm method, conditional identity-cast removal.

CPU setup is available without importing TTNN. Device execution is root-owned.
"""

import argparse
import ast
import hashlib
import json
import os
import runpy
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    assert args.overwrite or not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline": DOC / "v3_movement_baseline.py.snapshot",
        "candidate": DOC / "v3_norm_identity_cast.py.snapshot",
    }
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights, methods = {}, {}
    roles = [
        (0, "post_attention_layernorm.weight"),
        (3, "input_layernorm.weight"),
        (3, "post_attention_layernorm.weight"),
        (3, "self_attn.q_norm.weight"),
        (3, "self_attn.k_norm.weight"),
    ]
    for layer, key in roles:
        name = f"model.language_model.layers.{layer}.{key}"
        with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as f:
            weights[layer, key] = (f.get_tensor(name).float() + 1).reshape(1, 1, -1)
    for variant, path in paths.items():
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods[variant] = {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {k for k in methods["baseline"] if methods["baseline"][k] != methods["candidate"][k]} == {"norm"}
    # The accuracy-sensitive linear-attention inputnorm branch must remain byte-equivalent as an AST.
    norms = []
    for path in paths.values():
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        norms.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "norm"))
    assert ast.dump(norms[0].body[0]) == ast.dump(norms[1].body[0])
    cases = []
    for layer, key in roles:
        heads = 16 if ".q_norm." in key else 2
        for rows, layout in ((1, "interleaved"), (1, "width_sharded"), (33, "interleaved")):
            shape = (
                ((1, 1, heads, 256) if rows == 1 else (1, heads, rows, 256)) if "self_attn" in key else (1, rows, 2048)
            )
            for dtype in ("bfloat16", "float32"):
                cases.append(dict(layer=layer, key=key, rows=rows, layout=layout, dtype=dtype, shape=shape))
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=digest(Path(__file__)),
        dependency_sha256=digest(Path(__file__).with_name("conv_first_tap_mixed_probe.py")),
        snapshots={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        seed=args.seed,
        input_description="Seeded BF16 or genuine FP32 activations; checkpoint layer-0/3 norm weights uploaded as FP32 (1+weight), matching setup.",
        timing_scope="Complete adapter: optional DRAM-to-width-sharded input, actual frozen norm method (including native norm, output-memory selection, and required final cast), output-to-DRAM. Common conversions remain timed.",
        planned_cases=cases,
        cases=[],
        timing=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        save()
        print("CPU_NORM_SETUP_PASS", len(cases))
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), f"Clean timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        dweights = {k: upload(v, ttnn.float32) for k, v in weights.items()}
        functions = {k: runpy.run_path(str(p))["FusedDecoder"].norm for k, p in paths.items()}
        cores = ttnn.num_cores_to_corerangeset(32, mesh.compute_with_storage_grid_size(), row_wise=True)
        model_shard = ttnn.create_sharded_memory_config(
            (32, 64),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        head_shard = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for number, case in enumerate(cases):
            host = torch.randn(case["shape"], generator=torch.Generator().manual_seed(args.seed + number))
            if case["dtype"] == "bfloat16":
                host = host.bfloat16()
            dx = upload(host, getattr(ttnn, case["dtype"]))
            obj = SimpleNamespace(
                kind="linear_attention" if case["layer"] == 0 else "full_attention",
                mesh_device=mesh,
                compute=compute,
                config=SimpleNamespace(rms_norm_eps=config["rms_norm_eps"]),
                w={case["key"]: dweights[case["layer"], case["key"]]},
            )
            memory = head_shard if "self_attn" in case["key"] else model_shard

            def call(variant):
                x = ttnn.to_memory_config(dx, memory) if case["layout"] == "width_sharded" else dx
                out = functions[variant](obj, x, case["key"])
                assert out.dtype == ttnn.bfloat16
                return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

            outputs, native = {}, {}
            original = ttnn.rms_norm
            for variant in paths:
                seen = []

                def capture(*a, **kw):
                    out = original(*a, **kw)
                    seen.append(out)
                    return out

                try:
                    ttnn.rms_norm = capture
                    with device_only():
                        out = call(variant)
                finally:
                    ttnn.rms_norm = original
                assert len(seen) == 1
                assert seen[0].dtype == getattr(ttnn, case["dtype"])
                outputs[variant] = ttnn.to_torch(out)
                native[variant] = ttnn.to_torch(seen[0])
            row = dict(
                case=case,
                output=compare(outputs["baseline"], outputs["candidate"]),
                native_output=compare(native["baseline"], native["candidate"]),
                native_dtype=case["dtype"],
                returned_dtype="bfloat16",
                input_preserved=compare(host, ttnn.to_torch(dx)),
            )
            result["cases"].append(row)
            save()
            if not all(row[k]["exact"] for k in ("output", "native_output", "input_preserved")):
                continue
            for variant in paths:
                trace, opened, out = None, False, None
                try:
                    for _ in range(args.warmup):
                        with device_only():
                            out = call(variant)
                        ttnn.synchronize_device(mesh)
                    del out
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    opened = True
                    with device_only():
                        out = call(variant)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    opened = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(outputs["baseline"], ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(outputs["baseline"], ttnn.to_torch(out))
                    result["timing"].append(
                        dict(
                            case_index=number,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first=first,
                            last=last,
                        )
                    )
                    assert first["exact"] and last["exact"]
                finally:
                    if trace is not None:
                        if opened:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    save()
            print("NORM_CASE_COMPLETE", number, case, flush=True)
        result["device_exact"] = all(
            all(r[k]["exact"] for k in ("output", "native_output", "input_preserved")) for r in result["cases"]
        )
        result["status"] = "exact" if result["device_exact"] else "different"
        assert all(digest(paths[k]) == v["sha256"] for k, v in result["snapshots"].items())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(result["status"], args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--seed", type=int, default=154)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", type=Path, default=DOC / "v3_norm_identity_probe.json")
    args = p.parse_args()
    assert min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
