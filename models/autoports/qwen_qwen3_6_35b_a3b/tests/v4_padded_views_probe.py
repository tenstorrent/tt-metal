# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: actual native concat/reduce adapters with padded row-zero views.

Synthetic finite operands at actual model shapes; no model acceptance claim.
No TTNN import or device access in --oracle-only mode.
"""

import argparse
import ast
import copy
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

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extracted(path):
    tree = ast.parse(path.read_text())
    methods = {
        n.name: n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "FusedDecoder"
        for n in c.body
        if isinstance(n, ast.FunctionDef)
    }
    result = {}
    for label, method in (("heads16", "_full_attention"), ("heads32", "_linear_attention")):
        fn = methods[method]
        branch = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Assign)
                and isinstance(x.value, ast.Call)
                and ast.unparse(x.value.func) == "ttnn.experimental.nlp_concat_heads_decode"
                for x in n.body
            )
        )
        index = next(
            i
            for i, n in enumerate(branch.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "ttnn.experimental.nlp_concat_heads_decode"
        )
        # Complete shard conversion, native concat, actual crop/reshape adapter.
        body = copy.deepcopy(branch.body[index - 1 : index + 2])
        spec = ast.parse(f"def {label}(self,o,shard):\n    pass").body[0]
        if label == "heads32":
            body.insert(0, ast.parse("head_shard = shard").body[0])
        spec.body = body + [ast.Return(ast.Name("merged", ast.Load()))]
        result[label] = ast.fix_missing_locations(ast.Module(body=[spec], type_ignores=[]))
        crop = ast.parse(f"def {label}_crop(merged):\n    pass").body[0]
        crop.body = [copy.deepcopy(branch.body[index + 1]), ast.Return(ast.Name("merged", ast.Load()))]
        result[label + "_crop"] = ast.fix_missing_locations(ast.Module(body=[crop], type_ignores=[]))
    fn = methods["_moe"]
    branch = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and any(isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "weighted" for x in n.body)
    )
    index = next(
        i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "weighted"
    )
    body = copy.deepcopy(branch.body[index : index + 3])
    assert [ast.unparse(n.targets[0]) for n in body] == ["weighted", "total", "routed"]
    spec = ast.parse("def moe(self,down,routing):\n    pass").body[0]
    spec.body = body + [ast.Return(ast.Name("routed", ast.Load()))]
    result["moe"] = ast.fix_missing_locations(ast.Module(body=[spec], type_ignores=[]))
    crop = ast.parse("def moe_crop(total):\n    pass").body[0]
    crop.body = [copy.deepcopy(body[-1]), ast.Return(ast.Name("routed", ast.Load()))]
    result["moe_crop"] = ast.fix_missing_locations(ast.Module(body=[crop], type_ignores=[]))
    return result


def run(args):
    torch.set_num_threads(2)
    assert not args.output.exists(), f"Preserve evidence: {args.output}"
    base = DOC / "v4_serial_l1_compact.py.snapshot"
    candidate = DOC / "v4_padded_views.py.snapshot"
    provenance = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert sha(base) == provenance["base_sha256"] and sha(candidate) == provenance["candidate_sha256"]
    extracted_sets = [extracted(p) for p in (base, candidate)]
    result = dict(
        status="cpu_setup_pass",
        diagnostic_only=True,
        probe_sha256=sha(__file__),
        base_sha256=sha(base),
        candidate_sha256=sha(candidate),
        inputs="Synthetic finite deterministic operands; actual model dimensions and extracted snapshot adapter statements",
        extracted_adapters={
            p.name: {k: ast.unparse(v) for k, v in sets.items()} for p, sets in zip((base, candidate), extracted_sets)
        },
        checks=[],
        timing=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def check(name, a, b):
        stats = compare(a, b)
        result["checks"].append(dict(name=name, **stats))
        save()
        assert stats["exact"], name

    # Row0 remains at exactly the same tile/face offsets; rows1..31 are retained padding.
    for width in (2048, 4096):
        source = torch.arange(32 * width).reshape(1, 1, 32, width)
        check(f"cpu_row0_w{width}", source[:, :, :1, :].reshape(1, 1, width), source.reshape(1, 32, width)[:, :1, :])
    if args.oracle_only:
        save()
        print("PADDED_VIEW_CPU_PASS", flush=True)
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
        params = dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768)
        result["device_open_parameters"] = params
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        obj = SimpleNamespace(compute=compute)
        modules = []
        for path, sets in zip((base, candidate), extracted_sets):
            module = runpy.run_path(str(path))
            for name, definition in sets.items():
                exec(compile(definition, str(path) + ":" + name, "exec"), module)
            modules.append(module)
        generator = torch.Generator().manual_seed(args.seed)

        def upload(value, dtype):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def read(value):
            return ttnn.to_torch(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))

        def shard(head_dim):
            return ttnn.create_sharded_memory_config(
                (32, head_dim),
                ttnn.CoreGrid(y=1, x=1),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        def timing(label, name, fn, inputs, reference):
            trace = None
            capture_open = False
            out = None
            try:
                for _ in range(args.warmup):
                    with device_only():
                        out = fn(*inputs)
                    ttnn.synchronize_device(mesh)
                    out = None
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                with device_only():
                    out = fn(*inputs)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                capture_open = False
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                check(label + "_" + name + "_first", reference, read(out))
                samples = []
                for _ in range(args.samples):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.replays):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.replays)
                check(label + "_" + name + "_last", reference, read(out))
                result["timing"].append(
                    dict(
                        case=label,
                        name=name,
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        scope="Extracted actual shard/concat/crop or multiply/reduce/crop adapter; no host conversions inside trace",
                    )
                )
                save()
            finally:
                if trace is not None:
                    if capture_open:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace)
                out = None

        for dtype_name, dtype in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
            for heads, dim in ((16, 256), (32, 128)):
                name = f"heads{heads}"
                host = torch.randn((1, 1, heads, dim), generator=generator)
                if dtype_name == "bf16":
                    host = host.bfloat16()
                x = upload(host, dtype)
                memory = shard(dim)
                outputs = [module[name](obj, x, memory) for module in modules]
                a, b = map(read, outputs)
                check(name + "_" + dtype_name + "_old_new", a, b)
                check(name + "_" + dtype_name + "_torch", host.reshape(1, 1, 4096), b)
                assert tuple(outputs[1].shape) == (1, 1, 4096) and tuple(outputs[1].padded_shape) == (1, 32, 4096)
                check(
                    name + "_" + dtype_name + "_consumer",
                    read(ttnn.multiply(outputs[0], 2.0)),
                    read(ttnn.multiply(outputs[1], 2.0)),
                )
                outputs = None
                for label, module in zip(("baseline", "candidate"), modules):
                    timing(name + "_" + dtype_name, label, module[name], (obj, x, memory), a)
                del x
            down_host = torch.randn((1, 8, 1, 2048), generator=generator)
            routing_host = torch.rand((1, 8, 1, 1), generator=generator)
            if dtype_name == "bf16":
                down_host = down_host.bfloat16()
                routing_host = routing_host.bfloat16()
            down, routing = upload(down_host, dtype), upload(routing_host, dtype)
            outputs = [module["moe"](obj, down, routing) for module in modules]
            a, b = map(read, outputs)
            check("moe_" + dtype_name + "_old_new", a, b)
            check(
                "moe_" + dtype_name + "_consumer",
                read(ttnn.multiply(outputs[0], 2.0)),
                read(ttnn.multiply(outputs[1], 2.0)),
            )
            assert tuple(outputs[1].shape) == (1, 1, 2048) and tuple(outputs[1].padded_shape) == (1, 32, 2048)
            outputs = None
            for label, module in zip(("baseline", "candidate"), modules):
                timing("moe_" + dtype_name, label, module["moe"], (obj, down, routing), a)
            del down, routing
            # Deterministic nonzero rows1..31 establish that candidate aliases rather than clears/copies padding.
            for width, name in ((4096, "heads16_crop"), (2048, "moe_crop")):
                host = torch.randn((1, 1, 32, width), generator=generator)
                host[:, :, 1:, :] += 7
                if dtype_name == "bf16":
                    host = host.bfloat16()
                x = upload(host, dtype)
                for placement in ("dram", "width_sharded"):
                    if placement == "width_sharded":
                        memory = ttnn.create_sharded_memory_config(
                            (32, width // 16),
                            ttnn.CoreGrid(y=2, x=8),
                            ttnn.ShardStrategy.WIDTH,
                            ttnn.ShardOrientation.ROW_MAJOR,
                            use_height_and_width_as_shard_shape=True,
                        )
                        source = ttnn.to_memory_config(x, memory)
                    else:
                        source = x
                    old, new = (m[name](source) for m in modules)
                    check(f"retained_{width}_{dtype_name}_{placement}", read(old), read(new))
                    assert new.buffer_address() == source.buffer_address(), "Expected metadata alias"
                    restored = ttnn.reshape(new, (1, 1, 32, width), (1, 1, 32, width))
                    check(f"retained_full_{width}_{dtype_name}_{placement}", host, read(restored))
                    check(f"unmodified_{width}_{dtype_name}_{placement}", host, read(source))
                    # Logical RM conversion must discard retained rows, not concatenate them into useful data.
                    compact = ttnn.to_layout(ttnn.to_memory_config(new, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
                    check(
                        f"consumer_rm_{width}_{dtype_name}_{placement}",
                        host[:, :, :1, :].reshape(1, 1, width),
                        ttnn.to_torch(compact),
                    )
                    del old, new, restored, compact, source
                del x
        result["status"] = "exact"
        assert sha(candidate) == result["candidate_sha256"] and sha(base) == result["base_sha256"]
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result.update(status="failed", close_error=traceback.format_exc())
        save()
    print(result["status"], args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DOC / "v4_padded_views_probe.json")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    assert min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
