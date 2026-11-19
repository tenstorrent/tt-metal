# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: actual prefill projection consumers and native gated norm at a smaller gate stride.

The measured adapter includes all extracted concat/slices, native norm and final
Z multiply. Projections, recurrence and output matmul are outside this isolated
comparison. --oracle-only never imports TTNN. Root owns hardware execution.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter(source, native):
    """Extract the actual consumer construction and native gated-output statements."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")

    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, n in enumerate(method.body) if assignment(n, "output_projected")) + 1
    end = next(i for i, n in enumerate(method.body) if assignment(n, "history"))
    packing = copy.deepcopy(method.body[start:end])
    branch = next(
        n for n in method.body if isinstance(n, ast.If) and any(assignment(child, "gated") for child in n.body)
    )
    norm = copy.deepcopy([n for n in branch.body if assignment(n, "gated") or assignment(n, "merged")])
    assert len(norm) == 2
    gate_name = norm[0].value.args[1].id
    offset = next(k.value.value for k in norm[0].value.keywords if k.arg == "gate_column_offset")
    fn = ast.parse(
        "def extracted(self, state_projected, output_projected, head_first, do_norm):\n"
        "    decode = False\n"
        "    gated = merged = None\n"
        "    if do_norm:\n"
        "        pass\n"
        "    return (qkv, projection_a, projection_b, projection_z, gated, merged)\n"
    ).body[0]
    conditional = fn.body[2]
    conditional.body = norm
    fn.body = fn.body[:2] + packing + [conditional, fn.body[-1]]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"ttnn": native}
    exec(compile(module, str(source) + ":actual_consumer_adapter", "exec"), namespace)
    return namespace["extracted"], {
        "source_sha256": digest(source),
        "adapter_ast_sha256": hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        "gate_operand": gate_name,
        "gate_column_offset": offset,
    }


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), f"Preserve existing evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"baseline": DOC / "selected_fused_decoder_v3.py.snapshot", "candidate": args.snapshot}
    cpu_ops = SimpleNamespace(concat=lambda tensors, dim: torch.cat(tensors, dim=dim))
    cpu = {name: adapter(path, cpu_ops) for name, path in paths.items()}
    # Unique INT32 labels prove channel and token mappings without BF16 collisions.
    mapping = []
    for rows in args.lengths:
        state = torch.arange(rows * 6208, dtype=torch.int32).reshape(1, rows, 6208)
        output = torch.arange(rows * 6144, dtype=torch.int32).reshape(1, rows, 6144) + rows * 6208
        old, new = [cpu[v][0](None, state, output, None, False) for v in ("baseline", "candidate")]
        checks = {name: bool(torch.equal(a, b)) for name, a, b in zip(("qkv", "a", "b", "z"), old[:4], new[:4])}
        packed = torch.cat([output[:, :, :2048], state[:, :, :6144], output[:, :, 2048:], state[:, :, 6144:]], -1)
        checks["native_gate_columns"] = bool(torch.equal(packed[:, :, 8192:12288], output[:, :, 2048:6144]))
        assert all(checks.values())
        mapping.append(dict(rows=rows, checks=checks))
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.linear_attn.norm.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as handle:
        gamma = handle.get_tensor(key)
    assert gamma.shape == (128,) and torch.isfinite(gamma).all()
    eps = json.loads((args.checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    result = dict(
        status="cpu_mapping_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        production_modified=False,
        probe_sha256=digest(Path(__file__)),
        snapshots={name: dict(path=str(paths[name]), **data[1]) for name, data in cpu.items()},
        checkpoint_revision=REVISION,
        gamma_sha256=hashlib.sha256(gamma.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        cpu_mapping=mapping,
        fixtures="Unique integer CPU mapping; deterministic finite BF16 synthetic projected operands and FP32 native core, real checkpoint gamma",
        timing_scope="Actual extracted consumer concat/slices + native gated norm + BF16 final Z multiply; excludes projection/recurrence/output matmul",
        logical_tail_scope="Non-aligned rows exercise consumer mapping only; native norm requires aligned physical rows, as in existing public planner",
        cases=[],
        timing=[],
        device_closed=None,
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        print("CPU_DIRECT_PREFILL_MAPPING_PASS", len(mapping))
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), f"Clean timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024)
        result.update(status="running", device_closed=False)

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        obj = SimpleNamespace(
            config=SimpleNamespace(rms_norm_eps=eps),
            output_norm_bf16=upload(gamma),
            compute=ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        functions = {name: adapter(path, ttnn)[0] for name, path in paths.items()}
        generator = torch.Generator().manual_seed(args.seed)
        names = ("qkv", "a", "b", "z", "native_gated_norm", "merged")
        for rows in args.lengths:
            state_host = torch.randn((1, rows, 6208), generator=generator).bfloat16()
            output_host = (3 * torch.randn((1, rows, 6144), generator=generator)).bfloat16()
            core_host = torch.randn((32, rows, 128), generator=generator)
            state, output = upload(state_host), upload(output_host)
            core = upload(core_host, ttnn.float32)
            do_norm = rows % 32 == 0
            baseline = None
            outputs = {}
            for variant in ("baseline", "candidate"):
                with device_only():
                    tensors = functions[variant](obj, state, output, core, do_norm)
                outputs[variant] = tuple(ttnn.to_torch(t) if t is not None else None for t in tensors)
                del tensors
            checks = {
                name: compare(a, b)
                for name, a, b in zip(names, outputs["baseline"], outputs["candidate"])
                if a is not None
            }
            checks.update(
                {
                    "a_vs_supplied": compare(state_host[:, :, 6144:6176], outputs["candidate"][1]),
                    "b_vs_supplied": compare(state_host[:, :, 6176:6208], outputs["candidate"][2]),
                    "z_vs_supplied": compare(output_host[:, :, 2048:], outputs["candidate"][3]),
                }
            )
            row = dict(rows=rows, native_norm_exercised=do_norm, checks=checks)
            result["cases"].append(row)
            save()
            if not all(c["exact"] for c in checks.values()) or not do_norm:
                row["timing_skipped"] = (
                    "Code difference"
                    if do_norm
                    else "Consumer mapping only; native norm requires tile-aligned physical sequence"
                )
                continue
            baseline = outputs["baseline"]
            for variant in ("baseline", "candidate"):
                trace, capture_open, tensors = None, False, None
                try:
                    for _ in range(args.warmup):
                        with device_only():
                            tensors = functions[variant](obj, state, output, core, True)
                        ttnn.synchronize_device(mesh)
                    tensors = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        tensors = functions[variant](obj, state, output, core, True)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = {name: compare(a, ttnn.to_torch(t)) for name, a, t in zip(names, baseline, tensors)}
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = {name: compare(a, ttnn.to_torch(t)) for name, a, t in zip(names, baseline, tensors)}
                    result["timing"].append(
                        dict(
                            rows=rows,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first=first,
                            last=last,
                        )
                    )
                    assert all(c["exact"] for c in list(first.values()) + list(last.values()))
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    tensors = None
                    save()
            del state, output, core, outputs
            print("DIRECT_PREFILL_CASE_COMPLETE", rows, flush=True)
        result["device_exact"] = all(c["exact"] for row in result["cases"] for c in row["checks"].values())
        result["status"] = "exact" if result["device_exact"] else "different"
        assert all(digest(path) == result["snapshots"][name]["source_sha256"] for name, path in paths.items())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()
    print(result["status"], args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--snapshot", type=Path, default=DOC / "endurance/v3_direct_prefill_consumers.py.snapshot")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 17, 32, 33, 128, 129, 256])
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "endurance/v3_direct_prefill_probe.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
