# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic current-source decode norm adapters; model PCC gates remain unchanged.

The probe reports arithmetic differences without rejecting on code equality.
Exact assertions cover layout/padding independence and trace determinism only.
--oracle-only performs source checks without importing TTNN or accessing devices.
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
EXP = DOC / "endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path, native=None):
    tree = ast.parse(path.read_text())
    method = next(
        n
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "FusedDecoder"
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention"
    )
    index = (
        next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "not decode"
            and any(isinstance(child, ast.Return) for child in n.body)
        )
        + 1
    )
    body = copy.deepcopy(method.body[index:])
    assert isinstance(body[-1], ast.Return) and ast.dump(body[-1].value) == ast.dump(
        ast.parse('self.linear(merged, "linear_attn.out_proj.weight")', mode="eval").body
    )
    body[-1] = ast.Return(ast.Name("merged", ast.Load()))
    fn = ast.parse("def adapter(self, o, output_projected, projection_z):\n    s = 1\n    decode = True\n").body[0]
    fn.body += body
    module = ast.fix_missing_locations(ast.Module([fn], []))
    metadata = dict(
        path=str(path), sha256=sha(path), adapter_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest()
    )
    if native is None:
        compile(module, str(path), "exec")
        return None, metadata
    namespace = {
        "ttnn": native,
        "reshape": lambda value, shape: native.to_layout(native.reshape(value, shape), native.TILE_LAYOUT),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["adapter"], metadata


def metrics(a, b):
    value = compare(a, b)
    af, bf = a.flatten().double(), b.flatten().double()
    value["pcc"] = float(torch.corrcoef(torch.stack((af, bf)))[0, 1])
    value["relative_l2"] = float(torch.linalg.vector_norm(af - bf) / torch.linalg.vector_norm(af))
    return value


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), "Preserve prior evidence with a fresh output filename"
    paths = {"baseline": args.baseline}
    for geometry in ("heads", "rows"):
        for dtype in ("bf16", "fp32"):
            paths[f"{geometry}_{dtype}"] = EXP / f"v4_decode_native_norm_{geometry}_{dtype}.py.snapshot"
    proof = json.loads((EXP / "v4_decode_native_norm_source_proof.json").read_text())
    assert sha(paths["baseline"]) == proof["baseline_sha256"]
    for label, path in paths.items():
        if label != "baseline":
            assert sha(path) == proof["variants"][label]["sha256"]
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.linear_attn.norm.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as handle:
        gamma = handle.get_tensor(key)
    eps = json.loads((args.checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    result = dict(
        status="cpu_source_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        probe_sha256=sha(Path(__file__)),
        snapshots={k: extract(p)[1] for k, p in paths.items()},
        checkpoint_revision=REVISION,
        gamma_sha256=hashlib.sha256(gamma.view(torch.uint8).numpy().tobytes()).hexdigest(),
        acceptance="Code differences are reported, not a model-quality failure. Existing whole-decoder PCC/stress gates decide.",
        timing_scope="Complete extracted decode norm+gate+head packing adapter, including all casts/layout changes; excludes recurrence, setup and final output projection",
        cases=[],
        timing=[],
        padding=[],
        device_closed=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        print("CPU_SOURCE_PASS", len(paths))
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), key
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024)
        result.update(status="running", device_closed=False)

        def upload(value, dtype=ttnn.bfloat16):
            host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, preserve_nan_values=True)
            return ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        round_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        obj = SimpleNamespace(
            config=SimpleNamespace(rms_norm_eps=eps),
            output_norm_bf16=upload(gamma),
            w={"linear_attn.norm.weight": upload(gamma.float().reshape(1, 1, 128), ttnn.float32)},
            round_widen_chain=[round_op, widen_op],
            compute=ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        funcs = {label: extract(path, ttnn)[0] for label, path in paths.items()}
        for seed in (42, 154, 913):
            gen = torch.Generator().manual_seed(seed)
            host_core = torch.randn((1, 1, 32, 128), generator=gen)
            host_projected = (3 * torch.randn((1, 1, 6144), generator=gen)).bfloat16()
            core = ttnn.to_memory_config(upload(host_core, ttnn.float32), ttnn.L1_MEMORY_CONFIG)
            projected = upload(host_projected)
            z = projected[:, :, 2048:]
            outputs = {}
            for label, fn in funcs.items():
                with device_only():
                    output = fn(obj, core, projected, z)
                assert tuple(output.shape) == (1, 1, 4096) and output.dtype == ttnn.bfloat16
                outputs[label] = ttnn.to_torch(output)
                assert torch.isfinite(outputs[label]).all()
                result["cases"].append(
                    dict(seed=seed, variant=label, vs_current=metrics(outputs["baseline"], outputs[label]))
                )
                save()
                if seed != 154:
                    continue
                trace, capturing = None, False
                try:
                    for _ in range(args.warmup):
                        output = fn(obj, core, projected, z)
                        ttnn.synchronize_device(mesh)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        output = fn(obj, core, projected, z)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capturing = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    repeat = compare(outputs[label], ttnn.to_torch(output))
                    assert repeat["exact"], "Trace must preserve the candidate's own output"
                    result["timing"].append(
                        dict(
                            seed=seed,
                            variant=label,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            own_replay=repeat,
                        )
                    )
                finally:
                    if trace is not None:
                        if capturing:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    save()
            # Independently fill the true-head physical rows: consumed row zero
            # must match the actual expanded metadata adapter for both policies.
            for fill in (17.0, -23.0, float("nan")):
                hc = torch.full((32, 32, 128), fill, dtype=torch.bfloat16)
                hg = torch.full((1, 32, 6144), fill, dtype=torch.bfloat16)
                hc[:, 0] = host_core.reshape(32, 128).bfloat16()
                hg[:, :1] = host_projected
                dc, dg = upload(hc), upload(hg)
                observed = ttnn.to_torch(dc)[:, 1:]
                observed_gate = ttnn.to_torch(dg)[:, 1:]
                injected = (
                    bool(torch.isnan(observed).all() and torch.isnan(observed_gate).all())
                    if fill != fill
                    else bool((observed == fill).all() and (observed_gate == fill).all())
                )
                assert injected, "Sentinel injection must be verified"
                for policy in ("bf16", "fp32"):
                    dtype = ttnn.bfloat16 if policy == "bf16" else ttnn.float32
                    out = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                        dc,
                        dg,
                        obj.output_norm_bf16,
                        32,
                        epsilon=eps,
                        output_dtype=dtype,
                        gate_column_offset=2048,
                        compute_kernel_config=obj.compute,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    out = ttnn.reshape(out, (1, 1, 4096), (1, 32, 4096))
                    out = ttnn.multiply(
                        out, ttnn.typecast(z, ttnn.float32), dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
                    )
                    check = compare(outputs[f"heads_{policy}"], ttnn.to_torch(out))
                    result["padding"].append(
                        dict(
                            seed=seed,
                            fill="nan" if fill != fill else fill,
                            variant=f"heads_{policy}",
                            injection_verified=injected,
                            consumed_output=check,
                        )
                    )
                    save()
                    assert check["exact"], "Unused rows must not alter consumed output"
            print("NATIVE_NORM_ADAPTER_COMPLETE", seed, flush=True)
        assert all(sha(p) == result["snapshots"][k]["sha256"] for k, p in paths.items())
        result["status"] = "diagnostic_pass_model_quality_pending"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DOC / "selected_fused_decoder_v4.py.snapshot")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
