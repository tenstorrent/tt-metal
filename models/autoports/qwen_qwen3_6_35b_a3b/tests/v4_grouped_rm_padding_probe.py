# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic native grouped-down pack-untilize with N padding; complete actual MoE.

All four real checkpoint expert banks and the real shared expert execute. Inputs
are seeded BF16 hidden rows. Compares each bank's consumed combine output and the
complete MoE, then measures all actual operations and adapters. No TTNN import in
--oracle-only mode. Parent owns serialized physical-chip0 execution.
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_serial_rm_probe import load_snapshot

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"baseline": args.baseline_snapshot, "candidate": args.candidate_snapshot}
    proof = json.loads(args.candidate_snapshot.with_suffix(".provenance.json").read_text())
    assert digest(paths["baseline"]) == proof["base_sha256"]
    assert digest(paths["candidate"]) == proof["candidate_sha256"]
    methods = []
    for p in paths.values():
        tree = ast.parse(p.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods.append({n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)})
    assert methods[0].keys() == methods[1].keys()
    assert set(k for k in methods[0] if methods[0][k] != methods[1][k]) == {"from_state_dict", "_moe"}
    index_path = args.checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = f"model.language_model.layers.{args.layer}."
    names = [
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert_gate.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    shapes = {}
    for name in names:
        with safe_open(args.checkpoint / index[prefix + name], framework="pt", device="cpu") as f:
            shapes[name] = f.get_slice(prefix + name).get_shape()
    assert shapes["mlp.experts.down_proj"] == [256, 2048, 512]
    result = dict(
        status="cpu_setup_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        production_modified=False,
        probe_sha256=digest(__file__),
        snapshots={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        dependencies={
            str(Path(__file__).with_name(n)): digest(Path(__file__).with_name(n))
            for n in ("v4_serial_rm_probe.py", "conv_first_tap_mixed_probe.py")
        },
        checkpoint_revision=REVISION,
        checkpoint_index_sha256=digest(index_path),
        weight_shapes=shapes,
        layer=args.layer,
        seed=args.seed,
        lengths=args.lengths,
        required_grid_x=11,
        added_candidate_resident_bytes=4 * 64 * 512 * 64 * 2,
        input_description="Seeded BF16 hidden rows; checkpoint expert/router/shared weights, not captured normalized model rows",
        timing_scope="Complete actual _moe, all four banks, native routing/metadata, dispatch, matmuls, activation, RM crop or untilize, combine, shared expert and final add",
        comparison_scope="All consumed per-bank routed_combine outputs and final MoE; intentionally excludes unwritten padding rows in native grouped outputs",
        cases=[],
        timing=[],
        eager_timing=[],
        device_closed=None,
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    record(args.output)
    save()
    if args.oracle_only:
        print("CPU_GROUPED_RM_SETUP_PASS", len(args.lengths), shapes["mlp.experts.down_proj"])
        return
    for k in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(k) in (None, "", "0"), f"Clean timing requires {k} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768
        )
        result.update(status="running", device_closed=False)
        assert mesh.compute_with_storage_grid_size().x == 11, "N2112 candidate is bound to 11-column grid"
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def weight(name, bank=None):
            key = prefix + name
            with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
                return f.get_tensor(key) if bank is None else f.get_slice(key)[bank * 64 : (bank + 1) * 64]

        w = {
            "mlp.gate.weight": upload(weight("mlp.gate.weight").T, ttnn.float32),
            "mlp.shared_expert.down_proj.weight": upload(weight("mlp.shared_expert.down_proj.weight").T),
        }
        shared = [weight(name) for name in names[1:4]]
        shared[-1] = torch.nn.functional.pad(shared[-1], (0, 0, 0, 31))
        w["shared_packed"] = upload(torch.cat(shared, dim=0).T)
        del shared
        gu = []
        down = {"baseline": [], "candidate": []}
        for bank in range(4):
            host = weight("mlp.experts.gate_up_proj", bank)
            gu.append(upload(host.permute(2, 0, 1).reshape(1, 1, 2048, 64 * 1024)))
            del host
            host = weight("mlp.experts.down_proj", bank)
            down["baseline"].append(upload(host.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2048)))
            host = torch.nn.functional.pad(host, (0, 0, 0, 64))
            down["candidate"].append(upload(host.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2112)))
            del host
        combine_bias = upload(torch.zeros(1, 1, 1, 2048))
        models = {}
        for variant, path in paths.items():
            obj = load_snapshot(path, "probe_grouped_rm_" + variant).FusedDecoder()
            obj.kind = "linear_attention"
            obj.mesh_device = mesh
            obj.compute = compute
            obj.w = w
            obj.decode_linear = {}
            obj.grouped_gu = gu
            obj.grouped_down = down[variant]
            obj.combine_bias = combine_bias
            models[variant] = obj
        generator = torch.Generator().manual_seed(args.seed)
        host_x = torch.randn((1, max(args.lengths), 2048), generator=generator).bfloat16()

        def call(variant, x):
            with device_only():
                out = models[variant]._moe(x)
            assert tuple(out.shape) == (1, x.shape[1], 2048)
            assert out.dtype == ttnn.bfloat16 and out.layout == ttnn.TILE_LAYOUT
            assert out.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            return out

        def capture(variant, x):
            original = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine
            parts = []

            def wrapper(*a, **kw):
                out = original(*a, **kw)
                parts.append(out)
                return out

            ttnn.experimental.nomic_embed_text_v2_moe.routed_combine = wrapper
            try:
                out = call(variant, x)
            finally:
                ttnn.experimental.nomic_embed_text_v2_moe.routed_combine = original
            assert len(parts) == 4
            ttnn.synchronize_device(mesh)
            return ttnn.to_torch(out), [ttnn.to_torch(p) for p in parts]

        for rows in args.lengths:
            x = upload(host_x[:, :rows])
            old, old_parts = capture("baseline", x)
            new, new_parts = capture("candidate", x)
            cs = [compare(a, b) for a, b in zip(old_parts, new_parts)]
            row = dict(
                rows=rows,
                memory_branch="L1" if rows <= 256 else "DRAM",
                final=compare(old, new),
                banks=cs,
                input_immutable=compare(host_x[:, :rows], ttnn.to_torch(x)),
            )
            result["cases"].append(row)
            save()
            if not row["final"]["exact"] or not all(c["exact"] for c in cs) or not row["input_immutable"]["exact"]:
                continue
            for variant in models:
                trace = None
                capture_open = False
                out = None
                try:
                    for _ in range(args.warmup):
                        out = call(variant, x)
                        ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(args.eager_samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call(variant, x)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start))
                    last = compare(old, ttnn.to_torch(out))
                    assert last["exact"]
                    result["eager_timing"].append(
                        dict(
                            rows=rows,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            last=last,
                        )
                    )
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    out = call(variant, x)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(old, ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(old, ttnn.to_torch(out))
                    assert first["exact"] and last["exact"]
                    result["timing"].append(
                        dict(
                            rows=rows,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            first=first,
                            last=last,
                        )
                    )
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            del x
            print("GROUPED_RM_CASE_COMPLETE", rows, flush=True)
        result["device_exact"] = all(
            r["final"]["exact"] and r["input_immutable"]["exact"] and all(c["exact"] for c in r["banks"])
            for r in result["cases"]
        )
        result["status"] = "exact" if result["device_exact"] else "different"
        assert all(digest(p) == result["snapshots"][k]["sha256"] for k, p in paths.items())
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result.update(status="error", close_error=traceback.format_exc())
        save()
    print(result["status"], args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-snapshot", type=Path, default=DOC / "v4_serial_l1_compact.py.snapshot")
    p.add_argument("--candidate-snapshot", type=Path, default=DOC / "v4_grouped_rm_padding.py.snapshot")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seed", type=int, default=154)
    p.add_argument("--lengths", nargs="+", type=int, default=[33, 128, 129, 256, 4097])
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--eager-samples", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--replays", type=int, default=3)
    p.add_argument("--output", type=Path, default=DOC / "v4_grouped_rm_padding_probe.json")
    args = p.parse_args()
    assert min(args.lengths) > 1 and min(args.warmup, args.eager_samples, args.samples, args.replays) > 0
    run(args)
