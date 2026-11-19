# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic direct slice of native sharded router outputs; all adapters timed.

Three seeded checkpoint-derived logit cases plus a captured real seed154 router
case. The common native gate is outside adapter timing; full decoder timing is
separate. No TTNN import or device execution in --oracle-only mode.
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
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_data_movement_probe import router_function

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
    old, new = [ast.parse(p.read_text()) for p in paths.values()]
    methods = []
    for tree in (old, new):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods.append({n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)})
    assert methods[0].keys() == methods[1].keys()
    assert [k for k in methods[0] if methods[0][k] != methods[1][k]] == ["_moe"]
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.0.mlp.gate.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key).bfloat16()
    fixtures = []
    for seed in (42, 154, 913):
        hidden = torch.randn((1, 2048), generator=torch.Generator().manual_seed(seed)).bfloat16()
        logits = torch.nn.functional.linear(hidden, weight).reshape(1, 16, 16)
        fixtures.append((f"checkpoint_seed{seed}", logits, "Checkpoint-derived CPU BF16 logits from seeded hidden row"))
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    fixtures.append(
        (
            "captured_seed154_pos5991",
            capture["tt_values"]["router"].reshape(1, 16, 16),
            "Captured actual TT router logits from historical seed154 position5991",
        )
    )
    result = dict(
        status="cpu_setup_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        production_modified=False,
        probe_sha256=digest(__file__),
        snapshots={
            k: dict(path=str(p), sha256=digest(p), adapter_ast_sha256=router_function(p, None)[1])
            for k, p in paths.items()
        },
        dependencies={
            str(Path(__file__).with_name(n)): digest(Path(__file__).with_name(n))
            for n in ("v3_data_movement_probe.py", "conv_first_tap_mixed_probe.py")
        },
        capture=dict(path=str(args.capture), sha256=digest(args.capture)),
        checkpoint_revision=REVISION,
        fixtures=[
            dict(name=n, description=d, logits_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
            for n, x, d in fixtures
        ],
        timing_scope="All actual post-native-gate score/id adapters through TILE routing and RM sparse ids, including direct slice and all reshapes/layouts. Common gate excluded; full decoder separate.",
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
        print("CPU_ROUTER_DIRECT_SETUP_PASS", len(fixtures))
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
        shard = ttnn.create_sharded_memory_config(
            (32, 32),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def upload(x, dtype, layout):
            return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=shard)

        kw = dict(
            bias_tensor=upload(torch.zeros(1, 16, 16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            input_indices_tensor=upload(
                torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, ttnn.TILE_LAYOUT
            ),
            output_tensor=upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            output_indices_tensor=upload(torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=8,
            output_softmax=True,
        )
        adapters = {k: router_function(p, ttnn)[0] for k, p in paths.items()}

        def host(values):
            return [ttnn.to_torch(v) for v in values]

        def checks(a, b):
            cs = [compare(x, y) for x, y in zip(a, b)]
            return dict(exact=all(c["exact"] for c in cs), tensors=cs)

        for name, logits, _ in fixtures:
            face = upload(logits, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            gate = ttnn.experimental.deepseek.moe.generalized_moe_gate(face, **kw)
            for value in gate:
                assert (
                    tuple(value.shape) == (1, 32, 32) and value.layout == ttnn.ROW_MAJOR_LAYOUT and value.is_sharded()
                )
            original = host(gate)

            def call(variant):
                with device_only():
                    routing, ids = adapters[variant](*gate)
                assert (
                    tuple(routing.shape) == (1, 8, 1, 1)
                    and routing.dtype == ttnn.bfloat16
                    and routing.layout == ttnn.TILE_LAYOUT
                )
                assert (
                    tuple(ids.shape) == (1, 1, 1, 8)
                    and ids.dtype == ttnn.uint16
                    and ids.layout == ttnn.ROW_MAJOR_LAYOUT
                )
                assert routing.memory_config() == ids.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                return routing, ids

            reference = host(call("baseline"))
            candidate = host(call("candidate"))
            row = dict(
                name=name, output=checks(reference, candidate), gate_input_immutable=checks(original, host(gate))
            )
            result["cases"].append(row)
            save()
            if not row["output"]["exact"] or not row["gate_input_immutable"]["exact"]:
                continue
            for variant in adapters:
                trace = None
                capture_open = False
                out = None
                try:
                    for _ in range(args.warmup):
                        out = call(variant)
                        ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(args.eager_samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call(variant)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start))
                    eager_check = checks(reference, host(out))
                    assert eager_check["exact"]
                    result["eager_timing"].append(
                        dict(
                            name=name,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            last=eager_check,
                        )
                    )
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    out = call(variant)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = checks(reference, host(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = checks(reference, host(out))
                    assert first["exact"] and last["exact"]
                    immutable = checks(original, host(gate))
                    assert immutable["exact"]
                    result["timing"].append(
                        dict(
                            name=name,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            first=first,
                            last=last,
                            gate_input_immutable=immutable,
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
            print("ROUTER_DIRECT_CASE_COMPLETE", name, flush=True)
        result["device_exact"] = all(
            r["output"]["exact"] and r["gate_input_immutable"]["exact"] for r in result["cases"]
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
    p.add_argument("--candidate-snapshot", type=Path, default=DOC / "v4_router_direct_slice.py.snapshot")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--capture", type=Path, default=DOC / "routing_seed154.json.pos5991.pt")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--eager-samples", type=int, default=5)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--output", type=Path, default=DOC / "v4_router_direct_slice_probe.json")
    args = p.parse_args()
    assert min(args.warmup, args.eager_samples, args.samples, args.replays) > 0
    run(args)
