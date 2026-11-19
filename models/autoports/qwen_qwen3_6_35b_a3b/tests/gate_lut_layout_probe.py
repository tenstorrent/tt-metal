# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FP32 gate-table gather layouts: exhaustive values and complete traced adapters.

Setup builds only checkpoint-derived tables on CPU. Each measured adapter starts
with a persistent BF16 TILE [1,S,32] projection and returns FP32 TILE [1,S,32].
No upload/readback is inside capture. Decode looks up decay alone; prefill looks
up both g and decay. This is a diagnostic, not a decoder performance claim.
"""

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def compare(expected, actual):
    a = expected.float().contiguous()
    b = actual.float().reshape(a.shape).contiguous()
    finite = torch.isfinite(a)
    differences = a.view(torch.int32) != b.view(torch.int32)
    categories = (
        (torch.isnan(a) != torch.isnan(b))
        | (torch.isposinf(a) != torch.isposinf(b))
        | (torch.isneginf(a) != torch.isneginf(b))
    )
    return dict(
        elements=a.numel(),
        finite_elements=int(finite.sum()),
        finite_code_differences=int((differences & finite).sum()),
        nonfinite_category_differences=int(categories.sum()),
        # NaN payload equality is deliberately not an acceptance condition.
        passed=not bool((differences & finite).any() or categories.any()),
    )


def load_tables():
    checkpoint = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION
    )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("A_log", "dt_bias"):
        key = "model.language_model.layers.0.linear_attn." + name
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            weights[name] = f.get_tensor(key).float().reshape(32, 1)
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
    g = -weights["A_log"].exp() * torch.nn.functional.softplus(values[None, :] + weights["dt_bias"])
    return {"g": g, "decay": g.exp()}


def run(args):
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    parameters = dict(
        physical_device_ids=[0], trace_region_size=args.trace_region_size, l1_small_size=args.l1_small_size
    )
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["device_open_parameters"] = parameters
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    result = dict(
        status="running",
        checkpoint_revision=REVISION,
        device_open_parameters=parameters,
        all_codes_per_head=65536,
        block_size=args.block_size,
        variants=[],
        trace_allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        timing_scope="Complete BF16 TILE projection -> FP32 TILE gate adapter; setup/readback excluded",
        source_contract={
            "dim0": "gather.cpp pre_gather_transform_tensor transposes gather axis to last, including table",
            "fp32_dispatch": "gather_codegen_supported.cpp rejects non-BFLOAT16 input; FP32 uses native",
            "rm": "RM single-row writer loads complete 65536-element (256 KiB FP32) input stick per head",
            "rm_parallelism": "Factory distributes H rows across cores; SingleRowSingleCore means one core per row",
            "codegen_alternative": "No direct FP32 codegen contract; BF16 conversion would change table values",
        },
    )

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    mesh = None
    stage = "cpu_setup"
    save()
    try:
        tables = load_tables()
        # A reproducible ordinary finite BF16 input with different indices by head.
        codes = (torch.arange(128)[:, None] * 43 + torch.arange(32)[None, :] * 97) % 4096 + 14336
        canonical = codes.to(torch.uint16).contiguous().view(torch.bfloat16).reshape(1, 128, 32)
        capture_names = []
        captured = []
        for position in (128, 129, 130, 1024, 3979, 5011, 7551, 7703, 8157):
            path = Path(f"{args.capture_base}.pos{position}.pt")
            if path.exists():
                captured.append(
                    torch.load(path, map_location="cpu", weights_only=False)["hf_values"]["projection_a"]
                    .bfloat16()
                    .reshape(1, 1, 32)
                )
                capture_names.append(str(path))
        if captured:
            sequence = torch.cat(captured, dim=1)
            canonical = sequence.repeat(1, (128 + sequence.shape[1] - 1) // sequence.shape[1], 1)[:, :128].contiguous()
        result["timing_input"] = dict(
            captures=capture_names, repeated_to_length=128, synthetic_if_no_capture=not bool(captured)
        )
        torch.save(dict(timing_projection_a_bf16=canonical), str(output) + ".inputs.pt")
        stage = "open_device"
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **parameters)

        def upload(x, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def reference(a, names):
            idx = a.contiguous().view(torch.uint16).to(torch.int64).reshape(-1, 32)
            return tuple(torch.gather(tables[name].T, 0, idx).reshape(a.shape) for name in names)

        def read(values):
            return tuple(ttnn.to_torch(v).clone() for v in values)

        def checks(expected, actual, names):
            return {name: compare(a, b) for name, a, b in zip(names, expected, actual)}

        def traced(fn, expected, names):
            for _ in range(args.warmup):
                warm = fn()
                ttnn.synchronize_device(mesh)
                del warm
            trace_id, capture_open, values = None, False, None
            try:
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                values = fn()
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                capture_open = False
                # One trace at a time; only these outputs are overwritten by
                # it, consumed before release, and never inputs to another trace.
                for v in values:
                    trace_allocation_tracker.acknowledge_corruptible(v)
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                first = read(values)
                samples = []
                for _ in range(args.iterations):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.trace_replays):
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.trace_replays)
                last = read(values)
                return dict(
                    median_ms=statistics.median(samples),
                    min_ms=min(samples),
                    samples_ms=samples,
                    replays_per_sample=args.trace_replays,
                    warm_replays=args.warmup,
                    same_input_each_replay=True,
                    blocking=False,
                    synchronization="group boundaries",
                    first_vs_reference=checks(expected, first, names),
                    last_vs_reference=checks(expected, last, names),
                    first_vs_last=checks(first, last, names),
                )
            finally:
                if trace_id is not None:
                    try:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                        ttnn.synchronize_device(mesh)
                    finally:
                        ttnn.release_trace(mesh, trace_id)
                del values

        for variant in args.variants:
            row = dict(name=variant, status="running", exhaustive={}, timing=[])
            result["variants"].append(row)
            tt_tables = None
            try:
                stage = variant + ":table_setup"
                transposed = variant.endswith("dim0")
                layout = ttnn.ROW_MAJOR_LAYOUT if variant.startswith("rm") else ttnn.TILE_LAYOUT
                tt_tables = {name: upload(v.T if transposed else v, ttnn.float32, layout) for name, v in tables.items()}

                def adapter(a, names):
                    s = a.shape[1]
                    idx = ttnn.reshape(ttnn.bitcast(a, ttnn.uint16), (s, 32))
                    if not transposed:
                        idx = ttnn.transpose(idx, -1, -2)
                    if layout == ttnn.ROW_MAJOR_LAYOUT:
                        idx = ttnn.to_layout(idx, layout)
                    outs = []
                    for name in names:
                        value = ttnn.gather(tt_tables[name], 0 if transposed else 1, idx)
                        if not transposed and not (s == 1 and layout == ttnn.TILE_LAYOUT):
                            value = ttnn.transpose(value, -1, -2)
                        value = ttnn.reshape(value, (1, s, 32))
                        if layout == ttnn.ROW_MAJOR_LAYOUT:
                            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
                        outs.append(value)
                    return tuple(outs)

                # Every BF16 code in every head, with a different circular offset
                # by head to expose incorrect broadcasting or column selection.
                totals = {
                    name: dict(
                        elements=0,
                        finite_elements=0,
                        finite_code_differences=0,
                        nonfinite_category_differences=0,
                        passed=True,
                    )
                    for name in tables
                }
                stage = variant + ":exhaustive"
                for start in range(0, 65536, args.block_size):
                    n = min(args.block_size, 65536 - start)
                    block = (torch.arange(start, start + n)[:, None] + torch.arange(32)[None, :] * 997) % 65536
                    a = block.to(torch.uint16).contiguous().view(torch.bfloat16).reshape(1, n, 32)
                    device_a = upload(a, ttnn.bfloat16)
                    got = adapter(device_a, tuple(tables))
                    measured = checks(reference(a, tuple(tables)), read(got), tuple(tables))
                    for name, stats in measured.items():
                        for key in totals[name]:
                            if key == "passed":
                                totals[name][key] &= stats[key]
                            else:
                                totals[name][key] += stats[key]
                    del got, device_a
                row["exhaustive"] = totals
                for s, names in ((1, ("decay",)), (128, ("g", "decay"))):
                    stage = f"{variant}:trace_S{s}"
                    a = canonical[:, :s].contiguous()
                    device_a = upload(a, ttnn.bfloat16)
                    expected = reference(a, names)
                    fn = lambda: adapter(device_a, names)
                    eager = fn()
                    item = dict(S=s, outputs=names, eager=checks(expected, read(eager), names))
                    del eager
                    row["timing"].append(item)
                    item["trace"] = traced(fn, expected, names)
                    del fn, device_a
                row["status"] = (
                    "passed"
                    if all(v["passed"] for v in totals.values())
                    and all(
                        v["passed"]
                        for item in row["timing"]
                        for group in (
                            item["eager"],
                            item["trace"]["first_vs_reference"],
                            item["trace"]["last_vs_reference"],
                            item["trace"]["first_vs_last"],
                        )
                        for v in group.values()
                    )
                    else "numerical_failure"
                )
            except Exception:
                row.update(status="failed", failure_stage=stage, error=traceback.format_exc())
            finally:
                ttnn.synchronize_device(mesh)
                del tt_tables
                save()
                print("GATE_LAYOUT", row["name"], row["status"], flush=True)
        result["status"] = (
            "passed" if all(r["status"] == "passed" for r in result["variants"]) else "completed_with_failures"
        )
    except Exception:
        result.update(status="failed", failure_stage=stage, error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("tile_dim1", "tile_dim0", "rm_dim1", "rm_dim0"),
        default=["tile_dim1", "rm_dim1", "tile_dim0", "rm_dim0"],
    )
    parser.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--trace-replays", type=int, default=20)
    parser.add_argument("--trace-region-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--l1-small-size", type=int, default=32768)
    args = parser.parse_args()
    if not 1 <= args.block_size <= 65536 or min(args.warmup, args.iterations, args.trace_replays) < 1:
        parser.error("block size must be 1..65536; warmup, iterations, and trace replays must be positive")
    run(args)
