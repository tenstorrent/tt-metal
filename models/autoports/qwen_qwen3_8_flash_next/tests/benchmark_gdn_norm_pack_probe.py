# SPDX-License-Identifier: Apache-2.0
"""Isolated logical-row GDN normalization packing; exclusive TP4 only.

The native convolution, arithmetic order, datatypes, recurrence and state routing
remain unchanged. Native B1/B4 are unchanged controls. No production edits or
profiler: trace completion synchronization is a standalone timing boundary.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import statistics
import textwrap
import time
import traceback
import types
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_row_fragment_probe as helpers
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as decoder
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState

WIDTHS = (512, 512, 1536)


def tensor_metadata(value):
    return {
        "shape": list(value.shape),
        "padded_shape": list(value.padded_shape),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "memory_config": str(value.memory_config()),
    }


def baseline_qk(model, value, weight):
    return ttnn.rms_norm(
        ttnn.typecast(value, ttnn.float32), weight=weight, epsilon=1e-06 / 128, compute_kernel_config=model.compute
    )


def candidate_qk(model, value, weight):
    batch = value.shape[0]
    if batch not in (8, 16, 32):
        return baseline_qk(model, value, weight)
    shape = tuple(value.shape)
    assert shape == (batch, 1, 4, 128)
    source = ttnn.to_layout(ttnn.typecast(value, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
    flat = ttnn.to_layout(ttnn.reshape(source, (1, 1, batch * 4, 128)), ttnn.TILE_LAYOUT)
    normalized = ttnn.rms_norm(flat, weight=weight, epsilon=1e-06 / 128, compute_kernel_config=model.compute)
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


def baseline_gate(model, output, gate):
    batch = output.shape[0]
    output = ttnn.typecast(output, ttnn.bfloat16)
    output = ttnn.pad(output, [(0, 0), (0, 0), (0, 31), (0, 0)], 0)
    gate = ttnn.pad(gate, [(0, 0), (0, 31), (0, 0)], 0)
    output = ttnn.reshape(output, (batch * 12, 32, 128))
    gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        output,
        gate,
        model.norm_weight_flat,
        12,
        epsilon=model.norm_eps,
        compute_kernel_config=model.compute,
        output_dtype=ttnn.bfloat16,
    )
    return ttnn.to_layout(gated[:, :1, :], ttnn.TILE_LAYOUT)


def candidate_gate(model, output, gate):
    batch = output.shape[0]
    if batch not in (8, 16, 32):
        return baseline_gate(model, output, gate)
    assert tuple(output.shape) == (batch, 12, 1, 128) and tuple(gate.shape) == (batch, 1, 1536)
    output = ttnn.to_layout(ttnn.typecast(output, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    flat = ttnn.to_layout(ttnn.reshape(output, (1, batch * 12, 128)), ttnn.TILE_LAYOUT)
    gate = ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(gate, ttnn.ROW_MAJOR_LAYOUT), (1, batch * 12, 128)), ttnn.TILE_LAYOUT
    )
    gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
        flat,
        gate,
        model.norm_weight_flat,
        1,
        epsilon=model.norm_eps,
        compute_kernel_config=model.compute,
        output_dtype=ttnn.bfloat16,
    )
    return ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(gated, ttnn.ROW_MAJOR_LAYOUT), (batch, 1, 1536)), ttnn.TILE_LAYOUT
    )


def transformed_methods():
    """Replace only two RMSNorm calls and the padded gated-normalization chain."""
    namespace = dict(decoder.__dict__, candidate_qk=candidate_qk, candidate_gate=candidate_gate)
    sources = {}
    result = []
    for name in ("_recurrence", "_chunk"):
        original = textwrap.dedent(inspect.getsource(getattr(decoder.ParallelGatedDeltaNet, name)))
        tree = ast.parse(original)
        method = tree.body[0]
        if name == "_recurrence":
            replaced = 0
            for i, node in enumerate(method.body):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "ttnn.rms_norm"
                ):
                    symbol = ast.unparse(node.targets[0])
                    assert symbol in ("q", "k")
                    method.body[i] = ast.parse(f"{symbol} = candidate_qk(self, {symbol}, self.{symbol}scale)").body[0]
                    replaced += 1
            assert replaced == 2
        else:
            start = next(
                i
                for i, n in enumerate(method.body)
                if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "o"
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "ttnn.typecast"
            )
            stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Return))
            assert len(method.body[start:stop]) == 6
            method.body[start:stop] = ast.parse("gated = candidate_gate(self, o, projections[..., 2560:4096])").body
        ast.fix_missing_locations(tree)
        exec(compile(tree, f"<isolated_gdn_norm_{name}>", "exec"), namespace)
        result.append(namespace[name])
        sources[name] = dict(baseline=original, candidate=ast.unparse(tree))
    return (*result, sources)


def hashes(snapshot):
    return {
        name: [hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in chips]
        for name, chips in snapshot.items()
    }


def measure(mesh, call, reset, snapshot, output_dir, label, repeats, warmup, reference, save):
    trace = output = None
    times, comparisons = [], []
    try:
        reset()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        for repeat in range(warmup + repeats):
            reset()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            if repeat < warmup:
                continue
            times.append(elapsed)
            actual = snapshot(output)
            if reference is None:
                reference = actual
            metrics = helpers.compare(reference, actual)
            exact = all(row["exact"] and row["finite"] for rows in metrics.values() for row in rows)
            for name in metrics:
                for row in metrics[name]:
                    row["pcc"] = 1.0 if row["exact"] and row["finite"] else None
                    row["pcc_basis"] = "exact finite element equality" if row["pcc"] == 1.0 else "not claimed"
            entry = {"repeat": repeat - warmup, "metrics": metrics, "sha256_per_asic": hashes(actual)}
            comparisons.append(entry)
            # Retain raw outputs, not repeated hundred-MB state dumps. Every
            # state element is still compared and every shard is hashed.
            if repeat == warmup or not exact:
                raw = (
                    actual
                    if not exact
                    else {name: values for name, values in actual.items() if name in ("q", "k", "v", "output")}
                )
                torch.save(raw, output_dir / f"{label}_{repeat - warmup}.pt")
            save()
            if not exact:
                raise AssertionError(f"Exact finite output/state mismatch: {label}/{repeat}")
        return {"seconds": times, "median_seconds": statistics.median(times), "comparisons": comparisons}, reference
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        del output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    assert args.repeats > 0 and args.warmup > 0 and all(b in (1, 4, 8, 16, 32) for b in args.batches)
    tracking = {
        "native": bool(ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()),
        "python": bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        "skip_program_cache": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "exemptions": [],
    }
    assert tracking["native"] and tracking["python"] and tracking["skip_program_cache"] == "0"
    assert all(
        os.environ.get(k, "0") in ("", "0")
        for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL")
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    alternate_recurrence, alternate_chunk, method_sources = transformed_methods()
    for name, variants in method_sources.items():
        for variant, source in variants.items():
            (args.output / f"{variant}{name}.py.txt").write_text(source + "\n")
    report = {
        "status": "started",
        "scope": "Isolated complete normalization bridges and real-weight GDN layer; not a full-model benchmark",
        "tracking": tracking,
        "method_sources": "retained exact current baseline and AST-transformed candidate",
        "poison_scope": "Zero/NaN implicit TILE padding on isolated Q/K/recurrence-output/gate inputs and full-GDN history. All logical outputs and full state must remain exact finite.",
        "cases": [],
        "timing_boundary": "host execute_trace through standalone completion synchronization; reset/readbacks excluded",
        "op_sequence": [
            {
                "stage": "Q/K RMSNorm",
                "baseline": "FP32 [B,1,4,128] padded head32",
                "candidate": "Same FP32 conversion/scale/epsilon/compute; RM logical-row reshape to[1,1,4B,128], TILE RMSNorm then restore",
            },
            {
                "stage": "Gated RMSNorm",
                "baseline": "BF16 [12B,32,128], H12, gate[B,32,1536]",
                "candidate": "Same BF16 conversion/weights/epsilon/compute; RM logical rows to[1,12B,128], native H1 then restore[B,1,1536]",
            },
            {
                "stage": "Rest of GDN",
                "baseline": "Current qualified history convolution, recurrence, state routing and output projection",
                "candidate": "Unchanged; B1/B4 use original methods",
            },
        ],
        "sources": {},
    }
    for path in (
        Path(__file__),
        Path(decoder.__file__),
        Path(precision.__file__),
        Path(helpers.__file__),
        Path(decoder.__file__).with_name("gdn_conv_pack.py"),
    ):
        report["sources"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    mesh = None
    save()
    try:
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
        assert cfg.layer_types[args.layer] == "linear_attention"
        policy = precision.for_layer(precision.load_policy(), args.layer)
        report["precision"] = policy
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=268435456)
        grid = mesh.compute_with_storage_grid_size()
        report.update(device_ids=list(mesh.get_device_ids()), grid=[grid.x, grid.y])
        for batch, poison in [(b, p) for b in args.batches for p in (False, True)]:
            ops = decoder.MeshOps(mesh)
            ops.precision_ccl = policy["ccl_dtype"]
            state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.linear_attn.")
            model = decoder.ParallelGatedDeltaNet(
                state, cfg, ops, batch_size=batch, max_requests=batch, chunk_size=1, allocate_column_weight=batch == 1
            )
            precision.cast_attr(model, "packed_weight", policy["weight_groups"]["gdn_input"])
            precision.cast_attr(model, "out_weight", policy["weight_groups"]["gdn_output"])
            model.projection_compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_input"])
            model.output_projection.row_weight = model.out_weight
            model.output_projection.compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_output"])
            model.recurrence_broadcast = batch > 1
            # Match shallow serving views: their owner prepares/shared this
            # persistent predecessor; standalone component construction does not.
            if batch in (8, 16):
                model.conv_decode_history_enabled = True
                model.conv_decode_predecessor = ops.upload(torch.zeros(1, 3, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
            rng = torch.Generator().manual_seed(2026101701 + batch)
            hidden = ops.upload(torch.randn(batch, 1, 2560, generator=rng).bfloat16() * 0.2)
            initial_recurrent = ops.shards(
                [torch.randn(batch, 12, 128, 128, generator=rng) * 0.01 for _ in range(4)], ttnn.float32
            )
            initial_conv = ops.shards(
                [
                    (
                        torch.randn(batch, 3, 2560, generator=rng) * 0.03
                        + torch.arange(batch).reshape(batch, 1, 1) * 0.007
                    ).bfloat16()
                    for _ in range(4)
                ]
            )
            if poison:
                initial_conv = ttnn.fill_implicit_tile_padding(initial_conv, float("nan"))
            slot_host = torch.zeros(batch, 32, dtype=torch.int32)
            position_host = slot_host.clone()
            slot_host[:, 0] = torch.arange(batch - 1, -1, -1)
            position_host[:, 0] = torch.arange(batch) + 17
            slots = ops.upload(slot_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            positions = ops.upload(position_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            q_input = ops.upload(torch.randn(batch, 1, 4, 128, generator=rng).bfloat16())
            k_input = ops.upload(torch.randn(batch, 1, 4, 128, generator=rng).bfloat16())
            output_input = ops.upload(torch.randn(batch, 12, 1, 128, generator=rng), ttnn.float32)
            gate_input = ops.upload(torch.randn(batch, 1, 1536, generator=rng).bfloat16())
            if poison:
                q_input = ttnn.fill_implicit_tile_padding(q_input, float("nan"))
                k_input = ttnn.fill_implicit_tile_padding(k_input, float("nan"))
                output_input = ttnn.fill_implicit_tile_padding(output_input, float("nan"))
                gate_input = ttnn.fill_implicit_tile_padding(gate_input, float("nan"))
            original_chunk, original_recurrence = model._chunk, model._recurrence
            candidate_bound = original_chunk if batch in (1, 4) else types.MethodType(alternate_chunk, model)
            candidate_recurrence = (
                original_recurrence if batch in (1, 4) else types.MethodType(alternate_recurrence, model)
            )
            row = {
                "batch": batch,
                "sequence": 1,
                "padding": "nan" if poison else "zero",
                "unchanged_control": batch in (1, 4),
                "convolution_compute": {
                    key: str(getattr(model.compute, key))
                    for key in (
                        "math_fidelity",
                        "math_approx_mode",
                        "fp32_dest_acc_en",
                        "packer_l1_acc",
                        "dst_full_sync_en",
                    )
                },
                "geometry": {
                    "q": tensor_metadata(q_input),
                    "k": tensor_metadata(k_input),
                    "output": tensor_metadata(output_input),
                    "gate": tensor_metadata(gate_input),
                    "qscale": tensor_metadata(model.qscale),
                    "kscale": tensor_metadata(model.kscale),
                    "norm_weight": tensor_metadata(model.norm_weight_flat),
                },
                "native_work": {
                    "baseline_gated_units": batch * 12,
                    "candidate_gated_units": batch * 12 if batch in (1, 4) else batch * 12 // 32,
                },
                "stages": {},
            }
            report["cases"].append(row)

            def reset():
                ttnn.copy(initial_recurrent, model.recurrent_bank)
                ttnn.copy(initial_recurrent, model.recurrent_state)
                ttnn.copy(initial_conv, model.conv_bank)
                ttnn.copy(initial_conv, model.conv_state)

            for stage in ("qk_chain", "gated_chain", "full_gdn"):
                row["stages"][stage] = {}
                if stage == "qk_chain":
                    calls = {
                        "baseline": lambda: (
                            baseline_qk(model, q_input, model.qscale),
                            baseline_qk(model, k_input, model.kscale),
                        ),
                        "candidate": lambda: (
                            candidate_qk(model, q_input, model.qscale),
                            candidate_qk(model, k_input, model.kscale),
                        ),
                    }
                    snapshot = lambda output: {name: helpers.host(value) for name, value in zip(("q", "k"), output)}
                elif stage == "gated_chain":
                    calls = {
                        "baseline": lambda: baseline_gate(model, output_input, gate_input),
                        "candidate": lambda: candidate_gate(model, output_input, gate_input),
                    }
                    snapshot = lambda output: {"output": helpers.host(output)}
                else:
                    calls = {
                        name: lambda: model.decode_forward(hidden, current_positions=positions, slots=slots)
                        for name in ("baseline", "candidate")
                    }
                    snapshot = lambda output: helpers.snapshot(model, output)
                for label, call in calls.items():
                    model._chunk = original_chunk if label == "baseline" else candidate_bound
                    model._recurrence = original_recurrence if label == "baseline" else candidate_recurrence
                    reset()
                    output = call()
                    ttnn.synchronize_device(mesh)
                    del output
                reference = None
                for label, call in calls.items():
                    model._chunk = original_chunk if label == "baseline" else candidate_bound
                    model._recurrence = original_recurrence if label == "baseline" else candidate_recurrence
                    result, reference = measure(
                        mesh,
                        call,
                        reset,
                        snapshot,
                        args.output,
                        f"b{batch}_{'nan' if poison else 'zero'}_{stage}_{label}",
                        args.repeats,
                        args.warmup,
                        reference,
                        save,
                    )
                    row["stages"][stage][label] = result
                    save()
                results = row["stages"][stage]
                results["median_speedup"] = (
                    results["baseline"]["median_seconds"] / results["candidate"]["median_seconds"]
                )
                print(
                    json.dumps(
                        {
                            "batch": batch,
                            "stage": stage,
                            "poison": poison,
                            "median_speedup": results["median_speedup"],
                            "baseline_s": results["baseline"]["median_seconds"],
                            "candidate_s": results["candidate"]["median_seconds"],
                        }
                    ),
                    flush=True,
                )
                del reference
            del model._chunk, model._recurrence
            del (
                model,
                ops,
                hidden,
                initial_recurrent,
                initial_conv,
                slots,
                positions,
                q_input,
                k_input,
                output_input,
                gate_input,
                original_recurrence,
                candidate_recurrence,
                original_chunk,
                candidate_bound,
            )
            save()
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
