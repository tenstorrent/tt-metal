# SPDX-License-Identifier: Apache-2.0
"""Isolated history/token sequence packing convolution; exclusive TP4 only.

The native convolution, arithmetic order, datatypes, recurrence and state routing
remain unchanged. Native B1 is outside this candidate scope. No production edits or
profiler: trace completion synchronization is a standalone timing boundary.
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


def baseline_conv(model, projected):
    batch, seq, width = projected.shape
    assert seq == 1 and width == 2560
    proj = ttnn.pad(projected, [(0, 0), (0, 31), (0, 0)], 0)
    lanes = []
    for lane in range(batch):
        incoming = ttnn.to_layout(model.conv_state[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT)
        lanes.append(
            ttnn.experimental.kda.qkv_causal_conv1d_silu(
                ttnn.to_layout(proj[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT),
                incoming,
                *model.conv_taps,
                *WIDTHS,
                actual_start=model.conv_zero,
                predecessor_carry=incoming,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                compute_kernel_config=model.compute,
            )
        )
    outputs = [ttnn.concat([row[i] for row in lanes], dim=0) if batch > 1 else lanes[0][i] for i in range(3)]
    return tuple(ttnn.to_layout(value[:, :1, :], ttnn.TILE_LAYOUT) for value in outputs)


BASELINE_SHA256 = "61227325235758233ad340fcea03285c229a89304af818c169ec20cbfdd97bbc"


def pinned_baseline(path):
    """Compile only the historical GDN class against unchanged current globals."""
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == BASELINE_SHA256, "Baseline source drift"
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelGatedDeltaNet")
    namespace = dict(decoder.__dict__)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_chunk")
    return namespace[cls.name], method


def candidate_conv(model, projected):
    batch, seq, width = projected.shape
    assert seq == 1 and width == 2560 and batch in (8, 16, 32)
    history = ttnn.to_layout(model.conv_state, ttnn.ROW_MAJOR_LAYOUT)
    current = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)
    joined = ttnn.concat([history, current], dim=1)
    sequence = ttnn.reshape(joined, (1, 4 * batch, width))
    assert tuple(sequence.shape) == (1, 4 * batch, 2560)
    assert sequence.dtype == ttnn.bfloat16 and sequence.layout == ttnn.ROW_MAJOR_LAYOUT
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        sequence,
        model._probe_predecessor,
        *model.conv_taps,
        *WIDTHS,
        actual_start=model.conv_zero,
        predecessor_carry=model._probe_predecessor,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
        compute_kernel_config=model.compute,
    )
    result = []
    for output, width in zip(outputs, WIDTHS):
        rows = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        selected = rows[:, 3 : 4 * batch : 4, :]
        assert tuple(selected.shape) == (1, batch, width)
        result.append(ttnn.to_layout(ttnn.reshape(selected, (batch, 1, width)), ttnn.TILE_LAYOUT))
    model._probe_conv_geometry = {
        "input": tensor_metadata(sequence),
        "predecessor": tensor_metadata(model._probe_predecessor),
        "widths": list(WIDTHS),
        "selected_rows": list(range(3, 4 * batch, 4)),
        "native_outputs": [tensor_metadata(value) for value in outputs],
        "restored_outputs": [tensor_metadata(value) for value in result],
    }
    return tuple(result)


def candidate_chunk(method):
    """Replace only the convolution in the retained, checksum-pinned baseline."""
    method = copy.deepcopy(method)
    tree = ast.Module(body=[method], type_ignores=[])
    start = next(
        i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "proj"
    )
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "self._roll_conv"
    )
    removed = method.body[start:end]
    assert len(removed) == 5 and isinstance(removed[2], ast.For)
    assert ast.unparse(removed[-1].targets[0]) == "(q, k, v)"
    original = copy.deepcopy(tree)
    method.body[start:end] = ast.parse("q, k, v = candidate_conv(self, projected)").body
    ast.fix_missing_locations(tree)
    namespace = dict(decoder.__dict__, candidate_conv=candidate_conv)
    exec(compile(tree, "<isolated_gdn_conv_history_chunk>", "exec"), namespace)
    return namespace["_chunk"], ast.unparse(original), ast.unparse(tree)


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
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "doc/benchmark/repair_deadline/gdn_conv_initial/multichip_decoder.py.txt",
    )
    args = parser.parse_args()
    assert args.repeats > 0 and args.warmup > 0 and all(b in (8, 16, 32) for b in args.batches)
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
    BaselineGDN, baseline_method = pinned_baseline(args.baseline_source)
    alternate_chunk, original_source, rewritten_source = candidate_chunk(baseline_method)
    (args.output / "baseline_chunk.py.txt").write_text(original_source + "\n")
    (args.output / "candidate_chunk.py.txt").write_text(rewritten_source + "\n")
    report = {
        "status": "started",
        "scope": "Isolated whole convolution chain and real-weight GDN layer; not a full-model benchmark",
        "tracking": tracking,
        "baseline_source_sha256": BASELINE_SHA256,
        "poison_scope": "Zero predecessor/normal padding and NaN predecessor/implicit TILE padding. Every lane has distinct finite history; all selected rows must remain exact finite.",
        "cases": [],
        "timing_boundary": "host execute_trace through standalone completion synchronization; reset/readbacks excluded",
        "op_sequence": [
            {
                "stage": "Input/history bridge",
                "baseline": "Pad [B,1,2560] to32; per-lane TILE slice then ROW_MAJOR",
                "candidate": "ROW_MAJOR history/current concat [B,4,2560], reshape [1,4B,2560]",
            },
            {
                "stage": "Convolution",
                "baseline": "B native calls, each20 channel blocks; BF16 partials, unchanged HiFi4 config",
                "candidate": "One native call,20 channel blocks times B/8 sequence tiles; original BF16 taps, same four-tap order and precise SiLU",
            },
            {
                "stage": "Output bridge",
                "baseline": "Three B-way TILE concatenations then first-token slice",
                "candidate": "ROW_MAJOR strided select rows3,7,...,4B-1 then reshape [B,1,W] and TILE",
            },
            {
                "stage": "Remaining attention",
                "baseline": "Original roll, normalization, recurrence, gating, output projection, state routing",
                "candidate": "Identical production method AST outside convolution block; roll-clear optimization NOT applied",
            },
        ],
        "sources": {},
    }
    for path in (
        Path(__file__),
        Path(decoder.__file__),
        Path(precision.__file__),
        Path(helpers.__file__),
        args.baseline_source,
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
            model = BaselineGDN(
                state, cfg, ops, batch_size=batch, max_requests=batch, chunk_size=1, allocate_column_weight=batch == 1
            )
            precision.cast_attr(model, "packed_weight", policy["weight_groups"]["gdn_input"])
            precision.cast_attr(model, "out_weight", policy["weight_groups"]["gdn_output"])
            model.projection_compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_input"])
            model.output_projection.row_weight = model.out_weight
            model.output_projection.compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_output"])
            model.recurrence_broadcast = batch > 1
            model._probe_predecessor = ops.upload(
                torch.full((1, 3, 2560), float("nan") if poison else 0, dtype=torch.bfloat16),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
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
            projected = model._linear(hidden, model.packed_weight)[..., :2560]
            assert projected.dtype == ttnn.bfloat16 and model.conv_state.dtype == ttnn.bfloat16
            if poison:
                projected = ttnn.fill_implicit_tile_padding(projected, float("nan"))
            original_chunk = model._chunk
            candidate_bound = original_chunk if batch == 1 else types.MethodType(alternate_chunk, model)
            row = {
                "batch": batch,
                "sequence": 1,
                "padding_and_predecessor": "nan" if poison else "zero",
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
                    "projected": tensor_metadata(projected),
                    "history": tensor_metadata(model.conv_state),
                    "taps": [tensor_metadata(t) for t in model.conv_taps],
                    "candidate_uses_original_taps": True,
                },
                "native_work": {
                    "baseline_calls": batch,
                    "baseline_channel_blocks_per_call": 20,
                    "candidate_calls": 1,
                    "candidate_channel_blocks": 20,
                    "candidate_sequence_tiles": batch // 8,
                    "candidate_workers": min(20 * (batch // 8), grid.x * grid.y),
                },
                "stages": {},
            }
            report["cases"].append(row)

            def reset():
                ttnn.copy(initial_recurrent, model.recurrent_bank)
                ttnn.copy(initial_recurrent, model.recurrent_state)
                ttnn.copy(initial_conv, model.conv_bank)
                ttnn.copy(initial_conv, model.conv_state)

            for stage in ("whole_conv_chain", "full_gdn"):
                row["stages"][stage] = {}
                calls = (
                    {
                        "baseline": lambda: baseline_conv(model, projected),
                        "candidate": lambda: candidate_conv(model, projected),
                    }
                    if stage == "whole_conv_chain"
                    else {
                        name: lambda: model.decode_forward(hidden, current_positions=positions, slots=slots)
                        for name in ("baseline", "candidate")
                    }
                )
                snapshot = (
                    (lambda output: {name: helpers.host(value) for name, value in zip(("q", "k", "v"), output)})
                    if stage == "whole_conv_chain"
                    else (lambda output: helpers.snapshot(model, output))
                )
                for label, call in calls.items():
                    model._chunk = original_chunk if label == "baseline" else candidate_bound
                    reset()
                    output = call()
                    ttnn.synchronize_device(mesh)
                    del output
                reference = None
                for label, call in calls.items():
                    model._chunk = original_chunk if label == "baseline" else candidate_bound
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
                    if label == "candidate" and batch > 1:
                        row["candidate_native_geometry"] = model._probe_conv_geometry
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
            del model._chunk
            del (
                model,
                ops,
                hidden,
                initial_recurrent,
                initial_conv,
                slots,
                positions,
                projected,
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
