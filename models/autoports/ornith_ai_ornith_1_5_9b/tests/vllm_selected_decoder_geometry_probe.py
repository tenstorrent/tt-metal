# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded selected-LoFi decoder geometry discriminator; no serving throughput.

Exclusive hardware and an external timeout are required. Build real layers0/3
at context262144; retain each production decode projection's natural-text input.
Sweep readers1/2/3/4 and K tile divisors up to16 at unchanged production input,
weight and output placements. Measure the complete decode_linear boundary,
including its existing layout conversions. This does not search other grids,
weight repadding, CCL families, or precision. Winners require integrated gates.
Use --layer/--role to bound an initial hardware launch. No profiler is used.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def candidate_schedule(k_tiles, production_readers, production_block, max_k_tiles):
    blocks = [k for k in range(1, min(k_tiles, max_k_tiles) + 1) if k_tiles % k == 0]
    baseline = (production_readers, production_block)
    return [baseline] + [(r, k) for k in blocks for r in (1, 2, 3, 4) if (r, k) != baseline] + [baseline]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 3))
    parser.add_argument(
        "--role",
        choices=(
            "packed_input",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "linear_attn.out_proj",
            "self_attn.o_proj",
        ),
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-k-tiles", type=int, default=16)
    args = parser.parse_args()
    if min(args.iterations, args.repetitions, args.max_k_tiles) <= 0:
        parser.error("iteration, repetition and K bounds must be positive")
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_DEVICE_PROFILER_DISPATCH", "TT_METAL_WATCHER"):
        if os.environ.get(name, "0") not in ("0", ""):
            parser.error(f"Disable {name} for this probe")
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")

    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator

    torch.set_num_threads(8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "rows": [],
        "context": 262144,
        "layers": [0, 3],
        "mesh": [1, 4],
        "logical_batch": 1,
        "physical_batch": 32,
        "prompt_tokens": 128,
        "decode_position": 128,
        "timed_scope": "Production decode_linear including existing conversion, fixed natural-text inputs",
        "serving_or_full_decode_throughput": False,
        "tracking": trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "timing_eligible": not trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "max_k_tiles": args.max_k_tiles,
        "search_scope": "Readers1/2/3/4 and K divisors within bound; production placement and weight padding fixed",
        "untried": "Different input/output grids, weight repadding, other pipelines and K divisors above max_k_tiles",
        "invocation": vars(args) | {"output": str(args.output)},
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def host(output):
        return [ttnn.to_torch(shard).float().clone() for shard in ttnn.get_device_tensors(output)]

    def compare(expected, outputs):
        comparisons = []
        for stimulus, (references, output) in enumerate(zip(expected, outputs, strict=True)):
            for shard, (reference, actual) in enumerate(zip(references, host(output), strict=True)):
                a, b = actual.double().reshape(-1), reference.double().reshape(-1)
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                exact = torch.equal(actual, reference)
                pcc = 1.0 if exact else float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if finite else None
                top = min(5, actual.shape[-1])
                comparisons.append(
                    {
                        "stimulus": stimulus,
                        "shard": shard,
                        "finite": finite,
                        "exact": exact,
                        "pcc": pcc,
                        "max_abs": float((a - b).abs().max()) if finite else None,
                        "top_channel_equal": bool(torch.equal(actual.argmax(-1), reference.argmax(-1))),
                        "top5_channel_set_equal": bool(
                            torch.equal(
                                actual.topk(top, dim=-1).indices.sort(-1).values,
                                reference.topk(top, dim=-1).indices.sort(-1).values,
                            )
                        ),
                    }
                )
        return comparisons

    def acceptable(rows):
        return all(
            r["finite"]
            and r["pcc"] is not None
            and math.isfinite(r["pcc"])
            and r["pcc"] >= 0.999
            and r["top_channel_equal"]
            and r["top5_channel_set_equal"]
            for r in rows
        )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        policy = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/configs/bfp8_all_lofi.json"
        gen = build_generator(
            None, mesh, layer_indices=[0, 3], max_batch_size=1, max_seq_len=262144, precision_config=policy
        )
        model = gen.model
        result["actual_policy"] = model.precision_summary()
        result["memory_full_context"] = model.memory_view()
        selected = {}
        for index, layer in zip(model.layer_indices, model.layers, strict=True):
            if args.layer is not None and index != args.layer:
                continue
            for role in layer.dram_weights:
                if args.role is not None and role != args.role:
                    continue
                assert layer.dram_weights[role].dtype == ttnn.bfloat8_b
                assert layer.matmul_compute(role).math_fidelity == ttnn.MathFidelity.LoFi
                assert layer.decode_activation_dtype == ttnn.bfloat16
                selected[(index, role)] = {
                    "layer": layer,
                    "saved": [],
                    "memory": [],
                    "original": layer.dram_layouts[role],
                }
        if not selected:
            raise ValueError("Requested layer/role combination does not exist")
        result["selected_roles"] = [f"{index}/{role}" for index, role in selected]
        stimuli = [
            "Explain how a computer learns patterns from examples and uses them to make predictions. Include a simple example and describe how to check whether it generalizes.",
            "Write a Python function that counts words in a sentence. Explain its inputs and output, and discuss what should happen when the sentence is empty.",
        ]
        result["stimuli"] = []
        for text in stimuli:
            encoded = gen.tokenizer.encode(text, add_special_tokens=False)
            ids = (encoded * math.ceil(129 / len(encoded)))[:129]
            gen.reset(clear_kv=True)
            gen.prefill_forward(
                torch.tensor([ids[:128]]), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[128]
            )
            token = model.upload(torch.tensor([[ids[128]]]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            position = model.upload(torch.tensor([128]), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            table = model.upload(gen.page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            with ExitStack() as stack:
                for index, layer in zip(model.layer_indices, model.layers, strict=True):
                    original = layer.decode_linear

                    def retain(x, role, *, index=index, original=original):
                        entry = selected.get((index, role))
                        if entry is not None:
                            entry["memory"].append(x.memory_config())
                            # Save independent DRAM copies, restoring original
                            # placement once per role outside all timed loops.
                            entry["saved"].append(ttnn.clone(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)))
                        return original(x, role)

                    stack.enter_context(patch.object(layer, "decode_linear", side_effect=retain))
                output = model.decode(token, cache=gen.cache, slots=[0], positions=[position], page_tables=[table])
            del output, token, position, table
            result["stimuli"].append(
                {"text": text, "token_ids": ids, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
            )
        assert all(
            len(entry["saved"]) == len(stimuli) for entry in selected.values()
        ), "Missing or repeated projection call"

        for (index, role), entry in selected.items():
            layer = entry["layer"]
            im, om, production = entry["original"]
            inputs = [
                ttnn.to_memory_config(x, memory) for x, memory in zip(entry["saved"], entry["memory"], strict=True)
            ]
            weight = layer.dram_weights[role]
            k, n = weight.shape
            expected = []
            for x in inputs:
                output = layer.decode_linear(x, role)
                expected.append(host(output))
                del output
            schedule = candidate_schedule(
                k // 32, production.num_workers_per_dram_bank, production.in0_block_w, args.max_k_tiles
            )
            for ordinal, (readers, block) in enumerate(schedule):
                row = {
                    "layer": index,
                    "role": role,
                    "readers": readers,
                    "K_block_tiles": block,
                    "baseline": ordinal in (0, len(schedule) - 1),
                    "baseline_repeat": ordinal == len(schedule) - 1,
                    "K": k,
                    "N": n,
                    "input_memory": str(im),
                    "output_memory": str(om),
                    "captured_input_memory": [str(x) for x in entry["memory"]],
                    "input_shapes": [list(x.shape) for x in inputs],
                    "weight_memory": str(weight.memory_config()),
                    "weight_dtype": str(weight.dtype),
                    "compute_fidelity": str(layer.matmul_compute(role).math_fidelity),
                    "production_program": str(production),
                }
                trace, outputs, capture_started = None, None, False
                try:
                    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block,
                        per_core_M=production.per_core_M,
                        per_core_N=production.per_core_N,
                        num_workers_per_dram_bank=readers,
                        use_multicore_pipeline=readers > 1,
                    )
                    layer.dram_layouts[role] = (im, om, program)
                    row["program"] = str(program)
                    outputs = [layer.decode_linear(x, role) for x in inputs]
                    row["eager_numerical"] = compare(expected, outputs)
                    if not acceptable(row["eager_numerical"]):
                        row["status"] = "rejected_numerical"
                        continue
                    outputs = None
                    gc.collect()
                    ttnn.synchronize_device(mesh)
                    row["programs_before_capture"] = mesh.num_program_cache_entries()
                    capture_started = True
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    outputs = [layer.decode_linear(x, role) for x in inputs]
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    row["programs_after_capture"] = mesh.num_program_cache_entries()
                    assert row["programs_before_capture"] == row["programs_after_capture"], "Compilation during capture"
                    for output in outputs:
                        trace_allocation_tracker.acknowledge_corruptible(output)
                    for _ in range(4):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    timings = []
                    for _ in range(args.repetitions):
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        timings.append(1000 * (time.perf_counter() - start) / (args.iterations * len(inputs)))
                    row["traced_numerical"] = compare(expected, outputs)
                    row.update(
                        status="pass" if acceptable(row["traced_numerical"]) else "rejected_trace_numerical",
                        ms_per_projection=timings,
                        median_ms=statistics.median(timings),
                        replay_blocking=False,
                    )
                except Exception as error:
                    message = str(error)
                    row.update(status="error", error=message)
                    # Continue only pre-capture compile-time constraints. Never
                    # classify device faults or an incomplete capture as legal.
                    if not capture_started and any(
                        marker in message
                        for marker in (
                            "beyond max L1 size",
                            "clash with L1 buffers",
                            "must be divisible by workers_per_bank",
                            "wider than the in0 shard",
                            "must divide one another",
                        )
                    ):
                        row["status"] = "rejected_factory_constraint"
                    else:
                        raise
                finally:
                    if trace is not None:
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    outputs = None
                    layer.dram_layouts[role] = entry["original"]
                    gc.collect()
                    result["rows"].append(row)
                    save()
                    print("SELECTED_DECODER_GEOMETRY", json.dumps(row), flush=True)
                if row["baseline"] and row["status"] != "pass":
                    raise AssertionError(f"Production baseline failed: {index}/{role}")
            role_rows = [r for r in result["rows"] if r["layer"] == index and r["role"] == role]
            assert role_rows[0]["status"] == role_rows[-1]["status"] == "pass"
            inputs = None
            gc.collect()
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="error", error=repr(error))
        raise
    finally:
        save()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
