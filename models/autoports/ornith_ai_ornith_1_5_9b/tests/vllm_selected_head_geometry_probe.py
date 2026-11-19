# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Selected-policy terminal norm/head geometry search; no serving performance claim.

Run with exclusive hardware under an external timeout. Natural-text residuals
come from layers 0/3 with the full 262144 context allocation. Decoder execution
only prepares fixed inputs; timing measures production terminal norm + unsplit
LMHead1D projection/conversion/concat on those same inputs for every candidate.
There is no sampler, decoder topology search, profiler, or runtime source edit.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if min(args.iterations, args.repetitions) <= 0:
        parser.error("iterations and repetitions must be positive")
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("0", ""):
        parser.error("Disable device profiling for this probe")

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
        "batch": 1,
        "prompt_tokens": 128,
        "decode_position": 128,
        "timed_scope": "Production terminal norm plus unsplit LMHead1D, three fixed natural-text reduced-stack residuals",
        "serving_or_full_decode_throughput": False,
        "tracking": trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "timing_eligible": not trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "factory_source": "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp",
        "larger_K_scope": "All larger divisors of 128 K tiles: 32,64,128. K>128 exceeds complete reduction dimension.",
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen, trace = None, None
    try:
        policy = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/configs/bfp8_all_lofi.json"
        gen = build_generator(
            None, mesh, layer_indices=[0, 3], max_batch_size=1, max_seq_len=262144, precision_config=policy
        )
        model, hidden_inputs = gen.model, []
        assert model.head.dtype == ttnn.bfloat8_b
        assert model.compute.math_fidelity == ttnn.MathFidelity.LoFi
        assert model.norm_compute.math_fidelity == ttnn.MathFidelity.HiFi2
        head = model.decode_head
        assert len(head.output_weights) == len(head.config.program_configs) == 1
        original_programs = head.config.program_configs
        production_pc = original_programs[0]
        target_N = math.ceil((model.vocab_size // 4) / (64 * 32))
        assert production_pc.in0_block_w == 2 and production_pc.per_core_M == 1
        assert production_pc.per_core_N == target_N and production_pc.num_workers_per_dram_bank == 3
        assert production_pc.use_multicore_pipeline
        assert head.output_weights[0].dtype == ttnn.bfloat8_b
        assert head.config.lm_head_dtype == ttnn.bfloat16
        result["actual_policy"] = model.precision_summary()
        result["terminal_norm_compute"] = str(model.norm_compute)
        result["head_compute"] = str(model.compute)
        result["production_program"] = str(production_pc)
        result["unsplit_weight_shape"] = list(head.output_weights[0].shape)
        result["weight_memory"] = str(head.output_weights[0].memory_config())
        result["input_memory"] = str(model.collectives.ar_mem)
        result["output_memory"] = str(head.config.output_memcfg)
        result["memory_full_context"] = model.memory_view()
        natural = [
            "Explain how a computer learns patterns from examples and uses them to make predictions. Include a simple example and describe how to check whether it generalizes.",
            "A gardener plants tomatoes in spring. Explain how sunlight, watering, and soil affect growth, then describe a sensible way to compare two watering schedules.",
            "Write a Python function that counts words in a sentence. Explain its inputs and output, and discuss what should happen when the sentence is empty.",
        ]
        result["stimuli"] = []
        for text in natural:
            encoded = gen.tokenizer.encode(text, add_special_tokens=False)
            ids = (encoded * math.ceil(129 / len(encoded)))[:129]
            gen.reset(clear_kv=True)
            gen.prefill_forward(
                torch.tensor([ids[:128]]), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[128]
            )
            token = model.upload(torch.tensor([[ids[128]]]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            position = model.upload(torch.tensor([128]), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            table = model.upload(gen.page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            original_logits = model.logits

            def retain_residual(hidden, *, decode=False):
                assert decode
                hidden_inputs.append(ttnn.clone(hidden))
                return original_logits(hidden, decode=decode)

            with patch.object(model, "logits", side_effect=retain_residual):
                output = model.decode(token, cache=gen.cache, slots=[0], positions=[position], page_tables=[table])
            del output, token, position, table
            result["stimuli"].append(
                {"text": text, "token_ids": ids, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
            )
        assert len(hidden_inputs) == 3
        assert all(x.memory_config() == model.collectives.ar_mem for x in hidden_inputs)
        result["residual_shapes"] = [list(x.shape) for x in hidden_inputs]
        result["residual_dtypes"] = [str(x.dtype) for x in hidden_inputs]
        assert all(x.dtype == ttnn.bfloat16 for x in hidden_inputs)
        reference = []
        for hidden in hidden_inputs:
            output = model.logits(hidden, decode=True)
            reference.append(model.host_logits(output).clone())
            del output

        def compare(outputs):
            rows = []
            for expected, output in zip(reference, outputs, strict=True):
                actual = model.host_logits(output)
                a, b = actual.double().reshape(-1), expected.double().reshape(-1)
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                exact = torch.equal(actual, expected)
                pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if finite else None
                rows.append(
                    {
                        "finite": finite,
                        "exact": exact,
                        "pcc": pcc,
                        "max_abs": float((a - b).abs().max()) if finite else None,
                        "top1_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
                    }
                )
            return rows

        candidates = [(3, 2)] + [(r, k) for r in (1, 2, 3, 4) for k in (2, 4, 8, 16, 32, 64, 128) if (r, k) != (3, 2)]
        # Repeat production last to expose clock/load drift across the search.
        candidates.append((3, 2))
        for candidate_index, (readers, block) in enumerate(candidates):
            row = {
                "id": f"r{readers}_k{block}",
                "readers": readers,
                "K_block_tiles": block,
                "target_cores": 64,
                "per_core_N": target_N,
                "baseline_repeat": candidate_index == len(candidates) - 1,
            }
            outputs = None
            trace = None
            try:
                assert 128 % block == 0
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=target_N,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=readers > 1,
                )
                head.config.program_configs = [pc]
                row["program"] = str(pc)
                outputs = [model.logits(hidden, decode=True) for hidden in hidden_inputs]
                row["eager_numerical"] = compare(outputs)
                if not all(x["finite"] and x["pcc"] >= 0.999 and x["top1_equal"] for x in row["eager_numerical"]):
                    row["status"] = "rejected_numerical"
                    continue
                del outputs
                outputs = None
                gc.collect()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                outputs = [model.logits(hidden, decode=True) for hidden in hidden_inputs]
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                # These are exact trace-produced final outputs, read only after
                # replay completes. No replay input aliases them; next candidate
                # allocations occur only after this trace has been released.
                for output in outputs:
                    trace_allocation_tracker.acknowledge_corruptible(output)
                for _ in range(4):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times = []
                for _ in range(args.repetitions):
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1000 / (args.iterations * len(hidden_inputs)))
                row["traced_numerical"] = compare(outputs)
                row["status"] = (
                    "pass"
                    if all(x["finite"] and x["pcc"] >= 0.999 and x["top1_equal"] for x in row["traced_numerical"])
                    else "rejected_trace_numerical"
                )
                row.update(ms_per_norm_head=times, median_ms=statistics.median(times), replay_blocking=False)
            except Exception as error:
                message = str(error)
                row.update(status="error", error=message)
                # Continue only after a specific compile-time geometry/capacity
                # rejection; unknown errors require inspection, not a new run.
                if any(
                    marker in message
                    for marker in (
                        "beyond max L1 size",
                        "clash with L1 buffers",
                        "must be divisible by workers_per_bank",
                        "wider than the in0 shard",
                    )
                ):
                    row["status"] = "rejected_factory_constraint"
                else:
                    raise
            finally:
                if trace is not None:
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace)
                    trace = None
                outputs = None
                head.config.program_configs = original_programs
                gc.collect()
                result["rows"].append(row)
                save()
                print("SELECTED_HEAD_GEOMETRY", json.dumps(row), flush=True)
        assert result["rows"][0]["status"] == result["rows"][-1]["status"] == "pass", "Production baseline failed"
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
