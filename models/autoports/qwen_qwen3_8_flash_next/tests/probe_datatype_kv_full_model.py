# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B32 full-vocabulary KV precision diagnostic through traced public APIs.

All 48 layers; 32 identical book table-of-contents 33-token prefixes and two identical
teacher-forced continuation tokens. Capacity128/256 is diagnostic only, never
advertised-context acceptance. No sampling, hidden-state argmax, performance,
HF accuracy, migration, or independent-prompt quality claim.

Parent owns device serialization. Run BF16 and BFP8 commands separately:
  TT_VISIBLE_DEVICES=0,1,2,3 TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
  timeout --signal=INT --kill-after=60 3600 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_datatype_kv_full_model \
  --precision-config <BF16-policy.json> --output <bf16-b32.json> --logits <bf16-b32.pt>

Repeat with BFP8-policy and separate outputs. Pass these .pt files as
--bf16-logits/--candidate-logits to analyze_datatype_kv_controls alongside its
paired decoder host captures. Each .pt contains logits [32,3,248320] and exact
input_token_ids [32,35]. Keep large logit files on persistent local storage,
not in source commits. Default32-way repeated input isolates per-user cache
behavior; it is not32 independent prompts or a throughput measurement.
"""

import argparse
import bz2
import json
import os
import time
import traceback
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"
)
BOOK = MODEL_DIR.parents[1] / "ttt_compat/tests/tale-of-two-cities.txt.bz2"


def make_inputs(tokenizer, book):
    """Host-only deterministic book inputs; never depend on TT predictions."""
    import torch

    with bz2.open(book, "rt", encoding="utf-8") as source:
        text = source.read(32768)
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Offset256 is table-of-contents text in this pinned tokenizer/book pair.
    # Preserve these exact historical diagnostic inputs; this is not the narrative readiness reference.
    selected = ids[256 : 256 + 35]
    if len(selected) != 35:
        raise ValueError("Book source must supply 35 tokens after offset256")
    full = torch.tensor(selected, dtype=torch.int64).unsqueeze(0).repeat(32, 1)
    return full[:, :33].clone(), full[:, 33:].clone(), full


def validate_snapshot(logits, input_ids):
    """Host shape/finite gate; vocabulary token argmax comes only from logits."""
    import torch

    if logits.ndim != 3 or tuple(logits.shape[:2]) != (32, 3) or logits.shape[-1] != 248320:
        raise ValueError("Expected all32 users, prefill+two decode outputs, and the complete248320-token vocabulary")
    if tuple(input_ids.shape) != (32, 35) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid input IDs or nonfinite vocabulary logits")
    return logits.argmax(dim=-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--capacity", type=int, choices=(128, 256), default=128)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--book", type=Path, default=BOOK)
    args = parser.parse_args()
    if args.output.exists() or args.logits.exists():
        parser.error("Use new output paths; prior diagnostic evidence is immutable")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache allocations must be tracked")
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        parser.error("Use the accepted mesh descriptor")
    for flag in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
        if os.environ.get(flag) not in (None, "", "0"):
            parser.error("Run this full-model diagnostic without profiler collection")

    import torch
    from transformers import AutoTokenizer

    from models.autoports.qwen_qwen3_8_flash_next.tt.precision import load_policy

    policy = load_policy(MODEL_DIR, args.precision_config)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts, forced, full_ids = make_inputs(tokenizer, args.book)
    torch.set_num_threads(8)

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.shared_pool_layout import shared_pool_layout
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

    result = {
        "passed": False,
        "scope": "all48_B32_same_input_short_prefix_KV_dtype_vocabulary_logit_diagnostic",
        "full_context_acceptance": False,
        "hf_accuracy_claim": False,
        "performance_claim": False,
        "capacity": args.capacity,
        "advertised_context_unchanged": 262144,
        "batch": 32,
        "prompt_length": 33,
        "decode_steps": 2,
        "source_prompt_count": 1,
        "precision": policy,
        "checkpoint": str(args.checkpoint),
        "book": str(args.book),
        "book_token_offset": 256,
        "prompt_format": "book table-of-contents teacher forcing; tokenizer.encode(add_special_tokens=False); not narrative readiness or chat evaluation",
        "prompt_content_class": "table_of_contents",
        "input_token_ids": full_ids.tolist(),
        "logits_artifact": str(args.logits),
        "trace_allocation_tracking": True,
        "skip_program_cache": False,
        "measurement_regime": "serial correctness with host vocabulary-logit readbacks",
    }
    mesh = generator = None
    phase = "setup"
    started = time.monotonic()

    def mark(name):
        nonlocal phase
        phase = name
        print(json.dumps({"phase": phase, "wall_seconds": time.monotonic() - started}), flush=True)

    def trace_ids():
        return {
            name: {repr(key): str(value) for key, value in getattr(generator, name).items()}
            for name in ("traces", "prefill_traces", "sampling_traces")
        }

    def memory():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return {
            "banks": view.num_banks,
            "allocated_bytes_per_rank": view.total_bytes_allocated_per_bank * view.num_banks,
            "free_bytes_per_rank": view.total_bytes_free_per_bank * view.num_banks,
        }

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        mark("load_all48")
        generator = build_generator(
            MODEL_DIR,
            mesh,
            checkpoint=str(args.checkpoint),
            precision_config=policy,
            max_batch_size=32,
            max_seq_len=args.capacity,
            defer_kv_cache=True,
        )
        assert tuple(generator.model.layer_indices) == tuple(range(48))
        assert generator.factored_decode is not None
        result["runtime_precision"] = generator.model.precision_summary()
        result["memory_loaded"] = memory()
        specs, tables = shared_pool_layout(32, args.capacity, 12)
        specs = [(shape, policy["kv_cache_dtype"], index) for shape, _, index in specs]
        pool = generator.allocate_kv_cache_per_layer(specs)
        result["pool_bytes_per_rank"] = pool.bytes_per_rank
        result["cache_spec"] = generator.cache_spec()
        mark("prepare_all_traces")
        generator.prepare()
        result["memory_prepared"] = memory()
        result["counters_before"] = dict(generator.counters)
        result["trace_ids_after_prepare"] = trace_ids()
        assert generator.traces, "No decode traces were captured"
        slots = tuple(range(32))
        request = {"page_table": tables[0], "page_tables_per_layer": tables, "kv_cache": pool, "slots": slots}
        mark("prefill_33_all32")
        prefill = generator.prefill_forward(prompts, prompt_lens=[33] * 32, **request)
        assert tuple(prefill.shape) == (32, 1, generator.model.config.vocab_size)
        outputs = [prefill[:, 0].float().clone()]
        result["trace_ids_after_prefill"] = trace_ids()
        decode_before = dict(generator.counters)
        for step in range(2):
            mark(f"traced_forced_decode_{step}")
            output = generator.decode_forward(
                forced[:, step : step + 1],
                [33 + step] * 32,
                sample_on_device=False,
                read_from_device=True,
                **request,
            )
            assert tuple(output.shape) == (32, generator.model.config.vocab_size)
            outputs.append(output.float().clone())
        logits = torch.stack(outputs, dim=1)
        result["token_top1"] = validate_snapshot(logits, full_ids)
        result["final_positions"] = [cache.position for cache in generator.caches]
        result["counters_after"] = dict(generator.counters)
        result["trace_ids_after_decode"] = trace_ids()
        result["decode_counter_delta"] = {
            name: generator.counters[name] - decode_before[name] for name in generator.counters
        }
        assert result["final_positions"] == [35] * 32
        assert result["decode_counter_delta"]["model_replays"] == 2
        assert result["decode_counter_delta"]["captures"] == 0
        assert result["decode_counter_delta"]["releases"] == 0
        assert result["trace_ids_after_prefill"] == result["trace_ids_after_decode"]
        assert generator.counters["prefill_replays"] - result["counters_before"]["prefill_replays"] == 64
        assert generator.prefill_traces, "No public prefill traces were used"
        payload = {
            "input_token_ids": full_ids,
            "logits": logits,
            "precision": policy,
            "capacity": args.capacity,
            "scope": result["scope"],
            "traced": True,
            "phases": ["prefill33", "forced_decode_position33", "forced_decode_position34"],
        }
        args.logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.logits)
        result["memory_final"] = memory()
        result["passed"] = True
        mark("snapshot_saved")
    except BaseException as error:
        result.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        if generator is not None:
            result["counters_at_exit"] = dict(generator.counters)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(json.dumps({"passed": result["passed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
