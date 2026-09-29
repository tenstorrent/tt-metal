# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer batched decode timing/profile with full-width page tables."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32, choices=(1, 2, 3, 8, 32))
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument(
        "--candidate",
        choices=(
            "shared",
            "attention",
            "attention_rowsdpa",
            "norms",
            "runtime",
            "experts",
            "experts_vector_mix",
            "experts_index_union",
        ),
    )
    parser.add_argument(
        "--mixed", action="store_true", help="Heterogeneous boundary lengths and shuffled physical pages"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.profile and args.all_layers:
        parser.error("Profiling is restricted to reduced layers 0/5")
    assert 1 <= args.input_length < 8192 - 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "scope": (
            "Full model/head/sampler; synthetic diverse prompts; generator timing, not serving TSU"
            if args.all_layers
            else "Reduced real layers 0/5, full head/sampler; synthetic diverse prompts; not a full-model TSU claim"
        ),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                Path(__file__).with_name("tsu_batch_candidate.py"),
                *sorted((root / "tt").glob("*.py")),
            ]
        },
        "rows": [],
    }
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=262144, layer_indices=None if args.all_layers else (0, 5))
        # Candidate/reference comparisons must not inherit the selected default.
        for decoder in gen.model.layers:
            decoder.batched_shared_decode = False
        report["baseline_batched_shared_decode"] = False
        cache, table = gen.model.allocate_cache(slots=32, context=8192)
        if args.mixed:
            torch.manual_seed(142)
            table = torch.randperm(table.numel(), dtype=torch.int32).reshape(table.shape)
        table = torch.nn.functional.pad(table[: args.batch], (0, 8192 - table.shape[1]))
        lengths = [
            (31, 63, 127, 128, 255, 256, 1023, 4095)[slot % 8] if args.mixed else args.input_length
            for slot in range(args.batch)
        ]
        prompts = torch.stack([torch.tensor([2] + [100 + slot] * (max(lengths) - 1)) for slot in range(args.batch)])
        gen.prefill_forward(prompts, page_table=table, kv_cache=cache, prompt_lens=lengths)
        positions = torch.tensor(lengths, dtype=torch.int32)
        tokens = torch.arange(100, 100 + args.batch, dtype=torch.int32)
        gen.decode_forward(tokens, positions, page_table=table, kv_cache=cache)
        ttnn.synchronize_device(mesh)
        if args.candidate:
            from models.autoports.google_gemma_4_26b_a4b_it.tests.tsu_batch_candidate import install_shared_batch

            if not args.profile:
                report["reference_rows"] = []
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    gen._replay()
                    ttnn.synchronize_device(mesh)
                    report["reference_rows"].append(
                        {"repeat": repeat, "mean_step_ms": (time.perf_counter() - started) * 1000}
                    )
                gen.decode_forward(tokens, positions, page_table=table, kv_cache=cache)
            vocab = gen.model.config.vocab_size
            reference = gen._read_logits(gen.trace_logits).reshape(-1, vocab)[: args.batch].clone()
            reference_steps = [reference]
            if args.mixed:
                for step in (1, 2):
                    gen.decode_forward(tokens.flip(0) + step, positions + step, page_table=table, kv_cache=cache)
                    reference_steps.append(gen._read_logits(gen.trace_logits).reshape(-1, vocab)[: args.batch].clone())
            gen._release_trace()
            if args.candidate == "runtime":
                for decoder in gen.model.layers:
                    decoder.batched_shared_decode = True
            else:
                install_shared_batch(
                    gen.model,
                    batch_attention=args.candidate.startswith("attention"),
                    batch_norms=args.candidate == "norms",
                    row_sdpa=args.candidate == "attention_rowsdpa",
                    batch_experts=args.candidate.startswith("experts"),
                    vector_mix=args.candidate in ("experts_vector_mix", "experts_index_union"),
                    indexed_union=args.candidate == "experts_index_union",
                )
            # Restore original prompt KV before replaying the same multi-step
            # sequence; old future rows are masked by each supplied position.
            gen.prefill_forward(prompts, page_table=table, kv_cache=cache, prompt_lens=lengths)
            gen.decode_forward(tokens, positions, page_table=table, kv_cache=cache)
            report["correctness"] = []
            for step, reference in enumerate(reference_steps):
                if step:
                    gen.decode_forward(tokens.flip(0) + step, positions + step, page_table=table, kv_cache=cache)
                actual = gen._read_logits(gen.trace_logits).reshape(-1, vocab)[: args.batch].clone()
                report["correctness"] += [
                    {
                        "step": step,
                        "slot": slot,
                        "pcc": float(torch.corrcoef(torch.stack((ref.float(), got.float())))[0, 1]),
                        "top1": int(ref.argmax()) == int(got.argmax()),
                        "max_abs": float((ref - got).abs().max()),
                    }
                    for slot, (ref, got) in enumerate(zip(reference, actual))
                ]
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert all(row["pcc"] >= 0.999 and row["top1"] for row in report["correctness"]), report["correctness"]
        if args.profile:
            from tracy import signpost

            signpost("PERF_DECODE")
            gen._replay()
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)
        else:
            for repeat in range(args.repeats):
                started = time.perf_counter()
                gen._replay()
                ttnn.synchronize_device(mesh)
                row = {"repeat": repeat, "mean_step_ms": (time.perf_counter() - started) * 1000}
                report["rows"].append(row)
                print("BATCH_TRACE", json.dumps(row), flush=True)
        report["cache_shapes"] = [[list(t.shape) for t in pair] for pair in cache]
        report["table_shape"] = list(table.shape)
        report["counters"] = gen.counters
        report["final_tokens"] = list(gen._read_tokens())
        report["precision_runtime"] = gen.model.precision_summary()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
