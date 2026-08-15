# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager vs traced serving prefill on the **real** qualitative prompts, one process.

The optimized-vLLM after-arm's served qualitative output came back as replacement
characters where the vLLM-integration arm's was coherent English, on the same pinned
64-token prompts.  Two things changed at once in that arm -- the serving prefill got a
captured trace, and the prefill page table became a single slot row with
``user_id=0`` -- so the first job is to say which, and the second is to say whether it
is the *prompt content* or the *bucket* that trips it.  A live server cannot answer
either cheaply: it costs a four-minute load per arm and mixes in the scheduler.

This drives the vLLM adapter directly, in one process, over one build:

    A. warm up with prefill tracing off, generate greedily from each pinned prompt;
    B. enable prefill tracing, capture the buckets, generate from the same prompts
       into *different* cache slots and blocks;
    C. compare token ids.

Both arms therefore share the build, the weights, the KV cache, the decode trace and
the sampler.  The only difference is the prefill path, which is what makes a
divergence attributable.

Usage::

    python doc/optimized_vllm/bench/prefill_trace_bisect.py --out doc/optimized_vllm/prefill_trace_bisect.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

MODEL_DOC = Path(__file__).resolve().parents[2]


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="", help="reduced layer indices, e.g. '0,3'")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gen-len", type=int, default=24)
    parser.add_argument("--prompts", type=int, default=3, help="how many pinned prompts to run")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    import os

    if args.layers:
        os.environ["MUSE_GLIMMER_VLLM_LAYER_INDICES"] = args.layers
    # Build with tracing off; arm B turns it on through the generator seam.
    os.environ["MUSE_GLIMMER_VLLM_PREFILL_TRACE"] = "0"

    from transformers import AutoConfig, AutoTokenizer

    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
        DEFAULT_TRACE_REGION_SIZE,
        close_generator_mesh,
        open_generator_mesh,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import PREFILL_TRACE_BUCKETS
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import (
        MuseGlimmerForConditionalGeneration as Adapter,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import HF_MODEL_ID, weights_snapshot_dir
    from models.common.sampling.generator import SamplingParams

    pinned = json.loads((MODEL_DOC / "full_model/qualitative/qualitative_prompts.json").read_text())[: args.prompts]
    tokenizer = AutoTokenizer.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)

    block_size = 64
    hf_config = AutoConfig.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)
    hf_config._name_or_path = HF_MODEL_ID

    report: dict = {
        "prompts": [{"id": p["id"], "prompt_tokens": len(p["token_ids"])} for p in pinned],
        "gen_len": args.gen_len,
        "layers": args.layers or "all",
        "buckets": list(PREFILL_TRACE_BUCKETS),
    }
    mesh = open_generator_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        model = Adapter.initialize_vllm_model(hf_config, mesh, args.max_num_seqs, max_seq_len=args.max_model_len)
        num_layers = len(model.generator.model.layers)
        tokens_all_users = Adapter.get_max_tokens_all_users(
            model_name=HF_MODEL_ID,
            num_devices=mesh.get_num_devices(),
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
        num_blocks = -(-(tokens_all_users + block_size * args.max_num_seqs) // block_size)
        kv_cache = model.allocate_kv_cache((num_blocks, 1, block_size, 128), torch.bfloat16, num_layers)
        blocks_per_seq = -(-args.max_model_len // block_size)

        model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
        for enable in (False, True):
            model.warmup_model_decode(
                kv_cache=kv_cache,
                max_batch_size=args.max_num_seqs,
                num_blocks=blocks_per_seq,
                can_sample_on_device=True,
                enable_trace=enable,
            )
        report["prefill_trace_after_warmup"] = list(model.generator.prefill_trace_buckets)

        def greedy(rows: int):
            return SamplingParams(temperature=[0.0] * rows, top_k=[1] * rows, top_p=[1.0] * rows, seed=[None] * rows)

        base_block = [1]

        def generate(ids: list[int], slot: int) -> list[int]:
            """One request through the adapter: prefill, then ``gen_len`` decode steps."""
            need = -(-(len(ids) + args.gen_len) // block_size) + 1
            first = base_block[0]
            base_block[0] += need
            row = torch.arange(first, first + need, dtype=torch.int32).reshape(1, need)
            out = model.prefill_forward(
                tokens=torch.tensor([ids], dtype=torch.int32),
                page_table=row,
                kv_cache=kv_cache,
                enable_trace=False,
                prompt_lens=[len(ids)],
                start_pos=torch.zeros(1, dtype=torch.int32),
                sampling_params=greedy(1),
                empty_slots=[slot],
            )
            emitted = [int(out[0])]
            table = torch.zeros(args.max_num_seqs, blocks_per_seq, dtype=torch.int32)
            table[slot, :need] = row[0]
            positions = torch.full((args.max_num_seqs,), -1, dtype=torch.int32)
            positions[slot] = len(ids)
            decode_tokens = torch.zeros(args.max_num_seqs, 1, dtype=torch.int32)
            decode_tokens[slot, 0] = emitted[0]
            for step in range(args.gen_len - 1):
                got = model.decode_forward(
                    tokens=decode_tokens,
                    page_table=table,
                    kv_cache=kv_cache,
                    start_pos=positions,
                    enable_trace=True,
                    read_from_device=False,
                    sampling_params=greedy(args.max_num_seqs),
                    reset_batch=(step == 0),
                )
                host, events = model.read_decode_output(got, async_read=True)
                import ttnn

                for event in events:
                    ttnn.event_synchronize(event)
                sampled = model.process_decode_output_host(host, is_tokens=True)
                emitted.append(int(sampled[slot]))
                decode_tokens[slot, 0] = emitted[-1]
                positions[slot] = len(ids) + step + 1
            return emitted

        rows = []
        eager = {}
        for index, prompt in enumerate(pinned):
            eager[prompt["id"]] = generate([int(t) for t in prompt["token_ids"]], slot=index)

        # --- arm B: same prompts, prefill traced --------------------------------
        buckets = [b for b in PREFILL_TRACE_BUCKETS if b <= model.generator.model.config.max_seq_len]
        model.generator.enable_prefill_trace(max_entries=len(buckets), max_padded_len=max(buckets))
        for length in buckets:
            model.generator.prefill_forward(
                torch.zeros(1, length, dtype=torch.int32),
                page_table=None,
                kv_cache=kv_cache,
                prompt_lens=[length],
                sample_on_device=True,
            )
        report["prefill_trace_buckets_resident"] = list(model.generator.prefill_trace_buckets)

        for index, prompt in enumerate(pinned):
            traced = generate([int(t) for t in prompt["token_ids"]], slot=index + len(pinned))
            want = eager[prompt["id"]]
            diverge = next((i for i in range(min(len(want), len(traced))) if want[i] != traced[i]), None)
            rows.append(
                {
                    "id": prompt["id"],
                    "prompt_tokens": len(prompt["token_ids"]),
                    "padded_len": ((len(prompt["token_ids"]) + 31) // 32) * 32,
                    "bucket_traced": ((len(prompt["token_ids"]) + 31) // 32) * 32 in buckets,
                    "identical": want == traced,
                    "first_divergence": -1 if diverge is None else diverge,
                    "eager_tokens": want,
                    "traced_tokens": traced,
                    "eager_text": tokenizer.decode(want),
                    "traced_text": tokenizer.decode(traced),
                }
            )
        report["comparison"] = rows
        report["all_identical"] = all(row["identical"] for row in rows)
        report["status"] = "ok"
    finally:
        try:
            model.generator.teardown()
        except Exception:  # noqa: BLE001
            pass
        close_generator_mesh(mesh)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparison"}, indent=2))
    for row in report.get("comparison", []):
        print(f"{row['id']} identical={row['identical']} first_div={row['first_divergence']}")
        print(f"   eager : {row['eager_text'][:110]!r}")
        print(f"   traced: {row['traced_text'][:110]!r}")
    return 0 if report.get("all_identical") else 1


if __name__ == "__main__":
    raise SystemExit(main())
