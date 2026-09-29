"""Bounded setup diagnostic for full-model host sampling; not benchmark scores."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample

import ttnn

from ..tt.generator_vllm import K2HorizonForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    os.environ["K2_VLLM_ALLOW_HOST_SAMPLING"] = "1"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    result = dict(
        scope="Setup diagnostic, full36 selected model, bounded diagnostic cache; not benchmark results",
        completed=False,
        threads=torch.get_num_threads(),
        records=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    mesh = adapter = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        adapter = K2HorizonForCausalLM(mesh, max_batch_size=32, num_layers=36)
        model = adapter.generator.model
        result.update(layers=model.num_layers, precision=model.precision_config, context=model.context)
        prompts = []
        for row in range(32):
            messages = [
                {
                    "role": "user",
                    "content": f"A library has {19 + row} shelves. Each shelf holds {23 + row} books. "
                    f"After lending {41 + row} books and receiving {12 + row} new books, "
                    "how many books remain? Explain your calculation carefully.",
                }
            ]
            ids = model.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, reasoning_effort="high", return_dict=False
            )
            prompts.append(dict(messages=messages, token_ids=ids))
        (args.output / "inputs.json").write_text(json.dumps(prompts, indent=2) + "\n")
        lengths = [len(x["token_ids"]) for x in prompts]
        if max(lengths) + args.steps >= 512:
            raise ValueError("Diagnostic cache is too short")
        tokens = torch.zeros((32, max(lengths)), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            tokens[row, : lengths[row]] = torch.tensor(prompt["token_ids"])
        table = torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16)
        cache = adapter.allocate_kv_cache((32 * 16, 2, 32, 128), torch.bfloat16, 36)
        generators = {row: torch.Generator().manual_seed(1234 + row) for row in range(32)}
        k = torch.full((32,), model.vocab_size, dtype=torch.int32)
        p = torch.full((32,), 0.95)

        def sample(logits, step):
            if step in (-1, 7, args.steps - 1):
                path = args.output / f"logits-{step}.pt"
                torch.save(logits.clone(), path)
            start = time.perf_counter()
            filtered = apply_top_k_top_p(logits.float(), k, p)
            sorted_at = time.perf_counter()
            probs = filtered.softmax(-1, dtype=torch.float32)
            softmax_at = time.perf_counter()
            next_tokens = random_sample(probs, generators).reshape(32, 1)
            sampled_at = time.perf_counter()
            return next_tokens, dict(
                sort_filter_ms=(sorted_at - start) * 1000,
                softmax_ms=(softmax_at - sorted_at) * 1000,
                random_sample_ms=(sampled_at - softmax_at) * 1000,
            )

        logits = adapter.prefill_forward(tokens, table, cache, lengths)
        tokens, timing = sample(logits[:, -1], -1)
        result["records"].append(dict(step=-1, **timing))
        positions = torch.tensor(lengths, dtype=torch.int32)
        for step in range(args.steps):
            begin = time.perf_counter()
            device_output = adapter.decode_forward(
                tokens, positions, table, cache, reset_batch=True, read_from_device=False
            )
            submitted = time.perf_counter()
            host, events = adapter.read_decode_output(device_output, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            ready = time.perf_counter()
            logits = adapter.process_decode_output_host(host, is_tokens=False)[:, -1]
            converted = time.perf_counter()
            tokens, timing = sample(logits, step)
            row = dict(
                step=step,
                submit_ms=(submitted - begin) * 1000,
                readback_completion_ms=(ready - submitted) * 1000,
                conversion_ms=(converted - ready) * 1000,
                **timing,
            )
            result["records"].append(row)
            print(json.dumps(row), flush=True)
            positions += 1
        result["generator_counters"] = dict(adapter.generator.counters)
        result["completed"] = True
    finally:
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        if adapter is not None:
            adapter.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
