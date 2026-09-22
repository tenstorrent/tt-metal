# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare full normalized logit distributions across live serving requests.

The API explicitly requests host-compatible all-vocabulary logprobs. This probe
is correctness evidence only, never a performance workload.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True, help="Temporary full-vector comparison data")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--standalone-control", type=Path, help="Serving report to compare after stopping its server")
    args = parser.parse_args()
    if args.standalone_control:
        standalone(args)
        return
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True)
    vocab_size = config.get_text_config().vocab_size
    prompts = ["Name the capital of France.", "What is seven plus five?"]

    def request(index):
        messages = [{"role": "user", "content": prompts[index]}]
        payload = dict(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=-1,
            return_tokens_as_token_ids=True,
        )
        response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        values = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        vector = np.full(vocab_size, np.nan, dtype=np.float32)
        for item in values:
            vector[int(item["token"].removeprefix("token_id:"))] = item["logprob"]
        assert np.isfinite(vector).all(), "Missing vocabulary logprobs"
        return vector, dict(
            prompt_id=index,
            messages=messages,
            prompt_token_ids=tokenizer.encode(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                add_special_tokens=False,
            ),
            usage=data["usage"],
            emitted_token=data["choices"][0]["logprobs"]["content"][0]["token"],
            sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            top20_ids=np.argsort(vector)[-20:][::-1].tolist(),
        )

    baseline = [request(i) for i in range(2)]
    rows = [row for _, row in baseline]
    differences = []
    for order in ([0, 1], [1, 0]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(request, order))
        for index, (vector, row) in zip(order, result):
            differences.append(float(np.max(np.abs(vector - baseline[index][0]))))
            rows.append(row)
    np.savez(args.vectors, **{f"prompt_{i}": v for i, (v, _) in enumerate(baseline)})
    report = dict(
        mode="explicit host compatibility; full-vocabulary normalized logits",
        endpoint="/v1/chat/completions",
        chat_template_present=bool(tokenizer.chat_template),
        rows=rows,
        max_abs_logprob_differences=differences,
        exact_reproducibility=all(x == 0 for x in differences),
        temporary_vectors=str(args.vectors),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


def standalone(args):
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric

    source = json.loads(args.standalone_control.read_text())
    prompts = [row["prompt_token_ids"] for row in source["rows"][:2]]
    expected = np.load(args.vectors)
    root = Path(__file__).resolve().parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    rows = []
    try:
        gen = build_generator(root, mesh, precision_config=root / "doc/datatype_sweep/selected_precision_config.json")
        cache = gen._ensure_cache(2, 128)
        for order in ([0, 1], [1, 0]):
            gen.reset()
            for slot, index in enumerate(order):
                result = gen.prefill_forward(
                    torch.tensor([prompts[index]]),
                    kv_cache=cache,
                    page_table=gen.page_table,
                    prompt_lens=[len(prompts[index])],
                    slots=[slot],
                )
                logits = gen._host_logits(result[0]).reshape(-1)[: len(expected[f"prompt_{index}"])]
                logprobs = torch.log_softmax(logits, dim=-1).numpy()
                rows.append(
                    dict(
                        prompt_id=index,
                        slot=slot,
                        max_abs_serving_logprob_difference=float(
                            np.max(np.abs(logprobs - expected[f"prompt_{index}"]))
                        ),
                        sha256=hashlib.sha256(logprobs.tobytes()).hexdigest(),
                        top20_ids=np.argsort(logprobs)[-20:][::-1].tolist(),
                    )
                )
        report = dict(mode="full-model standalone selected-policy host-logit control", rows=rows, cache_capacity=128)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
