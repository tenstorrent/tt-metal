"""Reconstruct seeded upstream vLLM performance inputs; no inference or HTTP."""

import hashlib
import json
from pathlib import Path

from vllm.benchmarks.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer

RUN = Path(__file__).resolve().parents[1] / "doc/benchmark/run"
TOKENIZER = "/mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/036114ce8d46c32b24c15423211069abb9c5d25e"


def main():
    tokenizer = get_tokenizer(TOKENIZER, trust_remote_code=True)
    records = []
    for concurrency in (32, 1):
        for warmup in (True, False):
            name = f"perf-b{concurrency}" + ("-warmup" if warmup else "")
            count = concurrency if warmup else max(8, concurrency * 3)
            seed = 4100 + concurrency + int(warmup)
            requests = RandomDataset(random_seed=seed).sample(
                tokenizer=tokenizer,
                num_requests=count,
                request_id_prefix=name + "-",
                prefix_len=0,
                range_ratio=0.0,
                input_len=4096,
                output_len=128,
                batchsize=1,
            )
            rows = []
            for request in requests:
                ids = tokenizer.encode(request.prompt, add_special_tokens=True)
                assert len(ids) == 4096
                rows.append(
                    {
                        "request_id": request.request_id,
                        "prompt": request.prompt,
                        "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
                        "input_ids": ids,
                        "input_tokens": len(ids),
                        "output_tokens": 128,
                    }
                )
            assert len({row["prompt_sha256"] for row in rows}) == count
            target = RUN / f"{name}-inputs.jsonl"
            with target.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(
                {
                    "profile": name,
                    "seed": seed,
                    "requests": count,
                    "unique_prompts": count,
                    "input_tokens_each": 4096,
                    "path": target.name,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    (RUN / "performance-input-reconstruction.json").write_text(
        json.dumps(
            {
                "method": "Deterministic replay of the same upstream vLLM 0.26 RandomDataset, tokenizer snapshot, seeds and arguments; no inference or HTTP",
                "server_alignment": "Original client logs report no tokenizer mismatch; native BOS makes 4095 prompt tokens into 4096 actual input tokens, independently validated in raw serving results",
                "profiles": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
