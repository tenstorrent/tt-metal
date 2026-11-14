# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the 32-prompt input file for the text_qwen_demo.py `prefill-perf-40k-b32` case.

The case needs 32 distinct prompts that each encode (instruct chat template, as the demo encodes them) to the same
length as input_data_prefill_40k_local.json (40928 tokens). The file is about 5.4 MB, over the repo's 500 KB file
limit, so it is generated instead of checked in.

Each prompt is a different rotation of the 40k prompt's token stream: the prompts are distinct and keep the same text
statistics. Prefill time depends on the length, not the content.

    python models/demos/llama3_70b_galaxy/demo/sample_prompts/make_prefill_40k_b32_prompts.py --out /path/b32.json
    pytest models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k prefill-perf-40k-b32 --input_prompts /path/b32.json
"""
import argparse
import json

from transformers import AutoTokenizer

from models.tt_transformers.tt.common import encode_prompt_hf

SOURCE = "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_prefill_40k_local.json"


def fit_to_length(tokenizer, raw_ids, target):
    """Text decoded from a prefix of raw_ids whose instruct encoding is exactly `target` tokens."""
    n = target - len(encode_prompt_hf(tokenizer, " "))
    for _ in range(50):
        text = tokenizer.decode(raw_ids[:n])
        got = len(encode_prompt_hf(tokenizer, text))
        if got == target:
            return text
        n += target - got
    raise RuntimeError(f"could not reach {target} tokens (last {got})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output JSON prompt file")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = json.load(open(SOURCE))[0]["prompt"]
    target = len(encode_prompt_hf(tokenizer, source))
    raw = tokenizer.encode(source, add_special_tokens=False)
    stream = raw * 2  # every rotation start has a full window after it
    stride = len(raw) // args.num_prompts
    prompts = [{"prompt": fit_to_length(tokenizer, stream[k * stride :], target)} for k in range(args.num_prompts)]
    assert len({p["prompt"] for p in prompts}) == args.num_prompts, "prompts are not distinct"
    with open(args.out, "w") as f:
        json.dump(prompts, f)
    print(f"wrote {args.num_prompts} distinct prompts x {target} tokens to {args.out}")


if __name__ == "__main__":
    main()
