# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF's greedy generation for a chat prompt, recorded for the demo and tests/test_model_full.py.

Runs the real HF ``hy_v4`` model on the CPU (reference/hf_harness.py record_decode_golden: the
prompt through the checkpoint's chat template, a cached greedy generation, then one recorded pass
over the prompt and its continuation). No device, no ttnn; it needs the environment that has
transformers with hy_v4:

    HY4_HF_MODEL=<checkpoint dir> python -m models.experimental.hy4_preview.demo.record_reference \
        --layers 78 --steps 32

All 78 layers in BF16 are 1.5 TB of memory-mapped weights: they pass through the page cache and
do not have to fit in RAM, but every layer is read from disk twice (about 25 minutes cold for a
short prompt). The file lands under $HY4_GOLDEN_DIR, named after the depth, the prompt length, the
step count and a digest of the prompt's token ids, which is how the test finds it.
"""

import argparse

from models.experimental.hy4_preview.reference import hf_harness

# The prompt the demo and tests/test_model_full.py test_generate use.
DEFAULT_PROMPT = "Explain in three sentences why the sky is blue."
DEFAULT_STEPS = 32


def main():
    parser = argparse.ArgumentParser(description="Record HF's greedy generation for a chat prompt.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="tokens to generate")
    parser.add_argument("--model", default=None, help=f"checkpoint dir (default ${hf_harness.HY4_HF_MODEL_ENV})")
    parser.add_argument("--root", default=None, help=f"golden dir (default {hf_harness.GOLDEN_ROOT})")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(hf_harness.model_dir(args.model)))
    ids = hf_harness.tokenize(args.prompt, args.model)
    golden = hf_harness.record_decode_golden(
        ids, args.steps, args.layers, args.model, root=args.root, recompute=args.recompute
    )
    n = ids.numel()
    print(hf_harness.decode_golden_path(args.layers, n, args.steps, ids, root=args.root))
    print(f"prompt: {n} tokens; seconds: {golden['meta']['seconds']}")
    print(f"continuation ids (cached generation): {golden['input_ids'][n:].tolist()}")
    print(f"full-pass argmax:                     {golden['logits'].argmax(-1).tolist()}")
    print(f"top1 - top2 margins: {[round(row.topk(2).values.diff().abs().item(), 2) for row in golden['logits']]}")
    continuation = golden["input_ids"][n:].tolist()
    print(f"text: {tokenizer.decode(continuation)!r}")


if __name__ == "__main__":
    main()
