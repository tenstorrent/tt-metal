# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 -- text -> text generation with `Qwen/Qwen3-Coder-Next` on Tenstorrent hardware.

Run it:

    ./python_env/bin/python -m models.demos.qwen3_coder_next.demo.demo_text_generation \
        --prompt "Write a Python function that returns the nth Fibonacci number." --layers 4

The wiring lives in `tt/pipeline.py` and is imported, not re-implemented -- this file only opens
the mesh (through `device_harness`, the package's sole opener, exactly as the e2e fixture does),
loads the reference weights and prints the result, so a green `tests/e2e` run and a working demo
cannot drift apart.
"""
from __future__ import annotations

import argparse
import time

from models.demos.qwen3_coder_next import device_harness
from models.demos.qwen3_coder_next.tt.pipeline import DEFAULT_CAPACITY, DEFAULT_PROMPT, build_pipeline
from models.demos.qwen3_coder_next.tt.reference import DEFAULT_LAYERS, load_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="the user prompt")
    parser.add_argument(
        "--layers",
        type=int,
        default=DEFAULT_LAYERS,
        help="decoder depth to build (the full 48-layer / 512-expert stack exceeds the mesh DRAM)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="decode horizon; default is the model-grounded cap (capacity - prompt length)",
    )
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY, help="pinned sequence capacity C")
    parser.add_argument("--mesh", default=None, help="force a mesh shape, e.g. 8x4")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="data-parallel replicas to materialise (each is a full TP group; >1 also runs a "
             "distinct prompt per replica to exercise the DP axis)",
    )
    parser.add_argument("--no-chat-template", action="store_true", help="feed the raw prompt, no chat template")
    parser.add_argument("--compare", action="store_true", help="also run the HF golden and report PCC")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape = tuple(int(v) for v in args.mesh.lower().split("x")) if args.mesh else None

    print(f"[demo] loading the reference checkpoint at depth {args.layers} ...", flush=True)
    model, tokenizer = load_reference(args.layers)

    device, _ = device_harness.open_mesh(shape)
    try:
        t0 = time.time()
        pipeline = build_pipeline(
            device,
            model=model,
            layers=args.layers,
            tokenizer=tokenizer,
            capacity=args.capacity,
            dp=args.dp,
        )
        print(f"[demo] pipeline resident in {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        result = pipeline.run_text_generation(
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            chat=not args.no_chat_template,
            collect_logits=args.compare,
        )
        elapsed = time.time() - t0

        print()
        print("=" * 78)
        print(f"PROMPT     : {args.prompt}")
        print(f"COMPLETION : {result['text']}")
        print("=" * 78)
        print(
            f"[demo] {len(result['tokens'])} token(s) in {elapsed:.1f}s "
            f"({len(result['tokens']) / max(elapsed, 1e-9):.2f} tok/s)"
        )

        if args.dp > 1:
            # Exercise the data-parallel axis: one distinct prompt per replica, run independently.
            extra = [f"{args.prompt} (replica {i})" for i in range(1, len(pipeline.replicas))]
            fan = pipeline.run_data_parallel(
                tokenizer, extra, max_new_tokens=args.max_new_tokens, chat=not args.no_chat_template
            )
            for i, r in enumerate(fan, start=1):
                print(f"[dp] replica {i}: {r['text']!r}")

        if args.compare:
            from models.demos.qwen3_coder_next.tt.pipeline import _pcc

            # Same two goldens the e2e gate uses, so the demo and the test report the SAME number.
            golden = pipeline._hf_reference_text_generation(
                tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, chat=not args.no_chat_template
            )
            golden_steps = pipeline._hf_score_sequence(result["prompt_ids"], result["tokens"])
            n = min(len(result["tokens"]), len(golden["tokens"]))
            matched = sum(int(a == b) for a, b in zip(result["tokens"][:n], golden["tokens"][:n]))
            achieved_pcc = _pcc(golden_steps, result["logits"])
            print(f"HF GOLDEN  : {golden['text']}")
            print(f"token agreement (free-running, both greedy): {matched}/{n}")
            print(f"free-running per-step logits PCC: {_pcc(golden['logits'][:n], result['logits'][:n]):.6f}")
            print(f"e2e PCC={achieved_pcc}")

    finally:
        device_harness.close_mesh(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
