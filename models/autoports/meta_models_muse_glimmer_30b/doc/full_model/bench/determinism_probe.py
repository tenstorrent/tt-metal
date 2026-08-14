# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why does greedy decode repeat, or not, across identical calls?

The reduced probe found ``generate()`` returning different tokens for the same
prompt on the second call.  This isolates the candidates:

1. is the *token* different, or only the sampled token while the logits agree?
   -> compare the sampler's pick against a host argmax of the very same logits;
2. are the top-2 logits **tied** at bf16?  With a 202048-wide vocab a tie is
   decided by ``ttnn.sampling``'s array order plus the tie-break pass, and by the
   device RNG state -- and ``SeedManager`` pushes fresh entropy on every
   ``reset_sampling_params``, so two calls do not share an RNG state;
3. does pinning the device RNG state make it deterministic?

Usage::

    python doc/full_model/bench/determinism_probe.py [--layers 0,3] [--repeats 4]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)


def say(*args) -> None:
    print(*args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--reference", default="readiness_aime24_chat.refpt")
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, max_seq_len=args.max_seq_len, max_batch_size=1, layer_indices=layer_indices
        )
        model = generator.model

        # A real prompt, not random ids: random ids give a near-degenerate logit
        # distribution where ties are common, which would confound the question.
        ref_path = ROOT / args.reference
        if ref_path.exists():
            from models.common.readiness_check.schema import load_reference

            prompt = [int(t) for t in load_reference(str(ref_path)).entries[0].prompt_tokens[0].tolist()]
            say(f"PROBE prompt: {len(prompt)} real tokens from {args.reference}")
        else:
            prompt = [int(t) for t in torch.randint(0, model.config.vocab_size, (37,)).tolist()]
            say("PROBE prompt: random ids (no reference found)")

        runs = []
        for index in range(args.repeats):
            generator.reset()
            runs.append(generator.generate(prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True))
            say(f"PROBE run[{index}] {runs[-1]}")
        say(f"PROBE all_equal={all(r == runs[0] for r in runs)}")

        # ---- host argmax of the same logits vs the sampler's pick
        generator.reset()
        logits, row = generator._prefill_user(prompt, user_id=0, page_table=model.normalize_page_table(None))
        host = model.logits_to_torch(logits)
        top = torch.topk(host[row].float(), k=5)
        say(f"PROBE prefill host top5 values={[round(v, 6) for v in top.values.tolist()]}")
        say(f"PROBE prefill host top5 indices={top.indices.tolist()}")
        say(f"PROBE prefill host argmax={int(top.indices[0])} gap(top1-top2)={float(top.values[0]-top.values[1]):.3e}")
        picks = []
        for _ in range(args.repeats):
            sampled = generator._sample_eager(logits, into_tokens=False)
            picks.append(int(sampled[row].item()))
        say(f"PROBE prefill sampler picks={picks} matches_host_argmax={[p == int(top.indices[0]) for p in picks]}")
        ttnn.deallocate(logits)

        # ---- host-sampling mode, which cannot depend on device RNG at all
        host_runs = []
        for index in range(2):
            generator.reset()
            host_runs.append(
                generator.generate(
                    prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True, host_sampling=True
                )
            )
            say(f"PROBE host_sampling run[{index}] {host_runs[-1]}")
        say(f"PROBE host_sampling all_equal={host_runs[0] == host_runs[1]}")
        say(f"PROBE device_vs_host_first_divergence={_first_diff(runs[0], host_runs[0])}")
        say("PROBE_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


def _first_diff(a, b) -> int:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
