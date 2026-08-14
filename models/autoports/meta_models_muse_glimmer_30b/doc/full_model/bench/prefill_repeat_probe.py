# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate why a repeated prefill of the same prompt returns different logits.

Three arms, each answering one question:

A. **prefill only, repeated** -- is prefill itself reproducible with nothing else
   touching the device?
B. **prefill, decode, reset, prefill** -- does decode leave state that the next
   prefill reads?  If A is stable and B is not, the cache reset or the attention
   op's read window is the cause, not the prefill math.
C. **prefill, reset, prefill** with the cache reset verified by reading it back --
   does ``reset()`` actually zero the paged cache?

Usage::

    python doc/full_model/bench/prefill_repeat_probe.py [--layers 0,3] [--prompt-len 37]
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
    parser.add_argument("--prompt-len", type=int, default=37)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    torch.manual_seed(7)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, max_seq_len=args.max_seq_len, max_batch_size=1, layer_indices=layer_indices
        )
        model = generator.model
        prompt = [int(t) for t in torch.randint(0, model.config.vocab_size, (args.prompt_len,)).tolist()]
        table = model.normalize_page_table(None)

        def prefill_top1() -> tuple[int, float]:
            logits, row = generator._prefill_user(prompt, user_id=0, page_table=table)
            host = model.logits_to_torch(logits)[row].float()
            ttnn.deallocate(logits)
            top = torch.topk(host, k=2)
            return int(top.indices[0]), float(top.values[0] - top.values[1])

        def cache_nonzero() -> int:
            total = 0
            for layer in model.layers:
                for cache in (layer.k_cache, layer.v_cache):
                    shard = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()
                    total += int((shard != 0).sum())
            return total

        # ---------------------------------------------------------------- arm A
        arm_a = []
        for _ in range(args.repeats):
            generator.reset()
            arm_a.append(prefill_top1())
        say(f"PROBE A prefill-only top1s={[t for t, _ in arm_a]} gaps={[round(g, 4) for _, g in arm_a]}")
        say(f"PROBE A stable={len({t for t, _ in arm_a}) == 1}")

        # ---------------------------------------------------------------- arm C
        generator.reset()
        say(f"PROBE C nonzero cache entries after reset (fresh) = {cache_nonzero()}")
        generator._prefill_user(prompt, user_id=0, page_table=table)
        filled = cache_nonzero()
        say(f"PROBE C nonzero cache entries after prefill = {filled}")
        generator.reset()
        after_reset = cache_nonzero()
        say(f"PROBE C nonzero cache entries after reset() = {after_reset}")
        say(f"PROBE C reset_zeroes_cache={after_reset == 0}")

        # ---------------------------------------------------------------- arm B
        generator.reset()
        base = prefill_top1()
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        generator.reset()
        after_decode = prefill_top1()
        say(f"PROBE B base={base[0]} after-decode-then-reset={after_decode[0]} same={base[0] == after_decode[0]}")

        # ---------------------------------------------------------------- arm D
        # Does a prefill read cache state written past its own prompt?  Same prompt,
        # once into a zeroed cache and once straight after a decode with no reset.
        generator.reset()
        clean = prefill_top1()
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        dirty = prefill_top1()
        say(f"PROBE D clean={clean} dirty(no reset)={dirty} same={clean[0] == dirty[0] and clean[1] == dirty[1]}")

        # ---------------------------------------------------------------- arm E
        # Full generate() with and without reset() between requests.  reset() is
        # contractually required between prompts, but if two no-reset runs agree
        # with each other and disagree with the reset baseline, the decode loop is
        # reading cache state the prefill did not write -- which matters for a
        # serving caller that owns the cache and never zeroes it.
        generator.reset()
        base_gen = generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        generator.reset()
        base_gen2 = generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        no_reset1 = generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        no_reset2 = generator.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
        say(f"PROBE E with-reset      {base_gen}")
        say(f"PROBE E with-reset      {base_gen2}")
        say(f"PROBE E no-reset        {no_reset1}")
        say(f"PROBE E no-reset        {no_reset2}")
        say(
            f"PROBE E reset_stable={base_gen == base_gen2} noreset_stable={no_reset1 == no_reset2} "
            f"reset_vs_noreset={base_gen == no_reset1}"
        )

        say("PROBE_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
