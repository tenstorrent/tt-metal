# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signposted profiling window over the **reduced** full-model variant.

``$full-model`` is explicit that Tracy must not be pointed at the all-layer stack:
2400 ops per decode trace over 52 layers produces multi-GB dumps and overflows the
profiler's marker buffers.  This runs the same wrapper with one real layer of each
kind (sliding + full attention) and the **real** terminal path -- real embedding
table, real final norm, real BFP4 LM head at the real padded vocab, the real
sampler, real traces -- so the terminal costs the full model adds over the decoder
stack are attributable, which is what the profile is for.

Windows:

* ``prefill`` -- embed + 2 layers + terminal norm + LM head + softcap for one prompt;
* ``decode`` -- the decode trace replayed ``--iters`` times (**1** by default: the
  dropped-marker check in run_tracy.sh established that even 2 overflows the DRAM
  marker buffer on the decode window and silently
  under-count);
* ``sampling`` -- the sampling trace replayed on its own, so the sampler's share of
  a token-out step is a measured row rather than a subtraction.

Usage (through tracy, one device job at a time, watcher unset)::

    python -m tracy -r -p -v doc/full_model/bench/perf_window.py --window decode --iters 1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from tracy import signpost

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=("prefill", "decode", "sampling"), required=True)
    parser.add_argument("--layers", default="0,3", help="one real layer of each kind")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(23)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    try:
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
            layer_indices=[int(i) for i in args.layers.split(",")],
        )
        model = generator.model
        prompt = [int(t) for t in torch.randint(0, model.config.vocab_size, (args.prompt_len,)).tolist()]

        # Warm every program and capture both traces before any signpost.
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        ttnn.synchronize_device(mesh)

        if args.window == "prefill":
            generator.reset()
            table = model.normalize_page_table(None)
            tt_page_table = model.page_table_to_device(table)
            tt_tokens, _ = model.prefill_tokens_to_device(prompt)
            # One untimed pass so the profiled one is warm.
            hidden = model.prefill_forward(model.embed_prefill(tt_tokens), page_table=tt_page_table, user_id=0)
            ttnn.deallocate(model.prefill_logits(hidden, last_token_index=len(prompt) - 1))
            ttnn.deallocate(hidden)
            ttnn.synchronize_device(mesh)

            signpost(header="PERF_PREFILL")
            embedded = model.embed_prefill(tt_tokens)
            hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0)
            logits = model.prefill_logits(hidden, last_token_index=len(prompt) - 1)
            ttnn.synchronize_device(mesh)
            signpost(header="PERF_PREFILL_END")
            ttnn.deallocate(logits)
            ttnn.deallocate(hidden)
        elif args.window == "decode":
            generator.reset()
            generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
            ttnn.synchronize_device(mesh)
            signpost(header="PERF_DECODE")
            for _ in range(args.iters):
                ttnn.execute_trace(mesh, generator._trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost(header="PERF_DECODE_END")
        else:
            generator.reset()
            generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
            slot = next(iter(generator.sampling._trace_states.values()))
            ttnn.synchronize_device(mesh)
            signpost(header="PERF_SAMPLING")
            for _ in range(args.iters):
                ttnn.execute_trace(mesh, slot["id"], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost(header="PERF_SAMPLING_END")
        print(f"PERF_WINDOW_OK window={args.window} iters={args.iters}", flush=True)
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
