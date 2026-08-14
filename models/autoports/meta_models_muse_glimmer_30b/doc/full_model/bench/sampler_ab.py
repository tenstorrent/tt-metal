# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why is the sampling trace 13 ms, and which knob is it?

This harness was written when token-out decode measured 36.164 ms/token -- a superseded
era, before both the embedding-gather fix and the topk split; the shipped step is now
23.8 ms with a 0.632 ms sampling trace (``evidence_perf.json``). At that time, of the
36.164 the **model** decode
trace is 23.166 -- essentially the 23.239 ms layer-stack floor -- and the **sampling**
trace is 12.970.  13 ms to pick one token out of a 32x50688 logit shard is not
credible as irreducible work, and ``$full-model`` gates on sampler ops not dominating
token-out decode.  This finds the knob.

Sampling cost does not depend on how many layers produced the logits, so this runs on
the **reduced** two-layer model: a 16 s build instead of 160 s, with the real padded
vocab, the real logits tensor, the real sampler and the real trace.  Each arm times
the sampling trace alone, replayed over the same captured logits.

Arms:

* ``pad_logits_to_power_of_2`` on/off -- the 50688 -> 65536 pad before ``ttnn.topk``;
* ``max_top_k`` 32 vs 8 -- candidates kept per shard *per topk call*.  ``ttnn.sampling``
  needs the *gathered* width to give a power-of-two tile count; under the shipped split
  each device contributes ``pieces * max_top_k``, so 32 gathers to 256 (8 tiles) and 8 to
  64 (2 tiles).  Without the split they gather to 128 and 32;
Force-argmax is **not** an arm here.  An earlier version of this docstring listed it as
"the honest latency reference" and said it "cannot feed ``tt_out_tok`` back (rank-3
output)".  That reason is withdrawn -- upstream passes ``output_tensor=tt_out_tok``
straight into ``ttnn.argmax``.  The real blocker is that its full-vocab gather needs
``self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)`` while this port constructs
``SamplingGenerator`` with ``tt_ccl=None`` on an L1_SMALL budget, and with it ``None``
the arm does not error, it **hangs** -- which is what it did when it was in the list.

Every arm also samples a real token so a faster arm that changes the *answer* is
caught rather than celebrated.

Usage::

    python doc/full_model/bench/sampler_ab.py [--rounds 3] [--replays 32]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

ARMS = [
    # The two arms that matter now. Both of the original pad arms ran ttnn.topk on its
    # single-core factory -- 50688 is not a power of two, and padding to 65536 exceeds
    # the multi-core uint16 bound -- so the pad A/B was measuring width, not a fast
    # path. Splitting the padded shard into 2 x 32768 is what reaches multi-core.
    {"label": "topk split to 2x32768 (shipped)", "max_top_k": 32, "topk_split_to_power_of_2": True},
    {"label": "no split: single-core topk over 50688", "max_top_k": 32, "topk_split_to_power_of_2": False},
    {
        "label": "rejected: top_k=32, pad_to_pow2, no split",
        "max_top_k": 32,
        "pad_logits_to_power_of_2": True,
        "topk_split_to_power_of_2": False,
    },
    {
        "label": "top_k=32, no pad",
        "max_top_k": 32,
        "pad_logits_to_power_of_2": False,
        "topk_split_to_power_of_2": False,
    },
    {"label": "top_k=8, pad_to_pow2", "max_top_k": 8, "pad_logits_to_power_of_2": True},
    {"label": "top_k=8, no pad", "max_top_k": 8, "pad_logits_to_power_of_2": False},
    # force_argmax is not an arm this harness can time, and the reason is exact rather
    # than incidental: ttnn's force-argmax path gathers the full vocab through
    # ``self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)``
    # (models/common/sampling/tt_sampling.py), and this port constructs
    # SamplingGenerator with ``tt_ccl=None`` -- a TT_CCL would put 36 more semaphores in
    # the main L1 pool, which the decode step has 7,296 B of headroom for. Left in the
    # arm list it does not error, it *hangs* the run. See README "Greedy is the top-k op
    # path, not force-argmax".
]


def say(*args) -> None:
    print(*args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(29)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    results = []
    try:
        for arm in ARMS:
            label = arm["label"]
            kwargs = {k: v for k, v in arm.items() if k != "label"}
            generator = None
            try:
                generator = build_generator(
                    ROOT,
                    mesh,
                    max_seq_len=args.max_seq_len,
                    max_batch_size=1,
                    layer_indices=[int(i) for i in args.layers.split(",")],
                    **kwargs,
                )
                vocab = generator.model.config.vocab_size
                prompt = [
                    int(t)
                    for t in torch.randint(
                        0, vocab, (args.prompt_len,), generator=torch.Generator().manual_seed(29)
                    ).tolist()
                ]

                generator.reset()
                tokens = generator.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
                ttnn.synchronize_device(mesh)

                model_only = []
                sampling_only = []
                slot = next(iter(generator.sampling._trace_states.values())) if generator._sampling_captured else None
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    for _ in range(args.replays):
                        ttnn.execute_trace(mesh, generator._trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    model_only.append((time.perf_counter() - started) / args.replays * 1e3)
                    if slot is not None and slot["id"] is not None:
                        ttnn.synchronize_device(mesh)
                        started = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, slot["id"], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        sampling_only.append((time.perf_counter() - started) / args.replays * 1e3)

                # An end-to-end token-out step, so the arm is judged on what a caller pays.
                generator.reset()
                started = time.perf_counter()
                generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True)
                one = time.perf_counter() - started
                generator.reset()
                started = time.perf_counter()
                generator.generate(prompt_token_ids=prompt, max_new_tokens=33, enable_trace=True)
                full = time.perf_counter() - started
                token_out = (full - one) / 32 * 1e3

                row = {
                    "label": label,
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                    "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
                    "sampling_trace_captured": slot is not None and slot["id"] is not None,
                    "model_trace_ms": round(min(model_only), 4),
                    "sampling_trace_ms": round(min(sampling_only), 4) if sampling_only else None,
                    "token_out_ms": round(token_out, 4),
                    "first_tokens": tokens,
                }
                results.append(row)
                say(f"AB {json.dumps(row)}")
            except Exception as exc:  # noqa: BLE001
                say(f"AB {label} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                results.append(
                    {"label": label, "kwargs": {k: str(v) for k, v in kwargs.items()}, "error": str(exc)[:400]}
                )
            finally:
                if generator is not None:
                    generator.teardown()
                clear_generator_cache()
        out = ROOT / "doc/full_model/sampler_ab.json"
        out.write_text(json.dumps(results, indent=2) + "\n")
        say(f"AB wrote {out}")
        say("AB_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
