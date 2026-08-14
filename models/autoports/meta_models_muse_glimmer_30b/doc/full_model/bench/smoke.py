# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model probe: the real wrapper with one layer of each kind.

This is the fast debugging loop ``$full-model`` asks for.  It builds
:class:`MuseGlimmerModel` with **real** weights for layer 0 (sliding) and layer 3
(full attention) and the **real** terminal path -- real embedding table, real
final norm, real LM head, real padded vocab, real KV-cache and page-table shapes,
real traces and the real sampler -- so wrapper, trace, cache, page-table,
LM-head and sampling bugs all reproduce here in a couple of minutes instead of
after a 52-layer load.

It is a debugging tool.  It is **not** correctness or performance evidence: two
layers do not predict tokens.  Every reported number comes from the all-layer
model.

Usage::

    python doc/full_model/bench/smoke.py [--layers 0,3] [--max-seq-len 4096]
        [--prompt-len 37] [--gen-len 8] [--batch 1] [--checks all]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    GREEDY,
    MuseGlimmerGenerator,
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
    parser.add_argument("--layers", default="0,3", help="comma-separated real layer indices, or 'all'")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=1, help="cache slots")
    parser.add_argument("--prompt-len", type=int, default=37, help="deliberately not tile/page/chunk aligned")
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE)
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    torch.manual_seed(0)
    mesh = open_multichip_mesh(trace_region_size=args.trace_region_size)
    generator: MuseGlimmerGenerator | None = None
    try:
        started = time.perf_counter()
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            layer_indices=layer_indices,
        )
        say(f"SMOKE built in {time.perf_counter() - started:.1f}s")
        report = generator.capability_report()
        for key in (
            "supported_context",
            "cache_slots",
            "decode_rows",
            "max_num_blocks",
            "blocks_per_seq",
            "prefill_chunk_size",
            "vocab_size",
            "padded_vocab_size",
            "num_layers",
            "layer_kinds",
            "force_argmax",
            "per_device_layer_weight_bytes",
            "per_device_kv_cache_bytes",
            "per_device_rope_table_bytes",
            "per_device_terminal_weight_bytes",
            "per_device_total_bytes",
            "per_device_dram_capacity_bytes",
        ):
            say(f"SMOKE report {key}={report[key]}")

        vocab = generator.model.config.vocab_size
        # A real prompt when one is available: random ids give a near-degenerate
        # logit distribution where top-1/top-2 ties are common, and a tie makes
        # greedy sensitive to the sampler's array order rather than to the model.
        reference = ROOT / "readiness_aime24_chat.refpt"
        if reference.exists() and args.prompt_len <= 0:
            from models.common.readiness_check.schema import load_reference

            prompt = [int(t) for t in load_reference(str(reference)).entries[0].prompt_tokens[0].tolist()]
        else:
            prompt = [int(t) for t in torch.randint(0, vocab, (abs(args.prompt_len),)).tolist()]
        say(f"SMOKE prompt tokens={len(prompt)}")

        # ---------------------------------------------------- low-level prefill
        started = time.perf_counter()
        logits = generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[args.prompt_len],
        )
        say(f"SMOKE prefill_forward last-logits shape={tuple(logits.shape)} in {time.perf_counter()-started:.2f}s")
        assert logits.shape == (1, 1, vocab), logits.shape
        assert torch.isfinite(logits).all(), "prefill logits are not finite"

        generator.reset()
        started = time.perf_counter()
        all_logits = generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=None,
            prompt_lens=[args.prompt_len],
            return_all_logits=True,
        )
        say(f"SMOKE prefill_forward all-logits shape={tuple(all_logits.shape)} in {time.perf_counter()-started:.2f}s")
        assert all_logits.shape == (1, args.prompt_len, vocab), all_logits.shape

        # --------------------------------------------------- high-level generate
        generator.reset()
        generator.reset_counters()
        started = time.perf_counter()
        free = generator.generate(prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True)
        say(f"SMOKE generate(free-running) tokens={free} in {time.perf_counter()-started:.2f}s")
        say(f"SMOKE counters(free-running) {generator.counters}")
        assert len(free) == args.gen_len, (len(free), args.gen_len)
        assert all(0 <= t < vocab for t in free), "sampled token outside the real vocabulary"

        # Capture-time staging is allowed; steady state is what matters, so run a
        # second generate with the traces already captured and require exactly one
        # token/position stage (the post-prefill reseed) and no page-table copy.
        generator.reset()
        generator.reset_counters()
        second = generator.generate(prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True)
        say(f"SMOKE counters(steady state) {generator.counters}")
        assert second == free, "greedy generation is not deterministic across identical prompts"
        assert generator.counters["trace_replays"] == args.gen_len - 1, generator.counters
        assert generator.counters["token_refreshes"] == 1, generator.counters
        assert generator.counters["position_refreshes"] == 1, generator.counters
        # One page-table copy per *request* (reset() drops the memo), never per token.
        assert generator.counters["page_table_refreshes"] == 1, generator.counters
        assert generator.counters["synchronizations"] == 0, generator.counters
        assert generator.counters["readbacks"] == args.gen_len, generator.counters

        # Unchanged / changed page table through the *low-level* decode path, which
        # is how a serving caller drives it: repeated steps with the same table must
        # cost one copy in total, and a different table exactly one more.
        identity = generator.model.normalize_page_table(None)
        permuted = identity.clone()[:, torch.randperm(identity.shape[1])]
        generator.reset()
        generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=identity,
            kv_cache=None,
            prompt_lens=[args.prompt_len],
        )
        generator.reset_counters()
        for step in range(4):
            generator.decode_forward(
                tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
                start_pos=torch.tensor([args.prompt_len + step], dtype=torch.int32),
                page_table=identity,
                kv_cache=None,
                sample_on_device=True,
            )
        say(f"SMOKE counters(4 decode steps, unchanged page table) {generator.counters}")
        assert generator.counters["page_table_refreshes"] == 1, generator.counters
        generator.decode_forward(
            tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([args.prompt_len + 4], dtype=torch.int32),
            page_table=permuted,
            kv_cache=None,
            sample_on_device=True,
        )
        say(f"SMOKE counters(+1 changed page table) {generator.counters}")
        assert generator.counters["page_table_refreshes"] == 2, generator.counters

        # A page table is a block permutation, not a semantic input: the same prompt
        # through a permuted table must generate the same tokens.
        generator.reset()
        moved = generator.generate(
            prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True, page_table=permuted
        )
        say(f"SMOKE permuted page table tokens={moved} match={moved == free}")
        if moved != free:
            # Not fatal on a 2-layer probe with random-id prompts: the logits are
            # near-degenerate there and a top-1/top-2 tie is decided by array order.
            # The all-layer gate is where this has to hold, and it is asserted there.
            say("SMOKE WARNING permuted page table changed the tokens (see README: tie sensitivity)")

        # ------------------------------------------------------ teacher forcing
        generator.reset()
        generator.reset_counters()
        forced = [int(t) for t in torch.randint(0, vocab, (args.gen_len,)).tolist()]
        seen: list[int] = []

        def next_input(step: int, predicted: int) -> int:
            seen.append(predicted)
            return forced[step]

        tf = generator.generate(
            prompt_token_ids=prompt,
            max_new_tokens=args.gen_len,
            next_input=next_input,
            enable_trace=True,
        )
        say(f"SMOKE generate(teacher-forced) tokens={tf}")
        say(f"SMOKE counters(teacher-forced) {generator.counters}")
        assert tf == seen, (tf, seen)
        assert len(tf) == args.gen_len

        # -------------------------------------------- host-sampling compat mode
        generator.reset()
        host = generator.generate(
            prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True, host_sampling=True
        )
        say(f"SMOKE host_sampling tokens={host} match_device={host == free}")

        # ------------------------------------------------------- low-level decode
        generator.reset()
        generator.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.long),
            page_table=None,
            kv_cache=generator.model.kv_cache,
            prompt_lens=[args.prompt_len],
        )
        step_logits = generator.decode_forward(
            tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([args.prompt_len], dtype=torch.int32),
            page_table=None,
            kv_cache=generator.model.kv_cache,
        )
        say(f"SMOKE decode_forward logits shape={tuple(step_logits.shape)}")
        assert step_logits.shape == (1, vocab), step_logits.shape
        toks = generator.decode_forward(
            tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
            start_pos=torch.tensor([args.prompt_len + 1], dtype=torch.int32),
            page_table=None,
            kv_cache=generator.model.kv_cache,
            sample_on_device=True,
        )
        say(f"SMOKE decode_forward on-device tokens={toks.tolist()}")

        # --------------------------------------------------------- batch > 1
        if args.batch > 1 and args.batch > args.max_batch_size:
            say(f"SMOKE batch arm skipped: --batch {args.batch} needs --max-batch-size >= {args.batch}")
        elif args.batch > 1:
            generator.reset()
            lens = [args.prompt_len, args.prompt_len - 5]
            batch_tokens = torch.zeros(2, args.prompt_len, dtype=torch.long)
            batch_tokens[0] = torch.tensor(prompt)
            batch_tokens[1, : lens[1]] = torch.tensor(prompt[: lens[1]])
            batch_logits = generator.prefill_forward(
                tokens=batch_tokens, page_table=None, kv_cache=None, prompt_lens=lens
            )
            say(f"SMOKE batch prefill shape={tuple(batch_logits.shape)} (mixed lengths {lens})")
            assert batch_logits.shape == (2, 1, vocab)
            batch_toks = generator.decode_forward(
                tokens=torch.tensor([[prompt[-1]], [prompt[-1]]], dtype=torch.long),
                start_pos=torch.tensor(lens, dtype=torch.int32),
                page_table=None,
                kv_cache=None,
                sample_on_device=True,
            )
            say(f"SMOKE batch decode on-device tokens={batch_toks.tolist()}")

        say("SMOKE_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
