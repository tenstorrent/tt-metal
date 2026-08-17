# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Can the verify forward be a *decode* step instead of a prefill?

The verify forward is the whole DFlash bottleneck: **64.5 ms**, and measured flat in
row count (32 rows costs more than 128), so it is host-dispatch bound rather than
work bound.  Meanwhile a decode step through the same 52 layers, reading the same
weights, costs **23.3 ms** -- because the decode path is L1-sharded, fused and traced
while the prefill path is DRAM-interleaved and eager.

The 16 verify positions can be expressed as a decode step: put the anchor and its 15
candidates in 16 of the decode batch's rows, all pointing at the *same* page-table
row, with ``current_pos[u] = anchor_pos + u``.  ``paged_update_cache`` writes all 16
K/V first and ``paged_scaled_dot_product_attention_decode`` then lets row ``u`` attend
``[0, anchor_pos + u]`` -- which includes the candidates before it and excludes those
after.  That is exactly the causal structure of a 16-row prefill.

And it should be free: the port documents that ``DECODE_ROWS`` "is **not** the batch
size and is deliberately independent of it... decode always runs 32 rows and inactive
rows carry ``current_pos = -1``".  So 16 active users cost what 1 costs.

This measures the two numbers that decide whether that redesign is worth building:

* eager ``ttnn_decode_forward`` at 1 vs 16 active users -- does batching cost anything,
  and is even the *eager* decode step cheaper than the 64.5 ms prefill;
* the same with taps armed, since DFlash needs the tapped hidden states and arming
  them adds a clone per tapped layer.

If eager decode already beats the prefill, the redesign pays off without touching the
trace machinery; if not, it needs a dedicated traced capture with taps armed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

PROMPT = "Write a Python function that merges two sorted lists."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        # max_batch_size 32 so the verify rows are addressable; the decode step runs
        # 32 rows regardless, so this costs nothing extra.
        gen = build_generator(".", mesh, max_batch_size=32, max_seq_len=args.max_seq_len)
        model = gen.model
        tok = gen.tokenizer
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = list(tok(text)["input_ids"])
        prompt_len = len(prompt_ids)

        table = gen._coerce_page_table(None)
        slot_row = model.page_table_row(table, 0)
        # Every verify row shares slot 0's blocks: they write different *positions*
        # inside the same sequence, which is the point.
        shared = slot_row.repeat(model.config.max_batch_size, 1)
        tt_page_table = model.page_table_to_device(shared) if hasattr(model, "page_table_to_device") else None
        if tt_page_table is None:
            tt_page_table = model._replicated(shared.to(torch.int32), ttnn.int32, device=True)

        # Seed the cache with a real prefix so the decode attends something real.
        one_row = model.page_table_row_to_device(slot_row)
        tt_tokens, _ = model.prefill_tokens_to_device(prompt_ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(model.prefill_forward(embedded, page_table=one_row, user_id=0, start_pos=0))
        ttnn.deallocate(one_row)

        results = {}
        for taps_on in (False, True):
            model.arm_hidden_state_taps((1, 13, 25, 37, 49) if taps_on else None)
            for active in (1, 16):
                positions = torch.arange(prompt_len, prompt_len + active)
                timings = []
                for repeat in range(args.repeats + 1):
                    tokens = model.tokens_to_device([7] * active)
                    current_pos, rope_pos_ids = model.positions_to_device(positions)
                    ttnn.synchronize_device(model.mesh_device)
                    started = time.perf_counter()
                    logits = model.ttnn_decode_forward(
                        tokens, current_pos, rope_pos_ids, tt_page_table, advance_positions=False
                    )
                    ttnn.synchronize_device(model.mesh_device)
                    elapsed = time.perf_counter() - started
                    ttnn.deallocate(logits)
                    ttnn.deallocate(tokens)
                    ttnn.deallocate(current_pos)
                    ttnn.deallocate(rope_pos_ids)
                    if taps_on:
                        for t in model.take_hidden_state_taps().values():
                            ttnn.deallocate(t)
                    if repeat:
                        timings.append(elapsed)
                key = f"taps={int(taps_on)},active={active}"
                results[key] = 1000.0 * statistics.median(timings)
                print(f"  eager decode  {key:22s} {results[key]:8.2f} ms", flush=True)
        model.arm_hidden_state_taps(None)
        ttnn.deallocate(tt_page_table)

        print("\n" + "=" * 72)
        print("reference: verify as an eager PREFILL forward is 64.5 ms/iteration")
        print("reference: one TRACED decode step (the shipped decode path) is 23.3 ms")
        best = results.get("taps=1,active=16")
        if best is not None:
            print(f"eager decode verify with taps, 16 rows: {best:.2f} ms")
            if best < 64.5:
                print(f"=> worth building: saves {64.5 - best:.1f} ms/iteration even without a trace")
            else:
                print("=> eager decode is no cheaper; this needs a traced capture with taps armed")
        print("=" * 72)

        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_decode_verify_cost.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"wrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
