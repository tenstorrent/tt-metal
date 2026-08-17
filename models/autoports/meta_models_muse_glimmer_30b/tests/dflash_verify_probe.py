# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is a page-aligned continuation prefill equivalent to a from-zero prefill?

This is the correctness instrument for the aligned verify forward, and it deliberately
**does not involve the drafter**.  Speculative decoding's verify forward wants to
restart at the page-block boundary below the anchor instead of re-forwarding the whole
prefix; that is only sound if the target computes the same logits either way.

Why an end-to-end token comparison cannot answer this:

* Committed tokens are the target's own argmax *by construction*, so a wrong verify
  forward still produces a self-consistent stream rather than an obvious failure.
* F2 established that this target's argmax depends on **forward width** in bf16 -- a
  0.0625 top-2 gap, one ulp, flips near-ties.  The aligned path changes the width *and*
  the op (``chunked_scaled_dot_product_attention`` over the paged cache instead of a
  plain ``scaled_dot_product_attention`` over the chunk), so *some* divergence is
  expected and proves nothing on its own.  A 96/128 token mismatch against greedy is
  therefore not evidence of a bug, and neither is 20/128 evidence of correctness.

So: teacher-force one fixed sequence both ways and compare per-position argmax over the
overlap, reporting the top-2 gap at each mismatch exactly as
``dflash_divergence_probe.py`` does.  Mismatches only at near-ties => the aligned path
is arithmetically equivalent and the token divergence is F2 rounding.  Mismatches at
wide gaps => the aligned forward is reading wrong history, and the run tells you at
which position.

Usage::

    python -m models.autoports.meta_models_muse_glimmer_30b.tests.dflash_verify_probe \
        --length 200 --restart 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

#: A near-tie at this gap or below is bf16 noise rather than a logic error; F2 measured
#: an argmax flip at a 0.0625 gap with no speculation involved at all.
NEAR_TIE_GAP = 0.25

PROMPT = "Write a Python function that merges two sorted lists, then explain how it works."


def _all_argmax_and_gap(model, hidden, rows: int) -> tuple[list[int], list[float]]:
    """Per-row argmax and top-2 gap, from a prefill hidden state."""
    TILE = 32
    ids: list[int] = []
    gaps: list[float] = []
    tiles = model.prefill_all_logits(hidden, prompt_len=rows)
    for tile_index, tile in enumerate(tiles):
        gathered = model.gather_and_untilize_logits(tile)
        host = model.logits_to_torch(gathered, gathered=True)
        ttnn.deallocate(gathered)
        ttnn.deallocate(tile)
        remaining = min(TILE, rows - tile_index * TILE)
        for r in range(remaining):
            row = host[r].float()
            top2 = torch.topk(row, 2)
            ids.append(int(top2.indices[0].item()))
            gaps.append(float((top2.values[0] - top2.values[1]).item()))
    return ids, gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=200, help="total sequence length to teacher-force")
    parser.add_argument("--restart", type=int, default=128, help="page-aligned position to restart at")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=args.max_seq_len)
        model = gen.model
        tok = gen.tokenizer

        # A real token sequence, extended by greedy decode so it is in-distribution
        # rather than random ids (near-tie statistics depend on that).
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        ids = list(tok(text)["input_ids"])
        if len(ids) < args.length:
            gen.reset()
            ids = ids + list(gen.generate(ids, args.length - len(ids)))
        ids = ids[: args.length]
        length, restart = len(ids), args.restart
        block = int(model.config.page_block_size)
        if restart % block:
            raise ValueError(f"--restart {restart} must be a multiple of the page block size {block}")
        if not 0 < restart < length:
            raise ValueError(f"--restart {restart} must be inside (0, {length})")
        logger.info(f"length={length} restart={restart} page_block={block}")

        table = gen._coerce_page_table(None)
        slot_row = model.page_table_row(table, 0)
        tt_page_table = model.page_table_row_to_device(slot_row)

        # ---- Path A: one from-zero prefill over the whole sequence.
        model.release_sliding_tails()
        tt_tokens, _ = model.prefill_tokens_to_device(ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=0)
        full_ids, full_gaps = _all_argmax_and_gap(model, hidden, length)
        ttnn.deallocate(hidden)

        # ---- Path B: from-zero up to `restart`, then an aligned continuation.
        # Exactly what the verify forward does: the prefix is already in the paged
        # cache, so the continuation re-forwards only the tail of the sequence.
        model.release_sliding_tails()
        tt_tokens, _ = model.prefill_tokens_to_device(ids[:restart])
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=0))

        model.release_sliding_tails()
        tail_ids = ids[restart:]
        tt_tokens, _ = model.prefill_tokens_to_device(tail_ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        hidden = model.prefill_forward(embedded, page_table=tt_page_table, user_id=0, start_pos=restart)
        cont_ids, cont_gaps = _all_argmax_and_gap(model, hidden, len(tail_ids))
        ttnn.deallocate(hidden)
        ttnn.deallocate(tt_page_table)

        # ---- Compare over the overlap.
        mismatches = []
        for offset, (a, b) in enumerate(zip(full_ids[restart:], cont_ids)):
            if a != b:
                position = restart + offset
                mismatches.append(
                    {
                        "position": position,
                        "from_zero": a,
                        "aligned": b,
                        "from_zero_top2_gap": full_gaps[position],
                        "aligned_top2_gap": cont_gaps[offset],
                    }
                )

        compared = len(cont_ids)
        wide = [m for m in mismatches if m["from_zero_top2_gap"] > NEAR_TIE_GAP]
        print("\n" + "=" * 72)
        print(f"positions compared          : {compared}  (restart={restart}, length={length})")
        print(f"argmax mismatches           : {len(mismatches)}")
        print(f"  of those at a near-tie    : {len(mismatches) - len(wide)}  (top-2 gap <= {NEAR_TIE_GAP})")
        print(f"  of those at a WIDE gap    : {len(wide)}")
        for m in mismatches[:12]:
            tag = "WIDE" if m["from_zero_top2_gap"] > NEAR_TIE_GAP else "near-tie"
            print(
                f"    pos {m['position']:>5d}: from_zero {m['from_zero']:>7d} vs aligned {m['aligned']:>7d}"
                f"  gap {m['from_zero_top2_gap']:.5f}  [{tag}]"
            )
        if wide:
            print("\nVERDICT: the aligned continuation is NOT equivalent -- wide-gap mismatches")
            print("mean it is reading different history, not rounding differently.")
        else:
            print("\nVERDICT: equivalent up to the bf16 near-tie floor (F2).  An end-to-end")
            print("token divergence against greedy is therefore rounding, not a verify bug.")
        print("=" * 72)

        payload = {
            "length": length,
            "restart": restart,
            "positions_compared": compared,
            "mismatches": mismatches,
            "wide_gap_mismatches": len(wide),
            "near_tie_gap_threshold": NEAR_TIE_GAP,
        }
        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_verify_probe.json")
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
