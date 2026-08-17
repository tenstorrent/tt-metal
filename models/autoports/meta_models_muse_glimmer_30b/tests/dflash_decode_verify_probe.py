# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is a 16-user batched decode step equivalent to a 16-row prefill?

This is the correctness gate for replacing DFlash's verify forward with a decode step,
which is worth 64.5 ms -> ~24 ms once traced (measured: eager decode 70.9 ms, eager
prefill 64.5 ms, *traced* decode 23.3 ms -- the whole gap is host dispatch, and the
port documents that the decode step's cost is independent of how many rows are active).

The claim being tested: putting the anchor and its 15 candidates in 16 decode rows that
all share **one** page-table row, with ``current_pos[u] = start + u``, reproduces a
16-row causal prefill.  It should, because ``paged_update_cache`` writes all 16 K/V
before ``paged_scaled_dot_product_attention_decode`` reads, and row ``u`` is limited to
``[0, start + u]`` -- so it sees the candidates before it and not those after.  But
"should" is exactly the kind of reasoning that produced the two wrong conclusions
earlier in this work, so it is measured against the prefill path per position, with the
top-2 gap at every mismatch to separate a real divergence from the bf16 near-tie floor
that F2 established for this model.

Usage::

    python -m models.autoports.meta_models_muse_glimmer_30b.tests.dflash_decode_verify_probe
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

NEAR_TIE_GAP = 0.25
PROMPT = "Write a Python function that merges two sorted lists, then explain how it works."


def _rows_argmax_gap(model, logits_tiles, rows: int) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    gaps: list[float] = []
    for tile_index, tile in enumerate(logits_tiles):
        gathered = model.gather_and_untilize_logits(tile)
        host = model.logits_to_torch(gathered, gathered=True)
        ttnn.deallocate(gathered)
        ttnn.deallocate(tile)
        remaining = min(32, rows - tile_index * 32)
        for r in range(remaining):
            top2 = torch.topk(host[r].float(), 2)
            ids.append(int(top2.indices[0].item()))
            gaps.append(float((top2.values[0] - top2.values[1]).item()))
    return ids, gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        gen = build_generator(".", mesh, max_batch_size=32, max_seq_len=args.max_seq_len)
        model = gen.model
        tok = gen.tokenizer
        block = args.block

        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        ids = list(tok(text)["input_ids"])
        gen.reset()
        ids = ids + list(gen.generate(ids, block + 8))
        prefix_len = len(ids) - block
        prefix, tail = ids[:prefix_len], ids[prefix_len : prefix_len + block]
        logger.info(f"prefix={prefix_len} block={block}")

        table = gen._coerce_page_table(None)
        slot_row = model.page_table_row(table, 0)
        one_row = model.page_table_row_to_device(slot_row)

        # ---- seed the cache with the prefix
        model.release_sliding_tails()
        tt_tokens, _ = model.prefill_tokens_to_device(prefix)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(model.prefill_forward(embedded, page_table=one_row, user_id=0, start_pos=0))

        # ---- (b) DECODE first, on a cache holding ONLY the prefix.
        #
        # Order matters and the first version of this probe got it wrong: running the
        # prefill verify first writes K/V for exactly these positions, so the decode step
        # then reads a pre-warmed cache and never has to chain row u onto row u-1's write
        # -- which is the whole property under test.  Decode must go first.
        shared = slot_row.repeat(model.config.max_batch_size, 1)
        tt_page_table = model._replicated(shared.to(torch.int32), ttnn.int32, device=True)
        positions = torch.arange(prefix_len, prefix_len + block)

        def seed_prefix() -> None:
            """Re-write the prefix into the cache.

            Required before *every* measurement: a decode step writes K/V for the block's
            positions, so a second run would read them back instead of chaining its own
            rows -- which silently turns this probe into a test of nothing.  Two earlier
            versions of this file were wrong in exactly that way.
            """
            model.release_sliding_tails()
            tt_seed, _ = model.prefill_tokens_to_device(prefix)
            seeded = model.embed_prefill(tt_seed)
            ttnn.deallocate(tt_seed)
            ttnn.deallocate(model.prefill_forward(seeded, page_table=one_row, user_id=0, start_pos=0))

        def run_decode(with_taps: bool) -> list[int]:
            seed_prefix()
            tokens = model.tokens_to_device(tail)
            current_pos, rope_pos_ids = model.positions_to_device(positions)
            model.arm_hidden_state_taps((1, 13, 25, 37, 49) if with_taps else None)
            out = model.ttnn_decode_forward(tokens, current_pos, rope_pos_ids, tt_page_table, advance_positions=False)
            gathered = model.gather_and_untilize_logits(out)
            host_logits = model.logits_to_torch(gathered, gathered=True)
            ttnn.deallocate(gathered)
            ttnn.deallocate(out)
            if with_taps:
                for t in model.take_hidden_state_taps().values():
                    ttnn.deallocate(t)
            model.arm_hidden_state_taps(None)
            for t in (tokens, current_pos, rope_pos_ids):
                ttnn.deallocate(t)
            return [int(torch.argmax(host_logits[u].float()).item()) for u in range(block)]

        # Taps armed is what the DFlash loop actually does, and arming them inserts a
        # clone of the width-sharded L1 decode residual into every tapped layer -- the
        # exact call that was already wrong once in this project.  Run both.
        decode_block = run_decode(with_taps=False)
        decode_block_taps = run_decode(with_taps=True)
        if decode_block_taps != decode_block:
            first = next(i for i, (a, b) in enumerate(zip(decode_block, decode_block_taps)) if a != b)
            print(f"!! arming taps CHANGES the decode logits, first at row {first}")
            print(f"   no taps: {decode_block}")
            print(f"   taps   : {decode_block_taps}")
        else:
            print("arming taps does not change the decode logits")
        ttnn.deallocate(tt_page_table)

        # ---- (a) now re-seed the prefix and verify the same block as a PREFILL,
        # which is the reference the decode step has to reproduce.
        model.release_sliding_tails()
        tt_tokens, _ = model.prefill_tokens_to_device(prefix)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(model.prefill_forward(embedded, page_table=one_row, user_id=0, start_pos=0))
        page_block = int(model.config.page_block_size)
        aligned = (prefix_len // page_block) * page_block
        lead = prefix_len - aligned
        verify_ids = ids[aligned:prefix_len] + tail
        model.release_sliding_tails()
        tt_tokens, _ = model.prefill_tokens_to_device(verify_ids)
        embedded = model.embed_prefill(tt_tokens)
        ttnn.deallocate(tt_tokens)
        hidden = model.prefill_forward(embedded, page_table=one_row, user_id=0, start_pos=aligned)
        tiles = model.prefill_all_logits(hidden, prompt_len=len(verify_ids), apply_softcap=False)
        prefill_ids, prefill_gaps = _rows_argmax_gap(model, tiles, len(verify_ids))
        ttnn.deallocate(hidden)
        prefill_block = prefill_ids[lead : lead + block]
        prefill_block_gaps = prefill_gaps[lead : lead + block]
        ttnn.deallocate(one_row)

        mismatches = [
            {"row": u, "prefill": a, "decode": b, "prefill_top2_gap": prefill_block_gaps[u]}
            for u, (a, b) in enumerate(zip(prefill_block, decode_block))
            if a != b
        ]
        wide = [m for m in mismatches if m["prefill_top2_gap"] > NEAR_TIE_GAP]

        print("\n" + "=" * 72)
        print(f"block positions compared : {block}  (prefix {prefix_len})")
        print(f"argmax mismatches        : {len(mismatches)}")
        print(f"  at a near-tie          : {len(mismatches) - len(wide)}  (gap <= {NEAR_TIE_GAP})")
        print(f"  at a WIDE gap          : {len(wide)}")
        for m in mismatches[:10]:
            tag = "WIDE" if m["prefill_top2_gap"] > NEAR_TIE_GAP else "near-tie"
            print(
                f"    row {m['row']:>3d}: prefill {m['prefill']:>7d} vs decode {m['decode']:>7d}"
                f"  gap {m['prefill_top2_gap']:.5f}  [{tag}]"
            )
        if wide:
            print("\nVERDICT: NOT equivalent -- batched decode is not reproducing the prefill's")
            print("causal structure, so it cannot replace the verify forward as written.")
        else:
            print("\nVERDICT: equivalent up to the bf16 near-tie floor.  A 16-user decode step")
            print("reproduces a 16-row prefill, so the verify forward can become a decode step.")
        print("=" * 72)

        out = Path(args.out) if args.out else Path(__file__).with_name("dflash_decode_verify_probe.json")
        out.write_text(
            json.dumps(
                {
                    "block": block,
                    "prefix_len": prefix_len,
                    "prefill_block": prefill_block,
                    "decode_block": decode_block,
                    "mismatches": mismatches,
                    "wide_gap_mismatches": len(wide),
                },
                indent=2,
            )
        )
        print(f"wrote {out}")
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
