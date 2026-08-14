# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The same prompt in two cache slots produced different logits.  Which is it?

``test_logits_are_reproducible_across_batch_positions`` prefills one 200-token
prompt into slot 0 and slot 1 and compares the two logit rows.  They differ, and
the difference is **not stable across runs** (1.6875 in the full pytest pass,
1.5955 in the ``-m slow`` pass), so a fixed page-table/position row mix-up is
already ruled out: that would give the same number every time.

Reading the code does not settle it either.  At ``start_pos == 0`` a 200-token
prompt is one prefill chunk (``prefill_chunk_size`` is 1024 at
``max_seq_len=1024``), the full-attention layers take the *non-paged*
``scaled_dot_product_attention`` over local q/k/v, the sliding layers take the
same op with ``sliding_window_size``, and neither reads the paged cache.  The
only thing ``user_id`` selects is where ``paged_fill_cache`` *writes*.  So by
inspection the two slots should be bit-identical.

This separates the hypotheses by measurement rather than by argument:

* **same-slot repeat, dirty cache** -- prefill slot 0 twice with no reset.  A
  difference here means the call is not reproducible at all and the slot index is
  a red herring;
* **cross-slot, clean cache** -- reset, then slot 0 and slot 1.  A difference
  here is a genuine slot dependence;
* **cross-slot, dirty cache** -- the failing test's exact conditions;
* **order** -- slot 1 first, then slot 0, to tell "slot 1 is wrong" from "the
  second call is wrong";
* **alignment** -- 200 (neither tile- nor block-aligned), 224 (tile-aligned,
  3.5 blocks), 256 (block-aligned).  The earlier non-aligned prefill bug
  (work log Section 6) was exactly this axis;
* **layer kind** -- the same matrix with only layer 0, then only layer 3, to say
  whether the sliding or the full-attention layer carries it.

Every comparison is ``max |a - b|`` over the full padded vocab row, plus whether
the argmax moved, so a difference that cannot change a token is not reported as
if it could.

Usage::

    python doc/full_model/bench/batch_slot_probe.py [--lengths 200,224,256]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

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

VOCAB = 202048


def say(*args) -> None:
    print(*args, flush=True)


def prompt_of(length: int, *, seed: int = 41) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(0, VOCAB, (length,), generator=gen).tolist()]


def one_prefill(generator, ids: list[int], slot: int, batch: int) -> torch.Tensor:
    """Prefill ``ids`` into cache slot ``slot`` and return that slot's logit row.

    The batch is padded out to ``slot + 1`` rows so the call really does target
    the requested slot: ``prefill_forward`` walks users 0..batch-1 in order.
    Only the requested row is returned.
    """
    tokens = torch.zeros(slot + 1, len(ids), dtype=torch.long)
    tokens[slot] = torch.tensor(ids, dtype=torch.long)
    lens = [1] * (slot + 1)
    lens[slot] = len(ids)
    # Rows before ``slot`` are 1-token throwaway prefills into their own slots;
    # they keep the loop honest without costing a full prompt each.
    logits = generator.prefill_forward(tokens=tokens, page_table=None, kv_cache=None, prompt_lens=lens)
    return logits[slot, 0].clone()


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = float((a - b).abs().max())
    row = {
        "comparison": name,
        "max_abs_diff": diff,
        "bit_identical": bool(torch.equal(a, b)),
        "argmax_a": int(a.argmax()),
        "argmax_b": int(b.argmax()),
        "argmax_moved": int(a.argmax()) != int(b.argmax()),
    }
    say(f"CMP {json.dumps(row)}")
    return row


def repeat_only(generator, length: int, layers_label: str, batch: int, repeats: int) -> list[dict]:
    """Is a single prefill call reproducible at all?  Slot 0 only, so batch 1 works.

    Both the CCL-impl arms came back non-reproducible, so the question is no longer
    "which slot" but "which length": the 37-token repeat test passes, these 200+
    ones do not, and the prefill path picks program configs and norm/SDPA branches
    off the row count.  This walks lengths to find the boundary.
    """
    ids = prompt_of(length)
    rows: list[dict] = []
    runs = []
    for _ in range(repeats):
        generator.reset()
        runs.append(one_prefill(generator, ids, 0, batch))
    for index in range(1, len(runs)):
        row = compare(f"repeat 0 vs {index} (reset before each)", runs[0], runs[index])
        row.update({"length": length, "layers": layers_label, "batch": batch, "mode": "repeat"})
        rows.append(row)
    return rows


def matrix(generator, length: int, layers_label: str, batch: int) -> list[dict]:
    ids = prompt_of(length)
    rows: list[dict] = []

    # 1. same slot, twice, no reset in between -> is one call even reproducible?
    generator.reset()
    slot0_clean = one_prefill(generator, ids, 0, batch)
    slot0_again = one_prefill(generator, ids, 0, batch)
    rows.append(compare("slot0 vs slot0 again (dirty cache)", slot0_clean, slot0_again))

    # 2. cross-slot on a clean cache.
    generator.reset()
    a0 = one_prefill(generator, ids, 0, batch)
    generator.reset()
    a1 = one_prefill(generator, ids, 1, batch)
    rows.append(compare("slot0 vs slot1 (each after reset)", a0, a1))

    # 3. cross-slot, dirty cache: the failing test's conditions.
    generator.reset()
    b0 = one_prefill(generator, ids, 0, batch)
    b1 = one_prefill(generator, ids, 1, batch)
    rows.append(compare("slot0 then slot1, one reset (the failing test)", b0, b1))

    # 4. reversed order, to separate "slot 1" from "the second call".
    generator.reset()
    c1 = one_prefill(generator, ids, 1, batch)
    c0 = one_prefill(generator, ids, 0, batch)
    rows.append(compare("slot1 then slot0, one reset", c1, c0))
    rows.append(compare("slot0-first vs slot0-second (same slot, both dirty)", b0, c0))

    for row in rows:
        row["length"] = length
        row["layers"] = layers_label
        row["batch"] = batch
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="200,224,256")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--layer-sets", default="0,3|0|3")
    # The suspect: the prefill CCL implementation.  ``async`` allocates a fresh
    # intermediate buffer per dispatch, and the decoder stage's own bisect found
    # the ring algorithm reads the penultimate intermediate before writing it on
    # a buffer's *first* use (see DEFAULT_CCL_PERSISTENT_BUFFERS).  Per-dispatch
    # allocation makes every prefill dispatch a first use.
    parser.add_argument("--prefill-ccl-impl", default=None, choices=[None, "async", "wrapper"])
    parser.add_argument("--mode", default="matrix", choices=["matrix", "repeat"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="batch_slot_probe.json")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    layer_sets = [[int(i) for i in group.split(",")] for group in args.layer_sets.split("|")]

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    results: list[dict] = []
    try:
        for layers in layer_sets:
            label = ",".join(str(i) for i in layers)
            generator = None
            try:
                decoder_kwargs = {}
                if args.prefill_ccl_impl is not None:
                    decoder_kwargs["prefill_ccl_impl"] = args.prefill_ccl_impl
                generator = build_generator(
                    ROOT,
                    mesh,
                    max_seq_len=args.max_seq_len,
                    max_batch_size=args.batch,
                    layer_indices=layers,
                    decoder_kwargs=decoder_kwargs,
                )
                for length in lengths:
                    say(f"--- layers=[{label}] length={length} batch={args.batch} mode={args.mode}")
                    if args.mode == "repeat":
                        results.extend(repeat_only(generator, length, label, args.batch, args.repeats))
                    else:
                        results.extend(matrix(generator, length, label, args.batch))
            except Exception as exc:  # noqa: BLE001
                say(f"PROBE layers=[{label}] FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                results.append({"layers": label, "error": str(exc)[:400]})
            finally:
                if generator is not None:
                    generator.teardown()
                clear_generator_cache()
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps(results, indent=2) + "\n")
        say(f"PROBE wrote {out}")
        say("PROBE_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
