# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does a repeated prefill first diverge from itself?

``batch_slot_probe.py --mode repeat`` established the shape of the defect: at
``max_batch_size=1``, prompt lengths 32 and 64 are bit-identical across repeats,
128 and up are not, and the *first* call is the odd one out (repeats 1 and 2 agree
with each other and differ from repeat 0 by the same amount).  So it is a
first-dispatch effect above 64 rows, not a batch-slot or cache-slot effect.

Guessing which branch crosses at 64 rows was not productive -- the sharded prefill
norm covers 64 and 128 alike, the fractured norm needs >256 rows, and both CCL
implementations reproduce it.  So this stops guessing and walks the graph: it runs
the prefill stage by stage, twice, and reports the *first* stage whose output moves.

Stages, in order: the embedding, then each layer's output, then the final norm
input/output and the LM head.  Layers are driven directly rather than through
``MuseGlimmerModel.prefill_forward`` so each intermediate can be read back.

Run 2 vs run 3 is also reported: if the first divergence is a first-use effect the
later pair agrees, which tells a warm-up fix from a genuine data-dependence one.

Usage::

    python doc/full_model/bench/prefill_divergence_probe.py [--length 128]
"""

from __future__ import annotations

import argparse
import json
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


def host(tensor: ttnn.Tensor) -> torch.Tensor:
    """Device-0's copy of a replicated tensor, on the host, as float32.

    The residual stream is replicated, so device 0 is representative; a
    divergence that appeared on only one device would still show up here because
    the layers all-gather/all-reduce before the next stage reads them.
    """
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def one_run(generator, ids: list[int]) -> dict[str, torch.Tensor]:
    """One prefill, stage by stage, returning a host copy of each intermediate."""
    model = generator.model
    stages: dict[str, torch.Tensor] = {}

    tt_tokens, padded_len = model.prefill_tokens_to_device(ids)
    page_table = model.page_table_to_device(model.normalize_page_table(None))
    hidden = model.embed_prefill(tt_tokens)
    ttnn.deallocate(tt_tokens)
    stages["embedding"] = host(hidden)

    for position, layer in enumerate(model.layers):
        out = layer.prefill_forward(hidden, page_table=page_table, user_id=0, start_pos=0)
        ttnn.deallocate(hidden)
        hidden = out
        stages[f"layer{position}({'sliding' if layer.config.is_sliding else 'full'})"] = host(hidden)

    row = model._slice_rows(hidden, len(ids) - 1)
    stages["last_tile_row"] = host(row)
    normed = model.final_norm.forward(row)
    stages["final_norm"] = host(normed)
    logits = model.lm_head.forward(normed)
    stages["logits"] = model.logits_to_torch(logits)[model.row_within_tile(len(ids) - 1)].float()

    ttnn.deallocate(row)
    ttnn.deallocate(normed)
    ttnn.deallocate(logits)
    ttnn.deallocate(hidden)
    ttnn.deallocate(page_table)
    return stages


def diff_runs(label: str, a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    first = None
    for name in a:
        d = float((a[name] - b[name]).abs().max())
        identical = bool(torch.equal(a[name], b[name]))
        if not identical and first is None:
            first = name
        rows.append({"pair": label, "stage": name, "max_abs_diff": d, "bit_identical": identical})
        say(f"STAGE {json.dumps(rows[-1])}")
    say(f"FIRST_DIVERGENCE {label} -> {first}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--out", default="prefill_divergence_probe.json")
    args = parser.parse_args()

    ids = prompt_of(args.length)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    results: list[dict] = []
    generator = None
    try:
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.batch,
            layer_indices=[int(i) for i in args.layers.split(",")],
        )
        # The defect is sporadic -- prefill_sync_bisect.py shows roughly one run in
        # three diverges, and an arm carrying *every* synchronise still failed while
        # arms carrying one each passed.  So a clean 3-run sample proves nothing;
        # this keeps running until a divergence is actually caught, and reports the
        # first stage that moved in the run that caught it.
        generator.reset()
        reference = one_run(generator, ids)
        caught = None
        for index in range(1, args.repeats + 1):
            generator.reset()
            current = one_run(generator, ids)
            divergent = [name for name in reference if not torch.equal(reference[name], current[name])]
            if divergent:
                caught = index
                say(f"CAUGHT divergence on run {index} after {index} clean comparisons")
                results.extend(diff_runs(f"run0 vs run{index}", reference, current))
                break
            say(f"run {index}: all stages bit-identical to run 0")
        if caught is None:
            say(f"NO_DIVERGENCE in {args.repeats} runs -- raise --repeats")
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps({"length": args.length, "comparisons": results}, indent=2) + "\n")
        say(f"DIVERGENCE wrote {out}")
        say("DIVERGENCE_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
