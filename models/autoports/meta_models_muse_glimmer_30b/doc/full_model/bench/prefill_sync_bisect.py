# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which seam does a device synchronise fix?

``batch_slot_probe.py --mode repeat`` shows ``generator.prefill_forward`` is
nondeterministic above 64 rows.  ``prefill_divergence_probe.py`` runs the *same*
graph stage by stage, reading each intermediate back, and is bit-identical over
three runs -- and a readback synchronises the device.  So the defect is a race
that synchronisation hides, and it crosses one of the stage boundaries the
divergence probe happened to synchronise.

This bisects the seam without touching shipped code: each arm monkeypatches a
``ttnn.synchronize_device`` into exactly one place and re-runs the repeat test.
The arm that turns three runs bit-identical names the seam.

Arms:

* ``none``          -- baseline, expected nondeterministic;
* ``after_embed``   -- one synchronise after the embedding all-gather;
* ``after_layer``   -- one after every layer;
* ``before_head``   -- one before the terminal norm/LM head;
* ``all``           -- the divergence probe's granularity, expected clean.

Usage::

    python doc/full_model/bench/prefill_sync_bisect.py [--length 128] [--repeats 3]
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
ARMS = ["none", "after_embed", "after_layer", "before_head", "all"]


def say(*args) -> None:
    print(*args, flush=True)


def prompt_of(length: int, *, seed: int = 41) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(0, VOCAB, (length,), generator=gen).tolist()]


def patched(generator, arm: str):
    """Install the arm's synchronise points; returns a restore callable."""
    model = generator.model
    mesh = model.mesh_device
    saved = {
        "embed_prefill": model.embed_prefill,
        "prefill_logits": model.prefill_logits,
        "layers": [layer.prefill_forward for layer in model.layers],
    }

    def restore():
        model.embed_prefill = saved["embed_prefill"]
        model.prefill_logits = saved["prefill_logits"]
        for layer, original in zip(model.layers, saved["layers"]):
            layer.prefill_forward = original

    if arm in ("after_embed", "all"):
        original_embed = saved["embed_prefill"]

        def embed(*a, **k):
            out = original_embed(*a, **k)
            ttnn.synchronize_device(mesh)
            return out

        model.embed_prefill = embed

    if arm in ("after_layer", "all"):
        for layer, original in zip(model.layers, saved["layers"]):

            def wrapped(*a, _original=original, **k):
                out = _original(*a, **k)
                ttnn.synchronize_device(mesh)
                return out

            layer.prefill_forward = wrapped

    if arm in ("before_head", "all"):
        original_logits = saved["prefill_logits"]

        def prefill_logits(*a, **k):
            ttnn.synchronize_device(mesh)
            return original_logits(*a, **k)

        model.prefill_logits = prefill_logits

    return restore


def run_arm(generator, arm: str, ids: list[int], repeats: int) -> dict:
    restore = patched(generator, arm)
    try:
        runs = []
        for _ in range(repeats):
            generator.reset()
            logits = generator.prefill_forward(
                tokens=torch.tensor([ids], dtype=torch.long),
                page_table=None,
                kv_cache=None,
                prompt_lens=[len(ids)],
            )
            runs.append(logits[0, 0].clone())
        diffs = [float((runs[0] - runs[i]).abs().max()) for i in range(1, len(runs))]
        identical = all(torch.equal(runs[0], runs[i]) for i in range(1, len(runs)))
        row = {
            "arm": arm,
            "all_bit_identical": bool(identical),
            "max_abs_diffs": diffs,
            "argmaxes": [int(r.argmax()) for r in runs],
        }
        say(f"ARM {json.dumps(row)}")
        return row
    finally:
        restore()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default="prefill_sync_bisect.json")
    args = parser.parse_args()

    ids = prompt_of(args.length)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    results = []
    try:
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.batch,
            layer_indices=[int(i) for i in args.layers.split(",")],
        )
        for arm in args.arms.split(","):
            results.append(run_arm(generator, arm, ids, args.repeats))
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps({"length": args.length, "arms": results}, indent=2) + "\n")
        say(f"BISECT wrote {out}")
        say("BISECT_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
