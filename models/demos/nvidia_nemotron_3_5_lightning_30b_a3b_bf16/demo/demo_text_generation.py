# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 -- text generation demo for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on a 4-chip TP=2 x DP=2 mesh.

Real input (the Source-A tokenizer over 32 distinct prompts) -> the chained
TTNN pipeline -> real task output (32 generated continuations, decoded to text).

The wiring lives in `tt/pipeline.py` and is imported from there; this file adds
no forward pass of its own, so a green `tests/e2e/test_e2e_pipeline.py`
guarantees this demo works.

    python -m models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.demo.demo_text_generation
"""
from __future__ import annotations

import argparse
import os
import time

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tests.e2e import make_golden
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--batch", type=int, default=P.BATCH, help="independent samples per call (default 32)")
    ap.add_argument(
        "--layers",
        type=int,
        default=P.DEFAULT_LAYERS,
        help="decoder depth to build; 0/negative means every layer (needs far more DRAM than 4 chips have)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=None, help="safety cap; default from the pipeline")
    ap.add_argument("--rows", type=int, default=2, help="mesh rows = DP")
    ap.add_argument("--cols", type=int, default=2, help="mesh cols = TP")
    ap.add_argument("--compare-hf", action="store_true", help="also run the HF reference and print both texts")
    ap.add_argument("--show", type=int, default=8, help="how many samples to print")
    a = ap.parse_args(argv)

    layers = None if a.layers is not None and a.layers <= 0 else a.layers
    os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")

    tok = _hf_ref.get_tokenizer()
    input_ids = make_golden.build_input_ids(a.batch)
    prompts = make_golden.prompts(a.batch)

    device = P.open_mesh(a.rows, a.cols)
    try:
        t0 = time.time()
        pipe = P.build_pipeline(device, layers=layers, batch=a.batch)
        print(f"[demo] built depth={pipe.n_layers} in {time.time() - t0:.1f}s  {pipe.describe()['block_types']}")
        print(f"[demo] variants={pipe.describe()['variants']}")
        print(f"[demo] batch={pipe.batch}  mesh={pipe.describe()['mesh_shape']}  sharded={pipe.sharded}")

        t0 = time.time()
        out = P.NemotronHPipeline.run_text_generation(pipe, input_ids, max_new_tokens=a.max_new_tokens, progress=True)
        print(f"[demo] generated {out['steps']} tokens x {a.batch} samples in {time.time() - t0:.1f}s")

        tt_text = tok.batch_decode(out["new_ids"], skip_special_tokens=True)
        hf_text = None
        if a.compare_hf:
            ref = pipe._hf_reference_text_generation(input_ids, max_new_tokens=out["steps"])
            hf_text = tok.batch_decode(ref["new_ids"], skip_special_tokens=True)

        print("\n=== Call 1: text generation ===")
        for i in range(min(a.show, a.batch)):
            print(f"[{i:2d}] prompt : {prompts[i]!r}")
            print(f"     TT     : {tt_text[i]!r}")
            if hf_text is not None:
                print(f"     HF ref : {hf_text[i]!r}")
        if pipe.n_layers < len(pipe.hf.config.layers_block_type) or layers is not None:
            print(
                f"\n[demo] NOTE: built {pipe.n_layers} of 52 decoder blocks (DRAM ceiling -- see README). "
                "The continuations are therefore not meaningful English; the PCC gate compares against the "
                "SAME depth-capped reference, so parity is exact-in-scope."
            )
    finally:
        P.close_mesh(device)


if __name__ == "__main__":
    main()
