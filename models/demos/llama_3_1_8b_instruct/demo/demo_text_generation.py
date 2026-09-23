# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo — text -> text for `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`.

Real input (HF chat template + tokenizer) -> the SHARED chained TTNN pipeline
(`tt/pipeline.py`, the exact same `run_text_generation` the e2e test asserts on)
-> real output (the assistant's text).

Run:
  ./python_env/bin/python -m models.demos.llama_3_1_8b_instruct.demo.demo_text_generation \
      --prompt "What is the capital of France? Answer in one short sentence."
"""
from __future__ import annotations

import argparse
import sys
import time

from models.demos.llama_3_1_8b_instruct.mesh_harness import close_mesh, open_mesh
from models.demos.llama_3_1_8b_instruct.tt import pipeline as pl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Llama-3.1-8B-Instruct TTNN text-generation demo (TP=4 x DP=1)")
    ap.add_argument("--prompt", default=pl.DEFAULT_PROMPT, help="prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="safety cap on the stop-token decode")
    ap.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all 32)")
    ap.add_argument("--max-seq-len", type=int, default=pl.DEFAULT_MAX_SEQ_LEN, help="pinned sequence capacity")
    ap.add_argument("--mesh", default="1x4", help="mesh shape rows x cols (DP x TP)")
    args = ap.parse_args(argv)

    rows, cols = (int(v) for v in args.mesh.lower().split("x"))

    # The demo is the device OWNER: one open here, threaded into build_pipeline.
    device = open_mesh(rows, cols)
    try:
        print(f"[demo] building resident pipeline on {device.get_num_devices()} chip(s) ...")
        t0 = time.time()
        pipe = pl.build_pipeline(device, layers=args.layers, max_seq_len=args.max_seq_len)
        print(f"[demo] built in {time.time() - t0:.1f}s: {pipe.describe()}")

        print(f"\n[prompt] {args.prompt}\n[assistant] ", end="", flush=True)
        t0 = time.time()
        out = pl.run_text_generation(
            pipe,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            collect_logits=False,
            verbose=True,
        )
        dt = time.time() - t0
        print(f"\n\n[demo] {len(out['new_ids'])} tokens in {dt:.1f}s ({len(out['new_ids']) / dt:.2f} tok/s)")
        print(f"[demo] graduated modules invoked: {sorted(out['invoked'])}")
    finally:
        close_mesh(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
