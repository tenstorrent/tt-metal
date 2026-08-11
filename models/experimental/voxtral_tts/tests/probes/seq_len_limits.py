# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How long can a prompt be, how long can an utterance be, and what does length cost? (§6.69)

STATUS said "prefill beyond ~1024 tokens needs chunked prefill" in three places. Nothing in the
code enforces 1024 -- the only guard is `Sp > self.max_seq_len` (gpt.py:344) and max_seq_len is a
constructor argument. This probe is what replaced that claim with measurements.

Two questions the phrase "long utterance" conflates:

  A. PROMPT length -> prefill. Attention here is explicit q@k^T, so the score matrix is
     [1, 32, S, S] and grows QUADRATICALLY. If anything broke with length it would be this.
  B. UTTERANCE length -> decode. Every generated frame writes ONE cache position, so the cache
     holds prompt + frames TOGETHER. This is the limit that actually caps a TTS utterance: a
     prompt is a few hundred tokens and the audio is thousands of frames.

C then asks what length COSTS -- allocation and depth measured separately, because "flat in depth"
and "free to allocate" are different claims and only both together mean length is free.

IDLE THE BOX FIRST. `_traced_frame` does real host work every frame, so a busy CPU inflates every
frame uniformly and independently of position -- indistinguishable by eye from a hardware effect.
§6.69 nearly shipped a thermal-droop story that was really a stray `find /`.

    python tests/probes/seq_len_limits.py [--deep]     # --deep adds the 3900-frame grind, ~2 min
"""
import argparse
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, HEAD_DIM, N_KV_HEADS
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipe
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAME_RATE = 12.5          # frames per second of audio; 80 ms real time per frame


def cache_mb(S, n_layers=26):
    return n_layers * 2 * N_KV_HEADS * S * HEAD_DIM * 2 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="add the 3900-frame grind")
    args = ap.parse_args()

    dev = open_device(trace_region_size=pipe.TRACE_REGION_SIZE)
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][1]
    try:
        # ---- A. PROMPT length --------------------------------------------------------------
        print("=== A. prompt length: does prefill have a ceiling? ===")
        wb = bref.load_backbone_state()
        g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=4096)
        print(f"  {'S':>6} {'padded':>7} {'prefill s':>10}  result")
        for S in (256, 512, 1024, 2048, 3072, 4096):
            g.pos = 0
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            h = g.prefill(torch.randn(1, S, DIM) * 0.02, last_only=True)
            ttnn.synchronize_device(dev)
            Sp = (S + gpt.PREFILL_MULTIPLE - 1) // gpt.PREFILL_MULTIPLE * gpt.PREFILL_MULTIPLE
            print(f"  {S:>6} {Sp:>7} {time.perf_counter()-t0:>10.2f}  "
                  f"{'ok' if torch.isfinite(h).all() else 'NON-FINITE'}")
        del g
        print("  params.json context is 65536 with no sliding window -- length costs KV cache only.")

        # ---- B/C. UTTERANCE length, and what allocation costs -------------------------------
        print("\n=== B+C. utterance length: cache holds prompt + frames, and allocation is free ===")
        print(f"  {'max_seq_len':>11} {'cache MB':>9} {'audio @350-tok prompt':>22} {'ms/frame':>9}")
        emb = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                       pref.load_voice(case["voice"]), bref.load_backbone_state())
        for M in (1024, 2048, 4096):
            p = pipe.TtVoxtralPipeline(dev, max_seq_len=M)
            c = p.flow(p.backbone.prefill_last(emb)[:, 0], cfg_alpha=pipe.CFG_ALPHA)
            p._trace_capture(pipe.CFG_ALPHA, pipe.N_DECODING_STEPS)
            try:
                c = _run(p, c, 40)                       # warm
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                c = _run(p, c, 60)
                ttnn.synchronize_device(dev)
                print(f"  {M:>11} {cache_mb(M):>9.0f} {(M-350)/FRAME_RATE:>20.0f} s "
                      f"{(time.perf_counter()-t0)/60*1e3:>9.2f}")
            finally:
                p._trace_release()
            del p

        # ---- D. does DEPTH cost anything? ---------------------------------------------------
        if args.deep:
            print("\n=== D. depth: 3900 frames deep vs a shallow warm band ===")
            p = pipe.TtVoxtralPipeline(dev, max_seq_len=4096)
            c = p.flow(p.backbone.prefill_last(emb)[:, 0], cfg_alpha=pipe.CFG_ALPHA)
            p._trace_capture(pipe.CFG_ALPHA, pipe.N_DECODING_STEPS)
            try:
                c = _run(p, c, 30)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                c = _run(p, c, 60)
                ttnn.synchronize_device(dev)
                shallow = (time.perf_counter() - t0) / 60 * 1e3
                n = 3900 - p.backbone.pos
                t0 = time.perf_counter()
                c = _run(p, c, n)
                ttnn.synchronize_device(dev)
                deep = (time.perf_counter() - t0) / n * 1e3
                print(f"  shallow warm band          {shallow:>7.2f} ms/frame")
                print(f"  {n} frames out to pos {p.backbone.pos:<5} {deep:>7.2f} ms/frame")
                print("  Flat => utterance length costs DRAM, not RTF.")
            finally:
                p._trace_release()
    finally:
        ttnn.close_device(dev)


def _run(p, c, n):
    """n traced frames, ignoring [END_AUDIO] -- this measures TIME, and a stop token would end the
    run long before the cache gets deep."""
    for _ in range(n):
        c = p._traced_frame(c[0])
        if int(c[0, 0]) == pipe.END_AUDIO_ID:
            c[0, 0] = 100
    return c


if __name__ == "__main__":
    main()
