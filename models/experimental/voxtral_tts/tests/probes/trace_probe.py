"""Is the p150's ~68 us per-op floor DEVICE time or HOST DISPATCH? 6.38's experiment, re-run.

This is the measurement that validates or undermines the reasoning behind six of today's
changes. Every per-op number in 6.41-6.48 is perf_counter around synchronize_device, which
INCLUDES host dispatch. 6.38 answered exactly this objection on the N150 -- _solve eager
19.145 ms vs traced 19.230, so dispatch was 0% -- but that was a different chip AND a different
host, and this box has 8 shared cores.

If dispatch is hidden here too, the op-count strategy is right and tracing is worth nothing.
If a large share of the 68 us is host, tracing removes it wholesale and several of today's
conclusions were measured through a host-cost lens.

TWO HAZARDS, both documented and both respected here:
  * trap #1 -- an exception escaping between begin_trace_capture and end_trace_capture hangs
    close_device and WEDGES THE CARD for every later run. Capture is in a try/finally.
  * 95dc26363f -- merely passing trace_region_size to open_device shifts the allocator enough to
    move a free-running trajectory (case 2 went 458 -> 464 frames with the trace OFF). So this
    process only ever TIMES; it never generates audio, and its numbers are not comparable to a
    run opened without the region.

This is trace-as-MEASUREMENT. 6.26's rejection of trace-as-SHIPPING (three silent failure modes
for 0.7%) is a separate question and untouched.
"""
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, HEAD_DIM, N_ACOUSTIC_CODEBOOK)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
L1_SMALL, TRACE_REGION = 65536, 200 * 1024 * 1024
REPS, ROUNDS = 40, 5


def timed(dev, fn, reps=REPS, rounds=ROUNDS):
    fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / reps * 1e3)
    return sum(out) / len(out), max(out) - min(out)


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL, trace_region_size=TRACE_REGION)
    tids = []
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=1024)
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
        embeds = pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), pipe.wb)
        h = pipe.backbone.prefill_last(embeds)[:, 0]
        gen, bb = pipe.flow, pipe.backbone
        pos = bb.pos
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()

        # ---------- Block 2: _solve, a pure device graph by construction ([flow-18]) ----------
        torch.manual_seed(0)
        x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK)
        xd = gen._up(x0.reshape(1, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
        hd = gen._up(gen._cfg_input(1, h))
        solve = lambda: gen._solve(xd, hd, 1, flow.N_DECODING_STEPS, flow.CFG_ALPHA)
        e_ms, e_spr = timed(dev, solve)
        solve()                                   # ensure every program is compiled
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            solve()
        finally:                                  # trap #1: never leave a capture open
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            tids.append(tid)
        t_ms, t_spr = timed(dev, lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False))
        print(f"\n=== Block 2  _solve (7 Euler steps, ~600 ops) ===")
        print(f"  eager  {e_ms:8.3f} ms  spread {e_spr:.3f}")
        print(f"  traced {t_ms:8.3f} ms  spread {t_spr:.3f}")
        print(f"  -> dispatch is {max(0.0, (e_ms - t_ms)) / e_ms * 100:5.1f}% of the eager time "
              f"({e_ms - t_ms:+.3f} ms)")
        print(f"  N150 (6.38): eager 19.145, traced 19.230, dispatch 0%")

        # ---------- Block 1: 26 decode layers, pure device once cos/sin/pos_t are hoisted ------
        up = lambda t: ttnn.from_torch(t.contiguous(), dtype=bb.dtype,
                                       layout=ttnn.TILE_LAYOUT, device=dev)
        cb, sb = gpt.rope_tables(1, offset=pos)
        cos = ttnn.to_memory_config(up(cb.reshape(1, 1, 1, HEAD_DIM)), gpt._ROPE_SHARD)
        sin = ttnn.to_memory_config(up(sb.reshape(1, 1, 1, HEAD_DIM)), gpt._ROPE_SHARD)
        pos_t = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=dev)
        xin = up(bref.embed_frame(pipe.wb, frames[0]).reshape(1, 1, DIM))
        layers = list(zip(bb.layers, bb.caches))

        def step26():
            x = ttnn.clone(xin)                   # add_ would otherwise consume xin (6.47)
            for lw, cache in layers:
                x = bb._layer_step(x, lw, cos, sin, cache, pos_t)
            return bb._norm(x, bb.norm)

        e2_ms, e2_spr = timed(dev, step26)
        step26()
        ttnn.synchronize_device(dev)
        tid2 = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            step26()
        finally:
            ttnn.end_trace_capture(dev, tid2, cq_id=0)
            tids.append(tid2)
        t2_ms, t2_spr = timed(dev, lambda: ttnn.execute_trace(dev, tid2, cq_id=0, blocking=False))
        print(f"\n=== Block 1  26-layer decode step (~470 ops) ===")
        print(f"  eager  {e2_ms:8.3f} ms  spread {e2_spr:.3f}")
        print(f"  traced {t2_ms:8.3f} ms  spread {t2_spr:.3f}")
        print(f"  -> dispatch is {max(0.0, (e2_ms - t2_ms)) / e2_ms * 100:5.1f}% of the eager time "
              f"({e2_ms - t2_ms:+.3f} ms)")
        print(f"  N150 (6.26): 24.86 -> 24.69, +0.17 ms (1.007x)")

        tot_e, tot_t = e_ms + e2_ms, t_ms + t2_ms
        print(f"\n=== both blocks ===")
        print(f"  eager {tot_e:.2f} ms/frame   traced {tot_t:.2f}   "
              f"delta {tot_e - tot_t:+.2f} ms/frame")
        print("  (timing only -- this process never generates audio: trace_region_size shifts "
              "the\n   allocator and 95dc26363f measured that moving a free-running trajectory.)")
    finally:
        for t in tids:
            try:
                ttnn.release_trace(dev, t)
            except Exception as e:
                print(f"  release_trace failed: {type(e).__name__}")
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
