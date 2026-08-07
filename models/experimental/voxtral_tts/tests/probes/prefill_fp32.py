"""Would a HIGHER-PRECISION PREFILL buy anything? Prefill runs once per utterance, so unlike
decode it can afford almost any precision -- the cost amortises over hundreds of frames.

6.55 found real-prompt prefill error PINNED at 0.70% relative across the whole WEIGHT ladder
(bf8 -> bf16 -> all bf16 moved PCC by +0.00004 and the relative error not at all). That points at
the ACTIVATIONS, which are bf16 (`DTYPE`), not at the weights. So testing higher-precision weights
alone would repeat 6.55; this raises both, separately, to find which one is actually binding:

    shipped     weights bf8/bf16, activations bf16       <- what ships
    w fp32      weights float32,  activations bf16       <- is it the weights?  (6.55 says no)
    a fp32      weights bf8/bf16, activations float32    <- is it the activations?
    both fp32   weights float32,  activations float32    <- the ceiling

If "both fp32" still reads ~0.70%, the residual is neither, and no prefill precision change can
help -- it would be the reference/device algebra itself (norm epsilon placement, accumulation
order, the halfsplit RoPE permute). If it drops sharply, prefill-only high precision is a real
lever and the question becomes whether it costs anything worth paying.

COSTS THAT WOULD MATTER IF IT WORKS, measured here rather than hand-waved:
  * prefill runs once, so its own time is amortised over ~450 frames -- +1 s of prefill is
    +2 ms/frame, which is NOT negligible at 39 ms/frame. Timed.
  * prefill shares `self.layers` with decode, so a separate fp32 copy DOUBLES weight memory
    (~13.6 GB for the 3.4B backbone at fp32, on top of the ~3.6 GB decode copy). Reported.
"""
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
F32, BF16, BF8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
#      label            w2     FF    attn   activations
ARMS = [("shipped   ", BF16, BF8, BF8, BF16),
        ("w fp32    ", F32, F32, F32, BF16),
        ("a fp32    ", BF16, BF8, BF8, F32),
        ("both fp32 ", F32, F32, F32, F32)]


def main():
    dev = open_device()
    try:
        wb = bref.load_backbone_state()
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
        real = pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), wb)
        print(f"  fp32 CPU reference prefill, P={real.shape[1]}")
        ref = bref.reference_forward(real, wb)[:, -1:]

        print(f"\n  {'arm':<12} {'PCC last':>11} {'rel err':>9} {'prefill s':>10} "
              f"{'/frame @450':>12}  note")
        base = None
        for lbl, w2d, ffd, attnd, actd in ARMS:
            gpt.WEIGHT_DTYPE, gpt.FF_WEIGHT_DTYPE = w2d, ffd
            gpt.ATTN_WEIGHT_DTYPE, gpt.DTYPE = attnd, actd
            try:
                g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=256)
                g.prefill(real)                                   # compile
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                got = g.prefill(real)[:, -1:]
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                p = pcc(got, ref)
                rel = ((got - ref).abs().max() / ref.abs().max()).item() * 100
                if base is None:
                    base = (p, dt)
                print(f"  {lbl:<12} {p:>11.6f} {rel:>8.2f}% {dt:>10.2f} "
                      f"{(dt-base[1])/450*1e3:>+11.2f}ms  "
                      f"{'' if base[0]==p else f'dPCC {p-base[0]:+.6f}'}")
                del g
            except Exception as e:
                print(f"  {lbl:<12} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:64]}")
        print("\n  6.55: the weight ladder alone (bf8 -> bf16 -> all bf16) left the real-prompt")
        print("  relative error at 0.70% throughout, for +4.91 ms/step of DECODE cost.")
        print("  A separate fp32 prefill copy would add ~13.6 GB of weights on top of decode's.")
    finally:
        gpt.WEIGHT_DTYPE, gpt.FF_WEIGHT_DTYPE = BF16, BF8
        gpt.ATTN_WEIGHT_DTYPE, gpt.DTYPE = BF8, BF16
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
