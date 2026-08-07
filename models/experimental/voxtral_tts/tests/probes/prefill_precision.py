"""Can prefill accuracy be improved, and is the synthetic gate a useful CANARY for it?

6.54 traced the codes gate's synthetic 29.5% to Block 1: PCC(h_dev,h_ref) 0.9865 off-manifold
against 0.9999 on real prompts, established at PREFILL and flat thereafter. Two questions follow.

  1. IS THERE ANYTHING TO FIX? Real-prompt prefill is already PCC 0.999894 last-position and beats
     tt_transformers' own 0.999564 at P=200. 6.16 measured the whole weight-precision ladder and
     priced it per millisecond, landing on BFP8 everywhere except w2. So the expected answer is
     "no, this is the knee" -- but that was measured on real prompts only, and it is worth knowing
     what the remaining precision levers actually buy.

  2. IS THE SYNTHETIC INPUT USEFUL RATHER THAN JUST ALARMING? It amplifies Block 1 error 22x. If a
     known precision CHANGE moves the synthetic number a lot while real prompts barely register
     it, the gate is a sensitivity amplifier -- an early warning for regressions that real prompts
     would hide -- and should be kept and baselined rather than dismissed. If both move together,
     it is just noise and adds nothing.

ARMS are weight dtypes, the only lever 6.16 found that matters:
    shipped     FF1/FF3 BFP8, wqkv/wo BFP8, w2 bf16
    +bf16 FF    FF1/FF3 -> bf16
    all bf16    also wqkv/wo -> bf16      (6.16: costs 3.3 ms/step, buys 0.04 pp on real prompts)

Measured on BOTH populations so the amplification factor is visible, against the fp32 reference.
"""
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, pcc
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
ARMS = [("shipped   (FF bf8, attn bf8)", ttnn.bfloat8_b, ttnn.bfloat8_b),
        ("+bf16 FF  (FF bf16, attn bf8)", ttnn.bfloat16, ttnn.bfloat8_b),
        ("all bf16  (FF bf16, attn bf16)", ttnn.bfloat16, ttnn.bfloat16)]


def main():
    dev = open_device()
    try:
        wb = bref.load_backbone_state()
        torch.manual_seed(0)
        syn = torch.randn(1, 128, 3072) * 0.02
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
        real = pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), wb)

        print("  computing fp32 reference prefill for both populations (CPU)")
        ref_syn = bref.reference_forward(syn, wb)[:, -1:]
        ref_real = bref.reference_forward(real, wb)[:, -1:]

        print(f"\n  {'arm':<32} {'REAL PCC':>10} {'REAL rel':>9} {'SYN PCC':>10} {'SYN rel':>9} "
              f"{'ms/step':>8}")
        base = {}
        for lbl, ffd, attnd in ARMS:
            gpt.FF_WEIGHT_DTYPE, gpt.ATTN_WEIGHT_DTYPE = ffd, attnd
            g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=1024)
            out = {}
            for name, emb, ref in (("real", real, ref_real), ("syn", syn, ref_syn)):
                got = g.prefill(emb)[:, -1:]
                out[name] = (pcc(got, ref),
                             ((got - ref).abs().max() / ref.abs().max()).item() * 100)
            # decode step cost, the thing any precision change is paid for in
            g.pos = 0
            x = bref.embed_frame(wb, torch.zeros(37, dtype=torch.long)).reshape(1, 1, DIM)
            g.step(x); ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(20):
                g.step(x)
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) / 20 * 1e3
            print(f"  {lbl:<32} {out['real'][0]:>10.6f} {out['real'][1]:>8.2f}% "
                  f"{out['syn'][0]:>10.6f} {out['syn'][1]:>8.2f}% {ms:>8.2f}")
            base[lbl] = (out["real"][0], out["syn"][0], ms)
            del g

        print("\n  AMPLIFICATION -- how much each arm moves each population, vs shipped:")
        s = base[ARMS[0][0]]
        for lbl, _, _ in ARMS[1:]:
            a = base[lbl]
            dr, ds = a[0] - s[0], a[1] - s[1]
            print(f"  {lbl:<32} real {dr:>+10.6f}   synthetic {ds:>+10.6f}   "
                  f"ratio {abs(ds)/max(abs(dr), 1e-9):>6.1f}x   cost {a[2]-s[2]:>+6.2f} ms/step")
        print("\n  6.16 priced reverting wqkv+wo to bf16 at 3.3 ms/step for 0.04 pp on real prompts.")
    finally:
        gpt.FF_WEIGHT_DTYPE, gpt.ATTN_WEIGHT_DTYPE = ttnn.bfloat8_b, ttnn.bfloat8_b
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
