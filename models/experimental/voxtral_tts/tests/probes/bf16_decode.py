"""What does bf16 through the WHOLE DECODE cost and buy? (weights, not activations)

6.16 measured this ladder on the N150 and landed on BFP8-everywhere-except-w2; 6.55 re-priced it
here but only on PREFILL accuracy, where the answer was "nothing" (real error pinned at 0.70%
across the whole ladder). Decode is the other half and the one that costs 26 weight reads per
frame, so it is where the ladder was chosen in the first place.

Accuracy is measured TEACHER-FORCED against the fp32 reference -- both sides advance on the same
real frames, so each step is independent (the trap gate_decode documents). PCC of the decode
hidden state is the quantity Block 2 actually consumes.
"""
import os, time, torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, pcc
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device
import json
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
dev = open_device()
try:
    wb = bref.load_backbone_state()
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
    embeds = pref.build_inputs_embeds(torch.tensor(fx["ids"], dtype=torch.long),
                                      pref.load_voice(fx["voice"]), wb)
    frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
    print(f"  {'decode weights':<28} {'ms/step':>9} {'vs ships':>9} {'min PCC':>10} "
          f"{'mean worst-sample':>18}")
    base = None
    for lbl, w2d, ffd, attnd in (("BFP8 FF+attn, w2 bf16 SHIPS", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b),
                                 ("bf16 FF, BFP8 attn",           ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b),
                                 ("bf16 EVERYTHING",              ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16)):
        gpt.WEIGHT_DTYPE, gpt.FF_WEIGHT_DTYPE, gpt.ATTN_WEIGHT_DTYPE = w2d, ffd, attnd
        g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=512)
        rd = bref.IncrementalBackbone(wb)
        h_ref = rd.prefill(embeds); g.prefill(embeds, last_only=True)
        ps, ws = [], []
        for i in range(12):
            emb = bref.embed_frame(wb, frames[i])
            h_ref = rd.step(emb); hd = g.step(emb.reshape(1, 1, DIM))
            ps.append(pcc(hd, h_ref))
            ws.append(((hd - h_ref).abs().max() / h_ref.abs().max()).item() * 100)
        ttnn.synchronize_device(dev)
        x = bref.embed_frame(wb, frames[0]).reshape(1, 1, DIM)
        t0 = time.perf_counter()
        for _ in range(20): g.step(x)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        if base is None: base = ms
        print(f"  {lbl:<28} {ms:>9.2f} {ms-base:>+9.2f} {min(ps):>10.6f} "
              f"{sum(ws)/len(ws):>17.2f}%")
        del g
    print("\n  6.16 on the N150: bf16-all 45.8 ms/step vs 31.4 for the shipped ladder, and it")
    print("  priced reverting w2 alone at 2.5 ms for 77% of the accuracy -- hence the split.")
finally:
    gpt.WEIGHT_DTYPE, gpt.FF_WEIGHT_DTYPE = ttnn.bfloat16, ttnn.bfloat8_b
    gpt.ATTN_WEIGHT_DTYPE = ttnn.bfloat8_b
    ttnn.close_device(dev)
