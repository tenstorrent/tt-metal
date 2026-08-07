"""WHY does the device flip 29.5% of codes on synthetic input and 3.9% on real? Close the loop.

ref_vs_ref.py just removed the explanation 6.54 offered. Running the reference against its own
float64 self gives 0/288 on BOTH populations, and the distance from the pre-round value to the
nearest FSQ flip boundary is the SAME for both (median 0.260 synthetic, 0.253 real). So random
inputs do NOT sit near bin boundaries, and 6.54's sentence saying they do is wrong.

What is left is the other factor in the same inequality: a code flips when

        |implementation error|  >  |distance to the .5 boundary|

The margins match, so the DEVICE's error must be the term that differs. fp32 lands at ~1.3e-6,
five orders below the margin, which is why the reference never flips at all. This measures the
device's error on the same pre-FSQ quantity, for both populations, and then checks the prediction
quantitatively: the fraction of codes whose margin is smaller than the device's error should
reproduce the observed flip rate. If it does, the mechanism is settled and it is about MAGNITUDE,
not about boundary proximity.

Everything is compared against float64, the only defensible truth here.
"""
import json
import os

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE, N_ACOUSTIC_CODEBOOK)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA, N_DECODING_STEPS
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
N_DRAWS, L = 8, ACOUSTIC_CODEBOOK_SIZE - 1


def _te_any(t, inv_freq):                       # dtype-generic; upstream hardcodes t.float()
    emb = t.to(inv_freq.dtype) @ inv_freq.unsqueeze(0)
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


fref.time_embedding = _te_any


def to_v(x):
    """the exact quantity FSQ rounds: (clamp(x,-1,1)+1)/2 * (levels-1)"""
    return (torch.clamp(x, -1, 1) + 1) / 2 * L


def main():
    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        wf = fref.load_flow_state()
        wf64 = {k: (v.double() if torch.is_floating_point(v) else v) for k, v in wf.items()}

        torch.manual_seed(0)
        h_syn = pipe.backbone.prefill_last(torch.randn(1, 128, 3072) * 0.02)[:, 0]
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
        h_real = pipe.backbone.prefill_last(pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long),
            pref.load_voice(case["voice"]), pipe.wb))[:, 0]

        print(f"  {'population':<12} {'|x_dev - x64|':>22} {'margin':>18} "
              f"{'predicted':>10} {'observed':>9}")
        for label, h in (("SYNTHETIC", h_syn), ("REAL", h_real)):
            errs, margs, flips, tot = [], [], 0, 0
            for d in range(N_DRAWS):
                g = torch.Generator().manual_seed(4000 + d)
                x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK, generator=g)
                sem = fref.semantic_code(h, wf)
                # fp64 truth, same inputs
                _, tr64 = fref.decode_frame(sem, h.double(), wf64, x_0=x0.double(),
                                            return_trace=True)
                v64 = to_v(tr64[-1])
                # device, same inputs -- the pre-FSQ solver state, before the host round
                xd = pipe.flow._solve(
                    pipe.flow._up(x0.reshape(1, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32),
                    pipe.flow._up(pipe.flow._cfg_input(1, h)), 1, N_DECODING_STEPS, CFG_ALPHA)
                v_dev = to_v(ttnn.to_torch(xd).double().reshape(1, N_ACOUSTIC_CODEBOOK))
                errs.append((v_dev - v64).abs().flatten())
                margs.append((0.5 - (v64 - v64.round()).abs()).flatten())
                flips += int((v_dev.round().long() != v64.round().long()).sum())
                tot += N_ACOUSTIC_CODEBOOK
            e, m = torch.cat(errs), torch.cat(margs)
            pred = float((m < e).float().mean()) * 100
            print(f"  {label:<12} median {e.median():.4f}  max {e.max():.3f}   "
                  f"median {m.median():.3f}   {pred:>8.1f}% {flips/tot*100:>8.1f}%")
        print("\n  A code flips when |device error| > |distance to the .5 boundary|.")
        print("  The margins are the same for both populations (ref_vs_ref.py), so the ERROR is")
        print("  what differs. fp32 sits at ~1.3e-6 and never flips anything.")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
