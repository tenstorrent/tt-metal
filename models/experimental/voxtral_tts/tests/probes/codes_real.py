"""Is the codes gate's 86/288 a REAL accuracy problem, or an artefact of its synthetic input?

`--gate codes` reads 29.5% of acoustic codes differing from the fp32 reference, which is alarming
next to a long-form WER of 1 wrong in 894. Something has to give. Two facts about that gate:

  * it is TEACHER-FORCED, so the number is NOT compounding trajectory divergence -- each frame is
    an independent "same input, same codes?" test. That was my first hypothesis and it is wrong.
  * it runs on `torch.randn(1, 128, 3072) * 0.02` -- SYNTHETIC embeddings. NOTES trap #12 says in
    as many words that random inputs are a pessimistic proxy, and gate_wiring prints the same
    warning. Nobody has re-read the codes gate in that light.

Random embeddings put the backbone far off its training manifold, where the semantic head's top
logits are near-tied and the flow's velocities land near FSQ bin boundaries. Both make integer
outputs flip on arithmetic noise that would be invisible on real text.

So this runs the IDENTICAL comparison on the real prompt fixture, and additionally reports what
the gate never has: the DISTRIBUTION of |delta|, and the margin at each flip. An off-by-one on a
21-level FSQ axis is the smallest perturbation representable; a flip with a large margin would be
a real bug. Those two cases are indistinguishable in "85/288".
"""
import json
import os
from collections import Counter

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
N_FRAMES = 8


def run(pipe, embeds, label, wf):
    """Exactly gate_codes' comparison: teacher-forced on the reference's own codes."""
    ref_dec = bref.IncrementalBackbone(pipe.wb)
    h_ref = ref_dec.prefill(embeds)
    h_dev = pipe.backbone.prefill_last(embeds)
    sem_bad = ac_bad = total = 0
    deltas = Counter()
    for i in range(N_FRAMES):
        torch.manual_seed(1000 + i)
        c_ref = fref.reference_frame(h_ref[:, 0], wf, cfg_alpha=CFG_ALPHA)
        torch.manual_seed(1000 + i)
        c_dev = pipe.flow(h_dev[:, 0], cfg_alpha=CFG_ALPHA)
        s_ref, s_dev = int(c_ref[0, 0]), int(c_dev[0, 0])
        d = (c_ref[0, 1:] - c_dev[0, 1:]).abs()
        for v in d.tolist():
            if v:
                deltas[int(v)] += 1
        sem_bad += s_ref != s_dev
        ac_bad += int((d != 0).sum())
        total += 36
        if s_ref == END_AUDIO_ID or s_dev == END_AUDIO_ID:
            break
        emb = bref.embed_frame(pipe.wb, c_ref[0])
        h_ref = ref_dec.step(emb)
        h_dev = pipe.backbone.step(emb).reshape(1, 1, -1)
    pct = ac_bad / total * 100
    off1 = deltas.get(1, 0)
    print(f"  {label:<34} semantic {sem_bad}/{total//36}   acoustic {ac_bad:>3}/{total} "
          f"({pct:>4.1f}%)   |delta| { {k: deltas[k] for k in sorted(deltas)} }")
    if ac_bad:
        print(f"  {'':<34} of those, {off1}/{ac_bad} ({off1/ac_bad*100:.0f}%) are OFF BY ONE "
              f"on a 21-level axis")
    return ac_bad, total


def main():
    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        wf = fref.load_flow_state()
        fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))

        print("=== what the gate actually runs: SYNTHETIC embeddings ===")
        torch.manual_seed(0)
        run(pipe, torch.randn(1, 128, 3072) * 0.02, "randn(1,128,3072)*0.02  [the gate]", wf)

        print("\n=== the same comparison on REAL prompts ===")
        tot_bad = tot_all = 0
        for ci in (0, 2, 3, 5, 10):
            case = fx["cases"][ci]
            embeds = pref.build_inputs_embeds(
                torch.tensor(case["ids"], dtype=torch.long),
                pref.load_voice(case["voice"]), pipe.wb)
            b, t = run(pipe, embeds, f"case {ci} ({case['voice']}, P={len(case['ids'])})", wf)
            tot_bad += b
            tot_all += t
        print(f"\n  REAL-PROMPT TOTAL: {tot_bad}/{tot_all} ({tot_bad/tot_all*100:.1f}%)")
        print("\nFor scale: --gate decode over 15 real prompts x 22 frames reads mean 0.91% "
              "worst-sample\nerror and min PCC 0.9994, and long-form WER is 1 wrong of 894.")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
