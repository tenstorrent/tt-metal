"""Does the REFERENCE also disagree with itself on synthetic inputs? (STATUS.md 6.54 follow-up)

6.54 showed the codes gate reads 85/288 on synthetic embeddings and 34/864 on real ones, and that
every real-prompt difference is off-by-one. But that compared DEVICE against reference, so it
still leaves the device as the only suspect. The control that removes it entirely: run the fp32
CPU reference against its own float64 self. No device, same code, same weights, only precision.

If fp32-vs-fp64 flips a comparable number of codes on synthetic input and ~none on real, then the
codes gate is measuring the INPUT's proximity to FSQ bin boundaries, not any property of the
device -- and a bf16/bf8 device is simply the same phenomenon with a bigger epsilon.

WHY THE INPUT WOULD MATTER. FSQ is code = round((clamp(x,-1,1)+1) * (L-1)/2), L=21. A code flips
when the pre-round value crosses a .5 boundary, so flips are governed by
    (how far the pre-round value sits from .5)  vs  (the implementation's error)
Random embeddings are off-manifold, so |x| is larger and clamps harder, and the accumulated
velocity error over 7 Euler steps is larger too. Both are measured here rather than asserted.

Block 2 only, 8 independent noise draws on a fixed h -- 8*36 = 288 codes, the gate's own
denominator, with no teacher-forced chain to confound it.
"""
import json
import os

import torch

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE, N_ACOUSTIC_CODEBOOK)

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
N_DRAWS = 8
L = ACOUSTIC_CODEBOOK_SIZE - 1


# reference/voxtral_flow_ref.time_embedding hardcodes t.float(), so the fp64 arm cannot run
# unpatched. This is dtype-generic and otherwise identical (same cat(cos, sin) ORDER).
def _time_embedding_any(t, inv_freq):
    emb = t.to(inv_freq.dtype) @ inv_freq.unsqueeze(0)
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


fref.time_embedding = _time_embedding_any


def codes_and_margin(h, w, x0, dtype):
    """Run Block 2 at `dtype`; return the integer codes and the pre-round value."""
    hh = h.to(dtype)
    ww = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in w.items()}
    sem = fref.semantic_code(hh, ww)
    _, trace = fref.decode_frame(sem, hh, ww, x_0=x0.to(dtype), return_trace=True)
    x = trace[-1]                                    # pre-FSQ solver state
    v = (torch.clamp(x, -1, 1) + 1) / 2 * L          # the value that gets rounded
    return v.round().long(), v


def report(label, h, w):
    n_diff = 0
    margins, errs, clamped = [], [], 0
    for d in range(N_DRAWS):
        g = torch.Generator().manual_seed(4000 + d)
        x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK, generator=g)
        c32, v32 = codes_and_margin(h, w, x0, torch.float32)
        c64, v64 = codes_and_margin(h, w, x0, torch.float64)
        n_diff += int((c32 != c64).sum())
        # distance from the rounded value to the nearest flip boundary, in the same units as v
        margins.append((0.5 - (v64 - v64.round()).abs()).flatten())
        errs.append((v32.double() - v64).abs().flatten())
        clamped += int((v64.abs() >= L - 1e-9).sum() + (v64 <= 1e-9).sum())
    m = torch.cat(margins)
    e = torch.cat(errs)
    tot = N_DRAWS * N_ACOUSTIC_CODEBOOK
    print(f"  {label:<38} fp32 vs fp64: {n_diff:>3}/{tot} ({n_diff/tot*100:>4.1f}%)")
    print(f"  {'':<38} fp32 error   median {e.median():.2e}  max {e.max():.2e}")
    print(f"  {'':<38} boundary margin median {m.median():.3f}  "
          f"frac within the error {float((m < e).float().mean())*100:.1f}%")
    return n_diff, tot


def main():
    print("loading fp32 reference weights (backbone + flow, CPU only -- no device)")
    wb = bref.load_backbone_state()
    wf = fref.load_flow_state()

    # h exactly as gate_codes builds it: the REFERENCE backbone, fed each kind of input
    torch.manual_seed(0)
    syn_embeds = torch.randn(1, 128, 3072) * 0.02
    h_syn = bref.IncrementalBackbone(wb).prefill(syn_embeds)[:, 0]

    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
    real_embeds = pref.build_inputs_embeds(
        torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), wb)
    h_real = bref.IncrementalBackbone(wb).prefill(real_embeds)[:, 0]

    print(f"\n  |h| synthetic: mean {h_syn.abs().mean():.4f}  max {h_syn.abs().max():.3f}")
    print(f"  |h| real     : mean {h_real.abs().mean():.4f}  max {h_real.abs().max():.3f}\n")

    print("=== the REFERENCE against itself, fp32 vs fp64. No device anywhere. ===")
    report("SYNTHETIC randn(1,128,3072)*0.02", h_syn, wf)
    report(f"REAL prompt (case 0, {case['voice']})", h_real, wf)
    print("\nFor comparison, DEVICE vs fp32 reference on the same two populations (6.54):")
    print("  synthetic 85/288 (29.5%)      real 34/864 (3.9%), 100% off-by-one")


if __name__ == "__main__":
    main()
