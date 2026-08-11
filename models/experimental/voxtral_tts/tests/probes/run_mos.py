"""Automated MOS estimate on every generated utterance, via DistillMOS.

6.58 said naturalness could not be assessed without human raters. That is true of MOS proper, but
a NO-REFERENCE MOS PREDICTOR is exactly the standard automated proxy and was never run. DistillMOS
(xls-r-sqa distilled) predicts a 1-5 MOS from the waveform alone -- no reference needed, so it
scores the device output on its own terms.

The number that matters is NOT the absolute MOS -- predictors are miscalibrated across domains --
but the DEVICE vs fp32 REFERENCE delta on identical prompts, and the spread across 90 utterances.
If the device scores within noise of the reference, the port costs nothing perceptually, which is
the actual question. Run from an ISOLATED venv: DistillMOS needs torchaudio, which STATUS 2
records as breaking transformers in the main one.
"""
import glob, json, os, sys
import numpy as np, soundfile as sf, torch
import distillmos

V = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(V, "generated")
m = distillmos.ConvTransformerSQAModel(); m.eval()

def score(path):
    x, sr = sf.read(path, dtype="float32")
    x = torch.from_numpy(np.asarray(x)).reshape(1, -1)
    if sr != 16000:                      # DistillMOS expects 16 kHz
        import torchaudio
        x = torchaudio.functional.resample(x, sr, 16000)
    with torch.no_grad():
        return float(m(x).item())

fx = json.load(open(os.path.join(V, "tests", "prompt_fixture.json")))["cases"]
print("=== DEVICE vs fp32 REFERENCE on identical prompts (the comparison that matters) ===")
print(f"  {'pair':<34} {'device':>8} {'fp32 ref':>9} {'delta':>7}")
for p in sorted(glob.glob(os.path.join(GEN, "*_FP32REF_s*.wav"))):
    b = os.path.basename(p); ci = int(b.split("_")[0].replace("case","")); sd = b.split("_s")[-1][:-4]
    d = os.path.join(GEN, f"case{ci}_{fx[ci]['voice']}_prg_s{sd}.wav")
    if not os.path.exists(d): continue
    md, mr = score(d), score(p)
    print(f"  case {ci} {fx[ci]['voice']:<16} seed {sd}   {md:>8.3f} {mr:>9.3f} {md-mr:>+7.3f}")

print("\n=== all 90 device utterances (15 cases x 3 seeds x 2 arms) ===")
by_arm = {}
for f in sorted(glob.glob(os.path.join(GEN, "case*_prg_s*.wav")) +
                glob.glob(os.path.join(GEN, "case*_base_s*.wav"))):
    arm = "prg (HEAD)" if "_prg_s" in f else "baseline"
    by_arm.setdefault(arm, []).append((score(f), os.path.basename(f)))
for arm, v in sorted(by_arm.items()):
    s = sorted(x[0] for x in v)
    print(f"  {arm:<12} n={len(s):<3} mean {np.mean(s):.3f}  median {np.median(s):.3f}  "
          f"min {s[0]:.3f}  max {s[-1]:.3f}")
    worst = sorted(v)[:3]
    print(f"  {'':<12} worst: " + ", ".join(f"{n.replace('.wav','')} {sc:.2f}" for sc, n in worst))
