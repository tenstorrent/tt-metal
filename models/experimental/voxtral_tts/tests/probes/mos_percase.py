"""Per-case DistillMOS for the sampler clips, so a listener knows what the predictor thought."""
import json, os, glob
import numpy as np, soundfile as sf, torch, torchaudio, distillmos
V = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(V, "generated")
m = distillmos.ConvTransformerSQAModel(); m.eval()
def score(p):
    x, sr = sf.read(p, dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(np.asarray(x)).reshape(1,-1), sr, 16000)
    with torch.no_grad(): return float(m(x).item())
fx = json.load(open(os.path.join(V, "tests/prompt_fixture.json")))["cases"]
res = {r["case"]: r for r in json.load(open(os.path.join(GEN, "results_prg_s0.json")))}
print(f"  {'case':<5} {'voice':<16} {'words':>5} {'MOS s0':>7} {'s1':>6} {'s2':>6}  text")
rows = []
for ci, c in enumerate(fx):
    ss = []
    for sd in (0,1,2):
        p = os.path.join(GEN, f"case{ci}_{c['voice']}_prg_s{sd}.wav")
        ss.append(score(p) if os.path.exists(p) else float("nan"))
    rows.append((ci, c, ss))
    print(f"  {ci:<5} {c['voice']:<16} {len(c['text'].split()):>5} {ss[0]:>7.2f} {ss[1]:>6.2f} "
          f"{ss[2]:>6.2f}  {c['text'][:42]!r}")
lf = [r for r in rows if len(r[1]['text'].split()) >= 20]
allv = [v for _,_,ss in rows for v in ss]
print(f"\n  long-form cases only (the WER bucket): mean MOS "
      f"{np.mean([v for _,_,ss in lf for v in ss]):.3f}")
print(f"  all 45 (15 cases x 3 seeds):            mean MOS {np.mean(allv):.3f}")
