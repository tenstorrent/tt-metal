"""Per-language DistillMOS over the clips generate_language_set.py wrote. Runs in the MOS venv.

Prints MOS_LANG_<code> lines for quality_report.py to parse, plus MOS_LANG_MIN, so a single language
degrading is visible where a pooled mean hides it.

    /tmp/mosvenv/bin/python tests/probes/mos_perlang.py base
"""
import json, os, sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import distillmos

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(HERE, "generated")

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
d = os.path.join(GEN, f"lang_{tag}")
rows = json.load(open(os.path.join(d, "manifest.json")))

m = distillmos.ConvTransformerSQAModel()
m.eval()


def score(path):
    x, sr = sf.read(path, dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(np.asarray(x)).reshape(1, -1), sr, 16000)
    with torch.no_grad():
        return float(m(x).item())


by_lang = {}
print(f"  {'lang':>5} {'voice':<16} {'s':>2} {'words':>5} {'sec':>6} {'MOS':>6}")
for r in rows:
    v = score(os.path.join(d, r["file"]))
    by_lang.setdefault(r["lang"], []).append(v)
    print(f"  {r['lang']:>5} {r['voice']:<16} {r['sentence']:>2} {r['words']:>5} "
          f"{r['seconds']:>6.1f} {v:>6.3f}", flush=True)

print()
means = {}
for lang in sorted(by_lang):
    vals = by_lang[lang]
    means[lang] = float(np.mean(vals))
    print(f"MOS_LANG_{lang} {means[lang]:.4f}   n={len(vals)} min={min(vals):.3f} "
          f"max={max(vals):.3f}")
print(f"MOS_LANG_MIN {min(means.values()):.4f}")
print(f"MOS_LANG_SPREAD {max(means.values()) - min(means.values()):.4f}")
