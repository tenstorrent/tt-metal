# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-language DistillMOS over a directory of clips. Runs in the MOS venv, NOT the main one.

Used by `test_mos.py` and the quality report (bring-up tooling, see the README). A separate venv
because DistillMOS needs torchaudio, which breaks transformers in the main venv.

    /tmp/mosvenv/bin/python tests/mos_score.py /path/to/clip_dir     # a dir with manifest.json
    /tmp/mosvenv/bin/python tests/mos_score.py base                  # = generated/lang_base

Prints MOS_LANG_<code> lines, MOS_LANG_MIN / MOS_LANG_SPREAD, and one MOS_JSON line with every
clip's score, for callers that must not depend on the table's formatting.
"""
import json, os, sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import distillmos

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(HERE, "generated")

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
d = tag if os.path.isdir(tag) else os.path.join(GEN, f"lang_{tag}")
rows = json.load(open(os.path.join(d, "manifest.json")))

m = distillmos.ConvTransformerSQAModel()
m.eval()


def score(path):
    x, sr = sf.read(path, dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(np.asarray(x)).reshape(1, -1), sr, 16000)
    with torch.no_grad():
        return float(m(x).item())


by_lang, clips = {}, []
print(f"  {'lang':>5} {'voice':<16} {'s':>2} {'words':>5} {'sec':>6} {'MOS':>6}")
for r in rows:
    v = score(os.path.join(d, r["file"]))
    by_lang.setdefault(r["lang"], []).append(v)
    clips.append({**r, "mos": v})
    print(
        f"  {r['lang']:>5} {r['voice']:<16} {r['sentence']:>2} {r['words']:>5} {r['seconds']:>6.1f} {v:>6.3f}",
        flush=True,
    )

print()
means = {}
for lang in sorted(by_lang):
    vals = by_lang[lang]
    means[lang] = float(np.mean(vals))
    print(f"MOS_LANG_{lang} {means[lang]:.4f}   n={len(vals)} min={min(vals):.3f} max={max(vals):.3f}")
print(f"MOS_LANG_MIN {min(means.values()):.4f}")
print(f"MOS_LANG_SPREAD {max(means.values()) - min(means.values()):.4f}")
print("MOS_JSON: " + json.dumps({"means": means, "clips": clips}))
