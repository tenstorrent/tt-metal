# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DistillMOS over the utterances of one or more quality-set runs, for `quality_report.py`.

MUST RUN FROM /tmp/mosvenv, never the main venv -- DistillMOS depends on torchaudio, which
STATUS.md §2 records as ABI-broken here AND as breaking `transformers` merely by being importable,
which takes the WER scorer down with it. `mos_setup.sh` builds the isolated venv.

    /tmp/mosvenv/bin/python mos_batch.py <tag> [<tag> ...]

Reads `generated/results<tag>.json` for the file list, so it scores exactly the utterances the run
produced. Prints machine-readable MOS_MEAN / MOS_LONGFORM / MOS_MIN lines for the report to parse,
plus a per-utterance table for a human.

READ THE NUMBERS AS A PAIRED DELTA, NOT AN ABSOLUTE (§6.59). A no-reference predictor's scale is
not calibrated for this domain -- short and adversarial prompts score 2.5-3.0 here for reasons
that have nothing to do with the port, and the same two cases are lowest in every arm. What the
metric resolves is device-vs-device and device-vs-fp32-reference on the SAME prompts.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

import distillmos

V = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(V, "generated")


def main():
    tags = sys.argv[1:]
    if not tags:
        raise SystemExit("usage: mos_batch.py <tag> [<tag> ...]")
    model = distillmos.ConvTransformerSQAModel()
    model.eval()

    def score(path):
        x, sr = sf.read(path, dtype="float32")
        x = torch.from_numpy(np.asarray(x)).reshape(1, -1)
        if sr != 16000:                       # DistillMOS expects 16 kHz
            x = torchaudio.functional.resample(x, sr, 16000)
        with torch.no_grad():
            return float(model(x).item())

    rows = []
    for t in tags:
        p = os.path.join(GEN, f"results{t}.json")
        if not os.path.exists(p):
            print(f"  (no results{t}.json)")
            continue
        for r in json.load(open(p)):
            w = r["wav"] if os.path.isabs(r["wav"]) else os.path.join(GEN, r["wav"])
            if os.path.exists(w):
                rows.append((score(w), r, t))

    if not rows:
        raise SystemExit("  no wavs found")
    print(f"  {'tag':<12} {'case':>4} {'voice':<16} {'words':>5} {'MOS':>6}")
    for m, r, t in rows:
        print(f"  {t:<12} {r['case']:>4} {r['voice']:<16} "
              f"{len(r['text'].split()):>5} {m:>6.3f}")
    allv = [m for m, _, _ in rows]
    # long form is the bucket the WER gate uses, and the one a listener actually hears
    lf = [m for m, r, _ in rows if len(r["text"].split()) >= 20]
    print(f"\nMOS_MEAN {np.mean(allv):.4f}")
    print(f"MOS_LONGFORM {np.mean(lf) if lf else float('nan'):.4f}")
    print(f"MOS_MIN {min(allv):.4f}")
    worst = sorted(rows)[:3]
    print("  worst three: " + ", ".join(
        f"case{r['case']} {r['voice']} {m:.2f}" for m, r, _ in worst))


if __name__ == "__main__":
    main()
