"""GSM8K scorer. Extraction rule copied from DSV4 `models/demos/blackhole/deepseek_v41_flash/reference/e2e_analyze.py` (`extract`, `norm`; branch
ssinghal/dsv4p1) as used by its `tools/score_gsm.py` / `wt/h43c/score_gsm_big.py`: final numeric answer of a completion = last `#### x`, else last `\\boxed{x}`,
else the last number in the text; commas stripped, trailing '.' dropped, integral floats printed as ints.

    python -m models.demos.blackhole.mimo_v26.reference.score_gsm8k <set.json> <outputs.json> [--min-correct K]

outputs.json: {"<idx>": "completion text", ...} or a list indexed by idx. Prints correct/N, Wilson 95% CI, unanswered count; exit 1 if --min-correct is not met.
"""
import argparse
import json
import math
import re


def norm(s):
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract(text):
    for pat in (r"####\s*\$?(-?[\d,]*\.?\d+)", r"\\boxed\{\s*\\?\$?\s*(-?[\d,]*\.?\d+)"):
        m = re.findall(pat, text)
        if m:
            return norm(m[-1])
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm(m[-1]) if m else None


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def score(items, outs):
    outs = {int(k): v for k, v in (outs.items() if isinstance(outs, dict) else enumerate(outs))}
    ok = [it["idx"] for it in items if it["idx"] in outs and extract(outs[it["idx"]]) == it["gold"]]
    return ok, len(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("outputs")
    ap.add_argument("--min-correct", type=int, default=None)
    a = ap.parse_args()
    d = json.load(open(a.set))
    ok, n_out = score(d["items"], json.load(open(a.outputs)))
    n = len(d["items"])
    lo, hi = wilson(len(ok), n)
    print(
        f"{len(ok)}/{n} correct ({100*len(ok)/n:.1f}%, Wilson95 {100*lo:.1f}-{100*hi:.1f}); outputs provided {n_out}; set sha256 {d['sha256']}"
    )
    if a.min_correct is not None and len(ok) < a.min_correct:
        raise SystemExit(1)
