#!/usr/bin/env python3
"""score.py f1.jsonl [f2.jsonl ...] [--ref 89.2] [--boot 10000]
Files = repeats of one task. Reports per-repeat accuracy, mean over repeats, question-clustered bootstrap 95% CI
(resample questions with replacement; per-question score = mean over that question's repeats), truncation rate
(finish_reason == length), no-answer rate, mean completion tokens. Errored requests are excluded and counted."""
import argparse
import json
import random
import statistics


def load(p):
    d, errs = {}, 0
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get("error"):
                errs += 1
            else:
                d[r["record_id"]] = r
    return d, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", type=float, default=None, help="reference score in percent")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    reps = []
    nerr = 0
    for p in a.files:
        d, e = load(p)
        reps.append(d)
        nerr += e
    per = []
    for p, d in zip(a.files, reps):
        n = len(d)
        k = sum(r["correct"] for r in d.values())
        per.append(k / n if n else float("nan"))
        print(f"  {p}: acc={100 * per[-1]:.2f} ({k}/{n})")
    allrecs = [r for d in reps for r in d.values()]
    N = len(allrecs)
    mean = statistics.mean(per)
    byq = {}
    for d in reps:
        for qid, r in d.items():
            byq.setdefault(qid, []).append(1.0 if r["correct"] else 0.0)
    qscore = [sum(v) / len(v) for v in byq.values()]
    rng = random.Random(a.seed)
    nq = len(qscore)
    bs = sorted(statistics.fmean(qscore[rng.randrange(nq)] for _ in range(nq)) for _ in range(a.boot))
    lo, hi = bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]
    toks = [r["completion_tokens"] for r in allrecs if r.get("completion_tokens") is not None]
    print(f"repeats={len(reps)} questions={nq} responses={N} errors_excluded={nerr}")
    print(
        f"mean over repeats: {100 * mean:.2f}  | per-repeat: {', '.join(f'{100 * x:.2f}' for x in per)}"
        f"  | pooled-question bootstrap 95% CI [{100 * lo:.2f}, {100 * hi:.2f}] (clustered by question, B={a.boot})"
    )
    if a.ref is not None:
        print(
            f"reference {a.ref:.2f}: delta(mean-ref)={100 * mean - a.ref:+.2f}; ref inside CI: {100 * lo <= a.ref <= 100 * hi}"
        )
    print(
        f"truncation rate (finish_reason=length): {100 * sum(r.get('finish_reason') == 'length' for r in allrecs) / max(N, 1):.2f}% "
        f"({sum(r.get('finish_reason') == 'length' for r in allrecs)}/{N})"
    )
    print(
        f"no-answer rate: {100 * sum(r['extracted'] is None for r in allrecs) / max(N, 1):.2f}% "
        f"({sum(r['extracted'] is None for r in allrecs)}/{N})"
    )
    print(
        f"mean completion tokens: {statistics.mean(toks) if toks else float('nan'):.1f}  max: {max(toks) if toks else None}"
    )


if __name__ == "__main__":
    main()
