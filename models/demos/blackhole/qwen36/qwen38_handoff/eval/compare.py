#!/usr/bin/env python3
"""Usage: compare.py A.jsonl B.jsonl"""
import json
import sys
from math import comb, sqrt


def load(p):
    d = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                d[r["record_id"]] = r
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    for name, d in (("A", A), ("B", B)):
        k = sum(r["correct"] for r in d.values())
        n = len(d)
        lo, hi = wilson(k, n)
        toks = [r["completion_tokens"] for r in d.values() if r.get("completion_tokens") is not None]
        mt = sum(toks) / len(toks) if toks else float("nan")
        tr = sum(r.get("finish_reason") == "length" for r in d.values())
        print(
            f"{name}: acc={k / max(n, 1):.4f} ({k}/{n}) 95% Wilson CI [{lo:.4f}, {hi:.4f}] mean_tokens={mt:.1f} truncated(length)={tr}"
        )
    common = sorted(set(A) & set(B))
    both = sum(A[i]["correct"] and B[i]["correct"] for i in common)
    ao = sum(A[i]["correct"] and not B[i]["correct"] for i in common)
    bo = sum(B[i]["correct"] and not A[i]["correct"] for i in common)
    neither = len(common) - both - ao - bo
    print(f"Paired on {len(common)} common ids: both right={both}, A only={ao}, B only={bo}, both wrong={neither}")
    print(f"McNemar exact two-sided p = {mcnemar_exact(ao, bo):.4g}")


if __name__ == "__main__":
    main()
