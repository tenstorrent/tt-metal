#!/usr/bin/env python3
"""Scratch, not for merge. gate_cmp.py <results.pt> <arm> [<other results.pt>]: per token, check the arm's selection
against an exact reference (float64 keys = score + bias on the bf16 inputs, score = sigmoid(logit) or the raw input):
the selected keys must be the top-k keys as a multiset and the weights the normalization of the selected scores. With a
second file, count tokens whose ids and weights are identical, and classify every difference as tie-only (same selected
key multiset) or not. Prints "GATE ..." lines."""
import sys

import torch

a = torch.load(sys.argv[1], weights_only=True)
arm = sys.argv[2]
b = torch.load(sys.argv[3], weights_only=True) if len(sys.argv) > 3 else None


def ref(lg, bias, sig, topk):
    s = torch.sigmoid(lg.double()) if sig else lg.double()
    key = s + bias.double()
    return s, key


tot = valid = 0
same = tie = other = 0
maxd = 0.0
for name, (lg, bias, w, idx) in sorted(a.items()):
    sig, topk, smax = (int(x) for x in name.split("_")[-3:])
    s, key = ref(lg, bias, sig, topk)
    for t in range(idx.shape[0]):
        tot += 1
        sel = idx[t, :topk].long()
        ok = len(set(sel.tolist())) == topk and int(sel.max()) < 256
        if ok:
            got = torch.sort(key[t, sel]).values
            want = torch.sort(torch.topk(key[t], topk).values).values
            ok = bool(torch.allclose(got, want, atol=1e-2))
        valid += int(ok)
        if b is not None:
            w2, idx2 = b[name][2], b[name][3]
            if torch.equal(idx[t], idx2[t]) and torch.equal(w[t], w2[t]):
                same += 1
            else:
                k1 = torch.sort(key[t, idx[t, :topk].long()]).values
                k2 = torch.sort(key[t, idx2[t, :topk].long()]).values
                d = float((k1 - k2).abs().max())
                maxd = max(maxd, d)
                if d <= 1e-2:
                    tie += 1
                else:
                    other += 1
print(f"GATE VALID {arm} {valid} of {tot} tokens select a valid top-k by the exact key")
if b is not None:
    print(f"GATE CMP identical {same}, tie-only {tie}, other {other} of {tot} tokens; largest key gap between the two selections {maxd:.4g}")
