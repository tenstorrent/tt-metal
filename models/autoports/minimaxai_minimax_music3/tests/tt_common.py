"""Shared helpers for the device tests (stages 03-06)."""
import json
import os
from pathlib import Path

import torch


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() < 2:
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def thresholds():
    t = {"c0_top1": 0.90, "c0_top5": 0.98, "depth_top1": 0.90, "depth_top5": 0.98}
    p = os.environ.get("MUSIC3_FLOORS")
    if p and Path(p).exists():
        t.update(json.load(open(p)).get("thresholds", {}))
    return t


class Report(dict):
    def write(self):
        out = os.environ.get("MUSIC3_REPORT")
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(self, open(out, "w"), indent=2, default=str)


def golden(golden_root, clip):
    d = Path(golden_root) / clip
    g = torch.load(d / "golden.pt")
    tf = torch.load(d / "tf" / "teacher_forced.pt")
    return g, tf


def agreement(guided_fn, logits_seq, targets, topk=5):
    """logits_seq [N, 2, V], targets [N] -> (top1, top5) of the guided argmax against the golden codes."""
    n = logits_seq.shape[0]
    t1 = t5 = 0
    for i in range(n):
        g = guided_fn(logits_seq[i]).reshape(-1)
        top = torch.topk(g, topk).indices.tolist()
        t1 += int(top[0] == int(targets[i]))
        t5 += int(int(targets[i]) in top)
    return t1 / n, t5 / n


def cpu_targets(guided_fn, cpu_logits_seq):
    """CPU reference guided argmax per step: [N] (targets for the TT agreement bars)."""
    return torch.stack([guided_fn(cpu_logits_seq[i]).reshape(-1).argmax() for i in range(cpu_logits_seq.shape[0])])
