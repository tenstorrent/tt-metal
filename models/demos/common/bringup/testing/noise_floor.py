# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Accuracy limits from the noise floor, never a fixed bar (F76).

Owner rule (DeepSeek-V4.1 run1): no fixed accuracy thresholds such as PCC 0.99 or a 1 % cap; every limit derives from
the noise floor of the reference's dtypes on the CPU op. A fixed bar either lets real errors through or, when the
reference's own dtypes cannot meet it, makes agents raise the device precision above the reference (exact paths, fp32
where the reference uses bf16) to clear it.

F80 (the precision brainstorm, owner-approved): the target is the fp32 CPU step; its noise floor, per statistic, is the
larger of
- the reference arithmetic: the official code at its own dtypes on the same inputs vs the fp32 step (tapped per step
  from the official model and cached with the goldens: official_floor.json; testing/official_floor.py), and
- the input nudge: the fp32 step on its inputs moved uniformly inside their rounding cell, in the dtype the official
  code holds each input (official_taps.yaml; default bf16; quantizer outputs `fixed`), SEEDS seeds, the worst.
A limit is calib (component_calib; component_calib_low for tests.low_precision_kinds) x that floor on the same
comparison (vs the CPU step on the same inputs, or vs the golden), never below what one bf16 rounding of the output
shows (an exact step) or one element of an integer output. For a correlation the miss is 1 - pcc ~ rel^2 / 2, so calib
applies squared. Selections (top-k): every flip must be explained (both swapped entries within 2x the nudge's spread
of the top-k cutoff), and the count is at most calib x the nudge's.

A gate metric written ``>= limit`` in tasks.yaml is compared with the ``<metric>_limit`` the test records
(core/gate.check_metrics).
"""

from __future__ import annotations

import json

import torch
from torch.overrides import TorchFunctionMode

from models.demos.common.bringup.core import defaults, metrics

SEEDS = int(defaults.get("noise_floor.seeds"))  # nudged runs per floor; the worst counts
FLIP_SPREAD = float(defaults.get("noise_floor.flip_spread"))  # a flip is explained within this x the nudge spread
MANT = {"bf16": 7, "fp32": 23}
TAPS_FILE = "official_taps.yaml"  # the reference role's map of the official code onto the framework's steps
FLOOR_FILE = "official_floor.json"  # next to the goldens: the reference arithmetic's statistics per step


def value(mode: str, got: torch.Tensor, want: torch.Tensor) -> float:
    """The comparison of harness.compare, without recording: pcc, match (equal elements), topk_overlap (per-row set
    overlap, order-free)."""
    if got.numel() != want.numel():
        return 0.0
    got = got.reshape(want.shape)
    if mode == "pcc":
        return metrics.pcc(got.float(), want.float())
    if mode == "match":
        return (got == want).float().mean().item()
    if mode == "topk_overlap":
        w, g = want.reshape(-1, want.shape[-1]), got.reshape(-1, got.shape[-1])
        return (
            torch.tensor([len(set(a.tolist()) & set(b.tolist())) / max(len(set(b.tolist())), 1) for a, b in zip(g, w)])
            .mean()
            .item()
        )
    raise ValueError(f"compare mode {mode!r}")


def gate_limit(mode: str, model_v: float, res_v: float, k: float, n: int) -> float:
    """The lower limit of a gate metric: 1 - calib x the miss of the precision model (model_v) or of one bf16 rounding
    of the output (res_v), whichever is larger; calib squared for pcc; an integer comparison never asks for fewer
    misses than one element of n."""
    miss = max(1.0 - model_v, 1.0 - res_v, 0.0 if mode == "pcc" else 1.0 / max(n, 1))
    return max(0.0, 1.0 - (k * k if mode == "pcc" else k) * miss)


def round_to(t: torch.Tensor, dtype: str = "bf16") -> torch.Tensor:
    """t as the official code hands it on: rounded to bf16, or kept (fp32)."""
    t = t.float()
    return t.to(torch.bfloat16).float() if dtype == "bf16" and t.is_floating_point() else t


def bf16_res(mode: str, ref: torch.Tensor, dtype: str = "bf16") -> float:
    """The metric of ``ref`` rounded to the output dtype vs ``ref``: the closest a correct output can get (1 for
    integers and for an fp32 output)."""
    if not ref.is_floating_point():
        return 1.0
    return value(mode, round_to(ref, dtype), ref.float())


# ---------------------------------------------------------------- the input nudge (F80)
def nudge(t, seed: int, dtype: str = "bf16"):
    """t moved uniformly inside its rounding cell in ``dtype`` (bf16 / fp32); `fixed` (a quantizer's output) and
    integer tensors stay as they are."""
    if not isinstance(t, torch.Tensor) or not t.is_floating_point() or dtype == "fixed":
        return t
    g = torch.Generator().manual_seed(seed)
    x = t.double()
    cell = torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-38))) - MANT[dtype])
    return (x + (torch.rand(t.shape, generator=g, dtype=torch.float64) - 0.5) * cell).to(t.dtype)


def input_dtypes(s, block_type: str | None, step: str, n: int) -> list[str]:
    """The dtype the official code holds each of a step's n inputs in (official_taps.yaml: steps.<step> or
    <block type>.<step>, inputs: [bf16 | fp32 | fixed, ...]); bf16 when not mapped."""
    import yaml

    p = s.bringup_dir / TAPS_FILE
    steps = ((yaml.safe_load(p.read_text()) if p.exists() else None) or {}).get("steps") or {}
    e = steps.get(f"{block_type}.{step}") or steps.get(step) or {}
    got = list(e.get("inputs") or [])
    bad = [d for d in got if d not in ("bf16", "fp32", "fixed")]
    if bad:
        raise ValueError(f"{p}: {step}: input dtypes {bad} not in bf16 / fp32 / fixed")
    return (got + ["bf16"] * n)[:n]


def explanation(s, block_type: str | None, step: str) -> str:
    """official_taps.yaml `explained`: why the official arithmetic of a step exceeds calib x its input nudge (its own
    internal roundings, with the evidence, e.g. the fp32 step with them emulated reproduces the official output). It
    lets R.5's tap-map check pass that step; the floor then is the official arithmetic, so the overseer reviews it."""
    import yaml

    p = s.bringup_dir / TAPS_FILE
    steps = ((yaml.safe_load(p.read_text()) if p.exists() else None) or {}).get("steps") or {}
    return str((steps.get(f"{block_type}.{step}") or steps.get(step) or {}).get("explained") or "").strip()


def output_dtype(s, block_type: str | None, step: str) -> str:
    """The dtype the official code hands a step's output on in (official_taps.yaml `out`; bf16 when not mapped): the
    resolution floor of the vs-CPU checks is one rounding to it (none for fp32)."""
    import yaml

    p = s.bringup_dir / TAPS_FILE
    steps = ((yaml.safe_load(p.read_text()) if p.exists() else None) or {}).get("steps") or {}
    d = (steps.get(f"{block_type}.{step}") or steps.get(step) or {}).get("out", "bf16")
    if d not in ("bf16", "fp32"):
        raise ValueError(f"{p}: {step}: out {d!r} not bf16 / fp32")
    return d


def nudged(run, inputs, dtypes, seeds: int = SEEDS) -> list:
    """run(*inputs) on the inputs nudged with each seed (the floor's outputs)."""
    return [run(*[nudge(t, 1000 * k + j, d) for j, (t, d) in enumerate(zip(inputs, dtypes))]) for k in range(seeds)]


def worst_float(outs, want, errors) -> dict:
    """The worst of ``errors(out, want)`` over the floor's outputs: largest rel / row / ratio spread / |bias|."""
    es = [errors(o, want) for o in outs]
    return {
        "rel": max(e["rel"] for e in es),
        "row": max(e["row"] for e in es),
        "ratio_min": min(e["ratio_min"] for e in es),
        "ratio_max": max(e["ratio_max"] for e in es),
        "bias": max((e["bias"] for e in es), key=abs),
    }


def merge_float(a: dict | None, b: dict) -> dict:
    """The larger floor per statistic (a: the reference arithmetic, cached; b: the nudge)."""
    if not a:
        return b
    return {
        "rel": max(a.get("rel", 0.0), b["rel"]),
        "row": max(a.get("row", 0.0), b["row"]),
        "ratio_min": min(a.get("ratio_min", 1.0), b["ratio_min"]),
        "ratio_max": max(a.get("ratio_max", 1.0), b["ratio_max"]),
        "bias": max((a.get("bias", 0.0), b["bias"]), key=abs),
    }


# ---------------------------------------------------------------- the reference arithmetic (cached, F80)
def arith(s, layer: int, step: str, chunk: int) -> dict | None:
    """The official code's statistics for one step (vs the fp32 step on the official inputs; vs the golden), from
    official_floor.json next to the component goldens; None when not generated (the nudge alone is the floor)."""
    from models.demos.common.bringup.testing.harness import component_golden

    g, _ = component_golden(s)
    p = g.dir / FLOOR_FILE
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return ((d.get("steps") or {}).get(f"L{layer:02d}.{step}") or {}).get(str(chunk))


# ---------------------------------------------------------------- selections: explained flips (F80)
class TopkTap(TorchFunctionMode):
    """Records the input and k of the last top-k a CPU step takes (its selection scores)."""

    def __init__(self):
        super().__init__()
        self.scores, self.k = None, None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.topk, torch.Tensor.topk) and args and isinstance(args[0], torch.Tensor):
            self.scores = args[0].detach().double().reshape(-1, args[0].shape[-1])
            self.k = int(args[1] if len(args) > 1 else kwargs.get("k"))
        return func(*args, **kwargs)


def tapped(run, inputs):
    """(output, scores, k) of run(*inputs); scores None when the step takes no top-k."""
    with TopkTap() as tap:
        out = run(*inputs)
    return out, tap.scores, tap.k


def support(t: torch.Tensor, k: int) -> torch.Tensor:
    """The selected entries of a selection output (dense, zeros elsewhere) or of scores (their top k)."""
    t = t.double().reshape(-1, t.shape[-1])
    if (t == 0).float().mean().item() >= 0.5:
        return t != 0
    return torch.zeros_like(t, dtype=torch.bool).scatter_(1, t.topk(k, dim=-1)[1], True)


def flips(scores, k: int, got_support, nudge_scores: list) -> dict:
    """Rows whose selection differs from the exact scores' top k, and how many of them are unexplained: a differing
    entry farther than FLIP_SPREAD x the nudge's spread (per row) from the cutoff between the k-th and k+1-th score."""
    exact = support(scores, k)
    srt = scores.sort(-1, descending=True).values
    bound = 0.5 * (srt[:, k - 1] + srt[:, k]) if scores.shape[-1] > k else srt[:, -1]
    delta = torch.stack([(n - scores).abs().amax(-1) for n in nudge_scores]).amax(0) if nudge_scores else 0 * bound
    diff = exact != got_support.reshape(exact.shape)
    far = ((scores - bound[:, None]).abs() > FLIP_SPREAD * delta[:, None]) & diff
    nrows = max(exact.shape[0], 1)
    return {
        "rows": int(diff.any(-1).sum()),
        "unexplained": int(far.any(-1).sum()),
        "frac": diff.any(-1).sum().item() / nrows,
    }
