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
  code holds each input (official_taps.yaml; default bf16; an fp4 quantizer's output `fp4/<block>`, F223; other
  quantizer outputs `fixed`), SEEDS seeds, the worst.
A limit is calib (component_calib) x that floor on the same
comparison (vs the CPU step on the same inputs, or vs the golden), never below what one rounding of the output to
its official dtype shows (F133: bf16, or fp32 for an fp32 output) (an exact step) or one element of an integer output. For a correlation the miss is 1 - pcc ~ rel^2 / 2, so calib
applies squared. Selections (top-k): every flip must be explained (both swapped entries within 2x the nudge's spread
of the top-k cutoff, or within tie_rtol of it), and the count is at most calib x the nudge's; the tie-aware recall
(F101) is recorded for every top-k output, dense router weights and index lists alike.

A gate metric written ``>= limit`` in tasks.yaml is compared with the ``<metric>_limit`` the test records
(core/gate.check_metrics).
"""

from __future__ import annotations

import json
import math

import torch
from torch.overrides import TorchFunctionMode

from models.demos.common.bringup.core import defaults, metrics

SEEDS = int(defaults.get("noise_floor.seeds"))  # nudged runs per floor; the worst counts
SWAP_SEEDS = int(defaults.get("noise_floor.swap_seeds"))  # a swap test's earlier steps and its nudged block
FLIP_SPREAD = float(defaults.get("noise_floor.flip_spread"))  # a flip is explained within this x the nudge spread
TIE_RTOL = float(defaults.get("noise_floor.tie_rtol"))  # ... or within this x |cutoff| (F101)
MANT = {"bf16": 7, "fp32": 23}
PAD_SENTINEL = 0xFFFFFFFF  # the uint32 pad of topk_large_indices
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
    """The lower limit of a gate metric: 1 - calib x the miss of the precision model (model_v) or of one rounding
    of the output to its dtype (res_v), whichever is larger; calib squared for pcc; an integer comparison never asks for fewer
    misses than one element of n."""
    miss = max(1.0 - model_v, 1.0 - res_v, 0.0 if mode == "pcc" else 1.0 / max(n, 1))
    return max(0.0, 1.0 - (k * k if mode == "pcc" else k) * miss)


def round_to(t: torch.Tensor, dtype: str = "bf16") -> torch.Tensor:
    """One rounding of t to the dtype the official code hands it on in: bf16, or fp32 (F133: t moved uniformly inside
    its fp32 rounding cell, returned in float64 so the move survives; an fp32 output's floor is ~1e-7, not 0 or bf16).
    The resolution floor of every limit: no output in that dtype can do better."""
    if not t.is_floating_point():
        return t
    if dtype == "bf16":
        return t.float().to(torch.bfloat16).float()
    g = torch.Generator().manual_seed(0)
    x = t.double()
    cell = torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-38))) - MANT[dtype])
    return x + (torch.rand(t.shape, generator=g, dtype=torch.float64) - 0.5) * cell


def bf16_res(mode: str, ref: torch.Tensor, dtype: str = "bf16") -> float:
    """The metric of ``ref`` rounded to the output dtype vs ``ref``: the closest a correct output can get (1 for
    integers and for an fp32 output)."""
    if not ref.is_floating_point():
        return 1.0
    return value(mode, round_to(ref, dtype), ref.float())


# ---------------------------------------------------------------- the input nudge (F80)
FP4_GRID_STEP = ((2.0, 0.5), (4.0, 1.0), (math.inf, 2.0))  # e2m1 magnitudes 0..6: the step below each bound


def _fp4_block(dtype: str) -> int | None:
    """``fp4/<block>``: an fp4 (e2m1) quantizer output with a scale per <block> values of the last dim; else None."""
    if not dtype.startswith("fp4/"):
        return None
    b = dtype[4:]
    if not b.isdigit() or int(b) < 1:
        raise ValueError(f"input dtype {dtype!r}: fp4/<block>, e.g. fp4/16")
    return int(b)


def nudge(t, seed: int, dtype: str = "bf16"):
    """t moved uniformly inside its rounding cell in ``dtype`` (bf16 / fp32, or fp4/<block>: the e2m1 grid step at
    |t| / s, s = the block's amax / 6, the official fp4 quantizer's rounding, F223); `fixed` (a quantizer's output whose
    rounding the step does not repeat) and integer tensors stay as they are."""
    if not isinstance(t, torch.Tensor) or not t.is_floating_point() or dtype == "fixed":
        return t
    g = torch.Generator().manual_seed(seed)
    x = t.double()
    block = _fp4_block(dtype)
    if block is not None:
        if x.shape[-1] % block:
            raise ValueError(f"input dtype {dtype}: last dim {x.shape[-1]} is not a multiple of {block}")
        xg = x.reshape(*x.shape[:-1], -1, block)
        s = (xg.abs().amax(-1, keepdim=True) / 6.0).clamp_min(1e-38)
        m = xg.abs() / s
        step = torch.full_like(m, FP4_GRID_STEP[-1][1])
        for bound, st in reversed(FP4_GRID_STEP[:-1]):
            step = torch.where(m < bound, torch.full_like(m, st), step)
        cell = (step * s).reshape(x.shape)
    else:
        cell = torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-38))) - MANT[dtype])
    return (x + (torch.rand(t.shape, generator=g, dtype=torch.float64) - 0.5) * cell).to(t.dtype)


def input_dtypes(s, block_type: str | None, step: str, n: int) -> list[str]:
    """The dtype the official code holds each of a step's n inputs in (official_taps.yaml: steps.<step> or
    <block type>.<step>, inputs: [bf16 | fp32 | fp4/<block> | fixed, ...]); bf16 when not mapped."""
    import yaml

    p = s.bringup_dir / TAPS_FILE
    steps = ((yaml.safe_load(p.read_text()) if p.exists() else None) or {}).get("steps") or {}
    e = steps.get(f"{block_type}.{step}") or steps.get(step) or {}
    got = list(e.get("inputs") or [])
    bad = [d for d in got if d not in ("bf16", "fp32", "fixed") and not str(d).startswith("fp4/")]
    if bad:
        raise ValueError(f"{p}: {step}: input dtypes {bad} not in bf16 / fp32 / fp4/<block> / fixed")
    for d in got:
        if str(d).startswith("fp4/"):
            _fp4_block(d)
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


# ---------------------------------------------------------------- selections: explained flips (F80, F101)
class TopkTap(TorchFunctionMode):
    """Records the input and k of every top-k a CPU step takes (its selection scores), in call order."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.topk, torch.Tensor.topk) and args and isinstance(args[0], torch.Tensor):
            k = int(args[1] if len(args) > 1 else kwargs.get("k"))
            self.calls.append((args[0].detach().double().reshape(-1, args[0].shape[-1]), k))
        return func(*args, **kwargs)


def _groups(calls) -> list:
    """The tapped top-ks by (width, k), their row blocks stacked in call order (a step that selects per row block)."""
    by = {}
    for sc, k in calls:
        by.setdefault((sc.shape[-1], k), []).append(sc)
    return [(torch.cat(b), k) for (_, k), b in by.items()]


def match_scores(calls, out):
    """(scores, k) the step's output selects from: the tapped top-k whose rows are the output's and whose own top k
    holds (>= 99 % of) the output's selection; (None, None) when none does (F101: a step taking several top-ks, e.g.
    candidate blocks then positions, or one per row block)."""
    if not isinstance(out, torch.Tensor) or out.dim() < 2 or not calls:
        return None, None
    rows = out.numel() // out.shape[-1]
    best = (0.0, None, None)
    for sc, k in _groups(calls):
        if sc.shape[0] != rows or (out.is_floating_point() and sc.shape[-1] != out.shape[-1]):
            continue
        got = support(out.reshape(rows, -1), k, sc.shape[-1])
        n = got.sum() if out.is_floating_point() else _valid(out).sum()  # an id beyond the width is a miss
        hit = ((got & support(sc, k)).sum() / n.clamp_min(1)).item()
        if hit > best[0]:
            best = (hit, sc, k)
    return (best[1], best[2]) if best[0] >= 0.99 else (None, None)


def _valid(i: torch.Tensor) -> torch.Tensor:
    """The valid entries of an index output (pads: negative, or the uint32 sentinel of topk_large_indices)."""
    i = i.to(torch.int64)
    return (i >= 0) & (i < PAD_SENTINEL)


def tapped(run, inputs):
    """(output, scores, k) of run(*inputs); scores None when no top-k the step takes is the one its output holds."""
    with TopkTap() as tap:
        out = run(*inputs)
    return (out, *match_scores(tap.calls, out))


def support(t: torch.Tensor, k: int, width: int | None = None) -> torch.Tensor:
    """The selected entries [rows, width] of a selection output (dense, zeros elsewhere), of an index output (its
    valid positions: pads and out-of-range ids dropped) or of scores (their top k)."""
    if not t.is_floating_point():
        i = t.to(torch.int64).reshape(-1, t.shape[-1])
        width = width or int(i.max().item()) + 1
        ok = (i >= 0) & (i < width)
        out = torch.zeros(i.shape[0], width + 1, dtype=torch.bool)
        return out.scatter_(1, torch.where(ok, i, width), True)[:, :width]
    t = t.double().reshape(-1, t.shape[-1])
    if (t == 0).float().mean().item() >= 0.5:
        return t != 0
    return torch.zeros_like(t, dtype=torch.bool).scatter_(1, t.topk(k, dim=-1)[1], True)


def _spread(nudge_scores: list, scores) -> torch.Tensor:
    """Per row, the largest move of a finite score under the nudge (masked -inf entries stay -inf)."""
    if not nudge_scores:
        return torch.zeros(scores.shape[0], dtype=scores.dtype)
    d = [torch.where(torch.isfinite(n) & torch.isfinite(scores), n - scores, 0.0).abs().amax(-1) for n in nudge_scores]
    return torch.stack(d).amax(0)


def flips(scores, k: int, got_support, nudge_scores: list, want_support=None, rtol: float | None = None) -> dict:
    """Rows whose selection differs from the reference's, how many of them are unexplained, and the tie-aware recall.

    The reference's selection is ``want_support`` (its own tie rule, its masked entries), else the top k of the exact
    scores. The cutoff of a row lies between its weakest selected and its strongest unselected score.
    - unexplained (F80, gated): a differing entry whose exact score lies farther from the cutoff than both
      FLIP_SPREAD x the nudge's score spread and TIE_RTOL x |cutoff|.
    - recall_tie (F101, recorded): Momcilo's MoE-gate rule, the nudge left out (on a step with quantized
      intermediates, e.g. an fp4 query, the nudge's spread covers most of the row): the reference's entries the device
      holds plus its swaps whose both scores lie within TIE_RTOL x |cutoff| of the cutoff, over the reference's
      entries (mean over rows; 1 for an empty row)."""
    rtol = TIE_RTOL if rtol is None else rtol
    exact = want_support if want_support is not None else support(scores, k)
    got = got_support.reshape(exact.shape)
    ninf = torch.full_like(scores, -math.inf)
    lo = torch.where(exact, scores, -ninf).amin(-1)  # the weakest selected score
    hi = torch.where(exact | ~torch.isfinite(scores), ninf, scores).amax(-1)  # the strongest unselected finite one
    bound = torch.where(torch.isfinite(hi), 0.5 * (lo + hi), lo)
    dist = (scores - bound[:, None]).abs()
    tie = (dist <= (rtol * bound.abs())[:, None]) & torch.isfinite(scores)
    near = tie | ((dist <= (FLIP_SPREAD * _spread(nudge_scores, scores))[:, None]) & torch.isfinite(scores))
    missed, extra = exact & ~got, got & ~exact
    diff = missed | extra
    credited = torch.minimum((missed & tie).sum(-1), (extra & tie).sum(-1))
    n = exact.sum(-1)
    recall = torch.where(n > 0, (exact & got).sum(-1).double() / n.clamp_min(1), 1.0)
    recall_tie = torch.where(n > 0, ((exact & got).sum(-1) + credited).double() / n.clamp_min(1), 1.0)
    nrows = max(exact.shape[0], 1)
    return {
        "rows": int(diff.any(-1).sum()),
        "unexplained": int((diff & ~near).any(-1).sum()),
        "frac": diff.any(-1).sum().item() / nrows,
        "recall": recall.mean().item(),
        "recall_tie": recall_tie.mean().item(),
        "worst_recall_tie": recall_tie.min().item() if recall_tie.numel() else 1.0,
    }


# ---------------------------------------------------------------- each row and column against its own floor (F83)
def _r(t: torch.Tensor, shape) -> torch.Tensor:
    return t.double().reshape(shape[0], -1) if len(shape) >= 1 else t.double().reshape(1, -1)


def _c(t: torch.Tensor, shape) -> torch.Tensor:
    return t.double().reshape(-1, shape[-1]) if len(shape) >= 2 else t.double().reshape(-1, 1)


def own_floor(got, want, outs, res_dtype: str = "bf16", scale: float = 1.0) -> dict:
    """F83: every row and every column of ``got`` against its own noise floor, and the signed additive bias.

    Per row r (and per column c, over the rows): ||got_r - want_r|| / max(the floor's ||out_r - want_r|| (the worst
    nudged run), one rounding of want_r to ``res_dtype``) x 1 / scale: no division by a tiny norm and no fixed floor;
    a row that is exact on the device and in the floor counts 0. Additive bias per column: |mean over the rows of
    (got - want)| over the larger of the floor's own |mean|, the largest random mean C columns of the floor's rms give
    (sqrt(2 ln 2C) x rms / sqrt(rows)) and the same for one rounding. The worst row (column) is allowed the expected
    excess of the worst of N (1 + sqrt(2 ln 2N / m), m its length) over a typical one.
    ``scale`` >= 1 widens every floor by the official arithmetic's excess over the nudge (a whole-output ratio).
    Returns the worst ratios (gated <= calib), where they are, and the worst column's signed bias over its rms."""
    w = want.double()
    shape = w.shape
    g = got.double().reshape(shape)
    res = round_to(w, res_dtype).double() if w.is_floating_point() else w
    out = {}
    for name, f, dim in (("row", _r, 1), ("col", _c, 0)):
        if name == "col" and len(shape) < 2:
            continue
        e_dev = (f(g, shape) - f(w, shape)).norm(dim=dim)
        e_fl = torch.stack([(f(o.double().reshape(shape), shape) - f(w, shape)).norm(dim=dim) for o in outs]).amax(0)
        e_res = (f(res, shape) - f(w, shape)).norm(dim=dim)
        # the worst of N rows (columns) of m elements sits ~ 1 + sqrt(2 ln 2N / m) above a typical one (chi statistics)
        n_, m_ = e_dev.numel(), max(f(w, shape).shape[dim], 1)
        denom = scale * torch.maximum(e_fl, e_res) * (1 + math.sqrt(2 * math.log(2 * n_) / m_))
        ratio = torch.where(
            e_dev == 0,
            torch.zeros_like(e_dev),
            torch.where(denom > 0, e_dev / denom.clamp_min(1e-300), torch.full_like(e_dev, math.inf)),
        )
        j = int(ratio.argmax().item())
        out[f"{name}_floor"], out[f"{name}_at"] = float(ratio[j].item()), j
    if len(shape) >= 2:
        n = _c(w, shape).shape[0]
        d_dev = (_c(g, shape) - _c(w, shape)).mean(0)
        d_fl = [(_c(o.double().reshape(shape), shape) - _c(w, shape)) for o in outs]
        # a random error's column mean is ~ rms / sqrt(rows); the largest of C such means ~ sqrt(2 ln 2C) of that
        spread = math.sqrt(2 * math.log(2 * d_dev.numel())) / math.sqrt(n)
        lim = torch.stack([x.mean(0).abs() for x in d_fl] + [x.pow(2).mean(0).sqrt() * spread for x in d_fl]).amax(0)
        r_res = _c(res, shape) - _c(w, shape)  # one rounding: its mean, and the random mean its rms allows
        lim = scale * torch.maximum(lim, torch.maximum(r_res.mean(0).abs(), r_res.pow(2).mean(0).sqrt() * spread))
        ratio = torch.where(
            d_dev == 0,
            torch.zeros_like(d_dev),
            torch.where(lim > 0, d_dev.abs() / lim.clamp_min(1e-300), torch.full_like(d_dev, math.inf)),
        )
        j = int(ratio.argmax().item())
        rms = _c(w, shape)[:, j].pow(2).mean().sqrt().item()
        out.update(
            bias_floor=float(ratio[j].item()), bias_at=j, bias_signed=float(d_dev[j].item() / rms) if rms > 0 else 0.0
        )
    return out


# ---------------------------------------------------------------- owner waivers (F85)
def waiver(s, block_type: str | None, step: str) -> str:
    """The owner's waiver of a step's noise-floor checks (spec tests.waivers: {<step> | <block type>.<step>: reason}),
    e.g. a gap the device's arithmetic cannot close at its normal settings, to be fixed later. '' when none."""
    w = s.get("tests.waivers") or {}
    return str(w.get(f"{block_type}.{step}") or w.get(step) or "").strip()


def waive(s, block_type, step: str, ok: bool, out, tag: str, ref=None) -> bool:
    """A waived step passes its noise-floor checks when its output is whole (finite, the right size) and, given the
    reference ``ref``, within the structural limit (component_scaled_rel, F97: a waiver covers a precision gap, never
    a wrong module such as a zero stub); the failure is printed and recorded (<tag>_waived = 1), never hidden."""
    why = waiver(s, block_type, step)
    if ok or not why:
        return ok
    whole = isinstance(out, torch.Tensor) and (not out.is_floating_point() or bool(torch.isfinite(out.float()).all()))
    note = "" if whole else " (but the output is not whole: FAIL)"
    if whole and isinstance(ref, torch.Tensor) and out.is_floating_point() and ref.is_floating_point():
        from models.demos.common.bringup.testing.component_checks import limits

        lim = limits(s)["struct_scaled_rel"]  # a structural bound (F97), not an accuracy limit (F138)
        r = ((out.float().reshape(ref.shape) - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
        if r > lim:
            whole, note = False, f" (but rel {r:.3g} > {lim} vs the reference, a structural error: FAIL)"
    metrics.record(f"{tag}_waived", 1)
    print(f"WAIVED {tag}: the noise-floor checks above failed; owner waiver: {why}{note}")
    return whole
