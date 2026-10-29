# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Built-in checks of a component test (F56): ``run_component_test(..., checks="auto")``.

The rendered component template had one gate, PCC (or exact match) vs the golden, and a test-role agent then added
checks by hand for every step (Hy4: 42 reviews, 4 h). The checks it added are the same few each time; these do them
generically, chosen by the kind of the step's output:

    float       vs the CPU step on the same inputs and vs the golden: finite, rel L2, every row and every column
                against its own floor (F83, warn only), every row's norm ratio, the median row norm ratio (a
                systematic scale). Limits: calib x the noise floor on the same comparison (testing/noise_floor.py:
                the larger of the official arithmetic and the input nudge), never below one rounding of the output to
                its official dtype (F133); the gated error vs the golden is rel L2 (err_<step>_L<i>, F81).
    selection   a float output with >= 75 % exact zeros (dense router weights): the same nonzero count per row, no
                negative weight when the reference has none, mean selection overlap, rel L2 on the rows with the same
                selection, their row sums; each limit calib x the noise floor's (every flip explained, F80).
    index       an integer [rows, k] output whose rows are sets (top-k positions, pads -1 or 0xFFFFFFFF): order-free
                overlap (mean and worst row, calib x the noise floor's miss, never below one entry), the same valid
                count per row, no repeats, and what the reference obeys: no position after its row (causal), every row
                selecting its own position, pads only after the valid positions. The golden gate uses
                ``topk_overlap``, never positional match.
    int         anything else: exact match as before; an unknown kind, so the freeze sweep sends it to a review.

Second inputs, each compared with the CPU step on the same inputs, for structural bugs the golden cannot show (the
noise-floor limits x struct_probe, at least the struct_second_* floors, no bias limit; F138: structural, never
accuracy limits):
    chunk0      stateful steps: the golden chunk 0 (start 0, empty prefix): short rows, the causal edge
    layer       the same step's inputs from the farthest other golden layer (distinct iHC streams where layer 0's
                are equal)
    mixed       float and selection outputs: every float input with its rows and columns permuted (seeded, per
                input): the real value distribution without its symmetries (layer 0's equal iHC streams, pre gates
                all near 1: a stream-order bug moves nothing on real inputs)
    small       float outputs: every float input scaled by a power of 2 to RMS ~1e-3, where a norm epsilon matters
    big         float outputs: every float input x 2, where a clamp (SwiGLU limit) matters
A hook ``swap_context(spec, ref, layer, golden, chunk, rctx, dctx)`` (F49), if the model has one, fills both contexts
of every input (e.g. a shared layer's source top-k).

Freeze sweep (``BRINGUP_IMPL=mutations``, CPU only, one process, the test must PASS): the reference passes; every
standard mistake of testing/mutate.py that applies to the output (applied on the golden input and on every second
input) fails; the bf16 controls (inputs and output rounded; the precision model itself) pass. It also reports, for
each float input scaled by 1.02, whether the checks see it (a scale-free step, e.g. a norm, does not move). An unknown
output kind or a miss fails the sweep: the orchestrator then starts the test-role review with the sweep's log.
"""

from __future__ import annotations

import math

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing import noise_floor as NF
from models.demos.common.bringup.testing.harness import compare, default_mode, device_ctx, reference_ctx

COMPONENT_DEFAULTS = {
    "component_calib": 2.0,  # every limit = this x the noise floor on the same comparison (Hy4 device: <= 1.7 x)
    "component_bias_share": 0.5,  # a systematic scale may use at most this share of the step's rel budget
    # F138: structural probes, never accuracy limits (they only loosen the second inputs, synthetic or permuted, for
    # structural bugs of 0.1 and more; recorded apart from the accuracy limits)
    "struct_probe": 1.5,  # second inputs: the noise-floor limits x this ...
    "struct_second_rel": 0.03,  # ... a rel limit of at least this, no bias limit ...
    "struct_second_row": 0.15,  # ... worst row at least this (GLM KDA attention x 1e-3: 0.09; hard cases >= 0.34)
    "struct_second_ratio": 0.1,  # ... row norm ratio within 1 +- at least this (GLM: 0.056; hard cases >= 0.13)
    # F96: at RMS 1e-3 a correct device meets absolute floors (fp8 scale, epsilon) the nudge does not model
    "struct_scaled_rel": 0.2,
    "struct_col_factor": 4.0,  # a column's limit is at least this x the whole-output rel limit (MiMo mlp: 2.8 x)
}
SPARSE_ZEROS = 0.75
SCALED_PROBES = ("small", "big")  # second inputs at synthetic scales (cases()): no own-floor checks (F83)
COL_FLOOR = 1e-6  # a column is measured against at least this fraction of the mean column norm (as rows are)
PAD_SENTINEL = NF.PAD_SENTINEL
SMALL_RMS = 1e-3
BIG_SCALE = 2.0


# ---------------------------------------------------------------- output kinds
def normalize_index(t: torch.Tensor) -> torch.Tensor:
    """int64 with every pad (negative, or the uint32 sentinel of topk_large_indices) as -1."""
    t = t.to(torch.int64)
    return torch.where((t < 0) | (t >= PAD_SENTINEL), torch.full_like(t, -1), t)


def _repeats(t: torch.Tensor) -> int:
    srt = t.sort(dim=-1).values
    return int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item())


def output_kind(t: torch.Tensor) -> str:
    if t.is_floating_point():
        if t.dim() >= 2 and t.shape[-1] > 1 and (t == 0).float().mean().item() >= SPARSE_ZEROS:
            return "selection"
        return "float"
    if t.dim() == 2 and t.shape[-1] > 1 and _repeats(normalize_index(t)) == 0:
        return "index"
    return "int"


def limits(s) -> dict:
    """The calib and the structural probe factors (F138: no fixed accuracy number, no spec override)."""
    out = dict(COMPONENT_DEFAULTS)
    out["component_col_factor"] = out["struct_col_factor"]  # F146: the pre-F138 name frozen tests read (structural)
    return out


# ---------------------------------------------------------------- the precision model
def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def rel(a, b) -> float:
    return ((a.double().reshape(b.shape) - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def calib(lim) -> float:
    return lim["component_calib"]


def rel_limit(lim, e, res=None):
    """F76: calib x the precision model's error e, never below calib x res, what one rounding of the output to its dtype
    shows (an exact step); no fixed floor or cap (owner: limits come from the noise floor, never a fixed bar). e (and
    res) is one error (the whole output) or a tensor of them (one per column)."""
    k = calib(lim)
    if isinstance(e, torch.Tensor):
        return k * (e if res is None else torch.maximum(e, res))
    return k * max(e, res or 0.0)


def col_errors(got, want) -> torch.Tensor | None:
    """rel L2 of every column (last dim) over all rows, against its own norm (at least COL_FLOOR x the mean column
    norm): a gate that is ~0 everywhere (Hy4 moe_shared iHC pre gate 0, <= 3e-5) with its sign flipped is a 200 %
    error of its own column; against 1e-3 x the mean column norm it read 1.6 %."""
    if want.dim() < 2 or got.numel() != want.numel():
        return None
    n = want.shape[-1]
    gc, wc = got.double().reshape(-1, n), want.double().reshape(-1, n)
    cn = wc.norm(dim=0)
    return (gc - wc).norm(dim=0) / cn.clamp_min(COL_FLOOR * cn.mean().item() + 1e-30)


# ---------------------------------------------------------------- checks per kind
def _rows(t, shape):
    t = t.double().reshape(shape)
    return t.reshape(shape[0], -1) if len(shape) >= 2 else t.reshape(1, -1)


def float_errors(got, want) -> dict:
    g, w = _rows(got, want.shape), _rows(want, want.shape)
    wn, gn = w.norm(dim=1), g.norm(dim=1)
    floor = 1e-6 * wn.mean().item() + 1e-30
    ratio = torch.where((wn < floor) & (gn < floor), torch.ones_like(wn), gn / wn.clamp_min(floor))
    return {
        "rel": ((g - w).norm() / w.norm().clamp_min(1e-30)).item(),
        "row": ((g - w).norm(dim=1) / wn.clamp_min(floor)).max().item(),
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
        "bias": ratio.median().item() - 1.0,
    }


def float_limits(
    lim, f: float, rel_lim: float, model: dict | None, second: bool = False, col_lim=None, res=None, k=None
) -> dict:
    """The float limits of one input, every one calib (k) x the noise floor's statistic (model), never below what one
    rounding of the output to its dtype shows (res); the bias may use half the rel budget. A second input is there for
    structural bugs (a wrong epsilon, stream order, a clamp: errors of 0.1 and more), not for precision: its limits
    are at least the struct_* probe floors (they only loosen) and it has no bias limit. col_lim: the per-column rel
    limits, on the golden input only (on synthetic inputs a correct device's single columns stray far from the model).
    F138: without a noise floor there is no limit (no fixed number to fall back to)."""
    if not model or res is None or k is None:
        raise ValueError("cannot judge: no noise floor for this output (F138: no fixed accuracy limit exists)")
    dev = lambda e: max(1 - e["ratio_min"], e["ratio_max"] - 1)  # noqa: E731
    row, ratio = k * max(model["row"], res["row"]), k * max(dev(model), dev(res))
    return {
        "rel": max(rel_lim, lim["struct_second_rel"]) if second else rel_lim,
        "row": max(f * row, lim["struct_second_row"]) if second else row,
        "col": None if second else col_lim,
        "ratio": max(f * ratio, lim["struct_second_ratio"]) if second else ratio,
        "bias": math.inf
        if second
        else max(k * abs(model["bias"]), k * abs(res["bias"]), lim["component_bias_share"] * rel_lim),
    }


def float_fails(got, want, L: dict) -> tuple[dict, list[str]]:
    if got.numel() != want.numel():
        return {}, [f"{got.numel()} elements, want {tuple(want.shape)}"]
    if not bool(torch.isfinite(got.float()).all()):
        return {}, ["non-finite output"]
    e = float_errors(got, want)
    e["rel_limit"] = L["rel"]
    ce = col_errors(got, want) if L.get("col") is not None else None
    if ce is not None:
        j = int((ce / L["col"]).argmax().item())
        e.update(col=ce[j].item(), col_limit=L["col"][j].item(), col_index=j)
    dev = max(1 - e["ratio_min"], e["ratio_max"] - 1)
    bad = [
        f"rel {e['rel']:.5f} > {L['rel']:.5f}" if not e["rel"] <= L["rel"] else "",
        f"worst row {e['row']:.5f} > {L['row']:.4f}" if not e["row"] <= L["row"] else "",
        (
            f"column {e['col_index']} rel {e['col']:.5f} > its limit {e['col_limit']:.5f}"
            if "col" in e and not e["col"] <= e["col_limit"]
            else ""
        ),
        (
            f"row norm ratio [{e['ratio_min']:.5f}, {e['ratio_max']:.5f}] outside 1 +- {L['ratio']:.4f}"
            if not dev <= L["ratio"]
            else ""
        ),
        f"median row norm ratio off by {e['bias']:+.5f} (limit {L['bias']:.4f})"
        if not abs(e["bias"]) <= L["bias"]
        else "",
    ]
    return e, [b for b in bad if b]


def f_lim(overlap_lim: float, f: float) -> float:
    """An overlap limit loosened by f (second inputs): the allowed miss grows by f."""
    return 1 - f * (1 - overlap_lim)


def selection_limits(lim, k: float, f: float, model: dict, res: dict, n: int) -> dict:
    """F76: the selection limits from the precision model's own selection stats (model) and one rounding of
    the output to its dtype (res): overlap miss, rel L2 and row sums on matched rows, each calib (k) x f x the larger, the miss
    never below one of the n selected entries."""
    return {
        "overlap": 1 - f * k * max(1 - model.get("overlap", 1.0), 1 - res.get("overlap", 1.0), 1.0 / max(n, 1)),
        "rel": f * k * max(model.get("rel", 0.0), res.get("rel", 0.0)),
        "rowsum": f * k * max(model.get("rowsum", 0.0), res.get("rowsum", 0.0)),
    }


def selection_fails(got, want, lim, f=1.0, slack: dict | None = None, L: dict | None = None) -> tuple[dict, list[str]]:
    """slack: added to the rel and row-sum limits (vs the golden: the fp32 step's own error vs the golden).
    L: the noise-floor limits (selection_limits); without it only the statistics and the structural checks (F138)."""
    slack = slack or {}
    if got.numel() != want.numel():
        return {}, [f"{got.numel()} elements, want {tuple(want.shape)}"]
    g, w = _rows(got, want.shape), _rows(want, want.shape)
    if not bool(torch.isfinite(g).all()):
        return {}, ["non-finite output"]
    gs, ws = g != 0, w != 0
    bad = []
    nbad = int((gs.sum(1) != ws.sum(1)).sum().item())
    if nbad:
        bad.append(f"{nbad} rows select a different number of entries")
    if bool((w >= 0).all()) and not bool((g >= 0).all()):
        bad.append(f"{int((g < 0).sum().item())} negative weights (the reference has none)")
    overlap = ((gs & ws).sum(1).float() / ws.sum(1).clamp_min(1)).mean().item()
    m = (gs == ws).all(1)
    e = {"overlap": overlap, "matched": m.float().mean().item()}
    ol = L["overlap"] if L else -math.inf  # F138: no noise floor, no limit (the caller only wants the statistics)
    if overlap < ol:
        bad.append(f"selection overlap {overlap:.5f} < {ol:.4f}")
    if m.any():
        gm, wm = g[m], w[m]
        e["rel"] = ((gm - wm).norm() / wm.norm().clamp_min(1e-30)).item()
        sw = wm.sum(1)
        nz = sw.abs() > 1e-30
        e["rowsum"] = ((gm.sum(1)[nz] / sw[nz]) - 1).abs().max().item() if nz.any() else 0.0
        rl = (L["rel"] if L else math.inf) + slack.get("rel", 0.0)
        sl = (L["rowsum"] if L else math.inf) + slack.get("rowsum", 0.0)
        if e["rel"] > rl:
            bad.append(f"rel L2 on matched rows {e['rel']:.5f} > {rl:.4f}")
        if e["rowsum"] > sl:
            bad.append(f"row sums off by {e['rowsum']:.5f} (limit {sl:.4f})")
    else:
        bad.append("no row selects the same entries")
    return e, bad


def index_fails(got, want, start: int, lim, thr: float, f=1.0, L: dict | None = None) -> tuple[dict, list[str]]:
    """L (F76): {overlap, worst_row} limits from the noise floor; without it only the structural checks (F138)."""
    if got.is_floating_point():
        return {}, [f"integer positions expected, got {got.dtype}"]
    if got.numel() != want.numel():
        return {}, [f"{got.numel()} elements, want {tuple(want.shape)}"]
    g, w = normalize_index(got.reshape(want.shape)), normalize_index(want)
    vg, vw = g >= 0, w >= 0
    pos = torch.arange(start, start + w.shape[0])[:, None]
    bad = []
    n = int((vg.sum(1) != vw.sum(1)).sum().item())
    if n:
        bad.append(f"{n} rows hold a different number of valid positions")
    if _repeats(g):
        bad.append(f"{_repeats(g)} repeated positions within rows")
    if not bool((vw & (w > pos)).any()):
        nc = int((vg & (g > pos)).sum().item())
        if nc:
            bad.append(f"{nc} positions after their query row (non-causal)")
    tail = lambda v: bool((v.int().diff(dim=1) <= 0).all())  # noqa: E731 - valid entries first, pads only after
    if tail(vw) and not tail(vg):  # a consumer may need it (Hy4: sparse_sdpa reads a row up to its first pad)
        bad.append(f"{int((vg.int().diff(dim=1) > 0).any(1).sum().item())} rows have a pad before a valid position")
    if bool((w == pos).any(1).all()) and not bool((g == pos).any(1).all()):
        bad.append(f"{int((~(g == pos).any(1)).sum().item())} rows do not select their own position")
    rows = torch.tensor(
        [torch.isin(b[b >= 0], a[a >= 0]).float().mean().item() if (b >= 0).any() else 1.0 for a, b in zip(g, w)]
    )
    e = {"overlap": rows.mean().item(), "worst_row": rows.min().item()}
    ol = L["overlap"] if L else -math.inf
    wl = L["worst_row"] if L else -math.inf
    if e["overlap"] < ol:
        bad.append(f"set overlap {e['overlap']:.5f} < {ol:.4f}")
    if e["worst_row"] < wl:
        bad.append(f"worst row overlap {e['worst_row']:.5f} < {wl:.4f}")
    return e, bad


def int_fails(got, want, thr: float) -> tuple[dict, list[str]]:
    if got.numel() != want.numel():
        return {}, [f"{got.numel()} elements, want {tuple(want.shape)}"]
    v = (got.reshape(want.shape) == want).float().mean().item()
    return {"match": v}, ([f"match {v:.5f} < {thr}"] if v < thr else [])


# ---------------------------------------------------------------- inputs: the golden and the second inputs
class Case:
    """One input set: the tensors, the chunk start, fresh reference / device contexts per call."""

    def __init__(self, name, inputs, start, rctx, dctx):
        self.name, self.inputs, self.start, self.rctx, self.dctx = name, inputs, start, rctx, dctx


def _floats(gl, names):
    return [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in names]


def _pow2_to_rms(t: torch.Tensor, rms: float) -> float:
    r = t.float().pow(2).mean().sqrt().item()
    return 2.0 ** -round(math.log2(r / rms)) if r > 0 else 1.0


def _mixed(t: torch.Tensor, i: int) -> torch.Tensor:
    """A float input with its rows and its last-dim columns permuted (seeded per input): values of the real
    distribution without its symmetries (equal iHC streams, near-equal gates), each input permuted differently."""
    if not (t.is_floating_point() and t.dim() >= 2):
        return t
    rows = torch.randperm(t.shape[0], generator=torch.Generator().manual_seed(2 * i + 1))
    cols = torch.randperm(t.shape[-1], generator=torch.Generator().manual_seed(2 * i + 2))
    return t[rows][..., cols].contiguous()


def _clone(x):
    return [t.clone() if isinstance(t, torch.Tensor) else t for t in x]


def cases(s, ref, layer: int, st, g, c: int, kind: str) -> list[Case]:
    hooks = s.hooks()

    def ctxs(c_):
        def d0():
            return device_ctx(layer, g, c_, s)

        def r():
            ctx = reference_ctx(ref, layer, g, c_, s)
            if hasattr(hooks, "swap_context"):
                hooks.swap_context(s, ref, layer, g, c_, ctx, d0())
            return ctx

        def d():
            ctx = d0()
            if hasattr(hooks, "swap_context"):
                hooks.swap_context(s, ref, layer, g, c_, reference_ctx(ref, layer, g, c_, s), ctx)
            return ctx

        return r, d

    gl = g.layer(c, layer)
    main = _floats(gl, st.inputs)
    out = [Case("golden", main, c * g.chunk, *ctxs(c))]
    if st.stateful and c != 0 and g.has_layer(0, layer):
        out.append(Case("chunk0", _floats(g.layer(0, layer), st.inputs), 0, *ctxs(0)))
    # the farthest layer first: iHC streams (and other per-layer structure) diverge with depth
    for other in sorted((L for L in g.layers if L != layer and g.has_layer(c, L)), key=lambda L: -abs(L - layer)):
        go = g.layer(c, other)
        same = all(i in go and go[i].shape == gl[i].shape and go[i].dtype == gl[i].dtype for i in st.inputs)
        if same and any(not torch.equal(go[i], gl[i]) for i in st.inputs):
            out.append(Case(f"layer{other}", _floats(go, st.inputs), c * g.chunk, *ctxs(c)))
            break
    if kind in ("float", "selection") and any(t.is_floating_point() and t.dim() >= 2 for t in main):
        out.append(Case("mixed", [_mixed(t, i) for i, t in enumerate(main)], c * g.chunk, *ctxs(c)))
    if kind == "float":
        f = [_pow2_to_rms(t, SMALL_RMS) if t.is_floating_point() else 1.0 for t in main]
        if any(x < 1 for x in f):
            out.append(Case("small", [t * x if x != 1.0 else t for t, x in zip(main, f)], c * g.chunk, *ctxs(c)))
        big = [t * BIG_SCALE if t.is_floating_point() else t for t in main]
        if any(t.is_floating_point() for t in main):
            out.append(Case("big", big, c * g.chunk, *ctxs(c)))
    return out


def run_cpu(cpu, case: Case, inputs=None):
    return cpu(case.rctx(), *_clone(case.inputs if inputs is None else inputs))


class Expect:
    """What a module's outputs are checked against: the CPU step's output per input, its precision-model limits, the
    golden (+ the fp32 step's own error vs it)."""

    def __init__(self, s, ref, layer, st, g, c, want, thr, mode):
        self.s, self.ref, self.layer, self.st, self.want, self.thr, self.mode = s, ref, layer, st, want, thr, mode
        self.kind = output_kind(want)
        self.lim = limits(s)
        self.k = calib(self.lim)
        self.cpu_step = ref.component(layer, st.name)
        self.cases = cases(s, ref, layer, st, g, c, self.kind)
        self.cpu = {x.name: run_cpu(self.cpu_step, x) for x in self.cases}
        # F80: the noise floor is the larger of the official code's arithmetic (cached with the goldens, golden input
        # only) and the input nudge (the fp32 step on inputs moved inside their official dtype's rounding cell)
        self.dtypes = NF.input_dtypes(s, s.block_type_of(layer), st.name, len(st.inputs))
        self.out_dtype = NF.output_dtype(s, s.block_type_of(layer), st.name)  # vs the CPU: one rounding to it
        self.arith = NF.arith(s, layer, st.name, c) or {}
        self.floor, self.scores = {}, {}
        for x in self.cases:
            run = lambda *inp, x=x: self.cpu_step(x.rctx(), *_clone(list(inp)))  # noqa: E731
            if self.kind in ("selection", "index"):
                _, sc, kk = NF.tapped(run, _clone(x.inputs))
                outs, nsc = [], []
                for k_ in range(NF.SEEDS):
                    with NF.inner(k_, self.dtypes):
                        o, ns, _ = NF.tapped(
                            run, [NF.nudge(t, 1000 * k_ + j, d) for j, (t, d) in enumerate(zip(x.inputs, self.dtypes))]
                        )
                    outs.append(o)
                    nsc.append(ns)
                self.floor[x.name] = outs
                if sc is not None and all(n is not None for n in nsc):
                    self.scores[x.name] = (sc, kk, nsc)
            else:
                self.floor[x.name] = NF.nudged(run, _clone(x.inputs), self.dtypes)
        self.rel_lim, self.model_err, self.res, self.col_lim, self.L = {}, {}, {}, {}, {}
        for x in self.cases:
            cpu, outs = self.cpu[x.name], self.floor[x.name]
            f = 1.0 if x.name == "golden" else self.lim["struct_probe"]
            if self.kind == "float":
                fe = NF.worst_float(outs, cpu, float_errors)
                self.model_err[x.name] = NF.merge_float(self.arith.get("vs_step") if x.name == "golden" else None, fe)
                self.res[x.name] = float_errors(NF.round_to(cpu, self.out_dtype), cpu)
                self.rel_lim[x.name] = rel_limit(self.lim, self.model_err[x.name]["rel"], self.res[x.name]["rel"])
                cols = [col_errors(o, cpu) for o in outs]
                mc = None if cols[0] is None else torch.stack(cols).amax(0)
                rc = col_errors(NF.round_to(cpu, self.out_dtype), cpu)
                self.col_lim[x.name] = None if mc is None else rel_limit(self.lim, mc, rc)
            elif self.kind == "selection":
                me = _worst_selection([selection_fails(o, cpu, self.lim)[0] for o in outs])
                re_ = selection_fails(NF.round_to(cpu, self.out_dtype), cpu, self.lim)[0]
                self.L[x.name] = selection_limits(self.lim, self.k, f, me, re_, int((cpu != 0).sum().item()))
            elif self.kind == "index":
                self.L[x.name] = self._index_limits(outs, cpu, x.start, f)
        if self.kind == "float":
            self.golden_err = rel(self.cpu["golden"], want)
            self.golden_col_err = col_errors(self.cpu["golden"], want)
            # vs the golden (full precision, F128): the floor's own statistics; one rounding to the output dtype (F133)
            fg = NF.worst_float(self.floor["golden"], want, float_errors)
            self.gold_model = NF.merge_float(self.arith.get("vs_golden"), fg)
            self.gold_res = float_errors(NF.round_to(self.cpu["golden"], self.out_dtype), want)
        if self.kind == "selection":
            gm = _worst_selection([selection_fails(o, want, self.lim)[0] for o in self.floor["golden"]])
            gr = selection_fails(NF.round_to(self.cpu["golden"], self.out_dtype), want, self.lim)[0]
            self.golden_L = selection_limits(self.lim, self.k, 1.0, gm, gr, int((want != 0).sum().item()))
        if self.kind == "index":
            self.golden_L = self._index_limits(self.floor["golden"], want, self.cases[0].start, 1.0)
        # the gate vs the golden (the template's metric): the noise floor; F138: a literal THRESHOLD never gates
        if thr is not None:
            raise ValueError("a literal THRESHOLD never gates (F138); run_component_test records it as a diagnostic")
        self.thr_literal = False
        gm = "topk_overlap" if self.kind == "index" and mode in (None, "topk_overlap") else (mode or default_mode(want))
        mv = min(gate_value(o, want, self.kind, mode) for o in self.floor["golden"])
        if "pcc_vs_golden" in (self.arith.get("vs_golden") or {}) and gm == "pcc":
            mv = min(mv, self.arith["vs_golden"]["pcc_vs_golden"])
        rv = (
            gate_value(NF.round_to(self.cpu["golden"], self.out_dtype), self.cpu["golden"], self.kind, mode)
            if want.is_floating_point()
            else 1.0
        )
        thr = NF.gate_limit(gm, mv, rv, self.k, want.numel())
        self.thr_how = (
            f"{self.k} x the noise floor's miss {1 - mv:.3g} (input nudge"
            + (", official arithmetic" if self.arith else "")
            + f"; {self.out_dtype} resolution {1 - rv:.3g})"
        )
        self.thr = thr
        # F81: the gated error vs the golden: rel L2 for float outputs (PCC is blind to a scale or an offset), the
        # missed fraction (1 - overlap / match) for integer outputs; limit = calib x the noise floor's error
        if self.kind in ("float", "selection"):
            fl = max(rel(o, want) for o in self.floor["golden"])
            ag = (self.arith.get("vs_golden") or {}).get("rel") or 0.0
            gres = rel(NF.round_to(self.cpu["golden"], self.out_dtype), want)
            m = max(fl, ag, gres)
            self.err_limit = self.k * m
            self.err_how = f"{self.k} x the noise floor's rel vs the golden {m:.3g}" + (
                " (official arithmetic)" if ag >= fl else ""
            )
        else:
            self.err_limit = 1.0 - self.thr
            self.err_how = f"1 - the overlap limit ({self.thr_how})"

    def _index_limits(self, outs, want, start: int, f: float) -> dict:
        es = [index_fails(m, want, start, self.lim, 0.0, L={"overlap": 0.0, "worst_row": 0.0})[0] for m in outs]
        e = {"overlap": min(x.get("overlap", 1.0) for x in es), "worst_row": min(x.get("worst_row", 1.0) for x in es)}
        valid = normalize_index(want) >= 0
        n, per_row = max(int(valid.sum().item()), 1), max(int(valid.sum(1).max().item()), 1)
        return {
            "overlap": 1 - f * self.k * max(1 - e.get("overlap", 1.0), 1.0 / n),
            "worst_row": 1 - f * self.k * max(1 - e.get("worst_row", 1.0), 1.0 / per_row),
        }

    def describe(self) -> str:
        lims = ", ".join(f"{k} {v:.4f}" for k, v in self.rel_lim.items())
        return (
            f"{self.st.name} L{self.layer}: output kind {self.kind}, input dtypes {self.dtypes}, out {self.out_dtype}"
            + (" + official arithmetic" if self.arith else "")
            + f", inputs {[x.name for x in self.cases]}"
            + (f"; rel limits (calibrated) {lims}" if lims else "")
        )

    def check(self, tag, got, ref, case, vs_golden, quiet):
        f = 1.0 if case.name == "golden" else self.lim["struct_probe"]
        warn = []  # F89: the own-floor row / column / bias checks warn, never gate (owner)
        if self.kind == "float":
            rl = self.rel_lim[case.name] + (self.golden_err if vs_golden else 0.0)
            model, res = self.model_err[case.name], self.res[case.name]
            if vs_golden:
                model, res = self.gold_model, self.gold_res
                rl = max(rl, self.k * max(model["rel"], res["rel"]))
            second = case.name != "golden"
            if case.name in SCALED_PROBES:  # F96
                rl = max(rl, self.lim["struct_scaled_rel"])
            cl = self.col_lim[case.name]
            if cl is not None and vs_golden and self.golden_col_err is not None:
                cl = cl + self.golden_col_err
            if cl is not None:  # never tighter than col_factor x the whole-output limit (Gemma mlp: one column 1.06 x)
                cl = cl.clamp_min(self.lim["struct_col_factor"] * rl)
            L = float_limits(self.lim, f, rl, model, second, cl, res, self.k)
            # F83: rows and columns against their own floors (below) replace the worst-row, row-ratio and per-column
            # limits, which divided by small norms behind a fixed 1e-6 floor
            L.update(row=math.inf, ratio=math.inf, col=None)
            e, bad = float_fails(got, ref, L)
            # F83 on inputs with real value scales only: the scaled probes (small: RMS 1e-3, big: x 2) are for
            # structural bugs, where a correct device's absolute limits (an fp8 scale floor, an epsilon) dominate and
            # the relative nudge does not model them (DeepSeek-V4.1 attention on `small`: one row 36x its floor)
            if got.numel() == ref.numel() and not bad and case.name not in SCALED_PROBES:
                outs = self.floor["golden"] if vs_golden else self.floor[case.name]
                base = self.gold_model["rel"] if vs_golden else self.model_err[case.name]["rel"]
                ar = (
                    (self.arith.get("vs_golden" if vs_golden else "vs_step") or {}).get("rel")
                    if case.name == "golden"
                    else None
                )
                nud = max(rel(o, ref) for o in outs)
                scale = (
                    max(1.0, (ar or 0.0) / nud) if nud > 0 else 1.0
                )  # the official arithmetic's excess over the nudge
                own = NF.own_floor(got, ref, outs, "bf16" if vs_golden else self.out_dtype, scale)
                e.update(own)
                kk = f * self.k
                if own["row_floor"] > kk:
                    warn.append(f"row {own['row_at']} is {own['row_floor']:.3g} x its own floor (limit {kk:g})")
                if case.name == "golden" and own.get("col_floor", 0) > kk:
                    warn.append(f"column {own['col_at']} is {own['col_floor']:.3g} x its own floor (limit {kk:g})")
                if case.name == "golden" and own.get("bias_floor", 0) > kk:
                    warn.append(
                        f"column {own['bias_at']} has an additive bias of {own['bias_signed']:+.3g} of its rms, "
                        f"{own['bias_floor']:.3g} x its floor (limit {kk:g})"
                    )
        elif self.kind == "selection":
            e, bad = selection_fails(got, ref, self.lim, f, L=self.golden_L if vs_golden else self.L[case.name])
            if not vs_golden and case.name in self.scores and got.numel() == ref.numel():
                e2, bad2 = self._flips(got, case.name, f, ref)
                e.update(e2)
                bad += bad2
        elif self.kind == "index":
            L = self.golden_L if vs_golden else self.L[case.name]
            e, bad = index_fails(got, ref, case.start, self.lim, self.thr, f, L=L)
            if not vs_golden and case.name in self.scores and got.numel() == ref.numel() and not bad:
                e2, bad2 = self._flips(got.reshape(ref.shape), case.name, f, ref)  # F101: near ties, tie-aware recall
                e.update(e2)
                bad += bad2
        else:
            outs = self.floor[case.name] if not vs_golden else self.floor["golden"]
            mm = min(int_fails(m, ref, 0.0)[0].get("match", 1.0) for m in outs)
            e, bad = int_fails(got, ref, 1 - f * self.k * max(1 - mm, 1.0 / max(ref.numel(), 1)))
        if not quiet:
            for k, v in e.items():
                metrics.record(f"{tag}_{k}", v)
            nums = " ".join(f"{k}={v:.6g}" for k, v in e.items())
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: {nums}" + (f" ({'; '.join(bad)})" if bad else ""))
            if warn:
                metrics.record(f"{tag}_floor_warn", 1)
                print(f"WARN {tag} (not gating, for the owner's review; do not work on it): {'; '.join(warn)}")
        return not bad

    def _flips(self, got, case: str, f: float, want) -> tuple[dict, list[str]]:
        """F80: every selection flip vs the CPU step must be explained by the nudge (or be a near tie, F101), and there
        are at most calib x the nudge's own flips (at least one row). Records the tie-aware recall (F101)."""
        sc, kk, nsc = self.scores[case]
        w = sc.shape[-1]
        ws = NF.support(want, kk, w)
        fl = NF.flips(sc, kk, NF.support(got, kk, w), nsc, ws)
        nudge_frac = max(NF.flips(sc, kk, NF.support(n, kk, w), [], ws)["frac"] for n in self.floor[case])
        lim = f * self.k * max(nudge_frac, 1.0 / max(sc.shape[0], 1))
        e = {
            "flip_rows": fl["rows"],
            "flips_unexplained": fl["unexplained"],
            "flip_frac": fl["frac"],
            "flip_limit": lim,
            "recall": fl["recall"],
            "recall_tie": fl["recall_tie"],
            "worst_recall_tie": fl["worst_recall_tie"],
        }
        bad = [f"{fl['unexplained']} rows select another entry the nudge does not explain" if fl["unexplained"] else ""]
        if fl["frac"] > lim:
            bad.append(f"{fl['frac']:.5f} of the rows select another entry (limit {lim:.5f})")
        return e, [b for b in bad if b]

    def evaluate(self, outs: dict, quiet=False, stop_early=False) -> list[str]:
        """Every check of one module's outputs (outs[input name]); the names of the failed checks."""
        tag, failed = f"auto_{self.st.name}_L{self.layer:02d}", []
        for case in self.cases:
            got = outs[case.name]
            if isinstance(got, Exception):
                failed.append(f"{tag}_{case.name}")
                if not quiet:
                    print(f"FAIL {tag}_{case.name}: the module raised {type(got).__name__}: {got}")
            elif case.name == "golden":
                if not self.check(f"{tag}_vs_cpu", got, self.cpu[case.name], case, False, quiet):
                    failed.append(f"{tag}_vs_cpu")
                if not self.check(f"{tag}_vs_golden", got, self.want, case, True, quiet):
                    failed.append(f"{tag}_vs_golden")
            elif not self.check(f"{tag}_{case.name}", got, self.cpu[case.name], case, False, quiet):
                failed.append(f"{tag}_{case.name}")
            if failed and stop_early:
                break
        return failed


def gate_value(out, want, kind, mode) -> float:
    """The golden gate's metric of ``out`` vs ``want`` (golden_gate's comparison, no print, no record)."""
    if out.numel() != want.numel():
        return 0.0
    out = out.reshape(want.shape)
    if kind == "index" and mode in (None, "topk_overlap") and not out.is_floating_point():
        g, w = normalize_index(out), normalize_index(want)
        return NF.value("topk_overlap", g, w)
    return NF.value(mode or default_mode(want), out, want)


def _worst_selection(es: list[dict]) -> dict:
    """The worst of the floor's selection stats: lowest overlap, largest matched-row rel and row-sum error."""
    out = {"overlap": min(e.get("overlap", 1.0) for e in es)}
    for k in ("rel", "rowsum"):
        v = [e[k] for e in es if k in e]
        if v:
            out[k] = max(v)
    return out


def error_value(out, want, ex) -> float:
    """F81: rel L2 of a float output vs ``want``; the missed fraction (1 - overlap or match) of an integer output."""
    if out.numel() != want.numel() or (out.is_floating_point() and not bool(torch.isfinite(out.float()).all())):
        return math.inf
    if ex.kind in ("float", "selection"):
        return rel(out, want)
    return 1.0 - gate_value(out, want, ex.kind, ex.mode)


def error_gate(name: str, out, want, ex, quiet: bool = False) -> bool:
    """The template's gated metric (F81): ``name`` = the error vs the golden, ``<name>_limit`` its limit (the ledger
    gates ``<= limit``)."""
    v, lim = error_value(out, want, ex), ex.err_limit
    if not quiet:
        metrics.record(name, v)
        metrics.record(f"{name}_limit", lim)
        print(f"{'ok  ' if v <= lim else 'FAIL'} {name}: {v:.6g} (<= {lim:.6g} = {ex.err_how})")
    return v <= lim


def golden_gate(name, out, want, kind, mode, thr, how: str = "") -> bool:
    """The template's gate (compare vs golden); index outputs are normalized and compared order-free. The limit is
    recorded as ``<name>_limit`` (F76: the ledger gates ``>= limit``)."""
    metrics.record(f"{name}_limit", thr)
    if how:
        print(f"note {name}: limit {thr:.6f} = {how}")
    if (
        kind == "index"
        and mode in (None, "topk_overlap")
        and not out.is_floating_point()
        and out.numel() == want.numel()
    ):
        return compare(name, normalize_index(out.reshape(want.shape)), normalize_index(want), "topk_overlap", thr)[1]
    return compare(name, out, want, mode or default_mode(want), thr)[1]


def golden_gate_quiet(out, want, kind, thr, mode=None) -> bool:
    """golden_gate without printing or recording (the sweep evaluates many modules in one process)."""
    if out.numel() != want.numel():
        return False
    out = out.reshape(want.shape)
    if kind == "index" and mode in (None, "topk_overlap") and not out.is_floating_point():
        g, w = normalize_index(out), normalize_index(want)
        v = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / len(set(b.tolist())) for a, b in zip(g, w)])
        return v.mean().item() >= thr
    mode = mode or default_mode(want)
    if mode == "pcc":
        return metrics.pcc(out.float(), want.float()) >= thr
    if mode in ("match", "topk_overlap"):
        return (out == want).float().mean().item() >= thr
    raise ValueError(f"compare mode {mode!r}")


# ---------------------------------------------------------------- the freeze sweep
def sweep(ex: Expect, gate_name: str) -> bool:
    """BRINGUP_IMPL=mutations: the CPU proof that the checks catch the standard mistakes (module docstring)."""
    print("sweep " + ex.describe())
    if ex.kind == "int":
        print(
            f"FAIL sweep: output kind 'int' ({ex.want.dtype} {tuple(ex.want.shape)}) has no built-in checks: review it"
        )
        return False

    def verdict(outs, stop_early):
        f = ex.evaluate(outs, quiet=True, stop_early=stop_early)
        ok = error_gate(gate_name, outs["golden"], ex.want, ex, quiet=True)
        return f + ([] if ok else [gate_name])

    rows, ok = [], True

    def control(name, outs):
        nonlocal ok
        f = verdict(outs, False)
        rows.append((name, "pass" if not f else "FAIL", ", ".join(f)))
        ok &= not f

    control("reference", ex.cpu)
    for k in [k for k in MU.KINDS if MU.applies(k, ex.want)]:
        mut = {n: MU.mutate(o, k) for n, o in ex.cpu.items()}
        f = verdict(mut, True)
        # F91 (owner): a mistake no larger than the step's own limit on every input is one a correct step may make
        size = {x.name: rel(mut[x.name], ex.cpu[x.name]) for x in ex.cases} if ex.kind == "float" else {}
        rl = {n: v if n == "golden" else max(v, ex.lim["struct_second_rel"]) for n, v in ex.rel_lim.items()}
        rl.update({n: max(rl[n], ex.lim["struct_scaled_rel"]) for n in SCALED_PROBES if n in rl})  # F96
        within = bool(size) and all(v <= rl.get(n, 0.0) for n, v in size.items())  # the limits check() applies
        how = f"size {max(size.values()):.3g} <= the step's limit" if within else ", ".join(f)
        rows.append((k, "caught" if f else "within limit" if within else "SLIPPED", how))
        ok &= bool(f) or within
    # F80: a correct step at the noise floor passes: inputs nudged with a seed the floor did not use, output in bf16
    nudged = {
        x.name: run_cpu(
            ex.cpu_step, x, [NF.nudge(t, 99991 + j, d) for j, (t, d) in enumerate(zip(x.inputs, ex.dtypes))]
        )
        for x in ex.cases
    }
    control("input nudge (new seed)", nudged)
    if ex.kind == "float" and ex.out_dtype == "bf16":
        control("bf16 output", {n: MU.bf16(o) for n, o in ex.cpu.items()})
    g0, base = ex.cases[0], ex.cpu["golden"]
    for i, t in enumerate(g0.inputs):
        if not (isinstance(t, torch.Tensor) and t.is_floating_point()):
            continue
        x = list(g0.inputs)
        x[i] = t * 1.02
        y = run_cpu(ex.cpu_step, g0, x)
        f = verdict(dict(ex.cpu, golden=y), True)
        move = rel(y, base) if y.is_floating_point() and base.is_floating_point() else float("nan")
        rows.append((f"input {ex.st.inputs[i]} x 1.02", "seen" if f else "not seen", f"output moves by rel {move:.3g}"))
    print(f"{'ok  ' if ok else 'FAIL'} sweep {ex.st.name} L{ex.layer}:")
    for r in rows:
        print(f"    {r[0]:<28} {r[1]:<9} {r[2][:160]}")
    metrics.record(f"sweep_{ex.st.name}_L{ex.layer:02d}_caught", sum(r[1] == "caught" for r in rows))
    metrics.record(f"sweep_{ex.st.name}_L{ex.layer:02d}_slipped", sum(r[1] in ("SLIPPED", "FAIL") for r in rows))
    return ok
