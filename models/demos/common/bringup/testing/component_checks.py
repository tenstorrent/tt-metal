# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Built-in checks of a component test (F56): ``run_component_test(..., checks="auto")``.

The rendered component template had one gate, PCC (or exact match) vs the golden, and a test-role agent then added
checks by hand for every step (Hy4: 42 reviews, 4 h). The checks it added are the same few each time; these do them
generically, chosen by the kind of the step's output:

    float       vs the CPU step on the same inputs and vs the golden: finite, rel L2, worst row rel L2
                (<= component_row), every row's norm ratio (1 +- component_ratio), the median row norm ratio
                (1 +- component_bias: a systematic scale), and every column's rel L2 over the rows (an iHC gate near 0
                with its sign flipped moves nothing else), each column against its own limit: the rule below applied
                to the precision model's error in that column, without the cap, and never below component_col_factor x
                the whole-output limit (the column check is for structural column bugs); on the golden input only. The rel L2 limit follows the step's precision: the CPU step
                run with every float intermediate rounded to bf16 (bf16 inputs and output; a model of a correct device
                step, fp32 accumulation inside each op) has rel L2 e vs the fp32 step; the limit is
                component_calib x e (component_calib_low for the step kinds in ``tests.low_precision_kinds``, default
                [moe]: bfp8 expert weights are not in the model), within [component_floor, component_rel], never
                below component_margin x e; vs the golden it adds the fp32 step's own error vs the bf16 golden.
    selection   a float output with >= 75 % exact zeros (dense router weights): the same nonzero count per row, no
                negative weight when the reference has none, mean selection overlap (>= component_select), rel L2 on
                the rows with the same selection (<= component_select_rel), their row sums (1 +- component_rowsum).
    index       an integer [rows, k] output whose rows are sets (top-k positions, pads -1 or 0xFFFFFFFF): order-free
                overlap (mean >= the gate threshold, worst row >= component_index_row), the same valid count per row,
                no repeats, and what the reference obeys: no position after its row (causal), every row selecting its
                own position, pads only after the valid positions. The golden gate uses ``topk_overlap``, never positional match.
    int         anything else: exact match as before; an unknown kind, so the freeze sweep sends it to a review.

Second inputs, each compared with the CPU step on the same inputs, for structural bugs the golden cannot show (rel
limit at least component_second_rel, the fixed limits x component_probe, no bias limit):
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
import weakref

import torch
from torch.overrides import TorchFunctionMode

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import Ctx
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.harness import compare, default_mode, reference_ctx

COMPONENT_DEFAULTS = {
    "component_rel": 0.015,  # rel L2 cap (Hy4 device components vs golden: <= 0.0084)
    "component_floor": 0.003,  # rel L2 limit never below this
    "component_calib": 2.0,  # rel L2 limit = this x the bf16 precision model's error (Hy4 device: <= 1.7 x)
    "component_calib_low": 3.5,  # the same for tests.low_precision_kinds (Hy4 bfp8 experts: 2.1 x)
    "component_col_factor": 4.0,  # a column's limit is at least this x the whole-output rel limit (MiMo mlp: 2.8 x)
    "component_margin": 1.2,  # ... and never below this x the model's error, even above the cap
    "component_row": 0.03,  # worst row rel L2 (Hy4 device: <= 0.016)
    "component_ratio": 0.015,  # every row's norm ratio within 1 +- this (Hy4 device: within 0.0073)
    "component_bias": 0.004,  # median row norm ratio within 1 +- this (a systematic scale) ...
    "component_bias_share": 0.5,  # ... or this x the rel limit, whichever is larger
    "component_select": 0.995,  # selection outputs: mean selection overlap (Hy4 device router: 0.9982)
    "component_select_rel": 0.005,  # rel L2 on rows with the same selection (Hy4 device: 0.0017)
    "component_rowsum": 0.004,  # row sums on those rows within 1 +- this
    "component_index_row": 0.95,  # index outputs: worst row set overlap (Hy4 device indexer: 0.9888)
    "component_probe": 1.5,  # second inputs: the fixed limits x this ...
    "component_second_rel": 0.03,  # ... and a rel limit of at least this, no bias limit ...
    "component_second_row": 0.15,  # ... worst row at least this (GLM KDA attention x 1e-3: 0.09; hard cases >= 0.34)
    "component_second_ratio": 0.1,  # ... row norm ratio within 1 +- at least this (GLM: 0.056; hard cases >= 0.13)
}
LOW_PRECISION_KINDS = ("moe",)
SPARSE_ZEROS = 0.75
COL_FLOOR = 1e-3  # a column is measured against at least this fraction of the mean column norm (a ~0 column)
PAD_SENTINEL = 0xFFFFFFFF
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
    return {k: s.threshold(k, v) for k, v in COMPONENT_DEFAULTS.items()}


# ---------------------------------------------------------------- the precision model
def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


class _Bf16Everywhere(TorchFunctionMode):
    """Rounds to bf16 the float32 result of every op that reads a value derived from the step's inputs; constants
    (weights, RoPE tables, masks, positions: anything built without the inputs) stay exact, as on a device where they
    are prepared on the host or in fp32."""

    def __init__(self, inputs):
        super().__init__()
        self.tainted = {}
        for t in inputs:
            self._mark(t)

    def _mark(self, t):
        if isinstance(t, torch.Tensor):
            self.tainted[id(t)] = weakref.ref(t)

    def _round(self, o, ptrs=()):
        # a view or an in-place result shares memory with an argument: replacing it by a rounded copy would cut the
        # alias (Gemma writes its attention output through out.view(...)); it is marked, not rounded
        if isinstance(o, torch.Tensor) and o.dtype == torch.float32 and _ptr(o) not in ptrs:
            o = o.to(torch.bfloat16).float()
        self._mark(o)
        return o

    def _is(self, t):
        r = self.tainted.get(id(t)) if isinstance(t, torch.Tensor) else None
        return r is not None and r() is t

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        flat = [a for a in list(args) + list(kwargs.values())]
        flat += [x for a in flat if isinstance(a, (list, tuple)) for x in a]
        if any(self._is(a) for a in flat):
            ptrs = {_ptr(a) for a in flat if isinstance(a, torch.Tensor)}
            if isinstance(out, torch.Tensor):
                out = self._round(out, ptrs)
            elif type(out) in (tuple, list):
                out = type(out)(self._round(o, ptrs) for o in out)
            elif isinstance(out, tuple):  # torch.return_types (topk, max, ...): marked, not rounded
                for o in out:
                    self._mark(o)
        return out


def precision_model(ref, cpu, case) -> torch.Tensor:
    """The CPU step on bf16 inputs with every float32 intermediate that depends on them rounded to bf16, output bf16.
    The reference object's attributes (one level of dicts too) are restored afterwards: a cache filled in this mode
    would otherwise hand its values to every later exact run."""
    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(ref).items()}
    x = [MU.bf16(t) for t in _clone(case.inputs)]
    try:
        with _Bf16Everywhere(x):
            y = cpu(case.rctx(), *x)
    finally:
        vars(ref).clear()
        vars(ref).update(saved)
    return MU.bf16(y)


def rel(a, b) -> float:
    return ((a.float().reshape(b.shape) - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def rel_limit(lim, e, low: bool):
    """calib x e within [floor, cap], but never below component_margin x e: a correct bf16 step must pass (when the
    cap would cut it, the looser limit may let a mistake through, and the sweep then sends the test to a review).
    e is one error (the whole output) or a tensor of them (one per column)."""
    k = lim["component_calib_low"] if low else lim["component_calib"]
    if isinstance(e, torch.Tensor):  # per column: no cap (the worst of thousands of columns sits above the mean)
        return torch.maximum((k * e).clamp_min(lim["component_floor"]), lim["component_margin"] * e)
    return max(lim["component_floor"], min(lim["component_rel"], k * e), lim["component_margin"] * e)


def col_errors(got, want) -> torch.Tensor | None:
    """rel L2 of every column (last dim) over all rows; a column is measured against at least COL_FLOOR x the mean
    column norm (a column that is ~0 everywhere, such as an iHC gate near 0, is judged on the output's scale)."""
    if want.dim() < 2 or got.numel() != want.numel():
        return None
    n = want.shape[-1]
    gc, wc = got.float().reshape(-1, n), want.float().reshape(-1, n)
    cn = wc.norm(dim=0)
    return (gc - wc).norm(dim=0) / cn.clamp_min(COL_FLOOR * cn.mean().item() + 1e-30)


# ---------------------------------------------------------------- checks per kind
def _rows(t, shape):
    t = t.float().reshape(shape)
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


def float_limits(lim, f: float, rel_lim: float, model: dict | None, second: bool = False, col_lim=None) -> dict:
    """The float limits of one input: rel from the precision model (rel_limit), the fixed ones x f; each at least
    component_margin x what the precision model itself reaches (a correct bf16 step must pass). A second input is
    there for structural bugs (a wrong epsilon, stream order, a clamp: errors of 0.1 and more), not for precision: its
    rel limit is at least component_second_rel and it has no bias limit (a correct device drifts on synthetic scales;
    Hy4 attention x 1e-3: rel 0.009, bias -0.0075). col_lim: the per-column rel limits (rel_limit of the precision
    model's per-column errors, no cap), on the golden input only: on synthetic inputs a correct device's single
    columns stray far from the model (Hy4 attention L1, one of 6144 columns: 0.043 on permuted, 0.14 on x 1e-3)."""
    out = {
        "rel": max(rel_lim, lim["component_second_rel"]) if second else rel_lim,
        "row": max(f * lim["component_row"], lim["component_second_row"]) if second else lim["component_row"],
        "col": None if second else col_lim,  # the golden input only (Hy4 attention on x 1e-3 inputs: one column 0.14)
        "ratio": max(f * lim["component_ratio"], lim["component_second_ratio"]) if second else lim["component_ratio"],
        # a systematic scale may use at most half of the step's error budget (GLM KDA attention: -0.45 % of 0.68 %)
        "bias": math.inf if second else max(lim["component_bias"], lim["component_bias_share"] * rel_lim),
    }
    if model:
        m = lim["component_margin"]
        out["row"] = max(out["row"], m * model["row"])
        out["ratio"] = max(out["ratio"], m * max(1 - model["ratio_min"], model["ratio_max"] - 1))
        out["bias"] = max(out["bias"], m * abs(model["bias"]))
    return out


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


def selection_fails(got, want, lim, f=1.0, slack: dict | None = None) -> tuple[dict, list[str]]:
    """slack: added to the rel and row-sum limits (vs the golden: the fp32 step's own error vs the bf16 golden)."""
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
    if overlap < f_lim(lim["component_select"], f):
        bad.append(f"selection overlap {overlap:.5f} < {f_lim(lim['component_select'], f):.4f}")
    if m.any():
        gm, wm = g[m], w[m]
        e["rel"] = ((gm - wm).norm() / wm.norm().clamp_min(1e-30)).item()
        sw = wm.sum(1)
        nz = sw.abs() > 1e-30
        e["rowsum"] = ((gm.sum(1)[nz] / sw[nz]) - 1).abs().max().item() if nz.any() else 0.0
        rl, sl = f * lim["component_select_rel"] + slack.get("rel", 0.0), f * lim["component_rowsum"] + slack.get(
            "rowsum", 0.0
        )
        if e["rel"] > rl:
            bad.append(f"rel L2 on matched rows {e['rel']:.5f} > {rl:.4f}")
        if e["rowsum"] > sl:
            bad.append(f"row sums off by {e['rowsum']:.5f} (limit {sl:.4f})")
    else:
        bad.append("no row selects the same entries")
    return e, bad


def index_fails(got, want, start: int, lim, thr: float, f=1.0) -> tuple[dict, list[str]]:
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
    if e["overlap"] < f_lim(thr, f):
        bad.append(f"set overlap {e['overlap']:.5f} < {f_lim(thr, f):.4f}")
    if e["worst_row"] < f_lim(lim["component_index_row"], f):
        bad.append(f"worst row overlap {e['worst_row']:.5f} < {f_lim(lim['component_index_row'], f):.4f}")
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
        start = c_ * g.chunk

        def d0():
            return Ctx(
                layer,
                start,
                g.chunk,
                None,
                {"state_prefix": g.state(layer, at=start), "prefix_len": start, "max_seq": g.seq},
            )

        def r():
            ctx = reference_ctx(ref, layer, g, c_)
            if hasattr(hooks, "swap_context"):
                hooks.swap_context(s, ref, layer, g, c_, ctx, d0())
            return ctx

        def d():
            ctx = d0()
            if hasattr(hooks, "swap_context"):
                hooks.swap_context(s, ref, layer, g, c_, reference_ctx(ref, layer, g, c_), ctx)
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
        self.low = st.kind in tuple(s.get("tests.low_precision_kinds") or LOW_PRECISION_KINDS)
        self.cpu_step = ref.component(layer, st.name)
        self.cases = cases(s, ref, layer, st, g, c, self.kind)
        self.cpu = {x.name: run_cpu(self.cpu_step, x) for x in self.cases}
        self.rel_lim, self.model, self.model_err, self.col_lim = {}, {}, {}, {}
        if self.kind == "float":
            for x in self.cases:
                self.model[x.name] = precision_model(ref, self.cpu_step, x)
                self.model_err[x.name] = float_errors(self.model[x.name], self.cpu[x.name])
                self.rel_lim[x.name] = rel_limit(self.lim, self.model_err[x.name]["rel"], self.low)
                mc = col_errors(self.model[x.name], self.cpu[x.name])
                self.col_lim[x.name] = None if mc is None else rel_limit(self.lim, mc, self.low)
            self.golden_err = rel(self.cpu["golden"], want)
            self.golden_col_err = col_errors(self.cpu["golden"], want)
        if self.kind == "selection":  # the fp32 step's own error vs the bf16 golden (Gemma router row sums: 0.0044)
            self.golden_sel = selection_fails(self.cpu["golden"], want, self.lim)[0]

    def describe(self) -> str:
        lims = ", ".join(f"{k} {v:.4f}" for k, v in self.rel_lim.items())
        return f"{self.st.name} L{self.layer}: output kind {self.kind}, inputs {[x.name for x in self.cases]}" + (
            f"; rel limits ({'low precision, ' if self.low else ''}calibrated) {lims}" if lims else ""
        )

    def check(self, tag, got, ref, case, vs_golden, quiet):
        f = 1.0 if case.name == "golden" else self.lim["component_probe"]
        if self.kind == "float":
            rl = self.rel_lim[case.name] + (self.golden_err if vs_golden else 0.0)
            second = case.name != "golden"
            cl = self.col_lim[case.name]
            if cl is not None and vs_golden and self.golden_col_err is not None:
                cl = cl + self.golden_col_err
            if cl is not None:  # never tighter than col_factor x the whole-output limit (Gemma mlp: one column 1.06 x)
                cl = cl.clamp_min(self.lim["component_col_factor"] * rl)
            e, bad = float_fails(got, ref, float_limits(self.lim, f, rl, self.model_err[case.name], second, cl))
        elif self.kind == "selection":
            e, bad = selection_fails(got, ref, self.lim, f, self.golden_sel if vs_golden else None)
        elif self.kind == "index":
            e, bad = index_fails(got, ref, case.start, self.lim, self.thr, f)
        else:
            e, bad = int_fails(got, ref, self.thr)
        if not quiet:
            for k, v in e.items():
                metrics.record(f"{tag}_{k}", v)
            nums = " ".join(f"{k}={v:.6g}" for k, v in e.items())
            print(f"{'ok  ' if not bad else 'FAIL'} {tag}: {nums}" + (f" ({'; '.join(bad)})" if bad else ""))
        return not bad

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


def golden_gate(name, out, want, kind, mode, thr) -> bool:
    """The template's gate (compare vs golden); index outputs are normalized and compared order-free."""
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
        return f + ([] if golden_gate_quiet(outs["golden"], ex.want, ex.kind, ex.thr, ex.mode) else [gate_name])

    rows, ok = [], True

    def control(name, outs):
        nonlocal ok
        f = verdict(outs, False)
        rows.append((name, "pass" if not f else "FAIL", ", ".join(f)))
        ok &= not f

    control("reference", ex.cpu)
    for k in [k for k in MU.KINDS if MU.applies(k, ex.want)]:
        f = verdict({n: MU.mutate(o, k) for n, o in ex.cpu.items()}, True)
        rows.append((k, "caught" if f else "SLIPPED", ", ".join(f)))
        ok &= bool(f)
    if ex.kind == "float":
        control(
            "bf16 inputs + output",
            {x.name: MU.bf16(run_cpu(ex.cpu_step, x, [MU.bf16(t) for t in x.inputs])) for x in ex.cases},
        )
        control("bf16 everywhere", ex.model)
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
