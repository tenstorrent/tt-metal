# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""model/decode_model.py DecodeModel against the corrected reference (PLAN.md section 54): generation from an EMPTY
context, nothing of the selection, the query or the cache from the host.

Every run: HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved, and HY4_MOCK=1 with the
same environment before the board job.

    HY4_DM_LAYERS    6 (default) | 78            the depth (a decode golden of that depth must exist)
    HY4_DM_PROMPT    31 (default) | 1000 | 1022 | 3000   the golden's prompt length
    HY4_DM_TRACE     0 (default): every step program by program | 1: the step as one trace (DecodeModel(trace=True))
    HY4_DM_ROWS      the rows that run the conversation in test_golden / test_free (default all 8; the others idle)
    HY4_DM_STEPS     test_golden: only the first steps (3: a first run on the board); HY4_DM_PROBE_FIRST: how many of
                     the first steps run program by program with the checks between programs (default 2);
                     HY4_DM_PROBE_EVERY k: untraced, every k-th golden step program by program (default 1: all);
                     HY4_DM_PROBE_AT "127,128,129": these steps program by program in either form (position 128 is
                     the first on chip 1: a P31 conversation never leaves chip 0)

test_golden   the golden's prompt fed from position 0 through ``step``, then its decode steps teacher-forced (the
              golden's own input ids). Checked at every probed step: every status; untraced, after each full-indexer
              layer's cut, the device's selection (exactly positions 0 .. p while the context fits the top-k; against
              the golden's recorded picks and the reference indexer's near-ties above it); at the golden's steps the
              logits and the greedy token against the golden's.
test_free     the prompt, then free-running greedy generation next to the golden's continuation.
test_rows     8 rows with different prompts, staggered starts, a row that is reset in the middle, a row whose
              conversation crosses chips (HY4_DM_LONG tokens, default 300: positions on chips 0, 1, 2) and one idle
              row in the same passes; then every row ALONE (the others idle) after a reset: tokens and logits of
              each row against its run among the others, bit for bit; the golden's row against the golden.

test_rewind   reset_row(row, keep=k): a row fed A[0 .. n) is taken back to position k and continued with other
              tokens B; then, after a full reset, the same row is fed A[0 .. k) + B afresh. The continuation's tokens
              and logits must be equal bit for bit, and the first step after the rewind, launched program by
              program, must select exactly positions 0 .. k on every full layer (nothing above k is visible: not
              the index keys in the scan, not a selection table the step before left, not the latent rows). A
              second row runs another conversation through both phases; a third is rewound to its own next
              position (nothing changes) and to 0.
test_open     DecodeModel.open(...) / close() without the test's mesh fixture, as a server starts it: the serving
              side's warm-up (two passes of all 8 rows on token 0 at positions 0 and 1, the second with
              want_logits, then a reset of every row), a short conversation, close.

Lines start with HY4_MF.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.model.decode_model import DecodeModel, mesh_requirements
from models.experimental.hy4_preview.fused.model.program import is_mock
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin, rope_pairing
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
LAYERS = int(os.environ.get("HY4_DM_LAYERS", "6"))
PROMPT = int(os.environ.get("HY4_DM_PROMPT", "31"))
TRACE = os.environ.get("HY4_DM_TRACE", "0") == "1"
ROWS_ON = [int(v) for v in os.environ.get("HY4_DM_ROWS", "0,1,2,3,4,5,6,7").split(",")]
MAX_POSITIONS = int(os.environ.get("HY4_DM_MAX_POSITIONS", "65536"))
TOPK = 2048
LONG = int(os.environ.get("HY4_DM_LONG", "300"))  # test_rows: the tokens of the row whose conversation crosses chips
TRACE_REGION = 64 << 20
# the host control of this form (tests/host_own_rows.py: the reference on its own nearest-packed rows with the device's
# expert formats): the logits of the golden's steps per variant
CONTROL = os.environ.get("HY4_DM_CONTROL") or f"generated/hy4_preview/logs/mf_own_rows_L{LAYERS}_P{PROMPT}.pt"
CONTROL_VARIANT = os.environ.get("HY4_DM_CONTROL_VARIANT", "near.bfp4")
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)


def golden_of(prompt=PROMPT):
    g = hf_harness.find_decode_golden(LAYERS, prompt)
    if g is None:
        pytest.skip(f"no {LAYERS}-layer decode golden of a {prompt}-token prompt under {hf_harness.GOLDEN_ROOT}")
    assert g["meta"]["rope_pairing"] == rope_pairing() == "interleaved", (g["meta"]["rope_pairing"], rope_pairing())
    return g


CHECK_EVERY = int(
    os.environ.get("HY4_DM_CHECK_EVERY", "64")
)  # DecodeModel(check_every=): status() after every n-th step


def build(mesh_device, trace=TRACE):
    return DecodeModel(
        mesh_device, layers=LAYERS, max_positions=MAX_POSITIONS, trace=trace, keep_logits=True, check_every=CHECK_EVERY
    )


def label_of(what):
    return (
        f"{what} [{LAYERS} layers, P{PROMPT}, {'ONE trace a step' if TRACE else 'untraced'}, pairing {rope_pairing()}]"
    )


class Probe:
    """What an untraced step is checked for between its programs."""

    def __init__(self, model, label, positions, active, picks_of=None):
        """positions[r]: the rows' positions of this step. picks_of(layer) -> per row the reference's picked positions
        (a sorted int64 tensor) or None: for a row whose context is above the top-k."""
        self.model, self.st, self.label = model, model.st, label
        self.positions, self.active, self.picks_of = positions, active, picks_of
        self.failed, self.shared = [], {}

    def __call__(self, name):
        st = self.st
        kind = name.split("[")[0]
        layer = int(name.split("[")[1][:-1]) if "[" in name else None
        if kind == "merge":
            flags = st.merge.read()["flags"]
            if bool((flags & 2).any()):
                self.failed.append(
                    f"{name}: the merge reports a wait that ran out, flags {sorted(set(flags.flatten().tolist()))}"
                )
        elif kind == "cut":
            picked = self.model.read_selection()
            for r in range(8):
                p = self.positions[r]
                if not self.active[r]:
                    if picked[r].numel():
                        self.failed.append(f"{name}: idle row {r} has {picked[r].numel()} picks")
                    continue
                if picked[r].unique().numel() != picked[r].numel() or (picked[r].numel() and int(picked[r].max()) > p):
                    self.failed.append(
                        f"{name} row {r}: a position twice or beyond its own {p}: {picked[r][-4:].tolist()}"
                    )
                    continue
                if p + 1 <= TOPK:
                    if picked[r].tolist() != list(range(p + 1)):
                        missing = sorted(set(range(p + 1)) - set(picked[r].tolist()))
                        self.failed.append(
                            f"{name} row {r} at position {p}: {picked[r].numel()} picks, missing {missing[:8]} (every position up to its own is wanted)"
                        )
                else:
                    if picked[r].numel() != TOPK:
                        self.failed.append(f"{name} row {r} at position {p}: {picked[r].numel()} picks of {TOPK}")
                    if self.picks_of is not None:
                        ref = self.picks_of(layer, r)
                        both = len(set(ref.tolist()) & set(picked[r].tolist()))
                        self.shared.setdefault(layer, {})[r] = (
                            both,
                            bool((picked[r] == p).any()),
                            bool((ref == p).any()),
                            picked[r],
                        )
        elif kind == "p2":
            status = {**st.a.status(), **{("full",) + k: v for k, v in st.af.t.status().items()}}
            if status:
                self.failed.append(f"{name}: words not at rest: {str(status)[:400]}")
        elif kind == "mlp":
            status = st.layers[layer].mlp.status()
            if status:
                self.failed.append(f"{name}: words not at rest: {str(status)[:400]}")
        elif kind == "head":
            status = self.model.status()
            if status:
                self.failed.append(f"after the step: {str(status)[:400]}")
        assert not self.failed, f"{self.label}: " + "; ".join(self.failed[:6])


def reference_scores(g, layer, j):
    """[1, n] the reference indexer's scores of the golden's step j on full layer ``layer`` over its cached keys
    (reference/indexer.py Hy4Indexer on the golden's q_resid and attn_in), n = position + 1."""
    from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig

    _, _, weights = PP.checkpoint_layer(layer)
    indexer = Hy4Indexer(IndexerConfig()).eval().load_weights(weights)
    hf = g["layers"][layer]
    p = g["meta"]["positions"][j]
    cos, sin = rope_cos_sin(torch.tensor([p]), AttentionConfig())
    with torch.no_grad():
        q = indexer.queries(hf["q_resid"][j : j + 1].float(), cos, sin)
        w = indexer.head_weights(hf["attn_in"][j : j + 1].float())
        return indexer.scores(q, w, g["cache"][layer]["index_key"].float()[: p + 1])


def log_selection(g, j, probe, label):
    """A golden step above the top-k: per full layer the device's picks against the golden's recorded ones, the
    differences split into near-ties of the reference's cut and real ones (reference/indexer.py compare_selection)."""
    from models.experimental.hy4_preview.reference.indexer import compare_selection

    p = g["meta"]["positions"][j]
    for layer, rows in sorted(probe.shared.items()):
        scores = reference_scores(g, layer, j)
        top = scores[0].topk(TOPK).values
        tol = (
            max(abs(float(top[-1])), float(top[0] - top[-1])) / 64
        )  # (the device ranks BF16 scores: layer_attn PLAN 21.4)
        rec = torch.zeros(1, p + 1, dtype=torch.bool)
        picks = g["layers"][layer]["topk_indices"][j].long()
        rec[0, picks[(picks >= 0) & (picks <= p)]] = True
        out = []
        for r, (both, own_dev, own_ref, picked) in sorted(rows.items()):
            mine = torch.zeros(1, p + 1, dtype=torch.bool)
            mine[0, picked] = True
            c = compare_selection(scores, rec, mine, TOPK, tol)
            out.append((r, both, c["near_tie"], c["real"], own_dev, own_ref))
        logger.info(
            f"HY4_MF {label} golden step {j} (position {p}) full layer {layer}: per row (row, picks shared with the golden's recorded "
            f"{TOPK}, differences that are near-ties of the reference's cut, real ones, own position picked: device, golden) {out}; "
            f"tolerance {tol:.3e} at a cut of {float(top[-1]):+.4f}"
        )
        assert all(
            o[1] >= 0.9 * TOPK for o in out
        ), f"{label} step {j} layer {layer}: picks shared with the golden {[o[1] for o in out]}"


def golden_picks(g, j):
    def of(layer, row):
        p = g["meta"]["positions"][j]
        picks = g["layers"][layer]["topk_indices"][j].long()
        return torch.unique(picks[(picks >= 0) & (picks <= p)])

    return of


def control_logits():
    """[steps, vocab] of the host control's variant, or None when the control has not been run for this golden."""
    if not os.path.isfile(CONTROL):
        return None
    kept = torch.load(CONTROL, weights_only=False)
    return kept["logits"].get(CONTROL_VARIANT)


def compare_step(model, g, j, got, rows, table, control=None):
    """The golden's step j: device logits and tokens of ``rows`` against the golden's (and the host control's)."""
    logits = model.read_logits()
    gl, want = g["logits"][j].float(), int(g["cached_logits_argmax"][j])
    top2 = gl.topk(2).values
    for r in rows:
        assert bool(torch.isfinite(logits[r]).all()), f"golden step {j} row {r}: the logits are not finite"
        assert (
            int(logits[r].argmax()) == got[r]
        ), f"golden step {j} row {r}: id {got[r]} is not the first maximum of the device's logits"
        dev2 = logits[r].topk(2).values
        table.append(
            {
                "step": j,
                "row": r,
                "pcc": pcc(gl, logits[r]),
                "token": got[r],
                "want": want,
                "margin": float(top2[0] - top2[1]),
                "dev_margin": float(dev2[0] - dev2[1]),
                "rank": int((gl > gl[got[r]]).sum()),
                "ctl_token": None if control is None else int(control[j].argmax()),
                "ctl_pcc": None if control is None else pcc(control[j], logits[r]),
            }
        )


def summarise(table, label):
    rows = sorted({t["row"] for t in table})
    steps = sorted({t["step"] for t in table})
    wrong = [t for t in table if t["token"] != t["want"]]
    per_step = [min(t["pcc"] for t in table if t["step"] == j) for j in steps]
    logger.info(
        f"HY4_MF {label} GOLDEN STEPS: {len(steps)} steps x {len(rows)} rows: greedy token = the golden's on {len(table) - len(wrong)} of "
        f"{len(table)}; logits pcc against the golden's: min {min(t['pcc'] for t in table):.6f}, mean "
        f"{sum(t['pcc'] for t in table) / len(table):.6f}; per step (min over rows) {[round(v, 4) for v in per_step]}"
    )
    logger.info(
        f"HY4_MF {label} GOLDEN STEPS differing (step, row, golden's token, device's, the golden's top-1 margin in logits, the "
        f"device's own margin, the device's token's rank in the golden's logits): "
        f"{[(t['step'], t['row'], t['want'], t['token'], round(t['margin'], 3), round(t['dev_margin'], 3), t['rank']) for t in wrong][:40]}"
    )
    if table[0]["ctl_token"] is not None:
        by_formats = [t for t in wrong if t["ctl_token"] != t["want"]]
        logger.info(
            f"HY4_MF {label} GOLDEN STEPS against the HOST CONTROL ({CONTROL_VARIANT}: the reference on its own nearest-packed rows "
            f"with the device's expert formats, {CONTROL}): the device's token = the control's on "
            f"{sum(t['token'] == t['ctl_token'] for t in table)} of {len(table)}; the control's = the golden's on "
            f"{sum(t['ctl_token'] == t['want'] for t in table) // len(rows)} of {len(steps)} steps; device logits against the control's pcc min "
            f"{min(t['ctl_pcc'] for t in table):.6f} mean {sum(t['ctl_pcc'] for t in table) / len(table):.6f}; of the {len(wrong)} tokens that "
            f"differ from the golden's, the control loses the golden's token too on {len(by_formats)} (a format loss); NOT explained by the "
            f"control (step, row, golden, device, control) {[(t['step'], t['row'], t['want'], t['token'], t['ctl_token']) for t in wrong if t['ctl_token'] == t['want']][:40]}"
        )
    else:
        logger.info(f"HY4_MF {label}: no host control at {CONTROL} (tests/host_own_rows.py)")
    same = [
        all(
            t["token"] == table[i - i % len(rows)]["token"]
            for t in table[i - i % len(rows) : i - i % len(rows) + len(rows)]
        )
        for i in range(0, len(table), len(rows))
    ]
    logger.info(
        f"HY4_MF {label} GOLDEN STEPS: steps on which every row gave the same token (the rows run the same conversation): {sum(same)} of {len(same)}"
    )
    return wrong


def check_runs(model, steps, label):
    """Every step ran every layer's programs: the canaries (a kernel on each head but the root in every P2 and every
    MLP program) count their runs. A step that was recorded and not run (a trace capture) is one short."""
    heads = len(model.st.read_canary())
    runs = model.st.canary_summary()
    want = steps * model.layers * heads
    assert (
        runs["p2"]["runs"] == want and runs["mlp"]["runs"] == want
    ), f"{label}: {steps} steps x {model.layers} layers x {heads} heads = {want} runs of each canary, counted {runs['p2']['runs']} / {runs['mlp']['runs']}"


def idle_mix(tokens_by_row, positions_by_row):
    return [tokens_by_row.get(r) for r in range(8)], [positions_by_row.get(r, 0) for r in range(8)]


@galaxy
def test_golden(mesh_device):
    g = golden_of()
    N, steps, ids = g["meta"]["prompt_len"], g["meta"]["steps"], g["input_ids"]
    label = label_of("decode model, the golden fed from position 0")
    model = build(mesh_device)
    for r in range(8):
        model.reset_row(r)
    feed = lambda s: idle_mix({r: int(ids[s]) for r in ROWS_ON}, {r: s for r in ROWS_ON})
    active = [r in ROWS_ON for r in range(8)]
    if is_mock():
        model.step(*feed(0))
        model.step_probed(*feed(1), after=lambda name: None)
        logger.info(
            f"HY4_MF {label}: compiled on the mock: {len(model.programs) + 2} programs a step, program-cache entries "
            f"{mesh_device.num_program_cache_entries()}, L1 tensors {model.st.l1_bytes()} B per core; rows {ROWS_ON}"
        )
        return
    # HY4_DM_STEPS: only the first steps (a first run on the board: 3 steps, each program by program)
    total = min(N - 1 + steps, int(os.environ.get("HY4_DM_STEPS", "1000000")))
    probe_until = int(os.environ.get("HY4_DM_PROBE_FIRST", "2"))
    probe_every = int(
        os.environ.get("HY4_DM_PROBE_EVERY", "1")
    )  # (untraced: every k-th golden step program by program)
    # (steps launched program by program whatever the form: the positions at which a row's stripe moves to the next chip)
    probe_at = {int(v) for v in os.environ.get("HY4_DM_PROBE_AT", "").split(",") if v}
    table, t0, times, kept_logits = [], time.perf_counter(), [], []
    control = control_logits()
    for s in range(total):
        j = s - (N - 1)
        tokens, positions = feed(s)
        # (a golden step above the top-k is launched program by program in the traced form too: its selections per layer)
        probed = s < probe_until or s in probe_at or (j >= 0 and j % probe_every == 0 and (not TRACE or s + 1 > TOPK))
        t1 = time.perf_counter()
        if probed:
            probe = Probe(model, f"{label} step {s}", positions, active, golden_picks(g, j) if j >= 0 else None)
            got = model.step_probed(tokens, positions, after=probe)
        else:
            got = model.step(tokens, positions)
            probe = None
        times.append(time.perf_counter() - t1)
        assert [v is not None for v in got] == active
        if s in (0, 1, 2) or s == N - 2:
            logger.info(
                f"HY4_MF {label} step {s} ({'probed' if probed else 'plain'}, {times[-1]:.2f} s): tokens out {got}"
            )
        if j >= 0:
            if probe is not None and probe.shared:
                log_selection(g, j, probe, label)
            elif probe is None:
                # (a step that was not launched program by program: only the LAST full layer's table can be read)
                picked = model.read_selection()
                if s + 1 <= TOPK:
                    bad = [r for r in range(8) if picked[r].tolist() != (list(range(s + 1)) if active[r] else [])]
                    assert (
                        not bad
                    ), f"{label} golden step {j}: the last full layer's selection of rows {bad} is not positions 0 .. {s}"
                else:
                    ref = golden_picks(g, j)(model.full[-1], 0)
                    logger.info(
                        f"HY4_MF {label} golden step {j} (position {s}), the last full layer ({model.full[-1]}): picks shared with the "
                        f"golden's recorded {[len(set(ref.tolist()) & set(picked[r].tolist())) for r in ROWS_ON]} of {TOPK}"
                    )
            compare_step(model, g, j, got, ROWS_ON, table, control)
            kept_logits.append(model.read_logits()[ROWS_ON[0]].clone())
        if s % 256 == 255:
            logger.info(f"HY4_MF {label}: {s + 1} of {total} steps, {time.perf_counter() - t0:.0f} s")
    status = model.status()
    if not table:
        logger.info(
            f"HY4_MF {label}: the first {total} steps only, each selection exact, tokens out {got}; status {status or 'at rest'}; canary {model.st.canary_summary()}"
        )
        assert not status, status
        return
    wrong = summarise(table, label)
    plain = [t for i, t in enumerate(times) if i >= 3]
    logger.info(
        f"HY4_MF {label}: {total} steps ({N - 1} prompt steps and {steps} golden steps) in {time.perf_counter() - t0:.0f} s on the host "
        f"(a step {1e3 * sum(plain) / max(len(plain), 1):.0f} ms with its host writes and reads; not a latency figure); status "
        f"{status or 'at rest'}; canary {model.st.canary_summary()}; rows {ROWS_ON}"
    )
    assert not status, status
    check_runs(model, total, label)
    if ROWS_ON == list(range(8)) and total == N - 1 + steps:
        # row 0's logits of the golden's steps, kept per form: the traced job against the untraced one, bit for bit
        name = (
            lambda traced: f"generated/hy4_preview/logs/mf_dm_logits_L{LAYERS}_P{PROMPT}_{'traced' if traced else 'untraced'}.pt"
        )
        mine = [t["token"] for t in table if t["row"] == 0]
        torch.save({"logits": torch.stack(kept_logits), "tokens": mine}, name(TRACE))
        if os.path.isfile(name(not TRACE)):
            old = torch.load(name(not TRACE), weights_only=False)
            same = [bool(torch.equal(a, b)) for a, b in zip(old["logits"], kept_logits)]
            logger.info(
                f"HY4_MF {label} against the {'untraced' if TRACE else 'traced'} job ({name(not TRACE)}): row 0's logits equal in every "
                f"bit on {sum(same)} of {len(same)} golden steps, the same token on {sum(a == b for a, b in zip(old['tokens'], mine))}; "
                f"largest difference {max(float((a - b).abs().max()) for a, b in zip(old['logits'], kept_logits)):.4f}"
            )
    # (78 layers: the formats' own control is at 0.80 against the golden on its worst step, PLAN.md section 55; with a
    # control the device is held to IT)
    if control is not None:
        assert min(t["ctl_pcc"] for t in table) > 0.9, "logits far from the host control's of the device's formats"
    else:
        assert min(t["pcc"] for t in table) > 0.7, "logits far from the golden's"


@galaxy
def test_free(mesh_device):
    """The prompt, then greedy generation on the device's own tokens for the golden's number of steps."""
    g = golden_of()
    N, steps, ids = g["meta"]["prompt_len"], g["meta"]["steps"], g["input_ids"]
    label = label_of("decode model, free-running generation")
    model = build(mesh_device)
    for r in range(8):
        model.reset_row(r)
    if is_mock():
        model.step(*idle_mix({r: int(ids[0]) for r in ROWS_ON}, {r: 0 for r in ROWS_ON}))
        logger.info(f"HY4_MF {label}: compiled on the mock")
        return
    last = {r: None for r in ROWS_ON}
    out = {r: [] for r in ROWS_ON}
    margins = []
    for s in range(N - 1 + steps):
        tokens = {r: int(ids[s]) if s < N else last[r] for r in ROWS_ON}
        got = model.step(*idle_mix(tokens, {r: s for r in ROWS_ON}))
        for r in ROWS_ON:
            last[r] = got[r]
            if s >= N - 1:
                out[r].append(got[r])
        if s >= N - 1:
            top2 = model.read_logits()[ROWS_ON[0]].topk(2).values
            margins.append(round(float(top2[0] - top2[1]), 3))
    status = model.status()
    check_runs(model, N - 1 + steps, label)
    want = [int(v) for v in g["cached_logits_argmax"][:steps]]
    gold_margin = [round(float(v[0] - v[1]), 3) for v in g["logits"][:steps].float().topk(2, dim=-1).values]
    for r in ROWS_ON:
        first = next((k for k in range(steps) if out[r][k] != want[k]), None)
        logger.info(
            f"HY4_MF {label} row {r}: {sum(a == b for a, b in zip(out[r], want))} of {steps} generated tokens are the golden's "
            f"continuation's; the first that differs: step {first}"
            + (
                ""
                if first is None
                else f" (golden {want[first]}, device {out[r][first]}; the golden's own top-1 margin there {gold_margin[first]})"
            )
        )
    logger.info(f"HY4_MF {label} row {ROWS_ON[0]} generated {out[ROWS_ON[0]]}")
    logger.info(f"HY4_MF {label} the golden's continuation {want}")
    logger.info(
        f"HY4_MF {label} top-1 margins per step: the golden's {gold_margin}; the device's (row {ROWS_ON[0]}) {margins}; status {status or 'at rest'}"
    )
    assert not status, status


def schedules(g, others):
    """Row -> [(pass at which it starts, token ids)]: conversations of a row in pass order. Row 4 stays idle."""
    ids = [int(v) for v in g["input_ids"]]
    alt = [[int(v) for v in o["input_ids"][:48]] for o in others]
    long_ids = [int(v) for v in others[2]["input_ids"]]
    return {
        0: [(0, ids)],  # the golden's conversation: prompt and continuation, teacher-forced
        1: [(0, alt[0][:40])],  # another prompt
        2: [(5, alt[1][:44])],  # starts five passes later
        3: [
            (11, long_ids[:LONG])
        ],  # a conversation that leaves chip 0 (positions 128 .. are on chip 1, 256 .. on chip 2)
        5: [(3, ids)],  # the golden's conversation again, three passes behind row 0
        6: [(0, ids[:31][::-1] + ids[:9])],
        7: [(0, alt[0][8:28]), (24, ids[:30])],  # a short conversation, then the row is reset and starts another
    }


def run_schedule(model, sched, passes, keep_logits=True, first=0):
    """The passes ``first`` .. ``passes`` - 1 of ``sched`` ({row: [(start, ids)]}; no conversation may be under way at
    ``first``); returns {(row, pass): (token out, logits)}."""
    for r in range(8):
        model.reset_row(r)
    out = {}
    for t in range(first, passes):
        tokens, positions = {}, {}
        for r, convs in sched.items():
            for start, ids in convs:
                if t == start:
                    model.reset_row(r)
                if start <= t < start + len(ids):
                    tokens[r], positions[r] = ids[t - start], t - start
        got = model.step(*idle_mix(tokens, positions))
        assert [v is not None for v in got] == [r in tokens for r in range(8)]
        logits = model.read_logits() if keep_logits and tokens else None
        for r in tokens:
            out[(r, t)] = (got[r], None if logits is None else logits[r].clone())
    return out


@galaxy
def test_rows(mesh_device):
    g = golden_of()
    others = [hf_harness.find_decode_golden(LAYERS, p) for p in (1000, 1022, 3000)]
    if any(o is None for o in others):
        pytest.skip("the other prompts' goldens (their token ids are the other rows' prompts) are not there")
    label = label_of("decode model, 8 rows of different conversations")
    sched = schedules(g, others)
    passes = max(start + len(ids) for convs in sched.values() for start, ids in convs)
    model = build(mesh_device)
    if is_mock():
        run_schedule(model, {r: [(0, c[0][1][:1])] for r, c in sched.items()}, 1, keep_logits=False)
        logger.info(
            f"HY4_MF {label}: compiled on the mock; {passes} passes, rows {sorted(sched)} with (start, tokens) {[(r, [(s, len(i)) for s, i in c]) for r, c in sched.items()]}"
        )
        return
    t0 = time.perf_counter()
    mixed = run_schedule(model, sched, passes)
    status = model.status()
    assert not status, f"{label}: after the mixed passes: {status}"
    check_runs(model, passes, label)
    logger.info(
        f"HY4_MF {label}: {passes} passes with rows (row, [(start pass, tokens)]) {[(r, [(s, len(i)) for s, i in c]) for r, c in sched.items()]}, "
        f"row 4 idle in every pass, in {time.perf_counter() - t0:.0f} s; status at rest"
    )
    # the golden's rows against the golden (row 0 from pass 0, row 5 three passes behind)
    N, steps = g["meta"]["prompt_len"], g["meta"]["steps"]
    for r, start in ((0, 0), (5, 3)):
        hits, low = 0, 1.0
        for j in range(steps):
            token, logits = mixed[(r, start + N - 1 + j)]
            hits += int(token == int(g["cached_logits_argmax"][j]))
            low = min(low, pcc(g["logits"][j].float(), logits))
        logger.info(
            f"HY4_MF {label} row {r} (the golden's conversation from pass {start}): {hits} of {steps} golden steps give the golden's token; logits pcc min {low:.6f}"
        )
    both = [
        (t, mixed[(0, t)][0], mixed[(5, t + 3)][0], bool(torch.equal(mixed[(0, t)][1], mixed[(5, t + 3)][1])))
        for t in range(len(sched[0][0][1]))
    ]
    logger.info(
        f"HY4_MF {label}: rows 0 and 5 run the same conversation three passes apart: the same token on {sum(a == b for _, a, b, _ in both)} of "
        f"{len(both)} positions, logits equal in every bit on {sum(e for _, _, _, e in both)}"
    )
    # every row alone, after a reset, the others idle
    result = {}
    for r in sorted(sched):
        # (its own passes only: from its first conversation's start to its last one's end)
        alone = run_schedule(
            model, {r: sched[r]}, max(s + len(i) for s, i in sched[r]), first=min(s for s, _ in sched[r])
        )
        status = model.status()
        assert not status, f"{label}: row {r} alone: {status}"
        keys = sorted(k for k in mixed if k[0] == r)
        assert keys == sorted(alone)
        tokens = sum(mixed[k][0] == alone[k][0] for k in keys)
        bits = sum(bool(torch.equal(mixed[k][1], alone[k][1])) for k in keys)
        worst = min(pcc(mixed[k][1], alone[k][1]) for k in keys)
        diff = max(float((mixed[k][1] - alone[k][1]).abs().max()) for k in keys)
        first = next((k[1] for k in keys if not torch.equal(mixed[k][1], alone[k][1])), None)
        result[r] = (len(keys), tokens, bits, worst, diff, first)
        logger.info(
            f"HY4_MF {label} row {r} ALONE against the same row among the others: {len(keys)} passes: the same token on {tokens}, logits "
            f"equal in every bit on {bits} (lowest pcc {worst:.7f}, largest difference {diff:.4f}; the first pass that differs: {first})"
        )
    logger.info(
        f"HY4_MF {label} ROWS (row: passes, same token, logits bit-equal, lowest pcc, largest difference, first differing pass) {result}; {time.perf_counter() - t0:.0f} s"
    )
    assert all(
        v[1] == v[0] and v[2] == v[0] for v in result.values()
    ), "a row's result depends on the other rows of its passes"


@galaxy
def test_rewind(mesh_device):
    g = golden_of()
    others = [hf_harness.find_decode_golden(LAYERS, p) for p in (1000, 1022)]
    if any(o is None for o in others):
        pytest.skip("the other prompts' goldens (their token ids are the other rows' prompts) are not there")
    label = label_of("decode model, a row rewound with reset_row(keep=)")
    ids = [int(v) for v in g["input_ids"]]
    A, B, K = ids[:40], [int(v) for v in others[0]["input_ids"][100:124]], 17
    C = [int(v) for v in others[1]["input_ids"][:80]]  # row 0's conversation, running through every pass
    model = build(mesh_device)
    for r in range(8):
        model.reset_row(r)
    if is_mock():
        model.step(*idle_mix({0: C[0], 2: A[0], 6: A[0]}, {0: 0, 2: 0, 6: 0}))
        model.reset_row(2, keep=1)
        logger.info(f"HY4_MF {label}: compiled on the mock")
        return
    passes = [0]

    def feed(tokens, positions, probe=False):
        """One pass: row 0 always runs the next token of C."""
        tokens, positions = dict(tokens), dict(positions)
        tokens[0], positions[0] = C[passes[0] % len(C)], passes[0] % len(C)
        if passes[0] % len(C) == 0:
            model.reset_row(0)
        passes[0] += 1
        tk, ps = idle_mix(tokens, positions)
        if probe:
            got = model.step_probed(
                tk, ps, after=Probe(model, f"{label} pass {passes[0]}", ps, [v is not None for v in tk])
            )
        else:
            got = model.step(tk, ps)
        return got, model.read_logits()

    # phase 1: row 2 runs A, is rewound to K and continues with B; row 6 runs A and is rewound to its own next
    # position (nothing changes) in the middle
    first = {}
    for p, token in enumerate(A):
        if p == 20:
            model.reset_row(6, keep=20)
        got, logits = feed({2: token, 6: token}, {2: p, 6: p}, probe=p < 1)
        first[("A", p)] = (got[2], logits[2].clone(), got[6], logits[6].clone())
    same_6 = sum(bool(torch.equal(v[1], v[3])) for v in first.values())
    model.reset_row(2, keep=K)
    assert model.next_position(2) == K
    for i, token in enumerate(B):
        got, logits = feed(
            {2: token}, {2: K + i}, probe=i < 2
        )  # (the first two steps after the rewind: every selection read)
        first[("B", i)] = (got[2], logits[2].clone())
    # phase 2: the same row afresh: A[0 .. K) + B
    model.reset_row(2)
    second = {}
    for p, token in enumerate(A[:K] + B):
        got, logits = feed({2: token}, {2: p})
        second[p] = (got[2], logits[2].clone())
    status = model.status()
    assert not status, status
    prefix = [bool(torch.equal(first[("A", p)][1], second[p][1])) for p in range(K)]
    cont = [
        (first[("B", i)][0] == second[K + i][0], bool(torch.equal(first[("B", i)][1], second[K + i][1])))
        for i in range(len(B))
    ]
    diff = max(float((first[("B", i)][1] - second[K + i][1]).abs().max()) for i in range(len(B)))
    # how far the continuation is from what the row gave at the same positions BEFORE the rewind (other tokens: it must differ)
    before = sum(bool(torch.equal(first[("A", K + i)][1], first[("B", i)][1])) for i in range(min(len(B), len(A) - K)))
    logger.info(
        f"HY4_MF {label}: row 2 fed {len(A)} tokens, rewound to position {K}, continued with {len(B)} other tokens; afresh the same row fed "
        f"the first {K} and those {len(B)}: the prefix's logits equal in every bit on {sum(prefix)} of {K} positions; the continuation: "
        f"the same token on {sum(c[0] for c in cont)} of {len(B)}, logits equal in every bit on {sum(c[1] for c in cont)} (largest "
        f"difference {diff:.4f}); positions at which the continuation's logits equal the ones the row had there before the rewind "
        f"(other tokens: none expected) {before}; row 6 rewound to its own next position in the middle: its logits equal row 2's "
        f"on {same_6} of {len(A)} positions (the same conversation on another row: reported, not asserted); row 0 another "
        f"conversation in all {passes[0]} passes; status at rest"
    )
    check_runs(model, passes[0], label)
    assert all(prefix) and all(
        c[0] and c[1] for c in cont
    ), "after reset_row(keep=) the row's continuation is not what a fresh feed gives"
    assert before == 0


def test_open():
    """DecodeModel.open / close as a server uses them (no mesh fixture of the test). Mock: the mesh is opened and the
    model built and compiled on it, then closed."""
    g = golden_of()
    N, steps, ids = g["meta"]["prompt_len"], g["meta"]["steps"], [int(v) for v in g["input_ids"]]
    label = label_of("decode model, open() / close()")
    t0 = time.perf_counter()
    model = DecodeModel.open(layers=LAYERS, keep_logits=True, max_positions=16_384, check_every=CHECK_EVERY)
    try:
        assert model.supports_logits and model.traced and model.max_positions == 16_384 and model.rows == 8
        t1 = time.perf_counter()
        # the serving side's warm-up: two passes of all 8 rows on token 0, the second with the logits, then a reset
        model.step([0] * 8, [0] * 8)
        out, logits = model.step([0] * 8, [1] * 8, want_logits=True)
        assert len(out) == 8 and tuple(logits.shape) == (8, model.vocab)
        for r in range(8):
            model.reset_row(r)
        t2 = time.perf_counter()
        if is_mock():
            logger.info(
                f"HY4_MF {label}: opened, built and compiled on the mock; mesh requirements {mesh_requirements()}"
            )
            return
        assert model.trace_id is not None, "the warm-up's second pass captures the trace"
        view = ttnn.get_memory_view(model.mesh, ttnn.BufferType.TRACE)
        # a conversation on row 3 beside idle rows: the golden's prompt and steps
        hits, low, times = 0, 1.0, []
        for s in range(N - 1 + steps):
            t3 = time.perf_counter()
            out, logits = model.step(*idle_mix({3: ids[s]}, {3: s}), want_logits=True)
            times.append(time.perf_counter() - t3)
            j = s - (N - 1)
            if j >= 0:
                hits += int(out[3] == int(g["cached_logits_argmax"][j]))
                low = min(low, pcc(g["logits"][j].float(), logits[3]))
        t4 = time.perf_counter()
        for _ in range(8):
            model.step(*idle_mix({3: ids[0]}, {3: model.next_position(3)}))
        plain = (time.perf_counter() - t4) / 8
        t5 = time.perf_counter()
        status = model.status()
        status_s = time.perf_counter() - t5
        logger.info(
            f"HY4_MF {label}: open + build {t1 - t0:.0f} s; the warm-up (pass 1 compiles every program, pass 2 captures the trace and is "
            f"its first replay; reset of every row) {t2 - t1:.1f} s; trace buffer {view.total_bytes_allocated_per_bank} B a bank of the "
            f"region; then row 3 alone from position 0 on the golden's tokens: {hits} of {steps} golden tokens, logits pcc min {low:.6f}; "
            f"a pass with want_logits {1e3 * sum(times) / len(times):.0f} ms on the host, without {1e3 * plain:.0f} ms (check_every "
            f"{model.check_every}: one status() read costs {1e3 * status_s:.0f} ms); status {status or 'at rest'}"
        )
        assert not status, status
    finally:
        model.close()
    logger.info(f"HY4_MF {label}: closed")
