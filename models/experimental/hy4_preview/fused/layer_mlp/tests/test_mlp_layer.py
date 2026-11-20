# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MLP half of a decoder layer as one program on the 8x4 mesh (fused/layer_mlp/program.py), checkpoint weights of
layers 1 to 5, against the HF golden of a 128-token prompt (reference/hf_harness.py), 8 golden token rows per step.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo"   (root alone)
    ... -k "rung and pair_row", "rung and pair_col", "rung and first"                    (one step, layer 1, 2 / 2 / 32 chips)
    pytest <this file> -x -k "layers"                                         (layers 1 to 5)
    pytest <this file> -x -k "trace"

A step: the golden's streams after the attention sublayer (streams_mid) go into the root's site; the program leaves
x on every chip, the picks, the partial outputs and F; a post-only program on the root then updates the streams.
Checked per step (every number is logged; lines start with HY4_MK8):
  words      nothing left in any status word or counter of any chip
  y, 1/RMS   the site's pre-mix against the golden's mlp_hc
  x          the root's BF16 x against the golden's mlp_in; bit-equal in every chip's hub buffer
  picks      identical on all chips; against a host router on the device's own x (reference/moe.py route,
             fused/moe/layout.py route_golden): the rows with exactly the host's 8 experts are counted, and a row may
             differ only where the 8th and 9th scores are closer than the device's scores are to the host's (the
             worst device pick at most PICK_TOLERANCE below the host's cut); agreement with HF's own picks is reported
  partials   every chip's partial against exact arithmetic on its BFP4 share for the device's picks
  reduce     F against the host model of the two-stage sum of the 32 device partials: bit-equal
  F          (a) against the reference on the device's picks with the checkpoint's weights (PCC, gain, max abs),
             (b) against exact arithmetic on the BFP4 weights (two BFP4 terms for the shared expert): what is left
             is the LoFi matmuls and the BF16 hand-overs; and against the golden's mlp_out (HF's own picks)
  streams    after the post program, against the golden's streams_out; the layer update's PCC and gain
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective.collective import to_bf16
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_mlp.program import E, HIDDEN, MlpLayer, is_mock
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 64 << 20
I = L.INTERMEDIATE
PICK_TOLERANCE = 2e-3

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)


def gain(ref, dev):
    ref, dev = ref.double().flatten(), dev.double().flatten()
    return float((ref * dev).sum() / (ref * ref).sum())


def site_params(layer, site="hc_mlp_layer"):
    prefix = f"model.layers.{layer}.{site}.hc_pre."
    return tuple(hy4_tensors([prefix + "hc_fn", prefix + "hc_base", prefix + "hc_scale"]))


class Weights:
    """The checkpoint's MLP-half weights of a layer on the host, and the BFP4 values the device computes with."""

    def __init__(self, layer):
        self.layer = layer
        prefix = f"model.layers.{layer}."
        self.gamma, self.router_w, self.router_b = hy4_tensors(
            [
                prefix + "post_attention_layernorm.weight",
                prefix + "mlp.gate.weight",
                prefix + "mlp.gate.e_score_correction_bias",
            ]
        )
        self.site = site_params(layer)
        self.gate_up, self.down = prepare.layer_experts(
            layer, shared=True, shared_terms=1
        )  # BF16, entry 256 the shared
        self._rounded = {}

    def exact(self, e):
        """(gate_up [4096, 6144], down [6144, 2048]) of entry e, fp32 (the checkpoint's BF16 values)."""
        return self.gate_up[e].float(), self.down[e].float()

    def rounded(self, e):
        """The same as the device holds it: BFP4 tiles of the transposed matrices; the shared expert as two terms."""
        if e not in self._rounded:
            value = L.two_terms_value if e >= E else (lambda w: L.bfp4_round(w.transpose(1, 2)).transpose(1, 2))
            self._rounded[e] = (value(self.gate_up[e : e + 1].float())[0], value(self.down[e : e + 1].float())[0])
        return self._rounded[e]

    def moe(self, x, ids, w, rounded=False, per_chip=False):
        """The MoE output [T, 6144] for per-row picks ids [T, 8] with weights w [T, 8], plus the shared expert
        (reference/moe.py Hy4MoE: routed with the clamp, shared without), in fp32 matmuls summed in fp64.
        per_chip: also every chip's partial [32, T, 6144] (its 1/32 of the intermediate channels)."""
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64)
        parts = torch.zeros(32, x.shape[0], HIDDEN, dtype=torch.float64) if per_chip else None
        xf = x.float()
        step = I // 32
        for e in sorted(set(ids.flatten().tolist())) + [E]:
            gu, dn = self.rounded(e) if rounded else self.exact(e)
            if e < E:
                rows = (ids == e).any(dim=-1).nonzero().flatten()
                we = (w * (ids == e))[rows].sum(dim=-1, keepdim=True).float()
                limit = 10.0
            else:
                rows, we, limit = torch.arange(x.shape[0]), 1.0, float("inf")
            pre = xf[rows] @ gu.T
            act = L.swiglu(pre[:, :I], pre[:, I:], limit) * we
            out[rows] += (act @ dn.T).double()
            if per_chip:
                for c in range(32):
                    parts[c, rows] += (act[:, c * step : (c + 1) * step] @ dn[:, c * step : (c + 1) * step].T).double()
        return (out.float(), parts.float()) if per_chip else out.float()


def two_stage_sum(layer_obj, partials):
    """The host model of the reduce: per mesh row the Float32 sum of the chips' partials in column order, rounded to
    BFloat16 (ties away from zero), then the same over the rows."""
    topo = layer_obj.topo
    rows = []
    for r in topo.rows:
        total = None
        for c in topo.members[r]:
            total = partials[c].clone() if total is None else total + partials[c]
        rows.append(to_bf16(total, "away"))
    total = rows[0].clone()
    for other in rows[1:]:
        total = total + other
    return to_bf16(total, "away")


def build(mesh_device, layer, timing=False, **kw):
    w = Weights(layer)
    mlp = MlpLayer(mesh_device, timing=timing, **kw)
    mlp.set_layer(layer, [w.site], 0, w.gamma, w.router_w, w.router_b)
    return mlp, w


def check_step(mlp, w, hf, rows, label, post=True, per_chip=True, full=True):
    """Everything of the module docstring for the step whose input rows are ``rows`` of the golden. The program
    (and the post program) have run."""
    cfg = mlp.cfg
    failed = []
    status = mlp.status()
    if status:
        failed.append(f"words not at rest: {status}")
    streams_mid = hf["streams_mid"][rows].float()
    root = mlp.read_root()
    r_y, _, r_post, r_stat = golden_pre(cfg, *w.site, streams_mid)
    logger.info(
        f"HY4_MK8 {label} pre-mix: y vs golden mlp_hc pcc={pcc(hf['mlp_hc'][rows].float(), root['y']):.7f}, vs the "
        f"reference on the same streams max_abs={(r_y - root['y']).abs().max():.3e}; 1/RMS max_rel="
        f"{((root['inv_rms'] - r_stat).abs() / r_stat).max():.3e}"
    )
    x = root["x"]
    r_x = rms_norm(r_y, w.gamma.float(), cfg.norm_eps)
    logger.info(
        f"HY4_MK8 {label} x: vs golden mlp_in pcc={pcc(hf['mlp_in'][rows].float(), x):.7f}; vs the reference norm "
        f"max_abs={(r_x - x).abs().max():.3e} (BF16 rounding alone {(r_x - r_x.to(torch.bfloat16).float()).abs().max():.3e})"
    )
    chips = {c: mlp.read_chip(c) for c in mlp.topo.chips}
    x_ok = sum(torch.equal(v["x"], x) for v in chips.values())
    first = chips[mlp.topo.root]
    picks_ok = sum(
        torch.equal(v["ids"], first["ids"]) and torch.equal(v["weights"], first["weights"]) for v in chips.values()
    )
    if x_ok != len(chips):
        failed.append(f"x bit-equal on {x_ok} of {len(chips)} chips")
    if picks_ok != len(chips):
        failed.append(f"picks equal to the root's on {picks_ok} of {len(chips)} chips")
    ids, weights = first["ids"].long(), first["weights"]
    r_scores, r_ids, r_w = L.route_golden(x, w.router_w, w.router_b)
    exact_rows = sum(ids[r].tolist() == r_ids[r].tolist() for r in range(L.ROWS))
    same_sets = sum(set(ids[r].tolist()) == set(r_ids[r].tolist()) for r in range(L.ROWS))
    hf_sets = sum(set(ids[r].tolist()) == set(hf["router_indices"][rows][r].tolist()) for r in range(L.ROWS))
    biased = r_scores + w.router_b.float()
    cut = biased.topk(8, dim=-1).values[:, -1]
    margin = (cut - biased.topk(9, dim=-1).values[:, -1]).min().item()
    below = (cut - biased.gather(1, ids).min(dim=-1).values).max().item()  # the worst device pick under the host's cut
    score_err = (first["scores"] - r_scores).abs().max().item()
    r_w_for = r_scores.gather(1, ids)
    r_w_for = r_w_for / r_w_for.sum(dim=-1, keepdim=True) * 2.827
    w_off = ((weights - r_w_for).abs() / r_w_for).max().item()
    union = sorted(set(ids.flatten().tolist()))
    logger.info(
        f"HY4_MK8 {label} picks: x bit-equal on {x_ok}/{len(chips)} chips, picks identical on {picks_ok}/{len(chips)}; "
        f"against a host router on the device's x: {same_sets}/8 rows the same 8 experts ({exact_rows}/8 in the same "
        f"order), smallest 8th-9th margin {margin:.2e}, the worst device pick {below:.2e} below the host's cut, device "
        f"scores within {score_err:.2e} of the host's; weights within {w_off:.2e} (BF16); {hf_sets}/8 rows have HF's "
        f"own picks; union of {len(union)} experts"
    )
    out_picks = dict(rows_exact=same_sets, margin=margin, below=below, score_err=score_err)
    # a pick may differ from the host router's only where the 8th and 9th scores are closer than the device's scores
    # are to the host's (tests/test_decoder_layer.py PICK_TOLERANCE)
    if below > PICK_TOLERANCE:
        failed.append(f"a device pick scores {below:.2e} below the host router's cut")

    partials = {c: v["out"] for c, v in chips.items()}
    f = root["f"]
    model = two_stage_sum(mlp, partials)
    wrong = int((model != f).sum())
    logger.info(
        f"HY4_MK8 {label} reduce: F vs the host model of the two-stage sum of the device partials: {wrong} of {f.numel()} elements differ"
    )
    if wrong:
        failed.append(f"F differs from the two-stage sum of the partials in {wrong} elements")
    ref_bfp4 = None
    if per_chip:
        ref_bfp4, parts = w.moe(x, ids, weights, rounded=True, per_chip=True)
        scores = {
            c: (pcc(parts[mlp.topo.chip(c)], partials[c]), gain(parts[mlp.topo.chip(c)], partials[c]))
            for c in mlp.topo.chips
        }
        worst = min(scores, key=lambda c: scores[c][0])
        pccs = sorted(s[0] for s in scores.values())
        gains = sorted(s[1] for s in scores.values())
        logger.info(
            f"HY4_MK8 {label} partials vs exact arithmetic on each chip's BFP4 share: pcc min {pccs[0]:.5f} (chip "
            f"{worst}) median {pccs[len(pccs) // 2]:.5f}; gain {gains[0]:.4f} to {gains[-1]:.4f}"
        )
        if pccs[0] < 0.97:
            failed.append(f"partial of chip {worst}: pcc {pccs[0]:.5f}")

    out = {"union": len(union), "picks": out_picks}
    if not full:  # part of the mesh: F is the sum of those chips' shares only, checked chip by chip above
        if post:
            streams = mlp.read_streams()
            mine = golden_post(cfg, streams_mid, f, r_post)
            logger.info(
                f"HY4_MK8 {label} streams after the post vs the reference post of the device's own F: max_abs={(mine - streams).abs().max():.3e}"
            )
            if (mine - streams).abs().max() > 1e-3 * streams.abs().max():
                failed.append("streams are not the post-mix of the device's F")
        assert not failed, f"{label}: " + "; ".join(failed)
        return out
    ref_dev = w.moe(x, ids, r_w_for)  # the reference's arithmetic and weights on the device's picks
    if ref_bfp4 is None:
        ref_bfp4 = w.moe(x, ids, weights, rounded=True)  # exact arithmetic on what the device holds
    for name, ref in (
        ("reference on the device's picks", ref_dev),
        ("exact arithmetic on the BFP4 weights", ref_bfp4),
        ("golden mlp_out (HF's picks)", hf["mlp_out"][rows].float()),
    ):
        out[name] = (pcc(ref, f), gain(ref, f), float((ref - f).abs().max()))
        logger.info(
            f"HY4_MK8 {label} F vs {name}: pcc={out[name][0]:.5f} gain={out[name][1]:.4f} max_abs={out[name][2]:.3e} "
            f"(rms of the reference {ref.pow(2).mean().sqrt():.3e})"
        )
    logger.info(
        f"HY4_MK8 {label} what BFP4 costs alone (host): exact arithmetic on the BFP4 weights vs the reference "
        f"pcc={pcc(ref_dev, ref_bfp4):.5f} gain={gain(ref_dev, ref_bfp4):.4f}"
    )
    if out["exact arithmetic on the BFP4 weights"][0] < 0.97:
        failed.append(
            f"F vs exact arithmetic on the BFP4 weights: pcc {out['exact arithmetic on the BFP4 weights'][0]:.5f}"
        )
    if post:
        streams = mlp.read_streams()
        gold = hf["streams_out"][rows].float()
        mine = golden_post(cfg, streams_mid, f, r_post)
        upd = lambda s: s - streams_mid
        out["streams"] = (pcc(gold, streams), pcc(upd(gold), upd(streams)), gain(upd(gold), upd(streams)))
        logger.info(
            f"HY4_MK8 {label} streams after the post: vs golden streams_out pcc={out['streams'][0]:.6f}, update "
            f"pcc={out['streams'][1]:.5f} gain={out['streams'][2]:.4f}; vs the reference post of the device's own F "
            f"max_abs={(mine - streams).abs().max():.3e}"
        )
        if (mine - streams).abs().max() > 1e-3 * streams.abs().max():
            failed.append("streams are not the post-mix of the device's F")
    assert not failed, f"{label}: " + "; ".join(failed)
    return out


def golden_layer(layer):
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden under {hf_harness.GOLDEN_ROOT}")
    return golden["layers"][layer]


@galaxy
def test_mock(mesh_device):
    """Compiles the 32 programs and the post program and checks every buffer against L1 (nothing runs)."""
    if not is_mock():
        pytest.skip("the mock's compile and size check")
    g = torch.Generator().manual_seed(1)
    timing = bool(os.environ.get("HY4_MLP_TIMING"))  # the build with stamps and the MoE piece's timing words
    mlp = MlpLayer(mesh_device, timing=timing)
    site = ((torch.randn(8, 4 * HIDDEN, generator=g) * 0.02), torch.randn(8, generator=g), torch.tensor([0.01, 0.005]))
    small = lambda: L.to_device_stream(mesh_device, torch.zeros(64, 256))
    mlp.set_layer(
        1,
        [site],
        0,
        torch.ones(HIDDEN),
        torch.randn(E, HIDDEN, generator=g) * 0.01,
        torch.zeros(E),
        streams=(small(), small()),
    )
    logger.info(f"HY4_MK8 mlp mock: L1 address space taken by persistent tensors {mlp.l1_bytes()} B per core")
    for post in (False, True):
        mlp.run(post=post)
    mlp.run_post()
    # the largest set of circular buffers a core allocates (tensor-backed ones take nothing more), per kind of chip
    total = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
    for name, coord in (("root", (0, 0)), ("head", (1, 0)), ("other", (0, 1))):
        desc, _ = mlp._chip_program(coord, post=True)
        per_core = {}
        for cb in desc.cbs:
            backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
            if backed:
                continue
            for c in ttnn.corerange_to_cores(cb.core_ranges):
                per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
        worst = max(per_core, key=per_core.get)
        logger.info(
            f"HY4_MK8 mlp mock L1 {name} chip: {len(desc.kernels)} kernels, {len(desc.cbs)} CBs, {len(desc.semaphores)} "
            f"program semaphores; largest CB set {per_core[worst]} B on core {worst}; with the tensors "
            f"{per_core[worst] + mlp.l1_bytes()} of {total} B"
        )
    logger.info("HY4_MK8 mlp mock: the programs (pre and post + pre form) and the post program compiled")


RUNGS = {"solo": [(0, 0)], "pair_row": [(0, 0), (0, 1)], "pair_col": [(0, 0), (1, 0)], "first": None}
FIRST_WAIT = 100_000_000  # polls (about 1 s): an untraced first run launches the chips one after another


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    """One step of layer 1 on the root alone (no fabric connection), on two chips, on all 32: the first runs of the
    program on the board. On part of the mesh F is the sum of those chips' shares."""
    layer = 1
    hf = golden_layer(layer)
    mlp, w = build(mesh_device, layer, chips=RUNGS[rung], wait_limit=FIRST_WAIT)
    rows = slice(0, 8)
    if is_mock():  # the host path of a board run, and the compile of this rung's programs
        mlp.set_streams(hf["streams_mid"][rows].float())
        mlp.run()
        mlp.run_post()
        return
    mlp.set_streams(hf["streams_mid"][rows].float())
    ttnn.synchronize_device(mesh_device)
    mlp.run()
    ttnn.synchronize_device(mesh_device)
    status = mlp.status()
    logger.info(f"HY4_MK8 {rung} L{layer}: status after the program: {status or 'all at rest'}")
    assert not status, f"words not at rest: {status}"
    mlp.run_post()
    ttnn.synchronize_device(mesh_device)
    check_step(mlp, w, hf, rows, f"{rung} L{layer} rows 0..7", full=RUNGS[rung] is None)


@galaxy
@pytest.mark.parametrize("layer", range(1, 6), ids=lambda layer: f"layer{layer}")
def test_layers(mesh_device, layer):
    """Four steps (golden rows 0..7, 40..47, 80..87, 120..127) of a layer, each run directly."""
    if is_mock():
        pytest.skip("needs the board")
    hf = golden_layer(layer)
    mlp, w = build(mesh_device, layer, wait_limit=FIRST_WAIT)
    results = []
    for start in (0, 40, 80, 120):
        rows = slice(start, start + 8)
        mlp.set_streams(hf["streams_mid"][rows].float())
        ttnn.synchronize_device(mesh_device)
        mlp.run()
        mlp.run_post()
        ttnn.synchronize_device(mesh_device)
        results.append(check_step(mlp, w, hf, rows, f"L{layer} rows {start}..{start + 7}", per_chip=start == 0))
    a, b = "reference on the device's picks", "exact arithmetic on the BFP4 weights"
    logger.info(
        f"HY4_MK8 summary L{layer}: unions {[r['union'] for r in results]}; rows with the host router's picks "
        f"{[r['picks']['rows_exact'] for r in results]} of 8 (worst pick below the cut "
        f"{max(r['picks']['below'] for r in results):.1e}, smallest margin {min(r['picks']['margin'] for r in results):.1e}); F vs reference pcc "
        f"{[round(r[a][0], 4) for r in results]} gain {[round(r[a][1], 4) for r in results]}; F vs exact arithmetic on "
        f"BFP4 pcc {[round(r[b][0], 4) for r in results]} gain {[round(r[b][1], 4) for r in results]}; streams update "
        f"pcc {[round(r['streams'][1], 4) for r in results]} gain {[round(r['streams'][2], 4) for r in results]}"
    )


@galaxy
def test_trace(mesh_device):
    """Several steps in a row replayed from ONE trace of [the program, the post program], the root's streams
    rewritten in place between replays (layer 2, golden rows 0..63 in 8 steps)."""
    if is_mock():
        pytest.skip("needs the board")
    layer = 2
    hf = golden_layer(layer)
    mlp, w = build(mesh_device, layer, wait_limit=FIRST_WAIT)

    def step():
        mlp.run()
        mlp.run_post()

    mlp.set_streams(hf["streams_mid"][0:8].float())
    ttnn.synchronize_device(mesh_device)
    step()
    ttnn.synchronize_device(mesh_device)
    trace_id, _ = tracing.capture(mesh_device, step)
    try:
        for n in range(8):
            rows = slice(8 * n, 8 * n + 8)
            mlp.set_streams(hf["streams_mid"][rows].float())
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            check_step(mlp, w, hf, rows, f"trace L{layer} replay {n} rows {8 * n}..{8 * n + 7}", per_chip=False)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


# ---------------------------------------------------------------------------------------------------------------
# Layer 0: the dense FFN (6144 -> 18432 -> 6144) as 9 always-on plain slots, two BFP4 terms each, no router


class DenseWeights:
    """Layer 0's MLP-half weights on the host: the FFN as 9 slices in the experts' form (prepare.dense_experts)."""

    def __init__(self, layer=0):
        prefix = f"model.layers.{layer}."
        (self.gamma,) = hy4_tensors([prefix + "post_attention_layernorm.weight"])
        self.site = site_params(layer)
        self.gate_up, self.down = prepare.dense_experts(layer, terms=1)
        self._rounded = {}

    def ffn(self, x, rounded=False):
        """(reference/moe.py Hy4FFN's output [T, 6144], every chip's partial [32, T, 6144]); rounded: with the two
        BFP4 terms the device holds."""
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64)
        parts = torch.zeros(32, x.shape[0], HIDDEN, dtype=torch.float64)
        xf, step = x.float(), I // 32
        for j in range(self.gate_up.shape[0]):
            if rounded and j not in self._rounded:
                self._rounded[j] = (
                    L.two_terms_value(self.gate_up[j : j + 1].float())[0],
                    L.two_terms_value(self.down[j : j + 1].float())[0],
                )
            gu, dn = self._rounded[j] if rounded else (self.gate_up[j].float(), self.down[j].float())
            pre = xf @ gu.T
            act = L.swiglu(pre[:, :I], pre[:, I:], float("inf"))
            out += (act @ dn.T).double()
            for c in range(32):
                parts[c] += (act[:, c * step : (c + 1) * step] @ dn[:, c * step : (c + 1) * step].T).double()
        return out.float(), parts.float()


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_dense(mesh_device, rung):
    """Layer 0's dense FFN through the same program (MlpLayer(dense=True)): one step on the root alone, on two chips,
    on all 32. The same checks as a MoE layer's, without the router."""
    hf = golden_layer(0)
    w = DenseWeights(0)
    mlp = MlpLayer(mesh_device, chips=RUNGS[rung], dense=True, wait_limit=FIRST_WAIT)
    mlp.set_layer(0, [w.site], 0, w.gamma)
    rows = slice(0, 8)
    mlp.set_streams(hf["streams_mid"][rows].float())
    ttnn.synchronize_device(mesh_device)
    mlp.run()
    if is_mock():
        mlp.run_post()
        return
    ttnn.synchronize_device(mesh_device)
    status = mlp.status()
    logger.info(f"HY4_MK8 dense {rung}: status after the program: {status or 'all at rest'}")
    assert not status, f"words not at rest: {status}"
    mlp.run_post()
    ttnn.synchronize_device(mesh_device)
    failed = []
    root = mlp.read_root()
    streams_mid = hf["streams_mid"][rows].float()
    r_y, _, r_post, _ = golden_pre(mlp.cfg, *w.site, streams_mid)
    x, f = root["x"], root["f"]
    chips = {c: mlp.read_chip(c) for c in mlp.topo.chips}
    x_ok = sum(torch.equal(v["x"], x) for v in chips.values())
    partials = {c: v["out"] for c, v in chips.items()}
    wrong = int((two_stage_sum(mlp, partials) != f).sum())
    ref, _ = w.ffn(x)
    ref4, parts = w.ffn(x, rounded=True)
    scores = {
        c: (pcc(parts[mlp.topo.chip(c)], partials[c]), gain(parts[mlp.topo.chip(c)], partials[c]))
        for c in mlp.topo.chips
    }
    pccs, gains = sorted(s[0] for s in scores.values()), sorted(s[1] for s in scores.values())
    logger.info(
        f"HY4_MK8 dense {rung}: y vs the reference max_abs={(r_y - root['y']).abs().max():.3e}; x vs golden mlp_in "
        f"pcc={pcc(hf['mlp_in'][rows].float(), x):.7f}, bit-equal on {x_ok}/{len(chips)} chips; F vs the two-stage sum "
        f"of the partials: {wrong} elements differ; partials vs exact arithmetic on each chip's two-term BFP4 share: "
        f"pcc min {pccs[0]:.5f} median {pccs[len(pccs) // 2]:.5f}, gain {gains[0]:.4f} to {gains[-1]:.4f}"
    )
    if x_ok != len(chips) or wrong or pccs[0] < 0.97:
        failed.append("x, reduce or a partial is off")
    streams = mlp.read_streams()
    mine = golden_post(mlp.cfg, streams_mid, f, r_post)
    if (mine - streams).abs().max() > 1e-3 * streams.abs().max():
        failed.append("streams are not the post-mix of the device's F")
    if RUNGS[rung] is None:
        gold = hf["streams_out"][rows].float()
        logger.info(
            f"HY4_MK8 dense {rung}: F vs reference (Hy4FFN, checkpoint weights) pcc={pcc(ref, f):.5f} gain={gain(ref, f):.4f}; "
            f"vs exact arithmetic on the two-term BFP4 weights pcc={pcc(ref4, f):.5f} gain={gain(ref4, f):.4f}; vs golden "
            f"mlp_out pcc={pcc(hf['mlp_out'][rows].float(), f):.5f}; what two BFP4 terms cost alone pcc={pcc(ref, ref4):.5f}; "
            f"streams vs golden streams_out pcc={pcc(gold, streams):.6f}, update pcc={pcc(gold - streams_mid, streams - streams_mid):.5f} "
            f"gain={gain(gold - streams_mid, streams - streams_mid):.4f}"
        )
        if pcc(ref4, f) < 0.97:
            failed.append("F vs exact arithmetic on the BFP4 weights")
    assert not failed, f"dense {rung}: " + "; ".join(failed)
