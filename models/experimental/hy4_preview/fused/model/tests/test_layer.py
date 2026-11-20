# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ONE whole decoder layer on the 8x4 mesh (fused/model/program.py: P1, P2, the MLP program with its leading post, POST)
with checkpoint weights, layers 0 to 5 one at a time, each fed the HF golden's streams of a 128-token prompt.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "L1"    (the first 32-chip run)
    pytest <this file> -x

Setup: the attention half's own accuracy case (layer_attn/tests/test_layer.py LayerCase: 8 users = the golden prompt
read up to positions 120..127, the reference's cache rows packed by the prefill stack's ops, the selection every
cached position, "recommended" projections), taken into a Stack with the layer's MLP half ("p02" experts). The programs
are run one by one (not traced) so that the host can read between them. Checked (lines start with HY4_MK11; the
halves' own checks log as HY4_MK8A and HY4_MK8):
  after P2    the attention half's checks (check_layer): x against attn_in, queries, ctx, F against the reference on the
              stored rows and against HF's attn_out; and what the FP8 storage does to the rows (latent and RoPE parts
              separately) and to the attention output
  after MLP   streams against the golden's streams_mid (the MLP program's leading pass did the attention's post-mix)
  after POST  the MLP half's checks (check_step, on the device's own streams_mid): y, x against mlp_in, picks against a
              host router and HF's, partials, reduce, F against the reference and HF's mlp_out; streams against the
              golden's streams_out; the layer's update (streams_out - streams_in): PCC and gain against HF's
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_attn.tests import test_layer as AT
from models.experimental.hy4_preview.fused.layer_mlp.tests import test_mlp_layer as MT
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 64 << 20
FIRST_WAIT = 100_000_000
ROWS = AT.POSITIONS
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)
gain = MT.gain


def fp8_effect(case, label):
    """What the packed storage does to the cached rows, part by part (host): gain and PCC of the stored rows against
    the reference's rows, for the latent (FP8 with a scale per 128 values) and the RoPE (BF16) parts."""
    stored = case.stored_latents()
    lat = case.cfg.latent
    cat = lambda rows, sl: torch.cat([r[:, sl] for r in rows])
    out = {}
    for name, sl in (("latent", slice(0, lat)), ("rope", slice(lat, None))):
        ref, got = cat(case.latents, sl), cat(stored, sl)
        out[name] = (gain(ref, got), pcc(ref, got), float(ref.pow(2).mean().sqrt()))
    below = float((cat(stored, slice(0, lat)).abs() <= cat(case.latents, slice(0, lat)).abs() + 1e-12).float().mean())
    logger.info(
        f"HY4_MK11 {label} FP8 storage, stored rows against the reference's: latent part gain {out['latent'][0]:.4f} pcc "
        f"{out['latent'][1]:.6f} (rms {out['latent'][2]:.3e}; {100 * below:.1f} % of its values are not larger in magnitude "
        f"than the reference's); RoPE part gain {out['rope'][0]:.4f} pcc {out['rope'][1]:.6f} (rms {out['rope'][2]:.3e})"
    )
    return out


@galaxy
@pytest.mark.parametrize("layer", [1, 0, 2, 3, 4, 5], ids=lambda n: f"L{n}")
def test_layer(mesh_device, layer):
    dense = layer == 0
    case = AT.LayerCase(mesh_device, layer)
    a, hf = case.a, case.hf
    w = MT.DenseWeights(0) if dense else MT.Weights(layer)
    fidelity = os.environ.get("HY4_MOE_FIDELITY", "p02")
    st = Stack(mesh_device, attn=a, wait_limit=FIRST_WAIT, moe_fidelity=fidelity)
    st.set_sites([case.site, w.site], [case.gamma, w.gamma])
    state = st.add_layer(
        layer,
        None,
        case.sink,
        0,
        1,
        dense=dense,
        router_weight=None if dense else w.router_w,
        router_bias=None if dense else w.router_b,
    )
    m, cfg = state.mlp, a.cfg
    audit = st.cb_audit(state, post=False)
    report = st.l1_report(state)
    logger.info(
        f"HY4_MK11 layer {layer}: L1 tensors {report['tensors']} B per core, largest free block {report['largest_free']}, "
        f"largest CB set of P2 {report[('p2', (0, 0))][0]}; cb_audit rows {len(audit)}"
    )
    p1, p2, mlp = st.layer_programs(state, post=False, audit=True)
    post = ("post", *st.post_program())
    label = f"layer {layer}"
    try:
        Stack.launch([p1, p2])
        if is_mock():
            Stack.launch([mlp, post])
            return
        ttnn.synchronize_device(mesh_device)
        fp8_effect(case, label)
        res_attn = AT.check_layer(case, label)
        f_attn = a.read_root()["f"]

        Stack.launch([mlp])
        ttnn.synchronize_device(mesh_device)
        status = st.status()
        assert not status, f"{label}: words not at rest after the MLP program: {status}"
        streams_in = hf["streams_in"][ROWS].float()
        gold_mid = hf["streams_mid"][ROWS].float()
        dev_mid = st.read_streams()
        _, _, post1, _ = golden_pre(cfg, *case.site, streams_in)
        mine = golden_post(cfg, streams_in, f_attn, post1)
        upd = lambda s: s - streams_in
        logger.info(
            f"HY4_MK11 {label} streams after the attention's post-mix (the MLP program's leading pass): vs the golden's "
            f"streams_mid pcc={pcc(gold_mid, dev_mid):.6f}, the update's pcc={pcc(upd(gold_mid), upd(dev_mid)):.5f} gain="
            f"{gain(upd(gold_mid), upd(dev_mid)):.4f}; vs the reference post-mix of the device's own F max_abs="
            f"{(mine - dev_mid).abs().max():.3e}"
        )
        assert (
            mine - dev_mid
        ).abs().max() <= 1e-3 * dev_mid.abs().max(), f"{label}: streams_mid is not the post-mix of the attention's F"

        Stack.launch([post])
        ttnn.synchronize_device(mesh_device)
        gold_out = hf["streams_out"][ROWS].float()
        dev_out = st.read_streams()
        # the MLP half's own checks, on the streams the device had (its own streams_mid in place of the golden's)
        hf2 = dict(hf)
        hf2["streams_mid"] = hf["streams_mid"].float().clone()
        hf2["streams_mid"][ROWS] = dev_mid
        m.read_chip = lambda coord: st.read_mlp_chip(state, coord)
        if dense:
            root = m.read_root()
            chips = {c: m.read_chip(c) for c in m.topo.chips}
            x, f = root["x"], root["f"]
            x_ok = sum(torch.equal(v["x"], x) for v in chips.values())
            wrong = int((MT.two_stage_sum(m, {c: v["out"] for c, v in chips.items()}) != f).sum())
            ref, _ = w.ffn(x)
            ref4, _ = w.ffn(x, rounded=True)
            gold = hf["mlp_out"][ROWS].float()
            logger.info(
                f"HY4_MK11 {label} dense FFN: x vs the golden's mlp_in pcc={pcc(hf['mlp_in'][ROWS].float(), x):.7f}, bit-equal on "
                f"{x_ok}/{len(chips)} chips; F vs the two-stage sum: {wrong} elements differ; F vs the reference (checkpoint "
                f"weights) on the device's x pcc={pcc(ref, f):.5f} gain={gain(ref, f):.4f}; vs exact arithmetic on the two-term "
                f"BFP4 weights pcc={pcc(ref4, f):.5f} gain={gain(ref4, f):.4f}; vs the golden's mlp_out (HF) pcc={pcc(gold, f):.5f} "
                f"gain={gain(gold, f):.4f}"
            )
            assert x_ok == len(chips) and not wrong and pcc(ref4, f) > 0.97
            res_mlp = {"hf": (pcc(gold, f), gain(gold, f)), "hf_rows": None}
        else:
            r = MT.check_step(m, w, hf2, ROWS, label, post=True, per_chip=True)
            ids = m.read_chip(m.topo.root)["ids"].long()
            hf_rows = sum(set(ids[k].tolist()) == set(hf["router_indices"][ROWS][k].tolist()) for k in range(8))
            shared = sum(len(set(ids[k].tolist()) & set(hf["router_indices"][ROWS][k].tolist())) for k in range(8))
            res_mlp = {
                "hf": r["golden mlp_out (HF's picks)"][:2],
                "hf_rows": hf_rows,
                "shared": shared,
                "host_rows": r["picks"]["rows_exact"],
                "union": r["union"],
            }
        upd = lambda s: s - streams_in
        logger.info(
            f"HY4_MK11 {label} SUMMARY [experts {fidelity}] vs HF: attn_out pcc={res_attn['the golden attn_out (HF)'][0]:.5f} gain="
            f"{res_attn['the golden attn_out (HF)'][1]:.4f}; streams_mid pcc={pcc(gold_mid, dev_mid):.6f}; mlp_out pcc="
            f"{res_mlp['hf'][0]:.5f} gain={res_mlp['hf'][1]:.4f}; picks: {res_mlp.get('hf_rows')} of 8 rows HF's 8 experts "
            f"({res_mlp.get('shared')} of 64 picks shared; {res_mlp.get('host_rows')} of 8 rows a host router's on the device's "
            f"x; union {res_mlp.get('union')}); streams_out pcc={pcc(gold_out, dev_out):.6f}; the layer's update pcc="
            f"{pcc(upd(gold_out), upd(dev_out)):.5f} gain={gain(upd(gold_out), upd(dev_out)):.4f}"
        )
        status = st.status()
        assert not status, f"{label}: words not at rest after POST: {status}"
        canary = st.canary_summary()
        logger.info(f"HY4_MK11 {label} [experts {fidelity}] canary on the 7 heads: {canary}")
        for kind, v in canary.items():
            assert (
                v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"] == 7
            ), f"{label}: canary {kind}: {v}"
    finally:
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
