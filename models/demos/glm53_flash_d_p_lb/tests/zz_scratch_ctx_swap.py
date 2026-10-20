# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: which context state costs the long-context accuracy. The ladder: golden context + the device's last chunk
gives final hidden PCC 0.938 / top-1 vs the reference 0.897; the device's own 51200-token context 0.790 / 0.703.
This builds the device context (chunks 0-9), keeps every layer's state, then runs the last chunk with each layer's
state from the device or the golden (s56320, at 51200) per variant:
  dev (all device), gold (all golden), gold_kda (KDA recurrent + conv golden), gold_dsa (MLA latent + indexer keys),
  gold_kv (MLA latent only), gold_idx (indexer keys only), gold_kda_lt<N> / gold_kda_ge<N> (KDA golden below / from
  layer N), from GLM_CS_VARIANTS.
Per variant: final hidden PCC vs the golden, top-1 / top-5 vs the reference's argmax on the ladder's sampled rows,
top-1 vs the text on all rows, L44 PCC.

    GLM_EXPERTS_MODE=ts TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_ctx_swap.py -s
"""

import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
KDA = ("kda_recurrent", "kda_conv")


def _pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def _pick(variant, i, name):
    """True: layer i's state tensor name comes from the golden."""
    if variant == "dev":
        return False
    if variant == "gold":
        return True
    if variant == "gold_kda":
        return name in KDA
    if variant == "gold_dsa":
        return name in ("kv_latent", "index_key")
    if variant == "gold_kv":
        return name == "kv_latent"
    if variant == "gold_idx":
        return name == "index_key"
    if variant.startswith("gold_kda_lt"):
        return name in KDA and i < int(variant[11:])
    if variant.startswith("gold_kda_ge"):
        return name in KDA and i >= int(variant[11:])
    raise ValueError(variant)


@mesh_parametrize
def test_ctx_swap(mesh_device):
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.common.bringup.testing.ladder import sampled_rows

    g = Golden.for_rung(S, "s56320")
    tokens = g.tokens().to(torch.long)
    seq, chunk = 56320, 5120
    last = seq - chunk
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    for start in range(0, last, chunk):
        h = model.embed(tokens[start : start + chunk])
        for i in layers:
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        model.free(h)
    dev = {i: {k: v.clone() for k, v in model.blocks[i].state_torch(last).items()} for i in layers}
    gold = {i: g.state(i, at=last) for i in layers}
    gm = g.model(10)
    rows = sampled_rows(chunk)
    gtop = gm["top32_ids"][rows].long()
    want = tokens[last + 1 : last + chunk]
    g44 = g.layer(10, layers[-1])["out"].float()
    variants = os.environ.get("GLM_CS_VARIANTS", "dev,gold,gold_kda,gold_dsa,gold_kv,gold_idx").split(",")
    for v in variants:
        for i in layers:
            st = {}
            for k, dv in dev[i].items():
                gv = gold[i].get(k)
                st[k] = gv.float() if (gv is not None and _pick(v, i, k)) else dv
            model.blocks[i].load_state(st, last)
        h = model.embed(tokens[last:seq])
        for i in layers:
            h2 = model.layer(i, h, last, None)
            model.free(h)
            h = h2
        l44 = _pcc(model.to_host(h).float(), g44)
        hid = model.final_norm(h)
        model.free(h)
        fh = _pcc(model.to_host(hid).float(), gm["final_norm"].float())
        lg = model.logits(hid, rows).float()
        t5 = lg.topk(5, dim=-1).indices
        r1 = (t5[:, 0] == gtop[:, 0]).float().mean().item()
        r5 = (t5 == gtop[:, :1]).any(-1).float().mean().item()
        hits = 0
        allr = list(range(chunk - 1))
        for r0 in range(0, len(allr), 512):
            rr = allr[r0 : r0 + 512]
            hits += int((model.logits(hid, rr).float().argmax(-1) == want[r0 : r0 + len(rr)]).sum())
        model.free(hid)
        print(
            f"[cs] {v:14s} final hidden pcc {fh:.4f}  L44 pcc {l44:.4f}  top1 vs ref {r1:.4f} top5 {r5:.4f}  "
            f"top1 vs text {hits / len(allr):.4f}",
            flush=True,
        )
