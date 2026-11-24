# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (HY4_MOCK=1: no board). The control of tests/test_decode_model.py: the reference fed the golden's tokens
from position 0, every layer attending over ITS OWN rows of the positions before (never the golden's cache rows),
as DecodeModel does. A variant is "<rows>.<experts>" in tests/test_control_rows.py's terms:

    rows     own       the rows as computed (fp32)
             near      the rows through the host model of the pack that rounds to nearest (what append="nearest"
                       stores; the pack's input is BF16)
    experts  exact     the checkpoint's BF16 values
             bfp4      what the decode streams hold: routed experts one BFP4 term, the shared expert and layer 0's
                       dense FFN two terms

    HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved HY4_CHAIN_LAYERS=6 \\
    HY4_CHAIN_PROMPT=31 HY4_OWN_VARIANTS=near.bfp4,near.exact TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

The contexts are below the top-k (every position up to the row's own is attended: no indexer in the control). Logs,
per variant, the golden steps' logits and tokens against the golden's, and saves the logits of the golden's steps
to generated/hy4_preview/logs/mf_own_rows_L<layers>_P<prompt>.pt ({variant: [steps, vocab]}), which
tests/test_decode_model.py reads (HY4_DM_CONTROL) to say whether a token the device loses is lost by the formats.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.ihc_site.op import golden_post, golden_pre
from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import DenseWeights
from models.experimental.hy4_preview.fused.model.tests.test_chain import (
    LAYERS,
    PROMPT,
    head_reference,
    host_logits,
    site,
)
from models.experimental.hy4_preview.fused.model.tests.test_control_rows import (
    E,
    Experts,
    plain_format,
    routed_format,
    store,
)
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import _checkpoint_reference
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import rms_norm, rope_pairing
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc

VARIANTS = os.environ.get("HY4_OWN_VARIANTS", "").split(",")
OUT = os.environ.get("HY4_OWN_OUT") or f"generated/hy4_preview/logs/mf_own_rows_L{LAYERS}_P{PROMPT}.pt"
TOPK = 2048


def test_own_rows():
    if not os.environ.get("HY4_OWN_VARIANTS"):
        pytest.skip("a host measurement, not a regression test: give HY4_OWN_VARIANTS (see the module's text)")
    torch.set_num_threads(int(os.environ.get("HY4_CTL_THREADS", "8")))
    g = hf_harness.find_decode_golden(LAYERS, PROMPT)
    assert g is not None and g["meta"]["rope_pairing"] == rope_pairing()
    N, steps = g["meta"]["prompt_len"], g["meta"]["steps"]
    S = N - 1 + steps
    assert S <= TOPK, "the control attends every position: contexts below the top-k"
    ids = g["input_ids"][:S].long()
    (table,) = hy4_tensors(["model.embed_tokens.weight"])
    emb = (
        table[ids].float()[:, None, :].expand(S, 4, table.shape[1]).contiguous()
    )  # the 4 streams start as the embedding row
    gold0 = g["layers"][0]["streams_in"].float()
    assert torch.equal(emb[N - 1 :], gold0), "the embedding rows are not the golden's streams_in of layer 0"
    cfg = IHCConfig()
    streams = {name: emb.clone() for name in VARIANTS}
    J = list(range(steps))
    gold = lambda layer, stage: g["layers"][layer][stage][J].float()
    t0 = time.perf_counter()
    for layer in range(LAYERS):
        ref = _checkpoint_reference(layer)
        (gamma_attn,) = hy4_tensors([f"model.layers.{layer}.input_layernorm.weight"])
        s_attn, s_mlp = site(layer, "hc_attn_layer"), site(layer, "hc_mlp_layer")
        w = DenseWeights(0) if layer == 0 else Experts(layer)
        stage = {}
        with torch.no_grad():
            for name in VARIANTS:
                rows_kind, experts = name.split(".")
                y, _, post, _ = golden_pre(cfg, *s_attn, streams[name])
                x = rms_norm(y, gamma_attn.float(), cfg.norm_eps)
                held, outs = [], []
                for i in range(S):  # position i appends its row, then attends positions 0 .. i
                    token = ref.new_token(x[i : i + 1], i)
                    held.append(store(token.latent.float(), "near") if rows_kind == "near" else token.latent.float())
                    outs.append(ref.absorbed_stages(x[i : i + 1], token, torch.cat(held))["out"].float())
                f_attn = torch.cat(outs)
                mid = golden_post(cfg, streams[name], f_attn, post)
                y2, _, post2, _ = golden_pre(cfg, *s_mlp, mid)
                x2 = rms_norm(y2, w.gamma.float(), cfg.norm_eps)
                ids_, weights = (None, None) if layer == 0 else ML.route_golden(x2, w.router_w, w.router_b)[1:]
                stage[name] = {
                    "attn_out": f_attn,
                    "mid": mid,
                    "post2": post2,
                    "mlp_in": x2,
                    "ids": ids_,
                    "weights": weights,
                }
            if layer:
                need = set()
                for name in VARIANTS:
                    experts = name.split(".")[1]
                    need |= {(e, routed_format(experts, layer)) for e in set(stage[name]["ids"].flatten().tolist())}
                    need.add((E, plain_format(experts)))
                with ThreadPoolExecutor(16) as pool:
                    list(pool.map(w._exact, sorted({e for e, _ in need})))
                    list(pool.map(lambda k: w.get(*k), sorted(need)))
            for name in VARIANTS:
                experts = name.split(".")[1]
                st_ = stage[name]
                if layer == 0:
                    f_mlp, _ = w.ffn(st_["mlp_in"], rounded=plain_format(experts) != "x")
                else:
                    f_mlp = w.moe(
                        st_["mlp_in"], st_["ids"], st_["weights"], routed_format(experts, layer), plain_format(experts)
                    )
                st_["mlp_out"] = f_mlp
                streams[name] = golden_post(cfg, st_["mid"], f_mlp, st_["post2"])
        del w
        if layer < 6 or layer % 6 == 5 or layer == LAYERS - 1:
            logger.info(
                f"HY4_MF own rows P{PROMPT} layer {layer} ({time.perf_counter() - t0:.0f} s), the golden's {steps} steps, pcc against the golden "
                f"(attn_out / mlp_out / streams_out): "
                + "; ".join(
                    f"{name} {pcc(gold(layer, 'attn_out'), stage[name]['attn_out'][N - 1 :]):.5f} / "
                    f"{pcc(gold(layer, 'mlp_out'), stage[name]['mlp_out'][N - 1 :]):.5f} / {pcc(gold(layer, 'streams_out'), streams[name][N - 1 :]):.5f}"
                    for name in VARIANTS
                )
            )
    head, lm = head_reference()
    gl = g["logits"][J].float()
    want = g["cached_logits_argmax"][J]
    margin = [round(float(v[0] - v[1]), 3) for v in gl.topk(2, dim=-1).values]
    keep = {"variants": VARIANTS, "want": want.tolist(), "margin": margin, "logits": {}, "tokens": {}}
    for name in VARIANTS:
        _, logits = host_logits(head, lm, streams[name][N - 1 :])
        got = logits.argmax(-1)
        rows_pcc = [pcc(gl[r], logits[r]) for r in range(steps)]
        keep["logits"][name], keep["tokens"][name] = logits.clone(), got.tolist()
        logger.info(
            f"HY4_MF own rows P{PROMPT} L{LAYERS} head on {name}: logits vs the golden pcc={pcc(gl, logits):.6f} (per step min {min(rows_pcc):.6f}); "
            f"top-1 the golden's on {int((got == want).sum())} of {steps} steps; differing (step, golden, got, the golden's margin, the "
            f"golden's token below the top here) "
            f"{[(r, int(want[r]), int(got[r]), margin[r], round(float(logits[r].max() - logits[r, want[r]]), 3)) for r in range(steps) if got[r] != want[r]]}"
        )
    torch.save(keep, OUT)
    logger.info(
        f"HY4_MF own rows P{PROMPT} L{LAYERS}: the logits of the golden's {steps} steps per variant saved to {OUT} ({time.perf_counter() - t0:.0f} s)"
    )
