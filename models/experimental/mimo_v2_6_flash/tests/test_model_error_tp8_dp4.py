# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the 48-layer model's error comes from: every position of the golden prompts, not only the last.

Each golden prompt is prefilled whole (prompt g on replica g % 4). For every layer, the device hidden states of all
positions are compared with the fp32 reference's (relative error and PCC), which shows how the error builds up
through the stack. The last layer's hidden states then go through the reference's final norm and LM head in fp32
on the host, and every position's logits are scored against the reference: top-1 agreement and KL(reference ||
device). With ~120 positions this is a steadier accuracy measure than the 40 rows of test_model_tp8_dp4.py, and
it isolates the decoder stack from the device LM head.

MIMO_ERROR_DUMP=<file> writes the per-layer and per-position numbers as JSON. A golden from
reference/make_golden_large.py (MIMO_GOLDEN; only the last layer's hidden states, many more positions) is scored the
same way, with the reference logits from its hidden states and no per-layer comparison.
"""

import json
import os

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference.model import rms_norm
from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    golden_path,
    load_config_and_ckpt,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_error_tp8_dp4(mesh_device):
    golden = torch.load(golden_path())
    cfg, ckpt = load_config_and_ckpt()
    gen = MiMoGalaxyGenerator(
        mesh_device, cfg, ckpt, cache_dir("model_bfp4experts"), dp=DP, users_per_replica=4, max_seq_len=1024
    )
    num_prompts = len(golden["token_ids"])
    large = "final_hidden" in golden
    device_hidden = {}
    for first in range(0, num_prompts, DP):  # one prompt per replica at a time; slots are reused
        pending = {}
        for g in range(first, min(first + DP, num_prompts)):
            model = gen.models[g % DP]
            pending[g] = (model, model.prefill_enqueue(golden["token_ids"][g], 0, return_hidden=True))
        for g, (model, handle) in pending.items():
            hidden = model.prefill_read(handle)[1]
            device_hidden[g] = [hidden[-1]] if large else hidden

    num_layers = 0 if large else len(device_hidden[0])
    layer_rel, layer_pcc = [], []
    for i in range(num_layers):
        dev = torch.cat([device_hidden[g][i] for g in range(num_prompts)])
        ref = torch.cat([golden["hidden"][g][i + 1].float() for g in range(num_prompts)])
        layer_rel.append(float((dev - ref).norm() / ref.norm()))
        layer_pcc.append(pcc(dev, ref))
    for i in range(num_layers):
        logger.info(f"layer {i:2d}: hidden rel. error {layer_rel[i]:.4f}, PCC {layer_pcc[i]:.5f}")

    norm, lm_head = ckpt.final_norm(), ckpt.lm_head().float()
    rows = []
    for g in range(num_prompts):
        with torch.no_grad():
            dev = rms_norm(device_hidden[g][-1], norm, cfg.layernorm_epsilon) @ lm_head.T
        if large:
            with torch.no_grad():
                ref = rms_norm(golden["final_hidden"][g].float(), norm, cfg.layernorm_epsilon) @ lm_head.T
        else:
            ref = golden["logits"][g].float()
        for p in range(ref.shape[0]):
            lp_ref, lp_dev = ref[p].log_softmax(-1), dev[p].log_softmax(-1)
            top = ref[p].topk(2)
            rows.append(
                dict(
                    golden=g,
                    pos=p,
                    match=int(dev[p].argmax()) == int(top.indices[0]),
                    ref_margin=float(top.values[0] - top.values[1]),
                    kl=float((lp_ref.exp() * (lp_ref - lp_dev)).sum()),
                    pcc=pcc(dev[p], ref[p]),
                )
            )
    n = len(rows)
    top1 = sum(r["match"] for r in rows) / n
    confident = [r for r in rows if r["ref_margin"] >= 0.5]
    top1_confident = sum(r["match"] for r in confident) / len(confident)
    kl = sum(r["kl"] for r in rows) / n
    logger.info(
        f"all {n} positions (host fp32 head on device hidden states): top-1 {top1:.4f}, top-1 where the reference "
        f"margin >= 0.5 {top1_confident:.4f} ({len(confident)} rows), mean KL {kl:.4f}, "
        f"PCC min {min(r['pcc'] for r in rows):.4f} mean {sum(r['pcc'] for r in rows) / n:.4f}"
    )
    later = [r for r in rows if r["pos"] >= 2]  # positions 0-1 carry the first token's massive activation
    logger.info(
        f"positions >= 2 ({len(later)}): top-1 {sum(r['match'] for r in later) / len(later):.4f}, "
        f"mean KL {sum(r['kl'] for r in later) / len(later):.4f}"
    )
    for r in rows if n <= 200 else []:
        if not r["match"]:
            logger.info(f"miss: golden {r['golden']} pos {r['pos']}, reference margin {r['ref_margin']:.3f}")
    if os.environ.get("MIMO_ERROR_DUMP"):
        with open(os.environ["MIMO_ERROR_DUMP"], "w") as f:
            json.dump(dict(layer_rel=layer_rel, layer_pcc=layer_pcc, rows=rows), f, indent=1)
    assert top1 >= 0.8, top1
