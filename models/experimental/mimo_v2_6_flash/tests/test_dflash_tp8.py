# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TtDFlash (tt/dflash.py) against reference/dflash.py on one Galaxy row (TP=8). Galaxy only.

Target features are synthetic (the drafter's math does not depend on where they come from): each draft group's prompt
rows go into its ring through prefill_context, then each step writes 1-4 committed rows per group from the taps (the
decode / verify traces' buffers; group g's rows at tap rows 4 g ..) and drafts every group in one program. Checks: the
block's final normed hidden of the eager draft program against the fp32 reference of each group (PCC), and the traced
drafts against each group's reference greedy drafts. Also the draft program's replay time. With draft groups
(MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2) the groups' requests are independent.
"""

import os
import time

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.experimental.mimo_v2_6_flash.reference.dflash import DFlashConfig, ReferenceDFlash, load_dflash_weights
from models.experimental.mimo_v2_6_flash.reference.spec_decode import DRAFT_GROUPS, MAX_DRAFTS
from models.experimental.mimo_v2_6_flash.tests.common import (
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    dp_submeshes,
    load_config_and_ckpt,
    model_path,
)
from models.experimental.mimo_v2_6_flash.tt.common import TILE
from models.experimental.mimo_v2_6_flash.tt.dflash import TtDFlash
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

PROMPT = 200
STEPS = 12


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_dflash(mesh_device):
    path = model_path()
    if not os.path.exists(os.path.join(path, "dflash", "config.json")):
        pytest.skip(f"no DFlash draft model in {path}/dflash")
    cfg, ckpt = load_config_and_ckpt()
    dcfg = DFlashConfig.from_model_path(path)
    row = dp_submeshes(mesh_device)[0]
    model = TtMiMoModel(row, cfg, ckpt=ckpt, cache_dir=cache_dir("model_bfp4experts"), num_layers=1)
    model.setup_decode_rope_tables()
    dflash = TtDFlash(model, path, cache_dir("dflash"))
    dflash.setup_cache()
    with safe_open(os.path.join(path, "model_pp0_ep0_shard0.safetensors"), "pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight").float()
        lm_head = f.get_tensor("lm_head.weight").float()
    weights = load_dflash_weights(dcfg)
    refs = [ReferenceDFlash(dcfg, weights, embed, lm_head) for _ in range(DRAFT_GROUPS)]

    g = torch.Generator().manual_seed(0)
    total = PROMPT + 4 * STEPS + 1
    # Group g's request: its own features and tokens, its prompt at a different length.
    prompts = [PROMPT + 37 * k for k in range(DRAFT_GROUPS)]
    feats = [
        (torch.randn(total + 40 * k, len(dflash.tap_layers), cfg.hidden_size, generator=g) * 2).bfloat16()
        for k in range(DRAFT_GROUPS)
    ]
    tokens = [torch.randint(1000, 100000, (total + 40 * k,), generator=g) for k in range(DRAFT_GROUPS)]
    rep = model.plan.replicate()

    def host(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)

    # Each prompt's context, as a prefill chunk writes it into its group's ring (after the request's reset).
    for grp in range(DRAFT_GROUPS):
        dflash.reset(grp)
        n = prompts[grp]
        padded = -(-n // TILE) * TILE
        acc = None
        for k, layer in enumerate(dflash.tap_layers):
            t = torch.zeros(1, 1, padded, cfg.hidden_size)
            t[0, 0, :n] = feats[grp][:n, k].float()
            x = ttnn.to_device(host(t), row)
            acc = dflash.prefill_tap(layer, x, acc)
            x.deallocate(True)
        dflash.prefill_context(acc, 0, n, grp)
        refs[grp].add_context(feats[grp][:n], torch.arange(n))

    starts, contexts, pccs, agree, n_drafts = list(prompts), [[] for _ in range(DRAFT_GROUPS)], [], 0, 0
    for step in range(STEPS):
        for k in range(len(dflash.taps)):
            t = torch.zeros(1, 1, TILE, cfg.hidden_size)
            for grp in range(DRAFT_GROUPS):
                for r, pos in contexts[grp]:
                    t[0, 0, r] = feats[grp][pos, k].float()
            ttnn.copy_host_to_device_tensor(host(t), dflash.taps[k])
        drafts_in = [
            (grp, contexts[grp], int(tokens[grp][starts[grp]]), starts[grp], 4 * grp) for grp in range(DRAFT_GROUPS)
        ]
        if step < 2:
            dflash.debug = []
            dflash.upload(drafts_in)
            dflash._body()
            got = model.plan.to_torch_replicated(dflash.debug.pop()).float()[0, 0]
            dflash.debug = None
            for grp in range(DRAFT_GROUPS):
                block = got[grp * dcfg.block : (grp + 1) * dcfg.block]
                pccs.append(pcc(block, refs[grp].block_hidden(int(tokens[grp][starts[grp]]), starts[grp])))
        else:
            if dflash.trace is None:
                dflash.capture_trace()
            dflash.enqueue(drafts_in)
        for grp in range(DRAFT_GROUPS):
            drafts = dflash.read_drafts(grp)
            want = refs[grp].draft(int(tokens[grp][starts[grp]]), starts[grp], MAX_DRAFTS)
            agree += sum(a == b for a, b in zip(drafts, want))
            n_drafts += len(want)
        # The next step commits 1-4 rows per group (its anchor and some drafts), whose features the taps then hold.
        for grp in range(DRAFT_GROUPS):
            m = 1 + (step + grp) % 4
            s0 = starts[grp]
            refs[grp].add_context(feats[grp][s0 : s0 + m], torch.arange(s0, s0 + m))
            contexts[grp] = [(4 * grp + j, s0 + j) for j in range(m)]
            starts[grp] += m
    logger.info(f"DFlash block hidden PCC {pccs}, draft agreement {agree}/{n_drafts} ({DRAFT_GROUPS} groups)")

    ttnn.synchronize_device(row)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(row, dflash.trace, cq_id=0, blocking=False)
    dflash.read_drafts()
    logger.info(f"DFlash draft program: {1e3 * (time.perf_counter() - t0) / 20:.2f} ms per replay")
    dflash.release_trace()
    # Random features of scale 2 lose more to bf16 through the 5 layers than the target's (the captured greedy paths
    # of real prompts give 0.9997 and 99.6% agreement): 0.998 and 86% here.
    assert min(pccs) > 0.997, pccs
    assert agree >= 0.8 * n_drafts, (agree, n_drafts)
