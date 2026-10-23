# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TtDFlash (tt/dflash.py) against reference/dflash.py on one Galaxy row (TP=8). Galaxy only.

Target features are synthetic (the drafter's math does not depend on where they come from): a prompt's rows go into
the ring through prefill_context, then each step writes 1-4 committed rows from the taps (the decode / verify traces'
buffers) and drafts. Checks: the block's final normed hidden of the eager draft program against the fp32 reference
(PCC), and the traced drafts against the reference's greedy drafts. Also the draft program's replay time.
"""

import os
import time

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.experimental.mimo_v2_6_flash.reference.dflash import DFlashConfig, ReferenceDFlash, load_dflash_weights
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS
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
    ref = ReferenceDFlash(dcfg, load_dflash_weights(dcfg), embed, lm_head)

    g = torch.Generator().manual_seed(0)
    total = PROMPT + 4 * STEPS + 1
    feat = (torch.randn(total, len(dflash.tap_layers), cfg.hidden_size, generator=g) * 2).bfloat16()
    tokens = torch.randint(1000, 100000, (total,), generator=g)
    rep = model.plan.replicate()

    def host(t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)

    # The prompt's context, as a prefill chunk writes it.
    padded = -(-PROMPT // TILE) * TILE
    acc = None
    for k, layer in enumerate(dflash.tap_layers):
        t = torch.zeros(1, 1, padded, cfg.hidden_size)
        t[0, 0, :PROMPT] = feat[:PROMPT, k].float()
        x = ttnn.to_device(host(t), row)
        acc = dflash.prefill_tap(layer, x, acc)
        x.deallocate(True)
    dflash.prefill_context(acc, 0, PROMPT)
    ref.add_context(feat[:PROMPT], torch.arange(PROMPT))

    start, context, pccs, agree, n = PROMPT, [], [], 0, 0
    for step in range(STEPS):
        for k in range(len(dflash.taps)):
            t = torch.zeros(1, 1, TILE, cfg.hidden_size)
            for r, pos in context:
                t[0, 0, r] = feat[pos, k].float()
            ttnn.copy_host_to_device_tensor(host(t), dflash.taps[k])
        anchor = int(tokens[start])
        if step < 2:
            dflash.debug = []
            dflash.upload(context, anchor, start)
            dflash._body()
            got = model.plan.to_torch_replicated(dflash.debug.pop()).float()[0, 0, : dcfg.block]
            dflash.debug = None
            pccs.append(pcc(got, ref.block_hidden(anchor, start)))
        else:
            if dflash.trace is None:
                dflash.capture_trace()
            dflash.enqueue(context, anchor, start)
        drafts = dflash.read_drafts()
        want = ref.draft(anchor, start, MAX_DRAFTS)
        agree += sum(a == b for a, b in zip(drafts, want))
        n += len(want)
        # The next step commits 1-4 rows (the anchor and some drafts), whose features the taps then hold.
        m = 1 + step % 4
        ref.add_context(feat[start : start + m], torch.arange(start, start + m))
        context = [(j, start + j) for j in range(m)]
        start += m
    logger.info(f"DFlash block hidden PCC {pccs}, draft agreement {agree}/{n}")

    ttnn.synchronize_device(row)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(row, dflash.trace, cq_id=0, blocking=False)
    dflash.read_drafts()
    logger.info(f"DFlash draft program: {1e3 * (time.perf_counter() - t0) / 20:.2f} ms per replay")
    dflash.release_trace()
    assert min(pccs) > 0.999, pccs
    assert agree >= 0.9 * n, (agree, n)
