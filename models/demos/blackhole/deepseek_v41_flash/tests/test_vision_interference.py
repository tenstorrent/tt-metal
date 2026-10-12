# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Running the vision tower right before a traced prefill must not change the prefill: the first-token logits of the same image prompt are bit-identical over repeated
rounds (tower encode -> prefill_interleaved with the fresh rows), with the tower's eager ops between the trace replays.

    VIS_LAYERS (0-3), VIS_ROUNDS (6), VIS_SYNC (0/1: a full-mesh synchronize after every encode)
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.vision import DSV41Vision, VisionArgs
from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionSeq
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.tt_transformers.tt.common import PagedAttentionConfig

U, CHUNK = 4, 512


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1600000000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
def test_vision_interference(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()
        raise


@torch.no_grad()
def _body(md):
    os.environ["DSV41_VISION"] = "1"
    os.environ.setdefault("DSV41_PF_UMASK", "1")
    a, _, b = os.environ.get("VIS_LAYERS", "0-3").partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B, ISL = 4 * U, 700
    max_ctx = 1024 + 128
    pp = default_page_params(max_ctx, U)
    _, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=lambda s: None,
    )
    va = VisionArgs()
    vision = DSV41Vision(md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va)
    vision.warmup()
    sync = os.environ.get("VIS_SYNC", "0") == "1"
    r = np.random.RandomState(0)
    pil = Image.fromarray(r.randint(0, 255, (13, 17, 3)).astype(np.uint8)).resize((560, 420), Image.BICUBIC)
    patches, n_h, n_w, h, w = VP.patchify_pil(pil, va)
    types = VP._ip.image_token_types(h, w)
    g = torch.Generator().manual_seed(7)
    prompts = torch.randint(1000, 100000, (B, ISL), generator=g)
    S_pad = -(-ISL // CHUNK) * CHUNK
    start = 100

    def one_round():
        rows = vision.image_rows(patches, n_h, n_w, types)  # eager tower ops right before the traced prefill
        if sync:
            ttnn.synchronize_device(md)
        tt = torch.full((ISL,), -1, dtype=torch.long)
        tt[start : start + types.numel()] = types
        pr = prompts.clone()
        pr[:, start : start + types.numel()] = va.image_token_id
        for u in range(B):
            m.release_user(u)
        m.pf_resume = {}
        res = m.prefill_interleaved(
            [(u, pr[u], 0, ISL) for u in range(B)],
            CHUNK,
            s_pad_max=S_pad,
            want_logits=True,
            vis={u: VisionSeq(tt, [(start, rows)]) for u in range(B)},
        )
        # then plain decode steps (traced) with the greedy tokens fed back: the generated tokens must repeat as well
        tok = torch.tensor([int(res[u][0]) for u in range(B)])
        gen = [tok.clone()]
        for k in range(int(os.environ.get("VIS_STEPS", "24"))):
            out = m.decode_forward(tok, torch.full((B,), ISL + k), enable_trace=True, reload_inputs=True)
            tok = out.reshape(-1)[:B].clone().long()
            gen.append(tok.clone())
        return torch.stack([res[u][1].float() for u in range(B)]), torch.stack(gen)

    outs = [one_round() for _ in range(int(os.environ.get("VIS_ROUNDS", "6")))]
    bad = 0
    for i, (lg, gen) in enumerate(outs[1:], 1):
        eq, eg = torch.equal(outs[0][0], lg), torch.equal(outs[0][1], gen)
        first = (outs[0][1] != gen).any(1).nonzero()
        print(
            f"round {i}: prefill logits bit-equal: {eq}; decode tokens equal: {eg}"
            + ("" if eg else f" (first differing step {int(first[0])})"),
            flush=True,
        )
        bad += not (eq and eg)
    assert bad == 0, f"{bad} rounds differ"
