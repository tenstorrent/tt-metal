# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Image input in the interleaved prefill (DSV41_VISION=1, tt/vision_state.py): the embedding-override / image-mask path through the column-split chunk trace.

Few layers (VIS_LAYERS, default "0": no Engram layer, so the mask only acts through the embedding and the router). Checks, on the SAME model:
  (a) identity: image spans whose rows ARE the table rows of their token ids (and bias_vl forced equal to bias) give logits bit-identical to the same prompts without images,
  (b) the override is effective: random rows in the spans change the logits,
  (c) a window / call without images after a call with images gives the baseline again (the persistent buffers are cleared).

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_vision_splice.py -s
Env: VIS_LAYERS (0), VIS_ISL (900), VIS_CHUNK (512).
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionSeq
from models.tt_transformers.tt.common import PagedAttentionConfig

U = 4
ISL = int(os.environ.get("VIS_ISL", "900"))
CHUNK = int(os.environ.get("VIS_CHUNK", "512"))
LAYERS = os.environ.get("VIS_LAYERS", "0")


def pcc(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


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
def test_vision_splice(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()  # (a failed trace capture hangs the mesh teardown: the pytest report never comes)
        raise


@torch.no_grad()
def _body(mesh_device):
    os.environ["DSV41_VISION"] = "1"
    os.environ.setdefault("DSV41_PF_UMASK", "1")
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    S_pad = -(-ISL // CHUNK) * CHUNK
    max_ctx = S_pad + 128
    pp = default_page_params(max_ctx, U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    pm = m.prefill_model
    assert pm.vis, "DSV41_VISION=1 must reach the prefill model"
    # bias_vl == bias: the identity check isolates the embedding path from the routing path
    zero = ttnn.from_torch(
        torch.zeros(1, 1, 1, 384),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    for _, pl in pm.layers:
        assert pl.L.moe.gate._dbias is not None
        pl.L.moe.gate._dbias = zero

    g = torch.Generator().manual_seed(7)
    prompts = torch.randint(1000, 100000, (B, ISL), generator=g)
    lens = [ISL - 29 * (u % 5) for u in range(B)]
    table = _Shards().get("embed.weight")  # [129280, 5120] bf16

    def spans_for(u, rows_fn):
        """three image spans per user (one crossing the 512 chunk boundary, one just before the end of the prompt) with image token types"""
        starts = [
            60 + 7 * u,
            470 + 3 * u,
            lens[u] - 110,
        ]  # (the last one is inside the sliding window of the last token)
        types = torch.full((lens[u],), -1, dtype=torch.long)
        sp = []
        for s in starts:
            n = 90
            types[s : s + n] = 1
            sp.append((s, rows_fn(u, s, n)))
        return VisionSeq(types, sp)

    def run(vis):
        for u in range(B):
            m.release_user(u)
        m.pf_resume = {}
        items = [(u, prompts[u], 0, lens[u]) for u in range(B)]
        res = m.prefill_interleaved(items, CHUNK, s_pad_max=S_pad, want_logits=True, vis=vis)
        return {u: (int(res[u][0]), res[u][1].float().clone()) for u in range(B)}

    base = run(None)
    ident = run({u: spans_for(u, lambda u, s, n: table[prompts[u, s : s + n]]) for u in range(B)})
    for u in range(B):
        p = pcc(base[u][1], ident[u][1])
        eq = torch.equal(base[u][1], ident[u][1])
        log(f"identity user {u}: token {base[u][0]} vs {ident[u][0]}, logits PCC {p:.6f}, bit-equal {eq}")
        assert base[u][0] == ident[u][0] and p > 0.99999, (u, p)
    rnd = torch.Generator().manual_seed(11)
    diff = run(
        {
            u: spans_for(u, lambda u, s, n: (torch.randn(n, 5120, generator=rnd) * 0.05).to(torch.bfloat16))
            for u in range(B)
        }
    )
    ps = [pcc(base[u][1], diff[u][1]) for u in range(B)]
    log(f"random rows: PCC vs baseline {[round(x, 4) for x in ps]}")
    assert max(ps) < 0.999, "the override rows have no effect"
    again = run(None)
    for u in range(B):
        assert torch.equal(
            base[u][1], again[u][1]
        ), f"user {u}: logits changed after a call with images (stale override buffers)"
    log("clean call after images: bit-identical to the baseline")
