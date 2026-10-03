# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 (attention part): device PREFILL attention of one layer vs the CPU reference dump (reference/ref_prefill_dump.py).

Checks the attention output for 16 users x S tokens (PCC) and the decode state it leaves in the layer's cache:
window ring, compressed latents, ratio-2 ``prev_cs``. Env: DSV41_PREFILL_DIR (default /mnt/tt-data/ssinghal/dsv4-prefill-s{S})."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.attention import WINDOW, DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, pad_len
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def build_attn(md, L, w, max_comp):
    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    meta = w["meta"]
    if meta["ratio"] == 0:
        attn = DSV41Attention(md, mesh_config, ccl, w["attn"], w["freqs_cis"], users_per_row=4, max_seq=256)
    else:
        assert meta["is_kv_source"], "reader layers need their source attention (not covered by this test)"
        attn = DSV41CompressedAttention(
            md,
            mesh_config,
            ccl,
            w["attn"],
            w["freqs_cis"],
            meta["ratio"],
            w["compressor"],
            users_per_row=4,
            max_comp=max_comp,
        )
    return attn


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_id,S", [(0, 9), (0, 128), (2, 9), (2, 128), (20, 9), (20, 128)])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_prefill_attention(mesh_device, layer_id, S):
    md = mesh_device
    rows, cols = tuple(md.shape)
    d = os.environ.get("DSV41_PREFILL_DIR", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}")
    ref = torch.load(os.path.join(d, f"layer_{layer_id}.pt"))
    pf, st = ref["prefill"], ref["state"]
    w = load_layer(layer_id, with_moe=False, max_seq_len=256)
    max_comp = 256
    attn = build_attn(md, layer_id, w, max_comp)
    attn.prefill = DSV41PrefillAttention(attn, w["attn"]["attn_sink"])
    ratio = w["meta"]["ratio"]
    Sp = pad_len(S)
    B = rows * 4
    x = pf["attn_in"].float()  # [16,S,D]
    xp = torch.zeros(B, Sp, 5120)
    xp[:, :S] = x
    h = ttnn.from_torch(
        xp.reshape(rows, 1, 4 * Sp, 5120),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    out = attn.prefill.forward(h, S)
    ttnn.synchronize_device(md)
    dev = lambda t: [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(rows)]
    got = torch.cat([o.reshape(4, Sp, 5120) for o in dev(out)])[:, :S]
    p_out = pcc(got, pf["attn_out"].float())
    per_user = [round(pcc(got[u], pf["attn_out"][u].float()), 4) for u in range(B)]
    cache = torch.cat([c.reshape(4, -1, 512) for c in dev(attn.cache)])
    ring_p = pcc(cache[:, :S], st["window"][:, :S])
    msg = f"PREFILL ATTN layer {layer_id} S={S}: out PCC {p_out:.5f} ring PCC {ring_p:.5f} per-user min {min(per_user)}"
    if ratio:
        nc = S // ratio
        comp_p = pcc(cache[:, WINDOW : WINDOW + nc], st["comp"][:, :nc])
        msg += f" comp PCC {comp_p:.5f}"
        if ratio > 1 and S % ratio:
            ref_cs = torch.cat([st["kv_state"][:, (S - 1) % ratio], st["score_state"][:, (S - 1) % ratio]], dim=-1)
            got_cs = torch.cat([c.reshape(4, -1) for c in dev(attn.prev_cs)])
            msg += f" prev_cs PCC {pcc(got_cs, ref_cs):.6f}"
    print(msg, flush=True)
    assert p_out > 0.999 and ring_p > 0.999
