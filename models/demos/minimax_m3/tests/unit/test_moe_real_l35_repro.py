# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-MoE-layer reproduction of token-0 using REAL layer-35 weights + the REAL dumped L35 input
(post_attn_norm reconstructed from /tmp/m3_dump). Runs the FULL MoE pipeline (gate->dispatch->expert->
combine->reduce) — including the dispatch capacity buffer whose uninitialized padding is the suspected
source. Loads only ONE layer's experts -> ~1-2 min (not the 2.5hr full model). Optionally poisons DRAM
(DBG_POISON=1) to recreate the resident-garbage condition. Checks ALL devices for the ~1e38 garbage.

If this reproduces the explosion -> we have a minutes-long iteration loop (swap to swiglu_oai + re-run).
If it stays finite -> the bug truly needs the full-60-layer footprint (single layer can't recreate it)."""

import os

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.minimax_m3.tt.experts_throughput.tt_minimax_moe import TtMiniMaxMoE
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

EMB, HID, E, K = 6144, 3072, 128, 4
CKPT = os.environ.get("M3_CKPT", "/data/vmelnykov/MiniMax-M3-ref")
DUMP = os.environ.get("M3_DUMP", "/tmp/m3_dump")
P = "language_model.model.layers.35.block_sparse_moe."
LN = "language_model.model.layers.35.post_attention_layernorm.weight"


def _poison(mesh, rows, cols):
    junk = [
        ttnn.from_torch(
            torch.full((1, 1, 8192, EMB), 3.0e38),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, None)),
        )
        for _ in range(4)
    ]
    for t in junk:
        t.deallocate(True)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_moe_real_l35_repro(mesh_device, device_params, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    sp_axis = 0
    s_total = 2048
    s_local = s_total // rows  # 256

    # --- real L35 weights + the real dumped L35 input ---
    with safe_open(f"{CKPT}/model-00036-of-00059.safetensors", framework="pt") as f:
        pass  # (layer-35 spans shard 35/36; resolve via index below)
    import json

    wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]

    def gt(name):
        with safe_open(f"{CKPT}/{wm[name]}", framework="pt") as f:
            return f.get_tensor(name).float()

    gate_w = {"weight": gt(P + "gate.weight"), "e_score_correction_bias": gt(P + "e_score_correction_bias")}
    routed_w = [
        {
            "gate_proj": gt(P + f"experts.{e}.w1.weight"),
            "up_proj": gt(P + f"experts.{e}.w3.weight"),
            "down_proj": gt(P + f"experts.{e}.w2.weight"),
        }
        for e in range(E)
    ]
    pln = gt(LN)

    # reconstruct the MoE input = post_attn_layernorm(residual_in + attn_out)  [gemma: *(1+w)]
    ri = torch.load(f"{DUMP}/L35_residual_in.pt").float().reshape(s_total, EMB)
    ao = torch.load(f"{DUMP}/L35_attn_out.pt").float().reshape(s_total, EMB)
    h = ri + ao
    x = h * (h.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt() * (1.0 + pln)  # [2048, 6144], ~unit-RMS
    logger.info(f"reconstructed L35 MoE input: absmax={x.abs().max():.3e} finite={bool(torch.isfinite(x).all())}")

    mc = extract_mesh_config(mesh_device)
    dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
    epc, mlen, max_buf, max_tok = compute_constants(s_local, E, K, mesh_device.get_num_devices(), dgs, 2)
    moe = TtMiniMaxMoE(
        mesh_device=mesh_device,
        dispatch_group_size=dgs,
        num_dispatch_groups=ndg,
        experts_per_chip=epc,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=mlen,
        max_dispatched_tokens_per_expert=max_tok,
        max_dispatch_buffer_token_size=max_buf,
        seq_len_per_chip=s_local,
        emb_dim=EMB,
        hidden_dim=HID,
        gate_weights=gate_w,
        routed_expert_weights=routed_w,
        num_links=get_default_num_links(mesh_device),
        topology=ttnn.Topology.Linear,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
    )

    # M3-correct routing (host), passed in like the model's mlp.py (bypasses the internal gate).
    scores = torch.sigmoid(x @ gate_w["weight"].T)
    idx = torch.topk(scores + gate_w["e_score_correction_bias"], K, dim=-1).indices  # select by biased
    wsel = scores.gather(1, idx)
    wsel = wsel / wsel.sum(-1, keepdim=True)  # unbiased, normalized

    def sp_shard(t, dt, layout):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dt,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(1, None)),
        )

    x_tt = sp_shard(x.reshape(1, s_total, EMB).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    idx_tt = sp_shard(idx.reshape(1, s_total, K).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
    wts_tt = sp_shard(wsel.reshape(1, s_total, K).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    if os.getenv("DBG_POISON") == "1":
        _poison(mesh_device, rows, cols)
        logger.info("DRAM poisoned (3e38 pattern) before MoE call")

    out = moe(x_tt, topk_indices=idx_tt, topk_weights=wts_tt)

    dts = ttnn.get_device_tensors(out)
    gmax, gdev = 0.0, -1
    for i, d in enumerate(dts):
        m = ttnn.to_torch(d).float().abs().max().item()
        if m > gmax:
            gmax, gdev = m, i
    logger.info(
        f">>> MoE out GLOBALmax|x|={gmax:.3e} @dev{gdev}(row{gdev // cols},col{gdev % cols}) "
        f"-> {'REPRODUCED token-0 explosion' if gmax > 1e30 else 'finite (no explosion single-layer)'}"
    )
