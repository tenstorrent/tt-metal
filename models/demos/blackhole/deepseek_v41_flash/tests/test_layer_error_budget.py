# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does a layer's device error come from?  For one layer, seeded from the saved reference chain state, run
the checkpoint's own sub-blocks on the CPU and the device layer on the same input, then compare:

  attention output, FFN (MoE + shared) output, and the router's expert choice per token.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-ne")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("layer_id", [int(x) for x in os.environ.get("DSV41_BUDGET_LAYERS", "1,8").split(",")])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_error_budget(mesh_device, layer_id):
    ref_kernels.FAKE_QUANT = False
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    d = torch.load(os.path.join(CHAIN, f"layer_{layer_id}.pt"))
    d["S"] = S
    B = d["dec_in"].shape[0]

    # ---- reference sub-blocks from the saved prefill state
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)
    st = d["state"]
    blk.attn.window_kv_cache.copy_(st["window"])
    ratio = blk.attn.compress_ratio
    assert ratio == 0 or blk.attn.is_kv_source, "pick a self-contained layer (ratio 0 or a kv source)"
    if ratio and blk.attn.is_kv_source:
        blk.attn.compress_kv_cache[:, : st["comp"].shape[1]].copy_(st["comp"])
        if ratio > 1:
            blk.attn.compressor.kv_state.copy_(st["kv_state"])
            blk.attn.compressor.score_state.copy_(st["score_state"])
    cap = {}
    blk.attn.register_forward_hook(lambda m, i, o: cap.update(attn_in=i[0].detach(), attn_out=o.detach()))
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(ffn_in=i[0].detach(), ffn_out=o.detach()))
    gate_idx = {}
    blk.ffn.gate.register_forward_hook(lambda m, i, o: gate_idx.update(idx=o[1].detach()))
    blk(d["dec_in"].to(torch.bfloat16), S, d["pre_in"], None)
    # clamp attribution: the same FFN input through the reference MoE with the swiglu clamp switched off
    clamped = blk.ffn(cap["ffn_in"]).float().reshape(B, 5120)
    saved = blk.ffn.shared_experts.swiglu_limit
    blk.ffn.shared_experts.swiglu_limit = 0.0
    for e in blk.ffn.experts:
        if e is not None:
            e.swiglu_limit = 0.0
    noclamp = blk.ffn(cap["ffn_in"]).float().reshape(B, 5120)
    print(
        f"BUDGET layer {layer_id}: reference MoE without the swiglu clamp vs with: rel-err {float((noclamp - clamped).norm() / clamped.norm()):.4f} PCC {R.pcc(noclamp, clamped):.5f}"
    )

    # ---- device layer on the same input
    chain = DSV41DecodeChain(mesh_device, log=lambda m: None)
    from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

    w = load_layer(layer_id)
    layer, attn = chain.build_layer(layer_id, d, w)
    layer.debug = {}
    sp = attn.step_inputs(torch.full((B,), S))
    out, nxt = layer.forward(
        chain.to_dev(d["dec_in"].reshape(B, -1), 4 * 5120), chain.to_dev(d["pre_in"].reshape(B, -1), 4), sp
    )
    ttnn.synchronize_device(mesh_device)
    dbg = layer.debug
    dev_attn = chain.to_host(dbg["attn_out"], 5120)
    dev_ffn = chain.to_host(dbg["ffn_out"], 5120)
    dev_routed = chain.to_host(dbg["routed"], 5120)
    rel = lambda a, b: float((a - b).norm() / b.norm())
    ra, rf = cap["attn_out"].reshape(B, 5120).float(), cap["ffn_out"].reshape(B, 5120).float()
    print(
        f"BUDGET layer {layer_id}: device ffn vs reference-WITHOUT-clamp: PCC {R.pcc(dev_ffn, noclamp):.5f} rel-err {rel(dev_ffn, noclamp):.4f}"
    )
    dev_in = chain.to_host(dbg["ffn_in"], 5120)
    ref_in = cap["ffn_in"].reshape(B, 5120).float()
    ref_shared = blk.ffn.shared_experts(cap["ffn_in"].reshape(-1, 5120)).float().reshape(B, 5120)
    ref_routed = rf - ref_shared
    dev_shared = chain.to_host(dbg["shared"], 5120)
    print(f"BUDGET layer {layer_id}: ffn INPUT  PCC {R.pcc(dev_in, ref_in):.5f} rel-err {rel(dev_in, ref_in):.4f}")
    print(
        f"BUDGET layer {layer_id}: shared expert  PCC {R.pcc(dev_shared, ref_shared):.5f} rel-err {rel(dev_shared, ref_shared):.4f}  (|ref shared| {float(ref_shared.norm()):.1f})"
    )
    print(
        f"BUDGET layer {layer_id}: routed experts PCC {R.pcc(dev_routed, ref_routed):.5f} rel-err {rel(dev_routed, ref_routed):.4f}  (|ref routed| {float(ref_routed.norm()):.1f})"
    )
    # the device FFN input differs from the reference's (it comes from the device's own attention output): feed-through effect
    print(f"BUDGET layer {layer_id}: attention  PCC {R.pcc(dev_attn, ra):.5f} rel-err {rel(dev_attn, ra):.4f}")
    print(f"BUDGET layer {layer_id}: ffn (moe+shared) PCC {R.pcc(dev_ffn, rf):.5f} rel-err {rel(dev_ffn, rf):.4f}")
    # routing agreement
    _, tt_idx = layer.moe.last_routing
    devs = ttnn.get_device_tensors(tt_idx)
    dev_idx = torch.cat(
        [ttnn.to_torch(devs[r * mesh_device.shape[1]]).reshape(-1, 6) for r in range(mesh_device.shape[0])]
    ).long()[:B]
    ref_idx = gate_idx["idx"].reshape(B, 6).long()
    same = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(dev_idx, ref_idx))
    print(f"BUDGET layer {layer_id}: router same expert set for {same}/{B} tokens")
    # contribution of tokens with identical routing vs different routing to the FFN error
    ok = torch.tensor([set(a.tolist()) == set(b.tolist()) for a, b in zip(dev_idx, ref_idx)])
    if ok.any():
        print(f"BUDGET layer {layer_id}: ffn rel-err on tokens with SAME routing {rel(dev_ffn[ok], rf[ok]):.4f}")
    if (~ok).any():
        print(f"BUDGET layer {layer_id}: ffn rel-err on tokens with DIFFERENT routing {rel(dev_ffn[~ok], rf[~ok]):.4f}")
    ref_kernels.FAKE_QUANT = True
