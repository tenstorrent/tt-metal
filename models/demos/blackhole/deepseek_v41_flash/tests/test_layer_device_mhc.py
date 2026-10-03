# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: a full decoder layer (decode step) vs the checkpoint's own Block, real weights.

The reference prefills S tokens per user on CPU (window cache, MoE and all); its window cache seeds the device
cache; then one decode step runs on both. Layer 0 is the window-only layer (compress_ratio 0).
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.calibrate import calibrate_gate_cutoff
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("layer_id,S", [(0, 8), (2, 23), (20, 24)])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_layer_decode(mesh_device, layer_id, S):
    torch.manual_seed(0)
    ref_kernels.FAKE_QUANT = False  # unquantised reference: the fp4/fp8 activation rounding is not reproduced on device
    rows, cols = tuple(mesh_device.shape)
    per_row = 4
    B = rows * per_row
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)

    h_pre, pm_pre = R.embed_tokens(torch.randint(1000, 100000, (B, S)))
    h_dec, pm_dec = R.embed_tokens(torch.randint(1000, 100000, (B, 1)))
    blk(h_pre, 0, pm_pre, None)  # prefill on CPU: fills the window cache (and compressed cache / state)
    ratio = blk.attn.compress_ratio
    snap = None
    if ratio:
        comp = blk.attn.compressor
        snap = dict(
            window=blk.attn.window_kv_cache.clone().float(),
            comp=blk.attn.compress_kv_cache[:, : S // ratio].clone().float(),
            kv_state=comp.kv_state.clone() if ratio > 1 else None,
            score_state=comp.score_state.clone() if ratio > 1 else None,
        )
    ref_x, ref_pre = blk(h_dec, S, pm_dec, None)  # decode step
    ref_x, ref_pre = ref_x.float(), ref_pre.float()

    mesh_config = mesh_4x8()
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    weights = R.dequantized_attention_weights(blk)
    if ratio == 0:
        attn = DSV41Attention(
            mesh_device, mesh_config, ccl, weights, blk.attn.freqs_cis, users_per_row=per_row, max_seq=256
        )
        attn.load_window(blk.attn.window_kv_cache[:, :S].float())
    else:
        comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            comp_w["wgate"] = comp.wgate.weight.data.float()
        attn = DSV41CompressedAttention(
            mesh_device,
            mesh_config,
            ccl,
            weights,
            blk.attn.freqs_cis,
            ratio,
            comp_w,
            users_per_row=per_row,
            max_comp=128,
        )
        attn.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"])
    mhc = lambda n: (
        getattr(blk, f"hc_{n}_fn").data,
        getattr(blk, f"hc_{n}_base").data,
        getattr(blk, f"hc_{n}_scale").data,
    )
    layer = DSV41Layer(
        mesh_device,
        mesh_config,
        ccl,
        attn,
        norms={"attn_norm": blk.attn_norm.weight.data.float(), "ffn_norm": blk.ffn_norm.weight.data.float()},
        mhc_params={"attn": mhc("attn"), "ffn": mhc("ffn")},
        moe_weights=load_moe_layer(layer_id),
        gate_bias_shift=calibrate_gate_cutoff(layer_id, seed=99),
        users_per_row=per_row,
    )

    if os.environ.get("MHC_FUSED_NORM", "1") == "1":  # exercise DSV41MHC.collapse_norm without editing layer.py
        for mname, wname in (("mhc_attn", "attn_norm_w"), ("mhc_ffn", "ffn_norm_w")):
            m, w_ = getattr(layer, mname), getattr(layer, wname)
            m.collapse = (lambda m_, w2: lambda x, pre: m_.collapse_norm(x, pre, w2, layer.eps))(m, w_)
        layer._norm = lambda h, w: h
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t: ttnn.from_torch(
        t.float(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    tt_x = up(h_dec.reshape(B, 1, 4, 5120))
    tt_pre = up(pm_dec.reshape(B, 1, 1, 4))
    st = attn.step_inputs(torch.full((B,), S))
    out, nxt = layer.forward(tt_x, tt_pre, st)

    rows_only = lambda t, d: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1, d) for r in range(rows)]
    )
    got_x = rows_only(out, 4 * 5120).float()[:B]
    got_pre = rows_only(nxt, 4).float()[:B]
    p_x, p_pre = R.pcc(got_x, ref_x.reshape(B, -1)), R.pcc(got_pre, ref_pre.reshape(B, -1))
    streams = [round(R.pcc(got_x.reshape(B, 4, 5120)[:, i], ref_x.reshape(B, 4, 5120)[:, i]), 4) for i in range(4)]
    ref_kernels.FAKE_QUANT = True
    print(f"LAYER {layer_id} decode: streams PCC {p_x:.5f} (per stream {streams}), next-pre PCC {p_pre:.5f}")
    assert p_x > 0.98 and p_pre > 0.98
