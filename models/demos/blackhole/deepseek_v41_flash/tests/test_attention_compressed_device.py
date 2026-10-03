# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: compressed-KV attention (layers 2 and 20) decode step vs the checkpoint's Attention, real weights.

Reference prefills S tokens on CPU (window ring, compressed cache, compressor state); a snapshot of that state
seeds the device; one decode step runs on both. Contexts stay short enough that the indexer selects every
compressed position (compress_len <= 512), which the module asserts.
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41CompressedAttention
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("layer_id,S", [(2, 23), (2, 24), (20, 24)])
@pytest.mark.parametrize("fake_quant", [False, True], ids=["unquantised_ref", "fakequant_ref"])
@torch.no_grad()
def test_compressed_attention(mesh_device, layer_id, S, fake_quant):
    torch.manual_seed(0)
    ref_kernels.FAKE_QUANT = fake_quant
    rows, cols = tuple(mesh_device.shape)
    per_row, B = 4, rows * 4
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)
    ratio = blk.attn.compress_ratio

    def block_input(tok):
        h, pm = R.embed_tokens(tok)
        return blk.attn_norm(blk.hc_pre(h, pm))

    x_pre = block_input(torch.randint(1000, 100000, (B, S)))
    x_dec = block_input(torch.randint(1000, 100000, (B, 1)))
    blk.attn(x_pre, 0)
    comp = blk.attn.compressor
    snap = dict(
        window=blk.attn.window_kv_cache.clone().float(),
        comp=blk.attn.compress_kv_cache[:, : S // ratio].clone().float(),
        kv_state=comp.kv_state.clone() if ratio > 1 else None,
        score_state=comp.score_state.clone() if ratio > 1 else None,
    )
    ref = blk.attn(x_dec, S).float()

    mesh_config = mesh_4x8()
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
    if ratio > 1:
        comp_w["wgate"] = comp.wgate.weight.data.float()
    attn = DSV41CompressedAttention(
        mesh_device,
        mesh_config,
        ccl,
        R.dequantized_attention_weights(blk),
        blk.attn.freqs_cis,
        ratio,
        comp_w,
        users_per_row=per_row,
        max_comp=128,
    )
    attn.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"])
    st = attn.step_inputs(torch.full((B,), S))
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols))
    tt_x = ttnn.from_torch(
        x_dec.reshape(1, 1, B, 5120).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    out = attn.forward(tt_x, st)
    devs = ttnn.get_device_tensors(out)
    got = torch.cat([ttnn.to_torch(devs[r * cols]).reshape(-1, 5120) for r in range(rows)]).float()[:B]
    p = R.pcc(got, ref.reshape(B, 5120))
    print(f"layer {layer_id} S={S} ratio={ratio} fake_quant={fake_quant}: attention decode PCC {p:.5f}")
    ref_kernels.FAKE_QUANT = True
    assert p > (0.97 if fake_quant else 0.98)
