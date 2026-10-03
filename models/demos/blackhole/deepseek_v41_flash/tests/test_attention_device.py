# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: window-only attention (layer 0) decode step vs the checkpoint's own Attention, real weights.

State injection: the reference prefills S tokens per user on CPU; its window cache seeds the device cache;
then one decode step runs on both and the outputs are compared.
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("layer_id", [0])
@pytest.mark.parametrize("S", [24])
@torch.no_grad()
def test_attention_decode(mesh_device, layer_id, S):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    per_row = 4
    B = rows * per_row
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)

    def block_input(tok):  # real-scale attention input: attn_norm of the collapsed embedding streams
        h, pm = R.embed_tokens(tok)
        return blk.attn_norm(blk.hc_pre(h, pm))

    x_pre = block_input(torch.randint(1000, 100000, (B, S)))
    x_dec = block_input(torch.randint(1000, 100000, (B, 1)))
    blk.attn(x_pre, 0)  # prefill: fills blk.attn.window_kv_cache
    ref = blk.attn(x_dec, S).float()  # [B, 1, 5120]

    mesh_config = mesh_4x8()
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    attn = DSV41Attention(
        mesh_device,
        mesh_config,
        ccl,
        R.dequantized_attention_weights(blk),
        blk.attn.freqs_cis,
        users_per_row=per_row,
        max_seq=256,
    )
    attn.load_window(
        blk.attn.window_kv_cache[:, :S].float()
    )  # NOTE: ref cache already includes the decode token at slot S
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
    print(
        f"layer {layer_id} attention decode PCC: {p:.5f}  per-user:",
        [round(R.pcc(got[i], ref[i, 0]), 3) for i in range(B)],
    )
    assert p > 0.98
