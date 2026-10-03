# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: DSV41MoEBlock on a Blackhole Galaxy vs the torch golden on real layer weights/activations."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.calibrate import calibrate_gate_cutoff
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_weights_vs_reference import golden_moe
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500_000,
            },
            id="fabric_1D_ring",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_id", [0, 2, 20])
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_moe_block(mesh_device, layer_id):
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    per_dev = 4
    batch = per_dev * rows

    # real activations: the FFN input of the checkpoint's own layer, for `batch` distinct tokens
    blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)
    tok = torch.randint(1000, 100000, (2, batch // 2))
    h, pm = R.embed_tokens(tok)
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    blk(h, 0, pm, None)
    x = cap["x"].reshape(batch, 1, 1, -1).to(torch.bfloat16)  # [batch, 1, 1, hidden]

    w = load_moe_layer(layer_id)
    golden, _ = golden_moe(x.reshape(batch, -1), w)  # [batch, hidden]

    K = calibrate_gate_cutoff(layer_id, seed=99)  # calibration tokens differ from the test batch
    moe = DSV41MoEBlock(mesh_device, w, topology=ttnn.Topology.Linear, batch_per_device=per_dev, gate_bias_shift=K)

    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(rows, cols))
    tt_tok = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    shard_gate = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols))
    tt_gate_in = ttnn.from_torch(
        x.reshape(1, 1, batch, -1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard_gate,
    )
    out = moe.forward(tt_gate_in, tt_tok)
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(
        ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(rows, cols)),
    )
    got = got.reshape(batch, -1).float()
    p = R.pcc(got, golden)
    print(f"layer {layer_id}: cutoff K={K:.3f}, MoE block PCC vs golden: {p:.5f}")
    # Layers 0/2 reach 0.994/0.988. Layer 20 reaches ~0.977: its bfp4 baseline is ~0.983 per token and the
    # bf16 gate cannot resolve one near-tie token (6th/7th margin 0.001, per-token PCC 0.90) in a batch of 16.
    assert p > 0.97
