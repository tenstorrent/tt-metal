# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of the decode swiglu clamp: DSV41MoEBlock (moe_compute SILU_CLAMPED, swiglu_limit 10) vs the torch golden with and without the clamp.

The routed experts' gate / up weights are scaled by DSV41_CLAMP_SCALE so the projections exceed the limit often (with real-size activations the clamp rarely
triggers); scaling the weights instead of the input leaves the routing unchanged. DSV41_DEC_SWIGLU_CLAMP=0 runs the block without the clamp (then it must agree with the unclamped golden).
"""
import os

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
@pytest.mark.parametrize("layer_id", [2])
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_moe_clamp(mesh_device, layer_id):
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    per_dev = int(
        os.environ.get("DSV41_CLAMP_PER_DEV", "4")
    )  # tokens per device: 4 = B 16; the server also runs the decode / verify buckets of other sizes
    batch = per_dev * rows
    scale = float(os.environ.get("DSV41_CLAMP_SCALE", "16"))
    clamp_on = os.environ.get("DSV41_DEC_SWIGLU_CLAMP", "1") != "0"

    blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)
    tok = torch.randint(1000, 100000, (2, batch // 2))
    h, pm = R.embed_tokens(tok)
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    blk(h, 0, pm, None)
    x = cap["x"].reshape(batch, 1, 1, -1).to(torch.bfloat16)
    if (
        os.environ.get("DSV41_CLAMP_SAME") == "1"
    ):  # every token identical: they all route to the same experts, so an expert sees up to `batch` tokens (several 32-token chunks)
        x = x[:1].repeat(batch, 1, 1, 1).contiguous()
    w = load_moe_layer(
        layer_id, experts=range(384)
    )  # host copies of every routed expert (a warm cache would skip them); the block is built from them
    w["w0"] *= scale
    w["w1"] *= scale
    g_clamp, _ = golden_moe(x.reshape(batch, -1), w, limit=10.0, shared=False)
    g_plain, _ = golden_moe(x.reshape(batch, -1), w, limit=0.0, shared=False)
    print(
        f"golden clamp vs plain PCC {R.pcc(g_clamp, g_plain):.5f} (the lower, the more the clamp matters at this scale)"
    )

    K = calibrate_gate_cutoff(layer_id, seed=99)
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
    got = (
        ttnn.to_torch(
            ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(rows, cols)),
        )
        .reshape(batch, -1)
        .float()
    )
    p_clamp, p_plain = R.pcc(got, g_clamp), R.pcc(got, g_plain)
    print(
        f"layer {layer_id} scale {scale} device clamp={'on' if clamp_on else 'off'}: PCC vs clamped golden {p_clamp:.5f}, vs unclamped golden {p_plain:.5f}"
    )
    want, other = (p_clamp, p_plain) if clamp_on else (p_plain, p_clamp)
    assert want > 0.97
    assert want > other, "the device output follows the wrong variant"
