"""Fused MoE tail (tt/moe_tail.py) vs the stock tilize+fast_reduce path: bit-level comparison and traced time of the full MoE block."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

T = int(os.environ.get("DSV41_T", "4"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_moe_tail(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("DSV41_L", "2"))
    blk = DSV41MoEBlock(md, load_moe_layer(layer), batch_per_device=T, gate_bias_shift=0.0)
    blk.warmup()
    d = torch.load(f"/mnt/tt-data/ssinghal/dsv4-chain-e/ffn_inputs_{layer}.pt")
    x = d["x"].to(torch.bfloat16)  # [16, 5120]
    B = 4 * T
    x = x[:B] if B <= 16 else torch.cat([x, torch.randn(B - 16, 5120).to(torch.bfloat16)])
    rows, cols = 4, 8
    sh = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    tok = ttnn.from_torch(
        x.reshape(B, 1, 1, 5120),
        device=md,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sh,
    )
    gate = ttnn.from_torch(
        x.reshape(1, 1, B, 5120),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
    sc, idx = blk.gate.forward(gate)
    outs = {}
    for mode in ("0", "1"):
        os.environ["DSV41_MOE_TAIL"] = mode
        o = blk.forward(gate, tok, (sc, idx))
        ttnn.synchronize_device(md)
        outs[mode] = ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp).float()
    a, b = outs["0"], outs["1"]
    print(
        f"TAIL stock-vs-fused: max abs diff {float((a - b).abs().max()):.4e} ref max {float(a.abs().max()):.3e} PCC {R.pcc(a.reshape(B, -1), b.reshape(B, -1)):.6f} finite {bool(torch.isfinite(b).all())}",
        flush=True,
    )
    for mode in ("0", "1", "0", "1"):
        os.environ["DSV41_MOE_TAIL"] = mode
        ms = chain_ms(md, lambda: blk.forward(gate, tok, (sc, idx)))
        print(f"TAIL traced MoE block tail={mode}: {ms * 1e3:.1f} us", flush=True)
