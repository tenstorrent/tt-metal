"""moe_compute time vs number of active experts on the busiest device, for the env-selected knobs
(MOE_COMPUTE_BFP8_WEIGHTS / FP32_ACC / FIDELITY / APPROX; read at program creation, so one process per setting)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms, stages
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

T = 4


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
def test_scaling(mesh_device):
    md = mesh_device
    import faulthandler

    faulthandler.dump_traceback_later(150, exit=False)
    print("MCS start", flush=True)
    tag = f"bfp8={os.environ.get('MOE_COMPUTE_BFP8_WEIGHTS','0')} acc={os.environ.get('MOE_COMPUTE_FP32_ACC','0')} fid={os.environ.get('MOE_COMPUTE_FIDELITY','LoFi')} approx={os.environ.get('MOE_COMPUTE_APPROX','1')}"
    blk = DSV41MoEBlock(md, load_moe_layer(2), batch_per_device=T, gate_bias_shift=0.0)
    print("MCS block built", flush=True)
    blk.warmup()
    print("MCS warm", flush=True)
    h = ttnn.from_torch(
        torch.randn(1, 1, T, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    h_tok = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, 5120])
    rep_rows = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
    mk = lambda ids: ttnn.from_torch(
        torch.tensor(ids, dtype=torch.int32).repeat(T * 4, 1).reshape(T * 4, 1, 1, 6),
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    wts = ttnn.from_torch(
        torch.full((T * 4, 1, 1, 6), 0.25).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    # dispatch-only baseline, then moe_compute (stage 1 total); active experts live on ONE device (ids 0..n-1, device 0 owns 0-11)
    pre = {
        n: mk((list(range(n)) + [24, 36, 48, 60, 72, 84])[:6]) for n in (1, 2, 3, 6)
    }  # n experts on device 0, the rest 1 per other device (duplicate ids hang)
    i0, i6 = mk([0] * 6), mk([0, 12, 24, 36, 48, 60])
    for n in (1, 2, 3, 6):
        ms = chain_ms(md, lambda: stages(blk.decode, h_tok, wts, pre[n], 1)) * 1e3
        print(f"MCS [{tag}] active experts on one device {n}: dispatch+moe_compute {ms:7.1f} us", flush=True)
    ms = chain_ms(md, lambda: stages(blk.decode, h_tok, wts, i6, 1)) * 1e3
    print(f"MCS [{tag}] 6 experts on 6 devices (1 each): {ms:7.1f} us", flush=True)
