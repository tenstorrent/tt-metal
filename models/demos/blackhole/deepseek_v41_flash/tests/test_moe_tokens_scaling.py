"""moe_compute (router + dispatch + moe_compute + tail, the DSV41PrefillMoE path) time vs tokens per device T = 32*G; checks each 32-token slice
against the T=32 block. Env DSV41_MTS="32,64,128,256", DSV41_MTS_LAYER (3). Router runs per 32-token slice (kernel limit)."""
import gc
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillMoE


def run(pm, h, h_tok, T):
    return pm.forward(h, h_tok)


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
def test_tokens_scaling(mesh_device):
    md = mesh_device
    w = load_moe_layer(int(os.environ.get("DSV41_MTS_LAYER", "3")))
    blk = DSV41MoEBlock(md, w, batch_per_device=32, gate_bias_shift=0.0)
    blk.warmup()
    for T in [int(x) for x in os.environ.get("DSV41_MTS", "32,64,128,256").split(",")]:
        try:
            os.environ["DSV41_MOE_G"] = str(T // 32)
            pm = DSV41PrefillMoE(blk, T=32)
            h = ttnn.from_torch(
                torch.randn(1, 1, T, 5120).to(torch.bfloat16),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            h_tok = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, 5120])
            ms = chain_ms(md, lambda: ttnn.deallocate(pm.forward(h, h_tok)))
            print(f"MTS T={T} tokens/call={4 * T} ms/call={ms:.3f} us/token={ms * 1e3 / (4 * T):.2f}", flush=True)
            o = ttnn.to_torch(ttnn.get_device_tensors(pm.forward(h, h_tok))[0]).float().reshape(T, -1)
            for j in range(T // 32):
                hj = ttnn.slice(h, [0, 0, 32 * j, 0], [1, 1, 32 * (j + 1), 5120])
                hj_tok = ttnn.reshape(ttnn.to_layout(hj, ttnn.ROW_MAJOR_LAYOUT), [32, 1, 1, 5120])
                oj = ttnn.to_torch(ttnn.get_device_tensors(blk.forward(hj, hj_tok))[0]).float().reshape(32, -1)
                print(
                    f"MTS T={T} slice {j}: PCC vs T=32 {R.pcc(o[32 * j : 32 * (j + 1)], oj):.6f} maxabs {(o[32 * j : 32 * (j + 1)] - oj).abs().max():.4f}",
                    flush=True,
                )
        except Exception as e:
            print(f"MTS T={T} FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        ttnn.synchronize_device(md)
        gc.collect()
