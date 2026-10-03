# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 4: SDPA decode (non-causal + mask + sink, K=256 slots) program-config variants. Prints 'P4 name us'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms

T, H, D = 4, 8, 512


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_probe4(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    q = up(torch.randn(1, T, H, D))
    K = up(torch.randn(T, 1, 256, D))
    m = up(torch.zeros(T, 1, H, 256))
    sink = up(torch.zeros(32, 32))
    ck = {
        n: ttnn.init_device_compute_kernel_config(
            md.arch(), math_fidelity=f, math_approx_mode=a, fp32_dest_acc_en=fp, packer_l1_acc=False
        )
        for n, (f, a, fp) in {
            "HiFi4f32": (ttnn.MathFidelity.HiFi4, False, True),
            "HiFi2f32": (ttnn.MathFidelity.HiFi2, False, True),
            "HiFi2": (ttnn.MathFidelity.HiFi2, False, False),
            "LoFi": (ttnn.MathFidelity.LoFi, True, False),
        }.items()
    }

    def run(name, **kw):
        ckn = kw.pop("ck", "HiFi4f32")
        pc = ttnn.SDPAProgramConfig(**kw)
        try:
            f = lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                K,
                K,
                is_causal=False,
                attn_mask=m,
                attention_sink=sink,
                scale=D**-0.5,
                program_config=pc,
                compute_kernel_config=ck[ckn],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print(f"P4 {name:50s} {chain_ms(md, f) * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"P4 {name:50s} FAIL {str(e).splitlines()[0][:100]!r}", flush=True)

    for ckn in ("HiFi4f32", "HiFi2f32", "HiFi2", "LoFi"):
        for gx, gy in ((4, 1), (8, 1), (8, 4)):
            for mc in (1, 2):
                run(
                    f"grid {gx}x{gy} kc=128 mc={mc} {ckn}",
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    q_chunk_size=0,
                    k_chunk_size=128,
                    exp_approx_mode=False,
                    max_cores_per_head_batch=mc,
                    ck=ckn,
                )
    for kc in (256,):
        run(
            f"grid 4x1 kc={kc} mc=1",
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
            q_chunk_size=0,
            k_chunk_size=kc,
            exp_approx_mode=False,
            max_cores_per_head_batch=1,
        )
