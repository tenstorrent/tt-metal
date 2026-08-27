# SPDX-License-Identifier: Apache-2.0
"""Phase-0 de-risk spike for the on-device HunyuanImage-3.0 VAE decode.

Micro-benchmarks ttnn.experimental.conv3d at the hottest AutoencoderKLConv3D
decoder conv sizes (single-chip, symmetric pad=1, bf16) for:
  (a) correctness: PCC vs torch nn.Conv3d  (gate 0.99)
  (b) wall-clock: median of warm runs

Go/no-go: does a full-decoder conv3d budget plausibly beat the ~36s host VAE tail?

Run (on box):
  ./python_env/bin/python -m pytest -o timeout=0 -s \
    models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_vae_conv3d_spike.py
"""
from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

import ttnn
from models.tt_dit.utils.conv3d import get_conv3d_config, register_conv3d_configs
from tests.ttnn.utils_for_testing import check_with_pcc

# Register safe blockings for Hunyuan's big-in-channel combos so they don't hit the
# get_conv3d_config fallback (C_in_block=in_channels -> L1 overflow / program.cpp:1492).
# Modest C_in_block=32 runs everything; the mesh's small per-chip spatial keeps it fast.
register_conv3d_configs(
    {
        (1024, 1024, (3, 3, 3)): (32, 64, 1, 1, 1),
        (1024, 8192, (3, 3, 3)): (32, 32, 1, 1, 1),
        (1024, 4096, (3, 3, 3)): (32, 32, 1, 1, 1),
        (512, 512, (3, 3, 3)): (32, 64, 1, 1, 1),
        (512, 1024, (3, 3, 3)): (32, 64, 1, 1, 1),
    }
)

ALIGNMENT = 32


def _prep_input(x_ncdhw, C, device, dtype=ttnn.bfloat16):
    t = x_ncdhw.permute(0, 2, 3, 4, 1).contiguous()  # N D H W C
    if C % ALIGNMENT:
        t = torch.nn.functional.pad(t, (0, ALIGNMENT - C % ALIGNMENT))
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


# TUNED blocking via tt_dit get_conv3d_config (single-chip: h_factor=w_factor=1).
def _cfg(grid_obj, C, oc, k, T, H, W, dtype=ttnn.bfloat16):
    return get_conv3d_config(C, oc, k, dtype, grid_size=grid_obj, h_factor=1, w_factor=1, T=T, H=H, W=W)


# name, (N,C,T,H,W), out_c, count  -- all k=3 s=1 pad=1 (symmetric zeros).
# Resolutions are per-level INPUT res (ResBlocks run before the level's upsample).
# count = number of ~this-class k3 convs in the real Hunyuan decoder (for a budget sum).
CASES = [
    ("conv_in_32to1024@64T1", (1, 32, 1, 64, 64), 1024, 1),  # conv_in
    ("mid+i0_1024@64T1", (1, 1024, 1, 64, 64), 1024, 10),  # mid(4) + i0 ResBlocks(6)
    ("up0_1024to8192@64T1", (1, 1024, 1, 64, 64), 8192, 1),  # i0 UpsampleDCAE (temporal, factor8)
    ("i1_1024@128T2", (1, 1024, 2, 128, 128), 1024, 6),  # i1 ResBlocks
    ("up1_1024to4096@128T2", (1, 1024, 2, 128, 128), 4096, 1),  # i1 UpsampleDCAE (temporal, 512*8)
    ("i2_512@256T4", (1, 512, 4, 256, 256), 512, 6),  # i2 ResBlocks
    ("up2_512to1024@256T4", (1, 512, 4, 256, 256), 1024, 1),  # i2 UpsampleDCAE (spatial, 256*4)
    ("i3_256@512T4", (1, 256, 4, 512, 512), 256, 6),  # i3 ResBlocks
    ("up3_256to512@512T4", (1, 256, 4, 512, 512), 512, 1),  # i3 UpsampleDCAE (spatial, 128*4)
    ("i4_128@1024T4", (1, 128, 4, 1024, 1024), 128, 6),  # i4 ResBlocks
    ("conv_out_128to3@1024T4", (1, 128, 4, 1024, 1024), 3, 1),  # conv_out
]


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.DISABLED}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_vae_conv3d_spike(device_params, mesh_device):
    dev = mesh_device
    g = dev.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    torch.manual_seed(0)
    print(f"\nSPIKE_GRID {grid}", flush=True)

    kcfg = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    budget_ms = 0.0
    results = []
    for name, (N, C, T, H, W), oc, weight in CASES:
        k, pad, stride = (3, 3, 3), (1, 1, 1), (1, 1, 1)
        try:
            x = torch.randn(N, C, T, H, W)
            m = nn.Conv3d(C, oc, kernel_size=k, stride=stride, padding=pad, bias=True, padding_mode="zeros")
            with torch.no_grad():
                gt = m(x)

            ttx = _prep_input(x, C, dev)
            cfg = _cfg(g, C, oc, k, T, H, W)
            w = ttnn.from_torch(m.weight.data, dtype=ttnn.bfloat16, pad_value=0)
            w = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=w, groups=1, C_in_block=cfg.C_in_block, alignment=ALIGNMENT, device=dev
            )
            b = ttnn.from_torch(
                m.bias.data.reshape(1, -1), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
            )

            def run():
                o = ttnn.experimental.conv3d(
                    input_tensor=ttx,
                    weight_tensor=w,
                    device=dev,
                    bias_tensor=b,
                    dtype=ttnn.bfloat16,
                    output_channels=oc,
                    kernel_size=k,
                    stride=stride,
                    groups=1,
                    padding=pad,
                    dilation=(1, 1, 1),
                    padding_mode="zeros",
                    config=cfg,
                    compute_kernel_config=kcfg,
                )
                ttnn.synchronize_device(dev)
                return o

            # warmup + correctness (stride1/pad1/k3 -> same T,H,W)
            o = run()
            to = ttnn.to_torch(o, device=dev, dtype=torch.float32).reshape(N, T, H, W, oc).permute(0, 4, 1, 2, 3)
            ok, msg = check_with_pcc(gt, to, pcc=0.99)
            ttnn.deallocate(o)

            ts = []
            for _ in range(3):
                t0 = time.time()
                o = run()
                ts.append(time.time() - t0)
                ttnn.deallocate(o)
            ts.sort()
            med_ms = ts[1] * 1000.0
            budget_ms += med_ms * weight
            results.append((name, ok, med_ms, weight))
            print(
                f"SPIKE_CASE {name:22s} in=({C},{T},{H},{W})->{oc:<4d} "
                f"pcc={'OK ' if ok else 'FAIL'} {msg} ms={med_ms:8.1f} x{weight}",
                flush=True,
            )
            ttnn.deallocate(ttx)
            ttnn.deallocate(w)
        except Exception as e:
            results.append((name, None, None, weight))
            print(f"SPIKE_CASE {name:22s} in=({C},{T},{H},{W})->{oc:<4d} ERROR {type(e).__name__}: {e}", flush=True)

    print(
        f"\nSPIKE_BUDGET_MS_TOTAL={budget_ms:.1f}  (~weighted single-chip conv3d sum; "
        f"host VAE tail ~36000ms). lower is better.",
        flush=True,
    )
    print(
        f"SPIKE_VERDICT {'GO' if 0 < budget_ms < 36000 else 'CHECK'} "
        f"(single-chip, untuned default blocking; mesh HW-parallel would divide further)",
        flush=True,
    )
