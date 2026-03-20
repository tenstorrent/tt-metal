#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark conv3d with/without fp32 intermediate reduction on real Wan decoder layer sizes.

Uses the device profiler to measure actual on-device kernel execution time (not host wall-clock).

Usage:
    # Set profiler env vars
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_MID_RUN_DUMP=1
    export TT_METAL_PROFILER_CPP_POST_PROCESS=1

    python models/tt_dit/tests/models/wan2_2/bench_conv3d_fp32_reduction.py
"""

import os

import torch

import ttnn
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, prepare_conv3d_weights
from models.tt_dit.utils.tensor import typed_tensor_2dshard

WARMUP = 2
RUNS = 3


def get_device_kernel_duration_us(device):
    """Read device profiler and return kernel duration in microseconds."""
    try:
        ttnn.ReadDeviceProfiler(device)
        latest = ttnn.get_latest_programs_perf_data()
        if not latest:
            return None
        for _dev_id, programs in latest.items():
            for p in programs:
                for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
                    if key in p.program_analyses_results:
                        d = p.program_analyses_results[key].duration
                        if d is not None:
                            return d / 1000.0  # ns -> us
        return None
    except Exception:
        return None


def run_conv3d(mesh_device, inp, w, b, C_out, kernel, c_in_block, fp32_dest):
    grid_size = mesh_device.compute_with_storage_grid_size()
    padding = tuple(k // 2 for k in kernel)

    conv_config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.DataType.BFLOAT16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=4,
        H_out_block=8,
        C_out_block=32,
        C_in_block=c_in_block,
        compute_with_storage_grid_size=grid_size,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest,
        packer_l1_acc=False,
    )

    w_p, b_p = prepare_conv3d_weights(w, b, conv_config)
    tt_w = typed_tensor_2dshard(
        w_p, mesh_device, shard_mapping={0: 0, 1: 1}, layout=ttnn.TILE_LAYOUT, dtype=ttnn.DataType.BFLOAT16
    )
    tt_b = typed_tensor_2dshard(
        b_p, mesh_device, shard_mapping={0: 0, 1: 1}, layout=ttnn.TILE_LAYOUT, dtype=ttnn.DataType.BFLOAT16
    )

    durations = []
    for i in range(WARMUP + RUNS):
        tt_in = inp.permute(0, 2, 3, 4, 1)
        tt_in = conv_pad_in_channels(tt_in)
        tt_in, _ = conv_pad_height(tt_in, 1)
        tt_in = typed_tensor_2dshard(
            tt_in, mesh_device, shard_mapping={0: 2, 1: 3}, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        tt_out = ttnn.experimental.conv3d(
            input_tensor=tt_in,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            config=conv_config,
            output_channels=C_out,
            kernel_size=kernel,
            stride=(1, 1, 1),
            padding=padding,
            padding_mode="zeros",
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=ckc,
        )
        ttnn.synchronize_device(mesh_device)

        if i >= WARMUP:
            dur = get_device_kernel_duration_us(mesh_device)
            if dur is not None:
                durations.append(dur)

    r = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(1, 1), dims=(2, 3)))
    out = r.permute(0, 4, 1, 2, 3)[:, :C_out]
    return out, durations


def main():
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    torch.manual_seed(42)

    # Real Wan decoder layer shapes
    test_cases = [
        # (name, C_in, C_out, T, H, W, kernel, conservative_cin, aggressive_cin)
        ("384x384 k3 90x160", 384, 384, 1, 90, 160, (3, 3, 3), 128, 128),
        ("192x192 k3 32x32", 192, 192, 1, 32, 32, (3, 3, 3), 192, 96),
        ("96x96 k3 32x32", 96, 96, 1, 32, 32, (3, 3, 3), 96, 32),
        ("384x384 k3 32x32", 384, 384, 1, 32, 32, (3, 3, 3), 384, 128),
    ]

    for name, C_in, C_out, T, H, W, kernel, cin_cons, cin_aggr in test_cases:
        print(f"\n=== {name} ===", flush=True)
        inp = torch.randn(1, C_in, T, H, W)
        m = torch.nn.Conv3d(C_in, C_out, kernel, 1, tuple(k // 2 for k in kernel), bias=True)
        w, b = m.weight.data, m.bias.data
        with torch.no_grad():
            ref = m(inp)

        configs = [
            (f"Conservative C_in={cin_cons}", cin_cons, True),
            (f"Aggressive   C_in={cin_aggr} bf16", cin_aggr, False),
            (f"Aggressive   C_in={cin_aggr} fp32", cin_aggr, True),
        ]

        for label, c_in_block, fp32_dest in configs:
            try:
                out, durations = run_conv3d(mesh_device, inp, w, b, C_out, kernel, c_in_block, fp32_dest)
                pcc = torch.corrcoef(torch.stack([out.flatten().double(), ref.flatten().double()]))[0, 1].item()
                mae = (out - ref).abs().mean().item()

                if durations:
                    min_us = min(durations)
                    print(f"  {label}:  device={min_us:8.1f}us  PCC={pcc:.6f}  MAE={mae:.6f}", flush=True)
                else:
                    print(f"  {label}:  device=     N/A  PCC={pcc:.6f}  MAE={mae:.6f}", flush=True)
            except Exception as e:
                err_msg = str(e).split("\n")[0][:60]
                print(f"  {label}:  FAILED — {err_msg}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
