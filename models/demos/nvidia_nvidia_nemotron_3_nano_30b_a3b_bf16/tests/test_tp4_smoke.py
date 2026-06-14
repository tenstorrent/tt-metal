# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 smoke test — verify mamba2_layer_forward runs end-to-end on 4-chip mesh."""

import os
import sys

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import ttnn


def run_smoke():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_layer import mamba2_layer_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import (
        _host_rep,
        close_device_tp4,
        open_device_tp4,
    )

    B = 1
    p = "backbone.layers.0"

    print("Opening 4-chip mesh...", flush=True)
    mesh = open_device_tp4()
    ttnn.synchronize_device(mesh)
    print("  SYNC OK: mesh open", flush=True)

    wc = WeightCache()

    hidden = torch.randn(B, 1, 2688, dtype=torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.synchronize_device(mesh)
    print("  SYNC OK: hidden_states uploaded", flush=True)

    print("Running mamba2_layer_forward (full TTNN-native path)...", flush=True)
    out_tt = mamba2_layer_forward(
        mesh,
        hidden_tt,
        norm_weight=wc[f"{p}.norm.weight"],
        in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
        conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
        conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
        dt_bias=wc[f"{p}.mixer.dt_bias"],
        A_log=wc[f"{p}.mixer.A_log"],
        norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
        D=wc[f"{p}.mixer.D"],
        out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
    )
    ttnn.synchronize_device(mesh)
    print("  SYNC OK: mamba2_layer_forward", flush=True)

    out_cpu = _host_rep(out_tt, mesh, B)
    print(f"  Output shape: {out_cpu.shape}, finite: {torch.isfinite(out_cpu).all().item()}", flush=True)
    assert out_cpu.shape == (B, 1, 2688), f"unexpected shape {out_cpu.shape}"
    assert torch.isfinite(out_cpu).all(), "output contains NaN/Inf"

    print("MAMBA2 LAYER DONE — PASSED.", flush=True)
    close_device_tp4(mesh)


if __name__ == "__main__":
    run_smoke()
