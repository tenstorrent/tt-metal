#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Wall-clock timing: gdn_prefill_fused vs gdn_prefill_ttnn_ops.

Usage (from tt-metal root):
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen35_27b/tt/tests/bench_gdn_prefill.py
"""

import os
import time

import torch

os.environ.setdefault("ARCH_NAME", "blackhole")
HF_MODEL = (
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)
os.environ.setdefault("HF_MODEL", HF_MODEL)

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused
from models.demos.qwen35_27b.tt.model import create_qwen35_model

WARMUP = 3
REPEATS = 5
TOKEN_COUNTS = [32, 64, 256, 512, 1024]
MESH_SHAPE = ttnn.MeshShape(1, 4)


def _to_mesh(t, mesh_device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _unshard(t):
    """Convert sharded layout to DRAM-interleaved (mirrors test helper)."""
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def main():
    mesh_device = ttnn.open_mesh_device(
        MESH_SHAPE,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    print("Loading model (3 layers)...")
    model = create_qwen35_model(
        mesh_device,
        model_path=HF_MODEL,
        max_batch_size=32,
        max_seq_len=2048,
        dtype=ttnn.bfloat8_b,
        n_layers=3,
    )
    args = model.args
    gdn_layer_idx = next(i for i in range(args.n_layers) if args.layer_types[i] == "linear_attention")
    gdn = model.layers[gdn_layer_idx].attention
    tw = gdn.tw

    Nv_TP = gdn.Nv_TP
    Nk_TP = gdn.Nk_TP
    Dk = gdn.Dk
    Dv = gdn.Dv
    qkv_dim_tp = gdn.qkv_dim_tp
    key_dim_tp = gdn.key_dim_tp
    num_pairs = Nv_TP  # B=1
    repeat_factor = Nv_TP // Nk_TP

    print(f"GDN params: Nv_TP={Nv_TP}, Nk_TP={Nk_TP}, Dk={Dk}, Dv={Dv}, num_pairs={num_pairs}")
    print(f"  qkv_dim_tp={qkv_dim_tp}, key_dim_tp={key_dim_tp}, repeat_factor={repeat_factor}")

    def run_once(N, use_ttnn_ops):
        conv_bf16 = torch.randn(1, N, qkv_dim_tp, dtype=torch.bfloat16) * 0.1
        a_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1
        b_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1

        conv_3d = _unshard(_to_mesh(conv_bf16, mesh_device))
        a_3d = _unshard(_to_mesh(a_bf16, mesh_device))
        b_3d = _unshard(_to_mesh(b_bf16, mesh_device))
        st = _to_mesh(torch.zeros(num_pairs, Dk, Dv, dtype=torch.bfloat16), mesh_device)
        out = _to_mesh(torch.zeros(num_pairs * N, 1, Dv, dtype=torch.bfloat16), mesh_device)

        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()

        gdn_prefill_fused(
            conv_3d,
            a_3d,
            b_3d,
            gdn.neg_exp_A,
            tw["dt_bias"],
            tw["norm_w"],
            gdn.scale_tt,
            gdn.rms_scale_tt,
            gdn.rms_eps_tt,
            st,
            out,
            num_pairs=num_pairs,
            num_tokens=N,
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
            use_ttnn_ops=use_ttnn_ops,
        )

        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()

        ttnn.deallocate(conv_3d)
        ttnn.deallocate(a_3d)
        ttnn.deallocate(b_3d)
        ttnn.deallocate(st)
        ttnn.deallocate(out)

        return (t1 - t0) * 1000.0  # ms

    print()
    print(f"{'N':>6}  {'path':<12}  {'mean_ms':>10}  {'min_ms':>10}  {'max_ms':>10}")
    print("-" * 58)

    for N in TOKEN_COUNTS:
        for use_ttnn_ops, label in [(False, "fused"), (True, "ttnn_ops")]:
            # Warmup
            for _ in range(WARMUP):
                run_once(N, use_ttnn_ops)

            # Timed
            times = [run_once(N, use_ttnn_ops) for _ in range(REPEATS)]
            mean_ms = sum(times) / len(times)
            min_ms = min(times)
            max_ms = max(times)
            print(f"{N:>6}  {label:<12}  {mean_ms:>10.3f}  {min_ms:>10.3f}  {max_ms:>10.3f}")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
