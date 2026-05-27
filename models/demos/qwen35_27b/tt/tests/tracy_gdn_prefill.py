#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tracy device-kernel profiling for gdn_prefill fused vs ttnn_ops.

Usage (from tt-metal root):
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    # Fused kernel:
    USE_TTNN_OPS=0 NUM_TOKENS=64 python -m tracy -p -v -r -o /tmp/tracy_fused \
        models/demos/qwen35_27b/tt/tests/tracy_gdn_prefill.py

    # TTNN ops:
    USE_TTNN_OPS=1 NUM_TOKENS=64 python -m tracy -p -v -r -o /tmp/tracy_ttnn \
        models/demos/qwen35_27b/tt/tests/tracy_gdn_prefill.py
"""

import os

import torch

import ttnn

os.environ.setdefault("ARCH_NAME", "blackhole")
HF_MODEL = (
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)
os.environ.setdefault("HF_MODEL", HF_MODEL)

USE_TTNN_OPS = os.environ.get("USE_TTNN_OPS", "0") == "1"
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "64"))
MESH_SHAPE = ttnn.MeshShape(1, 4)

from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _to_mesh(t, mesh_device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


mesh_device = ttnn.open_mesh_device(
    MESH_SHAPE,
    dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
)

print(f"Loading model (3 layers)...")
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
num_pairs = Nv_TP
repeat_factor = Nv_TP // Nk_TP
N = NUM_TOKENS

print(f"path={'ttnn_ops' if USE_TTNN_OPS else 'fused'}, N={N}")

torch.manual_seed(42)
conv_bf16 = torch.randn(1, N, qkv_dim_tp, dtype=torch.bfloat16) * 0.1
a_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1
b_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1


def run():
    conv_3d = _unshard(_to_mesh(conv_bf16, mesh_device))
    a_3d = _unshard(_to_mesh(a_bf16, mesh_device))
    b_3d = _unshard(_to_mesh(b_bf16, mesh_device))
    st = _to_mesh(torch.zeros(num_pairs, Dk, Dv, dtype=torch.bfloat16), mesh_device)
    out = _to_mesh(torch.zeros(num_pairs * N, 1, Dv, dtype=torch.bfloat16), mesh_device)

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
        use_ttnn_ops=USE_TTNN_OPS,
    )
    ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(conv_3d)
    ttnn.deallocate(a_3d)
    ttnn.deallocate(b_3d)
    ttnn.deallocate(st)
    ttnn.deallocate(out)


# Warmup (not profiled — Tracy's -p flag captures only enabled zones)
print("Warming up...")
run()
run()

print("Profiled run...")
run()
print("Done.")

ttnn.close_mesh_device(mesh_device)
