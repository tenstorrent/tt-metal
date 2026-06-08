# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-mesh TP4 (Wormhole, 4x n150) test for the dots.ocr TEXT-decoder DECODE MLP.

Decode (seq_len=1, M=batch=32). Same Megatron scheme as prefill but with the decode
memory-layout ops (see ``TTNNDotsOCRMLP.forward`` decode path in dots_ocr_mlp.py):

    fused gate+up : COLUMN(N)-split matmul, width-sharded output.
                    32 x 1536 x 17920  ->  32 x 1536 x 4480 per device
    ShardedToInterleaved : width-sharded fused output -> L1 interleaved (cheap tile-aware chunk).
    chunk (Slice)        : split [32,4480] -> gate [32,2240] / up [32,2240].
    silu(gate)*up (BinaryNg) : per device on the column shard.
    InterleavedToSharded : silu*up -> width-sharded (down's input layout).
    down          : ROW(K)-split matmul.
                    32 x 8960 x 1536  ->  32 x 2240 x 1536 per device.
    ReduceScatter : sum partials, scatter on N -> [32, 384] per device (next layer's K shard).

Op-level test: reproduces the sharding inline, reads weights off the HF module (no model-package
import). The reduce-scatter output is gathered host-side and PCC-checked against the float SwiGLU.

Run (correctness):  pytest .../test_dots_ocr_text_mlp_decode_tp.py -s
Run (Tracy):        python -m tracy -v -r -p -m pytest .../test_dots_ocr_text_mlp_decode_tp.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HIDDEN_SIZE = 1536
DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path():
    import os

    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _text_mlp_weights():
    """dots.ocr text-decoder layer-0 MLP weights (Qwen2MLP: gate/up [I,1536], down [1536,I]; no bias)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(_resolve_model_path(), trust_remote_code=True)
    cfg.num_hidden_layers = 1
    hf = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    mlp = hf.model.layers[0].mlp
    w_gate = mlp.gate_proj.weight.data.clone()
    w_up = mlp.up_proj.weight.data.clone()
    w_down = mlp.down_proj.weight.data.clone()
    del hf
    return w_gate, w_up, w_down


def _width_sharded(shape, device):
    """L1 WIDTH_SHARDED config for [1,1,M,W]; shards W across a core grid that divides W/32."""
    w_tiles = shape[-1] // 32
    grid = device.compute_with_storage_grid_size()
    cores = 1
    for c in range(min(grid.x * grid.y, w_tiles), 0, -1):
        if w_tiles % c == 0:
            cores = c
            break
    gx = min(grid.x, cores)
    gy = max(1, cores // gx)
    while gx * gy > cores or w_tiles % (gx * gy) != 0:
        gx -= 1
        gy = max(1, cores // gx) if gx else 1
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_dots_ocr_text_mlp_decode_tp4(mesh_device):
    """TP4 text-decoder DECODE MLP (M=32): fused col(N)-split gate/up + row(K)-split down + reduce_scatter."""
    tp = int(mesh_device.get_num_devices())
    if tp < 4:
        pytest.skip("text-MLP decode TP4 requires 4 devices")
    tp = 4
    if tuple(mesh_device.shape) != (1, 4):
        mesh_device.reshape(ttnn.MeshShape(1, 4))

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    mem = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG

    w_gate, w_up, w_down = _text_mlp_weights()
    inter = int(w_gate.shape[0])  # 8960
    assert inter % tp == 0 and HIDDEN_SIZE % tp == 0
    shard = inter // tp  # 2240

    m = 32  # decode: batch=32, seq_len=1
    x_torch = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16)

    xf = x_torch.float()
    ref = (F.silu(xf @ w_gate.t().float()) * (xf @ w_up.t().float())) @ w_down.t().float()  # [M, 1536]

    ckc = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    def up(t, dtype, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem, mesh_mapper=mapper
        )

    x_tt = up(x_torch.reshape(1, 1, m, HIDDEN_SIZE), ttnn.bfloat8_b, replicate)

    # --- fused gate+up: column(N)-split, width-sharded output (decode layout) ---
    fused_slabs = [
        torch.cat([w_gate[d * shard : (d + 1) * shard], w_up[d * shard : (d + 1) * shard]], dim=0)  # [4480, 1536]
        for d in range(tp)
    ]
    fused_w_tt = up(torch.stack([s.t().contiguous() for s in fused_slabs], dim=0), ttnn.bfloat8_b, shard0)
    gate_up = ttnn.linear(
        x_tt,
        fused_w_tt,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=ckc,
    )

    # ShardedToInterleaved -> chunk (Slice) -> silu*up (BinaryNg)
    gate_up = ttnn.sharded_to_interleaved(gate_up, l1)
    gate, up_t = ttnn.chunk(gate_up, 2, dim=-1)  # each [1,1,32,2240]
    ttnn.deallocate(gate_up)
    h = ttnn.mul(
        gate,
        up_t,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        fast_and_approximate_mode=True,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
    )
    ttnn.deallocate(gate)
    ttnn.deallocate(up_t)

    # InterleavedToSharded (down's input layout)
    h = ttnn.to_memory_config(h, _width_sharded([1, 1, m, shard], mesh_device))

    # --- down: row(K)-split + reduce_scatter ---
    down_slabs = [w_down[:, d * shard : (d + 1) * shard].t().contiguous() for d in range(tp)]  # [2240, 1536]/dev
    down_w_tt = up(torch.stack(down_slabs, dim=0), ttnn.bfloat8_b, shard0)
    partial = ttnn.linear(
        h, down_w_tt, dtype=ttnn.bfloat16, memory_config=mem, compute_kernel_config=ckc
    )  # [1,1,32,1536]
    ttnn.deallocate(h)
    out_rs = ttnn.reduce_scatter(
        partial, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(partial)

    out_full = (
        ttnn.to_torch(out_rs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        .float()
        .reshape(m, HIDDEN_SIZE)
    )
    passing, value = comp_pcc(ref, out_full, 0.98)
    print(f"\n[text MLP decode TP4] M={m} inter={inter}  fused col(N) gate/up + row(K) down  PCC={value}")
    assert passing, f"TP4 text MLP decode PCC below 0.98: {value}"
