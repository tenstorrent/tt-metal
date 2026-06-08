# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-mesh TP4 (Wormhole, 4x n150) test for the dots.ocr TEXT-decoder prefill MLP.

The text decoder MLP is a fused-gate/up SwiGLU (see ``TTNNDotsOCRMLP`` in dots_ocr_mlp.py).
Its Megatron tensor-parallel prefill op sequence, at S=2816 (hidden=1536, intermediate=8960):

    fused gate+up : COLUMN-parallel -> one matmul on the [gate|up] weight (N=2*8960=17920).
                    2816 x 1536 x 17920  ->  2816 x 1536 x 4480 per device
                    (per device = [gate_cols_d (2240) | up_cols_d (2240)]).
    chunk         : split the fused [M,4480] into gate [M,2240] and up [M,2240] (2 slices).
    silu(gate)*up : per device on the column shard (BinaryNg, no comm).
    down          : ROW-parallel -> split the K (intermediate=8960) dim.
                    2816 x 8960 x 1536  ->  2816 x 2240 x 1536 per device.
    ReduceScatter : sum the partials and scatter on N -> [M, 384] per device. NO all_gather --
                    the hidden stays K-sharded (384/dev) as the next layer's input.

This is an op-level test (reproduces the sharding inline; reads weights off the HF module) so
it does NOT import the model package. The reduce-scatter output is gathered host-side
(``ConcatMeshToTensor`` on the scattered N dim reconstructs the full summed [M,1536]) and
PCC-checked against the float SwiGLU. Matmuls use the model's tuned 2D-mcast program config.

Run (correctness):  pytest .../test_dots_ocr_text_mlp_prefill_tp.py -s
Run (Tracy):        python -m tracy -v -r -p -m pytest .../test_dots_ocr_text_mlp_prefill_tp.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc
from models.experimental.tt_symbiote.tests._vision_tp_matmul import vision_matmul_program_config

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


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_dots_ocr_text_mlp_prefill_tp4(mesh_device):
    """TP4 text-decoder prefill MLP: fused col(N)-split gate/up + row(K)-split down + reduce_scatter."""
    tp = int(mesh_device.get_num_devices())
    if tp < 4:
        pytest.skip("text-MLP prefill TP4 requires 4 devices")
    tp = 4
    if tuple(mesh_device.shape) != (1, 4):
        mesh_device.reshape(ttnn.MeshShape(1, 4))

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    mem = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG

    w_gate, w_up, w_down = _text_mlp_weights()
    inter = int(w_gate.shape[0])  # 8960
    assert int(w_gate.shape[1]) == HIDDEN_SIZE and int(w_down.shape[0]) == HIDDEN_SIZE
    assert inter % tp == 0 and HIDDEN_SIZE % tp == 0
    shard = inter // tp  # 2240

    m = 2816
    x_torch = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16)

    # Float reference SwiGLU.
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

    # --- fused gate+up: column(N)-split. Per device slab = [gate_cols_d | up_cols_d]. ---
    fused_slabs = [
        torch.cat([w_gate[d * shard : (d + 1) * shard], w_up[d * shard : (d + 1) * shard]], dim=0)  # [4480, 1536]
        for d in range(tp)
    ]
    fused_w = torch.stack([s.t().contiguous() for s in fused_slabs], dim=0)  # [tp, 1536, 4480]
    fused_w_tt = up(fused_w, ttnn.bfloat8_b, shard0)
    gu_pc = vision_matmul_program_config(mesh_device, m, HIDDEN_SIZE, 2 * shard)  # M x 1536 x 4480
    gate_up = ttnn.linear(
        x_tt, fused_w_tt, dtype=ttnn.bfloat8_b, memory_config=l1, compute_kernel_config=ckc, program_config=gu_pc
    )

    # --- chunk into gate / up halves (per-device column shards) ---
    gate, up_t = ttnn.chunk(gate_up, 2, dim=-1)  # each [1,1,M,2240]
    ttnn.deallocate(gate_up)

    # --- silu(gate) * up per device, L1 ---
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

    # --- down: row(K)-split. W is [out=1536, in=8960]; W.T=[in,out], shard in(K). ---
    down_slabs = [w_down[:, d * shard : (d + 1) * shard].t().contiguous() for d in range(tp)]  # [2240, 1536]/dev
    down_w_tt = up(torch.stack(down_slabs, dim=0), ttnn.bfloat8_b, shard0)
    down_pc = vision_matmul_program_config(mesh_device, m, shard, HIDDEN_SIZE)  # M x 2240 x 1536
    partial = ttnn.linear(
        h, down_w_tt, dtype=ttnn.bfloat16, memory_config=l1, compute_kernel_config=ckc, program_config=down_pc
    )
    ttnn.deallocate(h)

    # --- reduce_scatter ONLY: sum partials, scatter on N -> [M, 384]/dev (next layer's K shard) ---
    out_rs = ttnn.reduce_scatter(
        partial, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(partial)

    # Gather the scattered N shards (host) to reconstruct the full summed [M, 1536].
    out_full = (
        ttnn.to_torch(out_rs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        .float()
        .reshape(m, HIDDEN_SIZE)
    )

    passing, value = comp_pcc(ref, out_full, 0.98)
    print(f"\n[text MLP prefill TP4] M={m} inter={inter}  fused col(N) gate/up + row(K) down  PCC={value}")
    assert passing, f"TP4 text MLP prefill PCC below 0.98: {value}"
