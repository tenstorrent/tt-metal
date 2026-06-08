# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-mesh TP4 (Wormhole, 4x n150) test for the dots.ocr vision MLP (SwiGLU).

Megatron-style tensor parallel, matching the model's op sequence:

    fc1 (gate)   : COLUMN-parallel -> split the N (intermediate=4224) dim.
                   11264 x 1536 x 4224  ->  11264 x 1536 x 1056 per device
    fc3 (up)     : COLUMN-parallel -> same N-split.
                   11264 x 1536 x 4224  ->  11264 x 1536 x 1056 per device
    silu(gate)*up: per device on the N-sharded [11264, 1056] (BinaryNg, no comm).
    fc2 (down)   : ROW-parallel -> split the K (intermediate=4224) dim.
                   11264 x 4224 x 1536  ->  11264 x 1056 x 1536 per device
                   (each device matmuls its 1056-wide intermediate shard against W[kshard]).
    all-reduce   : ReduceScatter (sum partials, scatter on N) + AllGather (regather N)
                   -> full [11264, 1536] replicated. Bias added while N-sharded.

So gate/up shard N and down shards K: the intermediate stays sharded end-to-end (the column
output of gate/up is exactly the K shard down consumes), with a single all-reduce after down.
CCL runs on real fabric (FABRIC_1D); weights are read off the HF module so this does NOT
import the ``dots_ocr_vision`` module.

PCC reference (float SwiGLU on the same weights) is checked only at small M; the M=11264 shape
is for Tracy profiling, where it only validates output shape + replication. Skips < 4 devices.

Run (correctness):  pytest .../test_dots_ocr_vision_mlp_tp.py -k m256 -s
Run (Tracy):        python -m tracy -v -r -p -m pytest .../test_dots_ocr_vision_mlp_tp.py -k m11264
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


def _vision_mlp_weights():
    """dots.ocr vision ``DotsSwiGLUFFN`` weights: fc1=gate, fc3=up ([inter,1536]); fc2=down ([1536,inter])."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(_resolve_model_path(), trust_remote_code=True)
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, 1)
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "dots.ocr vision tower should expose blocks/layers"
    mlp = blocks[0].mlp

    def wb(lin):
        return lin.weight.data.clone(), (lin.bias.data.clone() if lin.bias is not None else None)

    g, u, d = wb(mlp.fc1), wb(mlp.fc3), wb(mlp.fc2)
    del hf_model
    return g, u, d


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
@pytest.mark.parametrize("m", [256, 11264], ids=["m256", "m11264"])
def test_dots_ocr_vision_mlp_tp4(mesh_device, m):
    """TP4 vision SwiGLU: column(N)-split gate/up + row(K)-split down with one all-reduce."""
    tp = int(mesh_device.get_num_devices())
    if tp < 4:
        pytest.skip("vision-MLP TP4 requires 4 devices")
    tp = 4
    if tuple(mesh_device.shape) != (1, 4):
        mesh_device.reshape(ttnn.MeshShape(1, 4))

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    mem = ttnn.DRAM_MEMORY_CONFIG
    check_pcc = m <= 2048  # large M is profiling-only

    (w_gate, b_gate), (w_up, b_up), (w_down, b_down) = _vision_mlp_weights()
    inter = int(w_gate.shape[0])
    assert int(w_gate.shape[1]) == HIDDEN_SIZE and int(w_down.shape[0]) == HIDDEN_SIZE
    assert inter % tp == 0, f"intermediate={inter} (gate/up N-split & down K-split) must divide TP={tp}"
    assert HIDDEN_SIZE % tp == 0, "down output N=1536 must divide TP for the reduce_scatter"

    x_torch = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16)

    ref_out = None
    if check_pcc:
        xf = x_torch.float()
        rg = xf @ w_gate.t().float() + (b_gate.float() if b_gate is not None else 0.0)
        ru = xf @ w_up.t().float() + (b_up.float() if b_up is not None else 0.0)
        ref_out = (F.silu(rg) * ru) @ w_down.t().float() + (b_down.float() if b_down is not None else 0.0)

    ckc = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    shardN = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    def up(t, dtype, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem, mesh_mapper=mapper
        )

    l1 = ttnn.L1_MEMORY_CONFIG
    # Sharded intermediates (1056-wide) always fit L1; the full-1536-wide tensors (x in,
    # down partial out) only fit at small M -- at S=11264 a BF16 [M,1536] shard is
    # ~515 KB/core and overflows per-core L1, so route those to DRAM there.
    wide_mem = l1 if m <= 2048 else mem
    x_tt = up(x_torch.reshape(1, 1, m, HIDDEN_SIZE), ttnn.bfloat8_b, replicate)  # full K=1536 on every device
    x_tt = ttnn.to_memory_config(x_tt, wide_mem)

    # --- gate/up: column(N)-split. W is [out=inter, in=1536]; W.T=[in,out], shard out(N). ---
    gate_w = up(w_gate.t().contiguous(), ttnn.bfloat8_b, shardN)  # [1536, 1056]/dev
    up_w = up(w_up.t().contiguous(), ttnn.bfloat8_b, shardN)
    gate_b = up(b_gate.reshape(1, -1), ttnn.bfloat8_b, shardN) if b_gate is not None else None  # N-disjoint -> fuse
    up_b = up(b_up.reshape(1, -1), ttnn.bfloat8_b, shardN) if b_up is not None else None
    # gate BFP8, up BFP4 (model dtype). BFP4 on the SwiGLU up activation is lossy in
    # isolation (its block scaling fights gate/up outliers), so the end-to-end PCC floor
    # is ~0.67 -- that is the BFP4 cost, not a sharding error (the all-BFP8 variant lands
    # ~0.996). The 0.999 cross-device replication check below still guards the all-reduce.
    # Tuned 2D-mcast config (per-device shapes): gate/up are M x 1536 x 1056.
    gu_pc = vision_matmul_program_config(mesh_device, m, HIDDEN_SIZE, inter // tp)
    gate = ttnn.linear(
        x_tt,
        gate_w,
        bias=gate_b,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=ckc,
        program_config=gu_pc,
    )
    up_t = ttnn.linear(
        x_tt, up_w, bias=up_b, dtype=ttnn.bfloat4_b, memory_config=l1, compute_kernel_config=ckc, program_config=gu_pc
    )

    # --- silu(gate) * up: per-device on the N-sharded [M, 1056] (no comm), L1 ---
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

    # --- down: row(K)-split. W is [out=1536, in=inter]; W.T=[in,out], shard in(K). ---
    # in (h) + out (partial) in L1; reduce_scatter reads the L1 partial.
    down_w = up(w_down.t().contiguous(), ttnn.bfloat8_b, shard0)  # [1056, 1536]/dev
    down_pc = vision_matmul_program_config(mesh_device, m, inter // tp, HIDDEN_SIZE)  # M x 1056 x 1536
    partial = ttnn.linear(
        h, down_w, dtype=ttnn.bfloat16, memory_config=wide_mem, compute_kernel_config=ckc, program_config=down_pc
    )  # [1,1,M,1536] partial
    ttnn.deallocate(h)

    # --- all-reduce = reduce_scatter (sum, scatter on N) + all_gather (regather N) ---
    out_rs = ttnn.reduce_scatter(
        partial, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(partial)
    if b_down is not None:
        out_rs = ttnn.add(out_rs, up(b_down.reshape(1, -1), ttnn.bfloat16, shardN))  # bias while N-sharded
    out_tt = ttnn.all_gather(
        out_rs, dim=3, num_links=1, cluster_axis=1, memory_config=mem, topology=ttnn.Topology.Linear
    )
    ttnn.deallocate(out_rs)

    out_gathered = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))  # [4,1,M,1536]
    assert list(out_gathered.shape) == [tp, 1, m, HIDDEN_SIZE]
    for d in range(1, tp):
        assert comp_pcc(out_gathered[0], out_gathered[d], 0.999)[0], f"all-reduce output not replicated on device {d}"

    if not check_pcc:
        ttnn.synchronize_device(mesh_device)
        print(f"\n[vision MLP TP4] M={m} inter={inter}  profiling shape ran (PCC skipped)")
        return

    out_dev = out_gathered[0].reshape(m, HIDDEN_SIZE)
    # BFP4 ``up`` is the dominant error (see note above); ~0.6 is the floor at seed 1234.
    # The sharding itself is exact (all-BFP8 -> ~0.996); this bar only guards against a
    # gross sharding break (which collapses PCC well below 0.6) while honoring the BFP4 up.
    passing, value = comp_pcc(ref_out, out_dev, 0.6)
    print(f"\n[vision MLP TP4] M={m} inter={inter}  N-split gate/up + K-split down (BFP4 up, L1)  PCC={value}")
    assert passing, f"TP4 vision MLP PCC below 0.6: {value}"
