# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.1 device MLA vs the BF16 functional reference (MLACPU) — the bf16-golden discriminator.

The whole-model test (test_glm_model_vs_reference) compares the device against the fp8 GPU trace
and bottoms out at ~0.88 mid-stack. That is the bf16(device)-vs-fp8(trace) cross-precision gap, NOT
a bug — DS-V3/Kimi share it (their test hardcodes TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88). To prove
the device MLA is faithful you must compare it to a BF16 reference, exactly like DS/Kimi's
test_prefill_transformer compares against a bf16-HF model at PCC_THRESHOLD=0.99.

This test runs the device GLM MLA at SEQ=2048 (≤ index_topk=2048 → every query selects ALL its causal
keys → the DENSE-equivalent path, no top-2048 sparsity) so the in-repo MLACPU golden is tractable on
CPU (the full 5120 sparse forward is CPU-hopeless, ~10 min/layer). It teacher-forces the GPU trace's
mla_input[:SEQ] into BOTH the device MLA and MLACPU and PCCs the two outputs.

Expectation: device-vs-bf16 ≥ 0.996 (matching DS-V3's mla_pcc_threshold=0.996 / Kimi's 0.995). A pass
confirms the GLM MLA introduces no error beyond the documented bf8/bf16-vs-fp8 precision the family
already accepts. The sparse band (rows≥2048) is validated separately: the sparse_mla op vs the trace
= 0.9997 flat (test_sparse_sdpa_vs_gpu.py) and the device-vs-trace band-split shows rows≥2048 ≈
rows<2048 (test_mla_output_device_vs_reference).

SEQ override: GLM_MLA_BF16_SEQ (default 2048; keep ≤2048 to stay dense + CPU-tractable).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device
from models.demos.deepseek_v32.tests.test_mla import WEIGHT_NAME_MAP
from models.demos.deepseek_v32.tests.test_vs_gpu_ref import (
    _DEVICE_PARAMS,
    _build_glm_cpu_reference,
    _config_for,
    _glm_model_args,
    _skip_unsupported,
    load_reference,
)
from models.demos.deepseek_v32.tt.mla import ttMLA

# Dense-regime sequence length: ≤ index_topk=2048 so the indexer selects all causal keys (dense-equiv)
# and the CPU MLACPU golden is tractable. SEQ=2048 → MLACPU forward ~2-3 min/layer.
SEQ = int(os.environ.get("GLM_MLA_BF16_SEQ", "2048"))
# device-vs-bf16 threshold: DS-V3 mla=0.996, Kimi mla=0.995 (model_variants.py) — GLM should match.
MLA_BF16_PCC = 0.996

# Layers to check: a dense MLA layer (0) + MoE-stack layers spanning the would-be trough.
GLM_MLA_BF16_LAYERS = [int(x) for x in os.environ.get("GLM_MLA_BF16_LAYERS", "0,30,60").split(",")]


def _run_glm_mla_device(layer, mesh_device, mla_cpu, args, mla_in):
    """Device GLM ttMLA forward at SEQ on the (sliced) trace mla_input. Returns out [1, SEQ, hidden].
    Mirrors test_vs_gpu_ref._run_device_forward but with a local SEQ and a caller-supplied MLACPU
    (so the device weights and the bf16 golden come from the SAME reference instance)."""
    config = _config_for("glm_5_1")
    sp_axis, tp_axis = 0, 1
    sd = mla_cpu.state_dict()
    weights = {v3: sd[cpu].clone() for cpu, v3 in WEIGHT_NAME_MAP.items()}
    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=SEQ,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        index_args=_glm_model_args(),
    )
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=SEQ,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors(SEQ)

    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_x = ttnn.from_torch(
        mla_in.reshape(1, 1, SEQ, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    out = mla.forward(tt_x, rope_tensors, kvpe_cache)
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[
        0
    ]  # [1, SEQ, hidden]
    return out_t


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("layer", GLM_MLA_BF16_LAYERS, ids=[f"glm_5_1-L{_l}" for _l in GLM_MLA_BF16_LAYERS])
@pytest.mark.timeout(0)
def test_glm_mla_device_vs_bf16(mesh_device, layer, device_params):
    """Device GLM MLA @ SEQ=2048 (dense) vs the BF16 MLACPU golden, same teacher-forced input.
    PASS ⇒ the device MLA is faithful to bf16 (no bug); the whole-model 0.88 trough is the
    documented bf16-vs-fp8-trace gap, not this op."""
    _skip_unsupported("glm_5_1", mesh_device)  # skips tp>2
    # GLM MLA needs exactly tp=2: tp=1 puts all 64 q-heads on one chip → the MLA circular buffers
    # (~2.2 MB) exceed Blackhole L1 (1.5 MB). 64/tp=32 heads/chip (tp=2) is the only fit.
    if mesh_device.shape[1] != 2:
        pytest.skip(f"GLM MLA needs tp=2 (mesh tp={mesh_device.shape[1]}): tp=1 OOMs L1, tp>2 fails sparse_sdpa")
    assert SEQ <= 2048, f"SEQ={SEQ} > 2048 would invoke the sparse path (CPU golden intractable)"

    ref = load_reference("glm_5_1", layer)
    mla_in = ref["mla_in"][:SEQ].to(torch.bfloat16)  # [SEQ, hidden], trace mla_input

    # --- BF16 golden: MLACPU (functional, no fp8) on the same input ---
    args, mla_cpu = _build_glm_cpu_reference(layer)
    freqs = precompute_freqs_cis(args)[:SEQ]
    mask = torch.full((SEQ, SEQ), float("-inf")).triu_(1)
    with torch.no_grad():
        golden = mla_cpu.forward(mla_in.unsqueeze(0), 0, freqs, mask)[0]  # [SEQ, hidden]

    # --- Device MLA (same reference weights) ---
    out = _run_glm_mla_device(layer, mesh_device, mla_cpu, args, mla_in)  # [1, SEQ, hidden]
    got = out[0] if out.dim() == 3 else out

    _, pcc = comp_pcc(golden.float(), got.float(), 0)
    logger.info(f"[GLM L{layer}] device MLA vs BF16 MLACPU @ SEQ={SEQ}: PCC={pcc}")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= MLA_BF16_PCC, f"GLM L{layer} device-vs-bf16 MLA PCC {pcc} < {MLA_BF16_PCC} (real device bug)"
