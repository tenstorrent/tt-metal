# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.1 MoE correctness — the host reference reproduces the GPU trace's MoE contribution.

The whole-layer block's MoE contribution is exactly  decoder_output - decoder_input - mla_output
(the second residual add). This test reconstructs that contribution with the in-repo host MoE
reference (_host_glm_moe: fp32 single-group noaux_tc routing + the fp8-dequant bf16 experts) fed the
derived gate input  ffn_norm(decoder_input + mla_output), and PCCs it against the GPU trace. A pass
confirms the GLM routing (single-group top-8 of 256 + the mean-centered e_score_correction_bias),
expert math, and route_scale all match the GPU — independent of the device.

Established result (memory): host-MoE reconstruction of GPU decoder_output ≈ 0.99998 (L30). The
DEVICE MoE (bf4 routed weights + bf8 activations) vs this bf16 host reference is 0.977-0.991 — pure
quantization, in-line with DS-V3's moe_pcc_threshold=0.94 / Kimi's 0.971 (model_variants.py), i.e. no
dispatch/gate/combine bug. This test pins the HOST reference (the bf16 ceiling); the device-vs-host
bf4 gap is covered by the block test.

Host-only (no device); ~30 GB expert weights load JIT on first run per MoE layer.
"""

import os

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v32.reference_cpu.weights import load_moe_block_weights
from models.demos.deepseek_v32.tests.test_vs_gpu_ref import GLM_REPO, _host_glm_moe, _load_block_trace, load_reference

# Host MoE reconstruction of the GPU MoE contribution. fp32 routing + bf16 experts vs the fp8 GPU →
# near-exact (the residual/norm bf16 storage is the only gap). Memory: 0.99998 @ L30.
MOE_HOST_PCC = 0.995

# MoE layers (>= NUM_DENSE_LAYERS=3). Pick layers whose decoder_io trace is LFS-pulled.
GLM_MOE_LAYERS = [int(x) for x in os.environ.get("GLM_MOE_LAYERS", "30,60").split(",")]


def _rmsnorm(x, w, eps=GLM51Config.RMS_NORM_EPS):
    """GLM RMSNorm in fp32: x * rsqrt(mean(x^2) + eps) * weight."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


@pytest.mark.parametrize("layer", GLM_MOE_LAYERS, ids=[f"glm_5_1-L{_l}" for _l in GLM_MOE_LAYERS])
@pytest.mark.timeout(0)
def test_glm_moe_host_vs_gpu(layer):
    """Host MoE(gate_input) vs the GPU trace's MoE contribution = decoder_output - decoder_input -
    mla_output. PASS ⇒ GLM routing + experts + route_scale reproduce the GPU (no model-formulation
    or routing bug); the device's bf4/bf8 gap below this is pure quantization."""
    assert layer >= GLM51Config.NUM_DENSE_LAYERS, f"L{layer} is a dense MLP layer, not MoE"

    blk = _load_block_trace(layer)  # block_in (= decoder_input), block_out (= decoder_output)
    ref = load_reference("glm_5_1", layer)  # mla_out (the MLA module output)
    decoder_in = blk["block_in"].float()  # [S, emb]
    decoder_out = blk["block_out"].float()  # [S, emb]
    mla_out = ref["mla_out"].float()  # [S, emb]

    mw = load_moe_block_weights(layer, repo=GLM_REPO)

    # Derive the gate input the GPU MoE saw: ffn_norm over the post-attention residual.
    gate_input = _rmsnorm(decoder_in + mla_out, mw["ffn_norm_weight"])
    # The MoE's contribution to the layer output (the second residual add).
    moe_golden = decoder_out - decoder_in - mla_out

    moe_pred = _host_glm_moe(gate_input, mw)

    _, pcc = comp_pcc(moe_golden, moe_pred, 0)
    logger.info(f"[GLM L{layer}] host MoE vs GPU MoE-contribution: PCC={pcc}")
    assert pcc >= MOE_HOST_PCC, f"GLM L{layer} host MoE PCC {pcc} < {MOE_HOST_PCC} (routing/expert/formulation bug)"
