# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Component test for TtQwen36GatedDeltaNet.

Reference is HuggingFace's own Qwen3_5GatedDeltaNet, not a hand-written copy --
an independent source of truth, so a misreading of the algorithm can't hide by
being wrong in both places. Neither flash-linear-attention nor causal-conv1d is
installed here, so HF falls back to its pure-torch path and runs on CPU.

Note the reference takes the CHUNKED path for T > 1 while we run the RECURRENT
one, so this compares two different algorithms, not two copies of one.

Weights come from HF's random init, so this checks arithmetic, not accuracy on
real data. `to_device` below is a preview of the eventual weight loader.

Single chip for now. 8xP150 means adding shapes to the mesh_device parametrize
and a mesh_mapper in to_device.

Run:
    pytest models/experimental/qwen_3_27b/tests/test_gated_deltanet.py -v
"""

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.qwen_3_27b.tt.tt_gated_deltanet import (
    CONV_DIM,
    CONV_KERNEL,
    D,
    NUM_V_HEADS,
    TtQwen36GatedDeltaNet,
)

CONFIG_PATH = "models/tt_transformers/model_params/Qwen3.6-27B"
PCC_THRESHOLD = 0.99

# (batch, seq_len). 32 is under one chunk, 128 is two full chunks.
SHAPES = [(1, 32), (1, 128)]
SHAPE_IDS = ["b1_t32", "b1_t128"]


def build_reference(layer_idx: int = 0) -> Qwen3_5GatedDeltaNet:
    """HF layer at real Qwen3.6-27B dims, with sanely scaled random weights."""
    cfg = AutoConfig.from_pretrained(CONFIG_PATH)
    ref = Qwen3_5GatedDeltaNet(getattr(cfg, "text_config", cfg), layer_idx=layer_idx).eval()

    with torch.no_grad():
        for p in ref.parameters():
            if p.dim() >= 2:  # the projections and the conv weight
                p.normal_(0, 0.02)  # config.initializer_range
        # HF inits A_log as log(U(0,16)); clamp off zero so log() stays finite.
        ref.A_log.copy_(torch.log(torch.empty(NUM_V_HEADS).uniform_(0.5, 16.0)))
        ref.dt_bias.normal_(0, 0.5)
        ref.norm.weight.normal_(1.0, 0.1)  # plain gain, centered on 1 -- not zero-centered
    return ref


def to_device(device, ref: Qwen3_5GatedDeltaNet) -> TtQwen36GatedDeltaNet:
    """Bridge an HF state_dict into the module's injected-tensor constructor."""
    sd = ref.state_dict()

    def linear(w):
        """nn.Linear stores [out, in]; ttnn.linear wants [in, out]."""
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def per_head(t):
        """[48] -> [1, 1, 48] fp32, ready to broadcast over [B, T, 48]."""
        return ttnn.from_torch(t.reshape(1, 1, -1).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Depthwise conv: [10240, 1, 4] -> [1, 4, 10240], channel axis last.
    conv1d_weight = sd["conv1d.weight"].squeeze(1).T.contiguous().reshape(1, CONV_KERNEL, CONV_DIM)

    return TtQwen36GatedDeltaNet(
        device,
        in_proj_qkv=linear(sd["in_proj_qkv.weight"]),
        in_proj_z=linear(sd["in_proj_z.weight"]),
        in_proj_b=linear(sd["in_proj_b.weight"]),
        in_proj_a=linear(sd["in_proj_a.weight"]),
        conv1d_weight=ttnn.from_torch(
            conv1d_weight.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        neg_A=per_head(-sd["A_log"].float().exp()),  # folded on host, like rms_norm's +1
        dt_bias=per_head(sd["dt_bias"]),
        norm_weight=ttnn.from_torch(
            sd["norm.weight"].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        out_proj=linear(sd["out_proj.weight"]),
        layer_idx=ref.layer_idx,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_gated_deltanet_pcc(mesh_device, shape, reset_seeds):
    batch, seq_len = shape

    ref = build_reference()
    x = torch.randn(batch, seq_len, D, dtype=torch.float32) * 0.5
    with torch.no_grad():
        reference = ref(x)

    tt_model = to_device(mesh_device, ref)
    tt_x = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    tt_out = ttnn.to_torch(tt_model(tt_x)).float().reshape(reference.shape)

    passing, pcc = comp_pcc(reference, tt_out, PCC_THRESHOLD)
    logger.info(f"gated_deltanet PCC {shape}: {pcc}")
    assert passing, f"gated_deltanet PCC below {PCC_THRESHOLD}: {pcc}"
