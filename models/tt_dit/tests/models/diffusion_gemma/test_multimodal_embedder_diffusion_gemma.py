# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-module parity: TT ``DiffusionGemmaMultimodalEmbedder`` vs a pure-torch
reference matching the HF spec (RMSNorm without affine, then a no-bias Linear
that projects vision_hidden → text_hidden).

This module is the only bridge between the vision tower's hidden size and the
text encoder's hidden size, and isn't exercised by any other test directly.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_multimodal_embedder_diffusion_gemma.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma.multimodal_embedder import DiffusionGemmaMultimodalEmbedder
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

PCC_THRESHOLD = 0.9995
ALLCLOSE_ATOL = 2e-2
ALLCLOSE_RTOL = 2e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_galaxy_tp4"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_multimodal_embedder(
    mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology
) -> None:
    """TT DiffusionGemmaMultimodalEmbedder vs a pure-torch RMSNorm + Linear reference."""
    torch.manual_seed(0)
    torch_dtype = torch.float32

    # Representative real-config sizes: vision_hidden_size=1152, text_hidden_size=2816.
    multimodal_hidden_size = 1152
    text_hidden_size = 2816
    rms_norm_eps = 1e-6
    num_soft_tokens = 256
    B = 1

    # Torch reference. RMSNorm has no learned scale (affine=False), so no scale tensor.
    proj_weight = torch.randn(text_hidden_size, multimodal_hidden_size, dtype=torch_dtype) * 0.02
    inputs_embeds = torch.randn(B, num_soft_tokens, multimodal_hidden_size, dtype=torch_dtype)
    with torch.no_grad():
        rms = inputs_embeds.pow(2).mean(dim=-1, keepdim=True).add(rms_norm_eps).rsqrt()
        normed = inputs_embeds * rms
        torch_out = normed @ proj_weight.T

    # State-dict keys match the HF layout the TT module loads from (only the projection
    # weight; the norm has no affine parameter to load).
    hf_state = {"embedding_projection.weight": proj_weight}

    tt_model = DiffusionGemmaMultimodalEmbedder(
        multimodal_hidden_size=multimodal_hidden_size,
        text_hidden_size=text_hidden_size,
        rms_norm_eps=rms_norm_eps,
        mesh_device=mesh_device,
    )
    tt_model.load_state_dict(hf_state)

    tt_in = bf16_tensor(inputs_embeds.unsqueeze(0), device=mesh_device)
    tt_out = tt_model(tt_in)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(torch_dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(torch_out, tt_out_torch.to(torch_dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
