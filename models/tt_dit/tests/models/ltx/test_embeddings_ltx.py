# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.embeddings import LTXAdaLayerNormSingle
from models.tt_dit.utils.check import assert_quality

# Add LTX-2 reference to path
sys.path.insert(0, "LTX-2/packages/ltx-core/src")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("embedding_dim", [4096], ids=["dim4096"])
def test_ltx_adaln_single(mesh_device: ttnn.MeshDevice, embedding_dim: int):
    """
    Test LTXAdaLayerNormSingle: compare TT output vs LTX-2 PyTorch AdaLayerNormSingle.
    """
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle as TorchAdaLayerNormSingle

    B = 2

    # Create PyTorch reference model
    torch_model = TorchAdaLayerNormSingle(embedding_dim=embedding_dim, embedding_coefficient=6)
    torch_model.eval()
    torch_state = torch_model.state_dict()
    logger.info(f"PyTorch AdaLayerNormSingle state keys: {list(torch_state.keys())}")

    # Create TT model and load weights
    tt_model = LTXAdaLayerNormSingle(
        embedding_dim=embedding_dim,
        embedding_coefficient=6,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_model.load_torch_state_dict(torch_state)

    # Create timestep input
    torch.manual_seed(42)
    timestep = torch.rand(B) * 1000  # random timesteps in [0, 1000)

    # PyTorch reference forward
    with torch.no_grad():
        torch_modulation, torch_embedded = torch_model(timestep)
    logger.info(f"PyTorch modulation shape: {torch_modulation.shape}, embedded: {torch_embedded.shape}")

    # TT forward
    # Timesteps module expects (1, 1, B, 1) so B broadcasts with the 128-dim factor.
    # Use float32 for timestep input — sinusoidal embeddings are precision-sensitive.
    tt_timestep = ttnn.from_torch(
        timestep.reshape(1, 1, B, 1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )
    tt_modulation, tt_embedded = tt_model(tt_timestep)

    # Convert back and compare — output is (1, 1, B, dim) in ttnn
    tt_modulation_torch = ttnn.to_torch(tt_modulation).squeeze(0).squeeze(0)
    tt_embedded_torch = ttnn.to_torch(tt_embedded).squeeze(0).squeeze(0)

    logger.info(f"TT modulation shape: {tt_modulation_torch.shape}, embedded: {tt_embedded_torch.shape}")

    assert_quality(torch_modulation, tt_modulation_torch, pcc=0.999)
    assert_quality(torch_embedded, tt_embedded_torch, pcc=0.999)
    logger.info("PASSED: LTXAdaLayerNormSingle matches PyTorch reference")
