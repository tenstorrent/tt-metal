# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc

from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import SunrgbdDatasetConfig
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.ttnn.position_embedding import TtnnPositionEmbeddingCoordsSine


@pytest.mark.parametrize("num_points", [128, 1024])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_position_embedding(device, num_points, reset_seeds):
    ref_model, _ = build_3detr(Detr3dArgs(), SunrgbdDatasetConfig())
    load_torch_model_state(ref_model)
    torch_pos_embedding = ref_model.pos_embedding

    xyz = torch.rand(1, num_points, 3) * torch.tensor([6.0, 6.0, 2.5]) - torch.tensor([3.0, 0.0, 0.0])
    dims = [torch.min(xyz, 1)[0], torch.max(xyz, 1)[0]]

    with torch.no_grad():
        torch_out = torch_pos_embedding(xyz, input_range=dims)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    ttnn_pos_embedding = TtnnPositionEmbeddingCoordsSine(
        pos_type="fourier", normalize=True, parameters=parameters.pos_embedding, device=device
    )
    ttnn_out = ttnn.to_torch(ttnn_pos_embedding(xyz, input_range=dims)).float()
    ttnn_out = ttnn_out.reshape(1, num_points, -1).permute(0, 2, 1)

    passing, pcc_message = comp_pcc(torch_out, ttnn_out, 0.95)
    logger.info(f"Position embedding PCC: {pcc_message}")
    assert passing, f"Position embedding PCC check failed: {pcc_message}"
