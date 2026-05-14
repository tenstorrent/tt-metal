# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.reference.vitpose_reference import (
    extract_reference_parameters,
    vitpose_embeddings,
    vitpose_patch_embeddings,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_embeddings import (
    VitPosePatchEmbeddings,
    preprocess_embedding_parameters,
    vitpose_embeddings as ttnn_vitpose_embeddings,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_vitpose_patch_embeddings(device, batch_size):
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()
    ref_params = extract_reference_parameters(model)

    pixel_values = torch.randn(batch_size, 3, 256, 192, dtype=torch.bfloat16)
    torch_output = vitpose_patch_embeddings(pixel_values, parameters=ref_params["backbone"]["embeddings"])

    tt_params = preprocess_embedding_parameters(state_dict, dtype=ttnn.bfloat16)
    patch_embed = VitPosePatchEmbeddings(tt_params, device, batch_size=batch_size)

    pixel_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(pixel_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = patch_embed(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output.float(), tt_output.float(), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_vitpose_embeddings(device, batch_size):
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()
    ref_params = extract_reference_parameters(model)

    pixel_values = torch.randn(batch_size, 3, 256, 192, dtype=torch.bfloat16)
    torch_output = vitpose_embeddings(pixel_values, parameters=ref_params["backbone"]["embeddings"])

    tt_params = preprocess_embedding_parameters(state_dict, dtype=ttnn.bfloat16)
    patch_embed = VitPosePatchEmbeddings(tt_params, device, batch_size=batch_size)

    pixel_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(pixel_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    patch_emb = patch_embed(tt_input)
    pos_patches = ttnn.to_device(tt_params["pos_patches"], device)
    pos_cls = ttnn.to_device(tt_params["pos_cls"], device)
    tt_output = ttnn_vitpose_embeddings(patch_emb, pos_patches=pos_patches, pos_cls=pos_cls)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output.float(), tt_output.float(), 0.99)
