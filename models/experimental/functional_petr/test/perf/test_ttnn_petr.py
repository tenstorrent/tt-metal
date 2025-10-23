# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import tracy
from models.experimental.functional_petr.reference.petr import PETR
from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR
import os
from models.experimental.functional_petr.tt.common import get_parameters


def prepare_inputs():
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )

    inputs["imgs"] = inputs["imgs"][:, 0:1, :, :, :]
    for meta in modified_batch_img_metas:
        meta["cam2img"] = [meta["cam2img"][0]]
        meta["lidar2cam"] = [meta["lidar2cam"][0]]
        meta["img_shape"] = [meta["img_shape"][0]] if isinstance(meta["img_shape"], list) else meta["img_shape"]
    return inputs, modified_batch_img_metas


def prepare_torch_model():
    torch_model = PETR(use_grid_mask=True)

    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        import urllib.request

        print(f"Downloading PETR weights from {weights_url} ...")
        urllib.request.urlretrieve(weights_url, weights_path)
        print(f"Weights downloaded to {weights_path}")

    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()
    return torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr(device, reset_seeds):
    perf = False
    inputs, modified_batch_img_metas = prepare_inputs()

    torch_model = prepare_torch_model()

    ttnn_inputs = dict()
    ttnn_inputs["imgs"] = ttnn.from_torch(inputs["imgs"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_batch_img_metas = modified_batch_img_metas.copy()

    torch_output = torch_model.predict(inputs, modified_batch_img_metas, skip_post_processing=True)

    parameters, query_embedding_input = get_parameters(torch_model, device)

    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    tracy.signpost("start")
    ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas, skip_post_processing=True)
    tracy.signpost("end")
