# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import os
import torch
import ttnn
import urllib.request
from loguru import logger

from models.experimental.petr.reference.petr import PETR
from models.experimental.petr.tt.tt_petr import ttnn_PETR
from models.experimental.petr.tt.model_preprocessing import get_parameters
from models.experimental.petr.demo.visualization import (
    load_images_with_calibration,
    Det3DDataPreprocessor,
    create_combined_visualization,
)


DETECTION_THRESHOLD = 0.4
INCLUDE_TORCH_VISUALIZATION = False


def test_demo(device, reset_seeds, threshold=DETECTION_THRESHOLD, include_torch=INCLUDE_TORCH_VISUALIZATION):
    logger.info("Loading data...")
    input_data, camera_images = load_images_with_calibration("models/experimental/petr/resources/sample_input")

    data_preprocessor = Det3DDataPreprocessor()
    output_after_preprocess = data_preprocessor(input_data, False)
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/"
        "petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        logger.info("Downloading weights...")
        urllib.request.urlretrieve(weights_url, weights_path)

    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]

    logger.info("Running PyTorch inference...")
    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    torch_output = None
    if include_torch:
        with torch.no_grad():
            torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

    logger.info("Running TTNN inference...")
    ttnn_inputs = dict()
    imgs_tensor = output_after_preprocess["inputs"]["imgs"]
    if len(imgs_tensor.shape) == 4:
        imgs_tensor = imgs_tensor.unsqueeze(0)
    ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

    parameters, query_embedding_input = get_parameters(torch_model, device)
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    ttnn_output = ttnn_model.predict(ttnn_inputs, batch_img_metas)

    output_dir = "models/experimental/petr/resources/sample_output"
    os.makedirs(output_dir, exist_ok=True)

    create_combined_visualization(
        torch_output,
        ttnn_output,
        camera_images,
        batch_img_metas,
        os.path.join(output_dir, "visualization.jpg"),
        threshold=threshold,
        include_torch=include_torch,
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"Visualization saved to: {output_dir}/visualization.jpg")
    logger.info("Demo COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        test_demo(device, None)
    finally:
        ttnn.close_device(device)
