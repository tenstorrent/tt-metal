# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path

from models.experimental.efficientnetb0.reference import efficientnetb0
from efficientnet_pytorch import EfficientNet
from models.utility_functions import run_for_wormhole_b0
from models.experimental.efficientnetb0.demo.demo_utils import preprocess, download_images, load_imagenet_labels
from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

import cv2
import torch
import ttnn
from loguru import logger


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 7 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "source",
    [
        "models/experimental/efficientnetb0/demo/input_image.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
def test_demo(model_type, source, device, reset_seeds):
    download_images(source)

    categories = load_imagenet_labels("models/experimental/efficientnetb0/demo/imagenet_class_labels.txt")
    transform = preprocess()

    image = cv2.imread(str(source))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    batch_size = input_tensor.shape[0]

    if model_type == "torch_model":
        model = EfficientNet.from_pretrained("efficientnet-b0").eval()
        state_dict = model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        torch_model = efficientnetb0.Efficientnetb0()

        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()

        output = torch_model(input_tensor)
        logger.info("Inferencing [Torch] Model")
    else:
        performant_runner = EfficientNetb0PerformantRunner(
            device,
            batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            model_location_generator=None,
            resolution=(224, 224),
        )
        performant_runner._capture_efficientnetb0_trace_2cqs()
        output = performant_runner.run(torch_input_tensor=input_tensor)
        output = ttnn.to_torch(output)
        logger.info("Inferencing [TTNN] Model")
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Check the top 5 categories that are predicted.
    top5_prob, top5_catid = torch.topk(probabilities, 3)

    for i in range(top5_prob.size(0)):
        cv2.putText(
            image,
            f"{top5_prob[i].item()*100:.3f}%",
            (15, (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{categories[top5_catid[i]]}",
            (160, (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        logger.info(categories[top5_catid[i]], top5_prob[i].item())

    out_dir = ROOT / model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "output_image.jpg")
    cv2.imwrite(out_path, image)
    logger.info(f"Output image saved to {out_path}")
