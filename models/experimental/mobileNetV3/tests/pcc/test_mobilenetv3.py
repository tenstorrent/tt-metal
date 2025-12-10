# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc
from torchvision import models
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from torchvision.models import MobileNet_V3_Small_Weights
from models.experimental.mobileNetV3.tt.utils import conv_config as model_config
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


class MobilenetV3TestInfra:
    def __init__(self, device, batch_size, input_channels, height, width, load_weights=True, use_randn_input=False):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.use_randn_input = use_randn_input

        if use_randn_input:
            torch_input_tensor = torch.randn(batch_size, input_channels, height, width)
        else:
            self.img = Image.open("models/experimental/mobileNetV3/resources/dog.jpeg").convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            input_tensor = preprocess(self.img)
            torch_input_tensor = input_tensor.unsqueeze(0)

        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        self.ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

        if load_weights:
            mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v3_small(weights=None)

        torch_model = mobilenet

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
        )
        with torch.no_grad():
            torch_model.eval()
            self.torch_output_tensor = torch_model(torch_input_tensor)

        self.ttnn_model = ttnn_MobileNetV3(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            parameters=parameters,
            device=device,
            input_height=height,
            input_width=width,
        )

        self.run()
        self.validate()

    def run(self):
        logger.info("Running TTNN MobileNetV3 model...")
        self.output_tensor = self.ttnn_model(self.device, self.ttnn_input_tensor)
        return self.output_tensor

    def validate(self):
        logger.info("Validating TTNN output against PyTorch...")
        tt_output_tensor_torch = ttnn.to_torch(self.output_tensor)

        if not self.use_randn_input:
            # Postprocess
            probs = torch.nn.functional.softmax(tt_output_tensor_torch, dim=1)[0]
            top1_id = torch.argmax(probs).item()
            label = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"][top1_id]
            confidence = probs[top1_id].item()

            logger.info(f"prediction : {label} : {confidence:.2%}")

            # Draw label on image
            draw = ImageDraw.Draw(self.img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)  # Windows
            except:
                font = ImageFont.load_default()  # fallback

            text = f"{label}: {confidence:.2%}"
            draw.text((10, 10), text, fill="red", font=font)

            # Save / show
            self.img.save("models/experimental/mobileNetV3/resources/image_with_label.jpg")

        pcc_threshold = 0.99
        passed, msg = check_with_pcc(self.torch_output_tensor, tt_output_tensor_torch, pcc=pcc_threshold)
        assert passed, logger.error(f"MobileNetV3 PCC check failed: {msg}")

        logger.info(
            f"MobileNetV3 passed: "
            f"batch_size={self.batch_size}, "
            f"act_dtype={model_config['ACTIVATIONS_DTYPE']}, "
            f"weight_dtype={model_config['WEIGHTS_DTYPE']}, "
            f"math_fidelity={model_config['MATH_FIDELITY']}, "
            f"PCC={msg}"
        )

        return True, msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 12288}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 224, 224),
    ],
)
def test_MobilenetV3(device, batch_size, input_channels, height, width):
    MobilenetV3TestInfra(
        device=device,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
    )
