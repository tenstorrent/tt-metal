# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from PIL import Image
from loguru import logger

from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


class Demo:
    """Panoptic-DeepLab demo supporting both PyTorch and TTNN pipelines."""

    def __init__(self) -> None:
        self.torch_model: Optional[Any] = None
        self.ttnn_model: Optional[Any] = None
        self.ttnn_device: Optional[Any] = None

        # Mesh mappers for TTNN
        self.inputs_mesh_mapper = None
        self.weights_mesh_mapper = None
        self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def initialize_torch_model(self) -> None:
        """Initialize PyTorch model and load weights."""
        logger.info("Initializing PyTorch Panoptic-DeepLab model…")
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
        self.torch_model = model
        logger.info("PyTorch model ready.")

    def initialize_ttnn_model(self) -> None:
        """Initialize TTNN model, preprocess parameters, and build runtime graph."""
        logger.info("Initializing TTNN Panoptic-DeepLab model…")

        # Initialize TT device
        self.ttnn_device = ttnn.open_device(device_id=0, l1_small_size=24576)

        # Setup mesh mappers
        self._setup_mesh_mappers()

        # Create reference torch model to extract parameters
        reference_model = self.torch_model

        # Preprocess model parameters

        logger.info("Preprocessing model parameters for TTNN…")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(self.weights_mesh_mapper),
            device=None,
        )

        # Create TTNN model
        self.ttnn_model = ttnn_MobileNetV3(
            inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, parameters=parameters
        )
        logger.info("TTNN model ready.")

    def _setup_mesh_mappers(self) -> None:
        """Setup mesh mappers for multi-device support."""
        if self.ttnn_device.get_num_devices() != 1:
            self.inputs_mesh_mapper = ttnn.ShardTensorToMesh(self.ttnn_device, dim=0)
            self.weights_mesh_mapper = None
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)
        else:
            self.inputs_mesh_mapper = None
            self.weights_mesh_mapper = None
            self.output_mesh_composer = None

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def run_torch_inference(self, input_tensor: torch.Tensor):
        """Run PyTorch inference."""
        if self.torch_model is None:
            raise RuntimeError("Torch model not initialized.")
        logger.info("Running PyTorch inference…")
        start = time.time()
        with torch.no_grad():
            output = self.torch_model(input_tensor)
        logger.info("PyTorch inference completed in {:.4f}s", time.time() - start)
        return output

    def run_ttnn_inference(self, input_tensor: ttnn.Tensor):
        """Run TTNN inference."""
        if self.ttnn_model is None or self.ttnn_device is None:
            raise RuntimeError("TTNN model/device not initialized.")
        logger.info("Running TTNN inference…")
        start = time.time()
        output = self.ttnn_model(self.ttnn_device, input_tensor)
        logger.info("TTNN inference completed in {:.4f}s", time.time() - start)
        return output

    def preprocess_image(self, image_path: str, device):
        imgage = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(imgage)
        torch_input = input_tensor.unsqueeze(0)
        ttnn_input = ttnn.from_torch(
            torch_input.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        ttnn_input = ttnn.to_device(ttnn_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        return torch_input, ttnn_input

    def postprocess_output(self, image_path, output_path, output_tt, output_torch):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        output_tt = ttnn.to_torch(output_tt)
        outputs = {"tt": output_tt, "torch": output_torch}

        for name, output in outputs.items():
            img = Image.open(image_path).convert("RGB")

            # Softmax + top1
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top1_id = torch.argmax(probs).item()
            label = weights.meta["categories"][top1_id]
            confidence = probs[top1_id].item()

            # Draw label
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)  # Windows
            except OSError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 24)  # Linux
                except OSError:
                    font = ImageFont.load_default()  # fallback

            text = f"{label}: {confidence:.2%}"

            # Background rectangle for visibility
            text_size = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([text_size[0] + 5, text_size[1] + 5, text_size[2] + 15, text_size[3] + 15], fill="black")
            draw.text((10, 10), text, fill="red", font=font)

            # Build unique filename
            os.makedirs(output_path, exist_ok=True)
            file_path = os.path.join(output_path, f"{name}.jpg")

            img.save(file_path)

    # ---------------------------------------------------------------------
    # Run demo
    # ---------------------------------------------------------------------
    def run_demo(self, image_path: str, output_dir: str) -> None:
        """Run the full demo pipeline end-to-end."""
        logger.info("Starting demo for image: {}", image_path)

        # Initialize models (Torch + TTNN)
        self.initialize_torch_model()
        self.initialize_ttnn_model()

        # Preprocess image
        torch_input, ttnn_input = self.preprocess_image(image_path, self.ttnn_device)

        base_name = Path(image_path).stem

        # Run inference
        torch_output = self.run_torch_inference(torch_input)
        ttnn_output = self.run_ttnn_inference(ttnn_input)

        # Postprocess to comparable outputs
        self.postprocess_output(image_path, output_dir, ttnn_output, torch_output)

        logger.info("Demo completed. Output dir: {}", output_dir)

    def cleanup(self) -> None:
        """Release device resources."""
        if self.ttnn_device is not None:
            try:
                ttnn.close_device(self.ttnn_device)
                logger.info("TTNN device closed.")
            finally:
                self.ttnn_device = None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT MobilenetV3 Demo")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        "-o",
        default="models/experimental/mobileNetV3/resources/outputs",
        help="Output directory for results",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Validate input file
    if not args.input or not os.path.exists(args.input):
        logger.error("Input image not found: {}", args.input)
        return 1

    # Prepare output directory
    out_dir = args.output or "models/experimental/mobileNetV3/resources/outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    demo: Optional[Demo] = None

    logger.info("=== MobilenetV3 Demo ===")
    try:
        demo = Demo()
        demo.run_demo(args.input, out_dir)
        return 0
    except Exception as e:
        logger.exception("Demo failed: {}", e)
        return 1
    finally:
        if demo is not None:
            try:
                demo.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
