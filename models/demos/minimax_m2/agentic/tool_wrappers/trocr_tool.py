# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TrOCRTool: Optical Character Recognition using TrOCR on TTNN.

Uses microsoft/trocr-base-handwritten for reading text from images.
Particularly good at handwritten text recognition.
"""

import torch
from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import ttnn
from models.experimental.trocr.tt.trocr import trocr_causal_llm


class TrOCRTool:
    """
    TTNN-accelerated TrOCR for optical character recognition.

    Reads text from images, especially handwritten text.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self._init_model(mesh_device)

    def _init_model(self, mesh_device):
        """Load TrOCR components."""
        logger.info("Loading TrOCR components...")

        # TrOCR uses single device - get chip0 submesh
        if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
            self.device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        else:
            self.device = mesh_device

        # Load HuggingFace components
        self.model_name = "microsoft/trocr-base-handwritten"
        logger.info(f"Loading processor from {self.model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)

        logger.info("Loading encoder model...")
        self.hf_model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        # Create TTNN-accelerated decoder
        logger.info("Creating TTNN decoder...")
        self.tt_model = trocr_causal_llm(self.device)

        logger.info("TrOCR ready.")

    def read_text(self, image_path: str) -> str:
        """
        Read text from an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Recognized text from the image.
        """
        logger.info(f"Reading text from: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # Generate text using TTNN-accelerated decoder
        with torch.no_grad():
            generated_ids = self.tt_model.generate(pixel_values)

        # Decode to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Recognized text: {text}")
        return text

    def close(self):
        """Release resources."""
        self.tt_model = None
        self.hf_model = None
        logger.info("TrOCRTool closed.")
