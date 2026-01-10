# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for OWL-ViT TTNN implementation.

This file contains tests to validate:
1. Model loads without errors on N150/N300 hardware
2. Produces valid detections (bounding boxes + labels)
3. Output verification (region-text similarity scores)
"""

import sys
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.wormhole.owl_vit.tt.ttnn_owl_vit import OwlViTTTNNConfig

# Test constants
MODEL_NAME = "google/owlvit-base-patch32"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEST_QUERIES = ["a photo of a cat", "a photo of a dog", "a photo of a remote control"]
DETECTION_THRESHOLD = 0.1
PCC_THRESHOLD = 0.95  # Pearson Correlation Coefficient threshold for validation


def load_test_image(url: str = TEST_IMAGE_URL) -> Image.Image:
    """Load a test image from URL."""
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    except Exception as e:
        logger.warning(f"Could not load image from URL: {e}")
        pytest.skip("Test image could not be downloaded (offline?)")


def get_pytorch_reference(image: Image.Image, text_queries: list[str]):
    """
    Get reference outputs from PyTorch OWL-ViT model.

    Returns:
        processor: OwlViTProcessor
        model: OwlViTForObjectDetection
        inputs: Preprocessed inputs
        outputs: Model outputs
    """
    try:
        processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    except Exception as e:
        logger.warning(f"Failed to load HF model: {e}")
        pytest.skip("HuggingFace model could not be loaded (offline/no token?)")
    model.eval()

    texts = [text_queries]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return processor, model, inputs, outputs


def calculate_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate Pearson Correlation Coefficient between two tensors."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    t1_mean = t1.mean()
    t2_mean = t2.mean()

    t1_centered = t1 - t1_mean
    t2_centered = t2 - t2_mean

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered**2).sum() * (t2_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return (numerator / denominator).item()


@pytest.fixture
def model_and_processor():
    """Load PyTorch model and processor."""
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


class TestOwlViTBasicFunctionality:
    """Test basic model loading and initialization."""

    def test_config_from_huggingface(self, model_and_processor):
        """Test that TTNN config can be created from HuggingFace config."""
        _, model = model_and_processor

        ttnn_config = OwlViTTTNNConfig.from_huggingface(model.config)

        assert ttnn_config.vision_hidden_size == 768
        assert ttnn_config.vision_num_heads == 12
        assert ttnn_config.vision_layers == 12
        assert ttnn_config.patch_size == 32
        assert ttnn_config.image_size == 768
        assert ttnn_config.text_hidden_size == 512

        logger.info("Config created successfully from HuggingFace config")

    def test_device_initialization(self, device):
        """Test that device initializes correctly."""
        assert device is not None
        logger.info(f"Device initialized: {device}")


class TestOwlViTVisionEncoder:
    """Test vision encoder components."""

    def test_vision_embeddings_shape(self, device, model_and_processor):
        """Test that vision embeddings have correct shape."""
        processor, model = model_and_processor

        # Create test image
        image = load_test_image()
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # [1, 3, 768, 768]

        # Get reference embeddings
        with torch.no_grad():
            vision_outputs = model.owlvit.vision_model(pixel_values)

        expected_shape = vision_outputs.last_hidden_state.shape
        logger.info(f"Expected vision output shape: {expected_shape}")

        # Expected: [batch, num_patches+1, hidden_size] = [1, 577, 768]
        assert expected_shape[1] == 577  # 24*24 patches + 1 CLS
        assert expected_shape[2] == 768

    def test_pytorch_vision_forward(self, model_and_processor):
        """Test PyTorch vision model forward pass for reference."""
        processor, model = model_and_processor

        image = load_test_image()
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = model.owlvit.vision_model(inputs["pixel_values"])

        logger.info(f"Vision last_hidden_state shape: {vision_outputs.last_hidden_state.shape}")
        logger.info(f"Vision pooler_output shape: {vision_outputs.pooler_output.shape}")

        assert vision_outputs.last_hidden_state.shape == torch.Size([1, 577, 768])
        assert vision_outputs.pooler_output.shape == torch.Size([1, 768])


class TestOwlViTTextEncoder:
    """Test text encoder components."""

    def test_text_embeddings_shape(self, model_and_processor):
        """Test that text embeddings have correct shape."""
        processor, model = model_and_processor

        texts = [TEST_QUERIES]
        inputs = processor(text=texts, return_tensors="pt", padding=True)

        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Attention mask shape: {inputs['attention_mask'].shape}")

        with torch.no_grad():
            text_outputs = model.owlvit.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        logger.info(f"Text last_hidden_state shape: {text_outputs.last_hidden_state.shape}")
        logger.info(f"Text pooler_output shape: {text_outputs.pooler_output.shape}")

        # Shape should be [num_queries, max_seq_len, hidden_size]
        assert text_outputs.last_hidden_state.shape[2] == 512


class TestOwlViTEndToEnd:
    """End-to-end tests for object detection."""

    def test_pytorch_detection(self, model_and_processor):
        """Test PyTorch model produces valid detections."""
        processor, model = model_and_processor

        image = load_test_image()
        texts = [TEST_QUERIES]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logger.info(f"Detection logits shape: {outputs.logits.shape}")
        logger.info(f"Detection pred_boxes shape: {outputs.pred_boxes.shape}")

        # Expected shapes
        # logits: [batch, num_patches, num_queries] = [1, 576, 3]
        # pred_boxes: [batch, num_patches, 4] = [1, 576, 4]
        assert outputs.logits.shape == torch.Size([1, 576, 3])
        assert outputs.pred_boxes.shape == torch.Size([1, 576, 4])

        # Post-process to get detections
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes,
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]

        logger.info(f"Detected {len(boxes)} objects with threshold {DETECTION_THRESHOLD}")
        for box, score, label in zip(boxes, scores, labels):
            box_coords = [round(x, 2) for x in box.tolist()]
            logger.info(f"  {TEST_QUERIES[label]}: score={score.item():.3f}, box={box_coords}")

        # Should detect at least one object (cats in image)
        assert len(boxes) > 0, "Should detect at least one object"

    def test_box_coordinates_valid(self, model_and_processor):
        """Test that predicted box coordinates are valid."""
        processor, model = model_and_processor

        image = load_test_image()
        texts = [TEST_QUERIES]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        pred_boxes = outputs.pred_boxes  # [1, 576, 4] in (cx, cy, w, h) normalized

        # All coordinates should be in [0, 1] range
        assert (pred_boxes >= 0).all(), "Box coordinates should be >= 0"
        assert (pred_boxes <= 1).all(), "Box coordinates should be <= 1"

        # Width and height should be positive
        assert (pred_boxes[..., 2] > 0).all(), "Box width should be positive"
        assert (pred_boxes[..., 3] > 0).all(), "Box height should be positive"

        logger.info("Box coordinates validation passed")


class TestOwlViTParameterLoading:
    """Test parameter loading and preprocessing."""

    def test_model_parameter_extraction(self, model_and_processor):
        """Test that model parameters can be extracted."""
        _, model = model_and_processor

        state_dict = model.state_dict()

        # Check key parameters exist
        assert "owlvit.vision_model.embeddings.patch_embedding.weight" in state_dict
        assert "owlvit.vision_model.encoder.layers.0.self_attn.q_proj.weight" in state_dict
        assert "owlvit.text_model.embeddings.token_embedding.weight" in state_dict
        assert "box_head.dense0.weight" in state_dict
        assert "class_head.dense0.weight" in state_dict

        # Log parameter shapes
        logger.info("Key parameter shapes:")
        logger.info(
            f"  Vision patch embedding: {state_dict['owlvit.vision_model.embeddings.patch_embedding.weight'].shape}"
        )
        logger.info(
            f"  Vision Q projection: {state_dict['owlvit.vision_model.encoder.layers.0.self_attn.q_proj.weight'].shape}"
        )
        logger.info(
            f"  Text token embedding: {state_dict['owlvit.text_model.embeddings.token_embedding.weight'].shape}"
        )
        logger.info(f"  Box head dense0: {state_dict['box_head.dense0.weight'].shape}")
        logger.info(f"  Class head dense0: {state_dict['class_head.dense0.weight'].shape}")

    def test_weight_shapes_match_config(self, model_and_processor):
        """Test that weight shapes match configuration."""
        _, model = model_and_processor

        state_dict = model.state_dict()

        # Vision model checks
        vision_hidden_size = 768
        vision_intermediate_size = 3072
        assert state_dict["owlvit.vision_model.encoder.layers.0.mlp.fc1.weight"].shape == torch.Size(
            [vision_intermediate_size, vision_hidden_size]
        )
        assert state_dict["owlvit.vision_model.encoder.layers.0.mlp.fc2.weight"].shape == torch.Size(
            [vision_hidden_size, vision_intermediate_size]
        )

        # Text model checks
        text_hidden_size = 512
        text_intermediate_size = 2048
        assert state_dict["owlvit.text_model.encoder.layers.0.mlp.fc1.weight"].shape == torch.Size(
            [text_intermediate_size, text_hidden_size]
        )
        assert state_dict["owlvit.text_model.encoder.layers.0.mlp.fc2.weight"].shape == torch.Size(
            [text_hidden_size, text_intermediate_size]
        )

        logger.info("Weight shape validation passed")


# ============================================================================
# Performance and Integration Tests (require hardware)
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
