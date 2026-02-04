import pytest
import torch
from transformers import CLIPModel, CLIPTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size, seq_len, pcc", [(1, 77, 0.95), (4, 77, 0.95)])
def test_clip_get_image_features(batch_size, seq_len, pcc):
    """
    Test the CLIP model's get_image_features method.
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config
    vision_config = config.vision_config

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        ttnn_model = TtCLIPModel(config, torch_model, device)

        # Create random pixel values
        torch_pixel_values = torch.randn(
            batch_size,
            vision_config.num_channels,
            vision_config.image_size,
            vision_config.image_size,
            dtype=torch.bfloat16,
        )
        ttnn_pixel_values = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        with torch.no_grad():
            torch_image_features = torch_model.get_image_features(pixel_values=torch_pixel_values)
            torch_image_features = torch_image_features.to(torch.bfloat16)

        ttnn_image_features = ttnn_model.get_image_features(pixel_values=ttnn_pixel_values)
        ttnn_image_features_torch = ttnn.to_torch(ttnn_image_features)

        passed, pcc_value = comp_pcc(torch_image_features, ttnn_image_features_torch, pcc=pcc)
        assert_with_pcc(torch_image_features, ttnn_image_features_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc", [(1, 77, 0.95), (4, 77, 0.95)])
def test_clip_get_text_features(batch_size, seq_len, pcc):
    """
    Test the CLIP model's get_text_features method.
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config
    text_config = config.text_config

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        ttnn_model = TtCLIPModel(config, torch_model, device)

        texts = ["a photo of a cat", "a photo of a dog", "a car", "a house"][:batch_size]
        inputs = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt")
        torch_input_ids = inputs["input_ids"]

        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

        with torch.no_grad():
            torch_text_features = torch_model.get_text_features(input_ids=torch_input_ids)
            torch_text_features = torch_text_features.to(torch.bfloat16)

        ttnn_text_features = ttnn_model.get_text_features(ttnn_input_ids)
        ttnn_text_features_torch = ttnn.to_torch(ttnn_text_features)

        passed, pcc_value = comp_pcc(torch_text_features, ttnn_text_features_torch, pcc=pcc)
        assert_with_pcc(torch_text_features, ttnn_text_features_torch, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)


@pytest.mark.parametrize("batch_size, seq_len, pcc", [(1, 77, 0.95), (4, 77, 0.95)])
def test_clip_model_forward(batch_size, seq_len, pcc):
    """
    Test the full CLIP model forward pass computing similarity scores.
    """
    torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config
    text_config = config.text_config
    vision_config = config.vision_config

    device = None

    try:
        device = ttnn.open_device(device_id=0)

        ttnn_model = TtCLIPModel(config, torch_model, device)

        # Create random inputs
        torch_pixel_values = torch.randn(
            batch_size,
            vision_config.num_channels,
            vision_config.image_size,
            vision_config.image_size,
            dtype=torch.bfloat16,
        )
        texts = ["a photo of a cat", "a photo of a dog", "a car", "a house"][:batch_size]
        inputs = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt")

        torch_input_ids = inputs["input_ids"]

        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

        ttnn_pixel_values = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        with torch.no_grad():
            torch_outputs = torch_model(input_ids=torch_input_ids, pixel_values=torch_pixel_values)
            torch_logits_per_image = torch_outputs.logits_per_image.to(torch.bfloat16)
            torch_logits_per_text = torch_outputs.logits_per_text.to(torch.bfloat16)

        logits_per_image, logits_per_text = ttnn_model(
            input_ids=ttnn_input_ids,
            pixel_values=ttnn_pixel_values,
        )

        ttnn_logits_per_image = ttnn.to_torch(logits_per_image)
        ttnn_logits_per_text = ttnn.to_torch(logits_per_text)

        passed_img, pcc_img = comp_pcc(torch_logits_per_image, ttnn_logits_per_image, pcc=pcc)
        assert_with_pcc(torch_logits_per_image, ttnn_logits_per_image, pcc=pcc)

        passed_txt, pcc_txt = comp_pcc(torch_logits_per_text, ttnn_logits_per_text, pcc=pcc)
        assert_with_pcc(torch_logits_per_text, ttnn_logits_per_text, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)
