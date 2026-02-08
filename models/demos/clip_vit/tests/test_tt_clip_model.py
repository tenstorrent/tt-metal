import pytest
import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.clip_vit.tests.test_tt_clip_text import (
    test_clip_text_attention,
    test_clip_text_embeddings,
    test_clip_text_encoder,
    test_clip_text_encoder_layer,
    test_clip_text_mlp,
)
from models.demos.clip_vit.tests.test_tt_clip_vision import (
    test_clip_vision_attention,
    test_clip_vision_embeddings,
    test_clip_vision_encoder,
    test_clip_vision_encoder_layer,
    test_clip_vision_mlp,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_clip_get_image_features(torch_model, submodule, pixel_values, config, pcc, device):
    vision_config = config.vision_config
    vision_model = torch_model.vision_model

    torch_output, ttnn_output = None, None
    if submodule == "EMBEDDINGS":
        torch_output, ttnn_output = test_clip_vision_embeddings(vision_model, vision_config, pixel_values, device)
    if submodule == "MLP":
        torch_output, ttnn_output = test_clip_vision_mlp(vision_model, vision_config, pixel_values, device)
    if submodule == "ATTENTION":
        torch_output, ttnn_output = test_clip_vision_attention(vision_model, vision_config, pixel_values, device)
    if submodule == "ENCODER_LAYER":
        torch_output, ttnn_output = test_clip_vision_encoder_layer(vision_model, vision_config, pixel_values, device)
    if submodule == "FULL":
        torch_output, ttnn_output = test_clip_vision_encoder(vision_model, vision_config, pixel_values, device)

    """
    ttnn_pixel_values = ttnn.from_torch(
        pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with torch.no_grad():
        torch_image_features = torch_model.get_image_features(pixel_values=pixel_values)
        torch_image_features = torch_image_features.to(torch.bfloat16)

    ttnn_image_features = ttnn_model.get_image_features(pixel_values=ttnn_pixel_values)
    ttnn_image_features_torch = ttnn.to_torch(ttnn_image_features)
    """

    passed, pcc_value = comp_pcc(torch_output, ttnn_output, pcc=pcc)
    assert_with_pcc(torch_output, ttnn_output, pcc=pcc)


def test_clip_get_text_features(torch_model, submodule, input_ids, config, pcc, device):
    text_config = config.text_config
    text_model = torch_model.text_model

    torch_output, ttnn_output = None, None
    if submodule == "EMBEDDINGS":
        torch_output, ttnn_output = test_clip_text_embeddings(text_model, text_config, input_ids, device)
    if submodule == "MLP":
        torch_output, ttnn_output = test_clip_text_mlp(text_model, text_config, input_ids, device)
    if submodule == "ATTENTION":
        torch_output, ttnn_output = test_clip_text_attention(text_model, text_config, input_ids, device)
    if submodule == "ENCODER_LAYER":
        torch_output, ttnn_output = test_clip_text_encoder_layer(text_model, text_config, input_ids, device)
    if submodule == "FULL":
        torch_output, ttnn_output = test_clip_text_encoder(text_model, text_config, input_ids, device)

    passed, pcc_value = comp_pcc(torch_output, ttnn_output, pcc=pcc)
    assert_with_pcc(torch_output, ttnn_output, pcc=pcc)


@pytest.mark.parametrize(
    "model_name, image_url, text_queries, max_seq_len, pcc",
    [
        (
            "openai/clip-vit-base-patch32",
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            ["a photo of a cat", "a photo of a dog"],
            77,
            0.95,
        )
    ],
)
@pytest.mark.parametrize(
    "encoder, submodule",
    [
        ("VISION", "EMBEDDINGS"),
        ("VISION", "MLP"),
        ("VISION", "ATTENTION"),
        ("VISION", "ENCODER_LAYER"),
        ("VISION", "FULL"),
        ("TEXT", "EMBEDDINGS"),
        ("TEXT", "MLP"),
        ("TEXT", "ATTENTION"),
        ("TEXT", "ENCODER_LAYER"),
        ("TEXT", "FULL"),
    ],
)
def test_clip_model_forward(model_name, image_url, text_queries, max_seq_len, pcc, encoder, submodule):
    torch_model = CLIPModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config
    text_config = config.text_config
    vision_config = config.vision_config

    device = None

    try:
        device = ttnn.open_device(device_id=0)

        # ttnn_model = TtCLIPModel(config, torch_model, device)
        image = load_image(image_url)
        inputs = processor(text_queries, image, padding=True, return_tensors="pt")

        torch_input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)

        if encoder == "VISION":
            test_clip_get_image_features(torch_model, submodule, pixel_values, config, pcc, device)
        if encoder == "TEXT":
            test_clip_get_text_features(torch_model, submodule, torch_input_ids, config, pcc, device)

        """
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

        """

    finally:
        if device is not None:
            ttnn.close_device(device)
