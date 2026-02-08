import pytest
import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    "test_fn",
    [
        "IMAGE_FEATURES",
        "TEXT_FEATURES",
        "FULL",
    ],
)
def test_clip_model_full(model_name, image_url, text_queries, max_seq_len, pcc, test_fn):
    torch_model = CLIPModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    torch_model = torch_model.to(torch.bfloat16)
    torch_model.eval()

    config = torch_model.config

    device = None

    try:
        device = ttnn.open_device(device_id=0)

        ttnn_model = TtCLIPModel(config, torch_model, device)

        image = load_image(image_url)
        inputs = processor(text_queries, image, padding=True, return_tensors="pt")

        torch_input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        batch_size, seq_len = torch_input_ids.shape
        torch_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        ttnn_input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_position_ids = ttnn.from_torch(
            torch_position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn_pixel_values = ttnn.permute(ttnn_pixel_values, [0, 2, 3, 1])

        if test_fn == "IMAGE_FEATURES":
            with torch.no_grad():
                torch_image_features = torch_model.get_image_features(pixel_values=pixel_values)
                torch_image_features = torch_image_features.to(torch.bfloat16)

            ttnn_image_features = ttnn_model.get_image_features(pixel_values=ttnn_pixel_values)
            ttnn_image_features_torch = ttnn.to_torch(ttnn_image_features)

            passed, pcc_value = comp_pcc(torch_image_features, ttnn_image_features_torch, pcc=pcc)
            assert_with_pcc(torch_image_features, ttnn_image_features_torch, pcc=pcc)

        elif test_fn == "TEXT_FEATURES":
            with torch.no_grad():
                torch_text_features = torch_model.get_text_features(
                    input_ids=torch_input_ids, position_ids=torch_position_ids
                )
                torch_text_features = torch_text_features.to(torch.bfloat16)

            ttnn_text_features = ttnn_model.get_text_features(
                input_ids=ttnn_input_ids,
                position_ids=ttnn_position_ids,
            )
            ttnn_text_features_torch = ttnn.to_torch(ttnn_text_features)

            passed, pcc_value = comp_pcc(torch_text_features, ttnn_text_features_torch, pcc=pcc)
            assert_with_pcc(torch_text_features, ttnn_text_features_torch, pcc=pcc)

        elif test_fn == "FULL":
            with torch.no_grad():
                torch_outputs = torch_model(input_ids=torch_input_ids, pixel_values=pixel_values)
                torch_logits_per_image = torch_outputs.logits_per_image.to(torch.bfloat16)
                torch_logits_per_text = torch_outputs.logits_per_text.to(torch.bfloat16)

            logits_per_image, logits_per_text = ttnn_model(
                input_ids=ttnn_input_ids,
                pixel_values=ttnn_pixel_values,
                position_ids=ttnn_position_ids,
            )

            ttnn_logits_per_image = ttnn.to_torch(logits_per_image)
            ttnn_logits_per_text = ttnn.to_torch(logits_per_text)

            print(f"LOGITS TEXT{torch_logits_per_text}")
            print(f"LOGITS IMAGE{torch_logits_per_image}")

            print(f"LOGITS TEXT{ttnn_logits_per_text}")
            print(f"LOGITS IMAGE{ttnn_logits_per_image}")

            passed_img, pcc_img = comp_pcc(torch_logits_per_image, ttnn_logits_per_image, pcc=pcc)
            assert_with_pcc(torch_logits_per_image, ttnn_logits_per_image, pcc=pcc)

            passed_txt, pcc_txt = comp_pcc(torch_logits_per_text, ttnn_logits_per_text, pcc=pcc)
            assert_with_pcc(torch_logits_per_text, ttnn_logits_per_text, pcc=pcc)

    finally:
        if device is not None:
            ttnn.close_device(device)
