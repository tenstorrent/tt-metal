# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from PIL import Image

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.tt.ttnn_encoder_vit import custom_preprocessor, vit
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_vit_encoder(device, reset_seeds):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    config = model.encoder.config
    sequence_size = 384
    batch_size = 1
    model = model.encoder
    model_state_dict = model.state_dict()

    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    torch_pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

    torch_attention_mask = None
    torch_output, *_ = model(torch_pixel_values).last_hidden_state

    model_state_dict = model.state_dict()

    torch_cls_token = model_state_dict["embeddings.cls_token"]
    torch_position_embeddings = model_state_dict["embeddings.position_embeddings"]
    if batch_size > 1:
        torch_cls_token = torch.nn.Parameter(torch_cls_token.expand(batch_size, -1, -1))
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
    else:
        torch_cls_token = torch.nn.Parameter(torch_cls_token)
        torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)
    cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    torch_pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    torch_pixel_values = torch.nn.functional.pad(torch_pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    batch_size, img_h, img_w, img_c = torch_pixel_values.shape  # permuted input NHWC
    patch_size = 16
    torch_pixel_values = torch_pixel_values.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    N, H, W, C = torch_pixel_values.shape

    pixel_values = ttnn.from_torch(
        torch_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    if torch_attention_mask is not None:
        head_masks = [
            ttnn.from_torch(
                torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for index in range(config.num_hidden_layers)
        ]
    else:
        head_masks = [None for _ in range(config.num_hidden_layers)]

    ttnn_output = vit(
        config,
        pixel_values,
        head_masks,
        cls_token,
        position_embeddings,
        parameters=parameters,
        device=device,
    )
    ttnn_output = ttnn.to_torch(ttnn_output[0])[0, :, :]
    assert_with_pcc(torch_output, ttnn_output, 0.91)
