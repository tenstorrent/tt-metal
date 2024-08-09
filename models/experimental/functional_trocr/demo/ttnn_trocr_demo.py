# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image
import ttnn
import pytest

from datasets import load_dataset
import evaluate

from loguru import logger
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.ttnn_trocr_generation_utils import GenerationMixin
from models.experimental.functional_trocr.tt.ttnn_encoder_vit import custom_preprocessor


@pytest.mark.parametrize("model_name", ["microsoft/trocr-base-handwritten"])
def test_trocr_demo(device, model_name, input_path):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        config = model.decoder.config

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()

        iam_ocr_sample_input = Image.open(input_path)

        pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

        torch_generated_ids = model.generate(pixel_values)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, device=device, custom_preprocessor=custom_preprocessor
        )

        generationmixin = GenerationMixin(model=model, device=device, config=config, parameters=parameters)

        inputs_tensor = torch.permute(pixel_values, (0, 2, 3, 1))
        inputs_tensor = torch.nn.functional.pad(inputs_tensor, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = inputs_tensor.shape  # permuted input NHWC
        patch_size = 16
        sequence_size = 384
        inputs_tensor = inputs_tensor.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        cls_token = model.encoder.state_dict()["embeddings.cls_token"]
        torch_position_embeddings = model.encoder.state_dict()["embeddings.position_embeddings"]
        vit_config = model.encoder.config
        torch_attention_mask = None

        if torch_attention_mask is not None:
            head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(vit_config.num_hidden_layers)
            ]
        else:
            head_masks = [None for _ in range(vit_config.num_hidden_layers)]

        if batch_size > 1:
            cls_token = torch.nn.Parameter(cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            cls_token = torch.nn.Parameter(cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)

        cls_token = ttnn.from_torch(cls_token, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        inputs_tensor = ttnn.from_torch(inputs_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        ttnn_output = generationmixin.generate(
            inputs_tensor, cls_token=cls_token, position_embeddings=position_embeddings, head_masks=head_masks
        )
        torch_generated_text = processor.batch_decode(torch_generated_ids, skip_special_tokens=True)

        ttnn_generated_text = processor.batch_decode(ttnn_output, skip_special_tokens=True)

        logger.info(f"Torch output: {torch_generated_text}")
        logger.info(f"Ttnn output: {ttnn_generated_text}")


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("model_name", ["microsoft/trocr-base-handwritten"])
def test_trocr_demo_iam_dataset(device, batch_size, model_name):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        config = model.decoder.config

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()

        dataset = load_dataset("Teklia/IAM-line")
        bert_score = evaluate.load("bertscore")
        reference = []
        predicted = []
        input_images = []

        for index in range(batch_size):
            iam_ocr_sample_input = dataset["test"][index]["image"]
            reference.append(dataset["test"][index]["text"])
            iam_ocr_sample_input = iam_ocr_sample_input.convert("RGB")
            input_images.append(iam_ocr_sample_input)

        pixel_values = processor(images=input_images, return_tensors="pt").pixel_values

        torch_generated_ids = model.generate(pixel_values)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, device=device, custom_preprocessor=custom_preprocessor
        )

        generationmixin = GenerationMixin(model=model, device=device, config=config, parameters=parameters)

        inputs_tensor = torch.permute(pixel_values, (0, 2, 3, 1))
        inputs_tensor = torch.nn.functional.pad(inputs_tensor, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, img_c = inputs_tensor.shape  # permuted input NHWC
        patch_size = 16
        sequence_size = 384
        inputs_tensor = inputs_tensor.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

        cls_token = model.encoder.state_dict()["embeddings.cls_token"]
        torch_position_embeddings = model.encoder.state_dict()["embeddings.position_embeddings"]
        vit_config = model.encoder.config
        torch_attention_mask = None

        if torch_attention_mask is not None:
            head_masks = [
                ttnn.from_torch(
                    torch_attention_mask[index].reshape(1, 1, 1, sequence_size).expand(batch_size, -1, -1, -1),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for index in range(vit_config.num_hidden_layers)
            ]
        else:
            head_masks = [None for _ in range(vit_config.num_hidden_layers)]

        if batch_size > 1:
            cls_token = torch.nn.Parameter(cls_token.expand(batch_size, -1, -1))
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings.expand(batch_size, -1, -1))
        else:
            cls_token = torch.nn.Parameter(cls_token)
            torch_position_embeddings = torch.nn.Parameter(torch_position_embeddings)

        cls_token = ttnn.from_torch(cls_token, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        inputs_tensor = ttnn.from_torch(inputs_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        ttnn_output = generationmixin.generate(
            inputs_tensor, cls_token=cls_token, position_embeddings=position_embeddings, head_masks=head_masks
        )

        torch_generated_text = processor.batch_decode(torch_generated_ids, skip_special_tokens=True)

        ttnn_generated_text = processor.batch_decode(ttnn_output, skip_special_tokens=True)

        for i in ttnn_generated_text:
            predicted.append(i)

    results = bert_score.compute(predictions=predicted, references=reference, lang="en")
    accuracy = sum(results["f1"]) / len(results["f1"])

    logger.info(f"Reference output: {reference}")
    logger.info(f"Ttnn output: {predicted}")
    logger.info(f"ACCURACY: {accuracy}")
