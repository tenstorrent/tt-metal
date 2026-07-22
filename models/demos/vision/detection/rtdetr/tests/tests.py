import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from ttnn.model_preprocessing import ParameterList, make_parameter_dict, preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.tt.backbone import (
    TtRTDetrConvEncoder,
    TtRTDetrResNetBackBone,
    TtRTDetrResNetBottleNeckLayer,
    TtRTDetrResNetConvLayer,
    TtRTDetrResNetEmbeddings,
    TtRTDetrResNetEncoder,
    TtRTDetrResNetShortcut,
    TtRTDetrResNetStage,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

MODEL_NAME = "PekingU/rtdetr_r50vd"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def load_coco_image():
    try:
        response = requests.get(TEST_IMAGE_URL, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except requests.RequestException as error:
        pytest.skip(f"COCO validation image could not be downloaded: {error}")


def preprocess_resnet_conv_layer(torch_module, *_):
    """Preserve the convolution parameter and Frozen-BN buffers as Torch tensors."""
    return {
        "convolution": {"weight": torch_module.convolution.weight},
        "normalization": {
            "weight": torch_module.normalization.weight,
            "bias": torch_module.normalization.bias,
            "running_mean": torch_module.normalization.running_mean,
            "running_var": torch_module.normalization.running_var,
        },
    }


def preprocess_resnet_embeddings(torch_module, *_):
    return {
        "embedder": ParameterList(
            [make_parameter_dict(preprocess_resnet_conv_layer(layer)) for layer in torch_module.embedder]
        ),
    }


def preprocess_resnet_bottleneck(torch_module, *_):
    parameters = {
        "layer": ParameterList(
            [make_parameter_dict(preprocess_resnet_conv_layer(layer)) for layer in torch_module.layer]
        ),
    }
    projection = torch_module.shortcut if hasattr(torch_module.shortcut, "convolution") else None
    if projection is None:
        projection = next(
            (module for module in torch_module.shortcut.children() if hasattr(module, "convolution")),
            None,
        )
    if projection is not None:
        parameters["shortcut"] = make_parameter_dict(preprocess_resnet_conv_layer(projection))
    return parameters


def preprocess_resnet_stage(torch_module, *_):
    return {
        "layers": ParameterList(
            [make_parameter_dict(preprocess_resnet_bottleneck(layer)) for layer in torch_module.layers]
        ),
    }


def preprocess_resnet_encoder(torch_module, *_):
    return {
        "stages": ParameterList([make_parameter_dict(preprocess_resnet_stage(stage)) for stage in torch_module.stages]),
    }


def preprocess_resnet_backbone(torch_module, *_):
    return {
        "embedder": make_parameter_dict(preprocess_resnet_embeddings(torch_module.embedder)),
        "encoder": make_parameter_dict(preprocess_resnet_encoder(torch_module.encoder)),
    }


def preprocess_conv_encoder(torch_module, *_):
    return {
        "model": make_parameter_dict(preprocess_resnet_backbone(torch_module.model)),
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_conv_layer(device):
    batch_size = 1
    in_channels = 3
    out_channels = 32

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_module = torch_rtdetr.model.backbone.model.embedder.embedder[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_conv_layer,
    )

    image = load_coco_image()
    torch_input = image_processor(images=image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = torch_input.shape

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetConvLayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        activation="relu",
    )

    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
    tt_output = tt_output.permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_embeddings(device):
    batch_size = 1
    out_channels = 64

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_module = torch_rtdetr.model.backbone.model.embedder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_embeddings,
    )

    image = load_coco_image()
    torch_input = image_processor(images=image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = torch_input.shape

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetEmbeddings(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )

    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
    tt_output = tt_output.permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_shortcut(device):
    batch_size = 1
    in_channels = 256
    out_channels = 512

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_backbone = torch_rtdetr.model.backbone.model
    torch_module = torch_backbone.encoder.stages[1].layers[0].shortcut

    # The downsampling shortcut is AvgPool2d followed by the projection module.
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module[1],
        custom_preprocessor=preprocess_resnet_conv_layer,
    )

    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        torch_input = torch_backbone.embedder(pixel_values)
        torch_input = torch_backbone.encoder.stages[0](torch_input)
        torch_output = torch_module(torch_input)

    _, _, input_height, input_width = torch_input.shape
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetShortcut(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        in_channels=in_channels,
        out_channels=out_channels,
        downsample=True,
    )

    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
    tt_output = tt_output.permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_bottleneck(device):
    batch_size = 1
    in_channels = 64
    out_channels = 256

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_backbone = torch_rtdetr.model.backbone.model
    torch_module = torch_backbone.encoder.stages[0].layers[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_bottleneck,
    )

    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        torch_input = torch_backbone.embedder(pixel_values)
        torch_output = torch_module(torch_input)

    _, _, input_height, input_width = torch_input.shape
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetBottleNeckLayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=(1, 1),
    )

    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
    tt_output = tt_output.permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_stage(device):
    batch_size = 1
    in_channels = 64
    out_channels = 256

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_backbone = torch_rtdetr.model.backbone.model
    torch_module = torch_backbone.encoder.stages[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_stage,
    )

    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        torch_input = torch_backbone.embedder(pixel_values)
        torch_output = torch_module(torch_input)

    _, _, input_height, input_width = torch_input.shape
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetStage(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=3,
        stride=(1, 1),
    )

    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
    tt_output = tt_output.permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_encoder(device):
    batch_size = 1

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_backbone = torch_rtdetr.model.backbone.model
    torch_module = torch_backbone.encoder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_encoder,
    )

    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        torch_input = torch_backbone.embedder(pixel_values)
        torch_outputs = torch_module(torch_input, output_hidden_states=True).hidden_states[1:]

    _, _, input_height, input_width = torch_input.shape
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetEncoder(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )

    tt_outputs = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    hidden_sizes = torch_rtdetr.config.backbone_config.hidden_sizes
    for stage_index, (torch_output, (tt_output, output_height, output_width), out_channels) in enumerate(
        zip(torch_outputs, tt_outputs, hidden_sizes), start=1
    ):
        tt_output = ttnn.to_torch(tt_output)
        tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
        tt_output = tt_output.permute(0, 3, 1, 2)

        _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.97)
        logger.info(f"Stage {stage_index}: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_resnet_backbone(device):
    batch_size = 1

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_module = torch_rtdetr.model.backbone.model

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_resnet_backbone,
    )

    image = load_coco_image()
    torch_input = image_processor(images=image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = torch_input.shape

    with torch.no_grad():
        torch_outputs = torch_module(torch_input, output_hidden_states=True).hidden_states[1:]

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrResNetBackBone(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )

    tt_outputs = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    hidden_sizes = torch_rtdetr.config.backbone_config.hidden_sizes
    for stage_index, (torch_output, (tt_output, output_height, output_width), out_channels) in enumerate(
        zip(torch_outputs, tt_outputs, hidden_sizes), start=1
    ):
        tt_output = ttnn.to_torch(tt_output)
        tt_output = tt_output.reshape(batch_size, output_height, output_width, out_channels)
        tt_output = tt_output.permute(0, 3, 1, 2)

        _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.97)
        logger.info(f"Stage {stage_index}: {pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_conv_encoder(device):
    batch_size = 1

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    torch_module = torch_rtdetr.model.backbone

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_conv_encoder,
    )

    image = load_coco_image()
    torch_input = image_processor(images=image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = torch_input.shape
    pixel_mask = torch.ones((batch_size, input_height, input_width), dtype=torch.bool)

    with torch.no_grad():
        torch_outputs = [feature_map for feature_map, _ in torch_module(torch_input, pixel_mask)]

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrConvEncoder(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_outputs = tt_module(tt_input)

    hidden_sizes = torch_rtdetr.config.backbone_config.hidden_sizes
    out_indices = torch_rtdetr.config.backbone_config.out_indices
    out_channels = [hidden_sizes[index - 1] for index in out_indices]

    for output_index, (torch_output, (tt_output, output_height, output_width), channels) in enumerate(
        zip(torch_outputs, tt_outputs, out_channels)
    ):
        tt_output = ttnn.to_torch(tt_output)
        tt_output = tt_output.reshape(batch_size, output_height, output_width, channels)
        tt_output = tt_output.permute(0, 3, 1, 2)

        _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.97)
        logger.info(f"Output {output_index}: {pcc_message}")
