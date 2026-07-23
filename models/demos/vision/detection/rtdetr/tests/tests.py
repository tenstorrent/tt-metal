import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import (
    preprocess_conv_encoder,
    preprocess_resnet_backbone,
    preprocess_resnet_bottleneck,
    preprocess_resnet_conv_layer,
    preprocess_resnet_embeddings,
    preprocess_resnet_encoder,
    preprocess_resnet_stage,
    preprocess_rtdetr_aifi_layer,
    preprocess_rtdetr_conv_norm_layer,
    preprocess_rtdetr_csp_rep_layer,
    preprocess_rtdetr_encoder_layer,
    preprocess_rtdetr_hybrid_encoder,
    preprocess_rtdetr_mlp,
    preprocess_rtdetr_rep_vgg_block,
    preprocess_rtdetr_self_attention,
)
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
from models.demos.vision.detection.rtdetr.tt.encoder import (
    TtRTDetrAIFILayer,
    TtRTDetrConvNormLayer,
    TtRTDetrCSPRepLayer,
    TtRTDetrEncoderLayer,
    TtRTDetrHybridEncoder,
    TtRTDetrMLP,
    TtRTDetrRepVggBlock,
    TtRTDetrSelfAttention,
    build_2d_sinusoidal_position_embedding,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

MODEL_NAME = "PekingU/rtdetr_r50vd"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
ENCODER_HEIGHT = 20
ENCODER_WIDTH = 20


def load_coco_image():
    try:
        response = requests.get(TEST_IMAGE_URL, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except requests.RequestException as error:
        pytest.skip(f"COCO validation image could not be downloaded: {error}")


def make_encoder_hidden_states(config):
    torch.manual_seed(0)
    return torch.randn(1, ENCODER_HEIGHT * ENCODER_WIDTH, config.encoder_hidden_dim)


def make_encoder_position_embeddings(torch_aifi, hidden_states):
    return torch_aifi.position_embedding(
        width=ENCODER_WIDTH,
        height=ENCODER_HEIGHT,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


def make_hybrid_encoder_inputs(torch_rtdetr):
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    batch_size, _, height, width = pixel_values.shape
    pixel_mask = torch.ones((batch_size, height, width), dtype=torch.bool)

    with torch.no_grad():
        backbone_features = [feature_map for feature_map, _ in torch_rtdetr.model.backbone(pixel_values, pixel_mask)]
        return [
            projection(feature_map)
            for projection, feature_map in zip(torch_rtdetr.model.encoder_input_proj, backbone_features)
        ]


def to_tt_feature_map(torch_input, device, dtype=ttnn.bfloat16):
    batch_size, channels, height, width = torch_input.shape
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, batch_size * height * width, channels),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return tt_input, batch_size, height, width


def to_torch_feature_map(tt_output, batch_size, height, width, channels):
    return ttnn.to_torch(tt_output).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)


def test_rtdetr_2d_sinusoidal_position_embedding():
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_aifi = torch_rtdetr.model.encoder.aifi[0]

    torch_output = torch_aifi.position_embedding(
        width=ENCODER_WIDTH,
        height=ENCODER_HEIGHT,
        device="cpu",
        dtype=torch.float32,
    )
    output = build_2d_sinusoidal_position_embedding(
        height=ENCODER_HEIGHT,
        width=ENCODER_WIDTH,
        embed_dim=torch_rtdetr.config.encoder_hidden_dim,
        temperature=torch_rtdetr.config.positional_encoding_temperature,
    )

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.9999)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_mlp(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.encoder.aifi[0].layers[0].mlp
    torch_input = make_encoder_hidden_states(torch_rtdetr.config)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_mlp,
    )

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrMLP(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output = ttnn.to_torch(tt_module(tt_input))

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_self_attention(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_aifi = torch_rtdetr.model.encoder.aifi[0]
    torch_module = torch_aifi.layers[0].self_attn
    torch_input = make_encoder_hidden_states(torch_rtdetr.config)
    position_embeddings = make_encoder_position_embeddings(torch_aifi, torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_self_attention,
    )

    with torch.no_grad():
        torch_output, _ = torch_module(
            hidden_states=torch_input,
            attention_mask=None,
            position_embeddings=position_embeddings,
        )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_position_embeddings = ttnn.from_torch(
        position_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrSelfAttention(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output = ttnn.to_torch(tt_module(tt_input, position_embeddings=tt_position_embeddings))

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_encoder_layer(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_aifi = torch_rtdetr.model.encoder.aifi[0]
    torch_module = torch_aifi.layers[0]
    torch_input = make_encoder_hidden_states(torch_rtdetr.config)
    position_embeddings = make_encoder_position_embeddings(torch_aifi, torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_encoder_layer,
    )

    with torch.no_grad():
        torch_output = torch_module(
            hidden_states=torch_input,
            attention_mask=None,
            spatial_position_embeddings=position_embeddings,
        )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_position_embeddings = ttnn.from_torch(
        position_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrEncoderLayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output = ttnn.to_torch(tt_module(tt_input, position_embeddings=tt_position_embeddings))

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_aifi_layer(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.encoder.aifi[0]
    torch.manual_seed(0)
    torch_input = torch.randn(1, torch_rtdetr.config.encoder_hidden_dim, ENCODER_HEIGHT, ENCODER_WIDTH)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_aifi_layer,
    )

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, ENCODER_HEIGHT * ENCODER_WIDTH, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrAIFILayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=1,
        height=ENCODER_HEIGHT,
        width=ENCODER_WIDTH,
    )
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.reshape(1, output_height, output_width, -1).permute(0, 3, 1, 2)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_conv_norm_layer(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.encoder.lateral_convs[0]
    torch_input = make_hybrid_encoder_inputs(torch_rtdetr)[-1]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_conv_norm_layer,
    )

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input, batch_size, input_height, input_width = to_tt_feature_map(torch_input, device)
    tt_module = TtRTDetrConvNormLayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        in_channels=torch_rtdetr.config.encoder_hidden_dim,
        out_channels=torch_rtdetr.config.encoder_hidden_dim,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        activation=torch_rtdetr.config.activation_function,
    )
    tt_output, output_height, output_width = tt_module(
        tt_input,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )
    tt_output = to_torch_feature_map(
        tt_output,
        batch_size,
        output_height,
        output_width,
        torch_rtdetr.config.encoder_hidden_dim,
    )

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_rep_vgg_block(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_encoder = torch_rtdetr.model.encoder
    torch_csp = torch_encoder.fpn_blocks[0]
    projected_features = make_hybrid_encoder_inputs(torch_rtdetr)

    with torch.no_grad():
        top_feature = torch_encoder.lateral_convs[0](projected_features[-1])
        top_feature = torch.nn.functional.interpolate(top_feature, scale_factor=2.0, mode="nearest")
        csp_input = torch.concat([top_feature, projected_features[-2]], dim=1)
        torch_input = torch_csp.conv1(csp_input)
        torch_module = torch_csp.bottlenecks[0]
        torch_output = torch_module(torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_rep_vgg_block,
    )

    tt_input, batch_size, input_height, input_width = to_tt_feature_map(torch_input, device)
    tt_module = TtRTDetrRepVggBlock(
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
    tt_output = to_torch_feature_map(
        tt_output,
        batch_size,
        output_height,
        output_width,
        torch_rtdetr.config.encoder_hidden_dim,
    )

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_csp_rep_layer(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_encoder = torch_rtdetr.model.encoder
    torch_module = torch_encoder.fpn_blocks[0]
    projected_features = make_hybrid_encoder_inputs(torch_rtdetr)

    with torch.no_grad():
        top_feature = torch_encoder.lateral_convs[0](projected_features[-1])
        top_feature = torch.nn.functional.interpolate(top_feature, scale_factor=2.0, mode="nearest")
        torch_input = torch.concat([top_feature, projected_features[-2]], dim=1)
        torch_output = torch_module(torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_csp_rep_layer,
    )

    tt_input, batch_size, input_height, input_width = to_tt_feature_map(torch_input, device)
    tt_module = TtRTDetrCSPRepLayer(
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
    tt_output = to_torch_feature_map(
        tt_output,
        batch_size,
        output_height,
        output_width,
        torch_rtdetr.config.encoder_hidden_dim,
    )

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_hybrid_encoder(device):
    dtype = ttnn.bfloat16
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.encoder
    torch_inputs = make_hybrid_encoder_inputs(torch_rtdetr)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_hybrid_encoder,
    )

    with torch.no_grad():
        torch_outputs = torch_module(inputs_embeds=list(torch_inputs)).last_hidden_state

    tt_inputs = []
    for torch_input in torch_inputs:
        tt_input, _, height, width = to_tt_feature_map(torch_input, device, dtype=dtype)
        tt_inputs.append((tt_input, height, width))

    batch_size = torch_inputs[0].shape[0]
    tt_module = TtRTDetrHybridEncoder(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=dtype,
    )
    tt_outputs = tt_module(tt_inputs, batch_size=batch_size)

    for output_index, (torch_output, (tt_output, height, width)) in enumerate(zip(torch_outputs, tt_outputs)):
        tt_output = to_torch_feature_map(
            tt_output,
            batch_size,
            height,
            width,
            torch_rtdetr.config.encoder_hidden_dim,
        )
        _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.95)
        logger.info(f"Output {output_index}: {pcc_message}")


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
