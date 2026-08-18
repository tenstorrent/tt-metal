# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import (
    custom_preprocessor,
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
    preprocess_rtdetr_decoder,
    preprocess_rtdetr_decoder_layer,
    preprocess_rtdetr_encoder_layer,
    preprocess_rtdetr_hybrid_encoder,
    preprocess_rtdetr_mlp,
    preprocess_rtdetr_mlp_prediction_head,
    preprocess_rtdetr_multiscale_deformable_attention,
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
from models.demos.vision.detection.rtdetr.tt.decoder import (
    TtRTDetrDecoder,
    TtRTDetrDecoderLayer,
    TtRTDetrDecoderMLP,
    TtRTDetrMLPPredictionHead,
    TtRTDetrMultiscaleDeformableAttention,
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
)
from models.demos.vision.detection.rtdetr.tt.model import TtRTDetrModel
from tests.ttnn.utils_for_testing import assert_with_pcc

MODEL_NAME = "PekingU/rtdetr_r50vd"
V2_MODEL_NAME = "PekingU/rtdetr_v2_r50vd"
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
    output = TtRTDetrAIFILayer._build_2d_sinusoidal_position_embedding(
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
def test_rtdetr_decoder_mlp(device):
    torch.manual_seed(0)

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.decoder.layers[0].mlp
    torch_input = torch.randn(1, torch_rtdetr.config.num_queries, torch_rtdetr.config.d_model)

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
    tt_module = TtRTDetrDecoderMLP(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output = ttnn.to_torch(tt_module(tt_input))

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_mlp_prediction_head(device):
    torch.manual_seed(0)

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.decoder.query_pos_head
    torch_input = torch.rand(1, torch_rtdetr.config.num_queries, 4)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_mlp_prediction_head,
    )

    with torch.no_grad():
        torch_output = torch_module(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrMLPPredictionHead(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        input_dim=4,
        hidden_dim=2 * torch_rtdetr.config.d_model,
        output_dim=torch_rtdetr.config.d_model,
        num_layers=2,
    )
    tt_output = ttnn.to_torch(tt_module(tt_input))

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.99)
    logger.info(pcc_message)


def run_rtdetr_multiscale_deformable_attention_test(device, model_name, model_class):
    torch.manual_seed(0)

    batch_size = 1
    num_queries = 300
    hidden_size = 256
    spatial_shapes_list = ((80, 80), (40, 40), (20, 20))
    sequence_length = sum(height * width for height, width in spatial_shapes_list)

    torch_rtdetr = model_class.from_pretrained(model_name).eval()
    torch_module = torch_rtdetr.model.decoder.layers[0].encoder_attn

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_multiscale_deformable_attention,
    )

    hidden_states = torch.randn(batch_size, num_queries, hidden_size)
    encoder_hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
    position_embeddings = torch.randn(batch_size, num_queries, hidden_size)
    reference_points = torch.rand(batch_size, num_queries, 1, 4)
    reference_points[..., 2:] = reference_points[..., 2:] * 0.5 + 0.05
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
    level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

    with torch.no_grad():
        torch_output, torch_attention_weights = torch_module(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
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
    tt_reference_points = ttnn.from_torch(
        reference_points,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_spatial_shapes = ttnn.from_torch(
        spatial_shapes,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_level_start_index = ttnn.from_torch(
        level_start_index,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrMultiscaleDeformableAttention(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output, tt_attention_weights = tt_module(
        hidden_states=tt_hidden_states,
        encoder_hidden_states=tt_encoder_hidden_states,
        position_embeddings=tt_position_embeddings,
        reference_points=tt_reference_points,
        spatial_shapes=tt_spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=tt_level_start_index,
    )

    tt_output = ttnn.to_torch(tt_output)
    tt_attention_weights = ttnn.to_torch(tt_attention_weights)
    torch_attention_weights = torch_attention_weights.reshape(tt_attention_weights.shape)

    _, output_pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.98)
    logger.info(f"Output: {output_pcc_message}")

    _, weights_pcc_message = assert_with_pcc(torch_attention_weights, tt_attention_weights, pcc=0.98)
    logger.info(f"Attention weights: {weights_pcc_message}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_multiscale_deformable_attention(device):
    run_rtdetr_multiscale_deformable_attention_test(device, MODEL_NAME, RTDetrForObjectDetection)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_v2_multiscale_deformable_attention(device):
    run_rtdetr_multiscale_deformable_attention_test(device, V2_MODEL_NAME, RTDetrV2ForObjectDetection)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_decoder_layer(device):
    torch.manual_seed(0)

    batch_size = 1
    num_queries = 300
    hidden_size = 256
    spatial_shapes_list = ((80, 80), (40, 40), (20, 20))
    sequence_length = sum(height * width for height, width in spatial_shapes_list)

    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.decoder.layers[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_decoder_layer,
    )

    hidden_states = torch.randn(batch_size, num_queries, hidden_size)
    encoder_hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
    object_queries_position_embeddings = torch.randn(batch_size, num_queries, hidden_size)
    reference_points = torch.rand(batch_size, num_queries, 1, 4)
    reference_points[..., 2:] = reference_points[..., 2:] * 0.5 + 0.05
    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
    level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)

    with torch.no_grad():
        torch_output = torch_module(
            hidden_states=hidden_states,
            object_queries_position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            encoder_hidden_states=encoder_hidden_states,
        )

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_object_queries_position_embeddings = ttnn.from_torch(
        object_queries_position_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_reference_points = ttnn.from_torch(
        reference_points,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_spatial_shapes = ttnn.from_torch(
        spatial_shapes,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_level_start_index = ttnn.from_torch(
        level_start_index,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrDecoderLayer(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_output = tt_module(
        hidden_states=tt_hidden_states,
        object_queries_position_embeddings=tt_object_queries_position_embeddings,
        reference_points=tt_reference_points,
        spatial_shapes=tt_spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=tt_level_start_index,
        encoder_hidden_states=tt_encoder_hidden_states,
    )
    tt_output = ttnn.to_torch(tt_output)

    _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.98)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_decoder(device):
    torch_rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).eval()
    torch_module = torch_rtdetr.model.decoder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=preprocess_rtdetr_decoder,
    )

    captured_decoder_inputs = {}

    def capture_decoder_inputs(_, __, kwargs):
        for name in (
            "inputs_embeds",
            "encoder_hidden_states",
            "reference_points",
            "spatial_shapes",
            "spatial_shapes_list",
            "level_start_index",
        ):
            value = kwargs[name]
            captured_decoder_inputs[name] = value.detach().clone() if isinstance(value, torch.Tensor) else value

    hook = torch_module.register_forward_pre_hook(capture_decoder_inputs, with_kwargs=True)
    image_processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    try:
        with torch.no_grad():
            torch_outputs = torch_rtdetr(pixel_values=pixel_values)
    finally:
        hook.remove()

    inputs_embeds = captured_decoder_inputs["inputs_embeds"]
    encoder_hidden_states = captured_decoder_inputs["encoder_hidden_states"]
    reference_points = captured_decoder_inputs["reference_points"]
    spatial_shapes = captured_decoder_inputs["spatial_shapes"]
    spatial_shapes_list = tuple(captured_decoder_inputs["spatial_shapes_list"])
    level_start_index = captured_decoder_inputs["level_start_index"]

    tt_inputs_embeds = ttnn.from_torch(
        inputs_embeds,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_reference_points = ttnn.from_torch(
        reference_points,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_spatial_shapes = ttnn.from_torch(
        spatial_shapes,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_level_start_index = ttnn.from_torch(
        level_start_index,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_module = TtRTDetrDecoder(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
    )
    (
        tt_last_hidden_state,
        tt_intermediate_hidden_states,
        tt_intermediate_logits,
        tt_intermediate_reference_points,
    ) = tt_module(
        inputs_embeds=tt_inputs_embeds,
        encoder_hidden_states=tt_encoder_hidden_states,
        reference_points=tt_reference_points,
        spatial_shapes=tt_spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=tt_level_start_index,
    )

    tt_last_hidden_state = ttnn.to_torch(tt_last_hidden_state)
    tt_intermediate_hidden_states = ttnn.to_torch(tt_intermediate_hidden_states)
    tt_intermediate_logits = ttnn.to_torch(tt_intermediate_logits)
    tt_intermediate_reference_points = ttnn.to_torch(tt_intermediate_reference_points)

    for layer_index in range(torch_rtdetr.config.decoder_layers):
        _, hidden_state_pcc = assert_with_pcc(
            torch_outputs.intermediate_hidden_states[:, layer_index],
            tt_intermediate_hidden_states[:, layer_index],
            pcc=0.0,
        )
        logger.info(f"Layer {layer_index} hidden state: {hidden_state_pcc}")

        _, layer_logits_pcc = assert_with_pcc(
            torch_outputs.intermediate_logits[:, layer_index],
            tt_intermediate_logits[:, layer_index],
            pcc=0.0,
        )
        logger.info(f"Layer {layer_index} logits: {layer_logits_pcc}")

        _, layer_reference_points_pcc = assert_with_pcc(
            torch_outputs.intermediate_reference_points[:, layer_index],
            tt_intermediate_reference_points[:, layer_index],
            pcc=0.0,
        )
        logger.info(f"Layer {layer_index} reference points: {layer_reference_points_pcc}")

    _, last_hidden_state_pcc = assert_with_pcc(
        torch_outputs.last_hidden_state,
        tt_last_hidden_state,
        pcc=0.95,
    )
    logger.info(f"Last hidden state: {last_hidden_state_pcc}")

    _, hidden_states_pcc = assert_with_pcc(
        torch_outputs.intermediate_hidden_states,
        tt_intermediate_hidden_states,
        pcc=0.95,
    )
    logger.info(f"Intermediate hidden states: {hidden_states_pcc}")

    _, logits_pcc = assert_with_pcc(
        torch_outputs.intermediate_logits,
        tt_intermediate_logits,
        pcc=0.95,
    )
    logger.info(f"Intermediate logits: {logits_pcc}")

    _, reference_points_pcc = assert_with_pcc(
        torch_outputs.intermediate_reference_points,
        tt_intermediate_reference_points,
        pcc=0.95,
    )
    logger.info(f"Intermediate reference points: {reference_points_pcc}")


def run_rtdetr_model_test(device, model_name, model_class):
    torch_rtdetr = model_class.from_pretrained(model_name).eval()
    torch_module = torch_rtdetr.model

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_module,
        custom_preprocessor=custom_preprocessor,
    )

    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    image = load_coco_image()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    _, _, input_height, input_width = pixel_values.shape

    with torch.no_grad():
        torch_outputs = torch_module(pixel_values=pixel_values)

    tt_pixel_values = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_module = TtRTDetrModel(
        config=torch_rtdetr.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        input_height=input_height,
        input_width=input_width,
    )
    (
        tt_last_hidden_state,
        tt_intermediate_hidden_states,
        tt_intermediate_logits,
        tt_intermediate_reference_points,
        logits,
        pred_boxes,
    ) = tt_module(tt_pixel_values)

    torch_scores = torch_outputs.enc_outputs_class.max(dim=-1).values
    torch_topk_ind = torch.topk(torch_scores, torch_rtdetr.config.num_queries, dim=1).indices
    tt_topk_ind = ttnn.to_torch(tt_module.topk_ind).long().reshape(torch_topk_ind.shape)

    tt_last_hidden_state = ttnn.to_torch(tt_last_hidden_state)
    tt_intermediate_hidden_states = ttnn.to_torch(tt_intermediate_hidden_states)
    tt_intermediate_logits = ttnn.to_torch(tt_intermediate_logits)
    tt_intermediate_reference_points = ttnn.to_torch(tt_intermediate_reference_points)

    # Decoder queries are an unordered set. Align queries selected by both implementations.
    matches = torch_topk_ind[0, :, None] == tt_topk_ind[0, None, :]
    torch_positions, tt_positions = torch.nonzero(matches, as_tuple=True)
    logger.info(f"Shared top-k proposals: {len(torch_positions)}")

    _, last_hidden_state_pcc = assert_with_pcc(
        torch_outputs.last_hidden_state[:, torch_positions],
        tt_last_hidden_state[:, tt_positions],
        pcc=0.90,
    )
    logger.info(f"Aligned last hidden state: {last_hidden_state_pcc}")

    _, hidden_states_pcc = assert_with_pcc(
        torch_outputs.intermediate_hidden_states[:, :, torch_positions],
        tt_intermediate_hidden_states[:, :, tt_positions],
        pcc=0.90,
    )
    logger.info(f"Aligned intermediate hidden states: {hidden_states_pcc}")

    _, logits_pcc = assert_with_pcc(
        torch_outputs.intermediate_logits[:, :, torch_positions],
        tt_intermediate_logits[:, :, tt_positions],
        pcc=0.90,
    )
    logger.info(f"Aligned intermediate logits: {logits_pcc}")

    _, reference_points_pcc = assert_with_pcc(
        torch_outputs.intermediate_reference_points[:, :, torch_positions],
        tt_intermediate_reference_points[:, :, tt_positions],
        pcc=0.90,
    )
    logger.info(f"Aligned intermediate reference points: {reference_points_pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_model(device):
    run_rtdetr_model_test(device, MODEL_NAME, RTDetrForObjectDetection)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_rtdetr_v2_model(device):
    run_rtdetr_model_test(device, V2_MODEL_NAME, RTDetrV2ForObjectDetection)


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
    for stage_index, (
        torch_output,
        (tt_output, output_height, output_width),
        out_channels,
    ) in enumerate(zip(torch_outputs, tt_outputs, hidden_sizes), start=1):
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
    for stage_index, (
        torch_output,
        (tt_output, output_height, output_width),
        out_channels,
    ) in enumerate(zip(torch_outputs, tt_outputs, hidden_sizes), start=1):
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

    for output_index, (
        torch_output,
        (tt_output, output_height, output_width),
        channels,
    ) in enumerate(zip(torch_outputs, tt_outputs, out_channels)):
        tt_output = ttnn.to_torch(tt_output)
        tt_output = tt_output.reshape(batch_size, output_height, output_width, channels)
        tt_output = tt_output.permute(0, 3, 1, 2)

        _, pcc_message = assert_with_pcc(torch_output, tt_output, pcc=0.97)
        logger.info(f"Output {output_index}: {pcc_message}")
