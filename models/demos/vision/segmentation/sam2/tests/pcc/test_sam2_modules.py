# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0
from models.demos.vision.segmentation.sam2.common import load_sam2_model_and_processor
from models.demos.vision.segmentation.sam2.tt.model_preprocessing import (
    preprocess_memory_attention,
    preprocess_memory_encoder,
)
from models.demos.vision.segmentation.sam2.tt.tt_memory_attention import ATTENTION_BANK_DTYPE, TtMemoryAttention
from models.demos.vision.segmentation.sam2.tt.tt_memory_encoder import TtMemoryEncoder
from models.demos.vision.segmentation.sam2.tt.tt_sam2_video import SAM2_L1_SMALL_SIZE, build_tt_sam2_model

N300_DEVICE_PARAMS = {
    "l1_small_size": SAM2_L1_SMALL_SIZE,
    "require_exact_physical_num_devices": True,
}


def _processed_pixels(processor, seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    return processor(images=image, return_tensors="pt").pixel_values


def _assert_pcc(golden, actual, threshold=0.98):
    golden = golden.float()
    actual = actual.float()
    assert tuple(actual.shape) == tuple(
        golden.shape
    ), f"expected shape {tuple(golden.shape)}, got {tuple(actual.shape)}"
    passed, pcc_value = comp_pcc(golden, actual, pcc=threshold)
    assert passed, f"PCC {pcc_value} did not meet {threshold}"
    return float(pcc_value)


def _safe_deallocate(*tensors):
    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


def _tt_tensor(tensor, device, *, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _golden_prediction(model, vision, inputs, multimask_output=True):
    fpn = list(vision.fpn_hidden_states)
    image_embeddings = fpn[-1] + model.no_memory_embedding.permute(1, 2, 0).reshape(1, 256, 1, 1)
    high_resolution_features = [model.mask_decoder.conv_s0(fpn[0]), model.mask_decoder.conv_s1(fpn[1])]
    sparse, dense = model.get_prompt_embeddings(**inputs)
    return model.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=model.get_image_wide_positional_embeddings(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask_output,
        high_resolution_features=high_resolution_features,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_hiera_and_fpn(mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    pixels = _processed_pixels(processor)
    with torch.no_grad():
        backbone = hf_model.vision_encoder.backbone(pixels, output_hidden_states=True, return_dict=True)
        vision = hf_model.vision_encoder(pixels, output_hidden_states=True, return_dict=True)
        golden_fpn = (
            hf_model.mask_decoder.conv_s0(vision.fpn_hidden_states[0]),
            hf_model.mask_decoder.conv_s1(vision.fpn_hidden_states[1]),
            vision.fpn_hidden_states[2],
        )

    model = build_tt_sam2_model(hf_model, mesh_device)
    device_input = stages = fpn = None
    try:
        folded = model._folded_inputs.prepare(pixels, 0)
        device_input = ttnn.from_torch(
            folded,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=model.encoder_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stages = model.image_encoder.trunk.forward(device_input)
        hiera_pccs = [
            _assert_pcc(golden, ttnn.to_torch(actual))
            for golden, actual in zip(backbone.intermediate_hidden_states, stages)
        ]
        fpn = model.image_encoder.neck(stages)
        fpn_pccs = [
            _assert_pcc(golden, ttnn.to_torch(actual).permute(0, 3, 1, 2))
            for golden, actual in zip(golden_fpn, fpn[:-1])
        ]
        logger.info("SAM2 Hiera stage PCCs={} FPN PCCs={}", hiera_pccs, fpn_pccs)
    finally:
        _safe_deallocate(device_input)
        for tensor in stages or ():
            _safe_deallocate(tensor)
        for tensor in fpn or ():
            _safe_deallocate(tensor)
        model.close()


@pytest.mark.parametrize("prompt_type", ["point", "box", "mask", "point_box"])
@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_prompt_encoder(prompt_type, mesh_device, reset_seeds, model_location_generator):
    hf_model, _ = load_sam2_model_and_processor(model_location_generator)
    points = torch.tensor([[[[420.0, 500.0]]]])
    labels = torch.tensor([[[1]]], dtype=torch.int32)
    boxes = torch.tensor([[[100.0, 120.0, 800.0, 900.0]]])
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[:, :, 200:820, 260:760] = 1.0

    inputs = {}
    if prompt_type in ("point", "point_box"):
        inputs.update(input_points=points, input_labels=labels)
    if prompt_type in ("box", "point_box"):
        inputs["input_boxes"] = boxes
    if prompt_type == "mask":
        inputs["input_masks"] = mask

    hf_inputs = dict(inputs)
    if prompt_type == "mask":
        hf_inputs["input_masks"] = F.interpolate(
            mask,
            size=hf_model.prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    with torch.no_grad():
        golden_sparse, golden_dense = hf_model.get_prompt_embeddings(**hf_inputs)

    model = build_tt_sam2_model(hf_model, mesh_device)
    actual_sparse = actual_dense = None
    try:
        prompt_encoder = model.image_head.prompt_encoder
        actual_sparse = prompt_encoder.embed_sparse(
            inputs.get("input_points"), inputs.get("input_labels"), inputs.get("input_boxes")
        )
        if prompt_type == "mask":
            actual_dense = prompt_encoder.embed_dense(mask)
            dense_torch = ttnn.to_torch(actual_dense)
        else:
            dense_sequence = ttnn.to_torch(model.image_head.no_mask_dense_seq)
            dense_torch = dense_sequence.reshape(1, 64, 64, 256).permute(0, 3, 1, 2)

        sparse_pcc = None
        if golden_sparse is None:
            assert actual_sparse is None, "TT prompt encoder produced unexpected sparse embeddings"
        else:
            sparse_pcc = _assert_pcc(golden_sparse.squeeze(1), ttnn.to_torch(actual_sparse))
        dense_pcc = _assert_pcc(golden_dense, dense_torch)
        logger.info("SAM2 {} prompt encoder sparse PCC={} dense PCC={}", prompt_type, sparse_pcc, dense_pcc)
    finally:
        _safe_deallocate(actual_sparse, actual_dense)
        model.close()


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_mask_decoder(mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    pixels = _processed_pixels(processor)
    points = torch.tensor([[[[420.0, 500.0]]]])
    labels = torch.tensor([[[1]]], dtype=torch.int32)
    inputs = {"input_points": points, "input_labels": labels}
    with torch.no_grad():
        vision = hf_model.vision_encoder(pixels, return_dict=True)
        golden_masks, golden_iou, golden_tokens, golden_object = _golden_prediction(hf_model, vision, inputs)

    model = build_tt_sam2_model(hf_model, mesh_device)
    sparse = image_features = actual_masks = actual_iou = actual_tokens = actual_object = None
    try:
        model.set_image(pixels)
        image_head = model.image_head
        sparse = image_head.prompt_encoder.embed_sparse(points, labels)
        top = model._image_cache.top_nhwc
        image_features = ttnn.add(
            ttnn.reshape(top, (int(top.shape[0]), int(top.shape[1]) * int(top.shape[2]), image_head.hidden_dim)),
            image_head.no_mem_embed,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        actual_masks, actual_iou, actual_tokens, actual_object = image_head.sam_mask_decoder(
            image_embeddings=image_features,
            image_pe=image_head.dense_pe_seq,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=image_head.no_mask_dense_seq,
            multimask_output=True,
            high_res_features=model._image_cache.high_res,
        )
        mask_pcc = _assert_pcc(golden_masks.squeeze(1), ttnn.to_torch(actual_masks))
        iou_pcc = _assert_pcc(golden_iou.squeeze(1), ttnn.to_torch(actual_iou))
        token_pcc = _assert_pcc(golden_tokens.squeeze(1), ttnn.to_torch(actual_tokens), threshold=0.95)
        torch.testing.assert_close(
            ttnn.to_torch(actual_object).float(), golden_object.squeeze(1).float(), rtol=0.1, atol=0.5
        )
        logger.info("SAM2 mask decoder mask PCC={} IoU PCC={} mask-token PCC={}", mask_pcc, iou_pcc, token_pcc)
    finally:
        _safe_deallocate(sparse, image_features, actual_masks, actual_iou, actual_tokens, actual_object)
        model.close()


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": SAM2_L1_SMALL_SIZE}], indirect=True)
def test_sam2_memory_encoder(device, reset_seeds, model_location_generator):
    hf_model, _ = load_sam2_model_and_processor(model_location_generator)
    vision_features = torch.randn(1, 256, 64, 64)
    masks = torch.sigmoid(torch.randn(1, 1, 1024, 1024))
    with torch.no_grad():
        golden_memory, _ = hf_model.memory_encoder(vision_features, masks)

    memory_encoder = TtMemoryEncoder(preprocess_memory_encoder(hf_model.memory_encoder, device), device)
    vision_tt = _tt_tensor(vision_features.permute(0, 2, 3, 1).contiguous(), device)
    masks_tt = _tt_tensor(masks.permute(0, 2, 3, 1).contiguous(), device)
    actual_memory = None
    try:
        actual_memory = memory_encoder(vision_tt, masks_tt)
        memory_pcc = _assert_pcc(golden_memory, ttnn.to_torch(actual_memory).permute(0, 3, 1, 2))
        logger.info("SAM2 memory encoder PCC={}", memory_pcc)
    finally:
        _safe_deallocate(vision_tt, masks_tt, actual_memory)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": SAM2_L1_SMALL_SIZE}], indirect=True)
def test_sam2_memory_attention(device, reset_seeds, model_location_generator):
    hf_model, _ = load_sam2_model_and_processor(model_location_generator)
    vision_features = torch.randn(1, 256, 64, 64)
    masks = torch.sigmoid(torch.randn(1, 1, 1024, 1024))
    with torch.no_grad():
        memory, memory_pos_bchw = hf_model.memory_encoder(vision_features, masks)
    current = torch.randn(4096, 1, 256)
    current_pos = hf_model.get_image_wide_positional_embeddings().flatten(2).permute(2, 0, 1).contiguous()
    memory = memory.flatten(2).permute(2, 0, 1).contiguous()
    memory_pos = memory_pos_bchw.flatten(2).permute(2, 0, 1).contiguous()
    memory_pos = memory_pos + hf_model.memory_temporal_positional_encoding[-1]
    with torch.no_grad():
        golden = hf_model.memory_attention(
            current_vision_features=current,
            memory=memory,
            current_vision_position_embeddings=current_pos,
            memory_posision_embeddings=memory_pos,
            num_object_pointer_tokens=0,
        )

    parameters = preprocess_memory_attention(hf_model.memory_attention, device)
    memory_attention = TtMemoryAttention(parameters, device)
    memory_with_position_tt = _tt_tensor((memory + memory_pos).transpose(0, 1).contiguous(), device)
    packed_keys = latent_values = actual = None
    projected_keys = []
    current_tt = current_pos_tt = None
    try:
        packed_keys = ttnn.linear(
            memory_with_position_tt,
            parameters.bank_k_proj.weight,
            bias=parameters.bank_k_proj.bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        split_keys = ttnn.split(packed_keys, 256, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cosine, sine = memory_attention.rope.tables()
        for split_key in split_keys:
            key = ttnn.reshape(split_key, (1, 1, 4096, 256))
            projected_keys.append(memory_attention.rope.apply(key, cosine, sine))
            _safe_deallocate(key)
        latent_values = _tt_tensor(memory.transpose(0, 1).unsqueeze(1).contiguous(), device, dtype=ATTENTION_BANK_DTYPE)
        current_tt = _tt_tensor(current.transpose(0, 1).contiguous(), device)
        current_pos_tt = _tt_tensor(current_pos.transpose(0, 1).contiguous(), device)
        actual = memory_attention(current_tt, current_pos_tt, [(key, latent_values) for key in projected_keys])
        attention_pcc = _assert_pcc(golden.reshape(1, 4096, 256), ttnn.to_torch(actual))
        logger.info("SAM2 memory attention PCC={}", attention_pcc)
    finally:
        _safe_deallocate(memory_with_position_tt, packed_keys, latent_values, current_tt, current_pos_tt, actual)
        for key in projected_keys:
            _safe_deallocate(key)
