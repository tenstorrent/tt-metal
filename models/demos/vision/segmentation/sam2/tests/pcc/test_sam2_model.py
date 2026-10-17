# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0
from models.demos.vision.segmentation.sam2.common import load_sam2_model_and_processor
from models.demos.vision.segmentation.sam2.tt.tt_sam2_video import SAM2_L1_SMALL_SIZE, build_tt_sam2_model

MODELS_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "models")
SAMPLE_IMAGE_PATH = MODELS_ROOT / "sample_data" / "huggingface_cat_image.jpg"
N300_DEVICE_PARAMS = {
    "l1_small_size": SAM2_L1_SMALL_SIZE,
}
N300_VIDEO_DEVICE_PARAMS = {**N300_DEVICE_PARAMS, "num_command_queues": 2}


def _processed_pixels(processor, seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    return processor(images=image, return_tensors="pt").pixel_values


def _sample_pixels(processor):
    with Image.open(SAMPLE_IMAGE_PATH) as image:
        return processor(images=image.convert("RGB"), return_tensors="pt").pixel_values


def _assert_pcc(golden, actual, threshold):
    golden = golden.float()
    actual = actual.float()
    assert tuple(actual.shape) == tuple(
        golden.shape
    ), f"expected shape {tuple(golden.shape)}, got {tuple(actual.shape)}"
    passed, pcc_value = comp_pcc(golden, actual, pcc=threshold)
    assert passed, f"PCC {pcc_value} did not meet {threshold}"
    return float(pcc_value)


def _safe_deallocate_output(output):
    if output is None:
        return
    for tensor in output.values():
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


def _golden_prediction(model, vision, inputs, multimask_output=True):
    fpn = list(vision.fpn_hidden_states)
    image_embeddings = fpn[-1] + model.no_memory_embedding.permute(1, 2, 0).reshape(1, 256, 1, 1)
    high_resolution_features = [model.mask_decoder.conv_s0(fpn[0]), model.mask_decoder.conv_s1(fpn[1])]
    prompt_inputs = dict(inputs)
    if prompt_inputs.get("input_points") is None and prompt_inputs.get("input_boxes") is None:
        prompt_inputs["input_points"] = torch.zeros(1, 1, 1, 2)
        prompt_inputs["input_labels"] = -torch.ones(1, 1, 1, dtype=torch.int32)
    if prompt_inputs.get("input_masks") is not None:
        prompt_inputs["input_masks"] = F.interpolate(
            prompt_inputs["input_masks"].float(),
            size=model.prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    sparse, dense = model.get_prompt_embeddings(**prompt_inputs)
    return model.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=model.get_image_wide_positional_embeddings(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask_output,
        high_resolution_features=high_resolution_features,
    )


def _image_prompt(prompt_type):
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[:, :, 200:820, 260:760] = 1.0
    cases = {
        "point": {
            "input_points": torch.tensor([[[[420.0, 500.0]]]]),
            "input_labels": torch.tensor([[[1]]], dtype=torch.int32),
        },
        "box": {"input_boxes": torch.tensor([[[100.0, 120.0, 800.0, 900.0]]])},
        "point_box": {
            "input_points": torch.tensor([[[[420.0, 500.0]]]]),
            "input_labels": torch.tensor([[[1]]], dtype=torch.int32),
            "input_boxes": torch.tensor([[[100.0, 120.0, 800.0, 900.0]]]),
        },
        "mask": {"input_masks": mask},
    }
    return cases[prompt_type]


@pytest.mark.parametrize("prompt_type", ["point", "box", "mask", "point_box"])
@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_image_hf_pcc(prompt_type, mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    pixels = _sample_pixels(processor)
    inputs = _image_prompt(prompt_type)
    with torch.no_grad():
        vision = hf_model.vision_encoder(pixels, return_dict=True)
        golden_masks, golden_iou, golden_tokens, golden_object = _golden_prediction(hf_model, vision, inputs)

    model = build_tt_sam2_model(hf_model, mesh_device)
    actual = None
    try:
        model.set_image(pixels)
        actual = model.predict(**inputs, multimask_output=True)
        mask_pcc = _assert_pcc(golden_masks.squeeze(1), ttnn.to_torch(actual["low_res_masks"]), 0.98)
        iou_pcc = _assert_pcc(golden_iou.squeeze(1), ttnn.to_torch(actual["iou_scores"]), 0.95)
        token_pcc = _assert_pcc(golden_tokens.squeeze(1), ttnn.to_torch(actual["mask_tokens"]), 0.95)
        actual_object = ttnn.to_torch(actual["object_score_logits"]).float()
        golden_object = golden_object.squeeze(1).float()
        torch.testing.assert_close(actual_object, golden_object, rtol=0.1, atol=0.5)
        logger.info(
            "SAM2 image {}: mask PCC={:.6f}, IoU PCC={:.6f}, mask-token PCC={:.6f}, "
            "object-score max abs error={:.6f}",
            prompt_type,
            mask_pcc,
            iou_pcc,
            token_pcc,
            torch.max(torch.abs(actual_object - golden_object)).item(),
        )
    finally:
        _safe_deallocate_output(actual)
        model.close()


def _video_prompt(prompt_type):
    cases = {
        "point": {
            "input_points": torch.tensor([[[[512.0, 512.0]]]]),
            "input_labels": torch.tensor([[[1]]], dtype=torch.int32),
        },
        "box": {
            "input_boxes": torch.tensor([[[256.0, 256.0, 768.0, 768.0]]]),
        },
    }
    return cases[prompt_type]


@pytest.mark.parametrize("prompt_type", ["point", "box"])
@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_VIDEO_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_video_hf_pcc(prompt_type, mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    frame_count = 1 if prompt_type == "box" else 3
    frames = [_processed_pixels(processor, seed) for seed in range(frame_count)]
    prompts = _video_prompt(prompt_type)

    hf_session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        hf_session,
        frame_idx=0,
        obj_ids=0,
        input_points=prompts.get("input_points"),
        input_labels=prompts.get("input_labels"),
        input_boxes=prompts.get("input_boxes"),
        original_size=(1024, 1024),
    )
    with torch.no_grad():
        golden_frames = [
            hf_model(inference_session=hf_session, frame_idx=frame_idx, frame=frame)
            for frame_idx, frame in enumerate(frames)
        ]

    model = build_tt_sam2_model(hf_model, mesh_device, bridge_upload_cq_id=1)
    session = None
    try:
        session = model.start_video_session()
        actual_frames = list(session.run(frames, prompts))
        assert len(actual_frames) == len(golden_frames)
        if prompt_type == "box":
            golden_pointer = hf_session.output_dict_per_obj[0]["cond_frame_outputs"][0]["object_pointer"].float()
            actual_pointer = ttnn.to_torch(actual_frames[0]["obj_ptr"]).float().reshape_as(golden_pointer)
            pointer_pcc = _assert_pcc(golden_pointer, actual_pointer, 0.98)
            logger.info(
                "SAM2 video box first-frame object-pointer PCC={:.6f}; "
                "this validates the checkpoint's single-mask token policy",
                pointer_pcc,
            )
        else:
            pcc_values = [
                _assert_pcc(golden.pred_masks, ttnn.to_torch(actual["pred_masks"]), threshold)
                for golden, actual, threshold in zip(golden_frames, actual_frames, (0.98, 0.95, 0.95))
            ]
            logger.info("SAM2 video {} frame PCCs={}", prompt_type, pcc_values)
    finally:
        if session is not None:
            session.close()
        model.close()
