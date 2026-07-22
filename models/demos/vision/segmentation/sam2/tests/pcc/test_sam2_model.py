# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

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

N300_DEVICE_PARAMS = {
    "l1_small_size": SAM2_L1_SMALL_SIZE,
    "require_exact_physical_num_devices": True,
}
N300_VIDEO_DEVICE_PARAMS = {**N300_DEVICE_PARAMS, "num_command_queues": 2}


def _processed_pixels(processor, seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    return processor(images=image, return_tensors="pt").pixel_values


def _sample_pixels(processor):
    with Image.open("models/sample_data/huggingface_cat_image.jpg") as image:
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
def test_sam2_image_pcc(prompt_type, mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    pixels = _sample_pixels(processor)
    inputs = _image_prompt(prompt_type)
    with torch.no_grad():
        vision = hf_model.vision_encoder(pixels, return_dict=True)
        golden_masks, golden_iou, golden_tokens, golden_object = _golden_prediction(hf_model, vision, inputs)

    model = build_tt_sam2_model(hf_model, mesh_device)
    actual = None
    try:
        assert set(model.encoder_device.get_device_ids()) == set(
            ttnn.get_pcie_device_ids()
        ), "image inference must stay on the PCIe-attached ASIC"
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


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_VIDEO_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_sam2_three_frame_video(mesh_device, reset_seeds, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    frames = [_processed_pixels(processor, seed) for seed in range(3)]
    prompts = {
        "input_points": torch.tensor([[[[512.0, 512.0]]]]),
        "input_labels": torch.tensor([[[1]]], dtype=torch.int32),
    }

    hf_session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        hf_session,
        frame_idx=0,
        obj_ids=0,
        input_points=prompts["input_points"],
        input_labels=prompts["input_labels"],
        original_size=(1024, 1024),
    )
    with torch.no_grad():
        golden_frames = [
            hf_model(inference_session=hf_session, frame_idx=frame_idx, frame=frame)
            for frame_idx, frame in enumerate(frames)
        ]

    model = build_tt_sam2_model(hf_model, mesh_device, bridge_upload_cq_id=1)
    session = pipeline_session = early_close_session = None
    image_output = None
    try:
        encoder_ids = set(model.encoder_device.get_device_ids())
        tracker_ids = set(model.tracker_device.get_device_ids())
        assert encoder_ids == set(ttnn.get_pcie_device_ids()), "encoder must use the PCIe-attached ASIC"
        assert tracker_ids == set(mesh_device.get_device_ids()) - encoder_ids, "tracker must use the remote ASIC"
        assert encoder_ids.isdisjoint(tracker_ids), "encoder and tracker submeshes must not overlap"

        model.set_image(frames[0])
        image_output = model.predict(**prompts, multimask_output=False)
        assert tuple(image_output["low_res_masks"].shape) == (
            1,
            1,
            256,
            256,
        ), "unexpected image mask shape before video"
        _safe_deallocate_output(image_output)
        image_output = None

        session = model.start_video_session()
        actual_frames = [
            session.step(frames[0], prompts=prompts),
            session.step(frames[1]),
            session.step(frames[2]),
        ]
        for golden, actual, threshold in zip(golden_frames, actual_frames, (0.98, 0.95, 0.95)):
            _assert_pcc(golden.pred_masks, ttnn.to_torch(actual["pred_masks"]), threshold)
        golden_high_res = F.interpolate(
            golden_frames[0].pred_masks.float(), (1024, 1024), mode="bilinear", align_corners=False
        )
        _assert_pcc(golden_high_res, ttnn.to_torch(actual_frames[0]["pred_masks_high_res"]), 0.98)
        assert len(session.bank["cond_frame_outputs"]) == 1, "conditioning bank must retain frame zero"
        assert (
            len(session.bank["non_cond_frame_outputs"]) <= model.max_obj_ptrs_in_encoder - 1
        ), "non-conditioning bank exceeded its bound"
        session.close()
        session = None

        pipeline_session = model.start_video_session()
        pipelined = list(pipeline_session.run(frames, prompts))
        assert len(pipelined) == 3, f"expected 3 pipelined outputs, got {len(pipelined)}"
        for golden, actual, threshold in zip(golden_frames, pipelined, (0.98, 0.95, 0.95)):
            _assert_pcc(golden.pred_masks, ttnn.to_torch(actual["pred_masks"]), threshold)
        pipeline_session.close()
        pipeline_session = None

        early_close_session = model.start_video_session()
        iterator = early_close_session.run(frames, prompts)
        first = next(iterator)
        _assert_pcc(golden_frames[0].pred_masks, ttnn.to_torch(first["pred_masks"]), 0.98)
        early_close_session.close()
        early_close_session = None

        model.set_image(frames[0])
        image_output = model.predict(**prompts, multimask_output=False)
        assert tuple(image_output["low_res_masks"].shape) == (
            1,
            1,
            256,
            256,
        ), "unexpected image mask shape after video"
    finally:
        _safe_deallocate_output(image_output)
        for active_session in (session, pipeline_session, early_close_session):
            if active_session is not None:
                active_session.close()
        model.close()
