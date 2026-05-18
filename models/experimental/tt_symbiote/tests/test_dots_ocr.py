# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E TTNN pipeline tests for dots.ocr (text-only and vision+text)."""

import os
import time

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}


def _resolve_mesh_device_shape():
    mesh_device = os.environ.get("MESH_DEVICE")
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP":
        return DOTS_OCR_DP_MESH_DEVICE_MAP.get(
            mesh_device, MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))
        )
    return MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))


def _dots_ocr_mesh_num_devices():
    sh = _resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def _dots_ocr_device_params():
    dp = {"trace_region_size": 300000000, "num_command_queues": 1}
    if _dots_ocr_mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path():
    """Resolve dots.ocr model path: env var > HF cache > model ID for auto-download."""
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


def _dots_ocr_pipeline_batch_size():
    """Match ``TTNNDotsOCRPipeline`` batch to mesh size when DP is requested.

    DP sharding in the pipeline requires ``batch_size == num_devices`` on the
    mesh. ``DOTS_OCR_PARALLELISM=DP`` alone only changes the *fixture* mesh
    shape (e.g. N300 ``(2, 1)``); without this, tests still run batch 1.
    """
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() != "DP":
        return 1
    n = _dots_ocr_mesh_num_devices()
    return n if n > 1 else 1


def _dots_ocr_stack_input_ids_for_dp(input_ids: torch.Tensor) -> torch.Tensor:
    """Turn ``[1, S]`` into ``[B, S]`` by repeating the same prompt on each stream."""
    bs = _dots_ocr_pipeline_batch_size()
    if bs <= 1 or input_ids.shape[0] == bs:
        return input_ids
    if input_ids.shape[0] != 1:
        raise ValueError(f"DP batch stacking expects base shape [1, S], got {tuple(input_ids.shape)}")
    return input_ids.expand(bs, -1).contiguous()


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
def test_dots_ocr_text(mesh_device):
    """Test standalone TTNN pipeline for dots.ocr (text-only, no vision)."""

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
    )

    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "What is optical character recognition and how does it work?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])

    pipeline.warmup(input_ids)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(input_ids, max_new_tokens=128)
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        text = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline TEXT OUTPUT: {text}")

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Text Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(text.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_text_timing_stats.csv")
    pipeline.release()


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_link",
    [
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg",
    ],
)
def test_dots_ocr_vision(mesh_device, image_link):
    """Test standalone TTNN pipeline for dots.ocr with vision (image + text).

    Default mesh comes from ``MESH_DEVICE`` (e.g. N300 ``(1, 2)``). With
    ``DOTS_OCR_PARALLELISM=DP``, the fixture uses ``DOTS_OCR_DP_MESH_DEVICE_MAP``
    (N300 ``(2, 1)``). In DP mode the test sets ``batch_size == num_devices`` and
    repeats the same prompt on each stream so dual-stream sharding is exercised.
    """
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import requests

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
    )

    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    # Load and crop the image
    image = Image.open(requests.get(image_link, stream=True).raw)
    original_width, original_height = image.size

    # Crop to 57.5% of original height from the top
    new_height = int(original_height * 0.575)
    top = 0
    bottom = new_height

    # Crop box: (left, top, right, bottom)
    image = image.crop((0, top, original_width, bottom))

    print(f"Cropped image from {original_width}x{original_height} to {original_width}x{new_height}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        max_length=2800,
        return_tensors="pt",
    )

    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=180,
        stop_on_eos=False,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        decoded = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline VISION OUTPUT: {decoded}")

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Vision Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_vision_timing_stats.csv")
    pipeline.release()
