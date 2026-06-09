# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host (CPU) HuggingFace reference for dots.ocr vision+text.

Runs the real ``rednote-hilab/dots.ocr`` model on host with the SAME image,
crop, and prompt as the TTNN pipeline test (``test_dots_ocr.py::test_dots_ocr_vision``),
so its greedy output is the ground-truth reference to compare the TTNN pipeline
(and the TP4 text body) against -- e.g. to settle whether the page running-header
belongs in the transcription.

No Tenstorrent device is needed; this is pure torch + transformers on CPU.

Run::

    pytest -s models/experimental/tt_symbiote/tests/test_dots_ocr_hf_reference.py

Knobs (env):
    DOTS_OCR_MODEL_PATH   local checkpoint dir (else downloads from the HF hub)
    DOTS_OCR_HF_MAX_TOKENS max_new_tokens (default 180, matches the pipeline test)
    DOTS_OCR_HF_DTYPE     'bfloat16' (default; matches the pipeline) or 'float32'
                          (cleanest ground truth, more host memory)

NOTE: CPU greedy decode over the ~2.8k-token image sequence is slow (minutes);
lower DOTS_OCR_HF_MAX_TOKENS for a quick smoke read.
"""

import json
import os

import pytest
import torch

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
IMAGE_LINK = "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"
IMAGE_TOKEN = "<|imgpad|>"
IMAGE_TOKEN_ID = 151665


def _resolve_model_path():
    """dots.ocr model path: env var > HF cache/download > model ID."""
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download(DOTS_OCR_MODEL_ID)


def _build_processor(model_path):
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)
    with open(os.path.join(model_path, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = IMAGE_TOKEN
    processor.image_token_id = IMAGE_TOKEN_ID
    return processor, tokenizer


def _hf_dtype():
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        os.environ.get("DOTS_OCR_HF_DTYPE", "bfloat16").strip().lower()
    ]


@pytest.mark.parametrize("image_link", [IMAGE_LINK])
def test_dots_ocr_hf_reference(image_link):
    """Greedy host HF generation on the demo image; prints the reference text."""
    pytest.importorskip("qwen_vl_utils")
    import requests
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    max_new_tokens = int(os.environ.get("DOTS_OCR_HF_MAX_TOKENS", "180"))
    dtype = _hf_dtype()
    model_path = _resolve_model_path()
    processor, tokenizer = _build_processor(model_path)

    # Same image + crop as test_dots_ocr_vision: top 57.5% of the page.
    image = Image.open(requests.get(image_link, stream=True).raw)
    original_width, original_height = image.size
    new_height = int(original_height * 0.575)
    image = image.crop((0, 0, original_width, new_height))
    print(f"Cropped image from {original_width}x{original_height} to {original_width}x{new_height}")

    # Same prompt as test_dots_ocr_vision.
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
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(dtype)
    image_grid_thw = inputs["image_grid_thw"]
    prompt_len = int(input_ids.shape[1])

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).eval()
    hf_model.config.image_token_id = IMAGE_TOKEN_ID

    print(
        f"[dots_ocr HF ref] dtype={dtype} grid={image_grid_thw.tolist()} "
        f"prompt_len={prompt_len} max_new_tokens={max_new_tokens} (CPU greedy -- this is slow)"
    )
    gen = hf_model.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    new_ids = gen[0, prompt_len:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print(f"HF REFERENCE OUTPUT ({len(new_ids)} tokens):\n{text}")
    print("=" * 70)
    print(f"first 16 generated token ids: {new_ids[:16]}")

    assert len(text.strip()) > 0, "HF reference output should not be empty"
