# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""Test module for end-to-end Qwen2.5-VL model inference with optional TT vision model."""

import pytest
import torch
import os
from loguru import logger
import ttnn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from models.utility_functions import skip_for_grayskull
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_tt_vision",
    [False, True],
    ids=["hf_vision", "tt_vision"],
)
def test_qwen_vl_end_to_end(
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    use_tt_vision,
):
    """Test end-to-end Qwen2.5-VL model with options to replace vision component."""
    mesh_device.enable_async(True)
    max_new_tokens = 128

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Sample image input to trigger vision model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Optionally use TT vision model
    if use_tt_vision:
        # Create the TorchVisionTransformer wrapper using the original vision model as reference
        model_args = VisionModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_new_tokens)
        model.visual = DropInVisionTransformer(model.visual, model_args, debug=True)  # show PCC

    # Run inference
    logger.info("Running model generation...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Verify output
    expected_output = "The image depicts a serene beach scene with a person and a dog. The person is sitting on the sandy beach, facing the ocean. They are wearing a plaid shirt and black pants, and they have long hair. The dog, which appears to be a Labrador Retriever, is sitting on the sand and is interacting with the person by placing its paw on their hand. The dog is wearing a harness with a colorful collar. The background shows the ocean with gentle waves, and the sky is clear with a soft light, suggesting it might be early morning or late afternoon. The overall atmosphere of the image is peaceful and joyful."
    # TT output:      'The image depicts a person sitting on a rocky beach, likely at sunset or sunrise, given the warm, golden light illuminating the scene. The person is wearing a plaid shirt and appears to be holding a device, possibly a phone or camera, with a green screen visible. The water in the background is calm, reflecting the sunlight, and the overall atmosphere is serene and peaceful.'

    logger.info(f"Generated output: {output_text}")
    assert len(output_text) > 0, "No output generated from the model"
    logger.info(f"Expected output : {expected_output}")

    logger.info(f"Test passed with {'TorchVisionTransformer' if use_tt_vision else 'original vision model'}")
