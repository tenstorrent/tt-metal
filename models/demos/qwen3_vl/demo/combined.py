# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Test module for end-to-end Qwen2.5-VL model inference with optional TT vision model."""

import os

import pytest
import torch
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

import ttnn
from models.demos.qwen3_vl.tt.model import DropInVisionTransformer
from models.demos.qwen3_vl.tt.model_config import VisionModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_input, text_input, expected_output",
    [
        (
            "file://models/sample_data/house_in_field_1080p.jpg",
            "Describe this image.",
            """This image is a vibrant, highly saturated landscape photograph that captures a serene and idyllic rural scene. The composition is symmetrical and balanced, with a striking reflection in the foreground.
            **Key Elements:**

            - **The House:** A charming, two-story brick house with a warm, terracotta-colored tiled roof sits on a lush green hill. The house features multiple white-framed windows, some arched, and a small balcony or overhang on the right side. Two chimneys are visible on the roof. The architecture suggests a European countryside style.

            - **The Landscape:** The house is nestled on a gently sloping,""",
        ),
        (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "Describe this image.",
            """This is a heartwarming and serene photograph capturing a joyful moment between a woman and her dog on a beach at sunset.

            **Main Subjects:**
            - **The Woman:** She is sitting cross-legged on the sand, facing her dog. She has long, dark hair and is wearing a blue and white plaid shirt with rolled-up sleeves and dark pants. She is smiling warmly, looking at her dog, and holding a small treat in her hand. Her left wrist has a white watch or band.
            - **The Dog:** A golden-colored Labrador Retriever, likely a yellow lab, is sitting upright on the sand, facing the""",
        ),
        (
            "https://gist.github.com/gwangTT/ae3ac698de56020bc459018c7c2bff08/raw/a91b2df96c61234d83a7f61c4495bfc826786c74/paper.png",
            "Transcribe the text in the image.",
            """Fractal Generative Models

            4.4. Relation to Long-Sequence Modeling
            Most previous work on pixel-by-pixel generation formulates the problem as long-sequence modeling and leverages methods from language modeling to address it (Child et al., 2019; Roy et al., 2021; Ren et al., 2021; Hawthorne et al., 2022; Yu et al., 2023). However, the intrinsic structures of many data types, including but not limited to images, are beyond one-dimensional sequences. Different from these methods, we treat such""",
        ),
    ],
    ids=["72dpi", "240dpi", "300dpi"],
)
@pytest.mark.parametrize(
    "use_tt_vision",
    [False],
    ids=["hf_vision"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_qwen_vl_end_to_end(
    mesh_device,
    reset_seeds,
    ensure_gc,
    ensure_nltk,
    image_input,
    text_input,
    expected_output,
    use_tt_vision,
    is_ci_env,
    request,
):
    test_id = request.node.callspec.id
    if is_ci_env and not "240dpi" in test_id:
        pytest.skip("CI only runs 240dpi image test for compromise of coverage and time limit")

    """Test end-to-end Qwen3-VL model with options to replace vision component."""
    batch_size = 1  # use batch size 1 for now to run the test in reasonable amount of time in CI
    max_new_tokens = 200

    # Load model and processor
    model_name = os.environ.get("HF_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
    assert "Qwen3-VL-32B".lower() in model_name.lower(), "This test uses only Qwen3-VL-32B for fast accuracy checking"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    # Sample image input to trigger vision model
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_input,
                    },
                    {"type": "text", "text": text_input},
                ],
            }
        ]
    ] * batch_size

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Optionally use TT vision model
    if use_tt_vision:
        # Create the TorchVisionTransformer wrapper using the original vision model as reference
        model_args = VisionModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_new_tokens)
        model.model.visual = DropInVisionTransformer(model.visual, model_args, debug=True)  # show PCC

    # Run inference
    logger.info("Running model generation...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Verify output
    # Token-level BLEU score (standard for text generation)
    for i, output_text in enumerate(output_texts):
        logger.info(f"Expected output: {expected_output}")
        logger.info(f"Generated output {i}: {output_text}")

        reference = [word_tokenize(expected_output.lower())]
        candidate = word_tokenize(output_text.lower())
        bleu_score = sentence_bleu(reference, candidate)
        logger.info(f"BLEU score of output {i}: {bleu_score:.3f}")
        assert bleu_score > 0.5

    logger.info(f"Test passed with {'TorchVisionTransformer' if use_tt_vision else 'original vision model'}")
