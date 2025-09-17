# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

import ttnn
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs


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
            "The image depicts a serene and picturesque scene of a small, quaint house situated on a lush green lawn. The house has a traditional design with a red brick exterior and a sloping roof covered in dark tiles. It features multiple windows, including a prominent bay window on the front facade, which adds to its charming appearance. The lawn is expansive and well-maintained, extending towards a body of water that reflects the house and the surrounding landscape. The water appears calm, creating a perfect mirror image of the house and the sky above. The reflection is clear and detailed, enhancing the overall tranquility of the scene. In the background, there",
        ),
        (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "Describe this image.",
            "The image depicts a serene beach scene with a person and a dog. The person is sitting on the sandy beach, facing the ocean. They are wearing a plaid shirt and black pants, and they have long hair. The dog, which appears to be a Labrador Retriever, is sitting on the sand and is interacting with the person by placing its paw on their hand. The dog is wearing a harness with a colorful collar. The background shows the ocean with gentle waves, and the sky is clear with a soft light, suggesting it might be early morning or late afternoon. The overall atmosphere of the image is peaceful and joyful.",
        ),
        (
            "https://gist.github.com/gwangTT/ae3ac698de56020bc459018c7c2bff08/raw/a91b2df96c61234d83a7f61c4495bfc826786c74/paper.png",
            "Transcribe the text in the image.",
            "Fractal Generative Models 4.4. Relation to Long-Sequence Modeling Most previous work on pixel-by-pixel generation formulates the problem as long-sequence modeling and leverages methods from language modeling to address it (Child et al., 2019; Roy et al., 2021; Ren et al., 2021; Hawthorne et al., 2022; Yu et al., 2023). However, the intrinsic structures of many data types, including but not limited to images, are beyond one-dimensional sequences. Different from these methods, we treat such",
        ),
    ],
    ids=["72dpi", "240dpi", "300dpi"],
)
@pytest.mark.parametrize(
    "use_tt_vision",
    [False, True],
    ids=["hf_vision", "tt_vision"],
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

    """Test end-to-end Qwen2.5-VL model with options to replace vision component."""
    batch_size = 1  # use batch size 1 for now to run the test in reasonable amount of time in CI
    max_new_tokens = 128

    # Load model and processor
    model_name = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    assert "Qwen2.5-VL-3B".lower() in model_name.lower(), "This test uses only Qwen2.5-VL-3B for fast accuracy checking"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
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
