#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Run Qwen3-VL-2B-Instruct on CPU (HuggingFace reference) and print output.
Usage:
  HF_HOME=/home/yito/.cache/huggingface python models/demos/qwen3_vl/demo/cpu_reference_output.py
"""

import os
import json
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
PROMPT_FILE = "models/demos/qwen3_vl/demo/sample_prompts/demo.json"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.0  # argmax (greedy)


def load_prompts(path):
    with open(path) as f:
        return json.load(f)


def main():
    print(f"Loading model: {HF_MODEL}")
    print(f"Prompt file:   {PROMPT_FILE}")
    print()

    processor = AutoProcessor.from_pretrained(HF_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    prompts = load_prompts(PROMPT_FILE)

    for idx, conversation in enumerate(prompts):
        print(f"=== Prompt {idx} ===")
        for msg in conversation:
            if msg["role"] == "user":
                for part in msg["content"]:
                    if part["type"] == "text":
                        print(f"  [text] {part['text']}")
                    elif part["type"] == "image":
                        print(f"  [image] {part['image']}")
        print()

        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        print(f"  Input token count: {inputs.input_ids.shape[-1]}")
        print()

        with torch.no_grad():
            if TEMPERATURE == 0.0:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                )

        # Decode only the newly generated tokens
        new_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
        output_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]

        print(f"=== CPU Reference Output (Prompt {idx}) ===")
        print(output_text)
        print()


if __name__ == "__main__":
    main()
