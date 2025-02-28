# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""Record module inputs/outputs for testing functional implementations."""

import torch
from models.demos.qwen25_vl.model import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLMLP
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.demos.qwen25_vl.instrument import instrument

# Instrument the vision MLP
Qwen2_5_VLMLP.forward = instrument("Qwen2_5_VLMLP")(Qwen2_5_VLMLP.forward)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# Modify model config to only use a single layer
# This will make the model only run the first layer during forward pass
model.config.num_hidden_layers = 1
model.config.num_vision_layers = 1  # If the vision encoder has separate layers

# Default processor
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

# Run inference to trigger recording
outputs = model.generate(**inputs, max_new_tokens=1)

print("Done recording module inputs/outputs.")
