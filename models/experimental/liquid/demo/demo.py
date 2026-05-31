# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import requests
import torch
from PIL import Image
from transformers import AutoProcessor

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.liquid.tt.liquid_vl import LiquidVL


def main():
    model_id = "LiquidAI/LFM2.5-VL-1.6B"
    weights_path = os.environ.get("LIQUID_WEIGHTS", os.path.expanduser("~/liquid_weights/"))

    device = ttnn.open_device(device_id=0)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=weights_path)

    params = preprocess_model_parameters(
        model_name=model_id,
        device=device,
        model=None,
        convert_to_ttnn=lambda n, p: isinstance(p, (torch.nn.Linear, torch.nn.Conv2d)),
    )

    model = LiquidVL(device, params, model_args=None)

    url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )

    pixel_values = ttnn.from_torch(inputs["pixel_values"], device=device)
    input_ids = ttnn.from_torch(inputs["input_ids"], device=device)

    image_features = model.encode_image(pixel_values)
    outputs = model.generate(input_ids, image_features=image_features)

    print(outputs)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
