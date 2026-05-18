# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.integrations.finegrained_fp8 import Fp8Dequantize


_ORIGINAL_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    # Scalar FP8 scales (0-D): some HF paths expect a 2-D scale grid and crash without this.
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat

model_name = "mistralai/Devstral-Small-2-24B-Instruct-2512"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

_device = next(model.parameters()).device

# Load image
image = Image.open("models/experimental/devstarl2_small/reference/testimage1.jpeg").convert("RGB")

# Build prompt with the correct image placeholder(s). Mistral3 uses token id `image_token_id`
# (decoded as "[IMG]"), NOT the string "<image>". The processor expands one logical "[IMG]"
# in the chat template into one token per vision patch when `images=` is passed.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

# Move tensor inputs to the same device as the model (works with device_map="auto")
inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Generate output (avoid passing both max_length from config and max_new_tokens)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# Decode output
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print(output_text)
