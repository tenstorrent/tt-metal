# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Same flow as ``demo.py``, but the HF model's ``vision_tower`` is backed by
``DotsVisionTransformerTT`` (ttnn) instead of ``DotsVisionTransformer`` (torch).

Run from the tt-metal repo root (or ensure ``models`` is on ``PYTHONPATH``), with
Tenstorrent devices visible. Example::

    python models/demos/dots_ocr/reference/demo_tt.py
"""

from __future__ import annotations

import atexit
from pathlib import Path

import torch
import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

import ttnn
from models.demos.dots_ocr.tt.dots_visionTT import DotsVisionTransformerTT

_ref_dir = Path(__file__).resolve().parent
model_path = str(_ref_dir / "dots_ocr")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print("model: ", model)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("finished loading model and processor")
print("processor: ", processor)


def _open_mesh_for_demo():
    """Single-device mesh."""
    mesh_shape = ttnn.MeshShape(1, 1)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    def _cleanup(md=mesh_device):
        ttnn.close_mesh_device(md)

    atexit.register(_cleanup)
    return mesh_device


class _VisionTowerTTWrapper(nn.Module):
    """``forward`` contract matches ``DotsVisionTransformer`` (``pixel_values``, ``grid_thw``)."""

    def __init__(self, tt_tower: DotsVisionTransformerTT):
        super().__init__()
        self._tt = tt_tower

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16: bool = True):
        if bf16 and hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.bfloat16()
        out = self._tt(hidden_states, grid_thw, return_host_torch=True)
        return out.to(dtype=hidden_states.dtype, device=hidden_states.device)


_mesh_device = _open_mesh_for_demo()
_tt_vision = DotsVisionTransformerTT(
    vision_config=model.config.vision_config,
    mesh_device=_mesh_device,
    state_dict=model.state_dict(),
    state_dict_prefix="vision_tower.",
    weight_cache_path=None,
)
model.vision_tower = _VisionTowerTTWrapper(_tt_vision)
model.eval()
print("vision_tower: TTNN-backed DotsVisionTransformerTT")


image_path = str(_ref_dir / "test12.png")
prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("text: ", text)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=2000)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
