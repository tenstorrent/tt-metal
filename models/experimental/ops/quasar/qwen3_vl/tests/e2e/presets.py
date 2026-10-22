# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Input size presets: tiny for fast iteration, demo for the captured graph sizes."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

_DEMO_JPEG = Path(__file__).resolve().parents[7] / "models" / "sample_data" / "demo.jpeg"
PROMPT = "Describe this image."


def _demo_image():
    return Image.open(_DEMO_JPEG).convert("RGB")


def _tiny_image():
    # 256x256 = the processor's min_pixels, so it is not resized: 256 patches -> 64 image tokens.
    return _demo_image().resize((256, 256), Image.BICUBIC)


@dataclass(frozen=True)
class Preset:
    name: str
    max_seq_len: int
    kv_blocks: int
    make_image: Callable
    prompt: str = PROMPT
    block_size: int = 32


PRESETS = {
    # 78 tokens pad to a 128 prefill; 8 blocks x 32 = 256 tokens of KV for prefill plus decode.
    "tiny": Preset("tiny", max_seq_len=256, kv_blocks=8, make_image=_tiny_image),
    # Captured demo sizes: 2766 tokens pad to 4096; 1024 blocks as in the graph capture.
    "demo": Preset("demo", max_seq_len=4096, kv_blocks=1024, make_image=_demo_image),
}


def build_inputs(preset: Preset, processor):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": preset.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[preset.make_image()], padding=True, return_tensors="pt")
