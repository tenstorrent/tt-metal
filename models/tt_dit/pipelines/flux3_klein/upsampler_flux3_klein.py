# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein prompt upsampler: BFL's Qwen3.5-9B fine-tune (black-forest-labs/flux-3-image-klein-pup) on one
chip through models/demos/blackhole/qwen36, following BFL's flux3/pup.py: greedy decoding with a presence penalty
for text-to-image, the reference image JPEG-compressed to at most 1 MP."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

import ttnn

REPO_ID = "black-forest-labs/flux-3-image-klein-pup"
MAX_NEW_TOKENS = 3584
MAX_SEQ_LEN = 8192
BLOCK_SIZE = 64
PREFILL_CHUNK = 2048
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)
REFERENCE_MULTIPLE = 32
REFERENCE_MAX_PIXELS = 1024 * 1024
JPEG_QUALITY = 92
# What the upsampler needs from the mesh it runs on: qwen36's GDN conv uses L1_SMALL, and its prefill and decode run
# from captured traces on two command queues.
DEVICE_PARAMS = {"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 256 * 1024 * 1024}


def resolve_weights_dir() -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(REPO_ID, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(REPO_ID))


def reference_size(width: int, height: int) -> tuple[int, int]:
    def floor(v: float) -> int:
        return max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)

    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > REFERENCE_MAX_PIXELS:
        beta = math.sqrt(width * height / REFERENCE_MAX_PIXELS)
        w, h = floor(width / beta), floor(height / beta)
    return w, h


def prepare_reference(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def presence_greedy(logits: torch.Tensor, generated: list[int], penalty: float) -> int:
    logits = logits.clone()
    if penalty > 0 and generated:
        logits[torch.tensor(sorted(set(generated)))] -= penalty
    return int(torch.argmax(logits).item())


class Flux3KleinUpsampler:
    """Rewrites a prompt (and optional reference image) into the detailed caption the DiT was trained on."""

    def __init__(self, mesh_device: ttnn.MeshDevice, snapshot: str | os.PathLike | None = None) -> None:
        from transformers import AutoProcessor, AutoTokenizer

        from models.demos.blackhole.qwen36.tt.common import create_tt_model
        from models.ttt_compat.tt.generator import Generator

        snapshot = Path(snapshot) if snapshot is not None else resolve_weights_dir()
        self._device = mesh_device
        self.args, self.model, _ = create_tt_model(
            mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN, hf_model=str(snapshot), enable_mtp=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        self.system_prompts = json.loads((snapshot / "system_prompts.json").read_text())
        self.stop_ids = {int(self.tokenizer.convert_tokens_to_ids(t)) for t in STOP_TOKENS}

        num_blocks = MAX_SEQ_LEN // BLOCK_SIZE
        self.page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        self.model.allocate_kv_caches(
            [num_blocks, int(self.args.n_kv_heads), BLOCK_SIZE, int(self.args.head_dim)], ttnn.bfloat16, batch_size=1
        )
        self.generator = Generator([self.model], [self.args], mesh_device)

        # compile the vision tower and capture the prefill trace before any request: a program compiled while a
        # trace is parked can overwrite its buffers
        self.model.init_vision_model()
        warm = self._inputs("", Image.new("RGB", (1024, 1024)))
        ttnn.deallocate(self.model.get_image_features(warm[1], warm[2]))
        self.model.capture_prefill_trace_chunked(
            mesh_device, self.page_table, chunk_size=PREFILL_CHUNK, warmup_masked_buckets=True
        )

    def _inputs(self, prompt: str, image: Image.Image | None):
        task = "i2i" if image is not None else "t2i"
        if image is None:
            messages = [
                {"role": "system", "content": self.system_prompts[task]},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)], dtype=torch.int64)
            return input_ids, None, None, task
        content = [{"type": "image"}, {"type": "text", "text": I2I_SCAFFOLD.format(instruction=prompt)}]
        messages = [{"role": "system", "content": self.system_prompts[task]}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = self.processor(text=text, images=[prepare_reference(image)], return_tensors="pt")
        return (
            encoded["input_ids"].to(torch.int64),
            encoded["pixel_values"],
            encoded["image_grid_thw"].to(torch.int64),
            task,
        )

    def __call__(self, prompt: str, image: Image.Image | None = None) -> str:
        from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch, prime_decode_trace

        input_ids, pixel_values, grid, task = self._inputs(prompt, image)
        penalty = PRESENCE_PENALTY[task]
        vision_tokens = self.model.get_image_features(pixel_values, grid) if pixel_values is not None else None
        position = int(input_ids.shape[1])
        logits = prefill_dispatch(
            self.model, input_ids, self.page_table, torch.tensor([position]), True, vision_tokens=vision_tokens
        )
        token = presence_greedy(ttnn.to_torch(logits).reshape(-1).float(), [], penalty)
        if vision_tokens is not None:
            ttnn.deallocate(vision_tokens)

        generated = []
        max_new_tokens = min(MAX_NEW_TOKENS, MAX_SEQ_LEN - position)
        if token not in self.stop_ids:
            prime_decode_trace(
                self.generator, self.model, torch.tensor([[token]]), torch.tensor([position]), self.page_table
            )
        while token not in self.stop_ids and len(generated) < max_new_tokens:
            generated.append(token)
            out = self.generator.decode_forward(
                torch.tensor([[token]]),
                torch.tensor([position]),
                page_table=self.page_table,
                kv_cache=None,
                enable_trace=True,
                read_from_device=True,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )
            logits = out[0] if isinstance(out, tuple) else out
            token = presence_greedy(logits.reshape(-1).float(), generated, penalty)
            position += 1
        caption = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return caption or prompt

    def close(self) -> None:
        self.generator.release_persistent_capture()
        self.model.free_kv_caches()
