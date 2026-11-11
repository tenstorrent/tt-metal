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
SHARED_MESH_PARAMS = {"num_command_queues": 2, "trace_region_size": 256 * 1024 * 1024}
DEVICE_PARAMS = {"l1_small_size": 24576, **SHARED_MESH_PARAMS}

SHARED_DRAM_BOTTOM = 1938 << 20
SHARED_DRAM_TOP = 256 << 20
SHARED_L1_TOP = 34 << 10
DIT_L1_MARGIN = 2 << 10


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
    if w * h > REFERENCE_MAX_PIXELS:
        if w > h:
            w = REFERENCE_MAX_PIXELS // h // REFERENCE_MULTIPLE * REFERENCE_MULTIPLE
        else:
            h = REFERENCE_MAX_PIXELS // w // REFERENCE_MULTIPLE * REFERENCE_MULTIPLE
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
        if input_ids.shape[1] >= MAX_SEQ_LEN:
            raise ValueError(f"the prompt is {input_ids.shape[1]} tokens, the upsampler's limit is {MAX_SEQ_LEN - 1}")
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
        self.generator = self.model = None


def _hold(device: ttnn.MeshDevice, buffer_type: ttnn.BufferType, banks: int, page: int, pages: int) -> ttnn.Tensor:
    memory_config = ttnn.DRAM_MEMORY_CONFIG if buffer_type == ttnn.BufferType.DRAM else ttnn.L1_MEMORY_CONFIG
    return ttnn.empty([1, 1, banks * pages, page // 2], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config)


def _allocated(device: ttnn.MeshDevice, buffer_type: ttnn.BufferType) -> tuple[int, list[tuple[int, int, bool]]]:
    view = ttnn.get_memory_view(device, buffer_type)
    return view.num_banks, [(int(b["address"]), int(b["size"]), b["allocated"] == "yes") for b in view.block_table]


def reserve_upsampler_ranges(mesh_device: ttnn.MeshDevice) -> list[ttnn.Tensor]:
    """Hold, on the DiT's mesh and before anything else allocates on it, the ranges the upsampler's submesh will
    allocate from on its chip. The submesh's allocator does not see the mesh's, so both would start at the same
    addresses."""
    page = 8192
    banks, _ = _allocated(mesh_device, ttnn.BufferType.DRAM)
    holds = []
    pages = SHARED_DRAM_BOTTOM // page
    while pages:
        n = min(pages, 16384)
        holds.append(_hold(mesh_device, ttnn.BufferType.DRAM, banks, page, n))
        pages -= n

    _, blocks = _allocated(mesh_device, ttnn.BufferType.DRAM)
    _, size = max(((a, s) for a, s, used in blocks if not used and s > 2 * SHARED_DRAM_TOP), key=lambda b: b[0])
    fill_pages = (size - SHARED_DRAM_TOP) // page
    top = size - fill_pages * page
    fill = []
    while fill_pages:
        n = min(fill_pages, 16384)
        fill.append(_hold(mesh_device, ttnn.BufferType.DRAM, banks, page, n))
        fill_pages -= n
    holds.append(_hold(mesh_device, ttnn.BufferType.DRAM, banks, top, 1))
    for t in fill:
        ttnn.deallocate(t)

    l1_banks, _ = _allocated(mesh_device, ttnn.BufferType.L1)
    holds.append(_hold(mesh_device, ttnn.BufferType.L1, l1_banks, SHARED_L1_TOP, 1))
    return holds


class Flux3KleinSharedUpsampler:
    """The upsampler on a (1, 1) submesh of the DiT's mesh, its ranges held by reserve_upsampler_ranges.

    The DiT's mesh owns the command queues outside a call. A switch quiesces the mesh that ran and reads from the
    one about to run, which drops that mesh's stale prefetcher-cache records. The DiT's resident L1 (its collective
    semaphores) is held on the submesh too, so the upsampler's circular buffers stay clear of it.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, snapshot: str | os.PathLike | None = None) -> None:
        self._mesh = mesh_device
        self._mesh_probe = ttnn.from_torch(torch.zeros(32, 32), layout=ttnn.TILE_LAYOUT, device=mesh_device)
        mesh_device.quiesce_devices()
        self._submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1))
        self._upsampler = Flux3KleinUpsampler(self._submesh, snapshot)
        self._submesh_probe = ttnn.from_torch(torch.zeros(32, 32), layout=ttnn.TILE_LAYOUT, device=self._submesh)
        self._dit_l1 = None
        self._dit_l1_lowest = None
        self._to_mesh()

    def _hold_dit_l1(self) -> None:
        _, dit = _allocated(self._mesh, ttnn.BufferType.L1)
        lowest_dit = min(a for a, _, used in dit if used)
        if self._dit_l1_lowest is not None and lowest_dit >= self._dit_l1_lowest:
            return
        if self._dit_l1 is not None:
            ttnn.deallocate(self._dit_l1)
        banks, own = _allocated(self._submesh, ttnn.BufferType.L1)
        size = (min(a for a, _, used in own if used) - lowest_dit + DIT_L1_MARGIN + 31) // 32 * 32
        self._dit_l1 = _hold(self._submesh, ttnn.BufferType.L1, banks, size, 1)
        self._dit_l1_lowest = lowest_dit

    def _to_mesh(self) -> None:
        self._submesh.quiesce_devices()
        ttnn.from_device(self._mesh_probe)

    def _to_submesh(self) -> None:
        self._mesh.quiesce_devices()
        self._hold_dit_l1()
        ttnn.from_device(self._submesh_probe)

    def __call__(self, prompt: str, image: Image.Image | None = None) -> str:
        self._to_submesh()
        try:
            return self._upsampler(prompt, image)
        finally:
            self._to_mesh()

    def close(self) -> None:
        self._to_submesh()
        self._upsampler.close()
        for t in (self._dit_l1, self._submesh_probe):
            if t is not None:
                ttnn.deallocate(t)
        self._to_mesh()
        ttnn.deallocate(self._mesh_probe)
