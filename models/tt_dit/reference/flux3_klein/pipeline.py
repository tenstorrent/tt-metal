# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/pipeline.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

"""Prompt + reference images -> image. """

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from .autoencoder import DOWNSAMPLING_RATIO, AutoEncoder, latent_ids, latent_shape
from .checkpoint import is_flow_map, load_checkpoint
from .dit import IN_CHANNELS, MODES, DiT, Packed
from .sampling import denoise, denoise_flow_map, timesteps
from .text_encoder import TextEncoder

MODELS = {  # weights entry (the Hub repository, downloaded there), default steps, CFG scale (None: a flow map, sampled without CFG)
    "base": {"dit": "flux-3-image-klein-base", "steps": 50, "guidance": 3.5},
    "distilled": {"dit": "flux-3-image-klein", "steps": 16, "guidance": None},
}
IMAGE_ROPE_SCALE = 0.5
MATCH_MAX_PIXELS = 4 * 1024 * 1024  # first reference, padded canvas included
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024  # each reference of a multi-reference edit


def dit_path(weights_dir: str | Path, model: str) -> Path:
    """The bf16 DiT of ``model``: its repository root under ``weights_dir``. Upstream reads the root from the
    WEIGHTS_DIR environment variable and also serves the fp8 / nvfp4 ``variants/``; this copy loads bf16 only."""
    return Path(weights_dir) / MODELS[model]["dit"]


class Pipeline:
    """One or more DiTs sharing the text encoder and autoencoder. ``offload`` keeps the DiTs on the CPU and
    moves the one in use to the GPU."""

    def __init__(
        self,
        dits: dict[str, str],
        text_encoder: str,
        autoencoder: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False,
    ):
        self.device, self.offload = device, offload
        self.dits = {}
        for name, path in dits.items():
            flow_map = is_flow_map(path)
            if flow_map != (MODELS[name]["guidance"] is None):
                raise SystemExit(f"{path}: {name} expects {'a CFG' if flow_map else 'a flow-map'} checkpoint")
            with torch.device("meta"):
                dit = DiT(flow_map)
            self.dits[name] = load_checkpoint(dit, path, "cpu" if offload else device, torch.bfloat16)
        self.text_encoder = TextEncoder(text_encoder, device)
        self.autoencoder = load_checkpoint(AutoEncoder(), autoencoder, device, torch.float32)

    def dit(self, model: str) -> DiT:
        if self.offload:
            for name, dit in self.dits.items():
                dit.to(self.device if name == model else "cpu")
        return self.dits[model]

    def generate(
        self,
        model: str,
        prompt: str,
        references: list[Image.Image],
        width: int,
        height: int,
        steps: int,
        guidance: float | None,
        seed: int,
        alpha: float = 6.93,
    ) -> Image.Image:
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references")
        dit = self.dit(model)
        cond = self.encode_references(references)
        h, w = latent_shape(height, width)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(seed))
        noise = noise.to(torch.bfloat16).float().to(self.device)  # bf16 values, fp32 state across the Euler steps
        ids = latent_ids(h, w, self.device)
        image = (noise, scaled_ids(ids))

        schedule = timesteps(steps, alpha)
        if dit.flow_map:
            packed = pack([[self.text_encoder.encode(prompt), cond, image]])
            tokens = denoise_flow_map(dit, packed, schedule)
        else:
            text = [self.text_encoder.encode(""), self.text_encoder.encode(prompt)]
            packed = pack([[text[0], cond, image], [text[1], cond, image]])
            tokens = denoise(dit, packed, schedule, guidance)
        return to_pil(self.autoencoder.decode(tokens, ids))

    def encode_references(self, references: list[Image.Image]) -> tuple[Tensor, Tensor] | None:
        if not references:
            return None
        tokens, ids = [], []
        for index, reference in enumerate(references):
            t, i = self.autoencoder.encode(to_tensor(reference).to(self.device))
            i = scaled_ids(i)
            i[:, 3] = -(1 + index)  # the l axis tells references apart
            tokens.append(t.to(torch.bfloat16))
            ids.append(i)
        return torch.cat(tokens), torch.cat(ids)


def pack(samples: list[list[tuple[Tensor, Tensor] | None]]) -> Packed:
    """samples[s] = [text, image_cond or None, image], each a (tokens, ids) pair."""
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    device = samples[0][0][0].device
    cat = lambda parts, width, dtype: torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)
    data = {m: cat(data[m], IN_CHANNELS[m], torch.bfloat16) for m in MODES}
    ids = {m: cat(ids[m], 4, torch.float32) for m in MODES}
    return Packed(
        data=data,
        ids=ids,
        timesteps={m: torch.zeros(data[m].shape[0], device=device, dtype=torch.bfloat16) for m in MODES},
        mode_indices=mode_indices,
    )


def scaled_ids(ids: Tensor) -> Tensor:
    ids = ids.float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a single reference: the canvas (the fitted size rounded up to 16) at most 4 MiB,
    sides 64-4096."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """A single reference: RGB, fitted by reference_canvas and reflect-padded to its canvas, centred. Returns the
    image and, for a padded one, the (left, top, right, bottom) box of the fitted image inside it."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = (
        "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    )  # reflect needs a pad shorter than the side
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), (
        left,
        top,
        left + image.width,
        top + image.height,
    )


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Each reference of a multi-reference edit: RGB, fitted under 1 MP (sides floored), centre-cropped to the
    16-px grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def default_size(references: list[Image.Image]) -> tuple[int, int]:
    """1024x1024 for text-to-image; the first (prepared) reference's own size otherwise."""
    return references[0].size if references else (1024, 1024)


def to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def to_pil(image: Tensor) -> Image.Image:
    array = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)
