# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-side request handling for MiniMax-H3-Image (text-to-image and reference editing).

Everything here is torch/numpy/PIL only (no ttnn), so it unit-tests on any host. Each function reproduces one step of
the release's SGLang image pipeline exactly; the comments name the step, not the file, because the reference is not
public. The device pipeline (`pipeline_minimax_h3_image.py`) composes these.

Request flow (t2i: no references; edit: >= 1 reference image, numbered `Picture 1..n` in request order):
  canvas = resolve_image_canvas(w, h, min_area)                       # generated size; resized back afterwards
  refs: qwen copy = bound_short_edge(img, 1024, 1024); vae copy = bound_short_edge(img, 1024, 2048)
  noise (one CPU generator, fixed order): target eps, then per ref anchor eps (even grid), then per ref posterior eps
  ref rows  = patchify(normalize(fp16(posterior sample)) even-cropped);  anchors = 0.999*rows + (1-0.999)*eps_anchor
  packed sequence: [text L | ref 1 .. ref n | target | pad];  timestep slots [t_k, 0.999] (refs use slot 1)
  6 steps on the pdd_config sigma grid: DiT(t = 1 - sigma_k, head k) -> x += (sigma_k - sigma_{k+1}) * v  (fp32)
  decode: z*std+mean -> VAE -> ImageNet de-norm -> clamp(0,1) -> uint8 by truncation -> LANCZOS back to request size
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

CANVAS_GRID = 32
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MIN_GENERATION_AREA_RANGE = (1024, 16777216)
QWEN_REFERENCE_SHORT_EDGE = (1024, 1024)
VAE_REFERENCE_SHORT_EDGE = (1024, 2048)
# The per-request reference limit (the served API's rule): the references' total EFFECTIVE area -- each image counted at
# its VAE-copy size, i.e. after its short side is scaled into VAE_REFERENCE_SHORT_EDGE -- may not exceed this many
# pixels. Every cap the image pipeline compiles for derives from it (see `reference_caps`); over it a request is
# rejected, never shrunk (the release pipeline does not shrink references).
REFERENCE_AREA_LIMIT = 48_000_000
MAX_REFERENCES = 16
MAX_REFERENCE_ASPECT = 4.0
REFERENCE_SIDE_RANGE = (128, 8192)
VAE_RATIO = 16
LATENT_CHANNELS = 24
PATCH = (1, 2, 2)
REFERENCE_TIMESTEP = 0.999
MAX_PROMPT_TOKENS = 4096
ROPE_SPATIAL_SCALE = 32.0
TEXT_TAG, IMAGE_TAG = 1, 0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------------------------- canvas / references


def resolve_image_canvas(width: int, height: int, min_generation_area: int | None = DEFAULT_MIN_GENERATION_AREA):
    """`(canvas_w, canvas_h)`: scale up (keeping aspect) to at least `min_generation_area`, snap to multiples of 32.

    Python float64 math and `round()` (ties to even), as the reference does. 1024x1024 -> 2048x2048.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {width}x{height}")
    if min_generation_area is None:
        if width % CANVAS_GRID or height % CANVAS_GRID or min(width, height) < CANVAS_GRID:
            raise ValueError(f"without a canvas the size must be >= 32 and a multiple of 32, got {width}x{height}")
        return width, height
    lo, hi = MIN_GENERATION_AREA_RANGE
    if not isinstance(min_generation_area, int) or not lo <= min_generation_area <= hi:
        raise ValueError(
            f"min_generation_area must be an integer pixel area within [{lo}, {hi}], got {min_generation_area}"
        )
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))

    def nearest(value: float) -> int:
        return max(CANVAS_GRID, int(round(value / CANVAS_GRID)) * CANVAS_GRID)

    return nearest(width * scale), nearest(height * scale)


def load_reference_image(image: Image.Image) -> Image.Image:
    """Upright RGB, original pixels. Animated images are rejected."""
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("animated reference images are not supported")
    out = ImageOps.exif_transpose(image).convert("RGB")
    if min(out.size) < 32:
        raise ValueError(f"reference image short edge must be >= 32, got {out.size}")
    return out


def bounded_size(size: tuple[int, int], lo: int, hi: int) -> tuple[int, int]:
    """`size` when its short edge is in `[lo, hi]`, else scaled so the short edge is `lo`/`hi` (sides to 32 px)."""
    w, h = size
    short = min(w, h)
    if lo <= short <= hi:
        return w, h
    scale = (lo if short < lo else hi) / short
    return (
        max(CANVAS_GRID, int(round(w * scale / CANVAS_GRID)) * CANVAS_GRID),
        max(CANVAS_GRID, int(round(h * scale / CANVAS_GRID)) * CANVAS_GRID),
    )


def bound_short_edge(image: Image.Image, lo: int, hi: int) -> Image.Image:
    """Keep the original pixels when the short edge is in `[lo, hi]`, else LANCZOS it to `lo`/`hi` (sides to 32)."""
    size = bounded_size(image.size, lo, hi)
    if size == image.size:
        return image
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def effective_reference_area(sizes: list[tuple[int, int]]) -> int:
    """Total pixels of the references' VAE copies (`(width, height)` of each upright original): the API's measure."""
    return sum(w * h for w, h in (bounded_size(size, *VAE_REFERENCE_SHORT_EDGE) for size in sizes))


class ReferenceLimitError(ValueError):
    """A request's references are over the per-request limit (the served API answers 400)."""


def check_references(sizes: list[tuple[int, int]], area_limit: int = REFERENCE_AREA_LIMIT) -> int:
    """Validate a request's references by their upright `(width, height)`; returns the effective area.

    The rule: 1-16 references, each 128-8192 px a side with aspect ratio <= 4:1, whose total effective area (each
    counted after its short side is scaled into 1024-2048 px) is at most `area_limit` pixels.
    """
    if not 1 <= len(sizes) <= MAX_REFERENCES:
        raise ReferenceLimitError(f"between 1 and {MAX_REFERENCES} reference images are supported, got {len(sizes)}")
    lo, hi = REFERENCE_SIDE_RANGE
    for i, (w, h) in enumerate(sizes, 1):
        if not (lo <= min(w, h) and max(w, h) <= hi):
            raise ReferenceLimitError(f"reference image {i} is {w}x{h}: each side must be {lo}-{hi} px")
        if max(w, h) > MAX_REFERENCE_ASPECT * min(w, h):
            raise ReferenceLimitError(f"reference image {i} is {w}x{h}: aspect ratio over {MAX_REFERENCE_ASPECT:g}:1")
    area = effective_reference_area(sizes)
    if area > area_limit:
        raise ReferenceLimitError(
            f"reference images total {area / 1e6:.1f} MP effective area (each counted after scaling its short side "
            f"into {VAE_REFERENCE_SHORT_EDGE[0]}-{VAE_REFERENCE_SHORT_EDGE[1]} px); the limit is {area_limit / 1e6:g} MP"
        )
    return area


@dataclass(frozen=True)
class ReferenceCaps:
    """What the image pipeline must hold for references within `area_limit` (upper bounds, not estimates)."""

    rows: int  # VAE reference rows: one per 32x32 px of VAE copy, so <= area / 1024
    vision_tokens: int  # merged Qwen vision tokens across all copies
    vision_patches: int  # unmerged patches (2x2 per token)


def reference_caps(area_limit: int) -> ReferenceCaps:
    """A Qwen copy has short side 1024 and a VAE copy >= 1024 at the same aspect, so a reference's vision tokens
    (one per 32x32 px of Qwen copy) are at most its VAE rows, and at most 1024x4096 / 1024 = 4096 at 4:1 -- whatever
    the limit, 16 references carry <= 65536. Rounding each side to the 32-px grid adds at most half a token per row
    of the long side (<= 4096 / 32 / 2 = 64), per reference."""
    rows = area_limit // 1024
    per_ref = int(QWEN_REFERENCE_SHORT_EDGE[1] ** 2 * MAX_REFERENCE_ASPECT) // 1024
    vision = min(rows, MAX_REFERENCES * per_ref) + MAX_REFERENCES * 64
    return ReferenceCaps(rows=rows, vision_tokens=vision, vision_patches=4 * vision)


def vae_reference_pixels(image: Image.Image) -> torch.Tensor:
    """`(1, 3, 1, H16, W16)` uint8, top-left cropped to multiples of 16. ImageNet normalization happens on device."""
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    h16, w16 = (array.shape[0] // VAE_RATIO) * VAE_RATIO, (array.shape[1] // VAE_RATIO) * VAE_RATIO
    pixels = torch.from_numpy(array[:h16, :w16]).permute(2, 0, 1).contiguous()
    return pixels[None, :, None]


@dataclass
class ReferenceGeometry:
    """Latent geometry of one reference: the 16-px posterior grid and the even grid its rows live on."""

    posterior_h: int
    posterior_w: int

    @property
    def even_h(self) -> int:
        return self.posterior_h // 2 * 2

    @property
    def even_w(self) -> int:
        return self.posterior_w // 2 * 2

    @property
    def num_rows(self) -> int:
        return (self.even_h // 2) * (self.even_w // 2)


def reference_geometry(vae_image: Image.Image) -> ReferenceGeometry:
    w, h = vae_image.size
    return ReferenceGeometry(posterior_h=h // VAE_RATIO, posterior_w=w // VAE_RATIO)


def vae_reference_rows(vae_image: Image.Image) -> int:
    """Packed rows one VAE reference copy adds."""
    return reference_geometry(vae_image).num_rows


def qwen_vision_tokens(qwen_image: Image.Image) -> int:
    """Merged vision tokens one Qwen reference copy adds (16-px patches, 2x2 merge: one token per 32x32 px)."""
    w, h = qwen_image.size
    return max(1, round(w / 32)) * max(1, round(h / 32))


# ----------------------------------------------------------------------------------------------------- latents / RNG


def patchify(latents: torch.Tensor) -> torch.Tensor:
    """`[1, C, 1, H, W]` -> `[(H/2)*(W/2), C*4]`, row-major patches, channel-major features (c*4 + p*2 + q)."""
    b, c, t, h, w = latents.shape
    x = latents.reshape(b, c, t, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * 4).contiguous()


def unpatchify(rows: torch.Tensor, h: int, w: int, channels: int = LATENT_CHANNELS) -> torch.Tensor:
    """Inverse of :func:`patchify`: `[(h/2)*(w/2), C*4]` -> `[1, C, 1, h, w]`."""
    x = rows.reshape(1, 1, h // 2, w // 2, channels, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, channels, 1, h, w).contiguous()


@dataclass
class RequestNoise:
    """The request's noise, drawn from ONE generator in the reference's order."""

    target: torch.Tensor  # [1, 24, 1, Hc/16, Wc/16]
    anchors: list[torch.Tensor] = field(default_factory=list)  # per ref, [1, 24, 1, even_h, even_w]
    posteriors: list[torch.Tensor] = field(default_factory=list)  # per ref, [1, 24, 1, posterior_h, posterior_w]


def draw_request_noise(seed: int, canvas: tuple[int, int], refs: list[ReferenceGeometry]) -> RequestNoise:
    """Draw 1: target. Draw 2: each ref's anchor noise (even grid). Draw 3: each ref's posterior noise (16-px grid).

    CPU generator (the reference's `generator_device="cpu"`): the same seed gives the same noise on every machine.
    """
    if not 0 <= int(seed) <= 2**63 - 1:
        raise ValueError(f"seed must be in [0, 2^63-1], got {seed}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    cw, ch = canvas
    kw = {"generator": g, "dtype": torch.float32}
    noise = RequestNoise(target=torch.randn(1, LATENT_CHANNELS, 1, ch // VAE_RATIO, cw // VAE_RATIO, **kw))
    for ref in refs:
        noise.anchors.append(torch.randn(1, LATENT_CHANNELS, 1, ref.even_h, ref.even_w, **kw))
    for ref in refs:
        noise.posteriors.append(torch.randn(1, LATENT_CHANNELS, 1, ref.posterior_h, ref.posterior_w, **kw))
    return noise


def reference_rows(
    moments: torch.Tensor, posterior_eps: torch.Tensor, latents_mean, latents_std
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAE moments `[1, 48, 1, h, w]` -> (clean rows `[R, 96]` fp32, normalized even-cropped latent).

    Posterior *sample* (not the mean), rounded to fp16, even top-left crop, `(z - mean) / std`, patchify.
    """
    mean, logvar = torch.chunk(moments.float(), 2, dim=1)
    std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
    z = posterior_eps.to(mean.dtype) * std + mean
    z = z.to(torch.float16).float()
    z = z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2]
    m = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    s = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    z = (z - m) / s
    return patchify(z), z


def anchor_rows(clean_rows: torch.Tensor, anchor_eps: torch.Tensor) -> torch.Tensor:
    """`0.999f * clean + (1 - 0.999f) * eps`, fp32 -- the reference rows every step sees."""
    t = torch.tensor(REFERENCE_TIMESTEP, dtype=torch.float32)
    return t * clean_rows.float() + (1.0 - t) * patchify(anchor_eps).float()


# ------------------------------------------------------------------------------------------------ packed sequence


def _spatial_axis(dim: int, sqrt_area: float) -> torch.Tensor:
    """`linspace(left, left + ratio, dim // 2, endpoint=False) * 32` in float64 (numpy's endpoint=False form)."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(np.linspace(left, left + ratio, dim // 2, endpoint=False) * ROPE_SPATIAL_SCALE)


def _block_positions(t: float, latent_h: int, latent_w: int) -> torch.Tensor:
    sqrt_area = math.sqrt(latent_h * latent_w)
    hh, ww = torch.meshgrid(_spatial_axis(latent_h, sqrt_area), _spatial_axis(latent_w, sqrt_area), indexing="ij")
    return torch.stack([torch.full_like(hh, float(t)), hh, ww], dim=-1).reshape(-1, 3)


@dataclass
class ImagePackedSequence:
    """`[text L | ref 1 .. ref n | target | pad]` -- what the DiT sees, before device padding."""

    text_len: int
    ref_rows: list[int]
    target_rows: int
    position_ids: torch.Tensor  # float64 [used, 3]
    token_tags: torch.Tensor  # long [used]: text 1, Qwen vision tokens / image rows 0
    row_slot: torch.Tensor  # long [used]: 0 = the step's timestep, 1 = the pinned reference timestep

    @property
    def num_ref_rows(self) -> int:
        return sum(self.ref_rows)

    @property
    def sequence_length(self) -> int:  # "used"
        return self.text_len + self.num_ref_rows + self.target_rows

    @property
    def target_start(self) -> int:
        return self.text_len + self.num_ref_rows


def build_image_packed_sequence(
    text_token_tags: torch.Tensor, refs: list[ReferenceGeometry], canvas_latent_hw: tuple[int, int]
) -> ImagePackedSequence:
    """Text rows at `(j, 0, 0)`; ref i at `t = L + i` on its own sqrt-area grid; the target at `t = L + n_refs`."""
    L = int(text_token_tags.shape[0])
    lh, lw = canvas_latent_hw
    text_pos = torch.stack(
        [
            torch.arange(L, dtype=torch.float64),
            torch.zeros(L, dtype=torch.float64),
            torch.zeros(L, dtype=torch.float64),
        ],
        dim=-1,
    )
    blocks = [text_pos]
    for i, ref in enumerate(refs):
        blocks.append(_block_positions(L + i, ref.even_h, ref.even_w))
    blocks.append(_block_positions(L + len(refs), lh, lw))
    target_rows = (lh // 2) * (lw // 2)
    ref_rows = [ref.num_rows for ref in refs]
    image_rows = sum(ref_rows) + target_rows
    tags = torch.cat([text_token_tags.long(), torch.full((image_rows,), IMAGE_TAG, dtype=torch.long)])
    slot = torch.zeros(L + image_rows, dtype=torch.long)
    slot[L : L + sum(ref_rows)] = 1
    return ImagePackedSequence(
        text_len=L,
        ref_rows=ref_rows,
        target_rows=target_rows,
        position_ids=torch.cat(blocks),
        token_tags=tags,
        row_slot=slot,
    )


# ----------------------------------------------------------------------------------------------------------- sampler


@dataclass
class PddSchedule:
    sigmas: list[float]  # K + 1 values, 1.0 -> 0.0

    @classmethod
    def from_pretrained(cls, weights_dir: str | Path) -> "PddSchedule":
        cfg = json.loads((Path(weights_dir) / "pdd_config.json").read_text())
        sigmas = [float(s) for s in cfg["sigmas"]]
        if sigmas[0] != 1.0 or sigmas[-1] != 0.0 or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError(f"pdd sigmas must decrease strictly from 1.0 to 0.0, got {sigmas}")
        if len(sigmas) - 1 != int(cfg["num_inference_steps"]):
            raise ValueError("pdd_config num_inference_steps does not match its sigma grid")
        # The reference round-trips through fp32 `1 - sigma`; the values are unchanged by it.
        ts = torch.tensor([1.0 - s for s in sigmas[:-1]], dtype=torch.float32)
        return cls(sigmas=torch.cat([1.0 - ts, torch.zeros(1)]).tolist())

    @property
    def num_steps(self) -> int:
        return len(self.sigmas) - 1

    def timestep(self, k: int) -> float:
        """What the DiT is fed at step k: `1 - sigma_k` (fp32), unscaled. Step 0 is t = 0."""
        return float(torch.tensor(1.0 - self.sigmas[k], dtype=torch.float32))

    def dt(self, k: int) -> float:
        """Euler coefficient: `x += (sigma_k - sigma_{k+1}) * v` (fp32)."""
        s = torch.tensor(self.sigmas, dtype=torch.float32)
        return float(s[k] - s[k + 1])


# ------------------------------------------------------------------------------------------------------------ output


def to_uint8_truncated(pixels01: torch.Tensor) -> np.ndarray:
    """`[3, H, W]` float in [0, 1] -> HWC uint8 by truncation (`(x*255).to(uint8)`), as the reference writes PNGs."""
    # One float temporary (the input is left untouched), and an HWC copy made by torch's threaded `contiguous`
    # rather than by PIL on a strided numpy view: 5x faster at 4K, same bits.
    return (pixels01.float() * 255).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def finish_image(pixels01: torch.Tensor, requested: tuple[int, int]) -> Image.Image:
    """Canvas pixels -> the requested size: truncate to uint8, then LANCZOS when the canvas differs."""
    image = Image.fromarray(to_uint8_truncated(pixels01))
    if image.size != tuple(requested):
        image = image.resize(tuple(requested), Image.Resampling.LANCZOS)
    return image
