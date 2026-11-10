# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 media module for the encoder service, for the FLUX.3 prompt upsampler (Qwen3.5-9B VLM): image checks,
resize, token expansion and stored values. The vision encoder is vision/encoder.py (the head-parallel tower).

prepare reproduces the upsampler's own path pixel for pixel (FLUX.3 reference policy, then the HF Qwen2VL image
processor of the upsampler checkpoint):
- LANCZOS to multiples of 32 under the pixel cap of the request's image count (1 -> 1024^2, 2 -> 724^2, 3-10 ->
  512^2), then a JPEG q92 round trip (black-forest-labs flux3/pup.py, kept: the upsampler was trained on it);
- HF smart_resize (factor 32, 256^2 to 4096^2 pixels): a no-op on the policy's output unless it is under 256^2, then
  torchvision BICUBIC + antialias on uint8 like HF;
- the processor's patch rows. The item keeps them as uint8 [rows, 768] (one temporal copy, unnormalized: 1/8 of the
  bytes of HF's fp32 pixel_values); pixel_values() restores HF's values exactly through a 256-entry table.

The pixel cap depends on the number of images in the REQUEST, which prepare(kind, media) does not see: it takes it as
n_refs (default 1). The same image bytes therefore give different embeddings in a 1-image and a 3-image request, so
the media key must include media_key_salt(item).
"""

import hashlib
import io
import math
from dataclasses import dataclass

import numpy as np

from models.demos.blackhole.qwen36.vision import format as media_format

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
CHANNELS = 3
ROW_U8 = CHANNELS * PATCH_SIZE * PATCH_SIZE  # 768: one temporal copy of a patch
# HF processor size bounds (preprocessor_config.json "size" of the upsampler checkpoint)
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
MAX_RATIO = 200
# FLUX.3 upsampler reference policy (flux3/pup.py)
REFERENCE_MULTIPLE = 32
MAX_REFERENCES = 10
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFERENCES + 1)}}
JPEG_QUALITY = 92
# Only in the tokenizer: the chat template renders <|vision_start|><|image_pad|><|vision_end|> per image
VISION_START = 248053
VISION_END = 248054


@dataclass(frozen=True)
class Qwen35Image:
    patches: np.ndarray
    """uint8 [grid_h * grid_w, 768]: HF's patch rows (2x2 merge-block order), one temporal copy, unnormalized."""
    grid_h: int
    """Patch rows: height / 16 (HF image_grid_thw[1])."""
    grid_w: int
    """Patch columns: width / 16."""
    n_refs: int = 1
    """Images in the request, which picked the pixel cap."""

    @property
    def n_tokens(self) -> int:
        return self.grid_h * self.grid_w // (MERGE_SIZE * MERGE_SIZE)

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (1, self.grid_h, self.grid_w)


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    """flux3/pup.py reference_size: (width, height), multiples of 32, under the pixel cap for n_refs images."""
    if max(width, height) / min(width, height) > MAX_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Hugging Face Qwen2VLImageProcessor's resized (height, width) with the upsampler's factor and bounds."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width):.1f} exceeds {MAX_RATIO}:1")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def reference_policy(rgb: np.ndarray, n_refs: int) -> np.ndarray:
    """flux3/pup.py policy on an RGB image (orientation and alpha handled by the caller): LANCZOS to reference_size,
    JPEG q92 round trip."""
    from PIL import Image

    image = Image.fromarray(rgb, "RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    out = Image.open(io.BytesIO(buffer.getvalue()))
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def hf_resize(rgb: np.ndarray) -> np.ndarray:
    """HF's resize (torchvision uint8 BICUBIC + antialias) when smart_resize changes the size, else rgb itself."""
    h, w = smart_resize(*rgb.shape[:2])
    if (h, w) == rgb.shape[:2]:
        return rgb
    import torch
    from torchvision.transforms.v2 import functional as tvF

    chw = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)  # (a writable copy) HWC memory, CHW shape view
    out = tvF.resize(chw, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return out.permute(1, 2, 0).contiguous().numpy()


def patchify(rgb: np.ndarray) -> np.ndarray:
    """[H, W, 3] uint8 -> [H/16 * W/16, 768] uint8 in the HF processor's row order."""
    h, w, _ = rgb.shape
    gh, gw = h // PATCH_SIZE, w // PATCH_SIZE
    x = rgb.transpose(2, 0, 1).reshape(
        CHANNELS, gh // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE, gw // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE
    )
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # [gh/2, gw/2, 2, 2, C, 16, 16]
    return np.ascontiguousarray(x).reshape(gh * gw, ROW_U8)


def pixel_values(patches: np.ndarray, dtype=None, expand: bool = True):
    """uint8 patches [rows, 768] -> HF pixel_values [rows, 1536] (normalize (x - 127.5) / 127.5 in fp32, the image
    repeated over the temporal patch), bit-exact; dtype torch.float32 (default) or torch.bfloat16 (= the fp32 values
    cast). expand=False returns the [rows, 3, 2, 256] view without materializing the temporal copy."""
    import torch

    dtype = dtype or torch.float32
    lut = (torch.arange(256, dtype=torch.float32) - 127.5) / 127.5
    rows = patches.shape[0]
    one = lut.to(dtype)[torch.from_numpy(patches).long()].reshape(rows, CHANNELS, 1, PATCH_SIZE * PATCH_SIZE)
    four = one.expand(rows, CHANNELS, TEMPORAL_PATCH_SIZE, PATCH_SIZE * PATCH_SIZE)
    return four.reshape(rows, -1) if expand else four


class Qwen35Media:
    def __init__(self, hf_config: dict):
        vision = hf_config["vision_config"]
        assert (
            vision["patch_size"] == PATCH_SIZE
            and vision.get("spatial_merge_size", MERGE_SIZE) == MERGE_SIZE
            and vision.get("temporal_patch_size", TEMPORAL_PATCH_SIZE) == TEMPORAL_PATCH_SIZE
        ), vision
        assert not vision.get("deepstack_visual_indexes"), "deepstack features are not part of the stored value"
        self.hidden = vision["out_hidden_size"]
        self.placeholder_token_ids = {"image": hf_config["image_token_id"]}
        assert (hf_config["vision_start_token_id"], hf_config["vision_end_token_id"]) == (VISION_START, VISION_END)
        # The encoder's (1, N) mesh: N in {4, 8} (16 vision heads split over the chips), 1-D ring fabric
        self.device_params = {
            "l1_small_size": 24576,
            "trace_region_size": 256 * 1024 * 1024,
            "num_command_queues": 1,
            "fabric_config": "FABRIC_1D_RING",
        }
        self._stub = False
        self._encoder = None

    def prepare(self, kind: str, media: np.ndarray, n_refs: int = 1) -> Qwen35Image:
        if kind != "image":
            raise ValueError(f"Qwen3.5 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 3 or 0 in media.shape:
            raise ValueError(f"expected a non-empty uint8 RGB [height, width, 3], got {media.dtype} {media.shape}")
        if n_refs not in MAX_PIXELS_BY_REFS:
            raise ValueError(f"{n_refs} images in the request; the upsampler takes 1 to {MAX_REFERENCES}")
        rgb = hf_resize(reference_policy(media, n_refs))
        h, w = rgb.shape[:2]
        return Qwen35Image(patchify(rgb), h // PATCH_SIZE, w // PATCH_SIZE, n_refs)

    def expansion(self, item: Qwen35Image) -> list[tuple[int, int]]:
        # The template already wraps the marker in <|vision_start|> ... <|vision_end|>; HF's processor expands only
        # the <|image_pad|> itself, so the run replacing it is the pads alone.
        return [(self.placeholder_token_ids["image"], item.n_tokens)]

    def stored_nbytes(self, item: Qwen35Image) -> int:
        return media_format.stored_nbytes(item.n_tokens, self.hidden)

    @staticmethod
    def media_key_salt(item: Qwen35Image) -> bytes:
        """Bytes to add to the image bytes in the media key: the pixel cap bucket (1, 2 or 3-10 images)."""
        return b"qwen35-pup-refs-%d" % min(item.n_refs, 3)

    def load(self, mesh_device, weights_path: str, kind: str, options: dict) -> None:
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"Qwen3.5 does not accept {kind}")
        self.close()
        if options.get("stub_embeddings"):
            self._stub = True
            return
        from models.demos.blackhole.qwen36.vision.encoder import VisionEncoder

        encoder = VisionEncoder(mesh_device, weights_path, options)
        try:
            encoder.load()
        except BaseException:
            encoder.close()
            raise
        self._encoder = encoder

    def encode(self, items: list[Qwen35Image], out: list[memoryview]) -> None:
        assert self._stub or self._encoder is not None, "encode before load"
        if len(items) != len(out):
            raise ValueError(f"{len(items)} items for {len(out)} buffers")
        rows = [media_format.embeddings(buf, self.hidden) for buf in out]
        for item, r in zip(items, rows):
            if r.shape[0] != item.n_tokens:
                raise ValueError(f"buffer of {r.shape[0]} rows for an item of {item.n_tokens} tokens")
        if self._stub:
            import torch

            for item, r in zip(items, rows):
                digest = hashlib.sha256(np.asarray(item.patches.shape).tobytes() + item.patches.tobytes()).digest()
                seed = int.from_bytes(digest[:8], "little")
                stub = torch.randn(item.n_tokens, self.hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
                r[:] = stub.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            return
        self._encoder.encode(items, rows)

    def close(self) -> None:
        self._stub = False
        if self._encoder is not None:
            encoder, self._encoder = self._encoder, None
            encoder.close()
