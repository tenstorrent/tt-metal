# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host input preparation for Alpamayo 2 Edge driving inference.

Port of the NVlabs `helper.prepare_model_inputs` path (sample ordering, uint8 bicubic resize, chat template) and of
the Cosmos3 Edge processor (image-token expansion, block-major patchify, fused rescale + normalize), followed by the
model's history-token fusion and multimodal rope index. Both the reference and the TT model consume `ModelInputs`.
"""

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from models.experimental.alpamayo2_edge.reference.chat import CLOSED_THINKING, build_conversation, render_chat_template
from models.experimental.alpamayo2_edge.reference.config import Alpamayo2EdgeConfig
from models.experimental.alpamayo2_edge.reference.rope_index import get_rope_index
from models.experimental.alpamayo2_edge.reference.traj_tokens import DeltaTrajectoryTokenizer, fuse_traj_tokens

CAMERA_FRAME_COUNTS = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
CAMERA_PIXEL_BOUNDS = {
    camera: (163840, 196608) if camera in (1, 6) else (81920, 98304) for camera in CAMERA_FRAME_COUNTS
}
HISTORY_STEPS = 20
IMAGE_TOKEN = "<|image_pad|>"


@dataclass
class ModelInputs:
    input_ids: torch.Tensor  # [B, S] int64, history tokens fused
    attention_mask: torch.Tensor  # [B, S] int64
    mm_token_type_ids: torch.Tensor  # [B, S] int64: 1 on image tokens
    patches: torch.Tensor  # [N, 768] float32 normalized, block-major rows, (y, x, channel) within a patch
    image_grid_thw: torch.Tensor  # [I, 3] int64
    position_ids: torch.Tensor  # [3, B, S] int64
    rope_deltas: torch.Tensor  # [B, 1] int64
    ego_history_xyz: torch.Tensor  # [B, 20, 3] float32
    ego_history_rot: torch.Tensor  # [B, 20, 3, 3] float32


@dataclass(frozen=True)
class _CheckpointAssets:
    tokenizer: Tokenizer
    chat_template: str
    special_tokens: dict
    image_processor: dict
    history_tokenizer: DeltaTrajectoryTokenizer
    tokens_per_history_traj: int
    include_camera_ids: bool
    system_prompt: str | None


@lru_cache(maxsize=4)
def _checkpoint_assets(model_dir: Path) -> _CheckpointAssets:
    """Tokenizer, chat template, processor settings and history tokenizer, all read from the checkpoint."""
    raw = json.loads((model_dir / "config.json").read_text())
    tokenizer_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    processor_cfg = json.loads((model_dir / "processor_config.json").read_text())["image_processor"]
    history_cfg = {k: v for k, v in raw["hist_traj_tokenizer_cfg"].items() if k != "_target_"}
    return _CheckpointAssets(
        tokenizer=Tokenizer.from_file(str(model_dir / "tokenizer.json")),
        chat_template=(model_dir / "chat_template.jinja").read_text(),
        special_tokens={k: tokenizer_cfg[k] for k in ("bos_token", "eos_token", "unk_token", "pad_token")},
        image_processor=processor_cfg,
        history_tokenizer=DeltaTrajectoryTokenizer(**history_cfg),
        tokens_per_history_traj=raw["tokens_per_history_traj"],
        include_camera_ids=raw["include_camera_ids"],
        system_prompt=raw.get("system_prompt"),
    )


def canonical_frame_order(camera_indices: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
    """Frame indices sorted by camera, then capture timestamp."""
    by_time = torch.argsort(timestamps.reshape(-1), stable=True)
    return by_time[torch.argsort(camera_indices[by_time], stable=True)]


def resize_frame(frame: torch.Tensor, min_pixels: int, max_pixels: int, factor: int) -> torch.Tensor:
    """[3, H, W] uint8 -> [3, h, w] uint8 with h, w multiples of `factor` and h * w within the pixel bounds."""
    height, width = smart_resize(*frame.shape[-2:], factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    return resize(frame, [height, width], interpolation=InterpolationMode.BICUBIC)


def patchify(image: torch.Tensor, patch: int, merge: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """[C, H, W] -> ([gh * gw, patch * patch * C], (gh, gw)).

    Rows walk each merge x merge block of patches before the next block; values inside a patch are (y, x, channel).
    """
    channels, height, width = image.shape
    gh, gw = height // patch, width // patch
    x = image.reshape(channels, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(1, 4, 2, 5, 3, 6, 0)
    return x.reshape(gh * gw, patch * patch * channels), (gh, gw)


def normalize_patches(patches: torch.Tensor, mean, std, rescale_factor: float) -> torch.Tensor:
    """Fused rescale + normalize of uint8 patches with (y, x, channel) layout, as the reference helper applies it."""
    channels = len(mean)
    scale = 1.0 / rescale_factor
    mean = torch.tensor(mean, dtype=torch.float32) * scale
    std = torch.tensor(std, dtype=torch.float32) * scale
    x = patches.float().reshape(patches.shape[0], -1, channels)
    return ((x - mean) / std).reshape(patches.shape[0], -1)


def _history_tensor(value, tail: tuple) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    value = value.reshape(-1, *tail) if value.ndim > len(tail) else value.unsqueeze(0)
    if value.shape != (1, *tail):
        raise ValueError(f"expected one ego history of shape {tail}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("ego history poses must be finite")
    return value


def prepare_inputs(
    cfg: Alpamayo2EdgeConfig,
    frames: list[np.ndarray],
    camera_ids: list[int],
    timestamps_us: list[int],
    ego_history_xyz,
    ego_history_rot,
    nav_text: str,
) -> ModelInputs:
    """Pack one driving sample (B = 1): HxWx3 uint8 frames in any order, ego history in the t0 ego frame."""
    assets = _checkpoint_assets(Path(cfg.model_dir))
    proc = assets.image_processor
    patch_cfg = (proc["temporal_patch_size"], proc["patch_size"], proc["merge_size"])
    if patch_cfg != (1, cfg.vision.patch, cfg.vision.merge):
        raise ValueError(f"processor_config.json patching {patch_cfg} disagrees with the vision config")
    if not (len(frames) == len(camera_ids) == len(timestamps_us)):
        raise ValueError("frames, camera_ids and timestamps_us must have one entry per frame")
    if dict(Counter(camera_ids)) != CAMERA_FRAME_COUNTS:
        raise ValueError(
            f"the driving profile needs frames per camera {CAMERA_FRAME_COUNTS}, got {Counter(camera_ids)}"
        )
    if not nav_text.strip():
        raise ValueError("nav_text must be nonempty")
    xyz = _history_tensor(ego_history_xyz, (HISTORY_STEPS, 3))
    rot = _history_tensor(ego_history_rot, (HISTORY_STEPS, 3, 3))

    cameras = torch.tensor(camera_ids, dtype=torch.int64)
    timestamps = torch.tensor(timestamps_us, dtype=torch.int64)
    order = canonical_frame_order(cameras, timestamps).tolist()
    images = [torch.from_numpy(np.ascontiguousarray(frames[i])).permute(2, 0, 1).contiguous() for i in order]
    for image in images:
        if image.dtype != torch.uint8 or image.shape[0] != 3:
            raise ValueError("frames must be HxWx3 uint8")

    messages = build_conversation(
        images,
        cameras[order],
        timestamps[order],
        CAMERA_PIXEL_BOUNDS,
        nav_text,
        num_tokens_per_history_traj=assets.tokens_per_history_traj,
        include_camera_ids=assets.include_camera_ids,
        system_prompt=assets.system_prompt,
    )
    text = render_chat_template(
        assets.chat_template,
        messages,
        add_generation_prompt=True,
        add_vision_id=False,
        enable_thinking=False,
        **assets.special_tokens,
    )
    if not text.endswith(CLOSED_THINKING):
        raise ValueError(f"The chat template must end the generation prompt with {CLOSED_THINKING}")

    factor = cfg.vision.patch * cfg.vision.merge
    patch_rows, grids, image_tokens = [], [], []
    for entry in messages[1]["content"]:
        if entry["type"] != "image":
            continue
        resized = resize_frame(entry["image"], entry["min_pixels"], entry["max_pixels"], factor)
        rows, (gh, gw) = patchify(resized, cfg.vision.patch, cfg.vision.merge)
        patch_rows.append(rows)
        grids.append((1, gh, gw))
        image_tokens.append(IMAGE_TOKEN * (gh * gw // cfg.vision.merge**2))
    patches = normalize_patches(torch.cat(patch_rows), proc["image_mean"], proc["image_std"], proc["rescale_factor"])

    # The template wraps every image as <|vision_start|><|image_pad|><|vision_end|>; the processor expands the pad.
    pieces = text.split(IMAGE_TOKEN)
    if len(pieces) != len(image_tokens) + 1:
        raise ValueError(f"prompt has {len(pieces) - 1} image placeholders for {len(image_tokens)} images")
    text = "".join(piece + tokens for piece, tokens in zip(pieces, image_tokens + [""]))
    input_ids = torch.tensor([assets.tokenizer.encode(text).ids], dtype=torch.int64)

    mm_token_type_ids = (input_ids == cfg.tokens.image).long()
    input_ids = fuse_traj_tokens(assets.history_tokenizer, input_ids, xyz.unsqueeze(1), cfg.tokens)
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor(grids, dtype=torch.int64)
    position_ids, rope_deltas = get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw, attention_mask, merge=cfg.vision.merge
    )
    return ModelInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        patches=patches,
        image_grid_thw=image_grid_thw,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        ego_history_xyz=xyz,
        ego_history_rot=rot,
    )


def prepare_sample_inputs(cfg: Alpamayo2EdgeConfig, sample: dict) -> ModelInputs:
    """`prepare_inputs` on a dict in the NVlabs sample format (see `synthetic_sample`)."""
    return prepare_inputs(
        cfg,
        [frame[0].permute(1, 2, 0).numpy() for frame in sample["image_frames"]],
        sample["camera_indices"].tolist(),
        sample["absolute_timestamps"].reshape(-1).tolist(),
        sample["ego_history_xyz"],
        sample["ego_history_rot"],
        sample["nav_text"][0],
    )


def synthetic_sample(seed: int, src_hw: tuple[int, int] = (1080, 1920)) -> dict:
    """A driving sample in the NVlabs loader format with seeded uniform-noise images.

    22 camera-major frames (cameras 1 and 6 at -2, -1, -0.5, -0.1, 0 s; cameras 0, 2, 3, 5 at -1, -0.1, 0 s) and a
    straight 10 m/s ego history of 20 poses at 0.1 s steps, expressed in the ego frame at t0.
    """
    t0_us = 5_100_000
    cameras, timestamps = [], []
    for camera, count in CAMERA_FRAME_COUNTS.items():
        offsets = [-2_000_000, -1_000_000, -500_000, -100_000, 0] if count == 5 else [-1_000_000, -100_000, 0]
        cameras += [camera] * count
        timestamps += [t0_us + offset for offset in offsets]
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (len(cameras), 1, 3, *src_hw), dtype=torch.uint8, generator=generator)
    speed_mps, step_s = 10.0, 0.1
    xyz = torch.zeros(1, HISTORY_STEPS, 3)
    xyz[0, :, 0] = torch.arange(-(HISTORY_STEPS - 1), 1, dtype=torch.float32) * (speed_mps * step_s)
    return {
        "image_frames": images,
        "camera_indices": torch.tensor(cameras, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "ego_history_xyz": xyz,
        "ego_history_rot": torch.eye(3).expand(1, HISTORY_STEPS, 3, 3).contiguous(),
        "nav_text": ["Continue straight."],
    }
