# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static configuration of Qwen-Image-2.1 (Qwen/Qwen-Image-2.1 @ 790c926) and paths."""
from __future__ import annotations

import os
from dataclasses import dataclass

HF_REPO = "Qwen/Qwen-Image-2.1"
HF_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

PROMPT_DEMO = "White furry llama with black sunglasses, smiling and happy, jumping"
SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n" "<|im_start|>user\n{}<|im_end|>\n" "<|im_start|>assistant\n"
)
# Number of leading (system-role) tokens dropped from the encoder hidden states; equals
# len(processor.apply_chat_template([system message])) for the shipped processor (verified: 14).
DROP_IDX = 14
IMAGE_PAD_TOKEN_ID = 151655


@dataclass(frozen=True)
class DiTConfig:
    num_layers: int = 32
    hidden: int = 4096
    heads: int = 32
    head_dim: int = 128
    mlp_hidden: int = 12288  # mlp_ratio 3
    in_channels: int = 64
    out_channels: int = 64
    context_dim: int = 4096
    axes_dims_rope: tuple = (16, 56, 56)
    rope_theta: int = 10000
    eps: float = 1e-6
    timestep_dim: int = 256


@dataclass(frozen=True)
class TextEncoderConfig:
    num_layers: int = 36
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 12288
    rms_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    vocab: int = 151936


@dataclass(frozen=True)
class VAEConfig:
    z_dim: int = 64
    decoder_base_dim: int = 144
    dim_mult: tuple = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    out_channels: int = 4
    scale_factor_spatial: int = 16
    # decoder temporal upsample flags per up block (reverse of temperal_downsample [F,T,T,T])
    temporal_upsample: tuple = (True, True, True, False)


@dataclass(frozen=True)
class SchedulerConfig:
    base_image_seq_len: int = 256
    max_image_seq_len: int = 8192
    base_shift: float = 0.5
    max_shift: float = 0.9
    shift_terminal: float = 0.02
    num_train_timesteps: int = 1000


DIT = DiTConfig()
TE = TextEncoderConfig()
VAE = VAEConfig()
SCHED = SchedulerConfig()


def snapshot_dir() -> str:
    """Resolve the local HF snapshot of the weights (env QWEN_IMAGE_SNAPSHOT overrides)."""
    env = os.environ.get("QWEN_IMAGE_SNAPSHOT")
    if env:
        if not os.path.isdir(env):
            raise FileNotFoundError(f"QWEN_IMAGE_SNAPSHOT is not a directory: {env}")
        return env
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    d = os.path.join(cache, "models--" + HF_REPO.replace("/", "--"), "snapshots", HF_REVISION)
    if os.path.isdir(d):
        return d
    raise FileNotFoundError(
        f"Missing {HF_REPO} revision {HF_REVISION} under {cache}; "
        "download the pinned checkpoint or set QWEN_IMAGE_SNAPSHOT explicitly"
    )


GOLDENS_DIR = os.environ.get("QWEN_IMAGE_GOLDENS", "generated/qwen_image_2_1/goldens")
