# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Architecture config for NVIDIA Cosmos3-Super-Image2Video.

All values extracted from https://huggingface.co/nvidia/Cosmos3-Super-Image2Video
at revision 8ec97da4ec5afc56754b1ff67de96fbbb87c76f5 (config.json,
transformer/config.json, vae/config.json, vision_encoder/config.json,
scheduler/scheduler_config.json, model_index.json).
"""

from __future__ import annotations

import os

# COSMOS3_HF_REPO selects a different checkpoint of the same Cosmos3OmniTransformer
# family (e.g. nvidia/Cosmos3-Nano, 16B) without a separate config module — the
# trunk reads its dims from the checkpoint's hf_config at build time, not from the
# dicts below.
HF_REPO = os.environ.get("COSMOS3_HF_REPO", "nvidia/Cosmos3-Super-Image2Video")
HF_REVISION = "8ec97da4ec5afc56754b1ff67de96fbbb87c76f5"


# Cosmos3OmniTransformer — the 64B MoT diffusion+text trunk.
# Per-layer weights are duplicated: one "und" (text/AR) set and one "_moe_gen"
# (diffusion) set; they join only in the attention softmax via the two-way
# joint attention. Both expert sets are active during I2V inference.
TRANSFORMER_CONFIG = dict(
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=64,
    num_key_value_heads=8,  # GQA 8:1
    head_dim=128,
    intermediate_size=25600,  # SwiGLU; ~5x hidden (non-standard)
    hidden_act="silu",
    vocab_size=151936,  # Qwen2 tokenizer vocab
    max_position_embeddings=262144,
    rope_theta=5_000_000,
    rope_scaling=dict(
        rope_type="default",
        mrope_section=(24, 20, 20),  # (temporal, height, width)
        mrope_interleaved=True,
    ),
    position_embedding_type="unified_3d_mrope",
    rms_norm_eps=1e-6,
    latent_channel=48,  # matches VAE z_dim
    latent_patch_size=2,
    patch_latent_dim=192,
    joint_attn_implementation="two_way",
    qk_norm_for_text=True,
    qk_norm_for_diffusion=True,
    use_moe=True,  # MoT dual-expert (not classic routing)
    timestep_scale=0.001,
    enable_fps_modulation=True,
    base_fps=16,
)


# AutoencoderKLWan, TI2V-5B variant. Same class as `models/tt_dit/models/vae/
# vae_wan2_1.py` (Wan2.2) but configured with z_dim=48 instead of 16.
VAE_CONFIG = dict(
    z_dim=48,
    in_channels=12,
    out_channels=12,
    base_dim=160,
    decoder_base_dim=256,
    dim_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    patch_size=2,
    scale_factor_spatial=16,
    scale_factor_temporal=4,
    temperal_downsample=(False, True, True),
    is_residual=True,
    attn_scales=(),
    clip_output=False,
    dropout=0.0,
)


# Qwen3VLVisionModel — encodes the reference image. `out_hidden_size` matches
# the transformer hidden so features project directly. Three intermediate
# layers (8, 16, 24) are injected into the trunk, not just the final layer.
VISION_ENCODER_CONFIG = dict(
    hidden_size=1152,
    depth=27,
    num_heads=16,
    intermediate_size=4304,
    in_channels=3,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=5120,
    deepstack_visual_indexes=(8, 16, 24),
    hidden_act="gelu_pytorch_tanh",
    num_position_embeddings=2304,
)


# UniPCMultistepScheduler with rectified-flow prediction. Implementation
# already exists at `models/tt_dit/solvers/unipc.py` — wire it up directly.
SCHEDULER_CONFIG = dict(
    solver_order=2,
    solver_type="bh2",
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    use_karras_sigmas=True,
    flow_shift=5.0,  # Cosmos3 paper Table 21: Cosmos3-Super-Image2Video uses shift=5. shift=10 is the Audio-Visual omni preset.
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    sigma_min=0.147,
    sigma_max=200.0,
    timestep_spacing="linspace",
    predict_x0=True,
    lower_order_final=True,
)


TEXT_TOKENIZER_HF_PATH = f"{HF_REPO}"  # tokenizer/ subdir of this repo


# Approximate parameter counts (used for memory planning, not for execution).
TRANSFORMER_PARAMS = 64_000_000_000  # 27 safetensors shards, ~128 GB FP16
VAE_PARAMS = 700_000_000
VISION_ENCODER_PARAMS = 600_000_000
