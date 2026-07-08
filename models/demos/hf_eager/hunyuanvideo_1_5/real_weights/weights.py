# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load the REAL tencent/HunyuanVideo-1.5 DiT checkpoint into the diffusers
``HunyuanVideo15Transformer3DModel`` via the hyvideo->diffusers key converter.

Portable: fetches the checkpoint with ``hf_hub_download`` (uses the local HF cache
if present, downloads otherwise), so it works unchanged on QB2. Set
``HF_HUB_DISABLE_XET=1`` to avoid the xet stall seen on some networks.

Key facts baked in (discovered during bring-up, see README):
  * The real patch-embed is 65-channel (concat-condition: 32 latent + 32 cond + 1
    mask), NOT the config's nominal in_channels=32 -> build with in_channels=65.
  * t2v is signaled by all-zero image_embeds; the byt5 (context_embedder_2) path
    is mandatory in the forward.
"""
import os
import re

import torch

_REPO = "tencent/HunyuanVideo-1.5"
_DEFAULT_VARIANT = "transformer/720p_t2v"

# Full 720p_t2v config mapped to diffusers ctor kwargs (in_channels=65).
CONFIG = dict(
    in_channels=65,
    out_channels=32,
    num_attention_heads=16,
    attention_head_dim=128,
    num_layers=54,
    num_refiner_layers=2,
    mlp_ratio=4,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    text_embed_dim=3584,
    image_embed_dim=1152,
    rope_theta=256,
    rope_axes_dim=(16, 56, 56),
    task_type="t2v",
    use_meanflow=False,
)


def load_converted_state_dict(repo=_REPO, variant=_DEFAULT_VARIANT):
    """native hyvideo checkpoint -> diffusers-layout state_dict (fp32 on CPU)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from .hyvideo_to_diffusers import convert

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    path = hf_hub_download(repo, "diffusion_pytorch_model.safetensors", subfolder=variant)
    raw = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    return convert(raw)


def build_real_transformer(num_layers=54, dtype=torch.float32, repo=_REPO, variant=_DEFAULT_VARIANT):
    """Return ``HunyuanVideo15Transformer3DModel`` loaded with the real weights.

    ``num_layers`` < 54 loads only the first N double-blocks (for a shallower,
    memory-lighter model). ``dtype`` is the CPU dtype; for ttnn the stubs upload
    from this module (bf16-coerce the ttnn uploads to fit one chip — see README)."""
    from diffusers import HunyuanVideo15Transformer3DModel as M

    conv = load_converted_state_dict(repo, variant)
    sd = {}
    for k, v in conv.items():
        m = re.match(r"transformer_blocks\.(\d+)\.", k)
        if m and int(m.group(1)) >= num_layers:
            continue
        sd[k] = v.to(dtype)
    cfg = dict(CONFIG, num_layers=num_layers)
    with torch.device("meta"):
        model = M(**cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()
    return model
