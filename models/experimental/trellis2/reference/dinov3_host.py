# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the DINOv3 ViT-L/16 image encoder as TRELLIS.2 uses it (`modules/image_feature_extractor.py`
`DinoV3FeatureExtractor`, transformers `DINOv3ViTModel`, `facebook/dinov3-vitl16-pretrain-lvd1689m`).

TRELLIS.2 runs the embeddings and the 24 layers and then a LayerNorm without affine (not the model's own final norm);
the result is the cond for every flow stage: [1, 5 + (S / 16)^2, 1024] (1,029 tokens at 512 px).

The TT encoder pads the token sequence to a multiple of 512 so its SDPA can use large key chunks (LESSONS Device #16)
and masks the padding as a separate window, as the SLat DiTs do. `functional_forward` restates that decomposition in
torch: patches unfolded on the host and embedded by a matmul, RoPE as x * cos + (x @ R) * sin over all tokens with
cos = 1, sin = 0 on the prefix and padding rows, block-diagonal attention over [real | padding].
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

HIDDEN = 1024
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = HIDDEN // NUM_HEADS
MLP = 4096
PATCH = 16
NUM_REGISTERS = 4
NUM_PREFIX = 1 + NUM_REGISTERS
ROPE_THETA = 100.0
NORM_EPS = 1e-5
TOKEN_MULTIPLE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WEIGHTS = "dinov3-vitl16-pretrain-lvd1689m/model.safetensors"


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def preprocess(image, size: int) -> torch.Tensor:
    """PIL image -> [1, 3, size, size] as `DinoV3FeatureExtractor.__call__` does (LANCZOS, / 255, ImageNet norm)."""
    from PIL import Image

    image = image.resize((size, size), Image.LANCZOS)
    x = torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None]
    return (x - torch.tensor(MEAN)[None, :, None, None]) / torch.tensor(STD)[None, :, None, None]


def num_tokens(size: int) -> int:
    return NUM_PREFIX + (size // PATCH) ** 2


def padded_tokens(n: int) -> int:
    return -(-n // TOKEN_MULTIPLE) * TOKEN_MULTIPLE


def patchify(x: torch.Tensor) -> torch.Tensor:
    """[1, 3, H, W] -> [H/16 * W/16, 768] in the conv weight's (c, kh, kw) order, patches row-major."""
    _, c, h, w = x.shape
    x = x.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).permute(1, 3, 0, 2, 4)
    return x.reshape((h // PATCH) * (w // PATCH), c * PATCH * PATCH)


def rope_tables(size: int, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [n_pad, 64] in fp32: the HF patch-center RoPE on the patch rows, identity on prefix and padding."""
    n = size // PATCH
    inv_freq = 1 / ROPE_THETA ** torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32)
    ch = torch.arange(0.5, n, dtype=torch.float32) / n
    coords = torch.stack(torch.meshgrid(ch, ch, indexing="ij"), dim=-1).flatten(0, 1) * 2.0 - 1.0
    angles = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    cos, sin = torch.ones(n_pad, HEAD_DIM), torch.zeros(n_pad, HEAD_DIM)
    cos[NUM_PREFIX : NUM_PREFIX + n * n], sin[NUM_PREFIX : NUM_PREFIX + n * n] = torch.cos(angles), torch.sin(angles)
    return cos, sin


def rotate_half_matrix() -> torch.Tensor:
    """R with x @ R == rotate_half(x) = cat(-x[32:], x[:32])."""
    h = HEAD_DIM // 2
    r = torch.zeros(HEAD_DIM, HEAD_DIM)
    r[torch.arange(h) + h, torch.arange(h)] = -1.0
    r[torch.arange(h), torch.arange(h) + h] = 1.0
    return r


def embed(state: dict[str, torch.Tensor], pixels: torch.Tensor, n_pad: int) -> torch.Tensor:
    """[1, 1, n_pad, 1024]: CLS, registers, patch embeddings (matmul on unfolded patches), zero padding."""
    w = state["embeddings.patch_embeddings.weight"].float().reshape(HIDDEN, -1)
    patches = patchify(pixels.float()) @ w.T + state["embeddings.patch_embeddings.bias"].float()
    prefix = torch.cat([state["embeddings.cls_token"][0], state["embeddings.register_tokens"][0]]).float()
    h = torch.cat([prefix, patches, patches.new_zeros(n_pad - prefix.shape[0] - patches.shape[0], HIDDEN)])
    return h[None, None]


def _heads(x: torch.Tensor) -> torch.Tensor:
    b, _, n, _ = x.shape
    return x.reshape(b, n, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)


def functional_forward(state: dict[str, torch.Tensor], pixels: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the TT encoder in fp32: pixels [1, 3, S, S] -> cond [1, N, 1024] (N real tokens)."""
    s = {k: v.float() for k, v in state.items()}
    size = pixels.shape[-1]
    n, n_pad = num_tokens(size), padded_tokens(num_tokens(size))
    cos, sin = rope_tables(size, n_pad)
    rot = rotate_half_matrix()
    mask = torch.zeros(n_pad, n_pad, dtype=torch.bool)
    mask[:n, :n] = mask[n:, n:] = True
    h = embed(s, pixels, n_pad)
    for i in range(NUM_LAYERS):
        p = f"layer.{i}."
        a = F.layer_norm(h, (HIDDEN,), s[p + "norm1.weight"], s[p + "norm1.bias"], eps=NORM_EPS)
        q = _heads(F.linear(a, s[p + "attention.q_proj.weight"], s[p + "attention.q_proj.bias"]))
        k = _heads(F.linear(a, s[p + "attention.k_proj.weight"]))
        v = _heads(F.linear(a, s[p + "attention.v_proj.weight"], s[p + "attention.v_proj.bias"]))
        q, k = q * cos + (q @ rot) * sin, k * cos + (k @ rot) * sin
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).permute(0, 2, 1, 3).reshape(1, 1, n_pad, HIDDEN)
        h = (
            h
            + F.linear(o, s[p + "attention.o_proj.weight"], s[p + "attention.o_proj.bias"])
            * s[p + "layer_scale1.lambda1"]
        )
        a = F.layer_norm(h, (HIDDEN,), s[p + "norm2.weight"], s[p + "norm2.bias"], eps=NORM_EPS)
        a = F.gelu(F.linear(a, s[p + "mlp.up_proj.weight"], s[p + "mlp.up_proj.bias"]))
        h = h + F.linear(a, s[p + "mlp.down_proj.weight"], s[p + "mlp.down_proj.bias"]) * s[p + "layer_scale2.lambda1"]
    return F.layer_norm(h[0, :, :n], (HIDDEN,), eps=NORM_EPS)
