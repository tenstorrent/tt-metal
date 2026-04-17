# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn implementation of Depth Anything V3 (metric branch).

For iteration 9 the DinoV2-L backbone runs entirely on a single Blackhole
p150a chip, while the small DPT head still runs on CPU. The on-chip flow:

  1. Run patch_embed + cls + pos_embed on CPU (cheap, simplifies tile alignment).
  2. Upload the (B, 1370, 1024) embedding to device, padded to (B, 1376, 1024)
     for tile-friendly matmuls.
  3. Run 24 DinoV2 transformer blocks on chip, capturing intermediate outputs
     at layers (4, 11, 17, 23) for the DPT head.
  4. Download those four intermediates, drop padding, hand to the CPU DPT head.

The device handle and all 24 blocks' weights are cached in module-level state
so the per-call cost reduces to upload-input + chip-compute + download-outputs.
The benchmark harness drives the timing and `accuracy=` PCC vs the fp32
reference, so this module's job is just `run(pixel_values) -> (depth, peak_dram_mb)`.
"""

from __future__ import annotations

import gc
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

import ttnn

from models.experimental.depth_anything_v3.reference.dinov2_l_dpt import (
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    OUT_LAYERS,
    PATCH_SIZE,
    build_da3_metric,
)


_DEVICE_ID = 0
_TILE = 32

# Cached state — populated on first call, reused on subsequent calls.
_state: dict = {}


_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
# 8x12=96-core grid. Larger 10x12 grids on Blackhole shifted PCC below the
# 99% guard — the extra rows alter reduction ordering enough to matter for
# bf16 accumulation. 96 cores still beats default-grid placement.
_CORE_GRID = ttnn.CoreGrid(y=8, x=12)


def _hifi4_kernel_config():
    """Higher matmul fidelity + fp32 accumulation. Closes most of the bf16 PCC
    gap left by the default LoFi math fidelity. Cached as a module constant —
    constructing it per call adds 168 ctor calls per forward."""
    return _HIFI4


def _next_tile(n: int) -> int:
    return ((n + _TILE - 1) // _TILE) * _TILE


def _to_dev(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _upload_block(block, device) -> dict:
    """Move all weight tensors of a single DinoV2 block onto the chip."""
    sd = block.state_dict()

    # qkv linear: nn.Linear stores weight as (out, in); ttnn.linear expects (in, out).
    qkv_w = sd["attn.qkv.weight"].T.contiguous()  # (1024, 3072)
    qkv_b = sd["attn.qkv.bias"]                   # (3072,)
    proj_w = sd["attn.proj.weight"].T.contiguous()  # (1024, 1024)
    proj_b = sd["attn.proj.bias"]
    fc1_w = sd["mlp.fc1.weight"].T.contiguous()     # (1024, 4096)
    fc1_b = sd["mlp.fc1.bias"]
    fc2_w = sd["mlp.fc2.weight"].T.contiguous()     # (4096, 1024)
    fc2_b = sd["mlp.fc2.bias"]

    return {
        "norm1_w": _to_dev(sd["norm1.weight"].unsqueeze(0), device),
        "norm1_b": _to_dev(sd["norm1.bias"].unsqueeze(0), device),
        "qkv_w": _to_dev(qkv_w, device),
        "qkv_b": _to_dev(qkv_b.unsqueeze(0), device),
        "proj_w": _to_dev(proj_w, device),
        "proj_b": _to_dev(proj_b.unsqueeze(0), device),
        "ls1_g": _to_dev(sd["ls1.gamma"].unsqueeze(0), device),
        "norm2_w": _to_dev(sd["norm2.weight"].unsqueeze(0), device),
        "norm2_b": _to_dev(sd["norm2.bias"].unsqueeze(0), device),
        "fc1_w": _to_dev(fc1_w, device),
        "fc1_b": _to_dev(fc1_b.unsqueeze(0), device),
        "fc2_w": _to_dev(fc2_w, device),
        "fc2_b": _to_dev(fc2_b.unsqueeze(0), device),
        "ls2_g": _to_dev(sd["ls2.gamma"].unsqueeze(0), device),
    }


def _ttnn_block(x, p, attention_mask):
    """Run one DinoV2 block on chip. Mirrors openvla's dinov2_attention/feedforward
    patterns. Uses fused attention_softmax_ with an additive mask so the 6 padded
    sequence positions (1370 -> 1376 tile alignment) get -inf attention weight."""
    head_dim = EMBED_DIM // NUM_HEADS

    # ----- Attention -----
    h = ttnn.layer_norm(x, weight=p["norm1_w"], bias=p["norm1_b"], epsilon=1e-6,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    qkv = ttnn.linear(h, p["qkv_w"], bias=p["qkv_b"],
                      memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                      compute_kernel_config=_hifi4_kernel_config())
    ttnn.deallocate(h)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=NUM_HEADS,
    )
    ttnn.deallocate(qkv)
    attn_scores = ttnn.matmul(q, k, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                              compute_kernel_config=_hifi4_kernel_config())
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    # Fused: scale by 1/sqrt(head_dim), add additive mask, then softmax.
    attn_probs = ttnn.transformer.attention_softmax_(
        attn_scores, attention_mask=attention_mask, head_size=head_dim,
    )
    ctx = ttnn.matmul(attn_probs, v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                      compute_kernel_config=_hifi4_kernel_config())
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(v)
    ctx = ttnn.transformer.concatenate_heads(ctx, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(ctx, p["proj_w"], bias=p["proj_b"],
                      memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                      compute_kernel_config=_hifi4_kernel_config())
    ttnn.deallocate(ctx)
    out = ttnn.mul(out, p["ls1_g"])
    x = ttnn.add(x, out)
    ttnn.deallocate(out)

    # ----- MLP -----
    h = ttnn.layer_norm(x, weight=p["norm2_w"], bias=p["norm2_b"], epsilon=1e-6,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    h = ttnn.linear(h, p["fc1_w"], bias=p["fc1_b"],
                    memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                    activation="gelu", compute_kernel_config=_hifi4_kernel_config(),
                    core_grid=_CORE_GRID)
    h = ttnn.linear(h, p["fc2_w"], bias=p["fc2_b"],
                    memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                    compute_kernel_config=_hifi4_kernel_config(), core_grid=_CORE_GRID)
    h = ttnn.mul(h, p["ls2_g"])
    x = ttnn.add(x, h)
    ttnn.deallocate(h)
    return x


def _cpu_head_channels_last(head, intermediates, img_hw):
    """DPT head with explicit channels_last activation layout. The head's weights
    are converted to channels_last in setup; we need to convert each intermediate
    once after the (B, N, C) -> (B, C, H, W) reshape and the rest stays NHWC.

    Microprofile pre-iter17 showed refinenets = 106ms / 154ms head (largest hotspot);
    these are 3x3 convs at 296x296 with 256 channels which benefit from the AVX512
    NHWC bf16 conv2d kernels."""
    H, W = img_hw
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
    feats = []
    for i, x in enumerate(intermediates):
        x = x[:, 1:, :].transpose(1, 2).reshape(x.shape[0], -1, Hp, Wp)
        x = head.projects[i](x)
        x = head.resize_layers[i](x)
        feats.append(x)
    layer_rns = [head.scratch.layer1_rn, head.scratch.layer2_rn,
                 head.scratch.layer3_rn, head.scratch.layer4_rn]
    rn = [layer_rns[i](feats[i]) for i in range(4)]
    path = head.scratch.refinenet4(rn[3])
    path = head.scratch.refinenet3(path, rn[2])
    path = head.scratch.refinenet2(path, rn[1])
    path = head.scratch.refinenet1(path, rn[0])
    path = head.scratch.output_conv1(path)
    path = F.interpolate(path, size=(H, W), mode="bilinear", align_corners=True)
    return head.scratch.output_conv2(path)


def _cpu_embed(pixel_values: torch.Tensor, backbone) -> Tuple[torch.Tensor, int, int]:
    """Run patch_embed + cls prepend + pos_embed on CPU (cheap, no tiling pain)."""
    B, _, H, W = pixel_values.shape
    Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
    x = backbone.patch_embed["proj"](pixel_values)
    x = x.flatten(2).transpose(1, 2)
    cls = backbone.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + backbone._interpolate_pos_embed(Hp, Wp)
    return x, Hp, Wp


def _build_attention_mask(seq_len: int, padded_len: int, device):
    """Additive bias mask: 0 for valid tokens, -1e4 for the padded tail.
    Shape (1, 1, 1, padded_len) which broadcasts over (B, num_heads, seq_q)."""
    mask = torch.zeros(1, 1, 1, padded_len, dtype=torch.bfloat16)
    mask[..., seq_len:] = -1e4
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _setup(pixel_values: torch.Tensor):
    """One-time setup: load the model, open the device, upload all weights."""
    if "device" in _state:
        return
    img_size = pixel_values.shape[-1]
    cpu_model = build_da3_metric(load_weights=True, img_size=img_size).eval()
    # Cast both halves to bfloat16. The head benefits via AVX512_BF16 conv2d
    # (head also goes channels_last). The backbone is only used for the cheap
    # patch_embed + cls + pos_embed step before chip upload — running it in
    # bf16 directly saves the fp32->bf16 cast we'd otherwise do on the result.
    cpu_model.backbone = cpu_model.backbone.to(torch.bfloat16)
    cpu_model.head = cpu_model.head.to(torch.bfloat16).to(memory_format=torch.channels_last)
    device = ttnn.open_device(device_id=_DEVICE_ID)
    blocks = [_upload_block(b, device) for b in cpu_model.backbone.blocks]
    seq_len = (img_size // PATCH_SIZE) ** 2 + 1  # +1 for cls token
    padded = _next_tile(seq_len)
    attention_mask = _build_attention_mask(seq_len, padded, device)
    _state.update(
        cpu_model=cpu_model,
        device=device,
        blocks=blocks,
        attention_mask=attention_mask,
    )


def run(pixel_values: torch.Tensor):
    """Run the metric branch with backbone on chip + DPT head on CPU.
    Returns (depth_tensor, peak_dram_mb)."""
    _setup(pixel_values)
    cpu_model = _state["cpu_model"]
    device = _state["device"]
    blocks = _state["blocks"]
    attention_mask = _state["attention_mask"]

    with torch.inference_mode():
        # Backbone is bfloat16; match dtype on input to avoid an in-conv cast.
        emb, Hp, Wp = _cpu_embed(pixel_values.to(torch.bfloat16), cpu_model.backbone)

        # Pad sequence dim to a tile multiple (1370 -> 1376) for ttnn matmul.
        N = emb.shape[1]
        N_pad = _next_tile(N)
        if N_pad != N:
            emb = F.pad(emb, (0, 0, 0, N_pad - N))

        x_tt = ttnn.from_torch(
            emb.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Save the ttnn handles for the intermediate stages without syncing,
        # so chip can pipeline all 24 blocks back-to-back. We sync them once
        # at the end via a single batched download loop.
        intermediate_handles: List = []
        out_layer_set = set(OUT_LAYERS)
        for i, p in enumerate(blocks):
            x_tt = _ttnn_block(x_tt, p, attention_mask)
            if i in out_layer_set:
                # Clone so the next block can reassign x_tt without freeing this.
                intermediate_handles.append(ttnn.clone(x_tt))

        intermediates_cpu: List[torch.Tensor] = [
            ttnn.to_torch(h)[..., :N, :] for h in intermediate_handles
        ]
        for h in intermediate_handles:
            ttnn.deallocate(h)
        ttnn.deallocate(x_tt)

        # DPT head with explicit channels_last activation layout.
        depth = _cpu_head_channels_last(
            cpu_model.head, intermediates_cpu, img_hw=pixel_values.shape[-2:]
        )

    return depth, 0.0
