# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# PP-OCRv6_small_rec on ttnn. Activations are flattened NHWC: [1, 1, B*H*W, C].

import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.reference.rec import HEADS, HIDDEN, LN_EPS, STAGE_CONFIGS
from models.demos.blackhole.pp_ocrv6.tt.common import (
    SILU,
    Backbone,
    Conv,
    Linear,
    compute_config,
    fold_bn,
    to_row_major,
)

PADDED_HEAD_DIM = 32


class LayerNorm:
    def __init__(self, device, ln):
        def p(t):
            return ttnn.from_torch(
                t.detach().reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        self.weight, self.bias = p(ln.weight), p(ln.bias)
        self.compute_config = compute_config(device)

    def __call__(self, x):
        return ttnn.layer_norm(
            x, weight=self.weight, bias=self.bias, epsilon=LN_EPS, compute_kernel_config=self.compute_config
        )


def _pad_heads(t, head_dim):
    """[HEADS*head_dim, ...] -> [HEADS*PADDED_HEAD_DIM, ...] with zero rows per head."""
    t = t.reshape(HEADS, head_dim, *t.shape[1:])
    out = torch.zeros(HEADS, PADDED_HEAD_DIM, *t.shape[2:], dtype=t.dtype)
    out[:, :head_dim] = t
    return out.reshape(HEADS * PADDED_HEAD_DIM, *t.shape[2:])


class Attention:
    def __init__(self, device, attn):
        head_dim = HIDDEN // HEADS
        w, b = attn.qkv.weight.detach(), attn.qkv.bias.detach()
        scale = head_dim**-0.5
        parts_w, parts_b = [], []
        for i in range(3):
            pw, pb = w[i * HIDDEN : (i + 1) * HIDDEN], b[i * HIDDEN : (i + 1) * HIDDEN]
            if i == 0:
                pw, pb = pw * scale, pb * scale
            parts_w.append(_pad_heads(pw, head_dim))
            parts_b.append(_pad_heads(pb, head_dim))
        self.qkv = Linear(device, torch.cat(parts_w), torch.cat(parts_b))
        proj_w = _pad_heads(attn.projection.weight.detach().t(), head_dim).t().contiguous()
        self.proj = Linear(device, proj_w, attn.projection.bias)
        self.compute_config = compute_config(device)

    def __call__(self, x, batch):
        rows = x.shape[2]
        seq = rows // batch
        qkv = ttnn.reshape(self.qkv(x, ttnn.L1_MEMORY_CONFIG), (batch, seq, 3 * HEADS * PADDED_HEAD_DIM))
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=HEADS)
        scores = ttnn.matmul(q, k, compute_kernel_config=self.compute_config)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        ctx = ttnn.matmul(probs, v, compute_kernel_config=self.compute_config)
        ctx = ttnn.transformer.concatenate_heads(ctx)
        return self.proj(ttnn.reshape(ctx, (1, 1, rows, HEADS * PADDED_HEAD_DIM)))


class Block:
    def __init__(self, device, blk):
        self.ln1 = LayerNorm(device, blk.layer_norm1)
        self.attn = Attention(device, blk.self_attn)
        self.ln2 = LayerNorm(device, blk.layer_norm2)
        self.fc1 = Linear(device, blk.mlp.fc1.weight, blk.mlp.fc1.bias, activation="silu")
        self.fc2 = Linear(device, blk.mlp.fc2.weight, blk.mlp.fc2.bias)

    def __call__(self, x, batch):
        x = ttnn.add(x, self.attn(self.ln1(x), batch))
        return ttnn.add(x, self.fc2(self.fc1(self.ln2(x))))


class TtPPOCRv6SmallRec:
    """Input NHWC [B, 48, W, 3] in [-1, 1]. Returns character probabilities [1, 1, B*W/8, NUM_CLASSES]."""

    def __init__(self, device, torch_model):
        self.device = device
        self.backbone = Backbone(device, torch_model.model.backbone, STAGE_CONFIGS)
        enc = torch_model.head.encoder

        def cbn(m):
            return fold_bn(m.convolution.weight, None, m.normalization)

        self.skip = Linear(device, *cbn(enc.conv_block[0]), activation="silu")
        self.reduce = Linear(device, *cbn(enc.conv_block[1]), activation="silu")
        self.local = Conv(device, *cbn(enc.conv_block[2]), groups=HIDDEN, activation=SILU)
        self.blocks = [Block(device, b) for b in enc.svtr_block]
        self.norm = LayerNorm(device, enc.norm)
        self.head = Linear(device, torch_model.head.head.weight, torch_model.head.head.bias)

    def __call__(self, x, batch, h, w):
        x, h, w = self.backbone(x, batch, h, w)[-1]
        x = ttnn.avg_pool2d(
            to_row_major(x),
            batch_size=batch,
            input_h=h,
            input_w=w,
            channels=x.shape[-1],
            kernel_size=[3, 2],
            stride=[3, 2],
            padding=[0, 0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        h, w = h // 3, w // 2

        skip = self.skip(x)
        x = self.reduce(x)
        local, _, _ = self.local(x, batch, h, w)
        x = ttnn.add(x, local)
        for blk in self.blocks:
            x = blk(x, batch)
        x = ttnn.add(self.norm(x), skip)
        logits = self.head(x)
        return ttnn.softmax(logits, dim=-1, numeric_stable=True)
