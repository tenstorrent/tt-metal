# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side check of the vision tower's device layout, no device needed.

Runs the TP-sharded computation TtVisionTower performs in fp32 torch, with the exact weights prepare_weights
uploads, the interleaved rotary tables and the padded attention windows, and compares it to the reference. Catches a
wrong q/k permutation, qkv interleave, merge-row reorder or window plan before any device run.

    HF_MODEL=/mnt/models/MiniMax-M3 pytest models/demos/minimax_m3/tests/unit/test_vision_layout.py
"""

import dataclasses
import os

import pytest
import torch
import torch.nn.functional as F

from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionReference, VisionArgs, load_vision_state_dict
from models.demos.minimax_m3.tt.vision import padded_head_dim, plan_batch, prepare_weights, rope_tables

CKPT = os.getenv("HF_MODEL")
LAYERS = 2


def _rot_interleaved(x):
    out = torch.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


def emulate(w, args: VisionArgs, pixel_values, grid, sp: int, tp: int):
    """TtVisionTower.forward in fp32 torch, one TP shard at a time, over the full padded sequence."""
    batch = plan_batch(grid, args, sp)
    hd = padded_head_dim(args.head_dim)
    local_heads = args.heads // tp
    x = torch.zeros(batch.padded, w["patch.weight"][0].shape[0])
    x[: batch.patches, : args.patch_elems] = pixel_values.float()
    cos, sin = rope_tables(batch.segments, args, batch.padded)
    cu = [c for i, c in enumerate(batch.cu_windows()) if i == 0 or c != batch.cu_windows()[i - 1]]
    v_ = lambda n: w[n][0].reshape(-1)
    shard = lambda t, n, dim: t.chunk(tp, dim=dim)[n]
    ln = lambda t, name: F.layer_norm(t, (args.hidden,), v_(name + ".weight"), v_(name + ".bias"), args.eps)

    x = ln(x @ w["patch.weight"][0], "pre_ln")
    for i in range(args.layers):
        p = f"l{i}"
        h = ln(x, p + ".ln1")
        attn_sum = 0
        for c in range(tp):
            qkv = h @ shard(w[p + ".qkv.weight"][0], c, 1) + shard(v_(p + ".qkv.bias"), c, 0)
            q, k, v = qkv.reshape(batch.padded, 3, local_heads, hd).unbind(1)
            q = q * cos[:, None] + _rot_interleaved(q) * sin[:, None]
            k = k * cos[:, None] + _rot_interleaved(k) * sin[:, None]
            o = torch.zeros_like(q)
            for s0, s1 in zip(cu[:-1], cu[1:]):
                qs, ks, vs = (t[s0:s1].transpose(0, 1) for t in (q, k, v))
                o[s0:s1] = F.scaled_dot_product_attention(qs, ks, vs, scale=args.head_dim**-0.5).transpose(0, 1)
            attn_sum = attn_sum + o.reshape(batch.padded, -1) @ shard(w[p + ".o.weight"][0], c, 0)
        x = x + attn_sum + v_(p + ".o.bias")
        h = ln(x, p + ".ln2")
        mlp_sum = 0
        for c in range(tp):
            f = F.gelu(h @ shard(w[p + ".fc1.weight"][0], c, 1) + shard(v_(p + ".fc1.bias"), c, 0))
            mlp_sum = mlp_sum + f @ shard(w[p + ".fc2.weight"][0], c, 0)
        x = x + mlp_sum + v_(p + ".fc2.bias")

    z = sum(
        F.gelu(x @ shard(w["proj1.weight"][0], c, 1) + shard(v_("proj1.bias"), c, 0))
        @ shard(w["proj2.weight"][0], c, 0)
        for c in range(tp)
    )
    z_c = [shard(z, c, 1) + shard(v_("proj2.bias"), c, 0) for c in range(tp)]
    m = sum(z_c[c].reshape(batch.padded // 4, -1) @ shard(w["merge1.weight"][0], c, 0) for c in range(tp))
    m = F.gelu(m + v_("merge1.bias"))
    y = sum(shard(m, c, 1) @ shard(w["merge2.weight"][0], c, 0) for c in range(tp)) + v_("merge2.bias")
    return y[: batch.rows]


@pytest.mark.skipif(CKPT is None, reason="set HF_MODEL to the MiniMax-M3 checkpoint")
@pytest.mark.parametrize("sp,tp", [(4, 4), (1, 4), (4, 1)])
def test_device_layout_matches_reference(sp, tp):
    torch.manual_seed(0)
    args = dataclasses.replace(VisionArgs.from_checkpoint(CKPT), layers=LAYERS)
    keep = lambda n: "encoder.layers." not in n or int(n.split("encoder.layers.")[1].split(".")[0]) < LAYERS
    state = {n: t for n, t in load_vision_state_dict(CKPT).items() if keep(n)}
    # two images and a video longer than one attention segment
    grid = [(1, 8, 12), (1, 4, 6), (args.segment_max_frames + 2, 4, 4)]
    patches = sum(t * h * w for t, h, w in grid)
    pv = torch.randn(patches, args.patch_elems) * 0.5

    ref = MiniMaxM3VisionReference(state, args)(pv, grid)
    got = emulate(prepare_weights(state, args, tp), args, pv, grid, sp, tp)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    passing, pcc = comp_pcc(ref, got, 0.99999)
    assert passing, pcc
