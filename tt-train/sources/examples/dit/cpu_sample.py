# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sample a ttml DiT checkpoint on CPU via the PyTorch golden model.

    python cpu_sample.py --ckpt ckpt_ema_002000.npz --out grid.npy \
        [--dim 384 --depth 12 --heads 6 --patch 2] [--steps 50] [--cfg 2.0]

Loads tt-train .npz weights (names like DiT/blocks/0/attn/qkv/weight,
shapes [1,1,out,in]) into a torch DiT mirroring dit_ttml.py, then runs DDIM.
Doubles as a weight-level parity harness between the two implementations.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from reference_torch import DiffusionSchedule, sample_ddim


class TorchDiT(nn.Module):
    """Torch mirror of dit_ttml.DiT (affine LayerNorms, scale1p convention)."""

    def __init__(self, in_dim, dim, depth, num_heads, num_tokens, num_classes, mlp_ratio=4.0):
        super().__init__()
        self.dim, self.num_heads, self.num_classes = dim, num_heads, num_classes
        self.patch_proj = nn.Linear(in_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, num_tokens, dim))
        self.t_fc1, self.t_fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.label_emb = nn.Linear(num_classes + 1, dim, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            b = nn.Module()
            b.norm1 = nn.LayerNorm(dim, eps=1e-5)
            b.norm2 = nn.LayerNorm(dim, eps=1e-5)
            b.qkv, b.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
            b.fc1, b.fc2 = nn.Linear(dim, int(dim * mlp_ratio)), nn.Linear(int(dim * mlp_ratio), dim)
            b.branches = nn.ModuleList(nn.Linear(dim, dim) for _ in range(6))
            self.blocks.append(b)
        self.final_norm = nn.LayerNorm(dim, eps=1e-5)
        self.final_branches = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.final_proj = nn.Linear(dim, in_dim)

    @staticmethod
    def _mods(branches, c):
        s = torch.nn.functional.silu(c)
        return [br(s) for br in branches]

    def _attn(self, b, x):
        B, one, T, D = x.shape
        H, hd = self.num_heads, D // self.num_heads
        qkv = b.qkv(x).reshape(B, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return b.proj(out.transpose(1, 2).reshape(B, 1, T, D))

    def forward(self, tokens, t_feats, labels):
        onehot = torch.nn.functional.one_hot(labels, self.num_classes + 1).float()
        onehot = onehot.reshape(labels.shape[0], 1, 1, -1)
        x = self.patch_proj(tokens) + self.pos_emb
        c = self.t_fc2(torch.nn.functional.silu(self.t_fc1(t_feats))) + self.label_emb(onehot)
        for b in self.blocks:
            sh_a, sc1p_a, g_a, sh_m, sc1p_m, g_m = self._mods(b.branches, c)
            h = b.norm1(x) * sc1p_a + sh_a
            x = x + g_a * self._attn(b, h)
            h = b.norm2(x) * sc1p_m + sh_m
            x = x + g_m * b.fc2(torch.nn.functional.gelu(b.fc1(h), approximate="tanh"))
        f_sh, f_sc1p = self._mods(self.final_branches, c)
        x = self.final_norm(x) * f_sc1p + f_sh
        return self.final_proj(x)


def load_ttml_npz(model: TorchDiT, path: str):
    z = np.load(path)

    def W(name):  # [1,1,out,in] -> [out,in]
        return torch.from_numpy(z[name].reshape(z[name].shape[-2], z[name].shape[-1]).copy())

    def B(name):  # [1,1,1,out] -> [out]
        return torch.from_numpy(z[name].reshape(-1).copy())

    sd = {}
    sd["patch_proj.weight"], sd["patch_proj.bias"] = W("DiT/patch_proj/weight"), B("DiT/patch_proj/bias")
    sd["pos_emb"] = torch.from_numpy(z["DiT/pos_emb"].copy())
    sd["t_fc1.weight"], sd["t_fc1.bias"] = W("DiT/t_fc1/weight"), B("DiT/t_fc1/bias")
    sd["t_fc2.weight"], sd["t_fc2.bias"] = W("DiT/t_fc2/weight"), B("DiT/t_fc2/bias")
    sd["label_emb.weight"] = W("DiT/label_emb/weight")
    for i in range(len(model.blocks)):
        p = f"DiT/blocks/{i}"
        sd[f"blocks.{i}.norm1.weight"], sd[f"blocks.{i}.norm1.bias"] = B(f"{p}/norm1/gamma"), B(f"{p}/norm1/beta")
        sd[f"blocks.{i}.norm2.weight"], sd[f"blocks.{i}.norm2.bias"] = B(f"{p}/norm2/gamma"), B(f"{p}/norm2/beta")
        sd[f"blocks.{i}.qkv.weight"], sd[f"blocks.{i}.qkv.bias"] = W(f"{p}/attn/qkv/weight"), B(f"{p}/attn/qkv/bias")
        sd[f"blocks.{i}.proj.weight"], sd[f"blocks.{i}.proj.bias"] = W(f"{p}/attn/proj/weight"), B(f"{p}/attn/proj/bias")
        sd[f"blocks.{i}.fc1.weight"], sd[f"blocks.{i}.fc1.bias"] = W(f"{p}/mlp/fc1/weight"), B(f"{p}/mlp/fc1/bias")
        sd[f"blocks.{i}.fc2.weight"], sd[f"blocks.{i}.fc2.bias"] = W(f"{p}/mlp/fc2/weight"), B(f"{p}/mlp/fc2/bias")
        for j in range(6):
            sd[f"blocks.{i}.branches.{j}.weight"] = W(f"{p}/modulation/branches/{j}/weight")
            sd[f"blocks.{i}.branches.{j}.bias"] = B(f"{p}/modulation/branches/{j}/bias")
    sd["final_norm.weight"], sd["final_norm.bias"] = B("DiT/final_norm/gamma"), B("DiT/final_norm/beta")
    for j in range(2):
        sd[f"final_branches.{j}.weight"] = W(f"DiT/final_modulation/branches/{j}/weight")
        sd[f"final_branches.{j}.bias"] = B(f"DiT/final_modulation/branches/{j}/bias")
    sd["final_proj.weight"], sd["final_proj.bias"] = W("DiT/final_proj/weight"), B("DiT/final_proj/bias")

    missing, unexpected = model.load_state_dict(sd, strict=True), None
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_dim = args.patch * args.patch * args.channels
    num_tokens = (args.image_size // args.patch) ** 2
    model = TorchDiT(in_dim, args.dim, args.depth, args.heads, num_tokens, args.classes)
    load_ttml_npz(model, args.ckpt)
    model.eval()

    # sample_ddim expects the reference_torch DiT API surface; shim the bits it uses.
    model.pos_emb.data = model.pos_emb.data  # noop; pos_emb.shape[-1] read for t-features
    schedule = DiffusionSchedule()
    labels = torch.arange(args.classes)
    grid = sample_ddim(
        model, schedule, labels, (args.channels, args.image_size, args.image_size),
        patch=args.patch, steps=args.steps, cfg_scale=args.cfg, null_class=args.classes,
        generator=torch.Generator().manual_seed(args.seed),
    )
    np.save(args.out, grid.numpy())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
