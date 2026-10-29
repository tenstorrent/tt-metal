# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch fp64 reference for the loss trajectory of ``train.py``.

Rebuilds the ttml model exactly (fused qkv / gate_up weights in ttml's interleaved RoPE row order, LoRA adapters
on the fused projections, the same AdamW) from the HF checkpoint, replays the token windows ``train.py`` logged
(``--data-log``) from the same LoRA init (``--dump-lora-init``), and writes the per-step losses. CPU, fp64, so a
step at 8 layers takes minutes; run it in the background.

    python sources/examples/mistral_train/torch_reference.py --layers 8 --data-log data.json \
        --lora-init lora_init.npz --out loss_torch.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ttml.common.utils import unpermute_proj_rows

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB, EPS, THETA = 5120, 32, 8, 128, 32768, 131072, 1e-5, 1e8
LINEARS = ("qkv_linear", "out_linear", "w_gate_up", "w2")

torch.set_default_dtype(torch.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="HF snapshot dir; default: the HF cache")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--data-log", required=True)
    p.add_argument("--lora-init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=0, help="0 = every logged step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--threads", type=int, default=16)
    return p.parse_args()


def checkpoint_dir(arg):
    if arg:
        return Path(arg)
    root = Path.home() / ".cache/huggingface/hub/models--mistralai--Mistral-Small-24B-Base-2501/snapshots"
    return sorted(root.glob("*/config.json"))[-1].parent


class Checkpoint:
    def __init__(self, directory: Path):
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {name: directory / file for name, file in index.items()}

    def __getitem__(self, name: str) -> torch.Tensor:
        with safe_open(str(self.files[name]), framework="pt") as f:
            return f.get_tensor(name).to(torch.float64)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float64)


def rope(x: torch.Tensor) -> torch.Tensor:
    """ttml's interleaved-pair RoPE on [B, H, T, D]."""
    T, D = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv_freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def rmsnorm(x, gamma):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma


class Layer:
    def __init__(self, ckpt: Checkpoint, i: int):
        pre = f"model.layers.{i}."
        q = unpermute_proj_rows(ckpt[pre + "self_attn.q_proj.weight"].numpy(), HEADS)
        k = unpermute_proj_rows(ckpt[pre + "self_attn.k_proj.weight"].numpy(), KV_HEADS)
        v = ckpt[pre + "self_attn.v_proj.weight"].numpy()
        self.w = {
            "qkv_linear": bf16(torch.from_numpy(np.concatenate([q, k, v], 0))),
            "out_linear": bf16(ckpt[pre + "self_attn.o_proj.weight"]),
            "w_gate_up": bf16(torch.cat([ckpt[pre + "mlp.gate_proj.weight"], ckpt[pre + "mlp.up_proj.weight"]], 0)),
            "w2": bf16(ckpt[pre + "mlp.down_proj.weight"]),
        }
        self.g1 = bf16(ckpt[pre + "input_layernorm.weight"])
        self.g2 = bf16(ckpt[pre + "post_attention_layernorm.weight"])
        self.lora = {}  # name -> (A, B) trainable

    def linear(self, name, x, scaling):
        a, b = self.lora[name]
        return x @ self.w[name].T + ((x @ a.T) @ b.T) * scaling

    def forward(self, x, scaling):
        B, T, _ = x.shape
        h = rmsnorm(x, self.g1)
        qkv = self.linear("qkv_linear", h, scaling)
        q, k, v = torch.split(qkv, [HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
        q = rope(q.view(B, T, HEADS, HEAD_DIM).transpose(1, 2))
        k = rope(k.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)).repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2).repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float("-inf"))
        merged = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, HEADS * HEAD_DIM)
        x = x + self.linear("out_linear", merged, scaling)
        m = rmsnorm(x, self.g2)
        gu = self.linear("w_gate_up", m, scaling)
        return x + self.linear("w2", torch.nn.functional.silu(gu[..., :INTER]) * gu[..., INTER:], scaling)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    ckpt = Checkpoint(checkpoint_dir(args.checkpoint))
    data = json.loads(Path(args.data_log).read_text())
    if args.steps:
        data = data[: args.steps]
    init = np.load(args.lora_init)
    scaling = args.alpha / args.rank

    embed = bf16(ckpt["model.embed_tokens.weight"])
    head = bf16(ckpt["lm_head.weight"])
    norm = bf16(ckpt["model.norm.weight"])
    layers = [Layer(ckpt, i) for i in range(args.layers)]
    params = []
    for i, layer in enumerate(layers):
        for name in LINEARS:
            module = "attention" if name in ("qkv_linear", "out_linear") else "mlp"
            key = f"LoraModel/model/blocks/{i}/{module}/{name}/"
            a = torch.tensor(init[key + "lora_A"][0, 0], dtype=torch.float64, requires_grad=True)
            b = torch.tensor(init[key + "lora_B"][0, 0], dtype=torch.float64, requires_grad=True)
            layer.lora[name] = (a, b)
            params += [a, b]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    print(f"torch fp64 reference: {args.layers} layers, {len(params)} LoRA tensors, {len(data)} steps")

    out = []
    for entry in data:
        t0 = time.time()
        ids = torch.tensor(entry["inputs"], dtype=torch.int64)
        targets = torch.tensor(entry["targets"], dtype=torch.int64)
        x = embed[ids]  # [B, T, H]
        for layer in layers:
            x = layer.forward(x, scaling)
        logits = rmsnorm(x, norm) @ head.T
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append({"step": entry["step"], "loss": loss.item(), "s": time.time() - t0})
        print(f"step {entry['step']:4d} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
        Path(args.out).write_text(json.dumps({"layers": args.layers, "steps": out}, indent=1))


if __name__ == "__main__":
    main()
