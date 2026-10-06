# SPDX-License-Identifier: Apache-2.0
"""Layer-only reference for the pinned Kolibri checkpoint.

Transformers has no Kolibri modeling module. This follows Aleph Alpha's
Kolibri1DecoderLayer at 049a6a7bd2405b27d6d280d256bd3d585191c7ae, expressing
its fused residual and MoE operations with ordinary PyTorch operations.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

REVISION = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
MODEL_ID = "Aleph-Alpha/Kolibri-1-BF16"
SNAPSHOT = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--Aleph-Alpha--Kolibri-1-BF16/snapshots"
    / REVISION
)


class Kolibri1Config(Qwen3MoeConfig):
    model_type = "kolibri1"


AutoConfig.register("kolibri1", Kolibri1Config, exist_ok=True)


def config():
    return AutoConfig.from_pretrained(Path(__file__).parent, local_files_only=True)


def load_weights(layer_idx, *, stats_path=None, synthetic=False):
    prefix = f"model.layers.{layer_idx}."
    stats = {}
    if synthetic:
        stats = json.loads(Path(stats_path).read_text())
        gen = torch.Generator().manual_seed(2107 + layer_idx)
        return {
            k: (torch.randn(v["shape"], generator=gen) * v["std"] + v["mean"]).to(getattr(torch, v["dtype"]))
            for k, v in stats.items()
        }
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith(prefix)]
    collect_stats = stats_path and not Path(stats_path).exists()
    result = {}
    for shard in sorted({index[k] for k in keys}):
        with safe_open(SNAPSHOT / shard, framework="pt", device="cpu") as f:
            for k in keys:
                if index[k] != shard:
                    continue
                v = f.get_tensor(k)
                name = k[len(prefix) :]
                result[name] = v
                if collect_stats:
                    vf = v.float()
                    stats[name] = dict(
                        shape=list(v.shape),
                        dtype=str(v.dtype).split(".")[-1],
                        mean=vf.mean().item(),
                        std=vf.std().item(),
                    )
    if collect_stats:
        Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n")
    return result


def norm(x, w, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope(x, positions):
    inv = 1.0 / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
    phase = positions.float()[..., None] * inv
    phase = torch.cat([phase, phase], dim=-1)[:, None]
    rot = torch.cat([-x[..., 64:], x[..., :64]], dim=-1)
    return (x.float() * phase.cos() + rot.float() * phase.sin()).to(x.dtype)


class ReferenceDecoder:
    def __init__(self, weights, layer_idx):
        self.w, self.layer_idx = weights, layer_idx
        self.sliding = config().layer_types[layer_idx] == "sliding_attention"
        self.cache = None

    def reset(self):
        self.cache = None

    def attention(self, x, start):
        w = self.w
        b, s, _ = x.shape
        q = F.linear(x, w["self_attn.q_proj.weight"]).view(b, s, 48, 128).transpose(1, 2)
        k = F.linear(x, w["self_attn.k_proj.weight"]).view(b, s, 4, 128).transpose(1, 2)
        v = F.linear(x, w["self_attn.v_proj.weight"]).view(b, s, 4, 128).transpose(1, 2)
        q = norm(q, w["self_attn.q_norm.weight"])
        k = norm(k, w["self_attn.k_norm.weight"])
        if self.sliding:
            pos = torch.arange(start, start + s).expand(b, -1)
            q = rope(q, pos)
            k = rope(k, pos)
        if self.cache is not None:
            k = torch.cat([self.cache[0][:, :, :start], k], dim=2)
            v = torch.cat([self.cache[1][:, :, :start], v], dim=2)
        self.cache = (k, v)
        # Query tiling bounds CPU reference memory without changing attention semantics.
        outputs = []
        for off in range(0, s, 128):
            n = min(128, s - off)
            end = start + off + n
            left = max(0, start + off - 512) if self.sliding else 0
            rows = torch.arange(start + off, end)[:, None]
            cols = torch.arange(left, end)[None, :]
            mask = cols <= rows
            if self.sliding:
                mask &= cols >= rows - 512
            y = F.scaled_dot_product_attention(
                q[:, :, off : off + n].float(),
                k[:, :, left:end].float(),
                v[:, :, left:end].float(),
                attn_mask=mask,
                enable_gqa=True,
            )
            outputs.append(y.to(x.dtype))
        y = torch.cat(outputs, dim=2).transpose(1, 2).reshape(b, s, 6144)
        return F.linear(y, w["self_attn.o_proj.weight"])

    def moe(self, x):
        w = self.w
        shape = x.shape
        x = x.reshape(-1, 2560)
        logits = F.linear(x.float(), w["mlp.gate.weight"].float())
        ids = (logits + w["moe.router.expert_bias"].float()).topk(6, dim=-1).indices
        scores = logits.gather(-1, ids).sigmoid()
        y = torch.zeros_like(x)
        for e in ids.unique().tolist():
            rows, slots = torch.where(ids == e)
            xe = x[rows]
            base = f"mlp.experts.{e}."
            ye = F.linear(
                F.silu(F.linear(xe, w[base + "gate_proj.weight"])) * F.linear(xe, w[base + "up_proj.weight"]),
                w[base + "down_proj.weight"],
            )
            y.index_add_(0, rows, (ye.float() * scores[rows, slots, None]).to(x.dtype))
        base = "mlp.shared_experts."
        shared = F.linear(
            F.silu(F.linear(x, w[base + "gate_proj.weight"])) * F.linear(x, w[base + "up_proj.weight"]),
            w[base + "down_proj.weight"],
        )
        return (y + shared).reshape(shape)

    def __call__(self, x, *, start=0):
        w = self.w
        y = x + norm(self.attention(norm(x, w["input_layernorm.weight"]), start), w["post_attn_norm.weight"])
        return y + norm(self.moe(norm(y, w["post_attention_layernorm.weight"])), w["post_ffn_norm.weight"])
