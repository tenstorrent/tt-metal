# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fp32 reference of MiMo's DFlash draft model (``dflash/`` beside the checkpoint), as vLLM's ``qwen3_dflash`` runs it.

DFlash drafts a block in one non-causal pass. Its context is the target's residual stream after layers
``target_layer_ids`` (0, 11, 23, 35, 47: vLLM's aux hidden states at layer id + 1, hidden + residual, before the final
norm), concatenated per position: ``hidden_norm(fc(features))`` is projected once per committed position into every
draft layer's K / V (k_norm, then RoPE at that position). A draft is the block [anchor, mask x (BLOCK - 1)] at
positions [start, start + BLOCK): the anchor is the last committed token (the target's embedding), the masks the
checkpoint's separate ``mask_embedding``. Each of the 5 layers is a Qwen3 block (q_norm / k_norm per head, RoPE on the
first ``rope_dim`` of 128 dims, neox pairing, theta 10000; attention sinks; a dense SiLU MLP) whose queries attend,
non-causally, to the context within the sliding window and to the whole block. The final norm feeds the target's LM
head; the greedy tokens of block rows 1 .. BLOCK - 1 are the drafts.

``ReferenceDFlash`` keeps the context K / V of one request (the last ``window`` positions).
"""

import json
import os

import torch
from safetensors import safe_open


class DFlashConfig:
    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        d = c["dflash_config"]
        self.path = path
        self.hidden_size = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        self.num_heads = c["num_attention_heads"]
        self.num_kv_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.intermediate_size = c["intermediate_size"]
        self.eps = c["rms_norm_eps"]
        self.window = c["sliding_window"]
        self.block = c["block_size"]
        self.rope_dim = int(self.head_dim * c["partial_rotary_factor"])
        self.rope_theta = c["rope_theta"]
        self.target_layer_ids = list(d["target_layer_ids"])
        self.mask_token_id = d["mask_token_id"]

    @classmethod
    def from_model_path(cls, model_path):
        return cls(os.path.join(model_path, "dflash"))


def load_dflash_weights(cfg):
    """{name: fp32 tensor} of the draft model, plus ``mask_embedding`` [hidden]."""
    out = {}
    with safe_open(os.path.join(cfg.path, "dflash_draft_model.safetensors"), "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float()
    mask = torch.load(os.path.join(cfg.path, "mask_embedding.pt"), map_location="cpu")
    assert mask["mask_token_id"] == cfg.mask_token_id, (mask["mask_token_id"], cfg.mask_token_id)
    out["mask_embedding"] = mask["embedding"].float()
    return out


def rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope(x, positions, rope_dim, theta):
    """x [n, heads, head_dim], positions [n]: neox RoPE (pairs i, i + rope_dim / 2) on the first ``rope_dim`` dims."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    f = positions.double()[:, None] * inv_freq[None, :]
    emb = torch.cat([f, f], dim=-1)
    cos, sin = emb.cos().float()[:, None, :], emb.sin().float()[:, None, :]
    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    x1, x2 = xr[..., : rope_dim // 2], xr[..., rope_dim // 2 :]
    return torch.cat([xr * cos + torch.cat([-x2, x1], dim=-1) * sin, xp], dim=-1)


class ReferenceDFlash:
    def __init__(self, cfg, weights, embed, lm_head, window_from_last=False):
        """``embed`` / ``lm_head`` [vocab, hidden]: the target's. ``window_from_last``: every block row's window
        starts BLOCK - 1 positions after its own (measured from the block's last position), as the device's paged
        decode attention does with one ``cur_pos`` for the whole block."""
        self.cfg, self.w, self.embed, self.lm_head = cfg, weights, embed, lm_head
        self.window_from_last = window_from_last
        c = cfg
        self.k = [torch.zeros(0, c.num_kv_heads, c.head_dim) for _ in range(c.num_layers)]
        self.v = [torch.zeros(0, c.num_kv_heads, c.head_dim) for _ in range(c.num_layers)]
        self.pos = torch.zeros(0, dtype=torch.long)

    def _l(self, i, name):
        return self.w[f"layers.{i}.{name}"]

    def context_hidden(self, features):
        """features [n, len(target_layer_ids), hidden] -> hidden_norm(fc(cat)) [n, hidden]."""
        f = features.float().reshape(features.shape[0], -1)
        return rms(f @ self.w["fc.weight"].T, self.w["hidden_norm.weight"], self.cfg.eps)

    def add_context(self, features, positions):
        """Project the context of ``positions`` [n] (committed, in order) into every layer's K / V."""
        c = self.cfg
        h = self.context_hidden(features)
        keep = c.window + c.block
        for i in range(c.num_layers):
            k = (h @ self._l(i, "self_attn.k_proj.weight").T).view(-1, c.num_kv_heads, c.head_dim)
            k = rope(rms(k, self._l(i, "self_attn.k_norm.weight"), c.eps), positions, c.rope_dim, c.rope_theta)
            v = (h @ self._l(i, "self_attn.v_proj.weight").T).view(-1, c.num_kv_heads, c.head_dim)
            self.k[i] = torch.cat([self.k[i], k])[-keep:]
            self.v[i] = torch.cat([self.v[i], v])[-keep:]
        self.pos = torch.cat([self.pos, positions])[-keep:]

    def block_hidden(self, anchor, start):
        """The final normed hidden [BLOCK, hidden] of the block [anchor, mask x (BLOCK - 1)] at ``start``."""
        c = self.cfg
        pos = torch.arange(start, start + c.block)
        x = torch.cat([self.embed[anchor][None].float(), self.w["mask_embedding"][None].expand(c.block - 1, -1)])
        group = c.num_heads // c.num_kv_heads
        for i in range(c.num_layers):
            h = rms(x, self._l(i, "input_layernorm.weight"), c.eps)
            q = (h @ self._l(i, "self_attn.q_proj.weight").T).view(c.block, c.num_heads, c.head_dim)
            q = rope(rms(q, self._l(i, "self_attn.q_norm.weight"), c.eps), pos, c.rope_dim, c.rope_theta)
            k = (h @ self._l(i, "self_attn.k_proj.weight").T).view(c.block, c.num_kv_heads, c.head_dim)
            k = rope(rms(k, self._l(i, "self_attn.k_norm.weight"), c.eps), pos, c.rope_dim, c.rope_theta)
            v = (h @ self._l(i, "self_attn.v_proj.weight").T).view(c.block, c.num_kv_heads, c.head_dim)
            keys, vals = torch.cat([self.k[i], k]), torch.cat([self.v[i], v])
            kpos = torch.cat([self.pos, pos])
            qpos = torch.full_like(pos, start + c.block - 1) if self.window_from_last else pos
            allowed = kpos[None, :] > qpos[:, None] - c.window  # non-causal inside the block
            kh, vh = keys.repeat_interleave(group, dim=1), vals.repeat_interleave(group, dim=1)
            s = torch.einsum("qhd,khd->hqk", q, kh) * c.head_dim**-0.5
            s = s.masked_fill(~allowed[None], float("-inf"))
            sink = self._l(i, "self_attn.attention_sink_bias")[:, None, None].expand(c.num_heads, c.block, 1)
            p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
            o = torch.einsum("hqk,khd->qhd", p, vh).reshape(c.block, -1)
            x = x + o @ self._l(i, "self_attn.o_proj.weight").T
            h = rms(x, self._l(i, "post_attention_layernorm.weight"), c.eps)
            g = torch.nn.functional.silu(h @ self._l(i, "mlp.gate_proj.weight").T)
            x = x + (g * (h @ self._l(i, "mlp.up_proj.weight").T)) @ self._l(i, "mlp.down_proj.weight").T
        return rms(x, self.w["norm.weight"], c.eps)

    def draft(self, anchor, start, k=None):
        """The greedy drafts (first ``k``, default BLOCK - 1) after ``anchor`` at ``start``."""
        k = self.cfg.block - 1 if k is None else k
        h = self.block_hidden(anchor, start)[1 : 1 + k]
        return (h @ self.lm_head.float().T).argmax(-1).tolist()
