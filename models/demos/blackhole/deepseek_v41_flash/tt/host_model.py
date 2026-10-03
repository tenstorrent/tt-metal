# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side parts of the DeepSeek-V4.1-Flash decode step that are not transformer blocks.

Run on the CPU with torch, using the checkpoint's own code from ``inference/model.py`` / ``engram.py``:

  * ``HostEmbedding``: token ids -> the 4 identical residual streams (+ identity pre_mix).
  * ``HostEngram``: the n-gram hash-memory write at layers 1 and 14. The tables are ~100 GB each, so rows are
    read lazily from the safetensors (a token needs 24 of them) instead of materialising the table.
  * ``HostHead``: collapse the streams with the last layer's ``pre``, final RMSNorm, LM head, argmax.
"""

import os

import torch

_RD = torch.float32 if os.environ.get("REF_DTYPE", "bf16") == "fp32" else torch.bfloat16
from safetensors import safe_open

from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

CKPT = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")


class HostEmbedding:
    def __init__(self):
        sh = _Shards()
        self.weight = sh.get("embed.weight")  # [129280, 5120] bf16

    def __call__(self, token_ids: torch.Tensor):
        mod = R.load_model_module()
        h = self.weight[token_ids].to(_RD).unsqueeze(2).repeat(1, 1, 4, 1)
        return h, mod.make_identity_pre_mix(h, 4)


class _LazyEngramTable(torch.nn.Module):
    """Stands in for ``ParallelEngramEmbedding``: fp8 rows + e8m0 scales are read per lookup."""

    def __init__(self, sh: _Shards, layer_id: int):
        super().__init__()
        self.sh, self.wkey, self.skey = (
            sh,
            f"layers.{layer_id}.engram.embed.weight",
            f"layers.{layer_id}.engram.embed.scale",
        )
        path = os.path.join(sh.dir, sh.index[self.wkey])
        self.f = safe_open(path, "pt")
        self.w, self.s = self.f.get_slice(self.wkey), self.f.get_slice(self.skey)
        self.block = 32

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).tolist()
        rows = torch.cat([self.w[i : i + 1] for i in flat])  # [n, 256] fp8
        scales = torch.cat([self.s[i : i + 1] for i in flat])  # [n, 8] e8m0
        v = rows.float().unflatten(-1, (-1, self.block)) * scales.float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).reshape(*indices.shape, -1)


class HostEngram:
    """Engram layers (1 and 14). ``apply(layer_id, h, input_ids, start_pos)`` returns the updated streams."""

    def __init__(self, layer_ids=(1, 14), max_batch_size=16, max_seq_len=256):
        from transformers import AutoTokenizer  # tokenizer.json ships with the checkpoint

        mod = R.load_model_module()
        args = R.model_args(max_batch_size, max_seq_len)  # the hash cache is sized by these
        self.layout = mod.EngramLayout.from_args(args)
        tok = AutoTokenizer.from_pretrained(CKPT)
        self.hash = mod.NgramHashState(args, self.layout, tok)
        sh = _Shards()
        self.mods = {}
        for lid in layer_ids:
            e = mod.Engram(args, lid, self.layout)
            e.embed = _LazyEngramTable(sh, lid)  # drop the (virtual) 98 GB table
            with torch.no_grad():
                e.wkv.weight.data = sh.get(f"layers.{lid}.engram.wkv.weight")
                e.wkv.scale.data = sh.get(f"layers.{lid}.engram.wkv.scale")
                e.q_weight.data = sh.get(f"layers.{lid}.engram.q_weight").float()
                e.k_weight.data = sh.get(f"layers.{lid}.engram.k_weight").float()
            e.eval()
            self.mods[lid] = e

    @torch.no_grad()
    def hashes(self, input_ids, start_pos=0):
        return self.hash(input_ids, start_pos, None)

    @torch.no_grad()
    def apply(self, layer_id, h, hashes):
        e = self.mods[layer_id]
        return e(h, hashes[:, :, e.layer_hash_index, :], None)


class HostHead:
    def __init__(self):
        sh = _Shards()
        mod = R.load_model_module()
        self.eps = R.model_args().norm_eps
        self.norm_w = sh.get("norm.weight").float()
        self.head_w = sh.get("head.weight").float()  # [129280, 5120]
        self._mod = mod

    @torch.no_grad()
    def __call__(self, h, pre_mix):
        """h [B, 1, 4, D] streams after the last layer, pre_mix [B, 1, 4] (the last layer's ffn `pre`) -> logits [B, vocab]."""
        y = torch.sum(pre_mix.float().unsqueeze(-1) * h.float(), dim=2).to(_RD)  # hc_pre -> [B,1,D]
        x = y.float()
        x = (self.norm_w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps))).to(_RD)
        return torch.nn.functional.linear(x[:, -1].float(), self.head_w)
