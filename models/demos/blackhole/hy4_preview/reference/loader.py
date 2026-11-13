# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy checkpoint loader for Hy4-Preview (HYV4ForCausalLM).

Reads only the safetensors headers up front (a few KB per shard); tensors are read on demand
with ``safe_open(...).get_slice`` so one expert of the fused expert tensors can be read without
touching the other 255. Nothing here reads the whole 1.5 TB.

Fused expert tensors (per MoE layer, bf16):
    mlp.experts.gate_up_proj  [256, 4096, 6144]  = [E, 2*moe_inter, hidden]  (out, in per expert)
    mlp.experts.down_proj     [256, 6144, 2048]  = [E, hidden, moe_inter]
"""
import json
import os
import re
import struct
from dataclasses import dataclass, field

import torch

CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


@dataclass
class Hy4Config:
    raw: dict
    hidden: int = 6144
    n_layers: int = 78
    n_heads: int = 64
    q_lora: int = 2048
    kv_lora: int = 512
    nope: int = 192
    rope: int = 64
    v_dim: int = 256
    inter_dense: int = 18432
    inter_moe: int = 2048
    n_experts: int = 256
    topk: int = 8
    routed_scale: float = 2.827
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-5
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    rope_theta: float = 1e7
    idx_heads: int = 32
    idx_dim: int = 128
    idx_topk: int = 2048
    vocab: int = 120832
    indexer_types: list = field(default_factory=list)

    @classmethod
    def load(cls, ckpt=CKPT):
        c = json.load(open(os.path.join(ckpt, "config.json")))
        return cls(
            raw=c,
            hidden=c["hidden_size"],
            n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"],
            q_lora=c["q_lora_rank"],
            kv_lora=c["kv_lora_rank"],
            nope=c["qk_nope_head_dim"],
            rope=c["qk_rope_head_dim"],
            v_dim=c["v_head_dim"],
            inter_dense=c["intermediate_size"],
            inter_moe=c["moe_intermediate_size"],
            n_experts=c["n_routed_experts"],
            topk=c["num_experts_per_tok"],
            routed_scale=c["routed_scaling_factor"],
            swiglu_limit=c["swiglu_limit"],
            norm_eps=c["rms_norm_eps"],
            hc_mult=c["hc_mult"],
            hc_eps=c["hc_eps"],
            hc_magnitude=c["hc_magnitude"],
            rope_theta=c["rope_parameters"]["rope_theta"],
            idx_heads=c["index_n_heads"],
            idx_dim=c["index_head_dim"],
            idx_topk=c["index_topk"],
            vocab=c["vocab_size"],
            indexer_types=c["indexer_types"],
        )

    def is_dense(self, layer):
        return self.raw["mlp_layer_types"][layer] == "dense"

    def is_full_indexer(self, layer):
        return self.indexer_types[layer] == "full"


class Hy4Checkpoint:
    def __init__(self, ckpt=CKPT):
        self.dir = ckpt
        self.cfg = Hy4Config.load(ckpt)
        self.weight_map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self._hdr = {}

    def header(self, shard):
        if shard not in self._hdr:
            with open(os.path.join(self.dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._hdr[shard] = json.loads(fh.read(n))
        return self._hdr[shard]

    def meta(self, key):
        h = self.header(self.weight_map[key])[key]
        return _DT[h["dtype"]], tuple(h["shape"])

    def keys(self, prefix=""):
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def get(self, key, rows=None, dtype=None):
        """Read one tensor (or `rows` = index/slice along dim 0 for fused expert tensors) lazily."""
        from safetensors import safe_open

        with safe_open(os.path.join(self.dir, self.weight_map[key]), "pt") as f:
            if rows is None:
                t = f.get_tensor(key)
            else:
                t = f.get_slice(key)[rows]
        return t if dtype is None else t.to(dtype)

    def layer_prefix(self, layer):
        return f"model.layers.{layer}." if layer >= 0 else "model.mtp_layers.0."

    def layer_keys(self, layer):
        return self.keys(self.layer_prefix(layer))

    def layer_state(self, layer, dtype=torch.float32, experts=False):
        """Per-layer state dict with the prefix stripped. Fused expert tensors are NOT loaded
        unless experts=True (19 GB bf16): use expert_weights() for on-demand reads.
        layer=-1 -> the MTP layer."""
        p = self.layer_prefix(layer)
        sd = {}
        for k in self.layer_keys(layer):
            short = k[len(p) :]
            if short.startswith("mlp.experts."):
                if not experts:
                    continue
                sd[short] = self.get(k)
                continue
            t = self.get(k)
            sd[short] = t.to(dtype) if t.is_floating_point() else t
        return sd

    def expert_weights(self, layer, e, dtype=torch.float32):
        """(w_gate [I,H], w_up [I,H], w_down [H,I]) of routed expert e, split from the fused tensors.
        gate_up layout default = chunked [gate; up] along dim 1 (see MODEL_PROFILE NEEDS-CONFIRMATION)."""
        p = self.layer_prefix(layer) + "mlp.experts."
        gu = self.get(p + "gate_up_proj", rows=e).to(dtype)
        dn = self.get(p + "down_proj", rows=e).to(dtype)
        return split_gate_up(gu, GATE_UP_LAYOUT) + (dn,)

    def global_state(self, dtype=torch.float32):
        return {
            k: self.get(k).to(dtype) if self.meta(k)[0] != torch.float32 else self.get(k)
            for k in (
                "model.hc_head.hc_head_base",
                "model.hc_head.hc_head_fn",
                "model.hc_head.hc_head_scale",
                "model.norm.weight",
            )
        }

    def table(self, layers=(0, 1, 2, 3, -1)):
        rows = []
        for L in layers:
            for k in self.layer_keys(L):
                dt, sh = self.meta(k)
                rows.append((k, str(dt).replace("torch.", ""), list(sh)))
        for k in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.hc_head.hc_head_base",
            "model.hc_head.hc_head_fn",
            "model.hc_head.hc_head_scale",
        ):
            dt, sh = self.meta(k)
            rows.append((k, str(dt).replace("torch.", ""), list(sh)))
        return rows


GATE_UP_LAYOUT = os.environ.get("HY4_GATE_UP", "chunk")  # chunk | interleave | swap


def split_gate_up(gu, layout="chunk"):
    """gu [2I, H] -> (gate [I,H], up [I,H])."""
    if layout == "chunk":
        i = gu.shape[0] // 2
        return gu[:i], gu[i:]
    if layout == "swap":  # first half = up
        i = gu.shape[0] // 2
        return gu[i:], gu[:i]
    if layout == "interleave":
        return gu[0::2], gu[1::2]
    raise ValueError(layout)


if __name__ == "__main__":
    ck = Hy4Checkpoint()
    print(f"{'tensor':72s} {'dtype':9s} shape")
    for k, dt, sh in ck.table():
        print(f"{k:72s} {dt:9s} {sh}")
    print("full-indexer layers:", [i for i, t in enumerate(ck.cfg.indexer_types) if t == "full"])
    print(
        "layers with indexer weights in index:",
        sorted({int(m.group(1)) for k in ck.weight_map if (m := re.match(r"model\.layers\.(\d+)\..*indexer", k))}),
    )
