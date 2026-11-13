# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight loader skeleton for Hy4-Preview (CPU only; no ttnn).

* checkpoint key <-> logical name map (``logical_name`` / ``checkpoint_key``);
* expected shape and dtype of every tensor from Hy4Config (``expected``), checked against shard headers (``check_headers``);
* lazy reads through safetensors slices: nothing but the requested bytes is read (``load``, ``load_expert``);
* fused routed experts ``mlp.experts.gate_up_proj`` [256, 4096, 6144] and ``down_proj`` [256, 6144, 2048] split per expert.

UNVERIFIED (until the ref-oracle lands): the row order inside the fused gate_up_proj. ``GATE_UP_ORDER`` (env ``HY4_GATE_UP_ORDER``) selects it:
``gate_first`` (rows [0, 2048) = gate, [2048, 4096) = up; default), ``up_first``, or ``interleaved`` (gate = even rows, up = odd rows).
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Dict, Iterator, List, Optional, Tuple

from models.demos.blackhole.hy4_preview.config import DENSE, Hy4Config

DEFAULT_CKPT = "/mnt/tt-data/Hy4-Preview"
GATE_UP_ORDERS = ("gate_first", "up_first", "interleaved")
GATE_UP_ORDER = os.environ.get("HY4_GATE_UP_ORDER", "gate_first")

# checkpoint suffix (after "model.layers.<i>." or "model.mtp_layers.0.") -> logical suffix
_LAYER_SUFFIX = {
    "input_layernorm.weight": "norm.input",
    "post_attention_layernorm.weight": "norm.post_attn",
    "hc_attn_layer.hc_pre.hc_base": "hc.attn.base",
    "hc_attn_layer.hc_pre.hc_fn": "hc.attn.fn",
    "hc_attn_layer.hc_pre.hc_scale": "hc.attn.scale",
    "hc_mlp_layer.hc_pre.hc_base": "hc.mlp.base",
    "hc_mlp_layer.hc_pre.hc_fn": "hc.mlp.fn",
    "hc_mlp_layer.hc_pre.hc_scale": "hc.mlp.scale",
    "self_attn.q_a_proj.weight": "attn.q_a_proj",
    "self_attn.q_a_layernorm.weight": "attn.q_a_norm",
    "self_attn.q_b_proj.weight": "attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa.weight": "attn.kv_a_proj",
    "self_attn.kv_a_layernorm.weight": "attn.kv_a_norm",
    "self_attn.kv_b_proj.weight": "attn.kv_b_proj",
    "self_attn.linear_gate.weight": "attn.gate_proj",
    "self_attn.o_proj.weight": "attn.o_proj",
    "self_attn.learnable_sink_param": "attn.sink",
    "self_attn.indexer.wq_b.weight": "indexer.wq_b",
    "self_attn.indexer.wk.weight": "indexer.wk",
    "self_attn.indexer.weights_proj.weight": "indexer.weights_proj",
    "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
    "mlp.gate_proj.weight": "mlp.gate_proj",  # dense layer
    "mlp.up_proj.weight": "mlp.up_proj",
    "mlp.down_proj.weight": "mlp.down_proj",
    "mlp.gate.weight": "moe.router.weight",
    "mlp.gate.e_score_correction_bias": "moe.router.bias",
    "mlp.shared_experts.gate_proj.weight": "moe.shared.gate_proj",
    "mlp.shared_experts.up_proj.weight": "moe.shared.up_proj",
    "mlp.shared_experts.down_proj.weight": "moe.shared.down_proj",
    "mlp.experts.gate_up_proj": "moe.experts.gate_up",  # fused [E, 2*inter, hidden]
    "mlp.experts.down_proj": "moe.experts.down",  # [E, hidden, inter]
}
_MTP_ONLY_SUFFIX = {
    "eh_proj.weight": "eh_proj",
    "enorm.weight": "enorm",
    "hnorm.weight": "hnorm",
    "final_layernorm.weight": "final_norm",
}
_GLOBAL = {
    "model.embed_tokens.weight": "embed",
    "lm_head.weight": "lm_head",
    "model.norm.weight": "final_norm",
    "model.hc_head.hc_head_base": "hc_head.base",
    "model.hc_head.hc_head_fn": "hc_head.fn",
    "model.hc_head.hc_head_scale": "hc_head.scale",
}
_RE_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_RE_MTP = re.compile(r"^model\.mtp_layers\.(\d+)\.(.+)$")
_REV_GLOBAL = {v: k for k, v in _GLOBAL.items()}
_REV_LAYER = {v: k for k, v in _LAYER_SUFFIX.items()}
_REV_MTP = {v: k for k, v in _MTP_ONLY_SUFFIX.items()}


def logical_name(key: str) -> str:
    """'model.layers.3.self_attn.q_a_proj.weight' -> 'layers.3.attn.q_a_proj'; 'model.mtp_layers.0.eh_proj.weight' -> 'mtp.0.eh_proj'. KeyError if unknown."""
    if key in _GLOBAL:
        return _GLOBAL[key]
    m = _RE_LAYER.match(key)
    if m and m.group(2) in _LAYER_SUFFIX:
        return f"layers.{int(m.group(1))}.{_LAYER_SUFFIX[m.group(2)]}"
    m = _RE_MTP.match(key)
    if m:
        s = m.group(2)
        if s in _MTP_ONLY_SUFFIX:
            return f"mtp.{int(m.group(1))}.{_MTP_ONLY_SUFFIX[s]}"
        if s in _LAYER_SUFFIX:
            return f"mtp.{int(m.group(1))}.{_LAYER_SUFFIX[s]}"
    raise KeyError(f"unmapped checkpoint key {key!r}")


def checkpoint_key(logical: str) -> str:
    """Inverse of logical_name."""
    if logical in _REV_GLOBAL:
        return _REV_GLOBAL[logical]
    parts = logical.split(".", 2)
    if parts[0] == "layers" and len(parts) == 3 and parts[2] in _REV_LAYER:
        return f"model.layers.{int(parts[1])}.{_REV_LAYER[parts[2]]}"
    if parts[0] == "mtp" and len(parts) == 3:
        if parts[2] in _REV_MTP:
            return f"model.mtp_layers.{int(parts[1])}.{_REV_MTP[parts[2]]}"
        if parts[2] in _REV_LAYER:
            return f"model.mtp_layers.{int(parts[1])}.{_REV_LAYER[parts[2]]}"
    raise KeyError(f"unknown logical name {logical!r}")


# ---------------------------------------------------------------------------- expected inventory
def layer_logical_suffixes(cfg: Hy4Config, layer: int) -> List[str]:
    """Suffixes (after 'layers.<i>.' / 'mtp.0.') the checkpoint holds for a layer; the rules the tests verify against the real index."""
    spec = cfg.layer(layer)
    s = [
        "norm.input", "norm.post_attn",
        "attn.q_a_proj", "attn.q_a_norm", "attn.q_b_proj", "attn.kv_a_proj", "attn.kv_a_norm", "attn.kv_b_proj",
        "attn.gate_proj", "attn.o_proj", "attn.sink",
    ]  # fmt: skip
    if spec.has_hc:
        s += [f"hc.{a}.{b}" for a in ("attn", "mlp") for b in ("base", "fn", "scale")]
    if spec.has_indexer_weights:
        s += ["indexer.wq_b", "indexer.wk", "indexer.weights_proj", "indexer.k_norm.weight", "indexer.k_norm.bias"]
    if spec.mlp == DENSE:
        s += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    else:
        s += [
            "moe.router.weight", "moe.router.bias", "moe.shared.gate_proj", "moe.shared.up_proj", "moe.shared.down_proj",
            "moe.experts.gate_up", "moe.experts.down",
        ]  # fmt: skip
    if spec.is_mtp:
        s += ["eh_proj", "enorm", "hnorm", "final_norm"]
    return s


def _prefix(cfg: Hy4Config, layer: int) -> str:
    return f"mtp.{layer - cfg.num_hidden_layers}." if layer >= cfg.num_hidden_layers else f"layers.{layer}."


def layer_names(cfg: Hy4Config, layer: int) -> List[str]:
    p = _prefix(cfg, layer)
    return [p + s for s in layer_logical_suffixes(cfg, layer)]


def all_logical_names(cfg: Hy4Config) -> List[str]:
    out = list(_REV_GLOBAL)
    for i in range(cfg.n_total_layers):
        out += layer_names(cfg, i)
    return out


def expected(cfg: Hy4Config, logical: str) -> Tuple[str, Tuple[int, ...]]:
    """(safetensors dtype 'BF16'|'F32', shape) of a logical tensor, from the config."""
    H, I, Im, E = cfg.hidden_size, cfg.intermediate_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    glob = {
        "embed": ("BF16", (cfg.vocab_size, H)),
        "lm_head": ("BF16", (cfg.vocab_size, H)),
        "final_norm": ("BF16", (H,)),
    }
    if logical in glob:
        return glob[logical]
    if logical.startswith("hc_head."):
        raise KeyError("hc_head shapes are not derived (read from the header)")
    parts = logical.split(".", 2)
    suffix = parts[2]
    is_mtp = parts[0] == "mtp"
    f32 = "BF16" if is_mtp else "F32"  # the MTP layer stores sink and router bias as bf16
    table = {
        "norm.input": ("BF16", (H,)),
        "norm.post_attn": ("BF16", (H,)),
        "hc.attn.base": ("F32", (cfg.hc_fn_rows,)), "hc.mlp.base": ("F32", (cfg.hc_fn_rows,)),
        "hc.attn.fn": ("F32", (cfg.hc_fn_rows, cfg.hc_flat_dim)), "hc.mlp.fn": ("F32", (cfg.hc_fn_rows, cfg.hc_flat_dim)),
        "hc.attn.scale": ("F32", (2,)), "hc.mlp.scale": ("F32", (2,)),
        "attn.q_a_proj": ("BF16", (cfg.q_lora_rank, H)),
        "attn.q_a_norm": ("BF16", (cfg.q_lora_rank,)),
        "attn.q_b_proj": ("BF16", (cfg.q_b_out, cfg.q_lora_rank)),
        "attn.kv_a_proj": ("BF16", (cfg.kv_a_out, H)),
        "attn.kv_a_norm": ("BF16", (cfg.kv_lora_rank,)),
        "attn.kv_b_proj": ("BF16", (cfg.kv_b_out, cfg.kv_lora_rank)),
        "attn.gate_proj": ("BF16", (cfg.attn_out_dim, H)),
        "attn.o_proj": ("BF16", (H, cfg.attn_out_dim)),
        "attn.sink": (f32, (cfg.num_attention_heads,)),
        "indexer.wq_b": ("BF16", (cfg.indexer_q_out, cfg.q_lora_rank)),
        "indexer.wk": ("BF16", (cfg.index_head_dim, H)),
        "indexer.weights_proj": ("BF16", (cfg.index_n_heads, H)),
        "indexer.k_norm.weight": ("BF16", (cfg.index_head_dim,)),
        "indexer.k_norm.bias": ("BF16", (cfg.index_head_dim,)),
        "mlp.gate_proj": ("BF16", (I, H)), "mlp.up_proj": ("BF16", (I, H)), "mlp.down_proj": ("BF16", (H, I)),
        "moe.router.weight": ("BF16", (E, H)),
        "moe.router.bias": (f32, (E,)),
        "moe.shared.gate_proj": ("BF16", (Im, H)), "moe.shared.up_proj": ("BF16", (Im, H)), "moe.shared.down_proj": ("BF16", (H, Im)),
        "moe.experts.gate_up": ("BF16", (E, 2 * Im, H)),
        "moe.experts.down": ("BF16", (E, H, Im)),
        "eh_proj": ("BF16", (H, 2 * H)), "enorm": ("BF16", (H,)), "hnorm": ("BF16", (H,)), "final_norm": ("BF16", (H,)),
    }  # fmt: skip
    return table[suffix]


# ---------------------------------------------------------------------------- safetensors headers (no tensor bytes)
def read_header(path: str) -> Dict[str, dict]:
    """Parse a safetensors header with struct: {name: {'dtype', 'shape', 'data_offsets'}} (offsets relative to the data section). Reads 8 + header bytes only."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header size {n}")
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


class Hy4Weights:
    """Lazy view of the checkpoint: index and headers only until a tensor is asked for."""

    def __init__(
        self, ckpt_dir: Optional[str] = None, cfg: Optional[Hy4Config] = None, gate_up_order: Optional[str] = None
    ):
        self.ckpt_dir = ckpt_dir or os.environ.get("HY4_CKPT", DEFAULT_CKPT)
        self.cfg = cfg or Hy4Config.from_checkpoint(self.ckpt_dir)
        self.gate_up_order = gate_up_order or GATE_UP_ORDER
        if self.gate_up_order not in GATE_UP_ORDERS:
            raise ValueError(f"gate_up_order {self.gate_up_order!r} not in {GATE_UP_ORDERS}")
        with open(os.path.join(self.ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._headers: Dict[str, Dict[str, dict]] = {}

    # -- names
    def shard_of(self, logical: str) -> str:
        return os.path.join(self.ckpt_dir, self.weight_map[checkpoint_key(logical)])

    def header(self, logical: str) -> dict:
        shard = self.shard_of(logical)
        if shard not in self._headers:
            self._headers[shard] = read_header(shard)
        return self._headers[shard][checkpoint_key(logical)]

    def check_headers(self, logical_names: List[str]) -> List[str]:
        """Compare shard headers against config-derived expectations; returns a list of mismatch strings (empty = all good)."""
        bad = []
        for n in logical_names:
            try:
                dt, shape = expected(self.cfg, n)
            except KeyError:
                continue
            h = self.header(n)
            if h["dtype"] != dt or tuple(h["shape"]) != tuple(shape):
                bad.append(f"{n}: header {h['dtype']}{h['shape']} != expected {dt}{list(shape)}")
        return bad

    def unmapped_keys(self) -> List[str]:
        bad = []
        for k in self.weight_map:
            try:
                logical_name(k)
            except KeyError:
                bad.append(k)
        return bad

    # -- reads
    def _open(self, logical: str):
        from safetensors import safe_open

        return safe_open(self.shard_of(logical), framework="pt")

    def load(self, logical: str):
        """Whole tensor (do not use on moe.experts.*: 1.2 GB per layer; use load_expert)."""
        with self._open(logical) as f:
            return f.get_tensor(checkpoint_key(logical))

    def load_expert_fused(self, layer: int, e: int, which: str):
        """Slice of ONE expert out of the fused tensor ('gate_up' -> [4096, 6144], 'down' -> [6144, 2048]); only that slab is read."""
        name = f"{_prefix(self.cfg, layer)}moe.experts.{which}"
        if not 0 <= e < self.cfg.n_routed_experts:
            raise IndexError(e)
        with self._open(name) as f:
            return f.get_slice(checkpoint_key(name))[e : e + 1][0]

    def split_gate_up(self, gate_up):
        """[2*inter, hidden] -> (gate [inter, hidden], up [inter, hidden]) per ``self.gate_up_order`` (UNVERIFIED order, see module doc)."""
        two_i = gate_up.shape[0]
        if two_i % 2:
            raise ValueError(gate_up.shape)
        i = two_i // 2
        if self.gate_up_order == "gate_first":
            return gate_up[:i], gate_up[i:]
        if self.gate_up_order == "up_first":
            return gate_up[i:], gate_up[:i]
        return gate_up[0::2], gate_up[1::2]

    def load_expert(self, layer: int, e: int):
        """Per-expert (gate [2048, 6144], up [2048, 6144], down [6144, 2048]); reads two slabs of about 50 MB and 25 MB."""
        gate, up = self.split_gate_up(self.load_expert_fused(layer, e, "gate_up"))
        return gate, up, self.load_expert_fused(layer, e, "down")

    def iter_chip_experts(self, layer: int, chip: int) -> Iterator[Tuple[int, tuple]]:
        """Experts placed on ``chip`` (cfg.expert_chip), one at a time (lazy)."""
        for e in self.cfg.chip_experts(chip):
            yield e, self.load_expert(layer, e)
