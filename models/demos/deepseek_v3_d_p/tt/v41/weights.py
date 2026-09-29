# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side preparation of DeepSeek-V4.1-Flash checkpoint weights (F0).

Reads the raw HF shards by tensor name and returns exact bf16/fp32 torch tensors in the checkpoint's
``[out, in]`` orientation, keyed the way the V4.1 device modules consume them. Device materialization
(dtype, layout, mesh mapping, tensor caches) belongs to the modules that own the placement.

Checkpoint formats (HF ``deepseek-ai/DeepSeek-V4.1-Flash`` at ``CHECKPOINT_REVISION``):

* dense projections: ``F8_E4M3`` weight with one ``F8_E8M0`` scale per 32x32 block;
* routed experts: MXFP4 -- ``I8`` bytes packing two E2M1 codes (low nibble = even element), one ``F8_E8M0``
  scale per 32 consecutive elements along the input dimension;
* norms, compressor/indexer bf16 projections, gate weight: ``BF16``; attn_sink, gate biases, hc_*: ``F32``.

Both dequantizations are exact in bf16: an E4M3 (3 mantissa bits) or E2M1 (1 mantissa bit) value times a
power of two fits bf16's 7-bit mantissa, so the returned tensors carry no rounding.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig

CHECKPOINT_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
CHECKPOINT_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
# Git blob id of model.safetensors.index.json at CHECKPOINT_REVISION (the HF cache names the blob by it).
CHECKPOINT_INDEX_BLOB = "54c85064dd92c8550471302e9ae59bedbbf96ca4"
CHECKPOINT_ENV = "TT_V41_CHECKPOINT"
DEFAULT_CHECKPOINT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--deepseek-ai--DeepSeek-V4.1-Flash"
    / "snapshots"
    / CHECKPOINT_REVISION
)
INDEX_FILE = "model.safetensors.index.json"

FP8_BLOCK = 32  # 32x32 weight blocks per E8M0 scale
FP4_BLOCK = 32  # 32 input elements per E8M0 scale
EXPERT_READ_BATCH = 32  # routed experts dequantized per shard open (bounds the raw bytes held at once)

# E2M1 value by 4-bit code: bit 3 is the sign, bits 0-2 index the magnitude grid.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = torch.tensor(_E2M1_MAGNITUDES + tuple(-m for m in _E2M1_MAGNITUDES), dtype=torch.float32)
# (even, odd) element values by packed byte: one gather per byte instead of one per nibble.
_E2M1_PAIRS = torch.stack((E2M1_VALUES[torch.arange(256) & 0x0F], E2M1_VALUES[torch.arange(256) >> 4]), dim=-1)
_E8M0_NAN = 0xFF


def _check_scales(scale: torch.Tensor, name: str) -> torch.Tensor:
    if scale.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"{name}: scale dtype {scale.dtype}, expected float8_e8m0fnu")
    if (scale.view(torch.uint8) == _E8M0_NAN).any():
        raise ValueError(f"{name}: scale holds the E8M0 NaN encoding")
    return scale.float()


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, name: str = "weight") -> torch.Tensor:
    """``F8_E4M3 [out, in]`` with ``F8_E8M0 [out/32, in/32]`` block scales -> exact bf16 ``[out, in]``."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: weight dtype {weight.dtype}, expected float8_e4m3fn")
    out_dim, in_dim = weight.shape
    if out_dim % FP8_BLOCK or in_dim % FP8_BLOCK:
        raise ValueError(f"{name}: shape {tuple(weight.shape)} is not a whole number of {FP8_BLOCK}x{FP8_BLOCK} blocks")
    blocks = (out_dim // FP8_BLOCK, in_dim // FP8_BLOCK)
    if tuple(scale.shape) != blocks:
        raise ValueError(f"{name}: scale shape {tuple(scale.shape)}, expected {blocks}")
    s = _check_scales(scale, name)
    w = weight.float().view(blocks[0], FP8_BLOCK, blocks[1], FP8_BLOCK)
    return (w * s[:, None, :, None]).view(out_dim, in_dim).to(torch.bfloat16)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, name: str = "weight") -> torch.Tensor:
    """MXFP4 ``[out, in/2]`` bytes with ``F8_E8M0 [out, in/32]`` scales -> exact bf16 ``[out, in]``.

    Byte ``j`` of a row holds element ``2j`` in its low nibble and ``2j + 1`` in its high nibble.
    """
    if packed.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2):
        raise ValueError(f"{name}: packed dtype {packed.dtype}, expected int8/uint8/float4_e2m1fn_x2")
    out_dim, half_in = packed.shape
    in_dim = 2 * half_in
    if in_dim % FP4_BLOCK:
        raise ValueError(f"{name}: input dim {in_dim} is not a multiple of {FP4_BLOCK}")
    if tuple(scale.shape) != (out_dim, in_dim // FP4_BLOCK):
        raise ValueError(f"{name}: scale shape {tuple(scale.shape)}, expected {(out_dim, in_dim // FP4_BLOCK)}")
    s = _check_scales(scale, name)
    values = _E2M1_PAIRS[packed.view(torch.uint8).long()].view(out_dim, in_dim // FP4_BLOCK, FP4_BLOCK) * s[:, :, None]
    return values.view(out_dim, in_dim).to(torch.bfloat16)


def index_blob_id(index_path: Path) -> str:
    """Git blob id of a file (``sha1("blob <size>\\0" + bytes)``), the id the HF cache stores it under."""
    data = index_path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class V41Checkpoint:
    """A checkpoint directory: its index and shard-grouped reads by tensor name.

    Does not verify the revision; ``resolve_checkpoint`` does.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        with (self.root / INDEX_FILE).open(encoding="utf-8") as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def read(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Raw stored tensors for ``names``, opening each shard once (one shard holds a whole layer)."""
        missing = [name for name in names if name not in self.weight_map]
        if missing:
            raise KeyError(f"{self.root} index is missing {len(missing)} tensors, first: {missing[:4]}")
        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        tensors = {}
        for shard, keys in by_shard.items():
            with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensors[key] = handle.get_tensor(key)
        return tensors


def resolve_checkpoint() -> V41Checkpoint | None:
    """The pinned checkpoint from ``$TT_V41_CHECKPOINT``, else the HF cache snapshot; ``None`` if neither exists.

    Raises if the directory's index is not the one at ``CHECKPOINT_REVISION``.
    """
    override = os.environ.get(CHECKPOINT_ENV)
    root = Path(override) if override else DEFAULT_CHECKPOINT
    index = root / INDEX_FILE
    if not index.is_file():
        return None
    blob = index_blob_id(index)
    if blob != CHECKPOINT_INDEX_BLOB:
        raise ValueError(
            f"{index} has blob id {blob}; {CHECKPOINT_REPO}@{CHECKPOINT_REVISION} has {CHECKPOINT_INDEX_BLOB}"
        )
    return V41Checkpoint(root)


# --- layer schema ----------------------------------------------------------------------------------

FP8, BF16, F32 = "fp8", "bf16", "f32"
_STORED_DTYPE = {BF16: torch.bfloat16, F32: torch.float32}


@dataclass(frozen=True)
class _Param:
    key: tuple[str, ...]  # path in the returned dict
    name: str  # checkpoint name after ``layers.{layer}.``
    fmt: str
    shape: tuple[int, ...]


def _dense_params(config, layer: int) -> list[_Param]:
    """Every non-routed-expert tensor of backbone ``layer``, by role (Engram parameters belong to F5)."""
    dim, q_lora, heads, head_dim = config.EMB_SIZE, config.Q_LORA_RANK, config.NUM_ATTENTION_HEADS, config.HEAD_DIM
    o_rank, o_groups, inter = config.O_LORA_RANK * config.O_GROUPS, config.O_GROUPS, config.MOE_INTERMEDIATE_SIZE
    shared_inter = inter * config.NUM_SHARED_EXPERTS
    experts, hc = config.NUM_ROUTED_EXPERTS, config.HC_MULT
    mix_hc = (2 + hc) * hc
    idx_heads, idx_dim = config.INDEX_N_HEADS, config.INDEX_HEAD_DIM
    P = _Param
    params = [
        P(("attn", "wq_a"), "attn.wq_a", FP8, (q_lora, dim)),
        P(("attn", "q_norm"), "attn.q_norm.weight", BF16, (q_lora,)),
        P(("attn", "wq_b"), "attn.wq_b", FP8, (heads * head_dim, q_lora)),
        P(("attn", "wkv"), "attn.wkv", FP8, (head_dim, dim)),
        P(("attn", "kv_norm"), "attn.kv_norm.weight", BF16, (head_dim,)),
        P(("attn", "wo_a"), "attn.wo_a", FP8, (o_rank, heads * head_dim // o_groups)),
        P(("attn", "wo_b"), "attn.wo_b", FP8, (dim, o_rank)),
        P(("attn", "attn_sink"), "attn.attn_sink", F32, (heads,)),
        P(("attn_norm",), "attn_norm.weight", BF16, (dim,)),
        P(("ffn_norm",), "ffn_norm.weight", BF16, (dim,)),
        P(("gate_weights", "weight"), "ffn.gate.weight", BF16, (experts, dim)),
        P(("gate_weights", "e_score_correction_bias"), "ffn.gate.bias", F32, (experts,)),
        P(("gate_bias_vl",), "ffn.gate.bias_vl", F32, (experts,)),
        P(("shared_expert_weights", "gate_proj"), "ffn.shared_experts.w1", FP8, (shared_inter, dim)),
        P(("shared_expert_weights", "up_proj"), "ffn.shared_experts.w3", FP8, (shared_inter, dim)),
        P(("shared_expert_weights", "down_proj"), "ffn.shared_experts.w2", FP8, (dim, shared_inter)),
    ]
    for site in ("attn", "ffn"):
        params += [
            P((f"hc_{site}", "fn"), f"hc_{site}_fn", F32, (mix_hc, hc * dim)),
            P((f"hc_{site}", "base"), f"hc_{site}_base", F32, (mix_hc,)),
            P((f"hc_{site}", "scale"), f"hc_{site}_scale", F32, (3,)),
        ]
    kv_source = layer in config.KV_SOURCE_LAYERS
    if kv_source:
        params += [
            P(("compressor", "wkv"), "attn.compressor.wkv.weight", BF16, (head_dim, dim)),
            P(("compressor", "norm"), "attn.compressor.norm.weight", BF16, (head_dim,)),
        ]
        if config.compress_ratio(layer) > 1:
            params.append(P(("compressor", "wgate"), "attn.compressor.wgate.weight", BF16, (head_dim, dim)))
    if layer in config.INDEX_SOURCE_LAYERS:
        params += [
            P(("indexer", "wq_b"), "attn.indexer.wq_b", FP8, (idx_heads * idx_dim, q_lora)),
            P(("indexer", "weights_proj"), "attn.indexer.weights_proj.weight", BF16, (idx_heads, dim)),
        ]
        if kv_source:
            params += [
                P(("indexer", "wk"), "attn.indexer.wk.weight", BF16, (idx_dim, head_dim)),
                P(("indexer", "k_norm"), "attn.indexer.k_norm.weight", BF16, (idx_dim,)),
            ]
    return params


_EXPERT_PROJ = (("gate_proj", "w1", False), ("up_proj", "w3", False), ("down_proj", "w2", True))


def _stored_names(p: _Param, prefix: str) -> list[str]:
    return [f"{prefix}{p.name}.weight", f"{prefix}{p.name}.scale"] if p.fmt == FP8 else [f"{prefix}{p.name}"]


def _prepare(p: _Param, raw: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    full = f"{prefix}{p.name}"
    if p.fmt == FP8:
        t = dequant_fp8_block(raw[f"{full}.weight"], raw[f"{full}.scale"], full)
    else:
        t = raw[full]
        if t.dtype != _STORED_DTYPE[p.fmt]:
            raise ValueError(f"{full}: stored dtype {t.dtype}, expected {_STORED_DTYPE[p.fmt]}")
    if tuple(t.shape) != p.shape:
        raise ValueError(f"{full}: shape {tuple(t.shape)}, expected {p.shape}")
    return t


def _check_layer(config, layer: int):
    if not 0 <= layer < config.NUM_LAYERS:
        raise ValueError(f"layer {layer} is not a backbone layer (0..{config.NUM_LAYERS - 1})")


def load_layer_dense(ckpt: V41Checkpoint, layer: int, config=DeepSeekV41FlashConfig) -> dict:
    """Everything of backbone ``layer`` except the routed experts.

    Keys (bf16 unless noted; ``[out, in]`` orientation for projections):

    * ``attn``: wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink (fp32) -- ``TtV41Attention``;
    * ``attn_norm``, ``ffn_norm``; ``hc_attn`` / ``hc_ffn``: (fn, base, scale) fp32 tuples;
    * ``gate_weights``: weight, e_score_correction_bias (fp32); ``shared_expert_weights``: gate_proj (w1),
      up_proj (w3), down_proj (w2) -- the ``TtMoe`` state-dict entries;
    * ``gate_bias_vl`` (fp32): the gate bias the reference uses for vision tokens;
    * KV sources only: ``compressor``: wkv, norm, and wgate when the ratio is > 1 (stored bf16; the
      reference promotes the ratio > 1 pair to fp32);
    * index sources only: ``indexer``: wq_b, weights_proj, and wk, k_norm when the layer also owns the keys.
    """
    _check_layer(config, layer)
    prefix = f"layers.{layer}."
    params = _dense_params(config, layer)
    raw = ckpt.read([n for p in params for n in _stored_names(p, prefix)])
    out: dict = {}
    for p in params:
        node = out
        for k in p.key[:-1]:
            node = node.setdefault(k, {})
        node[p.key[-1]] = _prepare(p, raw, prefix)
    for site in ("hc_attn", "hc_ffn"):
        out[site] = (out[site]["fn"], out[site]["base"], out[site]["scale"])
    return out


def load_routed_experts(ckpt: V41Checkpoint, layer: int, config=DeepSeekV41FlashConfig) -> list[dict]:
    """``[{gate_proj: w1, up_proj: w3, down_proj: w2}]`` bf16 per routed expert, in expert order."""
    _check_layer(config, layer)
    dim, inter = config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE
    experts = []
    for start in range(0, config.NUM_ROUTED_EXPERTS, EXPERT_READ_BATCH):
        ids = range(start, min(start + EXPERT_READ_BATCH, config.NUM_ROUTED_EXPERTS))
        stems = {(e, key): f"layers.{layer}.ffn.experts.{e}.{w}" for e in ids for key, w, _ in _EXPERT_PROJ}
        raw = ckpt.read([f"{s}.{part}" for s in stems.values() for part in ("weight", "scale")])
        for e in ids:
            entry = {}
            for key, _, down in _EXPERT_PROJ:
                stem = stems[(e, key)]
                t = dequant_mxfp4(raw.pop(f"{stem}.weight"), raw.pop(f"{stem}.scale"), stem)
                expected = (dim, inter) if down else (inter, dim)
                if tuple(t.shape) != expected:
                    raise ValueError(f"{stem}: shape {tuple(t.shape)}, expected {expected}")
                entry[key] = t
            experts.append(entry)
    return experts


def load_layer(ckpt: V41Checkpoint, layer: int, config=DeepSeekV41FlashConfig) -> dict:
    """``load_layer_dense`` plus ``routed_expert_weights``: the full ``TtV41Block`` ``weights`` dict."""
    weights = load_layer_dense(ckpt, layer, config)
    weights["routed_expert_weights"] = load_routed_experts(ckpt, layer, config)
    return weights
