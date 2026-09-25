# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dequantized per-layer MiMo-V2.6 weights (HF names, bf16), cached locally as safetensors.

Checkpoint formats (verified against vLLM's ``_shard_fp8_qkv_proj`` + per-block amax == 448 probes):
  * ``qkv_proj.weight`` fp8 e4m3, 128x128 block scales ``weight_scale_inv``. Rows are pre-sharded for
    ``num_key_value_heads`` (=4) ranks: ``[Q_0|K_0|V_0|Q_1|K_1|V_1|...]``, scales tiled per chunk
    (``ceil(rows_per_chunk/128)`` scale rows each: GA 27, SWA 29). We return it re-ordered to the global
    ``[Q|K|V]`` layout the HF module splits.
  * dense MLP (layer 0) fp8 e4m3 with standard 128x128 block scales.
  * routed experts MXFP4: ``weight`` uint8 = two E2M1 codes (low nibble first), ``weight_scale`` uint8 E8M0
    per 32 input elements.
  * o_proj / norms / gate / sink bias bf16.
"""

import math
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file, save_file

from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL, fetch, weight_map

CACHE = Path(os.environ.get("MIMO_V2_EXTRACTED", LOCAL / "extracted"))
CKPT_TP = 4  # fused-qkv pre-shard count (== num_key_value_heads)
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_fp8_block(w, s, block=128):
    w = w.float()
    s = s.float().repeat_interleave(block, 0)[: w.shape[0]].repeat_interleave(block, 1)[:, : w.shape[1]]
    return w * s


def dequant_qkv(w, s, n_q, n_kv, hd, vhd, block=128):
    """Chunked fp8 fused qkv -> dequantized global [Q | K | V] (fp32)."""
    rows_chunk = w.shape[0] // CKPT_TP
    q_c, k_c, v_c = (n_q // CKPT_TP) * hd, (n_kv // CKPT_TP) * hd, (n_kv // CKPT_TP) * vhd
    assert q_c + k_c + v_c == rows_chunk, (w.shape, q_c, k_c, v_c)
    sc_rows = math.ceil(rows_chunk / block)
    rows = torch.arange(w.shape[0])
    if s.shape[0] == CKPT_TP * sc_rows:
        sidx = (rows // rows_chunk) * sc_rows + (rows % rows_chunk) // block
    else:
        assert s.shape[0] == math.ceil(w.shape[0] / block), s.shape
        sidx = rows // block
    deq = w.float() * s.float()[sidx].repeat_interleave(block, 1)[:, : w.shape[1]]
    chunks = deq.split(rows_chunk, 0)
    q = torch.cat([c[:q_c] for c in chunks])
    k = torch.cat([c[q_c : q_c + k_c] for c in chunks])
    v = torch.cat([c[q_c + k_c :] for c in chunks])
    return torch.cat([q, k, v])


def dequant_mxfp4(w, s, block=32):
    lo, hi = (w & 0x0F).long(), (w >> 4).long()
    codes = torch.stack([lo, hi], -1).flatten(-2)  # [out, in]
    vals = FP4[codes]
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(block, 1)
    return vals * scale


def _layer_names(i, wm):
    return [n for n in wm if n.startswith(f"model.layers.{i}.")]


def _raw_layer(i: int) -> dict:
    """Raw (still quantized) checkpoint tensors of layer ``i``, cached locally (~3.4 GB per MoE layer)."""
    path = CACHE / f"layer_{i}.raw.safetensors"
    if path.exists():
        return load_file(str(path))
    wm = weight_map()
    names = _layer_names(i, wm)
    logger.info(f"fetching layer {i}: {len(names)} tensors")
    pre = f"model.layers.{i}."
    raw = {k[len(pre) :]: v.contiguous() for k, v in fetch(names).items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    save_file(raw, str(path))
    return raw


def layer_state(i: int, cfg: MiMoTextConfig | None = None, *, experts: bool = True) -> dict:
    """{HF name without ``model.layers.{i}.``: bf16 tensor} for decoder layer ``i`` (dequantized)."""
    cfg = cfg or MiMoTextConfig()
    raw = _raw_layer(i)
    out = {}
    spec = cfg.layer_attn(i)
    for k, v in raw.items():
        if k.endswith(("weight_scale_inv", "weight_scale")) or (not experts and ".experts." in k):
            continue
        if k == "self_attn.qkv_proj.weight":
            t = dequant_qkv(v, raw[k + "_scale_inv"], spec.n_q, spec.n_kv, spec.head_dim, spec.v_head_dim)
        elif v.dtype == torch.float8_e4m3fn:
            t = dequant_fp8_block(v, raw[k + "_scale_inv"])
        elif v.dtype == torch.uint8:
            t = dequant_mxfp4(v, raw[k + "_scale"])
        else:
            t = v
        out[k] = (t.float() if k.endswith("e_score_correction_bias") else t.bfloat16()).contiguous()
    return out


def global_state(names=("model.embed_tokens.weight", "model.norm.weight")) -> dict:
    path = CACHE / "global.safetensors"
    if path.exists():
        return load_file(str(path))
    out = {k[len("model.") :] if k.startswith("model.") else k: v.bfloat16().contiguous() for k, v in fetch(list(names)).items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    save_file(out, str(path))
    return out


if __name__ == "__main__":
    import sys

    for a in sys.argv[1:]:
        if a == "global":
            global_state()
        else:
            sd = layer_state(int(a))
            logger.info(f"layer {a}: {len(sd)} tensors, {sum(t.numel() * t.element_size() for t in sd.values()) / 1e9:.2f} GB")
