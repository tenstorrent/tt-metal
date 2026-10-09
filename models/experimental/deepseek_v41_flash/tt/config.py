# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash config and checkpoint access, on top of the V4-Flash loader."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.experimental.deepseek_v4_flash.tt.quant import dequantize_fp8_block, dequantize_mxfp4
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

DEFAULT_MODEL_DIR = Path("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash").expanduser()
# V4.1 quantizes its fp8 weights in 32x32 blocks (V4-Flash: 128x128).
FP8_BLOCK = (32, 32)
# HF names the V4 attention block reads; the V4 loader maps them onto ``layers.N.attn.*``.
_V4_ATTENTION_KEYS = (
    "q_a_proj.weight",
    "q_a_norm.weight",
    "q_b_proj.weight",
    "kv_proj.weight",
    "kv_norm.weight",
    "o_a_proj.weight",
    "o_b_proj.weight",
    "sinks",
)
# HF names of the V4 MoE block's router and shared expert; the V4 loader maps them onto
# ``layers.N.ffn.*``.
_V4_MOE_KEYS = (
    "gate.weight",
    "gate.e_score_correction_bias",
    "shared_experts.gate_proj.weight",
    "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
)


def load_config(model_dir=DEFAULT_MODEL_DIR) -> SimpleNamespace:
    """The checkpoint's ``text_config``, plus the ``layer_types`` and ``num_local_experts`` the
    V4 blocks key on.

    Compressed layers get a type V4 has no compressor for, so the V4 block builds none.
    """
    with open(resolve_snapshot_dir(Path(model_dir)) / "config.json") as f:
        cfg = json.load(f)["text_config"]
    cfg["layer_types"] = ["sliding_attention" if r == 0 else "compressed_attention" for r in cfg["compress_ratios"]]
    cfg["num_local_experts"] = cfg["n_routed_experts"]
    return SimpleNamespace(**cfg)


def dequantized(loader: DeepseekV4WeightLoader, name: str, translate: bool = False):
    """Thunk returning checkpoint tensor ``name`` in fp32: fp8 blocks and MXFP4 (int8-packed, the
    routed experts) dequantized."""

    def load():
        w, s = loader.get_tensor(name, translate=translate), loader.get_scale(name, translate=translate)
        if s is None:
            return w.float()
        if w.dtype == torch.int8:
            return dequantize_mxfp4(w, s)
        return dequantize_fp8_block(w, s, block=FP8_BLOCK)

    return load


def attention_weights(loader: DeepseekV4WeightLoader, layer_idx: int) -> dict:
    """Lazy fp32 weights of layer ``layer_idx``'s attention, as thunks.

    The V4 keys are HF names; the V4.1 compressor / indexer keep their checkpoint names
    (``compressor.wkv.weight``, ``indexer.wq_b.weight``, ...).
    """
    weights = {k: dequantized(loader, f"layers.{layer_idx}.self_attn.{k}", True) for k in _V4_ATTENTION_KEYS}
    prefix = f"layers.{layer_idx}.attn."
    for name in loader.keys():
        if name.startswith((prefix + "compressor.", prefix + "indexer.")) and not name.endswith(".scale"):
            weights[name[len(prefix) :]] = dequantized(loader, name)
    return weights


def moe_weights(loader: DeepseekV4WeightLoader, layer_idx: int) -> dict:
    """Lazy fp32 router and shared-expert weights of layer ``layer_idx``'s MoE, as thunks under
    the V4 block's HF names. The routed experts come from :func:`expert_provider`."""
    return {k: dequantized(loader, f"layers.{layer_idx}.mlp.{k}", True) for k in _V4_MOE_KEYS}


def expert_provider(loader: DeepseekV4WeightLoader, layer_idx: int):
    """``provider(e) -> (gate_up [2I, D], down [D, I])``: routed expert ``e`` of layer
    ``layer_idx`` in fp32, gate_up being ``cat([gate, up])`` -- what the V4 experts upload."""
    prefix = f"layers.{layer_idx}.mlp.experts"

    def provider(e: int):
        gate, up, down = (
            dequantized(loader, f"{prefix}.{e}.{p}.weight", True)() for p in ("gate_proj", "up_proj", "down_proj")
        )
        return torch.cat([gate, up]), down

    return provider


def engram_weights(loader: DeepseekV4WeightLoader, layer_idx: int) -> dict:
    """Lazy fp32 ``wkv.weight`` ``[(hc+1)*D, n_hash_cols*head_dim]``, ``q_weight`` and ``k_weight`` ``[hc, D]``
    of layer ``layer_idx``'s Engram, as thunks. The n-gram table is not here: it stays on the host
    (see :mod:`.engram_lookup`)."""
    prefix = f"layers.{layer_idx}.engram."
    return {k: dequantized(loader, prefix + k) for k in ("wkv.weight", "q_weight", "k_weight")}
