# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint names -> the reference module's attribute names (``DeepseekV4DecoderLayer.state_dict()``).

The TT modules take the reference names (``TtHCA`` wants ``q_a_proj_weight`` etc.; ``TtMoe`` wants
``gate_proj/up_proj/down_proj`` in HF ``[out, in]`` orientation), and every PCC test builds its golden from the
reference module, so the per-layer dict this module returns IS the reference layer's state dict (minus the fused
expert stack, which is streamed one expert at a time -- see :func:`iter_layer_experts`).

Verified against ``DeepseekV4DecoderLayer(cfg, layer_idx).state_dict()`` for the three layer kinds (2026-09-24):

    native (layers.L.*)                       reference
    attn_norm.weight                          input_layernorm.weight
    ffn_norm.weight                           post_attention_layernorm.weight
    attn.attn_sink                            self_attn.sinks
    attn.wq_a.{weight,scale}   (fp8)          self_attn.q_a_proj.weight
    attn.q_norm.weight                        self_attn.q_a_norm.weight
    attn.wq_b.{weight,scale}   (fp8)          self_attn.q_b_proj.weight
    attn.wkv.{weight,scale}    (fp8)          self_attn.kv_proj.weight
    attn.kv_norm.weight                       self_attn.kv_norm.weight
    attn.wo_a.{weight,scale}   (fp8)          self_attn.o_a_proj.weight        (block-diagonal over o_groups, kept)
    attn.wo_b.{weight,scale}   (fp8)          self_attn.o_b_proj.weight
    attn.compressor.wkv.weight                self_attn.compressor.kv_proj.weight
    attn.compressor.wgate.weight              self_attn.compressor.gate_proj.weight
    attn.compressor.ape                       self_attn.compressor.position_bias
    attn.compressor.norm.weight               self_attn.compressor.kv_norm.weight
    attn.indexer.compressor.{wkv,wgate,ape,norm}  self_attn.compressor.indexer.{kv_proj,gate_proj,position_bias,kv_norm}
    attn.indexer.wq_b.{weight,scale} (fp8)    self_attn.compressor.indexer.q_b_proj.weight
    attn.indexer.weights_proj.weight          self_attn.compressor.indexer.scorer.weights_proj.weight
    hc_attn_{fn,base,scale}                   attn_hc.{fn,base,scale}          (fp32, kept fp32)
    hc_ffn_{fn,base,scale}                    ffn_hc.{fn,base,scale}
    ffn.gate.weight                           mlp.gate.weight
    ffn.gate.bias                             mlp.gate.e_score_correction_bias (learned-gate layers)
    ffn.gate.tid2eid                          mlp.gate.tid2eid                 (hash layers 0..2)
    ffn.shared_experts.{w1,w3,w2}.{weight,scale} (fp8)  mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight
    ffn.experts.E.{w1,w3,w2}.{weight,scale} (fp4)       experts: gate_proj / up_proj / down_proj  (per expert; the
                                              reference fuses them as experts.gate_up_proj [E, 2I, D] / down_proj [E, D, I])
    embed.weight / head.weight / norm.weight  model.embed_tokens.weight / lm_head.weight / model.norm.weight
    hc_head_{fn,base,scale}                   model.hc_head.{hc_fn,hc_base,hc_scale}
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from .dequant import fp8_e4m3_dequant_to_bf16, read_tensors, read_weight_map
from .layer_weights import FlashLayerKind, classify_layer, iter_routed_experts_bf16, load_layer_core_bf16

_ATTN = {
    "attn.wq_a": "self_attn.q_a_proj.weight",
    "attn.wq_b": "self_attn.q_b_proj.weight",
    "attn.wkv": "self_attn.kv_proj.weight",
    "attn.wo_a": "self_attn.o_a_proj.weight",
    "attn.wo_b": "self_attn.o_b_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.attn_sink": "self_attn.sinks",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    "shared.gate": "mlp.shared_experts.gate_proj.weight",
    "shared.up": "mlp.shared_experts.up_proj.weight",
    "shared.down": "mlp.shared_experts.down_proj.weight",
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.wq_b": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
}
for _a, _b in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
    for _p in ("fn", "base", "scale"):
        _ATTN[f"hc_{_a}_{_p}"] = f"{_b}.{_p}"

# the shared expert comes out of load_layer_core_bf16 TRANSPOSED (K x N); the reference keeps [out, in]
_TRANSPOSED_FROM_CORE = {"shared.gate", "shared.up", "shared.down"}

TOP_LEVEL = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}


def layer_torch_dict(model_dir: str | Path, layer_idx: int, *, weight_map=None) -> dict[str, torch.Tensor]:
    """One decoder layer's non-expert weights, dequantised to bf16 (fp32 where stored fp32), under the reference
    module's attribute names and orientations. Also carries ``"__kind__"`` (:class:`FlashLayerKind`)."""
    core = load_layer_core_bf16(model_dir, layer_idx, weight_map=weight_map)
    kind: FlashLayerKind = core.pop("kind")
    out: dict[str, torch.Tensor] = {"__kind__": kind}
    for native, t in core.items():
        ref = _ATTN.get(native)
        if ref is None:
            raise KeyError(f"layer {layer_idx}: no reference name for checkpoint tensor {native!r}")
        if native in _TRANSPOSED_FROM_CORE:
            t = t.t().contiguous()
        out[ref] = t
    return out


def iter_layer_experts(
    model_dir: str | Path, layer_idx: int, *, expert_ids=None, weight_map=None
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Stream ``(expert_id, {gate_proj, up_proj, down_proj})`` bf16 in HF ``[out, in]`` orientation (gate/up
    ``[I, D]``, down ``[D, I]``) -- what ``TtRoutedExpert`` consumes. ~50 MB per expert; the caller drops each."""
    for e, w in iter_routed_experts_bf16(model_dir, layer_idx, expert_ids=expert_ids, weight_map=weight_map):
        # blaze's iterator returns gate/up as K x N and down as N x K; undo that transpose
        yield e, {
            "gate_proj": w["gate"].t().contiguous(),
            "up_proj": w["up"].t().contiguous(),
            "down_proj": w["down"].t().contiguous(),
        }


def top_level_torch_dict(model_dir: str | Path, *, weight_map=None) -> dict[str, torch.Tensor]:
    """embedding, LM head, final norm and the HyperHead, under the reference names."""
    wm = weight_map if weight_map is not None else read_weight_map(model_dir)
    raw = read_tensors(model_dir, list(TOP_LEVEL), weight_map=wm)
    return {TOP_LEVEL[k]: v for k, v in raw.items()}


def layer_kind(model_dir: str | Path, layer_idx: int, *, weight_map=None) -> FlashLayerKind:
    return classify_layer(model_dir, layer_idx, weight_map=weight_map)


__all__ = [
    "layer_torch_dict",
    "iter_layer_experts",
    "top_level_torch_dict",
    "layer_kind",
    "TOP_LEVEL",
    "fp8_e4m3_dequant_to_bf16",
]
