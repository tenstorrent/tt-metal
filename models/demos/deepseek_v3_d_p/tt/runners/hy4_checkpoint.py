# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview checkpoint reader for the prefill weight-cache rebuild (``tt/runners/weight_categories.py``).

The cache rebuild reads the DeepSeek-V3 HF key layout. Hy4's checkpoints differ in four ways, all
handled here so the rest of the rebuild (category checks, the lazy per-layer reader, the model's own
cache writers) is shared with GLM-5.3 unchanged:

  * routed experts are FUSED per layer: ``mlp.experts.gate_up_proj`` ``[E, 2 * I, H]`` (gate rows
    first, then up, per expert) and ``mlp.experts.down_proj`` ``[E, H, I]``, instead of
    ``mlp.experts.{e}.{gate,up,down}_proj.weight``. One expert is read as a slice, not the whole tensor.
  * the MXFP8 release (``tencent/Hy4-preview-FP8``, ``quant_algo: MXFP8``) stores fp8 e4m3 weights
    with one E8M0 (uint8, bias 127) scale per 32 elements along the input dim, in ``<key>_scale``
    (``<name>.weight_scale`` / ``experts.gate_up_proj_scale``), instead of DeepSeek's 128 x 128
    ``weight_scale_inv``. The BF16 release has no scales; both go through ``bf16()``.
  * attention: the output gate is ``self_attn.linear_gate`` (cached as ttMLA's ``g_proj``), and each
    layer carries a ``learnable_sink_param`` ``[heads]`` no TT module consumes yet (#65921).
  * iHC: ``hc_attn_layer.hc_pre.*`` / ``hc_mlp_layer.hc_pre.*`` per layer plus ``model.hc_head.*``,
    not consumed by the prefill stack yet (#65891). ``pending_layer_tensors`` returns them, so those
    issues can add their cache category without re-deriving the key names.

Import-light (torch + safetensors only); nothing here opens a device.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc

MXFP8_BLOCK = 32  # elements per E8M0 scale, along the last (input) dim
E8M0_BIAS = 127

# Hy4 self_attn submodule -> ttMLA canonical key (see reference/cpu_deepseek_v32/weights.py CANONICAL_TO_CPU).
# Equal to the checkpoint name everywhere except the output gate (linear_gate -> g_proj) and the indexer
# LayerNorm bias, which ttMLA keeps as a separate tensor.
_MLA_KEYS = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "kv_b_proj.weight": "kv_b_proj.weight",
    "o_proj.weight": "o_proj.weight",
}
_INDEXER_KEYS = {
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm_bias.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}
_GATE_KEYS = {"linear_gate.weight": "g_proj.weight"}

# Tensors in the checkpoint that no prefill module consumes yet, by owning issue.
SINK_KEY = "self_attn.learnable_sink_param"  # [num_heads] fp32 -- #65921
IHC_LAYER_KEYS = tuple(
    f"{sub}.hc_pre.{field}" for sub in ("hc_attn_layer", "hc_mlp_layer") for field in ("hc_base", "hc_fn", "hc_scale")
)  # #65891
IHC_HEAD_KEYS = ("model.hc_head.hc_head_base", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_scale")


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor, block: int = MXFP8_BLOCK) -> torch.Tensor:
    """bf16 from an MXFP8 tensor: fp8 e4m3 values times 2**(scale - 127), one uint8 E8M0 scale per ``block``
    elements of the last dim. Works on any leading shape (a 2-D weight or a sliced 3-D expert stack)."""
    if scale.dtype != torch.uint8:
        raise TypeError(f"MXFP8 scale must be uint8 (E8M0), got {scale.dtype}")
    if weight.shape[:-1] != scale.shape[:-1] or scale.shape[-1] * block < weight.shape[-1]:
        raise ValueError(
            f"MXFP8 scale {tuple(scale.shape)} does not cover weight {tuple(weight.shape)} at block {block}"
        )
    factor = torch.exp2(scale.to(torch.float32) - E8M0_BIAS)
    factor = factor.repeat_interleave(block, dim=-1)[..., : weight.shape[-1]]
    return (weight.to(torch.float32) * factor).to(torch.bfloat16)


class Hy4Checkpoint(wc._Checkpoint):
    """``_Checkpoint`` over a Hy4 (BF16 or MXFP8) checkpoint that also answers DeepSeek-layout routed-expert
    keys (``model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight``) by slicing the fused tensors."""

    def _scale_key(self, key: str):
        """The MXFP8 scale for ``key``: ``<key>_scale`` (``foo.weight_scale``, ``experts.down_proj_scale``)."""
        cand = f"{key}_scale"
        return cand if cand in self.weight_map else None

    def _get_slice(self, key: str, index) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key} is not in the checkpoint at {self.path}")
        with safe_open(str(self.path / shard), framework="pt") as f:
            return f.get_slice(key)[index]

    def _read(self, key: str, index=None) -> torch.Tensor:
        """``key`` (optionally sliced along its leading dims) as bf16, dequantizing MXFP8 when a scale exists."""
        t = self.get(key) if index is None else self._get_slice(key, index)
        if t.dtype == torch.float8_e4m3fn:
            scale_key = self._scale_key(key)
            if scale_key is None:
                raise KeyError(f"{key} is fp8 but the checkpoint has no {key}_scale (expected MXFP8)")
            s = self.get(scale_key) if index is None else self._get_slice(scale_key, index)
            return dequant_mxfp8(t, s)
        return t.to(torch.bfloat16)

    @staticmethod
    def _parse_expert_key(key: str):
        # model.layers.{i}.mlp.experts.{e}.{proj}.weight  (also model.mtp_layers.{i}.)
        head, sep, tail = key.partition(".mlp.experts.")
        if not sep:
            return None
        parts = tail.split(".")
        if len(parts) != 3 or not parts[0].isdigit() or parts[2] != "weight" or parts[1] not in wc._PROJ:
            return None
        return head, int(parts[0]), parts[1]

    def routed_expert(self, layer_prefix: str, expert: int, proj: str, intermediate: int) -> torch.Tensor:
        """One routed expert's ``proj`` weight in HF Linear layout (``[out, in]``), from the fused tensors."""
        if proj == "down_proj":
            return self._read(f"{layer_prefix}.mlp.experts.down_proj", (slice(expert, expert + 1),))[0]
        fused = f"{layer_prefix}.mlp.experts.gate_up_proj"
        rows = slice(0, intermediate) if proj == "gate_proj" else slice(intermediate, 2 * intermediate)
        return self._read(fused, (slice(expert, expert + 1), rows))[0]

    def bf16(self, key: str) -> torch.Tensor:
        parsed = self._parse_expert_key(key)
        if parsed is not None and key not in self.weight_map:
            layer_prefix, expert, proj = parsed
            # gate_up_proj stacks gate then up, moe_intermediate rows each; down_proj's last dim is that size.
            intermediate = self._last_dim(f"{layer_prefix}.mlp.experts.down_proj")
            return self.routed_expert(layer_prefix, expert, proj, intermediate)
        return self._read(key)

    def _last_dim(self, key: str) -> int:
        from safetensors import safe_open

        with safe_open(str(self.path / self.weight_map[key]), framework="pt") as f:
            return int(f.get_slice(key).get_shape()[-1])


def hy4_mla_weights(ckpt: Hy4Checkpoint, layer_prefix: str, *, owns_indexer: bool, gated: bool) -> dict:
    """One layer's ttMLA weight dict in canonical names (norms and ``weights_proj`` fp32, the rest bf16, as
    ``load_attention_state_dict`` produces for GLM). A ``shared`` indexer layer has no indexer tensors."""
    names = dict(_MLA_KEYS)
    if owns_indexer:
        names.update(_INDEXER_KEYS)
    if gated:
        names.update(_GATE_KEYS)
    out = {}
    for src, canon in names.items():
        t = ckpt._read(f"{layer_prefix}.self_attn.{src}")
        out[canon] = t.float() if ("norm" in canon or "weights_proj" in canon) else t
    return out


def hy4_layer_state_dict(ckpt: Hy4Checkpoint, config, model_cfg, layer_idx: int, categories: set[str]) -> dict:
    """``weight_categories._layer_state_dict`` for Hy4: the same ``TtPrefillBlock`` state-dict shape, read through
    the Hy4 key layout. Norms, the dense FFN, the router and the shared expert use DeepSeek key names already."""
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused

    prefix = f"model.layers.{layer_idx}"
    mla_requested = wc.MLA in categories
    sd = wc._layer_state_dict(ckpt, config, model_cfg, layer_idx, categories - {wc.MLA, wc.ROUTED_EXPERTS})
    if mla_requested:
        sd["mla_weights"] = hy4_mla_weights(
            ckpt,
            prefix,
            owns_indexer=not indexer_layer_is_reused(config, layer_idx),
            gated=bool(getattr(config, "mla_use_output_gate", False)),
        )
    if wc.ROUTED_EXPERTS in categories:
        intermediate = model_cfg.MOE_INTERMEDIATE_SIZE
        sd["routed_expert_weights"] = [
            {p: ckpt.routed_expert(prefix, e, p, intermediate) for p in wc._PROJ}
            for e in range(model_cfg.NUM_ROUTED_EXPERTS)
        ]
    return sd


def pending_layer_tensors(ckpt: Hy4Checkpoint, layer_idx: int) -> dict:
    """The per-layer tensors only Hy4 has and no prefill module consumes yet:
    ``{"learnable_sink": [heads] fp32, "ihc": {"hc_attn_layer.hc_pre.hc_fn": ..., ...}}``.
    For #65921 (sink) and #65891 (iHC) to build their cache categories from."""
    prefix = f"model.layers.{layer_idx}."
    return {
        "learnable_sink": ckpt.get(prefix + SINK_KEY).float(),
        "ihc": {k: ckpt.get(prefix + k).float() for k in IHC_LAYER_KEYS},
    }


def load_hy4_trunk_state_dict(
    checkpoint_path,
    config,
    model_cfg: type,
    *,
    missing: dict[str, list[int]],
    first_layer_idx: int,
    num_layers: int,
) -> dict:
    """``weight_categories.load_trunk_state_dict`` over a Hy4 checkpoint."""
    return wc.load_trunk_state_dict(
        checkpoint_path,
        config,
        model_cfg,
        missing=missing,
        first_layer_idx=first_layer_idx,
        num_layers=num_layers,
        checkpoint=Hy4Checkpoint(checkpoint_path),
        layer_state_dict_fn=hy4_layer_state_dict,
    )
