#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for XiaomiMiMo/MiMo-V2.6-Flash-RL, layer by layer (engine.py), from the repo's own remote code
(modeling_mimo_v2.py at the pinned commit, trust_remote_code).

Stored tensors (N = isl, i = layer; 48 layers: layer 0 full_attention + dense MLP, then MoE layers with
hybrid_layer_pattern 5 sliding_attention : 1 full_attention, i.e. full on 0, 5, 11, 17, 23, 29, 35, 41, 47):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 4096]       bf16  token embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 4096]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.qkv_proj.output [1, N, 13568]      bf16  full     --sublayers: fused projection
                                               [1, N, 14848]      bf16  sliding  [q | k | v], before RoPE / v scale
    model.layers.{i}.self_attn.o_proj.output   [1, N, 4096]       bf16  all   --sublayers: attention after o_proj
    model.layers.{i}.mlp.output                [1, N, 4096]       bf16  all   --sublayers: dense MLP (layer 0) or
                                                                        routed-experts sum (no shared expert)
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 4, N, 192]     bf16  full     RoPE theta 1e7 on dims [0:64] only
                                               [1, 8, N, 192]     bf16  sliding  RoPE theta 1e4 on dims [0:64] only;
                                                                        every position (the device keeps the last 128)
    past_key_values.layers.{i}.values          [1, 4, N, 128]     bf16  full     v_proj output times
                                               [1, 8, N, 128]     bf16  sliding  attention_value_scale (0.707), no RoPE
    model.layers.{i}.mlp.gate.router_logits    [N, 256]           fp32  1-47  hidden @ gate.weight.T in fp32, before
                                                                        sigmoid / e_score_correction_bias
    model.layers.{i}.mlp.gate.top_k_index      [N, 8]             int32 1-47  chosen experts, sorted by expert id
    model.layers.{i}.mlp.gate.top_k_weights    [N, 8]             fp32  1-47  their weights, same order: sigmoid
                                                                        scores renormalized to sum 1
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 4096]       bf16  final norm
    logits                                     [1, 1, 152576]     fp32  last position, untied lm_head

What is peculiar about this model (see NOTES for the full wording):
  * Two attention kinds with different KV head counts (4 full / 8 sliding; 64 query heads, qk head_dim 192, v
    head_dim 128 for both), each with its own RoPE theta. RoPE covers only the first int(192 * 0.334) = 64 dims of
    q and k (half-split rotate_half within those 64); dims 64..191 pass through.
  * Sliding layers have an attention sink: a learned per-head logit self_attn.attention_sink_bias [64] (a weight,
    not stored here) appended as an extra key column before the softmax and dropped after it, so it only grows the
    denominator. Full layers have none (add_full_attention_sink_bias false). Scaling is 192**-0.5.
  * V is multiplied by attention_value_scale = 0.707 before it is cached and attended.
  * MoE gate: sigmoid scores, noaux_tc top-8 with n_group = topk_group = 1 (so a plain top-8 of sigmoid +
    e_score_correction_bias), weights from the uncorrected sigmoid, renormalized, routed_scaling_factor 1.
    Gate weight and the correction bias are held in the compute dtype (bf16 by default; the checkpoint has the
    bias in fp32), the logits are computed in fp32.

Weights (quantization_config: fp8 e4m3 dynamic, 128x128 blocks, experts stored mxfp4) are dequantized to the
compute dtype in ``convert`` with transformers' own routine (finegrained_fp8.Fp8Dequantize), then the bf16 / fp32
upstream code runs as-is. Activations are never quantized. Formats found in the checkpoint:
  * qkv_proj, dense mlp.{gate,up,down}_proj: float8_e4m3fn weight, fp32 ``weight_scale_inv`` per 128x128 block,
    value = fp8 * scale_inv. qkv_proj is stored as 4 tensor-parallel shards, each [q_r | k_r | v_r] with its own
    block grid (ceil(rows_per_rank / 128) block rows), so it is dequantized per shard and regrouped to the
    [q | k | v] rows the remote code splits (llama.cpp's convert_hf_to_gguf does the same).
  * experts.{e}.{gate,up,down}_proj: MXFP4, uint8 ``weight`` [out, in/2] (two e2m1 codes per byte, low nibble
    first) and uint8 E8M0 ``weight_scale`` [out, in/32], value = e2m1 * 2**(scale - 127).
  * o_proj (ignored_layers), norms, embeddings, lm_head, gate, sink bias: stored bf16 (correction bias fp32).
MTP layers (model.mtp.*), the vision and audio towers are never built nor read.

Attention runs eager, on CPU and GPU alike (the default here; --attn sdpa is refused): the remote model declares no
sdpa support, and forced through the config its sink layers fall back to the remote eager function, which then adds
the boolean sdpa mask as numbers instead of masking. No flash-attn / triton / CUDA kernel is needed.

The cache is a DynamicCache built without the config, so sliding layers keep K/V for every position rather than
the last window; the mask comes from config.sliding_window either way, so the forward is unchanged.

    python generate_golden_mimo_v2_6_flash_rl.py --isl 10240
    python generate_golden_mimo_v2_6_flash_rl.py --isl 256 --num-layers 2 --root /tmp/smoke   # dense+full, MoE+SWA
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings of the first n_tokens ids (no scaling, no BOS added).",
    "model.layers.{i}.output": (
        "Decoder layer output: h + mlp(post_attention_layernorm(h)) with h = x + self_attn(input_layernorm(x)). "
        "mlp is the dense SwiGLU MLP on layer 0 (moe_layer_freq 0) and the routed MoE (no shared expert) elsewhere. "
        "The next layer's input."
    ),
    "model.layers.{i}.self_attn.qkv_proj.output": (
        "Fused attention projection (attention_projection_layout fused_qkv), split [q | k | v] = "
        "[64*192 | kv_heads*192 | kv_heads*128] with kv_heads 4 (full) / 8 (sliding); before RoPE and the value "
        "scale. The checkpoint stores this weight as 4 TP shards [q_r | k_r | v_r]; this is the regrouped layout."
    ),
    "model.layers.{i}.self_attn.o_proj.output": "Attention output after o_proj (no post-attention norm), before the residual add.",
    "model.layers.{i}.mlp.output": "Dense MLP (layer 0) or routed MoE output (sum of top-8 expert outputs times their weights), before the residual add.",
    "past_key_values.layers.{i}.keys": (
        "K for every position, after RoPE. RoPE (half-split rotate_half) covers only dims [0:64] = int(head_dim 192 "
        "* partial_rotary_factor 0.334) of each head, dims [64:192] pass through. full_attention: [1, 4, N, 192], "
        "rope_theta 1e7. sliding_attention: [1, 8, N, 192], swa_rope_theta 1e4; the device only needs the last "
        "sliding_window (128) positions. Attention scaling is 192**-0.5. Sliding layers add an attention sink: "
        "self_attn.attention_sink_bias [64] (checkpoint weight, one logit per query head) is appended as an extra "
        "key column to the scaled, masked scores, the row max is subtracted, softmax in fp32, then the sink column "
        "is dropped (probabilities do not renormalize). Full layers have no sink."
    ),
    "past_key_values.layers.{i}.values": (
        "V for every position, the v slice of qkv_proj times attention_value_scale (0.707) in the compute dtype, no "
        "RoPE. full_attention: [1, 4, N, 128]; sliding_attention: [1, 8, N, 128]."
    ),
    "model.layers.{i}.mlp.gate.router_logits": (
        "Router logits [N, n_routed_experts], hidden (post_attention_layernorm output) @ gate.weight.T computed in "
        "fp32 from the compute-dtype operands, before the sigmoid. Selection: top-8 of sigmoid(logits) + "
        "e_score_correction_bias (n_group = topk_group = 1, so no group restriction). fp32."
    ),
    "model.layers.{i}.mlp.gate.top_k_index": "The 8 chosen experts per token, sorted by expert id (the gate's own top-k is unsorted). int32.",
    "model.layers.{i}.mlp.gate.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no correction bias) divided by "
        "their sum + 1e-20, times routed_scaling_factor (1.0). fp32."
    ),
    "logits": "Last position only: lm_head (not tied to embed_tokens) over the final norm output, fp32.",
}

QKV_TP = 4  # tensor-parallel shards in the stored qkv_proj; checked against every scale grid in _dequant_qkv


def _dequant_blocks(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 * per-128x128-block scale_inv, or MXFP4 (uint8 pairs of e2m1 codes, int8 view) * 2**(E8M0 - 127):
    transformers' FP8/MXFP4 dequantizer, which takes the block shape from the scale grid. fp32 out."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    if weight.dtype == torch.uint8:  # packed e2m1: the dequantizer unpacks int8 / float4_e2m1fn_x2
        weight = weight.view(torch.int8)
    return Fp8Dequantize(None)._dequantize_one(weight, scale, output_dtype=torch.float32)


def _dequant_qkv(weight: torch.Tensor, scale: torch.Tensor, q: int, k: int, v: int, block: int = 128) -> torch.Tensor:
    """qkv_proj is stored as QKV_TP shards, each [q/tp | k/tp | v/tp] rows with its own 128-row block grid. Pad each
    shard to whole blocks, dequantize it with its scales, then regroup the rows into the unsharded [q | k | v]."""
    tp = QKV_TP
    rows = (q + k + v) // tp
    blocks = -(-rows // block)
    assert weight.shape[0] == tp * rows and scale.shape[0] == tp * blocks, (weight.shape, scale.shape, tp)
    parts = []
    for r in range(tp):
        w = weight[r * rows : (r + 1) * rows]
        w = torch.cat([w, w.new_zeros(blocks * block - rows, w.shape[1])])  # fp8 zeros: dropped below
        parts.append(_dequant_blocks(w, scale[r * blocks : (r + 1) * blocks])[:rows])
    cuts = [q // tp, k // tp, v // tp]
    q_r, k_r, v_r = zip(*(p.split(cuts) for p in parts))
    return torch.cat([*q_r, *k_r, *v_r])


def _expose_backbone(causal_lm_class) -> None:
    """The remote MiMoV2ForCausalLM sets no base_model_prefix, so PreTrainedModel.get_decoder() returns the wrapper
    itself; the engine needs the text backbone (``.model``, which holds embed_tokens / layers / norm / rotary)."""
    causal_lm_class.get_decoder = lambda self: self.model


def _mask_api_shim(modeling) -> None:
    """MiMoV2Model.forward (written against transformers 5.3) calls the mask builders with ``input_embeds`` and
    ``cache_position``; this transformers takes ``inputs_embeds`` and derives positions from ``position_ids``. Only
    the whole-model forward of --selftest's reference calls them: the layer-by-layer run builds its masks itself."""
    from transformers import masking_utils

    def renamed(fn):
        return lambda input_embeds, cache_position=None, **kw: fn(inputs_embeds=input_embeds, **kw)

    modeling.create_causal_mask = renamed(masking_utils.create_causal_mask)
    modeling.create_sliding_window_causal_mask = renamed(masking_utils.create_sliding_window_causal_mask)


class MiMoV2Golden(LayerwiseGolden):
    hf_repo = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
    trust_remote_code = True
    # The remote MiMoV2 model declares no sdpa support (from_pretrained refuses it). Forced through the config, sink
    # layers fall back to the remote eager attention, which adds the boolean sdpa mask to the scores instead of masking.
    attn, attn_choices = "eager", ("eager",)
    recipe = Recipe(
        sublayers=("self_attn.qkv_proj", "self_attn.o_proj", "mlp"),
        # Routing is a top-k: bf16 rounding of the logits alone can flip an expert at the cut, so they stay exact.
        keep_fp32=("mlp.gate.router_logits", "mlp.gate.top_k_weights"),
        config=(
            "layer_types",
            "hybrid_layer_pattern",
            "moe_layer_freq",
            "sliding_window",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "v_head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_head_dim",
            "swa_v_head_dim",
            "partial_rotary_factor",
            "rope_theta",
            "swa_rope_theta",
            "rope_parameters",
            "attention_value_scale",
            "attention_projection_layout",
            "add_full_attention_sink_bias",
            "add_swa_attention_sink_bias",
            "layernorm_epsilon",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "scoring_func",
            "topk_method",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "quantization_config",
        ),
        notes=NOTES,
    )

    def build(self, config, dtype):
        # Text only: the vision / audio towers are not part of the prefill golden.
        config.vision_config, config.audio_config = {}, {}
        model = super().build(config, dtype)
        _expose_backbone(type(model))
        return model

    def reference(self, ckpt, ids, device, dtype, attn):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # The class from_pretrained is about to import from the tiny checkpoint (imports are cached, same object).
        cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2ForCausalLM", str(ckpt))
        _expose_backbone(cls)
        _mask_api_shim(sys.modules[cls.__module__])
        return super().reference(ckpt, ids, device, dtype, attn)

    def layer_types(self, text_config) -> list[str]:
        # MiMoV2Model.__init__ derives layer_types from hybrid_layer_pattern (1 = sliding); the engine needs them first.
        n = text_config.num_hidden_layers
        return ["sliding_attention" if p == 1 else "full_attention" for p in text_config.hybrid_layer_pattern[:n]]

    def make_cache(self, text_config):
        from transformers import DynamicCache

        return DynamicCache()  # no config: sliding layers keep full-length K/V too

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        state = super().prepare(backbone, text_config, hidden, position_ids)
        # MiMoV2Model.forward: full layers use rotary_emb, sliding layers swa_rotary_emb (own theta).
        state.rope["full_attention"] = backbone.rotary_emb(hidden, position_ids)
        state.rope["sliding_attention"] = backbone.swa_rotary_emb(hidden, position_ids)
        state.carry["cache_position"] = position_ids[0]
        return state

    def convert(self, sd, prefix, module):
        """Dequantize every FP8 / MXFP4 weight to fp32 (the Loader casts to the compute dtype); bf16 / fp32
        tensors, and everything in the unquantized --selftest checkpoint, pass through."""
        for name in [k for k in sd if k.endswith((".weight_scale_inv", ".weight_scale"))]:
            scale = sd.pop(name)
            weight_name = name.rsplit(".", 1)[0] + ".weight"
            weight = sd[weight_name]
            if weight_name.endswith("self_attn.qkv_proj.weight"):
                attn = module.get_submodule(weight_name.removesuffix(".qkv_proj.weight"))
                sd[weight_name] = _dequant_qkv(weight, scale, attn.q_size, attn.k_size, attn.v_size)
            else:
                sd[weight_name] = _dequant_blocks(weight, scale)
        return sd

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:  # the dense layer
            return handles
        key = f"model.layers.{i}.mlp.gate"

        def routing(gate, inputs, output):
            index, weights = output
            x = inputs[0]
            # MiMoV2MoEGate.forward keeps its logits local: this is its first line, verbatim.
            logits = F.linear(x.view(-1, x.shape[-1]).type(torch.float32), gate.weight.type(torch.float32), None)
            # ...and the weights it returned must follow from them exactly, else these are not the gate's logits.
            w = logits.sigmoid().gather(1, index)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * gate.routed_scaling_factor
            assert torch.equal(w, weights), "recomputed router logits do not reproduce the gate's top-k weights"
            index, order = index.sort(dim=-1)
            captured[f"{key}.router_logits"] = logits
            captured[f"{key}.top_k_index"] = index.to(torch.int32)
            captured[f"{key}.top_k_weights"] = weights.gather(-1, order)

        handles.append(gate.register_forward_hook(routing))
        return handles

    def tiny_config(self):
        from common import resolve_revision
        from transformers import AutoConfig

        revision = resolve_revision(self.hf_repo, "main")
        config = AutoConfig.from_pretrained(self.hf_repo, revision=revision, trust_remote_code=True)
        pattern = [0, 1, 1, 0, 1, 1]  # full+dense, sliding+MoE, ..., full+MoE: every kind of the real model
        config.update(
            dict(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                moe_intermediate_size=64,
                num_hidden_layers=len(pattern),
                hybrid_layer_pattern=pattern,
                moe_layer_freq=[0] + [1] * (len(pattern) - 1),
                num_attention_heads=4,
                num_key_value_heads=1,
                swa_num_attention_heads=4,
                swa_num_key_value_heads=2,
                head_dim=48,  # rope dim int(48 * 0.334) = 16: partial rotary, as in the real model
                swa_head_dim=48,
                v_head_dim=32,
                swa_v_head_dim=32,
                n_routed_experts=16,
                num_experts_per_tok=4,
                sliding_window=16,  # < selftest isl: the window really cuts
                sliding_window_size=16,
                vision_config={},
                audio_config={},
                processor_config={},
                quantization_config=None,  # the tiny checkpoint is plain bf16
            )
        )
        del config.quantization_config
        return config


if __name__ == "__main__":
    sys.exit(MiMoV2Golden().main())
