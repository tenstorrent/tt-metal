#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for Qwen/Qwen3.8-27B, layer by layer (engine.py), from transformers' qwen3_5 code (text backbone
only; the vision tower and the MTP layer are never built into a stage nor read).

Stored tensors (N = isl, i = layer; 64 dense layers: Gated DeltaNet linear attention on 48 of them, gated GQA on 3, 7,
11, ..., 63; SwiGLU MLP on every layer):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 5120]       bf16  embeddings, no scale
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 5120]       bf16  all   h + mlp(post_attn_norm(h)),
                                                                        h = x + mixer(input_norm(x))
    model.layers.{i}.linear_attn.output        [1, N, 5120]       bf16  GDN   --sublayers: after out_proj
    model.layers.{i}.self_attn.output          [1, N, 5120]       bf16  attn  --sublayers: after the sigmoid output
                                                                        gate and o_proj
    model.layers.{i}.mlp.output                [1, N, 5120]       bf16  all   --sublayers: SwiGLU MLP
  cache/layer_XX.safetensors
    model.layers.{i}.linear_attn.in_proj_a.output  [1, N, 48]     bf16  GDN   decay logits a: g =
                                                                        -exp(A_log) * softplus(a + dt_bias)
    model.layers.{i}.linear_attn.in_proj_b.output  [1, N, 48]     bf16  GDN   beta logits (sigmoid)
    past_key_values.layers.{i}.conv_states     [1, 10240, 4]      bf16  GDN   short-conv state: the last 4 positions
                                                                        of in_proj_qkv, pre-conv
    past_key_values.layers.{i}.recurrent_states [1, 48, 128, 128] fp32  GDN   delta-rule state per value head
                                                                        [k_dim, v_dim] after N tokens
    past_key_values.layers.{i}.keys            [1, 4, N, 256]     bf16  attn  k_norm'd, RoPE'd keys
    past_key_values.layers.{i}.values          [1, 4, N, 256]     bf16  attn
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 5120]       bf16  final RMSNorm
    logits                                     [1, 1, 248320]     fp32  last position, untied lm_head

GDN = layer_types "linear_attention" (layers 0-2, 4-6, ..., 60-62); attn = "full_attention": 3, 7, ..., 63.
metadata.json lists the layers each tensor exists on.

What is model-specific, and why it is handled here:

  * Qwen3_5TextModel.forward, before its layer loop (what prepare() / layer_kwargs() mirror): position ids [0, N) for
    the text row and the 3 mRoPE rows (text only, so all four are equal); the causal mask for "full_attention"
    layers, and for "linear_attention" layers the linear-attention mask, which is None without padding.
  * Gated DeltaNet (Qwen3-Next): in_proj_qkv -> depthwise causal conv (kernel 4) + SiLU; 16 key heads repeated to 48
    value heads, q / k l2-normed in the kernel; beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) in fp32;
    chunked delta rule (chunk 64, fp32 state); RMSNorm of each head gated by silu(z) (output_gate_type swish);
    out_proj. transformers 5.12's qwen3_5 picks fla / causal_conv1d when they are installed, else its pure-torch
    reference; the script refuses to run unless it is the torch reference, so CPU and GPU compute the same algorithm.
    metadata.json records it ("gdn_implementation").
  * Attention: q_proj gives 24 heads of [q 256 | gate 256]; q_norm / k_norm (RMSNorm, 1 + w); partial RoPE on the
    first 64 of 256 dims (rotate-half), theta 1e7, mRoPE sections [11, 11, 10] interleaved (all equal for text);
    4 KV heads; output * sigmoid(gate) before o_proj.
  * --prefill-chunk: each chunk's attention sees keys [0, e) through the cache; GDN carries its conv and recurrent
    states through the cache (upstream's chunked-continuation path). transformers 5.12's cache would hold that state
    in bf16, rounding it at every chunk boundary; make_cache keeps it fp32. Then a chunked bf16 run equals one whole
    forward bit for bit (selftest, chunks of 64 and 128); in fp32 compute the continuation path rounds differently
    in the last bit (state max |diff| ~1e-10).
  * Weights ship bf16, names model.language_model.* and lm_head.

    python generate_golden_qwen3_8_27b.py --isl 10240
    python generate_golden_qwen3_8_27b.py --isl 56320 --prefill-chunk 5120 --attn-chunk 4096
    python generate_golden_qwen3_8_27b.py --selftest --prefill-chunk 64
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers.cache_utils import LinearAttentionLayer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale).",
    "model.layers.{i}.output": (
        "Decoder layer output: h + mlp(post_attention_layernorm(h)), h = x + mixer(input_layernorm(x)), mixer the "
        "Gated DeltaNet or the attention. The next layer's input."
    ),
    "model.layers.{i}.linear_attn.output": "Gated DeltaNet output after the gated RMSNorm and out_proj (--sublayers).",
    "model.layers.{i}.self_attn.output": (
        "Attention output: GQA, times sigmoid(gate) from q_proj's second half, then o_proj (--sublayers)."
    ),
    "model.layers.{i}.mlp.output": "SwiGLU MLP output down(silu(gate(x)) * up(x)), before the residual add (--sublayers).",
    "model.layers.{i}.linear_attn.in_proj_a.output": (
        "Gated DeltaNet decay logits a [1, N, linear_num_value_heads]: per-token log decay g = -exp(A_log) * "
        "softplus(a + dt_bias), computed in fp32."
    ),
    "model.layers.{i}.linear_attn.in_proj_b.output": (
        "Gated DeltaNet beta logits [1, N, linear_num_value_heads]: the delta-rule write strength is sigmoid(.)."
    ),
    "past_key_values.layers.{i}.conv_states": (
        "Gated DeltaNet short-conv state after the prefill: the last linear_conv_kernel_dim positions of in_proj_qkv "
        "(channels [q 2048 | k 2048 | v 6144]), before the conv, oldest first."
    ),
    "past_key_values.layers.{i}.recurrent_states": (
        "Gated DeltaNet recurrent state S [1, linear_num_value_heads, k_dim, v_dim] after the last position, fp32."
    ),
    "past_key_values.layers.{i}.keys": (
        "Attention keys [1, num_key_value_heads, N, head_dim]: k_proj, k_norm, then partial RoPE (first 64 dims, "
        "rotate-half, rope_theta 1e7)."
    ),
    "past_key_values.layers.{i}.values": "Attention values [1, num_key_value_heads, N, head_dim]: v_proj, no norm, no RoPE.",
    "model.last_hidden_state": "Final RMSNorm (1 + w) of the last layer's output.",
    "logits": "Last position only: lm_head (not tied to embed_tokens) in the compute dtype, stored fp32.",
}

# The kernels qwen3_5 prefers over its torch reference when the packages are installed.
FAST_GDN = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")


class Fp32StateLayer(LinearAttentionLayer):
    """LinearAttentionLayer whose recurrent state keeps the dtype it is first written in."""

    def lazy_initialization(self, conv_states=None, recurrent_states=None):
        super().lazy_initialization(conv_states=conv_states, recurrent_states=recurrent_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(recurrent_states)


def torch_gdn() -> str:
    """qwen3_5 binds fla / causal_conv1d at import when installed (and builds fla's fused gated norm), else the
    pure-torch reference. Run only on the reference, so a trace does not depend on what is installed."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    fast = [name for name in (*FAST_GDN, "FusedRMSNormGated") if getattr(modeling_qwen3_5, name) is not None]
    if fast:
        raise SystemExit(f"qwen3_5 would run optimized GDN kernels {fast}: run without fla / causal_conv1d installed")
    return "transformers torch reference (no fla / causal_conv1d installed)"


class Qwen38Golden(LayerwiseGolden):
    hf_repo = "Qwen/Qwen3.8-27B"
    # The text as the model's own reply: the chat template (enable_thinking=False: an empty think block) up to the
    # assistant turn, the user turn naming the WikiText format, as Qwen3.8-Flash-Next. Wikitext ppl over the same 10100
    # tokens at isl 10240 (chunked 5120): 7.430 (top-1 54.6%) with it, 7.445 raw, 8.175 after <|endoftext|> (the
    # checkpoint has no BOS).
    prefix = (
        "<|im_start|>user\nContinue this WikiText-2 dump verbatim: a sequence of Wikipedia articles, one after "
        "another, each starting with a = Title = line, in raw formatting ( @-@ , = = headings = = ).<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    recipe = Recipe(
        modules=("linear_attn.in_proj_a", "linear_attn.in_proj_b"),
        sublayers=("linear_attn", "self_attn", "mlp"),
        cache=("conv_states", "recurrent_states", "keys", "values"),
        keep_fp32=("recurrent_states",),
        config=(
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_parameters",
            "partial_rotary_factor",
            "attn_output_gate",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "output_gate_type",
            "mamba_ssm_dtype",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # AutoModelForCausalLM maps qwen3_5 to Qwen3_5ForCausalLM, whose names (model.layers.*) are not the
        # checkpoint's (model.language_model.layers.*); register the checkpoint's class, as the Flash-Next script does.
        from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

        AutoModelForCausalLM.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration, exist_ok=True)

    def generate(self, args):
        gdn = torch_gdn()
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["gdn_implementation"] = gdn
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def make_cache(self, text_config):
        """transformers 5.12's LinearAttentionLayer allocates the recurrent state in the conv state's dtype (bf16), so
        the fp32 delta-rule state would be rounded to bf16 at every --prefill-chunk boundary and when stored. Keep it
        in the dtype the kernel returns (fp32, the config's mamba_ssm_dtype), as one whole forward carries it."""
        cache = super().make_cache(text_config)
        for layer in cache.layers:
            if type(layer) is LinearAttentionLayer:
                layer.__class__ = Fp32StateLayer
        return cache

    def layer_kwargs(self, i, kind, state):
        """Qwen3_5TextModel passes GDN layers the linear-attention mask, None without padding: leave it out, so
        --prefill-chunk does not turn it into a causal one."""
        kwargs = super().layer_kwargs(i, kind, state)
        if kind == "linear_attention":
            del kwargs["attention_mask"]
        return kwargs

    def selftest(self, args, compute, isl=192):
        # 192 tokens: 3 GDN chunks of 64. With --prefill-chunk 64 the layerwise run is a 3-step prefill, compared with
        # one whole-model forward.
        return super().selftest(args, compute, isl)

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("A_log"):
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)
            elif name.endswith("dt_bias"):
                t.copy_(torch.randn(t.shape, generator=g))
            elif name.endswith("linear_attn.norm.weight"):  # plain RMSNorm weight: w ~ 1
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)

    def tiny_config(self):
        from transformers import Qwen3_5Config

        # GDN x3, attention, twice: the real model's order.
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        return Qwen3_5Config(
            text_config=dict(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=8,
                layer_types=layer_types * 2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_conv_kernel_dim=4,
                max_position_embeddings=4096,
                tie_word_embeddings=False,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=1e7,
                    partial_rotary_factor=0.25,
                    mrope_section=[2, 1, 1],
                    mrope_interleaved=True,
                ),
            ),
            vision_config=dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64),
            tie_word_embeddings=False,
        )


if __name__ == "__main__":
    sys.exit(Qwen38Golden().main())
