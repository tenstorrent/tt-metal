#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for XingChen-AGI/Xing4.0-29B-A4B, layer by layer (engine.py), from the repo's own remote code
(modeling_xing4_0.py at the pinned commit, trust_remote_code).

Stored tensors (N = isl, i = layer; 40 layers, MLA attention on all of them; dense MLP on 0-1, MoE (64 routed
experts, top 4, 1 shared) on 2-39; the residual is hc_mult = 4 mHC streams):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 3584]     bf16  embeddings (no scale); the model
                                                                                  copies them into 4 streams
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 4, 3584]  bf16  all   the 4 mHC residual streams
                                                                                  after attention and MLP
    model.layers.{i}.self_attn.output                      [1, N, 3584]     bf16  all   --sublayers: attention
                                                                                  after o_proj
    model.layers.{i}.mlp.output                            [1, N, 3584]     bf16  all   --sublayers: dense MLP (0-1)
                                                                                  / routed + shared experts (2-39)
    model.layers.{i}.attn_hc.post                          [1, N, 4]        fp32  all   --sublayers: mHC weights
    model.layers.{i}.attn_hc.comb                          [1, N, 4, 4]     fp32  all   placing the attention
    model.layers.{i}.attn_hc.collapsed                     [1, N, 3584]     bf16  all   output / mixing streams,
                                                                                  and its input (pre norm)
    model.layers.{i}.ffn_hc.input                          [1, N, 4, 3584]  bf16  all   --sublayers: the streams
                                                                                  after attention, before the MLP
    model.layers.{i}.ffn_hc.{post,comb,collapsed}          as attn_hc             all   --sublayers: same, MLP site
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output   [1, N, 576]      bf16  all   MLA latent [c_kv (512) |
                                                                                  k_pe (64)], raw: before
                                                                                  kv_a_layernorm and RoPE
    model.layers.{i}.self_attn.kv_a_layernorm.output       [1, N, 512]      bf16  all   c_kv after kv_a_layernorm:
                                                                                  the latent kv_b_proj expands
    past_key_values.layers.{i}.k_pe                        [1, 1, N, 64]    bf16  all   k_pe after YaRN RoPE
                                                                                  (interleaved pairs in, half-split
                                                                                  layout out), shared by all heads
    model.layers.{i}.mlp.gate.router_logits                [N, 64]          fp32  2-39  hidden @ gate.weight.T in
                                                                                  fp32, before sigmoid / bias
    model.layers.{i}.mlp.gate.top_k_index                  [N, 4]           int32 2-39  chosen experts, ascending
    model.layers.{i}.mlp.gate.top_k_weights                [N, 4]           fp32  2-39  their weights, same order:
                                                                                  normalized sigmoid, x 2.0
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 3584]     bf16  norm(mean of the 4 streams)
    logits                                                 [1, 1, 131072]   fp32  last position, untied lm_head

What is model-specific, and why it is handled here:

  * Attention is DeepSeek-V3 MLA, the same on every layer: no sliding window, no sinks, no softcap, plain causal
    mask. q = q_b_proj(q_a_layernorm(q_a_proj(x))) gives 32 heads of [nope 128 | rope 64]; kv_a_proj_with_mqa gives
    the latent [c_kv 512 | k_pe 64]; kv_b_proj(kv_a_layernorm(c_kv)) expands to 32 heads of [k_nope 128 | v 128];
    k_pe is one RoPE head shared by all 32. RoPE is YaRN (factor 64 over 4096, theta 1e4, attention_factor 1) and
    rope_interleave (the config default): pairs (x0, x1), (x2, x3), ... rotate together and the result is written
    half-split [evens | odds], for q and k alike, so q.k is unchanged. Scaling is 192**-0.5 * mscale**2 with
    mscale = 0.1 * ln(64) + 1 (mscale_all_dim 1). The HF cache holds K/V decompressed to 32 heads [1, 32, N, 192 /
    128], which the device never stores, so only the latent pieces above are kept (the cache is freed per layer).
  * mHC: the residual is hc_mult = 4 streams [1, N, 4, 3584]; the embeddings are copied into all four before layer
    0 (a pre-hook here, as Xing4_0Model.forward does). At each site (attn_hc before attention, ffn_hc before the
    MLP) Xing4_0HyperConnection reads the flattened streams through an unweighted RMSNorm and hc_fn [24, 4 * 3584]
    to get pre = sigmoid(.), post = 2 * sigmoid(.) and a 4x4 comb (clamped to +-30, exp, 20 Sinkhorn row / column
    normalizations); the sub-layer input is collapsed = sum_j pre_j * stream_j, then input_layernorm /
    post_attention_layernorm; the new streams are post[k] * out + sum_j comb[k, j] * stream_j (comb @ streams, NOT
    transposed as in DeepSeek-V4). post / comb are returned in the stream dtype (bf16 under bf16 compute; stored
    fp32 so an fp32 run keeps them). After the last layer the streams are AVERAGED (no learned hc_head) and then
    normed, so the engine's final-norm stage is norm(mean(streams)) here.
  * MoE gate: sigmoid of fp32 logits, top-4 of sigmoid + e_score_correction_bias (fp32 in the checkpoint and kept
    fp32), weights from the uncorrected sigmoid renormalized and times routed_scaling_factor 2.0. The remote router
    has no group stage (n_group = topk_group = 1 in this config anyway). The routed sum is accumulated in fp32, then
    cast to the compute dtype before the shared expert is added.
  * Upstream bug, not drift: Xing4_0PreTrainedModel._init_weights initializes HyperConnection.fn / base / scale,
    which do not exist (the parameters are hc_fn / hc_base / hc_scale), so from_config dies in any transformers
    version. Only --selftest builds from a config; _init_weights_shim makes that init use the real names. It never
    runs on checkpoint weights and does not touch the forward. The remote code imports and runs unchanged under
    transformers 5.12 (written against 5.14).
  * The tokenizer is remote code too (Xing4_0Tokenizer, sentencepiece), so generate() builds the token cache with
    trust_remote_code before the engine looks for it.
  * Weights ship bf16 (e_score_correction_bias fp32), unquantized: convert() only renames the final-norm stage.
    The MTP layer (model.layers.40.*) is never built nor read.

    python generate_golden_xing4_0_29b_a4b.py --isl 10240
    python generate_golden_xing4_0_29b_a4b.py --isl 512 --num-layers 3 --root /tmp/smoke   # 2 dense + 1 MoE
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added); the model then copies them into the hc_mult streams.",
    "model.layers.{i}.output": (
        "Decoder layer output: the hc_mult mHC residual streams [1, N, hc_mult, hidden], after attention and MLP, each "
        "site writing back new[k] = post[k] * sublayer_output + sum_j comb[k, j] * streams[j]. The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": "MLA attention output after o_proj, before mHC.",
    "model.layers.{i}.mlp.output": (
        "Dense SwiGLU MLP (layers < first_k_dense_replace), or routed experts (weighted, summed in fp32, cast) + the "
        "shared expert; before mHC."
    ),
    "model.layers.{i}.attn_hc.post": (
        "mHC 'post' weights [1, N, hc_mult] = 2 * sigmoid(.): the attention output's weight per stream. Cast to the "
        "compute dtype by the module (bf16-valued under bf16 compute), stored fp32."
    ),
    "model.layers.{i}.attn_hc.comb": (
        "mHC stream mixer [1, N, hc_mult, hc_mult]: logits clamped to [mhc_h_res_clamp_min, max], exp(. - row max), "
        "hc_sinkhorn_iters x (row-normalize, column-normalize, + hc_eps). Applied as comb @ streams: new[k] = "
        "sum_j comb[k, j] * streams[j]. Cast to the compute dtype by the module, stored fp32."
    ),
    "model.layers.{i}.attn_hc.collapsed": (
        "Streams collapsed by the mHC 'pre' weights (sigmoid): sum_j pre_j * streams[j], the attention input before "
        "input_layernorm."
    ),
    "model.layers.{i}.ffn_hc.input": "The mHC streams after the attention write-back, before ffn_hc: the MLP site's input.",
    "model.layers.{i}.ffn_hc.post": "As attn_hc.post, for the MLP site.",
    "model.layers.{i}.ffn_hc.comb": "As attn_hc.comb, for the MLP site.",
    "model.layers.{i}.ffn_hc.collapsed": "As attn_hc.collapsed, for the MLP site (before post_attention_layernorm).",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA latent [c_kv (kv_lora_rank) | k_pe (qk_rope_head_dim)], the raw projection: before kv_a_layernorm on "
        "c_kv and before RoPE on k_pe."
    ),
    "model.layers.{i}.self_attn.kv_a_layernorm.output": (
        "c_kv after kv_a_layernorm (RMSNorm): the normed latent that kv_b_proj expands to per-head [k_nope | v]."
    ),
    "past_key_values.layers.{i}.k_pe": (
        "k_pe after RoPE, sliced from the HF cache keys [..., qk_nope_head_dim:] (identical on every head: one rope "
        "head shared by all). YaRN RoPE, rope_interleave: input pairs (2m, 2m+1) rotate by frequency m and the output "
        "is laid out half-split [rotated evens | rotated odds]; q's rope part gets the same permutation. "
        "Together with kv_a_layernorm.output this is the MLA latent cache [c_kv | k_pe]."
    ),
    "model.layers.{i}.mlp.gate.router_logits": (
        "Router logits [N, n_routed_experts] = hidden (post_attention_layernorm output) @ gate.weight.T, both cast "
        "to fp32 by the router; before the sigmoid. Selection: top-k of sigmoid(logits) + e_score_correction_bias "
        "(fp32), no group restriction. fp32."
    ),
    "model.layers.{i}.mlp.gate.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.gate.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum + "
        "1e-20 (norm_topk_prob), times routed_scaling_factor (2.0). fp32."
    ),
    "model.last_hidden_state": "Final norm of the MEAN of the last layer's hc_mult streams (no learned hc_head).",
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}


class StreamMeanNorm(torch.nn.Module):
    """What Xing4_0Model does after its last layer, as one stage: norm(streams.mean(dim=2))."""

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        return self.norm(streams.mean(dim=2))


def _init_weights_shim(modeling) -> None:
    """Xing4_0PreTrainedModel._init_weights initializes HyperConnection.fn / .base / .scale, attributes the module
    does not have (its parameters are hc_fn / hc_base / hc_scale), so from_config raises AttributeError. Same init
    under the real names; every other module goes to the shipped method. Only random-init builds (--selftest's tiny
    checkpoint) reach it: checkpoint weights are never initialized, and the forward is untouched."""
    from transformers import initialization as init

    cls = modeling.Xing4_0PreTrainedModel
    if getattr(cls, "_hc_init_shim", False):
        return
    shipped = cls._init_weights

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, modeling.Xing4_0HyperConnection):
            init.normal_(module.hc_fn, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        else:
            shipped(self, module)

    cls._init_weights, cls._hc_init_shim = _init_weights, True


def _modeling(repo: str, **hub):
    """The remote modeling module of ``repo`` (imported once per source and revision, then cached)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_xing4_0.Xing4_0ForCausalLM", repo, **hub)
    return sys.modules[cls.__module__]


class Xing4Golden(LayerwiseGolden):
    hf_repo = "XingChen-AGI/Xing4.0-29B-A4B"
    trust_remote_code = True
    recipe = Recipe(
        modules=("self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm"),  # + the gate, see register_hooks
        sublayers=("self_attn", "mlp"),  # + attn_hc / ffn_hc (post, comb, collapsed) and ffn_hc.input
        cache=("keys", "values"),  # freed per layer; only k_pe is stored, see cache_tensors
        # Routing is a top-k: bf16 rounding of the logits alone can flip an expert at the cut, so they stay exact.
        keep_fp32=("mlp.gate.router_logits", "mlp.gate.top_k_weights", "_hc.post", "_hc.comb"),
        config=(
            "first_k_dense_replace",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_theta",
            "rope_scaling",
            "rope_parameters",
            "rope_interleave",
            "max_position_embeddings",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "mhc_h_res_clamp_min",
            "mhc_h_res_clamp_max",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "scoring_func",
            "topk_method",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "num_nextn_predict_layers",
        ),
        notes=NOTES,
    )

    def generate(self, args):
        # common.token_cache loads the tokenizer without trust_remote_code, and this repo's tokenizer is remote code
        # (Xing4_0Tokenizer, sentencepiece). Build the token cache first with it; the engine then finds the file.
        if not args.selftest:  # the selftest writes its own token cache
            from common import TEXT, hub_kwargs, resolve_revision, token_cache

            revision = resolve_revision(args.hf_repo, args.revision)
            hub = hub_kwargs(args.hf_repo, None if revision == "local" else revision)
            token_cache(args.hf_repo, args.root, args.text or args.root / TEXT, trust_remote_code=True, **hub)
        return super().generate(args)

    def layer_types(self, text_config) -> list[str]:
        """Attention is the same everywhere; the MLP is dense before first_k_dense_replace, MoE after."""
        n, k = text_config.num_hidden_layers, text_config.first_k_dense_replace
        return ["full_attention/dense" if i < k else "full_attention/moe" for i in range(n)]

    def make_cache(self, text_config):
        self.rope_dim = text_config.qk_rope_head_dim  # cache_tensors slices k_pe out of the keys with it
        return super().make_cache(text_config)

    def stages(self, model, backbone, text_config):
        stages = super().stages(model, backbone, text_config)
        name, _, prefix = stages[-2]
        stages[-2] = (name, StreamMeanNorm(backbone.norm), prefix)
        return stages

    def convert(self, sd, prefix, module):
        if isinstance(module, StreamMeanNorm):  # the checkpoint's model.norm.weight is the wrapper's norm.weight
            return {f"norm.{k}": v for k, v in sd.items()}
        return sd

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        if i == 0:
            # Xing4_0Model copies the embeddings into hc_mult streams before layer 0; a no-op when they already are.
            def enter_streams(module, args):
                h = args[0]
                if h.dim() == 3:
                    return (h.unsqueeze(2).expand(-1, -1, module.attn_hc.hc_mult, -1).contiguous(), *args[1:])

            handles.append(layer.register_forward_pre_hook(enter_streams))
        if (gate := getattr(layer.mlp, "gate", None)) is not None:  # MoE layers only

            def routing(_m, _a, o, key=f"model.layers.{i}.mlp.gate"):
                logits, weights, index = o  # Xing4_0TopkRouter returns (router_logits, topk_weights, topk_indices)
                index, order = index.sort(dim=-1)  # torch.topk(sorted=False): order by expert id
                captured[f"{key}.router_logits"] = logits
                captured[f"{key}.top_k_index"] = index.to(torch.int32)
                captured[f"{key}.top_k_weights"] = weights.gather(-1, order)

            handles.append(gate.register_forward_hook(routing))
        if sublayers:
            for site in ("attn_hc", "ffn_hc"):

                def put(_m, _a, o, key=f"model.layers.{i}.{site}"):
                    captured[f"{key}.post"], captured[f"{key}.comb"], captured[f"{key}.collapsed"] = o

                handles.append(getattr(layer, site).register_forward_hook(put))

            def streams(_m, args, key=f"model.layers.{i}.ffn_hc.input"):
                captured[key] = args[0]

            handles.append(layer.ffn_hc.register_forward_pre_hook(streams))
        return handles

    def is_sublayer(self, key):
        return super().is_sublayer(key) or any(f".{site}." in key for site in ("attn_hc", "ffn_hc"))

    def cache_tensors(self, i, cache):
        """Only k_pe after RoPE: the HF cache holds K [1, heads, N, nope + rope] and V decompressed per head, which
        the device never stores; the rope part is one head broadcast to all."""
        keys = cache.layers[i].keys
        k_pe = keys[:, :1, :, -self.rope_dim :]
        assert torch.equal(k_pe.expand(-1, keys.shape[1], -1, -1), keys[..., -self.rope_dim :]), "k_pe differs by head"
        return {f"past_key_values.layers.{i}.k_pe": k_pe}

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def selftest(self, args, compute, isl=64):
        _init_weights_shim(_modeling(self.hf_repo, **self.hub()))  # the class tiny_config's from_config builds
        return super().selftest(args, compute, isl)

    def reference(self, ckpt, ids, device, dtype, attn):
        _init_weights_shim(_modeling(str(ckpt)))  # the class from_pretrained imports from the tiny checkpoint
        return super().reference(ckpt, ids, device, dtype, attn)

    def hub(self) -> dict:
        from common import hub_kwargs, resolve_revision

        return hub_kwargs(self.hf_repo, resolve_revision(self.hf_repo, "main"))

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("hc_scale"):  # O(1) scales so pre / post / comb really depend on the streams
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.2)
            elif name.endswith("hc_base"):
                t.copy_(torch.randn(t.shape, generator=g))

    def tiny_config(self):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.hf_repo, trust_remote_code=True, **self.hub())
        config.update(
            dict(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                moe_intermediate_size=64,
                num_hidden_layers=5,  # 2 dense (first_k_dense_replace) + 3 MoE
                num_attention_heads=4,
                num_key_value_heads=4,
                q_lora_rank=64,
                kv_lora_rank=32,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
                # Derived in Xing4_0Config.__init__ (update() does not rerun it): the RoPE dim and q/k head size.
                head_dim=16,
                qk_head_dim=48,
                n_routed_experts=8,
                num_experts_per_tok=4,
            )
        )
        return config  # YaRN rope, rope_interleave, hc_mult 4, 20 Sinkhorn steps, clamp: as the real model


if __name__ == "__main__":
    sys.exit(Xing4Golden().main())
