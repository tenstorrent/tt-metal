#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for deepseek-ai/DeepSeek-V4-Flash, layer by layer (engine.py), from transformers' DeepseekV4 code.

Stored tensors (N = isl, i = layer, E = 256 routed experts, k = min(512, N // 4); 43 layers: 2 sliding_attention,
then compressed_sparse_attention (CSA, ratio 4) and heavily_compressed_attention (HCA, ratio 128) alternating,
the last one CSA; layers 0-2 hash_moe, 3-42 moe):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 4096]       bf16  embeddings (no scale); the
                                                                                    model copies them into hc_mult=4
                                                                                    residual streams before layer 0
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 4, 4096]    bf16  all   the 4 mHC residual streams
                                                                                    after attention and MoE
    model.layers.{i}.self_attn.output                      [1, N, 4096]       bf16  all   --sublayers: attention
                                                                                    after o_b_proj
    model.layers.{i}.mlp.output                            [1, N, 4096]       bf16  all   --sublayers: routed experts
                                                                                    + shared expert
    model.layers.{i}.attn_hc.post                          [1, N, 4]          fp32  all   --sublayers: mHC weights
    model.layers.{i}.attn_hc.comb                          [1, N, 4, 4]       fp32  all   placing the attention
    model.layers.{i}.attn_hc.collapsed                     [1, N, 4096]       bf16  all   output / mixing streams,
                                                                                    and its input (pre-norm)
    model.layers.{i}.ffn_hc.{post,comb,collapsed}          as attn_hc               all   --sublayers: same, MoE site
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_norm.output              [1, N, 512]        bf16  all   shared K=V of every
                                                                                    position, pre-RoPE
    past_key_values.layers.{i}.keys                        [1, 1, 127, 512]   bf16  all   last sliding_window - 1
                                                                                    K=V, post RoPE (min(N, 127))
    past_key_values.layers.{i}.compressed_kv.compressor    [1, N // 4, 512]   bf16  CSA   compressed KV entries
                                                           [1, N // 128, 512] bf16  HCA
    past_key_values.layers.{i}.buffer_kv.compressor        [1, N % 4, 1024]   bf16  CSA   kv_proj of the tokens
                                                           [1, N % 128, 512]  bf16  HCA   after the last full window
    past_key_values.layers.{i}.buffer_gate.compressor      as buffer_kv       bf16  CSA, HCA  gate_proj of the same
    past_key_values.layers.{i}.overlap_kv.compressor       [1, 4, 512]        bf16  CSA   last window's Ca half
    past_key_values.layers.{i}.overlap_gate.compressor     [1, 4, 512]        fp32  CSA   its gate + position_bias
    past_key_values.layers.{i}.compressed_kv.indexer       [1, N // 4, 128]   bf16  CSA   indexer keys
    past_key_values.layers.{i}.buffer_kv.indexer           [1, N % 4, 256]    bf16  CSA   (the indexer's own
    past_key_values.layers.{i}.buffer_gate.indexer         [1, N % 4, 256]    bf16  CSA   compressor, same scheme
    past_key_values.layers.{i}.overlap_kv.indexer          [1, 4, 128]        bf16  CSA   at index_head_dim)
    past_key_values.layers.{i}.overlap_gate.indexer        [1, 4, 128]        fp32  CSA
    model.layers.{i}.self_attn.compressor.indexer.scorer.output
                                                           [1, N, N // 4]     fp32  CSA   index scores, pre mask
    model.layers.{i}.self_attn.compressor.indexer.output   [1, N, k]          int32 CSA   top-k compressed entries
                                                                                    per query, -1 = not causal
    model.layers.{i}.mlp.gate.output                       [N, 256]           fp32  all   router logits
    model.layers.{i}.mlp.experts.top_k_index               [N, 6]             int32 all   chosen experts (hash_moe:
                                                                                    tid2eid[token]), ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 6]             fp32  all   their weights
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 4096]       bf16  norm(hc_head(streams))
    logits                                                 [1, 1, 129280]     fp32  last position, lm_head (untied)

What is model-specific, and why it is handled here:

  * Layer kinds. config.json's compress_ratios (one per layer + one for the MTP layer, ignored) become layer_types:
    0 -> sliding_attention: MQA with one shared K=V head (head_dim 512, RoPE on the last 64 dims, theta 1e4), a
    128-token sliding window and a learned per-head attention sink. 128 -> heavily_compressed_attention: the same
    sliding branch plus one compressed KV entry per 128 tokens (softmax-gated pooling of kv_proj by gate_proj +
    position_bias, RMSNorm, YaRN RoPE theta 1.6e5 at the window's first position); every query attends to all
    causally complete entries. 4 -> compressed_sparse_attention: entries every 4 tokens over overlapping
    8-token windows (each token's projection has two halves: Ca feeds the next window, Cb its own), and a lightning
    indexer (its own compressor at index_head_dim 128, 64 query heads from the q LoRA) that keeps the top
    index_topk entries per query. The compressor/indexer state above is what decoding continues from.
  * mHC: the residual is 4 parallel streams [1, N, 4, 4096]. Each sub-layer collapses them with learned weights
    (pre), and writes back post * output + comb^T @ streams, comb Sinkhorn-projected; hc_head collapses them for the
    final norm. The engine's final-norm stage is therefore norm(hc_head(.)); hc_head's checkpoint keys (hc_head_fn,
    ...) share no prefix with norm.weight, so this script reads them into that stage itself.
  * Hash MoE: layers 0-2 route each token by a frozen tid2eid[token_id] table, so every layer receives input_ids.
  * Weights: the upstream names (attn.wq_a, ffn.experts.<e>.w1, hc_attn_fn, ...) are mapped with transformers' own
    deepseek_v4 renaming table. FP8 e4m3 linears (128x128 blocks) and FP4 e2m1 routed experts (two per int8 byte,
    1x32 blocks) both carry ue8m0 (power-of-two) scales and are dequantized with transformers' Fp8Dequantize: the
    products are exact in bf16, so nothing is rounded. Compressor/indexer projections, the router and norms ship
    bf16; sinks, position_bias, mHC and router bias fp32. Activations are not quantized (the reference inference
    code simulates FP8/FP4 activations in places; the golden is the unquantized model).
  * bf16: transformers pins the RMSNorm weights to fp32 (_keep_in_fp32_modules_strict), which turns their
    outputs fp32 and breaks the next bf16 linear, so its bf16 forward does not run as shipped. Here they load at
    the compute dtype instead; they ship bf16, so no value changes. Everything shipped fp32 stays fp32.
  * Attention is eager: the sink term is only in transformers' eager kernel (the model rejects sdpa/flash).

    python generate_golden_deepseek_v4_flash.py --isl 10240
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Shards, resolve_revision  # noqa: E402
from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale); the model then copies them into the hc_mult streams.",
    "model.layers.{i}.output": (
        "Decoder layer output: the hc_mult mHC residual streams [1, N, hc_mult, hidden], after attention and MoE, "
        "each written back as post * sublayer_output + comb^T @ streams. The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention output after o_b_proj (grouped o_a_proj LoRA, after de-rotating the RoPE dims of the output)."
    ),
    "model.layers.{i}.mlp.output": "MoE output: routed experts (weighted) + the shared expert, before mHC.",
    "model.layers.{i}.attn_hc.post": "mHC 'post' weights [1, N, hc_mult] (2*sigmoid): the attention output per stream.",
    "model.layers.{i}.attn_hc.comb": (
        "mHC stream mixer [1, N, hc_mult, hc_mult] after hc_sinkhorn_iters Sinkhorn steps; applied transposed: "
        "new[k] = sum_j comb[j, k] * streams[j]."
    ),
    "model.layers.{i}.attn_hc.collapsed": "Streams collapsed by the mHC 'pre' weights: attention input, pre norm.",
    "model.layers.{i}.ffn_hc.post": "As attn_hc.post, for the MoE site.",
    "model.layers.{i}.ffn_hc.comb": "As attn_hc.comb, for the MoE site.",
    "model.layers.{i}.ffn_hc.collapsed": "As attn_hc.collapsed, for the MoE site (before post_attention_layernorm).",
    "model.layers.{i}.self_attn.kv_norm.output": (
        "The single shared K=V head for every position: kv_norm(kv_proj(x)), before RoPE (which rotates its last "
        "qk_rope_head_dim dims, interleaved pairs; 'main' rope on sliding layers, 'compress' YaRN rope on CSA/HCA)."
    ),
    "past_key_values.layers.{i}.keys": (
        "Sliding-window K=V cache after prefill: the last sliding_window - 1 positions, post RoPE (values is the "
        "same tensor). A decode step appends one and attends to sliding_window keys."
    ),
    "past_key_values.layers.{i}.compressed_kv.compressor": (
        "Compressed KV entries, one per compress_rate tokens (CSA 4, HCA 128): softmax(gate + position_bias)-pooled "
        "kv_proj over the window (CSA: 2*rate tokens, overlapping), kv_norm, then 'compress' RoPE at position "
        "entry * rate. Attention concatenates them after the sliding-window keys."
    ),
    "past_key_values.layers.{i}.buffer_kv.compressor": (
        "kv_proj of the N % compress_rate tokens after the last complete window, waiting for decode to fill it "
        "(CSA: 2*head_dim wide, the Ca | Cb halves). Zero rows when N is a multiple of the rate."
    ),
    "past_key_values.layers.{i}.buffer_gate.compressor": "gate_proj of the same buffered tokens, no position_bias.",
    "past_key_values.layers.{i}.overlap_kv.compressor": (
        "CSA: the last complete window's Ca half [1, rate, head_dim], which the next window's entry pools with its Cb."
    ),
    "past_key_values.layers.{i}.overlap_gate.compressor": "CSA: its gate, position_bias already added (fp32).",
    "past_key_values.layers.{i}.compressed_kv.indexer": (
        "CSA: the lightning indexer's compressed keys [1, N // 4, index_head_dim], built like compressed_kv.compressor "
        "by the indexer's own compressor."
    ),
    "past_key_values.layers.{i}.buffer_kv.indexer": "CSA: indexer kv_proj of the buffered tokens (2*index_head_dim).",
    "past_key_values.layers.{i}.buffer_gate.indexer": "CSA: indexer gate_proj of the buffered tokens, no bias.",
    "past_key_values.layers.{i}.overlap_kv.indexer": "CSA: the indexer compressor's last Ca half.",
    "past_key_values.layers.{i}.overlap_gate.indexer": "CSA: its gate, position_bias added (fp32).",
    "model.layers.{i}.self_attn.compressor.indexer.scorer.output": (
        "CSA: index scores [1, N, N // 4] = sum_h w_h * relu(q_h . k) * index_head_dim^-0.5, fp32, BEFORE the causal "
        "mask (entry e is visible to query t iff e < (t + 1) // 4)."
    ),
    "model.layers.{i}.self_attn.compressor.indexer.output": (
        "CSA: top-k compressed entries per query, k = min(index_topk, N // 4), in descending score order; entries "
        "that are not causally visible are -1 (they are -inf ties, so row t has min(k, (t + 1) // 4) real indices). "
        "Compare rows as sets: equal scores may come out in any order."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, n_routed_experts], before sqrtsoftplus. Computed in the compute dtype (transformers' "
        "router; the reference inference code uses fp32: --compute-dtype float32), stored fp32. Present on hash_moe "
        "layers too, where they only weight the experts tid2eid chose."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": (
        "Routed experts per token, ascending by expert id. moe layers: top num_experts_per_tok of "
        "sqrtsoftplus(logits) + e_score_correction_bias; hash_moe layers: tid2eid[token_id]."
    ),
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sqrtsoftplus scores (no bias) normalized over the chosen experts, times "
        "routed_scaling_factor."
    ),
    "model.last_hidden_state": "Final norm of hc_head(streams) (the mHC collapse of the last layer's streams).",
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}

RMS_NORMS = ("q_a_norm", "kv_norm", "input_layernorm", "post_attention_layernorm", "norm")
CACHE_STATE = ("compressed_kv", "buffer_kv", "buffer_gate", "overlap_kv", "overlap_gate")
EXPERT_MATRICES = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}  # gate_up_proj = [w1; w3], as transformers


def fp32_pinned() -> list[str]:
    """What transformers keeps fp32 under a bf16 model, minus the RMSNorm weights (see the module docstring)."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

    return [m for m in DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict if m not in RMS_NORMS]


@contextlib.contextmanager
def runnable_dtype_plan():
    """The same dtype plan for from_pretrained (selftest reference) as build() gives the streamed model."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

    shipped = DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict
    DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = fp32_pinned()
    try:
        yield
    finally:
        DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = shipped


class HyperHeadNorm(torch.nn.Module):
    """What DeepseekV4Model does after its last layer, as one stage: norm(hc_head(streams))."""

    def __init__(self, hc_head: torch.nn.Module, norm: torch.nn.Module):
        super().__init__()
        self.hc_head, self.norm = hc_head, norm

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        return self.norm(self.hc_head(streams))


class DeepseekV4Golden(LayerwiseGolden):
    hf_repo = "deepseek-ai/DeepSeek-V4-Flash"
    attn, attn_choices = "eager", ("eager",)  # the attention sinks exist only in the eager kernel
    recipe = Recipe(
        modules=("self_attn.kv_norm", "self_attn.compressor.indexer.scorer", "mlp.gate"),  # + the indexer's top-k
        sublayers=("self_attn", "mlp"),  # + attn_hc / ffn_hc (post, comb, collapsed), see register_hooks
        routing="mlp.experts",
        cache=("keys",),  # + the compressor / indexer state, see cache_tensors
        keep_fp32=(
            "mlp.gate.output",
            "mlp.experts.top_k_weights",
            "indexer.scorer.output",
            "_hc.post",
            "_hc.comb",
            "overlap_gate.compressor",
            "overlap_gate.indexer",
        ),
        config=(
            "layer_types",
            "mlp_layer_types",
            "compress_rates",
            "sliding_window",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "q_lora_rank",
            "o_groups",
            "o_lora_rank",
            "qk_rope_head_dim",
            "rope_parameters",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "scoring_func",
            "routed_scaling_factor",
            "swiglu_limit",
            "rms_norm_eps",
            "expert_dtype",
            "quantization_config",
            "num_nextn_predict_layers",
        ),
        notes=NOTES,
    )

    # ---- the checkpoint ------------------------------------------------------------------------------------------

    def generate(self, args):
        self.compute_dtype = getattr(torch, args.compute_dtype)
        revision = resolve_revision(args.hf_repo, args.revision)
        args.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.tmp, prefix="hf_golden_hc_head_") as tmp:
            # A second reader for the hc_head tensors, which no single stage prefix reaches (see stages()). Built with
            # both namings' prefixes: upstream "hc_head_fn" and the "model.hc_head.hc_fn" of save_pretrained.
            self.extra = Shards(args.hf_repo, None if revision == "local" else revision, Path(tmp), list(self.HC_HEAD))
            self.upstream_names = not any(k.startswith("model.") for k in self.extra.weight_map)
            if args.selftest:
                self.randomize_router_buffers(Path(args.hf_repo))
            return super().generate(args)

    HC_HEAD = ("hc_head_", "model.hc_head.")  # upstream, transformers

    def build(self, config, dtype):
        model = super().build(config, dtype)  # plain nn.Linear: from_config does not apply the quantization_config
        model._keep_in_fp32_modules_strict = fp32_pinned()
        return model

    def stages(self, model, backbone, text_config):
        """Checkpoint prefixes for either naming: upstream (embed., layers.<i>., norm., head.; what the hub ships) or
        transformers' (model.embed_tokens., model.layers.<i>., ...; what the selftest's save_pretrained writes)."""
        m = "" if self.upstream_names else "model."
        final = HyperHeadNorm(backbone.hc_head, backbone.norm)
        stages = [("embed", backbone.embed_tokens, "embed." if self.upstream_names else "model.embed_tokens.")]
        stages += [(f"layer {i}", backbone.layers[i], f"{m}layers.{i}.") for i in range(text_config.num_hidden_layers)]
        stages += [("norm", final, f"{m}norm."), ("lm_head", model.lm_head, "head.")]
        # Each stage's module path in the model, which convert() strips from the renamed checkpoint keys.
        paths = ["embed_tokens."] + [f"layers.{i}." for i in range(text_config.num_hidden_layers)] + ["", "lm_head."]
        self.hf_path = {prefix: path for (_, _, prefix), path in zip(stages, paths)}
        return stages

    def convert(self, sd, prefix, module):
        """Dequantize, then map checkpoint keys to the module's parameter names with transformers' own renaming."""
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        sd = {prefix + k: v for k, v in sd.items()}
        if isinstance(module, HyperHeadNorm):
            sd.update({p + k: v for p in self.HC_HEAD for k, v in self.extra.load(len(self.HC_HEAD), p).items()})
        sd = dequantize(sd, self.compute_dtype)
        renamings = [t for t in get_checkpoint_conversion_mapping("deepseek_v4") if isinstance(t, WeightRenaming)]
        out = {}
        for k, v in sd.items():
            name = rename_source_key(k, renamings, [])[0].removeprefix("model.").removeprefix(self.hf_path[prefix])
            # mlp.experts.<e>.w1.weight -> mlp.experts.<e>.gate_proj.weight, which the Loader fuses into gate_up_proj
            parts = name.split(".")
            if len(parts) == 5 and parts[:2] == ["mlp", "experts"] and parts[3] in EXPERT_MATRICES:
                parts[3] = EXPERT_MATRICES[parts[3]]
            out[".".join(parts)] = v
        return out

    # ---- the forward ---------------------------------------------------------------------------------------------

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """DeepseekV4Model.forward before its layer loop: one sliding-window mask and both RoPE tables, all layers."""
        from transformers.masking_utils import create_sliding_window_causal_mask

        cache = self.make_cache(text_config)
        mask = create_sliding_window_causal_mask(
            config=text_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
        )
        rope = {t: backbone.rotary_emb(hidden, position_ids=position_ids, layer_type=t) for t in ("main", "compress")}
        kinds = set(self.layer_types(text_config))
        return RunState(
            position_ids=position_ids,
            cache=cache,
            masks=dict.fromkeys(kinds, mask),
            rope=dict.fromkeys(kinds, rope),
        )

    def layer_kwargs(self, i, kind, state):
        return dict(
            position_embeddings=state.rope[kind],
            position_ids=state.position_ids,
            attention_mask=state.masks[kind],
            past_key_values=state.cache,
            input_ids=state.input_ids,  # hash-routed layers pick experts by token id
            **state.carry,
        )

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        if i == 0:
            # DeepseekV4Model copies the embeddings into hc_mult streams before layer 0; a no-op when they already are.
            def enter_streams(module, args):
                h = args[0]
                if h.dim() == 3:
                    return (h.unsqueeze(2).expand(-1, -1, module.attn_hc.hc_mult, -1).contiguous(), *args[1:])

            handles.append(layer.register_forward_pre_hook(enter_streams))
        if (compressor := layer.self_attn.compressor) is not None and hasattr(compressor, "indexer"):

            def topk(_m, _a, o, key=f"model.layers.{i}.self_attn.compressor.indexer.output"):
                captured[key] = o.to(torch.int32)

            handles.append(compressor.indexer.register_forward_hook(topk))
        if sublayers:
            for site in ("attn_hc", "ffn_hc"):

                def put(_m, _a, o, key=f"model.layers.{i}.{site}"):
                    captured[f"{key}.post"], captured[f"{key}.comb"], captured[f"{key}.collapsed"] = o

                handles.append(getattr(layer, site).register_forward_hook(put))
        return handles

    def is_sublayer(self, key):
        return super().is_sublayer(key) or any(f".{site}." in key for site in ("attn_hc", "ffn_hc"))

    def cache_tensors(self, i, cache):
        """The sliding-window keys (values is the same tensor) and every compressor / indexer state dict entry."""
        layer = cache.layers[i]
        out = {f"past_key_values.layers.{i}.keys": layer.keys}
        for attr in CACHE_STATE:
            for name, t in (getattr(layer, attr, None) or {}).items():
                if t is not None:
                    out[f"past_key_values.layers.{i}.{attr}.{name}"] = t
        return out

    def free_cache(self, i, cache):
        layer = cache.layers[i]
        layer.keys = layer.values = None
        for attr in CACHE_STATE:
            for name in getattr(layer, attr, None) or {}:
                getattr(layer, attr)[name] = None

    def compare(self, name, got, ref, exact):
        if name.endswith("indexer.output"):  # equal scores tie in any order: compare each row as a set
            got, ref = got.sort(dim=-1).values, ref.sort(dim=-1).values
        return super().compare(name, got, ref, exact)

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def selftest(self, args, compute, isl=70):
        # 70 tokens: 17 CSA entries (> index_topk 8) and 4 HCA entries with tokens left in both buffers, 54 beyond the
        # sliding window.
        return super().selftest(args, compute, isl)

    def reference(self, *args, **kwargs):
        with runnable_dtype_plan():
            return super().reference(*args, **kwargs)

    @staticmethod
    def randomize_router_buffers(ckpt: Path) -> None:
        """from_config leaves tid2eid and e_score_correction_bias at zero and the selftest only randomizes parameters;
        random ones make the test check that input_ids reach the hash router and that the bias is loaded."""
        n_experts = json.loads((ckpt / "config.json").read_text())["n_routed_experts"]
        g = torch.Generator().manual_seed(2)
        for f in sorted(ckpt.glob("*.safetensors")):
            sd = load_file(str(f))
            hit = [k for k in sd if k.endswith(("gate.tid2eid", "gate.e_score_correction_bias"))]
            for k in hit:
                t = sd[k]
                if k.endswith("tid2eid"):  # distinct experts per token, as in the real table
                    experts = torch.rand(t.shape[0], n_experts, generator=g).argsort(-1)[:, : t.shape[1]]
                    sd[k] = experts.to(t.dtype).contiguous()
                else:
                    sd[k] = (torch.rand(t.shape, generator=g) * 0.2 - 0.1).to(t.dtype)
            if hit:
                save_file(sd, str(f), metadata={"format": "pt"})

    def tiny_config(self):
        from transformers import DeepseekV4Config

        sliding, csa, hca = "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"
        layer_types = [sliding, csa, hca, csa, hca, sliding]
        return DeepseekV4Config(
            vocab_size=512,
            hidden_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            mlp_layer_types=["hash_moe"] * 2 + ["moe"] * 4,
            num_attention_heads=4,
            head_dim=64,
            partial_rotary_factor=16 / 64,
            q_lora_rank=64,
            o_groups=2,
            o_lora_rank=32,
            n_routed_experts=8,
            num_experts_per_tok=2,
            compress_rates={csa: 4, hca: 16},
            sliding_window=16,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=8,
            hc_mult=4,
            rope_scaling=dict(type="yarn", factor=16, original_max_position_embeddings=32, beta_fast=32, beta_slow=1),
            max_position_embeddings=32 * 16,
        )


def dequantize(sd: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """<name>.weight with a sibling <name>.scale -> weight * scale, by transformers' Fp8Dequantize: FP8 e4m3 with a
    128x128 block grid, or FP4 e2m1 packed two per int8 (low nibble first) with a 1x32 grid; ue8m0 scales. Both
    products are exact in bf16. Unquantized tensors (and the tiny selftest checkpoint) pass through."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    dq = Fp8Dequantize(None)
    for scale in [k for k in sd if k.endswith(".scale")]:  # not mHC's attn_hc.scale: it has no sibling quantized weight
        weight = scale.removesuffix("scale") + "weight"
        if weight in sd and sd[weight].dtype in (torch.float8_e4m3fn, torch.int8):
            sd[weight] = dq._dequantize_one(sd.pop(weight), sd.pop(scale), output_dtype=dtype)
    return sd


if __name__ == "__main__":
    sys.exit(DeepseekV4Golden().main())
