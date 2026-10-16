#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for tencent/Hy4-preview, layer by layer (engine.py), from transformers' hy_v4 code.

REQUIRES transformers >= 5.17 (hy_v4 is not in 5.12, which tt-metal's python_env pins). Do not upgrade python_env:
run through a uv overlay that adds a newer transformers on top of it (same torch):

    uv run --no-project --python python_env/bin/python --with transformers==5.19.0 \\
        python models/common/hf_golden/generate_golden_hy4_preview.py --isl 10240

Stored tensors (N = isl, i = layer, k = min(2048, N); 78 layers, gated MLA + DSA sparse attention + learned sinks on
all of them, the indexer "full" on 0, 1, 5, 9, ..., 77 (21 layers) and "shared" on the rest (57); dense MLP on 0, MoE
(256 experts, top 8, 1 shared) on 1-77; the residual is hc_mult = 4 IHC streams):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 6144]       bf16  embeddings (no scale); the model
                                                                                    copies them into 4 streams
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 4, 6144]    bf16  all   the 4 IHC residual streams
                                                                                    after attention and MLP
    model.layers.{i}.self_attn.output                      [1, N, 6144]       bf16  all   --sublayers: attention
                                                                                    after o_proj
    model.layers.{i}.self_attn.gate_proj.output            [1, N, 16384]      bf16  all   --sublayers: attention
                                                                                    gate logits (64 heads x 256),
                                                                                    before sigmoid
    model.layers.{i}.self_attn.o_proj.input                [1, N, 16384]      bf16  all   --sublayers: attention
                                                                                    output AFTER the gate, before
                                                                                    o_proj
    model.layers.{i}.mlp.output                            [1, N, 6144]       bf16  all   --sublayers: dense MLP (0)
                                                                                    / routed + shared experts (1-77)
    model.layers.{i}.attn_hc.post                          [1, N, 4]          fp32  all   --sublayers: IHC weights
    model.layers.{i}.attn_hc.collapsed                     [1, N, 6144]       bf16  all   placing the attention
                                                                                    output, and its input (pre norm)
    model.layers.{i}.ffn_hc.input                          [1, N, 4, 6144]    bf16  all   --sublayers: the streams
                                                                                    after attention, before the MLP
    model.layers.{i}.ffn_hc.{post,collapsed}               as attn_hc               all   --sublayers: same, MLP site
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output   [1, N, 576]        bf16  all   MLA latent [c_kv (512) |
                                                                                    k_pe (64)], raw: before
                                                                                    kv_a_layernorm and RoPE
    past_key_values.layers.{i}.keys                        [1, 1, N, 512]     bf16  all   cached latent c_kv, after
                                                                                    kv_a_layernorm
    past_key_values.layers.{i}.values                      [1, 1, N, 64]      bf16  all   cached k_pe, after RoPE
                                                                                    (interleaved), one head
    past_key_values.layers.{i}.indexer_keys                [1, N, 128]        bf16  full  indexer keys: wk -> k_norm
                                                                                    -> RoPE on the LAST 64 dims
    model.layers.{i}.self_attn.indexer.output              [1, N, k]          int32 full  top-k key indices per
                                                                                    query, descending score; row s:
                                                                                    first min(s+1, k) real (sets)
    model.layers.{i}.mlp.gate.output                       [N, 256]           fp32  1-77  router logits, before
                                                                                    sigmoid / correction bias
    model.layers.{i}.mlp.experts.top_k_index               [N, 8]             int32 1-77  chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 8]             fp32  1-77  their weights, same order:
                                                                                    normalized sigmoid, x 2.827
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 6144]       bf16  norm(hc_head(streams))
    logits                                                 [1, 1, 120832]     fp32  last position, fp32 lm_head
                                                                                    (untied)

What is model-specific, and why it is handled here:

  * Attention (HYV4Attention) is DeepSeek-V3.2 MLA + DSA on every layer. q = q_b_proj(q_a_layernorm(q_a_proj(x)))
    gives 64 heads of [nope 192 | rope 64]; kv_a_proj_with_mqa gives the latent [c_kv 512 | k_pe 64]; the HF cache
    stores the latent itself (keys = kv_a_layernorm(c_kv), values = k_pe after RoPE, both one head), and kv_b_proj
    expands it to 64 heads of [k_nope 192 | v 256] after the cache. config.json's num_key_value_heads = 8 is
    overwritten by HYV4Config to num_attention_heads (MLA never groups keys). RoPE is "default" (theta 1e7) on the 64 rope dims,
    INTERLEAVED pairs as DeepSeek lays them out, not transformers' hy_v4 half-split (interleaved_mla_rope: ppl 199
    -> 3.91); yarn_apply_mscale is a no-op for default RoPE, so the scale is 256 ** -0.5.
  * DSA: a "full" layer's indexer (32 heads x 128, q from the shared q_resid, k = LayerNorm(wk(x)) kept fp32, RoPE on
    the last 64 dims) keeps index_topk = 2048 keys per query; a "shared" layer reuses the previous full layer's top-k,
    which the layer returns and the next layer is handed (prev_topk_indices), as HYV4Model.forward does. The top-k
    becomes a mask on top of the causal one (eager/sdpa path).
  * Gated MLA (gated_mla, gating_type elementwise in config.json): attn_output * sigmoid(gate_proj(x)) per head and
    channel, before o_proj (checkpoint name linear_gate). The ungated attention output is not a module output.
  * Sinks: one learned logit per head (checkpoint learnable_sink_param) joins the softmax and is dropped: only the
    eager kernel has it, and the model declares no sdpa support, so --attn is eager only.
  * Masks: HYV4Model.forward builds ONE causal mask with allow_is_causal_skip=False (the indexer needs a real one);
    the cache is DynamicCache, whose layer_types "deepseek_sparse_attention" maps to DynamicIndexedLayer. prepare()
    mirrors both.
  * IHC (enable_ihc): the residual is 4 streams [1, N, 4, 6144]; the embeddings are copied into all four before
    layer 0 (a pre-hook here). Each site (attn_hc, ffn_hc) computes, in fp32, pre = sigmoid(.) + eps and post =
    hc_magnitude (2.0) * sigmoid(.) + eps from fn @ flattened streams times their inverse RMS; the sub-layer input is
    collapsed = sum_j pre_j * stream_j, and the new streams are post[k] * output + stream[k] (NO comb / Sinkhorn
    mixing, unlike DeepSeek-V4). pre is not returned by the module, so it is not stored. After the last layer
    hc_head collapses the streams the same way with its own weights, then the final norm: the engine's final-norm
    stage is norm(hc_head(.)) here.
  * lm_head is pinned fp32 by transformers (enable_lm_head_fp32; _keep_in_fp32_modules_strict) and the hidden state
    is cast to fp32 before it, so logits() does the same. Also fp32 whatever the compute dtype: IHC fn / base /
    scale, hc_head, sinks, the indexer's weights_proj and k_norm, the router e_score_correction_bias.
  * MoE gate: sigmoid of fp32 logits, top-8 of sigmoid + e_score_correction_bias (no groups), weights from the
    uncorrected sigmoid renormalized and times 2.827; routed SwiGLU clamped at swiglu_limit 10.
  * Weights ship bf16, unquantized, experts already fused (gate_up_proj / down_proj). convert() applies
    transformers' own hy_v4 renaming table (learnable_sink_param -> sinks, linear_gate -> gate_proj,
    hc_attn_layer.hc_pre.hc_fn -> attn_hc.fn, hc_head.hc_head_fn -> hc_head.hc_fn, ...). The MTP layer
    (model.mtp_layers.0) is never built nor read.

    python generate_golden_hy4_preview.py --isl 10240
    python generate_golden_hy4_preview.py --isl 2304 --num-layers 3 --root /tmp/smoke   # dense/full, MoE/full, MoE/shared
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Shards, resolve_revision  # noqa: E402
from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

MIN_TRANSFORMERS = (5, 17)
UV = "uv run --no-project --python python_env/bin/python --with transformers==5.19.0 python"

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added); the model then copies them into the hc_mult streams.",
    "model.layers.{i}.output": (
        "Decoder layer output: the hc_mult IHC residual streams [1, N, hc_mult, hidden], after attention and MLP, each "
        "site writing back new[k] = post[k] * sublayer_output + streams[k] (no stream mixing). The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": "Attention output after o_proj, before the IHC write-back.",
    "model.layers.{i}.self_attn.gate_proj.output": (
        "Attention gate logits [1, N, num_heads * v_head_dim], before the sigmoid: the attention output of head h, "
        "channel c is multiplied by sigmoid of entry h * v_head_dim + c (elementwise gated MLA)."
    ),
    "model.layers.{i}.self_attn.o_proj.input": (
        "Attention output after the sinks softmax, @ V and the gate: attn * sigmoid(gate_proj(x)), heads "
        "concatenated [1, N, num_heads * v_head_dim]; the input of o_proj."
    ),
    "model.layers.{i}.mlp.output": (
        "Dense SwiGLU MLP (layer 0), or routed experts (weighted, SwiGLU clamped at swiglu_limit) + the shared expert; "
        "before the IHC write-back."
    ),
    "model.layers.{i}.attn_hc.post": (
        "IHC 'post' weights [1, N, hc_mult] = hc_magnitude * sigmoid(.) + hc_eps, computed in fp32: the attention "
        "output's weight per stream."
    ),
    "model.layers.{i}.attn_hc.collapsed": (
        "Streams collapsed by the IHC 'pre' weights (sigmoid(.) + hc_eps, fp32, not returned by the module): "
        "sum_j pre_j * streams[j], cast to the compute dtype; the attention input before input_layernorm."
    ),
    "model.layers.{i}.ffn_hc.input": "The IHC streams after the attention write-back, before ffn_hc: the MLP site's input.",
    "model.layers.{i}.ffn_hc.post": "As attn_hc.post, for the MLP site.",
    "model.layers.{i}.ffn_hc.collapsed": "As attn_hc.collapsed, for the MLP site (before post_attention_layernorm).",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA latent [c_kv (kv_lora_rank) | k_pe (qk_rope_head_dim)], the raw projection: before kv_a_layernorm on "
        "c_kv and before RoPE on k_pe."
    ),
    "past_key_values.layers.{i}.keys": (
        "The cached MLA latent c_kv [1, 1, N, kv_lora_rank], after kv_a_layernorm (RMSNorm); kv_b_proj expands it to "
        "per-head [k_nope | v] after the cache."
    ),
    "past_key_values.layers.{i}.values": (
        "The cached k_pe [1, 1, N, qk_rope_head_dim] after RoPE: default RoPE (theta from rope_parameters), interleaved "
        "pairs (x0, x1), (x2, x3), ... (see interleaved_mla_rope); one head shared by all query heads. With keys: the latent cache."
    ),
    "past_key_values.layers.{i}.indexer_keys": (
        "DSA indexer keys [1, N, index_head_dim]: wk -> k_norm (LayerNorm with bias, fp32) -> cast to the compute "
        "dtype -> RoPE on the LAST qk_rope_head_dim dims (half-split). Only on indexer_types == 'full' layers."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "DSA top-k key indices per query, k = min(index_topk, n_tokens), in descending score order, int32. Only the "
        "first min(s+1, k) entries of row s are real; the rest are causally masked ties in arbitrary order, so compare "
        "those first entries as sets. Present only on indexer_types == 'full' layers; a 'shared' layer reuses the "
        "previous full layer's. When n_tokens <= index_topk every key is selected and attention is effectively dense."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, n_routed_experts] = hidden @ gate.weight.T, both cast to fp32 by the router; before the "
        "sigmoid and e_score_correction_bias."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": (
        "Routed experts chosen per token (num_experts_per_tok): top-k of sigmoid(logits) + e_score_correction_bias "
        "(fp32), no group restriction; sorted ascending by expert id. int32."
    ),
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum + "
        "1e-20 (norm_topk_prob), times routed_scaling_factor; the shared expert is added unweighted. fp32."
    ),
    "model.last_hidden_state": "Final norm of hc_head(streams): the learned IHC collapse of the last layer's streams.",
    "logits": "Last position only: lm_head (not tied, pinned fp32) over the final hidden state cast to fp32.",
}

HC_SITES = ("attn_hc", "ffn_hc")
# hc_head's checkpoint prefix: upstream, and what transformers 5.19's save_pretrained writes (its reverse renaming).
HC_HEAD = ("model.hc_head.", "model.hc_pre.hc_head.")
NORM = "model.norm."


class HyperHeadNorm(torch.nn.Module):
    """What HYV4Model does after its last layer, as one stage: norm(hc_head(streams))."""

    def __init__(self, hc_head: torch.nn.Module, norm: torch.nn.Module):
        super().__init__()
        self.hc_head, self.norm = hc_head, norm

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        return self.norm(self.hc_head(streams))


def chunked_indexer_forward(chunk: int):
    """HYV4Indexer.forward with its [B, S, heads, T] fp32 scores built ``chunk`` query rows at a time (~406 GB whole
    at N = 56320). The body is upstream's; the projections, k_norm, RoPE and weights_proj still run on the whole
    sequence, since a bf16 linear rounds differently by row count (see the DeepSeek-V4 script). What is sliced reads
    only its own query row: the fp32 score matmul, the head sum, the mask and top-k. check_chunked_indexer compares
    with the installed upstream at startup."""
    import torch.nn.functional as F
    from transformers.models.hy_v4.modeling_hy_v4 import apply_rotary_pos_emb

    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values=None):
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pass, q_rot = torch.split(q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
        k = self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype).unsqueeze(2)
        k_pass, k_rot = torch.split(k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pass, q_rot], dim=-1)
        k = torch.cat([k_pass, k_rot], dim=-1).squeeze(2)
        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)
        k_t = k.transpose(-1, -2).float().unsqueeze(1)
        weights = (
            self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
            * (self.n_heads**-0.5)
            * self.softmax_scale
        )
        topk = min(self.index_topk, k.shape[-2])
        out = []
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            scores = F.relu(torch.matmul(q[:, s:e].float(), k_t))
            index_scores = torch.matmul(weights[:, s:e].unsqueeze(-2), scores).squeeze(-2)
            del scores
            mask = attention_mask[..., s:e, :] if attention_mask.shape[-2] == seq_len else attention_mask
            if mask.dtype == torch.bool:
                index_scores = index_scores.masked_fill(~mask, float("-inf"))
            else:
                index_scores = index_scores + mask
            out.append(index_scores.topk(topk, dim=-1).indices.to(torch.int32))
        return torch.cat(out, dim=-2)

    forward.chunk = chunk
    return forward


def check_chunked_indexer(upstream, device) -> None:
    """chunked_indexer_forward copies upstream's body; exit if the installed transformers has changed it. Real index
    widths (32 heads x 128 from q LoRA 2048 / hidden 6144, RoPE on 64), 3000 rows against a causal additive mask so
    top-2048 is really sparse, chunk sizes that do and do not divide them; on ``device``, since kernel choice is
    what this guards."""
    from transformers import HYV4Config
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4Indexer

    cfg = HYV4Config(hidden_size=6144, q_lora_rank=2048, index_n_heads=32, index_head_dim=128, qk_rope_head_dim=64)
    gen = torch.Generator().manual_seed(0)
    indexer = HYV4Indexer(cfg, 0)
    for p in indexer.parameters():
        p.data = torch.randn(p.shape, generator=gen) * (0.02 if p.dim() == 2 else 0.5) + (1 if p.dim() == 1 else 0)
    indexer = indexer.to(device, torch.bfloat16)
    indexer.k_norm.float()  # the checkpoint keeps it fp32, as from_pretrained does
    n = 3000
    h = torch.randn(1, n, 6144, generator=gen).to(device, torch.bfloat16)
    q_resid = torch.randn(1, n, 2048, generator=gen).to(device, torch.bfloat16)
    pos = torch.arange(n, device=device)[None]
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    freqs = pos[..., None].float() * inv
    emb = torch.cat([freqs, freqs], dim=-1)
    rope = (emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16))
    causal = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    mask = torch.zeros(1, 1, n, n, device=device, dtype=torch.bfloat16).masked_fill(~causal, torch.finfo(torch.bfloat16).min)
    with torch.no_grad():
        ref = upstream(indexer, h, q_resid, rope, mask, pos)
        for chunk in (256, 1000, 4096):
            got = chunked_indexer_forward(chunk)(indexer, h, q_resid, rope, mask, pos)
            # top-k among equal (masked) scores comes in any order: compare each row as a set
            if not torch.equal(got.sort(dim=-1).values, ref.sort(dim=-1).values):
                raise SystemExit(
                    f"chunked indexer (chunk {chunk}) differs from this transformers' HYV4Indexer.forward: upstream "
                    "changed; update chunked_indexer_forward"
                )


def interleaved_mla_rope(modeling) -> None:
    """Rotate the MLA's q / k rope dims as interleaved pairs (x0, x1), (x2, x3), ... (DeepSeek's layout), keeping the
    indexer's half-split rotation.

    transformers' hy_v4 (5.17 - 5.19) rotates both half-split. The checkpoint expects interleaved MLA rope: SGLang,
    which Tencent's model card points to, runs HYV4Attention through DeepseekV2AttentionMLA, whose rope is
    interleaved unless config.rope_interleave is false, and Hy4's config has no such key; its indexer is half-split
    (indexer_rope_interleave unset). Measured: wikitext ppl at isl 10240 is 199 (top-1 18%) half-split, 3.91 (top-1
    66.8%) interleaved. Both call sites use the module-level apply_rotary_pos_emb, the MLA with the default
    unsqueeze_dim=1 and the indexer with unsqueeze_dim=2; check that before rebinding it."""
    import inspect

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

    attn = inspect.getsource(modeling.HYV4Attention.forward)
    index = inspect.getsource(modeling.HYV4Indexer.forward)
    if "apply_rotary_pos_emb(q_rot, k_rot, cos, sin)" not in attn or "unsqueeze_dim=2" not in index:
        raise SystemExit("hy_v4's rope call sites changed: re-check interleaved_mla_rope against this transformers")
    half = getattr(modeling.apply_rotary_pos_emb, "half_split", modeling.apply_rotary_pos_emb)

    def rope(q, k, cos, sin, unsqueeze_dim=1):
        if unsqueeze_dim == 1:  # HYV4Attention: the MLA rope dims
            return apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1)
        return half(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)  # HYV4Indexer

    rope.half_split = half
    modeling.apply_rotary_pos_emb = rope


def check_transformers() -> None:
    import transformers

    version = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    if version < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} has no hy_v4: this script needs >= "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))}. Run it through the overlay (never upgrade python_env):\n"
            f"    {UV} {Path(__file__).name} ..."
        )


class Hy4Golden(LayerwiseGolden):
    hf_repo = "tencent/Hy4-preview"
    # No prefix: wikitext ppl over the same 10100 tokens at isl 10240 is 3.914 raw (top-1 66.8%), 3.927 after the
    # BOS token "<｜hy_start:opensource｜>", 3.927 under its chat template (no_think).
    prefix = ""
    attn, attn_choices = "eager", ("eager",)  # the sinks exist only in the eager kernel (_supports_sdpa = False)
    recipe = Recipe(
        modules=("self_attn.kv_a_proj_with_mqa", "self_attn.indexer", "mlp.gate"),
        # + o_proj.input, attn_hc / ffn_hc (post, collapsed) and ffn_hc.input, see register_hooks
        sublayers=("self_attn", "self_attn.gate_proj", "mlp"),
        routing="mlp.experts",
        cache=("keys", "values", "indexer_keys"),  # the latent [c_kv | k_pe] and the indexer keys: what decode reads
        # Routing is a top-k: bf16 rounding of the logits alone can flip an expert at the cut, so they stay exact.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights", "_hc.post"),
        config=(
            "layer_types",
            "indexer_types",
            "mlp_layer_types",
            "index_topk",
            "index_n_heads",
            "index_head_dim",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_parameters",
            "max_position_embeddings",
            "gated_mla",
            "gating_type",
            "learnable_sink",
            "enable_ihc",
            "enable_lm_head_fp32",
            "hc_mult",
            "hc_magnitude",
            "hc_eps",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "swiglu_limit",
            "tie_word_embeddings",
            "num_nextn_predict_layers",
        ),
        notes=NOTES,
    )

    def main(self, argv=None) -> int:
        check_transformers()
        from transformers.models.hy_v4 import modeling_hy_v4

        interleaved_mla_rope(modeling_hy_v4)  # before any forward, the selftest's reference included
        # HF_GOLDEN_INDEXER_CHUNK=<rows>: DSA indexer scores that many query rows at a time (chunked_indexer_forward).
        if chunk := int(os.environ.get("HF_GOLDEN_INDEXER_CHUNK", "0")):
            from transformers.models.hy_v4.modeling_hy_v4 import HYV4Indexer

            upstream = getattr(HYV4Indexer.forward, "upstream", HYV4Indexer.forward)
            check_chunked_indexer(upstream, "cuda" if torch.cuda.is_available() else "cpu")
            HYV4Indexer.forward = chunked_indexer_forward(chunk)
            HYV4Indexer.forward.upstream = upstream
            print(f"[indexer] scores built {chunk} query rows at a time", flush=True)
        return super().main(argv)

    # ---- the checkpoint ------------------------------------------------------------------------------------------

    def generate(self, args):
        revision = resolve_revision(args.hf_repo, args.revision)
        args.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.tmp, prefix="hf_golden_norm_") as tmp:
            # The final stage needs model.norm.weight AND hc_head, which share no prefix. The stage itself reads
            # hc_head; this second reader fetches model.norm.weight (its shard is re-downloaded if no longer on disk).
            self.extra = Shards(args.hf_repo, None if revision == "local" else revision, Path(tmp), [NORM])
            self.hc_head_prefix = next(p for p in HC_HEAD if self.extra.has(p))
            return super().generate(args)

    def stages(self, model, backbone, text_config):
        stages = super().stages(model, backbone, text_config)
        stages[-2] = ("norm", HyperHeadNorm(backbone.hc_head, backbone.norm), self.hc_head_prefix)
        return stages

    def convert(self, sd, prefix, module):
        """Checkpoint names -> HF parameter names, by transformers' own hy_v4 renaming table (what from_pretrained
        applies). Covers both the upstream names and the ones the selftest's save_pretrained writes."""
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        if isinstance(module, HyperHeadNorm):
            sd = {
                **{f"{prefix}{k}": v for k, v in sd.items()},
                **{f"{NORM}{k}": v for k, v in self.extra.load(1, NORM).items()},
            }
            strip = "model."
        else:
            sd = {f"{prefix}{k}": v for k, v in sd.items()}
            strip = prefix
        renamings = [t for t in get_checkpoint_conversion_mapping("hy_v4") if isinstance(t, WeightRenaming)]
        out = {}
        for k, v in sd.items():
            name = rename_source_key(k, renamings, [])[0]
            assert name.startswith(strip), f"{k} -> {name}: outside {strip}"
            out[name.removeprefix(strip)] = v
        return out

    # ---- the forward ---------------------------------------------------------------------------------------------

    def layer_types(self, text_config) -> list[str]:
        """'<indexer>/<mlp>', e.g. 'full/dense', 'shared/sparse': both decide what a layer stores."""
        n = text_config.num_hidden_layers
        return [f"{a}/{m}" for a, m in zip(text_config.indexer_types[:n], text_config.mlp_layer_types[:n])]

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """HYV4Model.forward before its layer loop: a DynamicCache (indexed layers), ONE causal mask forced to exist
        (allow_is_causal_skip=False: the indexer masks its scores with it) and one RoPE table, for every layer. The
        embeddings become streams in layer 0's pre-hook (register_hooks)."""
        from transformers.masking_utils import create_causal_mask

        cache = self.make_cache(text_config)
        mask = create_causal_mask(
            config=text_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
            allow_is_causal_skip=False,
        )
        rope = backbone.rotary_emb(hidden, position_ids=position_ids)
        kinds = set(self.layer_types(text_config))
        state = RunState(
            position_ids=position_ids, cache=cache, masks=dict.fromkeys(kinds, mask), rope=dict.fromkeys(kinds, rope)
        )
        state.carry["prev_topk_indices"] = None
        return state

    def after_layer(self, i, output, state):
        hidden, state.carry["prev_topk_indices"] = output  # a shared indexer layer reuses this top-k
        return hidden

    def logits(self, text_config, lm_head, hidden):
        return lm_head(hidden[:, -1:].to(lm_head.weight.dtype))  # HYV4ForCausalLM: the lm_head runs in its fp32

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        if i == 0:
            # HYV4Model copies the embeddings into hc_mult streams before layer 0; a no-op when they already are.
            def enter_streams(module, args):
                h = args[0]
                if h.dim() == 3:
                    return (h.unsqueeze(2).expand(-1, -1, module.attn_hc.hc_mult, -1).contiguous(), *args[1:])

            handles.append(layer.register_forward_pre_hook(enter_streams))
        if sublayers:
            for site in HC_SITES:

                def put(_m, _a, o, key=f"model.layers.{i}.{site}"):
                    captured[f"{key}.post"], captured[f"{key}.collapsed"] = o  # HYV4HyperConnection: (post, out)

                handles.append(getattr(layer, site).register_forward_hook(put))

            def streams(_m, args, key=f"model.layers.{i}.ffn_hc.input"):
                captured[key] = args[0]

            def gated(_m, args, key=f"model.layers.{i}.self_attn.o_proj.input"):
                captured[key] = args[0]

            handles.append(layer.ffn_hc.register_forward_pre_hook(streams))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(gated))
        return handles

    def is_sublayer(self, key):
        return super().is_sublayer(key) or key.endswith(".o_proj.input") or any(f".{site}." in key for site in HC_SITES)

    def compare(self, name, got, ref, exact):
        if name.endswith("indexer.output"):  # masked ties come in any order: compare each row as a set
            got, ref = got.sort(dim=-1).values, ref.sort(dim=-1).values
        return super().compare(name, got, ref, exact)

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith(("_hc.scale", "hc_head.hc_scale")):  # O(1): pre / post really depend on the streams
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.2)
            elif name.endswith(("_hc.base", "hc_head.hc_base", ".sinks")):
                t.copy_(torch.randn(t.shape, generator=g))

    def tiny_config(self):
        from transformers import HYV4Config

        indexer_types = ["full", "full", "shared", "shared", "full", "shared"]
        return HYV4Config(
            vocab_size=512,
            pad_token_id=0,  # the real ids (12000x) are outside the tiny vocab
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=len(indexer_types),
            num_attention_heads=4,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            index_head_dim=32,
            index_n_heads=4,
            index_topk=16,  # < selftest isl: DSA is really sparse
            indexer_types=indexer_types,
            mlp_layer_types=["dense"] + ["sparse"] * (len(indexer_types) - 1),
            n_routed_experts=8,
            num_experts_per_tok=2,
            swiglu_limit=0.05,  # small enough that the routed-expert clamp really cuts with these tiny weights
            rope_parameters={"rope_type": "default", "rope_theta": 1e7},
        )  # hc_mult 4, hc_magnitude 2, sinks, gate, fp32 lm_head: as the real model


if __name__ == "__main__":
    sys.exit(Hy4Golden().main())
