#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for zai-org/GLM-5.3-Flash-BF16, layer by layer (engine.py), from transformers' glm5_next code.

Stored tensors (N = isl, i = layer; 45 layers: KDA linear attention on 34 of them, DSA (MLA, no RoPE, k-pool sparse
indexer) on 3, 7, 11, ..., 43; dense MLP on 0-2, MoE (288 routed experts, top 8, 1 shared) on 3-44; the residual is
hc_mult = 4 mHC streams):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 4096]       bf16   embeddings (no scale); the
                                                                                     model copies them into 4 streams
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 4, 4096]    bf16   all   the 4 mHC residual streams
                                                                                     after attention and MLP
    model.layers.{i}.self_attn.output                      [1, N, 4096]       bf16   all   --sublayers: KDA / MLA
                                                                                     output after o_proj
    model.layers.{i}.self_attn.g_b_proj.output             [1, N, 8192]       bf16   KDA   --sublayers: output-gate
                                                                                     logits (o_norm applies sigmoid)
    model.layers.{i}.mlp.output                            [1, N, 4096]       bf16   all   --sublayers: dense MLP (0-2)
                                                                                     / routed + shared experts
    model.layers.{i}.attn_hc.post                          [1, N, 4]          fp32   all   --sublayers: mHC weights
    model.layers.{i}.attn_hc.comb                          [1, N, 4, 4]       fp32   all   placing the attention
    model.layers.{i}.attn_hc.collapsed                     [1, N, 4096]       bf16   all   output / mixing streams,
                                                                                     and its input (pre norm)
    model.layers.{i}.ffn_hc.input                          [1, N, 4, 4096]    bf16   all   --sublayers: the streams
                                                                                     after attention, before the MLP
    model.layers.{i}.ffn_hc.{post,comb,collapsed}          as attn_hc                all   --sublayers: MLP site
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.forget_gate.output          [1, N, 64, 128]    bf16   KDA   per-channel log decay g,
                                                                                     in [-5, 0] (computed fp32)
    model.layers.{i}.self_attn.b_proj.output               [1, N, 64]         bf16   KDA   beta logits (beta =
                                                                                     sigmoid), one per head
    past_key_values.layers.{i}.conv_states                 [1, 24576, 4]      bf16   KDA   short-conv state: the last
                                                                                     4 positions of [q | k | v]_proj,
                                                                                     before the conv
    past_key_values.layers.{i}.recurrent_states            [1, 64, 128, 128]  fp32   KDA   delta-rule state S per
                                                                                     head [k_dim, v_dim] after N tokens
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output   [1, N, 512]        bf16   DSA   MLA latent c_kv, raw
                                                                                     (no k_pe: qk_rope_head_dim 0)
    past_key_values.layers.{i}.keys                        [1, 1, N, 512]     bf16   DSA   MLA latent cache:
                                                                                     kv_a_layernorm(c_kv)
    past_key_values.layers.{i}.indexer_keys                [1, N, 257]        bf16   DSA   indexer state, packed
                                                                                     [k 128 | pool gate 128 | valid 1]
    model.layers.{i}.self_attn.indexer.output              [1, N, 2051]       int32  DSA   selected key positions per
                                                                                     query (2048 + 3 tail), -1 = none;
                                                                                     compare rows as sets
    model.layers.{i}.mlp.gate.output                       [N, 288]           fp32   3-44  router logits, before
                                                                                     sigmoid / correction bias
    model.layers.{i}.mlp.experts.top_k_index               [N, 8]             int32  3-44  chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 8]             fp32   3-44  their weights, same order:
                                                                                     normalized sigmoid, x 2.5
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 4096]       bf16   norm(mean of the 4 streams)
    logits                                                 [1, 1, 154880]     fp32   last position, untied lm_head

KDA = layers 0-2, 4-6, 8-10, ..., 40-42, 44 (layer_types "linear_attention"); DSA = 3, 7, ..., 43
("indexed_attention"). metadata.json lists the layers each tensor exists on; layer_kinds is "<attention>/<mlp>".

REQUIRES transformers >= 5.17 (glm5_next is in-library from 5.17; validated with 5.19.0). tt-metal's python_env pins
an older one, so run through a uv overlay that adds it on top of python_env, without touching it:

    uv run --no-project --python python_env/bin/python --with transformers==5.19.0 \\
        python models/common/hf_golden/generate_golden_glm_5_3_flash.py --isl 10240

What is model-specific, and why it is handled here:

  * Glm5NextTextModel.forward, before its layer loop (what prepare() mirrors): no RoPE anywhere (NoPE MLA,
    position_embeddings=None); one 2D boolean padding mask [1, N] for both layer types, all True here (no padding):
    create_recurrent_attention_mask returns None for an unpadded batch and the model substitutes ones. There is NO
    4D causal mask: KDA is causal by construction, and the DSA indexer applies causality itself (it only selects
    visible keys) and its top-k is turned into the attention's boolean mask. The cache is DynamicCache(config):
    LinearAttentionLayer on KDA layers, DynamicIndexedLayer on DSA layers. The 4 streams are the embeddings
    copied (a pre-hook on layer 0, as the model does), and the final stage is norm(hc_head(streams)), hc_head being
    an unweighted mean. Layers return (streams, top-k); the top-k is handed to the next layer as prev_topk_indices
    (always None here: every indexer is "full", none "shared"). Layers receive input_ids=None, like the VLM wrapper.
  * KDA (Kimi Delta Attention): q/k/v projections -> depthwise causal conv (kernel 4, fp32 weight) + SiLU ->
    l2-normed q/k; per-channel decay g = -5 * sigmoid(exp(A_log) * (f_b_proj(f_a_proj(x)) + dt_bias)) (safe gate,
    lower bound -5); beta = sigmoid(b_proj(x)); chunked delta rule (chunk 64, fp32); o_norm = RMSNorm gated by
    sigmoid(g_b_proj(g_a_proj(x))); o_proj. The kernels are transformers' PURE-TORCH reference functions
    (chunk_kimi_delta_attention, causal_conv1d_fn, ...). They are wrapped to prefer hub kernels / the `fla` and
    `causal_conv1d` packages when installed; this script checks the wrapping and, if anything but the torch function
    would run (e.g. fla on a GPU box), rebinds the torch reference, so CPU and GPU compute the same algorithm.
    metadata.json records which ran ("kda_implementation").
  * DSA: MLA with q LoRA 1536 and kv LoRA 512, 64 heads of qk 256 / v 256, all NoPE; the MLA cache holds only the
    normed latent (values are empty). The indexer (32 heads x 128, q from the q LoRA residual, k = LayerNorm(wk x))
    scores POOLS of 4 consecutive keys (softmax-gated average with a learned position bias), keeps the top
    index_topk / 4 = 512 pools of fully-visible pools, expands them to 2048 key positions, and always appends the
    current incomplete pool (up to 3 keys), hence 2051 columns. Rows hold -1 where nothing is selected; with
    N <= 2048 every visible key is selected and attention is effectively dense causal.
  * mHC: at each site (attn_hc, ffn_hc) the streams go through an unweighted RMSNorm and fn [24, 4 * 4096] in fp32:
    pre = sigmoid(.) + eps, post = 2 * sigmoid(.), comb = softmax(.) + eps then 20 Sinkhorn steps. The block input
    is sum_j pre_j * streams_j; the new streams are post[k] * out + sum_j comb[j, k] * streams[j] (comb TRANSPOSED,
    as DeepSeek-V4, unlike Xing). post / comb stay fp32 (stored fp32).
  * MoE: router logits in fp32, sigmoid, top-8 of sigmoid + e_score_correction_bias (fp32, no group stage),
    weights from the uncorrected sigmoid renormalized and x 2.5; SwiGLU with gate clamped <= 10, up to [-10, 10]
    (dense MLP and shared expert too).
  * Weights ship bf16 except fp32 hc_*_base / hc_*_scale, A_log, dt_bias and the router bias. Loading follows
    from_pretrained under --compute-dtype: conv1d, dt_bias, A_log and e_score_correction_bias are pinned fp32
    (_keep_in_fp32_modules_strict), everything else (hc base / scale included) is cast to the compute dtype.
    convert() applies transformers' own glm5_next renaming table (f_a_proj -> forget_gate.f_a_proj, hc_attn_fn ->
    attn_hc.fn, ...) and concatenates q/k/v_conv1d into conv1d as its converter does. The vision tower and the MTP
    layer (layers.45) are never built into a stage nor read.
  * Memory: the torch chunked KDA materializes [64 heads, N/64 chunks, 64, 64, 128] fp32 intermediates (~4.8 GB
    each at N = 2304, ~21 GB at N = 10240, a few alive at once); MoE layers hold ~15 GB of expert weights. Long ISL:
    HF_GOLDEN_KDA_CHUNK runs KDA a segment at a time (segmented_kda) and HF_GOLDEN_INDEXER_CHUNK the indexer a slice
    of query rows at a time (chunked_indexer_forward); both bit-identical, both checked at startup.

    HF_GOLDEN_KDA_CHUNK=4096 HF_GOLDEN_INDEXER_CHUNK=2048 \\
        python generate_golden_glm_5_3_flash.py --isl 56320 --attn-chunk 2048     # 4.1 min, 89 GB GPU, 117 GB

    python generate_golden_glm_5_3_flash.py --isl 10240
    python generate_golden_glm_5_3_flash.py --isl 2304 --num-layers 5 --root /tmp/smoke   # KDA dense, DSA + MoE,
                                                                                           # KDA + MoE; DSA sparse
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added); the model then copies them into the hc_mult streams.",
    "model.layers.{i}.output": (
        "Decoder layer output: the hc_mult mHC residual streams [1, N, hc_mult, hidden], after attention and MLP, each "
        "site writing back new[k] = post[k] * sublayer_output + sum_j comb[j, k] * streams[j]. The next layer's input."
    ),
    "model.layers.{i}.self_attn.output": (
        "KDA layers: linear-attention output after o_norm (gated RMSNorm) and o_proj. DSA layers: sparse MLA output "
        "after o_proj. Before mHC (--sublayers)."
    ),
    "model.layers.{i}.self_attn.g_b_proj.output": (
        "KDA output-gate logits g_b_proj(g_a_proj(x)) [1, N, linear_num_heads * linear_head_dim]; o_norm multiplies "
        "the normed core output by their sigmoid, per head (--sublayers)."
    ),
    "model.layers.{i}.mlp.output": (
        "Dense SwiGLU MLP (layers < 3), or routed experts (weighted) + the shared expert; SwiGLU clamps gate <= "
        "swiglu_limit and up to +-swiglu_limit. Before mHC (--sublayers)."
    ),
    "model.layers.{i}.attn_hc.post": (
        "mHC 'post' weights [1, N, hc_mult] = 2 * sigmoid(.), fp32: the attention output's weight per stream."
    ),
    "model.layers.{i}.attn_hc.comb": (
        "mHC stream mixer [1, N, hc_mult, hc_mult], fp32: softmax(.) + hc_eps, then hc_sinkhorn_iters column / row "
        "normalizations. Applied TRANSPOSED: new[k] = sum_j comb[j, k] * streams[j]."
    ),
    "model.layers.{i}.attn_hc.collapsed": (
        "Streams collapsed by the mHC 'pre' weights (sigmoid + hc_eps): sum_j pre_j * streams[j], the attention input "
        "before input_layernorm."
    ),
    "model.layers.{i}.ffn_hc.input": "The mHC streams after the attention write-back, before ffn_hc: the MLP site's input.",
    "model.layers.{i}.ffn_hc.post": "As attn_hc.post, for the MLP site.",
    "model.layers.{i}.ffn_hc.comb": "As attn_hc.comb, for the MLP site.",
    "model.layers.{i}.ffn_hc.collapsed": "As attn_hc.collapsed, for the MLP site (before post_attention_layernorm).",
    "model.layers.{i}.self_attn.forget_gate.output": (
        "KDA per-channel log decay g [1, N, linear_num_heads, linear_head_dim] = linear_lower_bound * sigmoid(exp(A_log) "
        "* (f_b_proj(f_a_proj(x)) + dt_bias)), computed in fp32; the state decays by exp(g) per token."
    ),
    "model.layers.{i}.self_attn.b_proj.output": (
        "KDA beta logits [1, N, linear_num_heads]; the delta-rule write strength is sigmoid(.)."
    ),
    "past_key_values.layers.{i}.conv_states": (
        "KDA short-conv state after the prefill: the last linear_conv_kernel_dim positions of the raw (pre-conv) "
        "projections, channels [q_proj | k_proj | v_proj] (3 * 8192), oldest position first. A decode step convolves "
        "it with the new token."
    ),
    "past_key_values.layers.{i}.recurrent_states": (
        "KDA recurrent state S [1, linear_num_heads, k_dim, v_dim] after the last position, fp32 (the cache casts it "
        "to fp32). q / k are l2-normed and q scaled by k_dim**-0.5 before they meet S."
    ),
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA latent c_kv [1, N, kv_lora_rank], the raw projection before kv_a_layernorm. There is no k_pe part "
        "(qk_rope_head_dim = 0, NoPE)."
    ),
    "past_key_values.layers.{i}.keys": (
        "MLA latent cache [1, 1, N, kv_lora_rank] = kv_a_layernorm(c_kv); kv_b_proj expands it to 64 heads of "
        "[k_nope 256 | v 256]. (The cache's values hold the empty k_pe and are not stored.)"
    ),
    "past_key_values.layers.{i}.indexer_keys": (
        "DSA indexer cache [1, N, 2 * index_head_dim + 1], packed as the model packs it: [k = k_norm(wk(x)) | pool "
        "gate scores = x @ index_kpool_compress_gate.T | valid (1.0 = real token)]. Pools of index_kpool keys are "
        "rebuilt from it every step (softmax(gate + index_kpool_compress_ape) over the pool, weighted sum of k)."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "DSA selection per query: index_topk / index_kpool top-scoring complete pools expanded to key positions "
        "(descending pool score), then the up to index_kpool - 1 keys of the current incomplete pool; -1 marks an "
        "empty slot. Compare each row as a set. When n_tokens <= index_topk every visible key is selected."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, n_routed_experts] = x @ gate.weight.T in fp32, before the sigmoid. Selection: top-k of "
        "sigmoid(logits) + e_score_correction_bias (fp32), n_group = 1."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum + "
        "1e-20, times routed_scaling_factor (2.5). fp32; the shared expert is added unweighted."
    ),
    "model.last_hidden_state": "Final norm of the MEAN of the last layer's hc_mult streams (hc_head has no weights).",
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}

# The KDA functions transformers wraps to prefer hub kernels / fla / causal_conv1d over its torch reference.
KDA_FUNCTIONS = (
    "chunk_kimi_delta_attention",
    "recurrent_kimi_delta_attention",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def torch_kda(modeling) -> str:
    """Make the KDA layers run transformers' pure-torch reference functions, whatever is installed, and say so.

    Each function is wrapped once at import: the wrapper calls `fla` / `causal_conv1d` when the package imports, else
    the torch function it wraps. If anything but that torch function was picked, rebind the module global (the
    layer calls it by name) to the torch function itself. Nothing is changed when the torch path already runs."""
    forced = []
    for name in KDA_FUNCTIONS:
        fn = getattr(modeling, name)
        reference = inspect.unwrap(fn)
        assert reference.__module__ == modeling.__name__, f"{name}: unexpected wrapped function {reference}"
        try:
            chosen = inspect.getclosurevars(fn).nonlocals.get("implementation", reference)
        except TypeError:  # not the expected closure: do not trust it
            chosen = None
        if chosen is not reference:
            setattr(modeling, name, reference)
            forced.append(name)
    if forced:
        print(f"[kda] torch reference forced for {forced} (an optimized kernel was installed)", flush=True)
        return f"transformers torch reference (forced over installed kernels for {', '.join(forced)})"
    return "transformers torch reference (no fla / causal_conv1d / hub kernel in use)"


def segmented_kda(reference, segment: int):
    """chunk_kimi_delta_attention run on ``segment`` positions at a time, each segment starting from the previous
    one's final recurrent state. The torch reference holds [heads, N / 64, 64, 64, 128] fp32 intermediates (~118 GB
    each at N = 56320). Its only cross-chunk dependency is that state, carried in fp32 by its sequential chunk loop,
    so segments that are whole numbers of its 64-position chunks run the same per-chunk ops on the same inputs.
    check_segmented_kda compares with the installed reference at startup."""

    def kda(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, **kwargs):
        assert segment % chunk_size == 0, f"KDA segment {segment} must be a multiple of chunk_size {chunk_size}"
        n = query.shape[1]
        if n <= segment:
            return reference(query, key, value, g, beta, chunk_size, initial_state, output_final_state, **kwargs)
        out, state = [], initial_state
        for s in range(0, n, segment):
            sl = slice(s, s + segment)
            o, state = reference(
                query[:, sl], key[:, sl], value[:, sl], g[:, sl], beta[:, sl], chunk_size, state, True, **kwargs
            )
            out.append(o)
        return torch.cat(out, dim=1), state if output_final_state else None

    kda.segment = segment
    return kda


def check_segmented_kda(reference, device) -> None:
    """segmented_kda against the reference on real KDA widths (64 heads x 128), 1000 positions (a padded last chunk),
    segments that do and do not divide them; on ``device``, since kernel choice is what this guards."""
    gen = torch.Generator().manual_seed(0)
    n, h, d = 1000, 64, 128
    q, k, v = (torch.randn(1, n, h, d, generator=gen).to(device, torch.bfloat16) for _ in range(3))
    g = (-5 * torch.rand(1, n, h, d, generator=gen)).to(device)
    beta = torch.rand(1, n, h, generator=gen).to(device, torch.bfloat16)
    s0 = torch.randn(1, h, d, d, generator=gen).to(device) * 0.1
    with torch.no_grad():
        for state in (None, s0):
            ref = reference(q, k, v, g, beta, 64, state, True, use_qk_l2norm_in_kernel=True)
            for segment in (128, 384, 4096):
                got = segmented_kda(reference, segment)(q, k, v, g, beta, 64, state, True, use_qk_l2norm_in_kernel=True)
                if not (torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])):
                    raise SystemExit(
                        f"segmented KDA (segment {segment}) differs from this transformers' "
                        "chunk_kimi_delta_attention: upstream changed; update segmented_kda"
                    )


def chunked_indexer_forward(chunk: int):
    """Glm5NextTextIndexer.forward with its [B, S, heads, pools] fp32 scores built ``chunk`` query rows at a time
    (~100 GB whole at N = 56320, plus the [B, S, S] visibility). The body is upstream's; every projection (wq_b, wk,
    gate, weights_proj) and the pooled keys still run on the whole sequence, since a bf16 linear rounds differently by
    row count (see the DeepSeek-V4 script). What is sliced reads only its own query row: the fp32 score matmul, the
    head sum, the visibility, top-k and the tail. check_chunked_indexer compares with the installed upstream."""
    import torch.nn.functional as F

    def forward(self, hidden_states, q_resid, attention_mask, past_key_values):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        kv_len = current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()

        valid_keys = packed_states[..., -1].bool()
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
        pool_keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        select_k = min(self.index_topk // self.index_kpool, pool_keys.shape[1])
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        output_width = self.index_topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)

        out = []
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            # get_visible_tokens for query rows s:e (their absolute positions)
            kv_positions = torch.arange(valid_keys.shape[-1], device=valid_keys.device)
            q_positions = current_length - seq_len + torch.arange(s, e, device=valid_keys.device)
            visible_tokens = (kv_positions[None, None, :] <= q_positions[None, :, None]) & valid_keys[:, None, :]

            scores = torch.matmul(q[:, s:e].float(), pool_keys_t)
            scores = F.relu(scores * self.softmax_scale)
            index_scores = torch.matmul(weights[:, s:e].unsqueeze(-2), scores).squeeze(-2)
            del scores
            pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, e - s, -1))
            valid_candidates = pool_visible & pool_valid[:, None]
            index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)

            selected = index_scores.topk(select_k, dim=-1).indices
            selected_valid = valid_candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_idx, selected]
            topk_indices = selected_indices.flatten(-2)
            topk_indices = topk_indices.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
            )
            if self.index_kpool_always_select_tail:
                topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
            topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
            topk_indices = topk_indices[..., :output_width]
            topk_indices = topk_indices.masked_fill(~attention_mask[:, s:e, None], -1)
            out.append(topk_indices.to(torch.int32))
        return torch.cat(out, dim=1)

    forward.chunk = chunk
    return forward


def check_chunked_indexer(upstream, device) -> None:
    """chunked_indexer_forward copies upstream's body; exit if the installed transformers has changed it. Real index
    widths (32 heads x 128 from q LoRA 1536 / hidden 4096, pools of 4, top 2048), 3000 rows so selection is really
    sparse, chunk sizes that do and do not divide them; on ``device``, since kernel choice is what this guards."""
    from transformers import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    cfg = Glm5NextConfig().text_config
    cfg.hidden_size, cfg.q_lora_rank, cfg.index_n_heads, cfg.index_head_dim = 4096, 1536, 32, 128
    cfg.index_topk, cfg.index_kpool, cfg.index_kpool_always_select_tail = 2048, 4, True
    gen = torch.Generator().manual_seed(0)
    indexer = Glm5NextTextIndexer(cfg, 0)
    for p in indexer.parameters():
        p.data = torch.randn(p.shape, generator=gen) * (0.02 if p.dim() == 2 else 0.5)
    indexer = indexer.to(device, torch.bfloat16)
    n = 3000
    h = torch.randn(1, n, 4096, generator=gen).to(device, torch.bfloat16)
    q_resid = torch.randn(1, n, 1536, generator=gen).to(device, torch.bfloat16)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    with torch.no_grad():
        ref = upstream(indexer, h, q_resid, mask, None)
        for chunk in (256, 1000, 4096):
            if not torch.equal(chunked_indexer_forward(chunk)(indexer, h, q_resid, mask, None), ref):
                raise SystemExit(
                    f"chunked indexer (chunk {chunk}) differs from this transformers' Glm5NextTextIndexer.forward: "
                    "upstream changed; update chunked_indexer_forward"
                )


class HcHeadNorm(torch.nn.Module):
    """What Glm5NextTextModel does after its last layer, as one stage: norm(hc_head(streams))."""

    def __init__(self, hc_head: torch.nn.Module, norm: torch.nn.Module):
        super().__init__()
        self.hc_head, self.norm = hc_head, norm

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        return self.norm(self.hc_head(streams))


class Glm53FlashGolden(LayerwiseGolden):
    hf_repo = "zai-org/GLM-5.3-Flash-BF16"
    # No prefix: raw wikitext is what this model predicts best. Wikitext ppl over the same 10100 tokens at isl 10240:
    # 2.87 raw (top-1 72.8%), 3.06 with "[gMASK]<sop>", 17.5 (top-1 51.8%) framed as its own reply by the chat
    # template ("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Continue this WikiText-2 dump verbatim: ...
    # <|assistant|><think></think>"), unlike DeepSeek-V4 and MiniMax-M3, which that framing helps.
    prefix = ""
    recipe = Recipe(
        modules=(
            "self_attn.forget_gate",  # KDA
            "self_attn.b_proj",  # KDA
            "self_attn.kv_a_proj_with_mqa",  # DSA
            "self_attn.indexer",  # DSA
            "mlp.gate",  # MoE
        ),
        sublayers=("self_attn", "mlp", "self_attn.g_b_proj"),  # + attn_hc / ffn_hc, see register_hooks
        routing="mlp.experts",
        cache=("conv_states", "recurrent_states", "keys", "indexer_keys"),  # see cache_tensors
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights", "recurrent_states", "_hc.post", "_hc.comb"),
        config=(
            "layer_types",
            "mlp_layer_types",
            "indexer_types",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_topk",
            "index_n_heads",
            "index_head_dim",
            "index_kpool",
            "index_kpool_always_select_tail",
            "linear_num_heads",
            "linear_head_dim",
            "linear_conv_kernel_dim",
            "linear_lower_bound",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "swiglu_limit",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # The engine (build, --selftest) builds through AutoModelForCausalLM, which has no glm5_next entry (only
        # AutoModelForImageTextToText does). Register the checkpoint's class, as the MiniMax-M3 script does.
        from transformers import AutoModelForCausalLM, Glm5NextConfig, Glm5NextForConditionalGeneration

        AutoModelForCausalLM.register(Glm5NextConfig, Glm5NextForConditionalGeneration, exist_ok=True)

    def generate(self, args):
        from transformers.models.glm5_next import modeling_glm5_next

        kda = torch_kda(modeling_glm5_next)  # before any forward, the selftest's reference included
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # HF_GOLDEN_KDA_CHUNK=<positions>: KDA one segment at a time (segmented_kda); a multiple of 64.
        if segment := int(os.environ.get("HF_GOLDEN_KDA_CHUNK", "0")):
            reference = modeling_glm5_next.chunk_kimi_delta_attention
            check_segmented_kda(reference, device)
            modeling_glm5_next.chunk_kimi_delta_attention = segmented_kda(reference, segment)
            kda += f", run {segment} positions at a time"
            print(f"[kda] {segment} positions at a time", flush=True)
        # HF_GOLDEN_INDEXER_CHUNK=<rows>: DSA indexer scores that many query rows at a time (chunked_indexer_forward).
        if chunk := int(os.environ.get("HF_GOLDEN_INDEXER_CHUNK", "0")):
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

            upstream = getattr(Glm5NextTextIndexer.forward, "upstream", Glm5NextTextIndexer.forward)
            check_chunked_indexer(upstream, device)
            Glm5NextTextIndexer.forward = chunked_indexer_forward(chunk)
            Glm5NextTextIndexer.forward.upstream = upstream
            print(f"[indexer] scores built {chunk} query rows at a time", flush=True)
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["kda_implementation"] = kda
        meta["indexer_chunk"] = chunk or None
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def layer_types(self, text_config) -> list[str]:
        """'<attention>/<mlp>', e.g. 'linear_attention/dense', 'indexed_attention/sparse': both decide what a layer
        stores."""
        n = text_config.num_hidden_layers
        return [f"{a}/{m}" for a, m in zip(text_config.layer_types[:n], text_config.mlp_layer_types[:n])]

    def stages(self, model, backbone, text_config):
        stages = super().stages(model, backbone, text_config)
        name, _, prefix = stages[-2]
        stages[-2] = (name, HcHeadNorm(backbone.hc_head, backbone.norm), prefix)
        return stages

    def convert(self, sd, prefix, module):
        """Checkpoint names -> HF names with transformers' glm5_next renamings; q/k/v_conv1d -> conv1d as its
        Concatenate(dim=0) converter does. Routed experts stay per expert: the Loader fuses them."""
        if isinstance(module, HcHeadNorm):  # the checkpoint's norm.weight is the wrapper's norm.weight
            return {f"norm.{k}": v for k, v in sd.items()}
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key

        renamings = [t for t in get_checkpoint_conversion_mapping("glm5_next") if isinstance(t, WeightRenaming)]
        out = {}
        for k, v in sd.items():
            out[rename_source_key(prefix + k, renamings, [])[0].removeprefix(prefix)] = v
        convs = [f"self_attn.{p}_conv1d.weight" for p in "qkv"]
        if convs[0] in out:
            out["self_attn.conv1d.weight"] = torch.cat([out.pop(c) for c in convs], dim=0)
        return out

    # ---- the forward ---------------------------------------------------------------------------------------------

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """Glm5NextTextModel.forward before its layer loop: one 2D boolean padding mask for both layer types (all
        True: nothing is padded), no RoPE, a DynamicCache with KDA and indexed layers, no top-k yet."""
        from transformers.masking_utils import create_recurrent_attention_mask

        cache = self.make_cache(text_config)
        mask = create_recurrent_attention_mask(
            config=text_config, inputs_embeds=hidden, attention_mask=None, past_key_values=cache
        )
        if mask is None:  # "Guarantee the mask to exist for the indexer"
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        mask = mask.bool()
        return RunState(
            position_ids=position_ids,
            cache=cache,
            masks={"indexed_attention": mask, "linear_attention": mask},
            rope={},
            carry={"prev_topk_indices": None},
        )

    def layer_kwargs(self, i, kind, state):
        return dict(
            attention_mask=state.masks[kind.split("/")[0]],
            position_ids=state.position_ids,
            position_embeddings=None,  # NoPE
            input_ids=None,  # the VLM wrapper calls the text model with inputs_embeds only
            past_key_values=state.cache,
            **state.carry,
        )

    def after_layer(self, i, output, state):
        hidden, state.carry["prev_topk_indices"] = output  # a "shared" indexer layer would reuse this top-k
        return hidden

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        if i == 0:
            # Glm5NextTextModel copies the embeddings into hc_mult streams before layer 0; a no-op when they already
            # are (the selftest's whole-model reference).
            def enter_streams(module, args):
                h = args[0]
                if h.dim() == 3:
                    return (h.unsqueeze(2).expand(-1, -1, module.attn_hc.hc_mult, -1).contiguous(), *args[1:])

            handles.append(layer.register_forward_pre_hook(enter_streams))
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
        """KDA: the conv and recurrent states. DSA: the MLA latent (keys; values hold the empty k_pe) and the packed
        indexer state."""
        if cache is None or i >= len(cache.layers):
            return {}
        layer, key = cache.layers[i], f"past_key_values.layers.{i}"
        if hasattr(layer, "recurrent_states"):
            return {f"{key}.conv_states": layer.conv_states[0], f"{key}.recurrent_states": layer.recurrent_states[0]}
        return {f"{key}.keys": layer.keys, f"{key}.indexer_keys": layer.indexer_keys}

    def free_cache(self, i, cache):
        if cache is None or i >= len(cache.layers):
            return
        layer = cache.layers[i]
        if hasattr(layer, "recurrent_states"):
            layer.conv_states[0] = layer.recurrent_states[0] = None
        else:
            layer.keys = layer.values = layer.indexer_keys = None

    def compare(self, name, got, ref, exact):
        if name.endswith("indexer.output"):  # pools come in score order, padding -1 anywhere: compare rows as sets
            got, ref = got.sort(dim=-1).values, ref.sort(dim=-1).values
        return super().compare(name, got, ref, exact)

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def selftest(self, args, compute, isl=160):
        # 160 tokens: 3 KDA chunks of 64 (the last one padded), 40 indexer pools of which 8 are kept: really sparse.
        return super().selftest(args, compute, isl)

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith(("norm.weight", "layernorm.weight", "k_norm.weight")):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("_hc.scale"):  # O(1) scales so pre / post / comb really depend on the streams
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.2)
            elif name.endswith("_hc.base"):
                t.copy_(torch.randn(t.shape, generator=g))
            elif name.endswith("forget_gate.A_log"):
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)
            elif name.endswith("forget_gate.dt_bias"):
                t.copy_(torch.randn(t.shape, generator=g))
            elif name.endswith("conv1d.weight"):
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)

    def tiny_config(self):
        from transformers import Glm5NextConfig

        # KDA + dense, DSA + dense, KDA + MoE, DSA + MoE; every indexer "full", as the real model.
        layer_types = ["linear_attention", "indexed_attention", "linear_attention"] * 2
        return Glm5NextConfig(
            text_config=dict(
                vocab_size=512,
                pad_token_id=None,
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=32,
                num_hidden_layers=len(layer_types),
                layer_types=layer_types,
                mlp_layer_types=["dense"] * 3 + ["sparse"] * 3,
                indexer_types=["full"] * len(layer_types),
                num_attention_heads=4,
                num_key_value_heads=4,
                q_lora_rank=32,
                kv_lora_rank=16,
                qk_nope_head_dim=16,
                qk_rope_head_dim=0,
                v_head_dim=16,
                index_n_heads=4,
                index_head_dim=16,
                index_topk=32,  # 8 pools of 4 < selftest isl: DSA is really sparse
                index_kpool=4,
                linear_num_heads=4,
                linear_head_dim=16,
                linear_conv_kernel_dim=4,
                linear_lower_bound=-5.0,
                n_routed_experts=8,
                num_experts_per_tok=4,
                n_shared_experts=1,
                routed_scaling_factor=2.5,
                hc_mult=4,
                hc_sinkhorn_iters=20,
                swiglu_limit=10.0,
            ),
            vision_config=dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64),
        )


if __name__ == "__main__":
    sys.exit(Glm53FlashGolden().main())
