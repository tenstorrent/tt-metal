#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for Qwen/Qwen3.8-Flash-Next, layer by layer (engine.py), from transformers' qwen4_exp code.

Stored tensors (N = isl, i = layer; 48 layers: Gated DeltaNet linear attention on 36 of them, gated GQA with a QSA
block-sparse indexer on 3, 7, 11, ..., 47; MoE (512 routed experts, top 10, 1 gated shared expert) on every layer;
the residual is hc_count = 4 hyper-connection streams, laid side by side in one [.., 4 * 2560] axis; layer 1 also adds
the PLE n-gram embedding):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 2560]       bf16   embeddings (no scale); the model
                                                                                     repeats them into the 4 streams
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 10240]      bf16   all   the 4 residual streams
                                                                                     [s0 | s1 | s2 | s3] after the layer
    model.layers.{i}.linear_attn.output                    [1, N, 2560]       bf16   GDN   --sublayers: after out_proj
    model.layers.{i}.self_attn.output                      [1, N, 2560]       bf16   attn  --sublayers: after the
                                                                                     sigmoid output gate and o_proj
    model.layers.{i}.mlp.output                            [1, N, 2560]       bf16   all   --sublayers: routed experts
                                                                                     + gated shared expert
    model.layers.{i}.ple.output                            [1, N, 10240]      bf16   1     --sublayers: what PLE adds to
                                                                                     the streams before attention
    model.layers.{i}.{attn,mlp}_hyper_connection.mixed_input  [1, N, 2560]   bf16   all   --sublayers: the block input
                                                                                     mixed from the normed streams
    model.layers.{i}.{attn,mlp}_hyper_connection.injection_weights  [1, N, 4]  bf16 all   --sublayers: the block
                                                                                     output's weight per stream
  cache/layer_XX.safetensors
    model.layers.{i}.linear_attn.in_proj_a.output          [1, N, 48]         bf16   GDN   decay logits a: g =
                                                                                     -exp(A_log) * softplus(a + dt_bias)
    model.layers.{i}.linear_attn.in_proj_b.output          [1, N, 48]         bf16   GDN   beta logits (sigmoid)
    past_key_values.layers.{i}.conv_states                 [1, 10240, 4]      bf16   GDN   short-conv state: the last 4
                                                                                     positions of in_proj_qkv, pre-conv
    past_key_values.layers.{i}.recurrent_states            [1, 48, 128, 128]  fp32   GDN   delta-rule state per value
                                                                                     head [k_dim, v_dim] after N tokens
    past_key_values.layers.{i}.keys                        [1, 2, N, 256]     bf16   attn  k_norm'd, RoPE'd keys
    past_key_values.layers.{i}.values                      [1, 2, N, 256]     bf16   attn
    past_key_values.layers.{i}.indexer_keys                [1, N, 128]        bf16   attn  QSA raw token keys (no norm,
                                                                                     no RoPE: both after pooling)
    model.layers.{i}.self_attn.indexer.output              [1, N, 2051]       int32  attn  key positions each query
                                                                                     attends to, ascending, -1 = none
    model.layers.{i}.ple.ple_embedding.ngram_ids           [1, N, 16]         int64  1     hashed 2-gram (heads 0-7) and
                                                                                     3-gram (8-15) rows of the table
    model.layers.{i}.ple.ple_embedding.output              [1, N, 2560]       bf16   1     their 16 x 160 embeddings
    past_key_values.layers.{i}.ple_conv_states             [1, 10240, 9]      bf16   1     PLE dilated-conv state
    past_key_values.layers.{i}.ngram_context               [1, 2]             int64  1     the last 2 token ids
    model.layers.{i}.mlp.gate.output                       [N, 512]           fp32   all   router logits (computed bf16)
    model.layers.{i}.mlp.experts.top_k_index               [N, 10]            int32  all   chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 10]            fp32   all   their weights, same order
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 2560]       bf16   hyper_connection_mixer(streams);
                                                                                     there is no final norm
    logits                                                 [1, 1, 248320]     fp32   last position, untied lm_head

GDN = layer_types "linear_attention" (layers 0-2, 4-6, ..., 44-46); attn = "indexed_attention" (the config's
"full_attention", renamed by the config class): 3, 7, ..., 47. metadata.json lists the layers each tensor exists on.

REQUIRES transformers >= 5.19 (qwen4_exp is in-library from 5.19; the checkpoint was saved with 5.8.0.dev0). tt-metal's
python_env pins an older one, so run through a uv overlay that adds it on top of python_env, without touching it:

    uv run --no-project --python python_env/bin/python --with transformers==5.19.0 \\
        python models/common/hf_golden/generate_golden_qwen3_8_flash_next.py --isl 10240

What is model-specific, and why it is handled here:

  * Qwen4ExpTextModel.forward, before its layer loop (what prepare() mirrors): position ids [0, N) for the text row
    and the 3 mRoPE rows (text only, so all four are equal); a 4D causal mask that is never skipped
    (allow_is_causal_skip=False: the indexer reads it row by row and overlays its selection), given to every layer;
    create_recurrent_attention_mask (None without padding) as the GDN / PLE conv_mask; (cos, sin) for ALL positions,
    because the indexer needs the RoPE of every key; a DynamicCache(config) with LinearAttentionLayer (3 conv states:
    GDN, PLE conv, n-gram context) on GDN layers and DynamicIndexedLayer on attention layers. The embeddings are
    repeated into the 4 streams (a pre-hook on layer 0, as the model does). Every layer gets the token ids
    (ple_input_ids). The final stage is hyper_connection_mixer: there is no final norm, the mixer's grouped RMSNorm is
    the last normalization.
  * Hyper-connections (Qwen4ExpTextGatedResidual), at each block (attention, MLP): x = grouped RMSNorm of the 4
    streams (one 2560 group each, weight 1 + w); mix = sigmoid(up(silu(down(x) / 4))) [.., 4, 2560]; block input =
    mean over streams of mix * x; injection = 2 * sigmoid(inject(x) / 4) [.., 4]; streams += injection[k] * block
    output (added to the un-normed streams). The final mixer is the same without the injection.
  * Gated DeltaNet (Qwen3-Next): in_proj_qkv -> depthwise causal conv (kernel 4) + SiLU; 16 key heads repeated to 48
    value heads, q / k l2-normed in the kernel; beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) in fp32;
    chunked delta rule (chunk 64, fp32 state); RMSNorm of each head gated by sigmoid(z) (output_gate_type sigmoid);
    out_proj. The kernels are transformers' PURE-TORCH reference functions. They are wrapped to prefer hub kernels /
    the `fla` and `causal_conv1d` packages, and fla IS installed in python_env; torch_gdn() rebinds the torch
    reference whenever anything else would run, so CPU and GPU compute the same algorithm. metadata.json records it
    ("gdn_implementation").
  * Attention: q_proj gives 24 heads of [q 256 | gate 256]; q_norm / k_norm (RMSNorm, 1 + w); partial RoPE on the
    first 64 of 256 dims (rotate-half), theta 1e7, mRoPE sections [11, 11, 10] interleaved (all equal for text);
    2 KV heads; output * sigmoid(gate) before o_proj. The keys / values cache holds the full N positions.
  * QSA indexer (Qwen4ExpTextQSAIndexer): index_qk_proj -> 4 query heads + 1 key head of 128; q is RMSNorm'd and
    RoPE'd. Per query, its visible keys are cut into blocks of 4 (oldest first); each block's raw keys are averaged
    (fp32), RMSNorm'd and RoPE'd at the block's first position; score = sum over heads of relu(q . k_block) / sqrt(128);
    the top 512 blocks plus the up-to-3 keys of the incomplete last block are selected (2051 at most); attention is
    masked to them. The model returns this as a mask; the stored output is each row's selected positions, sorted
    ascending (upstream's internal order is by score), -1 padded. For N <= 2048 every visible key is selected.
    Upstream loops over the queries one at a time (~0.4 ms each on an H200, ~22 s per attention layer at N = 56320);
    it runs as-is.
  * PLE (layer 1, ple_layer_ids [2]): per position, 2-gram and 3-gram token hashes (8 heads each, multiplier / XOR /
    prime-modulo, a sequence restarting after every <|endoftext|>, which the hash also pads with) index one table of
    ~320M rows x 160; the 16 rows are concatenated, projected to one value [2560] and a key per stream; the gate per
    stream is a signed sqrt of the scaled key . RMSNorm(stream) dot product; sigmoid(gate) * value goes through a
    dilated depthwise conv (kernel 4, dilation 3, SiLU) and is added to the streams. The table is 102 GB (128
    checkpoint shards, which from_pretrained concatenates). A lookup is a gather, so this script hashes the prompt
    with upstream's own code (ngram_ids()) and loads only the rows it uses (<= 16 N; ~900k at N = 56320, ~0.3 GB):
    SparseRows maps each id to its row and gathers it, the same values as nn.Embedding over the whole table. The
    checkpoint's layer_multipliers are loaded (and checked against the hash's own derivation); the per-head vocab
    sizes and offsets are not in the checkpoint and are rebuilt as _init_weights does.
  * MoE: router logits = x @ gate.weight.T in bf16 (stored fp32, exact); softmax in fp32, top 10, renormalized, cast to
    bf16 (the stored weights are those bf16 values in fp32); SwiGLU experts (640); shared expert (640) scaled by
    sigmoid(shared_expert_gate(x)) and added.
  * --prefill-chunk: each chunk's attention sees keys [0, e) through the cache, and its (cos, sin) are the full [0, e)
    positions, as the model passes them in a multi-step prefill (the indexer RoPEs pooled keys at their own positions);
    chunk_kwargs() cuts them so. GDN, the PLE conv and the n-gram hash carry their states through the cache.
  * Weights ship bf16. The vision tower and the MTP layer (mtp.*) are never built into a stage nor read.

    python generate_golden_qwen3_8_flash_next.py --isl 10240
    python generate_golden_qwen3_8_flash_next.py --isl 2048 --num-layers 4 --root /tmp/smoke   # GDN, GDN + PLE, GDN,
                                                                                                # attention + indexer
    python generate_golden_qwen3_8_flash_next.py --selftest --prefill-chunk 64
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import engine  # noqa: E402
from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added); the model then repeats them into the hc_count streams.",
    "model.layers.{i}.output": (
        "Decoder layer output: the hc_count residual streams side by side [1, N, hc_count * hidden] (stream k = "
        "channels k * hidden ...), after PLE (layer 1), attention and MLP, each block adding injection_weights[k] * "
        "its output to stream k. The next layer's input."
    ),
    "model.layers.{i}.linear_attn.output": "Gated DeltaNet output after the gated RMSNorm and out_proj (--sublayers).",
    "model.layers.{i}.self_attn.output": (
        "Attention output: sparse (indexer-selected) GQA, times sigmoid(gate) from q_proj's second half, then o_proj "
        "(--sublayers)."
    ),
    "model.layers.{i}.mlp.output": "Routed experts (weighted) + sigmoid(shared_expert_gate(x)) * shared expert (--sublayers).",
    "model.layers.{i}.ple.output": "What PLE adds to the streams before the attention block: gated value + its dilated conv (--sublayers).",
    "model.layers.{i}.attn_hyper_connection.mixed_input": (
        "Attention block input: mean over streams of sigmoid(up(silu(down(x) / hc_count))) * x, x = the streams after "
        "the grouped hc_norm (--sublayers)."
    ),
    "model.layers.{i}.attn_hyper_connection.injection_weights": (
        "2 * sigmoid(block_inject_weight(x) / hc_count) [1, N, hc_count]: the attention output's weight per stream "
        "(--sublayers)."
    ),
    "model.layers.{i}.mlp_hyper_connection.mixed_input": "As attn_hyper_connection.mixed_input, for the MLP block (--sublayers).",
    "model.layers.{i}.mlp_hyper_connection.injection_weights": "As attn_hyper_connection.injection_weights, for the MLP block (--sublayers).",
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
    "past_key_values.layers.{i}.keys": "Attention keys after k_norm and partial RoPE [1, num_key_value_heads, N, head_dim].",
    "past_key_values.layers.{i}.values": "Attention values [1, num_key_value_heads, N, head_dim].",
    "past_key_values.layers.{i}.indexer_keys": (
        "QSA indexer token keys [1, N, indexer_head_dim]: index_qk_proj's key head, raw. Blocks of "
        "indexer_compress_ratio are averaged, then k_layernorm'd and RoPE'd at the block's first position."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "QSA selection per query: the positions of the top indexer_budget / indexer_compress_ratio blocks and of the "
        "incomplete last block, ASCENDING (upstream orders them by score), -1 = empty slot. Attention sees exactly "
        "these keys. When n_tokens <= indexer_budget every visible key is selected."
    ),
    "model.layers.{i}.ple.ple_embedding.ngram_ids": (
        "PLE table rows per position [1, N, 2 * heads_per_ngram]: heads 0-7 hash the 2-gram ending here, 8-15 the "
        "3-gram, each modulo its own prime size plus its head offset (tokens before the start or an <|endoftext|> "
        "read as <|endoftext|>)."
    ),
    "model.layers.{i}.ple.ple_embedding.output": "The 2 * heads_per_ngram table rows of ngram_ids, concatenated [1, N, ple_embed_dim].",
    "past_key_values.layers.{i}.ple_conv_states": (
        "PLE conv state: the last (ple_conv_kernel_size - 1) * ngram_size positions of the normed gated value, before "
        "the dilated conv, oldest first."
    ),
    "past_key_values.layers.{i}.ngram_context": "PLE n-gram hash context: the last ngram_size - 1 token ids.",
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, num_experts] = x @ gate.weight.T, computed in bf16 (stored fp32). Selection: softmax in "
        "fp32, top num_experts_per_tok."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: their softmax probabilities renormalized to sum 1, cast to bf16 (stored "
        "fp32, exact)."
    ),
    "model.last_hidden_state": "hyper_connection_mixer(last layer's streams): hc_norm, mixed down to one stream. No final norm.",
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}

# The Gated DeltaNet functions transformers wraps to prefer hub kernels / fla / causal_conv1d over its torch reference.
GDN_FUNCTIONS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)
NGRAM = re.compile(r"\.ngram_embedding\.(?:shard_(\d+)\.)?weight$")


def torch_gdn(modeling) -> str:
    """Make the Gated DeltaNet layers run transformers' pure-torch reference functions, whatever is installed, and say
    so. As the GLM-5.3 script's torch_kda: each function is wrapped once at import to call fla / causal_conv1d / a hub
    kernel when available; if anything but the torch function it wraps was picked, rebind the module global (the layer
    calls it by name) to that torch function."""
    forced = []
    for name in GDN_FUNCTIONS:
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
        print(f"[gdn] torch reference forced for {forced} (an optimized kernel was installed)", flush=True)
        return f"transformers torch reference (forced over installed kernels for {', '.join(forced)})"
    return "transformers torch reference (no fla / causal_conv1d / hub kernel in use)"


class SparseRows(torch.nn.Module):
    """The rows of an embedding table a prompt uses: ``ids`` (ascending) and their rows ``weight``. A lookup maps each
    id to its row and gathers it, so it returns what nn.Embedding over the whole table returns, bit for bit."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, dim))
        self.ids = torch.nn.Buffer(torch.empty(n, dtype=torch.long))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = torch.searchsorted(self.ids, ids)
        if not torch.equal(self.ids[rows.clamp_max(self.ids.numel() - 1)], ids):
            raise RuntimeError("PLE n-gram id outside the rows loaded for this prompt")
        return torch.nn.functional.embedding(rows, self.weight)


def ngram_ids(embedding, ple_input_ids: torch.Tensor, buffers: dict[str, torch.Tensor]) -> torch.Tensor:
    """The table rows Qwen4ExpTextNGramEmbedding.forward looks up for these token ids, from no cache (what a chunked
    prefill computes too: its cached context is the previous tokens). Upstream's own forward and shift, run on a
    stand-in whose table only records the ids it is asked for."""
    cls = type(embedding)

    class Record:
        weight = torch.empty(0, device=ple_input_ids.device)

        def __call__(self, ids):
            self.ids = ids
            return ids.new_zeros((*ids.shape, 0), dtype=torch.float32)

    stand_in = types.SimpleNamespace(
        **{k: getattr(embedding, k) for k in ("layer_idx", "ngram_size", "context_len", "heads_per_ngram", "eos_token_id")},
        **buffers,
        ngram_embedding=Record(),
    )
    stand_in._shift_right_ignore_eos = functools.partial(cls._shift_right_ignore_eos, stand_in)
    cls.forward(stand_in, ple_input_ids, None)
    return stand_in.ngram_embedding.ids


def read_rows(shards, keys: list[str], ids: torch.Tensor) -> torch.Tensor:
    """Rows ``ids`` of the table stored as ``keys`` (shard_0, shard_1, ... concatenated along dim 0, as from_pretrained's
    Concatenate does, or one whole weight), read through a memory map: only the pages holding them."""
    ids = ids.cpu().numpy()
    out, start = [], 0
    for key in keys:
        path = shards.fetch(shards.weight_map[key])
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))[key]
        rows, dim = header["shape"]
        dtype = common.SAFETENSORS_DTYPES[header["dtype"]]
        itemsize = torch.empty(0, dtype=dtype).element_size()
        a, b = header["data_offsets"]
        assert b - a == rows * dim * itemsize, f"{key}: unexpected layout {header}"
        table = np.memmap(path, dtype=np.dtype(f"V{itemsize}"), mode="r", offset=8 + n + a, shape=(rows, dim))
        local = ids[(ids >= start) & (ids < start + rows)] - start
        out.append(torch.from_numpy(np.ascontiguousarray(table[local]).view(np.uint8)).view(dtype).view(-1, dim))
        start += rows
    assert ids.max() < start, f"n-gram id {ids.max()} past the table's {start} rows"
    return torch.cat(out)


class PleSkippingShards(common.Shards):
    """Shards that never read the PLE n-gram table with a stage: read_rows reads what a prompt uses. A file holding
    only table shards is not opened; one holding table shards and other tensors of the layer is read whole, and
    convert() drops the shards (a few GB, once)."""

    current = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PleSkippingShards.current = self

    def files(self, prefix):
        return sorted({f for k, f in self.weight_map.items() if k.startswith(prefix) and not NGRAM.search(k)})


class Qwen38FlashNextGolden(LayerwiseGolden):
    hf_repo = "Qwen/Qwen3.8-Flash-Next"
    # The text as the model's own reply: the chat template (enable_thinking=False: an empty think block) up to the
    # assistant turn, the user turn naming the WikiText format. Wikitext ppl over the same 10100 tokens at isl 10240
    # (chunked 5120): 3.99 (top-1 65.3%) with it, 4.30 raw, 4.40 after <|endoftext|> (the checkpoint has no BOS).
    prefix = (
        "<|im_start|>user\nContinue this WikiText-2 dump verbatim: a sequence of Wikipedia articles, one after "
        "another, each starting with a = Title = line, in raw formatting ( @-@ , = = headings = = ).<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    recipe = Recipe(
        modules=(
            "linear_attn.in_proj_a",  # GDN
            "linear_attn.in_proj_b",  # GDN
            "ple.ple_embedding",  # layer 1
            "mlp.gate",
        ),
        # + {attn,mlp}_hyper_connection outputs, see register_hooks
        sublayers=("linear_attn", "self_attn", "mlp", "ple"),
        routing="mlp.experts",
        cache=("conv_states", "recurrent_states", "keys", "values", "indexer_keys"),  # see cache_tensors
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights", "recurrent_states"),
        config=(
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_parameters",
            "partial_rotary_factor",
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "output_gate_type",
            "hc_count",
            "hc_lowrank",
            "ple_layer_ids",
            "ple_embed_dim",
            "ple_conv_kernel_size",
            "ngram_size",
            "heads_per_ngram",
            "ngram_vocab_size_base",
            "make_ngram_vocab_size_divisible_by",
            "seed",
            "split_ngram_parts",
            "eos_token_id",
            "rms_norm_eps",
            "hidden_act",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "norm_topk_prob",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        # AutoModelForCausalLM maps qwen4_exp to Qwen4ExpForCausalLM, whose names (model.layers.*) are not the
        # checkpoint's (model.language_model.layers.*); register the checkpoint's class, as the GLM-5.3 script does.
        from transformers import AutoModelForCausalLM, Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

        AutoModelForCausalLM.register(Qwen4ExpConfig, Qwen4ExpForConditionalGeneration, exist_ok=True)
        # The engine's final stage (and the selftest's reference hook) is backbone.norm; here that is the stream mixer.
        Qwen4ExpTextModel.norm = property(lambda self: self.hyper_connection_mixer)
        engine.Shards = PleSkippingShards

    def generate(self, args):
        from transformers.models.qwen4_exp import modeling_qwen4_exp

        gdn = torch_gdn(modeling_qwen4_exp)  # before any forward, the selftest's reference included
        self.ple_rows = {}
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["gdn_implementation"] = gdn
        meta["ple_rows_read"] = self.ple_rows
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def build(self, config, dtype):
        model = super().build(config, dtype)
        for layer in model.get_decoder().layers:
            if layer.ple is not None:  # the table is read row by row for the prompt: see read_rows
                emb = layer.ple.ple_embedding
                emb.ngram_embedding = SparseRows(0, emb.ngram_embedding.embedding_dim)
        return model

    def convert(self, sd, prefix, module):
        """Checkpoint names are the model's, except the PLE table: its shards are dropped (PleSkippingShards may have
        read some) and the rows this prompt uses are read instead. The table's buffers come from the checkpoint where
        it has them (layer_multipliers, as from_pretrained loads them), else from the module, as _init_weights builds
        them; the multipliers must equal the hash's own derivation."""
        ple = getattr(module, "ple", None)
        if ple is None:
            return sd
        from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers

        emb, base = ple.ple_embedding, "ple.ple_embedding."
        for k in [k for k in sd if NGRAM.search(k)]:
            del sd[k]
        derived = {
            "layer_multipliers": _build_layer_multipliers(
                emb.unigram_vocab_size, emb.ngram_size, emb.ple_layer_index, emb.seed
            ),
            "ngram_heads_vocab_sizes": torch.tensor(emb.head_vocab_sizes, dtype=torch.long),
            "ngram_heads_offsets": torch.tensor(emb.head_offsets, dtype=torch.long),
        }
        device = self.state.input_ids.device
        buffers = {}
        for name, value in derived.items():
            stored = sd.setdefault(base + name, value)
            if not torch.equal(stored.cpu().long(), value):
                raise SystemExit(f"{prefix}{base}{name}: checkpoint {stored.tolist()} != derived {value.tolist()}")
            buffers[name] = stored.to(device)
        ids = ngram_ids(emb, self.ple_input_ids(self.state), buffers).flatten().unique()  # ascending
        shards = PleSkippingShards.current
        keys = [k for k in shards.weight_map if k.startswith(prefix + base) and NGRAM.search(k)]
        keys.sort(key=lambda k: int(NGRAM.search(k)[1] or 0))
        sd[base + "ngram_embedding.weight"] = read_rows(shards, keys, ids)
        sd[base + "ngram_embedding.ids"] = ids
        with torch.device("meta"):
            emb.ngram_embedding = SparseRows(ids.numel(), emb.ngram_embedding.weight.shape[1])
        self.ple_rows[prefix] = ids.numel()
        print(f"[ple] {ids.numel()} n-gram rows read for {prefix}", flush=True)
        return sd

    # ---- the forward ---------------------------------------------------------------------------------------------

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """Qwen4ExpTextModel.forward before its layer loop, for text-only input_ids (the VLM wrapper passes no
        position ids then): positions [0, N) for the text row and the 3 mRoPE rows, kept on the cache as the model
        does; the never-skipped 4D causal mask; the recurrent conv_mask (None: nothing is padded); (cos, sin) over
        every position."""
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        cache = self.make_cache(text_config)
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        text_position_ids, mrope_position_ids = position_ids[0], position_ids[1:]
        cache.position_ids = mrope_position_ids
        mask_kwargs = dict(
            config=text_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=text_position_ids,
            allow_is_causal_skip=False,
        )
        masks = {
            "indexed_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        rope = backbone.rotary_emb(hidden, mrope_position_ids)
        eos = text_config.eos_token_id
        self.eos = eos[0] if isinstance(eos, list) else eos
        self.state = RunState(position_ids=text_position_ids, cache=cache, masks=masks, rope={"all": rope})
        return self.state

    def ple_input_ids(self, state):
        """The token ids PLE hashes: the model replaces positions conv_mask pads with <|endoftext|>."""
        ids, conv_mask = state.input_ids, state.masks["linear_attention"]
        if conv_mask is not None:
            ids = torch.where(conv_mask.bool(), ids, self.eos)
        return ids

    def layer_kwargs(self, i, kind, state):
        """What Qwen4ExpTextModel passes every layer: the attention mask for "indexed_attention" layers (the GDN ones
        ignore it), conv_mask, full-length (cos, sin), the cache and the token ids."""
        return dict(
            position_embeddings=state.rope["all"],
            attention_mask=state.masks["indexed_attention"],
            conv_mask=state.masks["linear_attention"],
            past_key_values=state.cache,
            ple_input_ids=self.ple_input_ids(state),
        )

    def chunk_kwargs(self, kwargs, s, e, n):
        """Rows [s, e) of a multi-step prefill: mask rows s:e over keys :e, (cos, sin) for positions [0, e) (the
        model's are always the full positions so far; attention takes their last e - s rows), conv_mask and token ids
        for s:e."""
        out = dict(kwargs)
        out["attention_mask"] = kwargs["attention_mask"][:, :, s:e, :e]
        out["position_embeddings"] = tuple(t[:, :e] for t in kwargs["position_embeddings"])
        if kwargs["conv_mask"] is not None:
            out["conv_mask"] = kwargs["conv_mask"][:, s:e]
        out["ple_input_ids"] = kwargs["ple_input_ids"][:, s:e]
        return out

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        if i == 0:
            # Qwen4ExpTextModel repeats the embeddings into hc_count streams before layer 0; a no-op when they already
            # are (the selftest's whole-model reference).
            def enter_streams(module, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                hc = module.attn_hyper_connection.hc_count
                if h.shape[-1] == hc * module.attn_hyper_connection.hidden_size:
                    return None
                h = h.repeat(1, 1, hc)
                return ((h, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})

            handles.append(layer.register_forward_pre_hook(enter_streams, with_kwargs=True))
        if layer.layer_type == "indexed_attention":
            indexer = layer.self_attn.indexer
            width = indexer.token_budget + indexer.compress_ratio - 1

            def selection(_m, _a, mask, key=f"model.layers.{i}.self_attn.indexer.output"):
                captured[key] = selected_positions(mask, width)

            handles.append(indexer.register_forward_hook(selection))
        if layer.ple is not None:

            def ids(_m, args, key=f"model.layers.{i}.ple.ple_embedding.ngram_ids"):
                captured[key] = args[0]

            handles.append(layer.ple.ple_embedding.ngram_embedding.register_forward_pre_hook(ids))
        if sublayers:
            for site in ("attn_hyper_connection", "mlp_hyper_connection"):

                def put(_m, _a, o, key=f"model.layers.{i}.{site}"):
                    captured[f"{key}.mixed_input"], captured[f"{key}.injection_weights"] = o[0], o[2]

                handles.append(getattr(layer, site).register_forward_hook(put))
        return handles

    def is_sublayer(self, key):
        return super().is_sublayer(key) or "_hyper_connection." in key

    def cache_tensors(self, i, cache):
        """GDN: the conv and recurrent states (+ on the PLE layer its conv state and n-gram context). Attention: keys,
        values and the indexer's raw keys."""
        if cache is None or i >= len(cache.layers):
            return {}
        layer, key = cache.layers[i], f"past_key_values.layers.{i}"
        if hasattr(layer, "recurrent_states"):
            out = {f"{key}.conv_states": layer.conv_states[0], f"{key}.recurrent_states": layer.recurrent_states[0]}
            if layer.conv_states.get(1) is not None:
                out[f"{key}.ple_conv_states"] = layer.conv_states[1]
                out[f"{key}.ngram_context"] = layer.conv_states[2]
            return out
        return {f"{key}.keys": layer.keys, f"{key}.values": layer.values, f"{key}.indexer_keys": layer.indexer_keys}

    def free_cache(self, i, cache):
        if cache is None or i >= len(cache.layers):
            return
        layer = cache.layers[i]
        if hasattr(layer, "recurrent_states"):
            for states in (layer.conv_states, layer.recurrent_states):
                for k in states:
                    states[k] = None
        else:
            layer.keys = layer.values = layer.indexer_keys = None

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def selftest(self, args, compute, isl=192):
        # 192 tokens: 3 GDN chunks of 64; indexer budget 16 = 4 blocks of 4 out of up to 48: really sparse. With
        # --prefill-chunk 64 the layerwise run is a 3-step prefill, compared with one whole-model forward.
        return super().selftest(args, compute, isl)

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in model.named_parameters():
            if name.endswith("A_log"):
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)
            elif name.endswith("dt_bias"):
                t.copy_(torch.randn(t.shape, generator=g))
            elif name.endswith("conv1d.weight"):  # the PLE conv is zero-initialized: make it matter
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)

    def tiny_config(self):
        from transformers import Qwen4ExpConfig

        # GDN, GDN + PLE, GDN, attention + indexer, twice: the real model's order.
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        return Qwen4ExpConfig(
            text_config=dict(
                vocab_size=512,
                hidden_size=64,
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
                moe_intermediate_size=32,
                shared_expert_intermediate_size=32,
                num_experts=8,
                num_experts_per_tok=3,
                hc_count=4,
                hc_lowrank=16,
                ple_layer_ids=[2],
                ple_embed_dim=64,
                ngram_vocab_size_base=1000,
                split_ngram_parts=4,
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=16,
                indexer_budget=16,
                indexer_compress_ratio=4,
                eos_token_id=7,
                bos_token_id=7,
                output_gate_type="sigmoid",
                max_position_embeddings=4096,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=1e7,
                    partial_rotary_factor=0.25,
                    mrope_section=[2, 1, 1],
                    mrope_interleaved=True,
                ),
            ),
            vision_config=dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64),
        )


def selected_positions(mask: torch.Tensor, width: int, rows: int = 1024) -> torch.Tensor:
    """The indexer's [B, 1, S, kv] selection mask (bool for sdpa; additive 0 / min for eager) as each row's selected
    key positions, ascending, -1 padded to ``width`` columns. int32 [B, S, width]."""
    sel = mask[:, 0] if mask.dtype == torch.bool else mask[:, 0] == 0
    if (most := sel.sum(-1).max().item()) > width:
        raise RuntimeError(f"indexer selected {most} keys in a row, more than {width}")
    kv = sel.shape[-1]
    positions = torch.arange(kv, device=sel.device)
    out = []
    for s in range(0, sel.shape[1], rows):
        p = torch.where(sel[:, s : s + rows], positions, kv).sort(dim=-1).values[..., :width]
        p = torch.nn.functional.pad(p, (0, width - p.shape[-1]), value=kv)
        out.append(torch.where(p == kv, -1, p).to(torch.int32))
    return torch.cat(out, dim=1)


if __name__ == "__main__":
    sys.exit(Qwen38FlashNextGolden().main())
