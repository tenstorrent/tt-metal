# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 language model (full assembly).

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`language_model_forward`

Qwen2ForCausalLM forward (causal, no KV cache)::

    hidden = embed_tokens(input_ids)
    cos, sin = rotary_emb(position_ids)          # shared across layers
    mask = causal additive mask                  # shared across layers
    for i in range(num_layers):
        hidden = decoder_layer_i(hidden, cos, sin, mask)
    hidden = norm(hidden)                         # final RMSNorm (eps 1e-6)
    logits = lm_head(hidden)

This is the LM ASSEMBLY. It does NOT re-implement any leaf maths -- it imports
and composes the already-verified TTNN modules:

    TtEmbedding   (DRAM gather over embed_tokens.weight)  -- tt/embedding.py
    TtDecoderLayer (N x pre-norm residual block)          -- tt/decoder_layer.py
    TtRMSNorm     (final model.norm, eps 1e-6)            -- tt/rmsnorm.py
    TtLMHead      (untied hidden->vocab linear, no bias)  -- tt/lm_head.py

The final ``model.norm`` (RMSNorm) is applied HERE, before the LM head -- it is
NOT part of the lm_head block (which is the bare projection), matching
``language_model_forward``.

The shared 1D-RoPE cos/sin tables (theta 1e6, head_dim 128) and the additive
causal mask are precomputed once on the host (a parameter-style precompute,
exactly like the cos/sin/mask the standalone decoder_layer block consumed) and
threaded into every decoder layer; each layer hands them to its TtAttention,
which uploads them like its weight tables. The forward path runs entirely with
ttnn ops (no host-side matmul / softmax / activation).

Config (reduced golden runs at num_layers=2; full model has 28):
hidden_size 1536, num_heads 12, num_kv_heads 2, head_dim 128, intermediate 8960,
rope_theta 1e6, rms_norm_eps 1e-6, attention_bias True, vocab_size 151936.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so the sibling
modules cannot be imported via the normal dotted package path -- they are loaded
by file path with importlib (the same convention the tests / decoder_layer use).

Reference TTNN impl this follows: models/tt_transformers/tt/model.py
"""
import importlib.util
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name: str, file_name: str, symbol: str):
    """Import a sibling module by file path (dir name has a dot)."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_TT_DIR, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)


TtEmbedding = _load_module("dots_tt_embedding", "embedding.py", "TtEmbedding")
TtDecoderLayer = _load_module("dots_tt_decoder_layer", "decoder_layer.py", "TtDecoderLayer")
TtRMSNorm = _load_module("dots_tt_rmsnorm", "rmsnorm.py", "TtRMSNorm")
TtLMHead = _load_module("dots_tt_lm_head", "lm_head.py", "TtLMHead")


def _build_rope_tables(seq_len: int, head_dim: int, rope_theta: float):
    """Host-side Qwen2 rotary cos/sin tables matching ``rope_forward``.

    inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)
    freqs    = outer(positions, inv_freq); emb = cat(freqs, freqs)
    returns (cos, sin) each [seq_len, head_dim] in fp32.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos(), emb.sin()


def _build_causal_mask(seq_len: int):
    """Host-side additive causal mask [seq, seq] matching ``language_model_forward``."""
    causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(causal, diagonal=1)


class TtLanguageModel(LightweightModule):
    """dots.ocr Qwen2ForCausalLM (embed -> N decoder layers -> final norm -> lm_head).

    Composes the verified leaf/composite TTNN modules. Weights are supplied via
    the golden's flat ``state_dict`` (HF Qwen2 naming): ``embed_tokens.weight``,
    ``norm.weight``, ``lm_head.weight``, and per-layer
    ``layers.{i}.{input_layernorm,post_attention_layernorm}.weight``,
    ``layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}``,
    ``layers.{i}.self_attn.o_proj.weight``,
    ``layers.{i}.mlp.{gate,up,down}_proj.weight``.

    Args:
        device: ttnn Device or MeshDevice.
        state_dict: flat dict of torch.Tensor weights (HF Qwen2 keys above).
        num_layers: number of decoder layers to assemble (reduced golden = 2).
        seq_len: sequence length of the input.
        num_heads: query heads (12).
        num_kv_heads: KV heads (2).
        head_dim: per-head dim (128).
        rope_theta: rotary base (1e6).
        eps: RMSNorm epsilon (1e-6).
        bias: attention QKV bias present (True).
        dtype: activation/weight dtype (bf16).
        weight_memory_config: storage for weight tables (DRAM by default).
    """

    def __init__(
        self,
        device,
        state_dict,
        num_layers: int,
        seq_len: int,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        eps: float = 1e-6,
        bias: bool = True,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        max_seq_len=None,
    ):
        super().__init__()
        self.device = device
        hidden = num_heads * head_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Shared rotary tables + causal mask (precomputed on host, like the
        # standalone decoder_layer block consumed). Threaded into every layer.
        cos, sin = _build_rope_tables(seq_len, head_dim, rope_theta)
        attention_mask = _build_causal_mask(seq_len)

        # Full-length RoPE tables for the cached AR-decode path (one row indexed
        # per decode step). Built once at max_seq_len and shared into every layer's
        # attention so a single decode step can RoPE its one token at any position.
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else int(seq_len)
        decode_cos, decode_sin = _build_rope_tables(self.max_seq_len, head_dim, rope_theta)

        self.embed_tokens = TtEmbedding(
            device=device,
            weight=state_dict["embed_tokens.weight"],
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )

        self.layers = []
        for i in range(num_layers):
            p = f"layers.{i}."
            layer = TtDecoderLayer(
                device=device,
                input_layernorm_weight=state_dict[p + "input_layernorm.weight"],
                q_weight=state_dict[p + "self_attn.q_proj.weight"],
                k_weight=state_dict[p + "self_attn.k_proj.weight"],
                v_weight=state_dict[p + "self_attn.v_proj.weight"],
                q_bias=state_dict.get(p + "self_attn.q_proj.bias") if bias else None,
                k_bias=state_dict.get(p + "self_attn.k_proj.bias") if bias else None,
                v_bias=state_dict.get(p + "self_attn.v_proj.bias") if bias else None,
                o_weight=state_dict[p + "self_attn.o_proj.weight"],
                post_attention_layernorm_weight=state_dict[p + "post_attention_layernorm.weight"],
                gate_weight=state_dict[p + "mlp.gate_proj.weight"],
                up_weight=state_dict[p + "mlp.up_proj.weight"],
                down_weight=state_dict[p + "mlp.down_proj.weight"],
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                seq_len=seq_len,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                dtype=dtype,
                weight_memory_config=weight_memory_config,
                decode_cos=decode_cos,
                decode_sin=decode_sin,
            )
            self.layers.append(layer)

        # Final model.norm (RMSNorm) -- applied before the LM head.
        self.norm = TtRMSNorm(
            device=device,
            dim=hidden,
            weight=state_dict["norm.weight"],
            eps=eps,
            weight_dtype=dtype,
            weight_memory_config=weight_memory_config,
        )

        # The lm_head's own optimized default is a bfloat8_b weight: the wide
        # 1536 -> 151936 projection is DRAM-bandwidth-bound on the ~233M-param
        # weight read, and bf8 halves that read for a ~-36% traced win on what
        # is ~72% of the whole assembly's device-kernel time. We must NOT force
        # the assembly-wide activation dtype (bf16) onto it -- that would revert
        # the inherited lm_head optimization. The bf8 weight is safe because the
        # logits feed an argmax downstream (PCC holds at 0.99991). The activation
        # stays bf16 (handled inside TtLMHead.forward) and the output is bf16.
        self.lm_head = TtLMHead(
            device=device,
            weight=state_dict["lm_head.weight"],
            weight_memory_config=weight_memory_config,
        )

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """input_ids: row-major uint32 [batch, seq_len] -> logits [seq, vocab].

        embed -> N x decoder_layer -> final RMSNorm -> lm_head.
        """
        # Gather -> [seq, hidden] (TILE layout) for the decoder stack.
        hidden = self.embed_tokens(input_ids)
        hidden = ttnn.reshape(hidden, (hidden.shape[-2], hidden.shape[-1]))

        for layer in self.layers:
            hidden = layer(hidden)

        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------ #
    # Cached AR-decode (O(1) per step) entered from embeddings.          #
    # ------------------------------------------------------------------ #
    def prefill_from_embeds(self, hidden: ttnn.Tensor, kv_cache):
        """Prefill from inputs_embeds [prompt_len, hidden]; populate ``kv_cache``.

        Runs the full-causal forward over the prompt while writing each layer's
        K/V into the cache, then norm + lm_head. Returns logits [prompt_len, vocab]
        (only the last row is used for the first generated token).
        """
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.prefill_kv(hidden, kv_cache, layer_idx)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def decode_step(self, hidden: ttnn.Tensor, pos: int, kv_cache):
        """One cached decode step from a single token's embed [1, hidden].

        Reads/writes ``kv_cache`` at ``pos`` per layer (O(1) layer-runs), then
        norm + lm_head. Returns logits [1, vocab].
        """
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.forward_decode(hidden, pos, kv_cache, layer_idx)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)
