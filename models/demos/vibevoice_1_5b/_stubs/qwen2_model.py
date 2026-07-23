# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_model` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.language_model`, a standard HF
`transformers.models.qwen2.modeling_qwen2.Qwen2Model`
(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
num_key_value_heads=2, head_dim=128, intermediate_size=8960,
rms_norm_eps=1e-6):

    x = embed_tokens(input_ids)
    position_embeddings = rotary_emb(x, position_ids)     # (cos, sin), RoPE, attention_scaling=1.0
    causal_mask = additive upper-triangular(-inf) mask     # sdpa causal self-attention (verified against
                                                            # the HF reference: token i's output is
                                                            # invariant to changes at position j > i)
    for layer in layers[:num_hidden_layers]:
        x = layer(x, attention_mask=causal_mask, position_embeddings=position_embeddings)
    x = norm(x)                                            # RMSNorm

Each `layer` is now COMPOSED from the already-graduated child stub
`_stubs/qwen2_decoder_layer.build` (GQA + RoPE + SwiGLU). qwen2_model owns the
embedding, the rotary_emb (cos/sin), the causal-mask construction, and the final
RMSNorm; the causal mask is threaded into every layer via the child's
`attention_mask` kwarg (the child adds it to the scaled scores before softmax).
The child's cos/sin convention (position_embeddings=(cos, sin), each shaped
[B, 1, T, head_dim] and reshaped to that internally) matches what qwen2_model
already produces, so no call-site adaptation is needed.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vibevoice_1_5b._stubs.qwen2_decoder_layer import build as _build_qwen2_decoder_layer

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _row(t_1d, device):
    return ttnn.from_torch(
        t_1d.detach().float().reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )


def _rms_norm(x, weight_row, eps, compute_config=None):
    return ttnn.rms_norm(x, epsilon=eps, weight=weight_row, compute_kernel_config=compute_config, memory_config=_DRAM)


def build(device, torch_module):
    """Bind the trained Qwen2Model weights and return a native ttnn forward closure."""
    m = torch_module
    num_layers = int(m.config.num_hidden_layers)
    layers = list(m.layers)[:num_layers]

    embed_w = m.embed_tokens.weight.detach().float()
    embed_w_tt = ttnn.from_torch(embed_w.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    final_norm_w = _row(m.norm.weight, device)
    final_norm_eps = float(m.norm.variance_epsilon)

    inv_freq = m.rotary_emb.inv_freq.detach().float()
    attention_scaling = float(getattr(m.rotary_emb, "attention_scaling", 1.0))
    head_dim = layers[0].self_attn.head_dim

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Compose the graduated per-layer child stub for every decoder layer.
    layer_forwards = [_build_qwen2_decoder_layer(device, layer) for layer in layers]

    def _tokens_tt(tokens):
        if isinstance(tokens, ttnn.Tensor):
            return tokens
        prepped = tokens.detach().to(torch.int32).contiguous()
        return ttnn.from_torch(prepped, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def forward(
        input_ids=None,
        *args,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        kv_buffers=None,
        cache_pos=None,
        ext_cos=None,
        ext_sin=None,
        ext_mask=None,
        write_onehot=None,
        **kwargs,
    ):
        # ── TRACEABLE decode fast path ────────────────────────────────────────────
        # When `write_onehot` is given (a single decode token, T=1), EVERY shape-dependent
        # constant is supplied by the caller as a RESIDENT device buffer already staged for this
        # step — the per-token RoPE cos/sin (`ext_cos`/`ext_sin`), the capacity mask (`ext_mask`),
        # the KV write position (`write_onehot`) — so this path issues NO host op (no `from_torch`)
        # and can be captured into a trace. The layer writes K/V in place via the masked-add
        # (`write_onehot`) instead of `update_cache`. Reused across steps by re-staging those buffers.
        if write_onehot is not None:
            h = (
                inputs_embeds
                if inputs_embeds.get_dtype() == ttnn.float32
                else ttnn.typecast(inputs_embeds, ttnn.float32)
            )
            for i, layer_forward in enumerate(layer_forwards):
                h = layer_forward(
                    h,
                    position_embeddings=(ext_cos, ext_sin),
                    attention_mask=ext_mask,
                    kv_buf=kv_buffers[i],
                    write_onehot=write_onehot,
                )
            return _rms_norm(h, final_norm_w, final_norm_eps, compute_config)
        # VibeVoice feeds precomputed inputs_embeds (speech embeds injected + AR feedback);
        # the isolated PCC test feeds input_ids. Support both; both paths are host-free.
        #
        # Optional KV-cache (opt-in, use_cache=True): `past_key_values` is a per-layer list of
        # (k, v) tensors (pre-`repeat_kv`, `[B, num_kv_heads, T_past, head_dim]`). On a decode step
        # the caller feeds only the ONE new token embed (T=1); RoPE is built for its absolute
        # position (past_len) and no causal mask is needed (a single query attends to all cached
        # keys). Prefill (past_key_values=None, use_cache=True) builds the causal mask exactly as the
        # legacy path. With the defaults (use_cache=False) the math and the return type are
        # byte-for-byte identical to the original, so the isolated qwen2_model PCC harness is
        # unaffected.
        if inputs_embeds is not None:
            x = (
                inputs_embeds
                if inputs_embeds.get_dtype() == ttnn.float32
                else ttnn.typecast(inputs_embeds, ttnn.float32)
            )
            B, T = int(x.shape[0]), int(x.shape[1])
        else:
            tok_tt = _tokens_tt(input_ids)
            B, T = int(tok_tt.shape[0]), int(tok_tt.shape[1])
            emb = ttnn.embedding(tok_tt, embed_w_tt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            x = ttnn.typecast(emb, ttnn.float32)

        # starting absolute position of this call's tokens
        if kv_buffers is not None:
            past_len = int(cache_pos) if (cache_pos is not None and T == 1) else 0
        elif past_key_values is not None:
            past_len = int(past_key_values[0][0].shape[2])
        else:
            past_len = 0

        if position_ids is None:
            pos = torch.arange(past_len, past_len + T, dtype=torch.float32).unsqueeze(0).expand(B, T)
        else:
            pos = position_ids.float()
        freqs = torch.einsum("bt,d->btd", pos, inv_freq)
        emb_ang = torch.concat([freqs, freqs], dim=-1)
        cos = (emb_ang.cos() * attention_scaling).reshape(B, 1, T, head_dim).contiguous()
        sin = (emb_ang.sin() * attention_scaling).reshape(B, 1, T, head_dim).contiguous()
        cos_t = ttnn.from_torch(cos, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        sin_t = ttnn.from_torch(sin, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        # ── fixed-capacity KV-cache path (in-place cache, fixed-shape decode) ──────
        if kv_buffers is not None:
            C = int(kv_buffers[0][0].shape[2])
            if T > 1:  # prefill: causal mask over the prompt block; layers fill_cache seq 0
                m = torch.triu(torch.full((T, T), -1.0e9, dtype=torch.float32), diagonal=1)
                mask = m.reshape(1, 1, T, T).expand(B, 1, T, T).contiguous()
            else:  # decode: block the not-yet-written capacity positions (> cache_pos)
                m = torch.zeros(C, dtype=torch.float32)
                m[past_len + 1 :] = -1.0e9
                mask = m.reshape(1, 1, 1, C).expand(B, 1, 1, C).contiguous()
            mask_t = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            h = x
            for i, layer_forward in enumerate(layer_forwards):
                h = layer_forward(
                    h,
                    position_embeddings=(cos_t, sin_t),
                    attention_mask=mask_t,
                    kv_buf=kv_buffers[i],
                    cache_pos=past_len,
                )
            return _rms_norm(h, final_norm_w, final_norm_eps, compute_config)

        # Causal mask only for a multi-token block with no cache (prefill / isolated test). A cached
        # decode step is a single query attending over all cached keys, so it needs no mask.
        causal_t = None
        if past_key_values is None and T > 1:
            causal = torch.triu(torch.full((T, T), -1.0e9, dtype=torch.float32), diagonal=1)
            causal = causal.reshape(1, 1, T, T).expand(B, 1, T, T).contiguous()
            causal_t = ttnn.from_torch(causal, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        new_past = [] if use_cache else None
        h = x
        for i, layer_forward in enumerate(layer_forwards):
            if use_cache:
                pkv = past_key_values[i] if past_key_values is not None else None
                h, kv = layer_forward(
                    h, position_embeddings=(cos_t, sin_t), attention_mask=causal_t, past_kv=pkv, use_cache=True
                )
                new_past.append(kv)
            else:
                h = layer_forward(h, position_embeddings=(cos_t, sin_t), attention_mask=causal_t)

        h = _rms_norm(h, final_norm_w, final_norm_eps, compute_config)
        if use_cache:
            return h, new_past
        return h

    return forward


def qwen2_model(*args, **kwargs):
    raise RuntimeError(
        "qwen2_model requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
