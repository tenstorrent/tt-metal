# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV-cached autoregressive decode for the XTTS-v2 GPT transformer core (Block 3).

Extends TTNNGPTCore (prefill) with a per-layer on-device KV cache and a single-token
decode step. At step t we compute Q/K/V for the one new token only, append its K/V to
the preallocated cache, and attend over positions 0..t via flash-decode SDPA — i.e. the
past K/V are cached, never recomputed through the c_attn projection.

Equivalence check (see tests/test_gpt_decode_pcc.py): because attention is causal, the
latent produced at decode step t must equal position t of the parallel prefill output.
So the decode loop is validated against the SAME golden `latents.pt` as prefill.

Tensor layouts (verified on device):
  - new token k/v : [1, n_head, 1, head_dim]         -> ttnn.update_cache(cache, kv, pos)
  - kv cache      : [1, n_head, max_seq, head_dim]
  - decode query  : [1, batch(=1), n_head, head_dim] -> scaled_dot_product_attention_decode
  - decode output : [1, batch(=1), n_head, head_dim]

Decode uses bf16 (flash-decode SDPA is bf16-only) — the native fast path.
"""

import ttnn

from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import TTNNGPTConfig, TTNNGPTCore


class TTNNGPTDecoder(TTNNGPTCore):
    def __init__(
        self,
        device,
        parameters,
        config: TTNNGPTConfig = None,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        max_seq: int = 128,
    ):
        # bf16 + sdpa: decode SDPA (flash-decode) only accepts bf16 q/k/v.
        super().__init__(
            device,
            parameters,
            config,
            math_fidelity=math_fidelity,
            activation_dtype=ttnn.bfloat16,
            attention="sdpa",
        )
        self.max_seq = max_seq
        self.pos = 0
        self.k_cache = []
        self.v_cache = []
        cfg = self.config
        import torch

        zeros = torch.zeros(1, cfg.n_head, max_seq, cfg.head_dim)
        for _ in range(cfg.n_layer):
            self.k_cache.append(ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
            self.v_cache.append(ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))

    def reset(self):
        """Start a new sequence. Positions are overwritten as we decode, and SDPA-decode
        only reads 0..cur_pos, so resetting the position counter is sufficient."""
        self.pos = 0

    def _attn_decode(self, x, li):
        cfg = self.config
        block = self.params["blocks"][li]
        qkv = self._linear(x, block["c_attn"])  # [1,1,3072]
        q = qkv[:, :, 0 : cfg.n_embd]
        k = qkv[:, :, cfg.n_embd : 2 * cfg.n_embd]
        v = qkv[:, :, 2 * cfg.n_embd : 3 * cfg.n_embd]
        ttnn.deallocate(qkv)

        def kv_heads(t):  # [1,1,E] -> [1, nh, 1, dh]
            t = ttnn.reshape(t, (1, 1, cfg.n_head, cfg.head_dim))
            return ttnn.permute(t, (0, 2, 1, 3))

        kh = kv_heads(k)
        vh = kv_heads(v)
        ttnn.update_cache(self.k_cache[li], kh, self.pos)
        ttnn.update_cache(self.v_cache[li], vh, self.pos)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        q_dec = ttnn.reshape(q, (1, 1, cfg.n_head, cfg.head_dim))  # [1, B=1, nh, dh]
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_dec,
            self.k_cache[li],
            self.v_cache[li],
            cur_pos=[self.pos],
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, B=1, nh, dh]

        attn = ttnn.reshape(attn, (1, 1, cfg.n_embd))  # merge heads
        return self._linear(attn, block["attn_proj"])

    def decode_step(self, x_t):
        """One token: x_t [1,1,1024] -> latent_t [1,1,1024]. Advances the cache position."""
        x = x_t
        for li, block in enumerate(self.params["blocks"]):
            x = ttnn.add(x, self._attn_decode(self._layer_norm(x, block["ln_1"]), li))
            x = ttnn.add(x, self._mlp(self._layer_norm(x, block["ln_2"]), block))
        x = self._layer_norm(x, self.params["ln_f"])
        x = self._layer_norm(x, self.params["final_norm"])
        self.pos += 1
        return x
