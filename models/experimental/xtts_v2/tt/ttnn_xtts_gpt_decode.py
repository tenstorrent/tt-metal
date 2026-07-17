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


class TTNNGPTTracedDecoder(TTNNGPTCore):
    """Trace-captured KV-cached decode: the whole 30-layer decode step is captured once
    into a device trace and replayed per token, eliminating per-token host op-dispatch.

    Position is a device tensor (`pos`) threaded into paged_update_cache(update_idxs_tensor)
    and scaled_dot_product_attention_decode(cur_pos_tensor), so a single captured graph
    works for every step. The token embedding is copied into a stable pre-allocated input
    tensor each step; the latent is read from a stable output tensor.

    Requires the device opened with a trace_region_size (e.g. ttnn.open_device(...,
    trace_region_size=50_000_000)). bf16 (flash-decode + paged cache are bf16-only).
    """

    def __init__(
        self,
        device,
        parameters,
        config: TTNNGPTConfig = None,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        max_seq: int = 128,
    ):
        super().__init__(
            device, parameters, config, math_fidelity=math_fidelity, activation_dtype=ttnn.bfloat16, attention="sdpa"
        )
        import torch

        cfg = self.config
        self.max_seq = max_seq
        zeros = torch.zeros(1, cfg.n_head, max_seq, cfg.head_dim)
        self.k_cache = [
            ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(cfg.n_layer)
        ]
        self.v_cache = [
            ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(cfg.n_layer)
        ]
        # paged_update_cache requires a height-sharded token input (1 core for B=1).
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        self._shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (32, cfg.head_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self._pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=device)
        self._in = ttnn.from_torch(
            torch.zeros(1, 1, cfg.n_embd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.trace_id = None
        self._out = None

    def _step_ops(self, x):
        cfg = self.config
        for li in range(cfg.n_layer):
            b = self.params["blocks"][li]
            qkv = self._linear(self._layer_norm(x, b["ln_1"]), b["c_attn"])
            q = ttnn.reshape(qkv[:, :, 0 : cfg.n_embd], (1, 1, cfg.n_head, cfg.head_dim))
            k = ttnn.reshape(qkv[:, :, cfg.n_embd : 2 * cfg.n_embd], (1, 1, cfg.n_head, cfg.head_dim))
            v = ttnn.reshape(qkv[:, :, 2 * cfg.n_embd : 3 * cfg.n_embd], (1, 1, cfg.n_head, cfg.head_dim))
            ttnn.experimental.paged_update_cache(
                self.k_cache[li],
                ttnn.interleaved_to_sharded(k, self._shard),
                update_idxs_tensor=self._pos,
                page_table=None,
            )
            ttnn.experimental.paged_update_cache(
                self.v_cache[li],
                ttnn.interleaved_to_sharded(v, self._shard),
                update_idxs_tensor=self._pos,
                page_table=None,
            )
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                self.k_cache[li],
                self.v_cache[li],
                cur_pos_tensor=self._pos,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config,
            )
            attn = ttnn.reshape(attn, (1, 1, cfg.n_embd))
            x = ttnn.add(x, self._linear(attn, b["attn_proj"]))
            x = ttnn.add(x, self._mlp(self._layer_norm(x, b["ln_2"]), b))
        x = self._layer_norm(x, self.params["ln_f"])
        return self._layer_norm(x, self.params["final_norm"])

    def reset_caches(self):
        for c in self.k_cache + self.v_cache:
            ttnn.copy(ttnn.zeros_like(c, memory_config=ttnn.DRAM_MEMORY_CONFIG), c)

    def _set_pos(self, p):
        import torch

        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([p], dtype=torch.int32)), self._pos)

    def _set_input(self, emb):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._in)

    def capture(self):
        """Compile (warmup) then capture the decode step into a trace. Resets caches after."""
        self._set_pos(0)
        self._step_ops(self._in)  # warmup compile (trace cannot compile new programs)
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._step_ops(self._in)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.reset_caches()

    def step(self, emb, pos, read=True):
        """emb: torch [1,1,1024]; pos: int. Returns latent torch [1,1,1024] if read."""
        self._set_input(emb)
        self._set_pos(pos)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        if read:
            import torch

            return ttnn.to_torch(self._out).to(torch.float32)
        return None

    def decode_sequence(self, inputs_embeds):
        """Feed a torch [1,S,1024] sequence token-by-token; return latents [1,S,1024]."""
        import torch

        self.reset_caches()
        lat = [
            self.step(inputs_embeds[:, t : t + 1, :].contiguous(), t, read=True) for t in range(inputs_embeds.shape[1])
        ]
        return torch.cat(lat, dim=1)
