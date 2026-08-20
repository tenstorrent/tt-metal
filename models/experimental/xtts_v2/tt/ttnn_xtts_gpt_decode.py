# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
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

import torch
import ttnn

from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import TTNNGPTConfig, TTNNGPTCore


DECODE_IN0_BLOCK_W = 4  # tiles of the K reduction per step; see _decode_matmul_cfg
DECODE_MAX_SUBBLOCK = 4  # out_subblock_h * w ceiling: fp32_dest_acc_en halves the register budget


def _decode_matmul_cfg(device, K, N, fused_activation=None):
    """1D-multicast matmul config for a single-token (M=1) decode linear.

    Spreads N across the largest usable core grid and cuts the K reduction into
    DECODE_IN0_BLOCK_W-tile chunks, so the weight stream pipelines against the math. Returns None
    when the shapes cannot express one, leaving ttnn's own heuristic in place. per_core_M=1 makes
    these DECODE ONLY — prefill shares _linear/_mlp and passes nothing.

    fused_activation folds an elementwise op into the matmul. Passing `activation=` to ttnn.linear
    alongside an explicit program_config does NOT fuse — it runs a second kernel — so the config
    is where it has to go."""
    Kt, Nt = K // 32, N // 32
    g = device.compute_with_storage_grid_size()
    rows = next((r for r in range(g.y, 0, -1) if Nt % (g.x * r) == 0), None)
    if rows is None or Kt % DECODE_IN0_BLOCK_W:
        return None
    per_core_N = Nt // (g.x * rows)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(g.x, rows),
        in0_block_w=DECODE_IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=next(w for w in range(min(per_core_N, DECODE_MAX_SUBBLOCK), 0, -1) if per_core_N % w == 0),
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _prefill_tiles(P, n_head, n_cores):
    """Smallest 32-row tile count covering P that ttnn.fill_cache seeds correctly.

    fill_cache splits n_head*tiles blocks over the cores as consecutive runs, but hands each core
    only its FIRST cache address and then walks forward. A run that crosses a head boundary writes
    the remainder into the previous head instead, leaving those positions zero -- and it does so
    identically on every repeat, so nothing downstream flags it. No run can straddle when every
    core gets at most one block, or when the blocks divide evenly over the cores (equal runs,
    each starting on a head boundary)."""
    t = (P + 31) // 32
    while n_head * t > n_cores and (n_head * t) % n_cores:
        t += 1
    return t


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
        self.ln_sharded = True  # single-token decode: use the width-sharded LayerNorm path
        # BUG-1: scaled_dot_product_attention_decode returns garbage when the KV-cache
        # sequence length is an ODD number of 32-tiles (e.g. 736 = 23 tiles -> PCC 0.63).
        # An even tile count (multiple of 64) is always correct. Round the cache up so the
        # flash-decode kernel never sees an odd tile count, whatever length the caller asks.
        self.max_seq = ((max_seq + 63) // 64) * 64
        self.pos = 0
        self.k_cache = []
        self.v_cache = []
        cfg = self.config
        zeros = torch.zeros(1, cfg.n_head, self.max_seq, cfg.head_dim)
        for _ in range(cfg.n_layer):
            self.k_cache.append(
                ttnn.from_torch(
                    zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=self.mesh_mapper
                )
            )
            self.v_cache.append(
                ttnn.from_torch(
                    zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=self.mesh_mapper
                )
            )

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
        batch: int = 1,
        data_mapper=None,
    ):
        super().__init__(
            device, parameters, config, math_fidelity=math_fidelity, activation_dtype=ttnn.bfloat16, attention="sdpa"
        )
        cfg = self.config
        # Data-parallel serving: `batch` is the number of requests carried in the tensor's
        # leading dim and `data_mapper` is how they are distributed (e.g.
        # ttnn.shard_tensor_to_mesh_mapper(mesh, dim=0) -> one request per chip, per-chip
        # batch 1). Weights/KV-cache keep self.mesh_mapper (replicate). Defaults reproduce the
        # single-request path exactly: batch=1 and data_mapper=self.mesh_mapper.
        self.batch = batch
        self.data_mapper = self.mesh_mapper if data_mapper is None else data_mapper
        self.ln_sharded = True  # single-token decode: use the width-sharded LayerNorm path
        b0 = self.params["blocks"][0]
        gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False)  # False = not the fast approximation
        self._prg = {  # decode-only matmul configs, keyed by weight; c_fc carries its gelu
            n: _decode_matmul_cfg(device, *tuple(b0[n]["weight"].shape)[-2:], gelu if n == "c_fc" else None)
            for n in ("c_attn", "attn_proj", "c_fc", "mlp_proj")
        }
        # BUG-1: sdpa_decode is wrong when the KV-cache length is an odd number of 32-tiles;
        # round up to an even tile count (multiple of 64). See TTNNGPTDecoder for details.
        self.max_seq = ((max_seq + 63) // 64) * 64
        zeros = torch.zeros(1, cfg.n_head, self.max_seq, cfg.head_dim)
        self.k_cache = [
            ttnn.from_torch(
                zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=self.mesh_mapper
            )
            for _ in range(cfg.n_layer)
        ]
        self.v_cache = [
            ttnn.from_torch(
                zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=self.mesh_mapper
            )
            for _ in range(cfg.n_layer)
        ]
        # paged_fused_update_cache does K+V in one kernel but needs them on non-overlapping cores;
        # nlp_create_qkv_heads_decode places both K and V on core (0,0), so move V to core (1,0) first.
        self._v_cfg1 = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                (32, cfg.head_dim),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        # Per-request state: one cache position and one token embedding per request. Sharded by
        # data_mapper, so on a 1xB mesh each chip holds its own [1]-shaped pos and [1,1,E] input.
        self._pos = ttnn.from_torch(
            torch.zeros(self.batch, dtype=torch.int32), device=device, mesh_mapper=self.data_mapper
        )
        self._in = ttnn.from_torch(
            torch.zeros(self.batch, 1, cfg.n_embd),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=self.data_mapper,
        )
        self.trace_id = None
        self._out = None

    def _step_ops(self, x):
        cfg = self.config
        for li in range(cfg.n_layer):
            b = self.params["blocks"][li]
            qkv = self._linear(self._layer_norm(x, b["ln_1"]), b["c_attn"], prg=self._prg["c_attn"])
            # Fused per-head Q/K/V split: outputs are height-sharded in L1 (K/V feed the cache-update
            # directly, Q feeds sdpa_decode) -- replaces 3 slice + 3 reshape + 2 interleaved_to_sharded.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, (1, 1, 1, 3 * cfg.n_embd)), num_heads=cfg.n_head, num_kv_heads=cfg.n_head
            )
            # Fused K+V cache write in one kernel (V moved to core (1,0) so K/V don't overlap).
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache[li],
                k,
                self.v_cache[li],
                ttnn.to_memory_config(v, self._v_cfg1),
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
            x = ttnn.add(x, self._linear(attn, b["attn_proj"], prg=self._prg["attn_proj"]))
            x = ttnn.add(x, self._mlp(self._layer_norm(x, b["ln_2"]), b, self._prg["c_fc"], self._prg["mlp_proj"]))
        x = self._layer_norm(x, self.params["ln_f"])
        return self._layer_norm(x, self.params["final_norm"])

    def prefill(self, prefix_emb):
        """Fill the KV-cache for prompt positions 0..P-1 in ONE batched pass over the P prompt tokens
        (ttnn.fill_cache), instead of P single-token decode steps -- each layer's K/V weights are read
        once, not P times. Latents are discarded (only the caches seed decode). Eager (not traced); run
        after reset_caches() and BEFORE capture() (allocating buffers under a live trace corrupts it).
        prefix_emb: torch [batch, P, 1024] (batch=1 for the single-request path). P is right-padded
        to a tile count fill_cache seeds correctly (_prefill_tiles), which also buckets the prefill
        program variants; callers keep their TRUE P for decode.

        Data-parallel note: when requests have different prompt lengths, right-pad them to a
        common P with zeros. Padded positions do get K/V written, but each request's decode starts
        at its own P_i <= P and sdpa_decode only attends to 0..cur_pos, so the pad K/V is never
        read and is overwritten as that request decodes. The real positions are unaffected because
        prefill attention is causal."""
        cfg = self.config
        E, nh = cfg.n_embd, cfg.n_head
        g = self.device.compute_with_storage_grid_size()
        P = 32 * _prefill_tiles(prefix_emb.shape[1], nh, g.x * g.y)
        x = ttnn.from_torch(
            torch.nn.functional.pad(prefix_emb, (0, 0, 0, P - prefix_emb.shape[1])).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self.data_mapper,
        )
        for li in range(cfg.n_layer):
            b = self.params["blocks"][li]
            # multi-token pass -> interleaved LayerNorm (the width-sharded decode path is single-token only)
            h = b["ln_1"](x, sharded=False, compute_kernel_config=self.compute_kernel_config)
            qkv = self._linear(h, b["c_attn"])  # [1, P, 3*E]
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.reshape(qkv, (1, 1, P, 3 * E)),
                num_heads=nh,
                num_kv_heads=nh,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # each [1, nh, P, dh]
            ttnn.fill_cache(self.k_cache[li], k, 0)  # write positions 0..P-1 in one shot
            ttnn.fill_cache(self.v_cache[li], v, 0)
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale, compute_kernel_config=self.compute_kernel_config
            )
            attn = ttnn.reshape(ttnn.experimental.nlp_concat_heads(attn), (1, P, E))  # fused head-merge
            x = ttnn.add(x, self._linear(attn, b["attn_proj"]))
            x = ttnn.add(x, self._mlp(b["ln_2"](x, sharded=False, compute_kernel_config=self.compute_kernel_config), b))

    def reset_caches(self):
        for c in self.k_cache + self.v_cache:
            ttnn.copy(ttnn.zeros_like(c, memory_config=ttnn.DRAM_MEMORY_CONFIG), c)

    def set_pos(self, pos):
        """Set the KV-cache write/attend position (cache-state control; host -> device _pos).

        `pos` is either one int (applied to every request) or a per-request sequence of length
        `batch` — data-parallel requests run at independent positions because their prompts have
        different lengths and they stop at different steps."""
        if isinstance(pos, int):
            t = torch.full((self.batch,), pos, dtype=torch.int32)
        else:
            t = torch.as_tensor(pos, dtype=torch.int32).reshape(self.batch)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, mesh_mapper=self.data_mapper), self._pos)

    def capture(self):
        """Compile (warmup) then capture the decode step into a trace. Warms at a scratch slot
        (max_seq-1, never a real decode position) and does NOT zero the cache, so capture() can run
        AFTER prefill() without clobbering the prompt's K/V. Allocating buffers under a live trace
        corrupts it, so prefill() must run before capture(). Teacher-forced callers reset themselves."""
        self.set_pos(self.max_seq - 1)
        self._step_ops(self._in)  # warmup compile (trace cannot compile new programs)
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._step_ops(self._in)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def step_device(self, emb_dev, pos):
        """Device-in -> device-out decode primitive: copy the (device) embedding into the stable
        trace input, set the position, replay the trace, and return the (device) latent output.
        No host<->device transfer and no loop -- the driver owns from_torch/to_torch, the mesh
        mapping, and the decode loop (see ttnn_xtts_gpt_generate.traced_decode_sequence)."""
        ttnn.copy(emb_dev, self._in)
        self.set_pos(pos)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        return self._out
