# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving adapter for poolside/Laguna-XS-2.1 on the 1×4 Blackhole mesh.

This is the thin translation layer between the Tenstorrent vLLM plugin and the model-specific
``LagunaGenerator`` / ``LagunaModel`` (``tt/generator.py`` / ``tt/model.py``). It implements exactly
the method surface the plugin calls (``initialize_vllm_model``, ``allocate_kv_cache``,
``prefill_forward``, ``decode_forward``, ``read_decode_output``, ``process_decode_output_host``,
``get_max_tokens_all_users``, ``model_capabilities``) and delegates all real compute to the
generator's low-level pieces (``model.embed_*`` / ``prefill_layers`` / ``decode_layers`` /
``lm_head_shards_*`` and the canonical ``Sampling1D`` split-sampling path). It adds NO new sampling
strategy, NO host argmax on the perf path, NO full-logits readback on the perf path, and NO
Python readback/writeback token-feedback loop: the traced decode replays a single captured graph
that samples on device and feeds ``tt_out_tok`` back into the persistent decode token buffer.

Cache ownership: in vLLM mode the KV cache is **owned by vLLM** — ``allocate_kv_cache`` builds the
paged buffers in the exact layer-dict format ``LagunaModel`` consumes, and every prefill/decode
call receives that cache plus vLLM's per-step page table and positions. The generator's own
standalone cache/reset path (``tt/generator.py``) is untouched and used only by the readiness
checks.

Attention: Laguna is a hybrid model (10 full + 30 sliding layers, ``sliding_window=512``). The
sliding layers trim attention on the READ side via the SDPA op's ``sliding_window_size`` kwarg over
a full paged cache, so — exactly like the TT plugin's currently-gated hybrid path — the adapter is
served as a **uniform full-attention model**: one page table, one ``FullAttentionSpec``-equivalent
cache per layer. No ``get_kv_cache_spec`` is needed (sliding correctness is inside the model's
SDPA, not vLLM page tables). This matches the documented full-context KV budget (all 40 layers hold
the full context), so no advertised capability is reduced.

Precision: construction goes through ``LagunaGenerator.from_pretrained`` →
``LagunaModel.from_pretrained``, which by default loads the datatype-sweep-selected precision
policy (``doc/datatype_sweep/selected_precision_config.json``): BFP8 attn/dense/shared weights,
BFP4 routed experts, BF16 router/norms/activations/CCL, BFP8 KV cache, BFP8 LM head, per-group
compute fidelities, fp32/HiFi4 SDPA. The serving path therefore uses the selected policy verbatim.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

import torch

import ttnn

try:
    from .generator import LagunaGenerator, _replicate
except ImportError:  # loaded as a standalone module by some tooling
    from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator, _replicate

# HF-advertised context (doc/context_contract.json). Served verbatim; no capability reduction.
ADVERTISED_MAX_CONTEXT = 262144


class LagunaForCausalLM:
    """vLLM bridge for TTLagunaForCausalLM.

    Registered as ``TTLagunaForCausalLM`` in the TT vLLM plugin (the plugin prepends ``TT`` to the
    HF architecture ``LagunaForCausalLM``).
    """

    # Capability flags read by the plugin platform hook. On-device sampling is REQUIRED here: the
    # readiness runner enforces ``sample_on_device_mode=all`` and the model serves its canonical
    # traced split-sampling path. Async decode is supported via the read/process split below.
    # Prefix caching is off (sliding-window model; not implemented/tested).
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    def __init__(self, generator: LagunaGenerator, mesh_device, max_batch_size: int, max_model_len: int):
        self.gen = generator
        self.model = generator.model
        self.mesh_device = mesh_device
        self.tokenizer = generator.tokenizer
        self.data_parallel = 1  # single 1×4 mesh; TP=4/EP=4 are intra-mesh, not vLLM DP
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.vocab = generator.vocab
        self.hidden = generator.hidden
        self.D = mesh_device.get_num_devices()
        # Per-batch captured decode trace + persistent device buffers.
        self._decode: dict[int, dict] = {}
        # Persistent prefill buffers (sampling tensors + fixed [1,1,1,H] terminal + B=1 sampler),
        # allocated once BEFORE the decode trace is captured (see warmup_model_prefill).
        self._pf: Optional[dict] = None
        # Persistent prefill page-table buffers keyed by shape (allocate-once, then copy-in), kept
        # SEPARATE from the decode trace's page table so a prefill never overwrites the decode pt.
        self._pf_pt: dict = {}
        # max_num_blocks_per_req, learned from warmup_model_decode; lets prefill warmup pre-allocate
        # the serving-shape page-table buffer before the decode trace is captured.
        self._max_blocks: Optional[int] = None
        self.already_warmed_up_prefill = False
        # vLLM-owned cache dtype (from the selected precision policy), used for allocation.
        self._kv_dtype = self.model.precision_policy.kv_cache

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=ADVERTISED_MAX_CONTEXT,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        """Plugin entry point (loader.py). ``optimizations`` (str|None) is accepted for interface
        parity but the precision policy comes from the datatype-sweep selection by default; a
        non-default policy is only used via ``TT_LAGUNA_PRECISION_CONFIG``. ``n_layers`` builds a
        reduced representative target for the minimum-surface bring-up loop."""
        assert tt_data_parallel == 1, (
            f"Laguna-XS-2.1 uses a single 1×4 mesh (intra-mesh TP=4/EP=4); tt_data_parallel must be 1, "
            f"got {tt_data_parallel}"
        )
        # Minimum-surface bring-up: TT_LAGUNA_VLLM_NUM_LAYERS builds a reduced representative target
        # (e.g. "0,1,4" = dense-full + sliding-MoE + full-MoE). vLLM still sees the full 40-layer HF
        # config (so it allocates 40 KV specs; the model's decode/prefill zip truncates to the built
        # layers). Reduced is an inner-loop debugging tool only — final evidence uses the full stack.
        if n_layers is None:
            import os as _os

            env_nl = _os.environ.get("TT_LAGUNA_VLLM_NUM_LAYERS")
            if env_nl:
                n_layers = [int(x) for x in env_nl.split(",")] if "," in env_nl else int(env_nl)
        gen = LagunaGenerator.from_pretrained(
            mesh_device,
            max_seq_len=int(max_seq_len),
            num_layers=n_layers,
            hf_config=hf_config,
        )
        return cls(gen, mesh_device, int(max_batch_size), int(max_seq_len))

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Emit a FullAttentionSpec for EVERY layer WITHOUT a sliding-window field.

        Laguna's HF config sets ``sliding_window=512``; the plugin's default uniform spec
        (``worker._build_default_kv_cache_spec``) would bake that into the spec, causing vLLM to cap
        each request's page table to ``sliding_window/block_size`` blocks and collapse positions
        beyond 512 onto physical block 0 — silently corrupting the cache and defeating the advertised
        262144 context. Sliding-window correctness for the 30 sliding layers is handled INSIDE the
        model on the SDPA read side (``paged_scaled_dot_product_attention_decode(sliding_window_size=
        512)``) over a full paged cache, so every layer needs a full-length cache here. All specs are
        identical, so vLLM unifies them into a single KV group → single page table (no per-layer
        routing), exactly the legacy uniform path this adapter's prefill/decode expect."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        layer_types = getattr(text_config, "layer_types", None)
        if num_layers is None and layer_types is not None:
            num_layers = len(layer_types)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        common = dict(
            block_size=cache_config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )  # NOTE: no sliding_window — full cache for all layers.
        return {f"model.layers.{i}.self_attn": FullAttentionSpec(**common) for i in range(int(num_layers))}

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1, **kwargs):
        """Total KV-cache token pool. Return the full advertised context so a single request can use
        the whole 262144-token window (per-device KV ≈5.4 GB at full context, which fits in 34 GB
        alongside ~5.1 GB weights + trace; see doc/context_contract.json). Bounded by the requested
        ``max_model_len`` when smaller (e.g. the reduced bring-up target)."""
        max_model_len = kwargs.get("max_model_len")
        if max_model_len:
            return min(int(max_model_len), ADVERTISED_MAX_CONTEXT)
        return ADVERTISED_MAX_CONTEXT

    @property
    def cache_path(self):
        # Not used by this adapter's own allocator (weights are cached inside LagunaModel), but the
        # plugin may query it; return a harmless path.
        return Path("/tmp")

    # --------------------------------------------------------------------- #
    # KV cache (vLLM-owned)
    # --------------------------------------------------------------------- #
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Build the vLLM-owned paged KV cache. ``kv_cache_shape`` =
        ``(num_blocks, num_kv_heads_local, block_size, head_dim)`` — already folded to the per-device
        local KV heads (2 = 8/TP4) by the plugin. Each layer gets its own ``[k, v]`` buffer,
        replicated across the mesh (each device stores its own local-head slice; identical shape).
        Returns the list of per-layer dicts that ``LagunaModel.prefill_layers`` / ``decode_layers``
        consume. KV dtype is the selected-policy BFP8, independent of vLLM's torch ``dtype`` hint."""
        num_blocks, local_kv_heads, block_size, head_dim = kv_cache_shape
        kv_cache = []
        for _ in range(num_layers):
            k = ttnn.from_torch(
                torch.zeros(kv_cache_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            v = ttnn.from_torch(
                torch.zeros(kv_cache_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            kv_cache.append(
                {
                    "k": k,
                    "v": v,
                    "block_size": int(block_size),
                    "blocks_per_user": int(num_blocks),
                    "dtype": self._kv_dtype,
                }
            )
        # A cache (re)allocation invalidates captured decode traces (they close over kv buffers).
        self._decode = {}
        return kv_cache

    # --------------------------------------------------------------------- #
    # Page-table / sampling helpers
    # --------------------------------------------------------------------- #
    def _page_table_to_device(self, page_table_torch):
        pt = torch.as_tensor(page_table_torch, dtype=torch.int32)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=_replicate(self.mesh_device),
        )

    def _prefill_pt(self, page_table):
        """Persistent, shape-keyed prefill page-table buffer: allocate once per shape (in warmup,
        before the decode trace exists) then only copy the contents in. This removes the per-prefill
        device allocation that would otherwise happen under the resident decode trace at serving time
        (allocator.cpp 'unsafe ... active trace'). Kept separate from the decode trace's own page
        table so a prefill can never overwrite it."""
        pt = torch.as_tensor(page_table, dtype=torch.int32)
        if pt.dim() == 1:
            pt = pt.reshape(1, -1)
        key = tuple(pt.shape)
        buf = self._pf_pt.get(key)
        if buf is None:
            buf = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
            self._pf_pt[key] = buf
        ttnn.copy_host_to_device_tensor(self.gen._host(pt, ttnn.int32), buf)
        return buf

    @staticmethod
    def _sampling_row_params(sp, row):
        """Map one row of a vLLM ``TTSamplingParams`` to (k, p, temp, seed). temperature==0 → greedy
        top-1. top_k<=0 (disabled) → the device candidate-set width (32).

        No explicit seed (``sp.seed[row] is None``) means "sample randomly" — so a FRESH random seed is
        drawn per call (via ``secrets``, independent of the torch/global RNG which vLLM pins to seed 0).
        Defaulting to a fixed 0 instead makes identical no-seed requests deterministic and collapses
        temperature/top-k variety (the plugin's no-seed / temperature-varied / top-k variety tests)."""
        temp = float(sp.temperature[row]) if sp.temperature is not None else 1.0
        top_k = int(sp.top_k[row]) if sp.top_k is not None else 0
        top_p = float(sp.top_p[row]) if sp.top_p is not None else 1.0
        seed = sp.seed[row] if sp.seed is not None else None
        if temp <= 0.0:  # greedy — seed irrelevant (top-k(k=1) is deterministic)
            return 1, 1.0, 1.0, 0
        k = top_k if 0 < top_k <= 32 else 32
        p = top_p if 0.0 < top_p <= 1.0 else 1.0
        s = int(seed) if seed is not None else secrets.randbelow(2_000_000_000)
        return k, p, temp, s

    def _sampling_buffers_from_params(self, sp, B):
        """Build host [B] arrays of k/p/temp/seed from a vLLM TTSamplingParams (lists), padding to B
        with greedy defaults for inactive rows."""
        k = torch.ones(B, dtype=torch.int32)
        p = torch.ones(B, dtype=torch.float32)
        t = torch.ones(B, dtype=torch.float32)
        s = torch.zeros(B, dtype=torch.int32)
        n = 0 if sp is None or sp.temperature is None else len(sp.temperature)
        for row in range(min(n, B)):
            kk, pp, tt_, ss = self._sampling_row_params(sp, row)
            k[row], p[row], t[row], s[row] = kk, pp, tt_, ss
        return k, p, t, s

    # --------------------------------------------------------------------- #
    # Prefill — trace-safe (bucketed length + fixed-shape terminal)
    # --------------------------------------------------------------------- #
    # Under vLLM continuous batching a NEW-request prefill is interleaved between decode-trace
    # replays, i.e. it runs while the decode trace is RESIDENT. ttnn forbids device-buffer allocation
    # while a trace is resident ("Allocating device buffers is unsafe due to the existence of an
    # active trace", allocator.cpp) — any such allocation can corrupt the captured trace (garbage
    # tokens, then a device wedge). So prefill must run ONLY already-compiled programs over
    # already-allocated buffers. Two things make that true:
    #   (1) The prompt is right-padded to a BUCKET length so `prefill_layers` sees a fixed shape per
    #       bucket (a bounded set of programs, all pre-compiled by warmup_model_prefill BEFORE the
    #       decode trace is captured). Right-padding is safe: causal attention means the last REAL
    #       token (plen-1) never attends to the pad positions, so its logits are exact; the padded
    #       cache slots (plen..L-1) are future positions, overwritten before any decode step reads
    #       them.
    #   (2) The last-real-token hidden is selected with a FIXED-shape one-hot matmul (the row index
    #       plen-1 is DATA copied into a persistent selector, not part of the program hash) instead of
    #       the old `ttnn.slice(h, [0, plen-1, 0], ...)` whose offset baked plen into a new program per
    #       distinct length. Sampling reuses persistent B=1 buffers (copy-in, no per-call allocation).

    def _prefill_bucket_lens(self):
        """Supported prefill bucket lengths (powers of two from 128 up to a warm cap, capped by
        max_model_len). Every request rounds UP to one of these; warmup compiles them all. The cap
        (``TT_LAGUNA_PREFILL_WARM_CAP``, default 8192 = the decoder single-shot boundary PIPE_CHUNK)
        bounds warmup cost; prompts beyond it round up to a multiple of the top bucket (pipelined
        path, whose fixed chunk program the top bucket also warms)."""
        import os as _os

        cap = min(int(self.max_model_len), int(_os.environ.get("TT_LAGUNA_PREFILL_WARM_CAP", "8192")))
        buckets, b = [], 128
        while b < cap:
            buckets.append(b)
            b *= 2
        buckets.append(cap)
        return sorted(set(x for x in buckets if x >= 1))

    def _bucket_len(self, plen):
        buckets = self._prefill_bucket_lens()
        for b in buckets:
            if plen <= b:
                return b
        top = buckets[-1]
        return ((plen + top - 1) // top) * top  # multiple of the top bucket (warmed pipeline chunk)

    def _prefill_state(self):
        """Allocate (once) the persistent prefill sampling buffers + B=1 sampler. Called from
        warmup_model_prefill BEFORE any decode trace is captured, so these allocations are safe."""
        if self._pf is not None:
            return self._pf
        z = torch.zeros([1], dtype=torch.int32)
        st = dict(
            tok=self.gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32),
            k=self.gen._rep(torch.ones([1], dtype=torch.int32), ttnn.uint32),
            p=self.gen._rep(torch.ones([1], dtype=torch.float32), ttnn.bfloat16),
            t=self.gen._rep(torch.ones([1], dtype=torch.float32), ttnn.bfloat16),
            seeds=self.gen._rep(z, ttnn.uint32),
            sampler=self.gen._sampler(1),
            # Persistent [1,1,1,H] buffer holding the selected last-real-token hidden. Fixed shape →
            # the terminal norm+LM-head+sample program is compiled ONCE (warmup) and reused, never
            # recompiled per prompt length under the resident decode trace.
            last_h=self.gen._rep(
                torch.zeros([1, 1, 1, self.hidden], dtype=torch.float32), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
        )
        self._pf = st
        return st

    def _last_token_shards(self, h, plen, L):
        """Select the last REAL token's logit shards in a way that is fixed-shape at the terminal.

        ``h`` is the bucketed prefill output ``[1, L, H]`` (L fixed per bucket, so its readback program
        is compiled once per bucket by warmup). Read back the replicated hidden, take row ``plen-1`` on
        host (free), copy it into the persistent ``[1,1,1,H]`` buffer, and run the column-sharded LM
        head over that fixed shape. The row index is data, not program shape, so no new program
        compiles per prompt length at serving time. The readback is prefill-only (not the decode perf
        path); decode never leaves the device."""
        st = self._prefill_state()
        hh = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).reshape(
            -1, L, self.hidden
        )
        hrow = hh[0, plen - 1].to(torch.float32).reshape(1, 1, 1, self.hidden)
        hsrc = ttnn.from_torch(
            hrow, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(hsrc, st["last_h"])
        return self.model.lm_head_shards_decode(st["last_h"])

    def _refresh_prefill_sampling(self, st, sp, u):
        """Copy per-request k/p/temp/seed into the persistent B=1 sampling buffers (no allocation)."""
        k, p, t, s = self._sampling_row_params(sp, u)
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([k], dtype=torch.int32), ttnn.uint32), st["k"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([p], dtype=torch.float32), ttnn.bfloat16), st["p"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([t], dtype=torch.float32), ttnn.bfloat16), st["t"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([s], dtype=torch.int32), ttnn.uint32), st["seeds"])

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        enable_trace=False,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        """One prefill step. ``tokens`` [num_reqs, padded_seq] int32, ``page_table`` [num_reqs, nb].
        Host-sampling (``sampling_params is None``): returns logits ``[num_reqs, 1, vocab]``.
        Device-sampling: samples the last position on device and returns ``(tokens[num_reqs,1], None)``.
        The logical prompt length ``prompt_lens[u]`` may be any value ≤ context (not block/tile/chunk
        aligned); internally the prompt is right-padded to a bucket length so the whole prefill runs
        pre-compiled, trace-safe programs (see the block comment above)."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch = tokens.shape[0]
        pt = self._prefill_pt(page_table)  # persistent, shape-keyed (no per-call alloc under trace)
        if prompt_lens is None:
            prompt_lens = [int(tokens.shape[1])] * batch
        starts = [0] * batch if start_pos is None else [int(x) for x in start_pos]

        device_sampling = sampling_params is not None
        st = self._prefill_state() if device_sampling else None
        last_logits = []
        sampled = []
        for u in range(batch):
            plen = int(prompt_lens[u])
            L = self._bucket_len(plen)
            padded = torch.zeros(L, dtype=torch.int64)
            padded[:plen] = tokens[u, :plen]
            tok_tt = self.gen._tokens_to_device(padded)
            x = self.model.embed_prefill(tok_tt)
            h = self.model.prefill_layers(x, kv_cache, pt, user_id=u, start_pos=starts[u])
            shards = self._last_token_shards(h, plen, L)  # fixed-shape terminal
            if device_sampling:
                self._refresh_prefill_sampling(st, sampling_params, u)
                st["sampler"].decode_forward(
                    shards, k=st["k"], p=st["p"], temp=st["t"], seeds=st["seeds"], tt_out_tok=st["tok"]
                )
                sampled.append(self.gen._read_token(st["tok"], 1)[0])
            else:
                logits = self.model.logits_to_host(shards).reshape(1, self.vocab)
                last_logits.append(logits)
        if device_sampling:
            toks = torch.tensor(sampled, dtype=torch.int64).reshape(batch, 1)
            return toks, None
        return torch.stack(last_logits, dim=0)  # [num_reqs, 1, vocab]

    # --------------------------------------------------------------------- #
    # Decode (traced split sampling + async split)
    # --------------------------------------------------------------------- #
    def _decode_state(self, B, kv_cache, pt_persist):
        """Capture (once per batch B) the decode trace over persistent device buffers:
        embed(tok) → 40-layer stack → norm → LM head → Sampling1D(k/p/temp/seed) → tt_out_tok, then
        plus_one(cur/ridx) on device. Nothing is rebuilt on host between replays except the page
        table (only when it changes) and positions/token (only on a batch-layout reset)."""
        st = self._decode.get(B)
        if st is not None:
            return st
        tok = self.gen._rep(torch.zeros([1, 1, 1, B], dtype=torch.int32), ttnn.uint32)
        cur = self.gen._rep(torch.zeros([B], dtype=torch.int32), ttnn.int32)
        ridx = self.gen._rep(torch.zeros([1, B], dtype=torch.int32), ttnn.uint32)
        k = self.gen._rep(torch.ones([B], dtype=torch.int32), ttnn.uint32)
        p = self.gen._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        t = self.gen._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        seeds = self.gen._rep(torch.zeros([B], dtype=torch.int32), ttnn.uint32)
        sampler = self.gen._sampler(B)

        def step():
            h = self.model.embed_decode(ttnn.reshape(tok, (1, B)))
            h = self.model.decode_layers(h, cur, ridx, pt_persist, kv_cache)
            shards = self.model.lm_head_shards_decode(h)
            sampler.decode_forward(shards, k=k, p=p, temp=t, seeds=seeds, tt_out_tok=tok)
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        step()  # compile
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        step()  # capture
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        st = dict(
            tid=tid,
            tok=tok,
            cur=cur,
            ridx=ridx,
            k=k,
            p=p,
            t=t,
            seeds=seeds,
            pt=pt_persist,
            staged=False,
            last_pt_host=None,
            last_sp_key=None,
        )
        self._decode[B] = st
        return st

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        **kwargs,
    ):
        """One decode step for the whole padded batch.

        Device sampling (``sampling_params`` given): traced split-sampling. Token/position refresh is
        done from host ONLY on a batch-layout change (``reset_batch``) or the first step after a
        (re)capture; otherwise the previous step's on-device sampled token (in ``tok``) and the
        device-advanced ``cur``/``ridx`` are reused (no host token/position work). The page table is
        copied only when its contents changed. Returns a per-DP list of device token tensors when
        ``read_from_device=False``, else host tokens.

        Host sampling (``sampling_params is None``, compat mode for min_p/logprobs/etc.): eager decode
        returning logits; never used for the measured perf path."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1, 1)
        B = tokens.shape[0]
        pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(B)

        if sampling_params is None:
            return self._decode_host_sampling(tokens, pos, page_table, kv_cache, read_from_device)

        st = self._decode.get(B)
        if st is None:
            pt_persist = self._page_table_to_device(page_table)
            st = self._decode_state(B, kv_cache, pt_persist)
        tok, cur, ridx, tid, pt = st["tok"], st["cur"], st["ridx"], st["tid"], st["pt"]

        # --- sampling params: refresh persistent buffers only when they change ---
        k_h, p_h, t_h, s_h = self._sampling_buffers_from_params(sampling_params, B)
        sp_key = (tuple(k_h.tolist()), tuple(p_h.tolist()), tuple(t_h.tolist()), tuple(s_h.tolist()))
        if sp_key != st["last_sp_key"]:
            ttnn.copy_host_to_device_tensor(self.gen._host(k_h, ttnn.uint32), st["k"])
            ttnn.copy_host_to_device_tensor(self.gen._host(p_h.to(torch.float32), ttnn.bfloat16), st["p"])
            ttnn.copy_host_to_device_tensor(self.gen._host(t_h.to(torch.float32), ttnn.bfloat16), st["t"])
            ttnn.copy_host_to_device_tensor(self.gen._host(s_h, ttnn.uint32), st["seeds"])
            st["last_sp_key"] = sp_key

        # --- token/position refresh: only on reset or first step (else device feedback) ---
        if reset_batch or not st["staged"]:
            ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens, B), tok)
            ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), cur)
            ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), ridx)
            st["staged"] = True
            self.gen.counters["token_refresh"] += 1
            self.gen.counters["pos_refresh"] += 1

        # --- page table: copy only when contents changed ---
        pt_host = torch.as_tensor(page_table, dtype=torch.int32)
        if st["last_pt_host"] is None or not torch.equal(pt_host, st["last_pt_host"]):
            ttnn.copy_host_to_device_tensor(self._page_table_to_device_host(pt_host), pt)
            st["last_pt_host"] = pt_host.clone()
            self.gen.counters["page_table_refresh"] += 1

        ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=read_from_device)
        self.gen.counters["trace_replay"] += 1

        if read_from_device:
            host = self._read_tokens_host(tok, B)
            return host
        return [tok]  # device token buffer, per-DP list; read via read_decode_output/process...

    # ---- async split ---- #
    def read_decode_output(self, tt_out, async_read=False):
        """Non-blocking readback of the on-device sampled tokens. ``tt_out`` is the per-DP list
        returned by ``decode_forward(read_from_device=False)``."""
        if not async_read:
            return [t.cpu() for t in tt_out]
        host_outputs = [t.cpu(blocking=False) for t in tt_out]
        read_events = [ttnn.record_event(self.mesh_device, 0) for _ in tt_out]
        return host_outputs, read_events

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """Convert the (host) ttnn tensors to torch. ``is_tokens`` True → sampled token ids [B];
        False → logits [B, vocab]. DP=1, so the single entry is returned directly."""
        out = tt_out[0] if isinstance(tt_out, list) else tt_out
        if isinstance(out, tuple):  # (tokens/logits, logprobs)
            out = out[0]
        if is_tokens:
            th = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            B = th.shape[-1] if th.dim() >= 1 else 1
            return th.reshape(-1)[:B].to(torch.int32)
        # logits: gather vocab shards → [B, 1, vocab] (rank-3: the plugin's host-sampling path indexes
        # `tt_out[rows, -1, :]`, so decode logits must carry the seq axis, exactly like the prefill
        # host path's [num_reqs, 1, vocab]). Returning a rank-2 [B, vocab] triggers
        # `IndexError: too many indices for tensor of dimension 2` in model_runner._get_output_tokens.
        th = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return th.reshape(-1, 1, self.vocab)

    # ---- host-sampling (compat) decode ---- #
    def _decode_host_sampling(self, tokens, pos, page_table, kv_cache, read_from_device):
        B = tokens.shape[0]
        pt = self._page_table_to_device(page_table)
        tok_tt = self.gen._rep(tokens.reshape(1, B).to(torch.int32), ttnn.uint32)
        cur = self.gen._rep(pos, ttnn.int32)
        ridx = self.gen._rep(pos.reshape(1, B), ttnn.uint32)
        h = self.model.embed_decode(tok_tt)
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache)
        shards = self.model.lm_head_shards_decode(h)
        if read_from_device:
            logits = self.model.logits_to_host(shards).reshape(B, self.vocab)
            return logits
        return [shards]

    def _read_tokens_host(self, tok_buf, B):
        th = ttnn.to_torch(tok_buf, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return th.reshape(-1)[:B].to(torch.int32)

    # ---- host-tensor builders for persistent-buffer refresh ---- #
    def _host_rank4_tok_batch(self, tokens, B):
        return self.gen._host(tokens.reshape(1, 1, 1, B).to(torch.int32), ttnn.uint32)

    def _host_pos_batch(self, pos):
        return self.gen._host(pos.reshape(-1).to(torch.int32), ttnn.int32)

    def _host_ridx_batch(self, pos):
        return self.gen._host(pos.reshape(1, -1).to(torch.int32), ttnn.uint32)

    def _page_table_to_device_host(self, pt_host):
        return self.gen._host(pt_host.to(torch.int32), ttnn.int32)

    # --------------------------------------------------------------------- #
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every supported prefill bucket length BEFORE the decode trace is captured.

        The plugin's two-phase warmup calls this (Phase 2, ``enable_trace=True``) immediately before
        ``warmup_model_decode`` captures the decode trace, so all prefill programs + persistent
        buffers allocated here are safe. This is REQUIRED (not a no-op): prefill has no trace of its
        own, but at serving time a new-request prefill runs while the decode trace is resident, and
        any first-time program compilation / buffer allocation then corrupts the trace. Warming every
        bucket (and the persistent sampling/selector buffers) makes serving-time prefill
        allocation-free. ``already_warmed_up_prefill`` is reset by the plugin between phases, so this
        runs once per phase; the persistent buffers themselves are allocated once (idempotent)."""
        if kv_cache is None or self.already_warmed_up_prefill:
            return None
        self.already_warmed_up_prefill = True
        self._prefill_state()  # allocate persistent sampling/selector buffers (pre-trace)
        bs = int(kv_cache[0]["block_size"])
        total_blocks = int(kv_cache[0]["blocks_per_user"])
        greedy = None
        if can_sample_on_device:
            from types import SimpleNamespace

            greedy = SimpleNamespace(temperature=[0.0], top_k=[0], top_p=[1.0], seed=[None])
        # Warm at the SERVING page-table width (max_num_blocks_per_req), NOT a bucket-tight arange.
        # Serving always passes a full-width page table (see comment below); the chunked prefill path
        # (seq > PIPE_CHUNK) slices it per chunk (ttnn.slice / chunked SDPA are keyed on that width).
        # Warming those programs at a narrow width leaves them to RECOMPILE under the resident decode
        # trace on the first real >PIPE_CHUNK prefill — a ~200x slowdown (measured: chunked 4096 is
        # 3.2s standalone but 11+ min at serving until the wide-page-table programs recompile). Warming
        # at the serving width makes serving-time prefill compile-free. Single-shot buckets use the
        # table whole (width-agnostic) so this is a no-op for them.
        serve_w = int(self._max_blocks) if self._max_blocks else 0
        for L in self._prefill_bucket_lens():
            nb = (L + bs - 1) // bs
            if nb > total_blocks:  # cache too small for this bucket (reduced bring-up); skip
                continue
            w = serve_w if serve_w >= nb else nb
            pt = torch.zeros((1, w), dtype=torch.int32)
            pt[0, :nb] = torch.arange(nb, dtype=torch.int32)
            dummy = torch.zeros((1, L), dtype=torch.int64)
            self.prefill_forward(
                dummy, page_table=pt, kv_cache=kv_cache, prompt_lens=[L], start_pos=[0], sampling_params=greedy
            )
        return None

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace=False,
        max_batch_size=None,
        num_blocks=None,
        can_sample_on_device=False,
        **kwargs,
    ):
        """Decode warmup. Phase 2 (``enable_trace=True``) pre-captures the single decode trace for the
        full padded batch (``max_batch_size``) over the vLLM-owned cache, so the first real decode
        replays a ready trace instead of compiling+capturing under a live request. ``_decode_state``
        compiles then captures internally, so Phase 1 (``enable_trace=False``) is a no-op. A dummy
        all-zeros page table is used for capture (writes land in block 0 at position 0 and are
        overwritten by the first real prefill); every real decode refreshes the persistent page
        table / positions from the scheduler before replay."""
        # Remember the per-request block width in BOTH phases so prefill warmup (which the plugin runs
        # just before the decode trace is captured) can pre-allocate the serving-shape page table.
        if num_blocks:
            self._max_blocks = int(num_blocks)
        if not enable_trace or kv_cache is None or max_batch_size is None:
            return None
        B = int(max_batch_size)
        if B in self._decode:
            return None
        nb = int(num_blocks) if num_blocks else 1
        pt_persist = self.gen._rep(torch.zeros([B, nb], dtype=torch.int32), ttnn.int32)
        self._decode_state(B, kv_cache, pt_persist)
        return None
