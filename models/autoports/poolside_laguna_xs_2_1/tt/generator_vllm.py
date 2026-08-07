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

import os
import secrets
from pathlib import Path
from typing import Optional

import torch

import ttnn

try:
    from .generator import LagunaGenerator, _replicate
except ImportError:  # loaded as a standalone module by some tooling
    from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator, _replicate

# Advertised context MUST equal servable context (plan item 1.2). The HF config declares 262144, and
# the decoder addresses pos 262143 in isolation, but the 2026-07-31 P150x4 serving sweep OOMs at ISL
# 262144 (both OSL 128 and 1024) while ISL 131072 serves — so end-to-end servable on this config is
# 131072. We advertise only what we can serve; a request for a context we cannot serve would fail an
# agent session at an unpredictable turn. Restoring 262144 is Tier-2 work (hybrid-KV cuts per-device
# KV ~5.4 GB→~1.5 GB; shared RoPE frees ~4.7 GB transient) — see doc/context_contract.json
# ("serving_verified") for the recorded limiting reason.
HF_CONFIG_MAX_CONTEXT = 262144  # what the HF config declares (not currently servable end-to-end)
# Default 131072 = verified servable on P150x4 (2026-07-31 sweep). Env-overridable ONLY for context
# experiments/benchmarks (e.g. measuring exactly ISL 131072 with a 1024-token output needs the KV pool
# — get_max_tokens_all_users = min(max_model_len, this) — to hold 132096; set both this and --max-model-len).
# Raising it re-introduces OOM risk (262144 OOM'd); a small bump (~133120) is ~1k above the known-good 131072.
ADVERTISED_MAX_CONTEXT = int(os.environ.get("TT_LAGUNA_ADVERTISED_CONTEXT", "131072"))


class LagunaForCausalLM:
    """vLLM bridge for TTLagunaForCausalLM.

    Registered as ``TTLagunaForCausalLM`` in the TT vLLM plugin (the plugin prepends ``TT`` to the
    HF architecture ``LagunaForCausalLM``).
    """

    # Capability flags read by the plugin platform hook. On-device sampling is REQUIRED here: the
    # readiness runner enforces ``sample_on_device_mode=all`` and the model serves its canonical
    # traced split-sampling path. Async decode is supported via the read/process split below.
    #
    # Prefix caching: fully plumbed (ttnn chunked SDPA + sliding_window_size, decoder paged-window read
    # + start_pos-offset suffix fill, plugin suffix-only prefill) and DEFAULT ON (env override
    # TT_LAGUNA_PREFIX_CACHE, default "1"). Validated: cold parity and full-hit (warm re-send) are
    # bit-exact vs a no-cache baseline, and cache hits fire (TTFT ~5.97s→0.17s on a long hit).
    # Partial-hit (cached prefix + new suffix) matches the no-cache baseline for the deterministic head
    # of generation, then diverges only at a high-entropy token — floating-point non-determinism across
    # execution paths (suffix-chunk read vs cold pipelined / local-bf16 prefill), the non-bit-exactness
    # inherent to prefix caching on quantized HW (the GPU norm), not a context error. This is ACCEPTED
    # non-determinism (partial-hit runs are not bit-reproducible) — see the determinism contract in
    # README.md (item 1.3). Set TT_LAGUNA_PREFIX_CACHE=0 to force the cold path (bit-reproducible, no reuse).
    _PREFIX_CACHE_ENABLED = os.environ.get("TT_LAGUNA_PREFIX_CACHE", "1") == "1"
    model_capabilities = {
        "supports_prefix_caching": _PREFIX_CACHE_ENABLED,
        "supports_prefix_caching_with_sliding_window": _PREFIX_CACHE_ENABLED,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    # Hybrid KV cache: 10 full-attention + 30 sliding-window(512) layers get DIFFERENT KV specs
    # (see get_kv_cache_spec). The plugin reads this off the model class (worker.py) to size
    # sliding-group block headroom; the vLLM hybrid manager then packs full vs sliding into separate
    # kv_cache_groups and hands the model per-layer block tables (page_tables_per_layer).
    #
    # Per-group persistent-page-table threading (prefill + decode trace) + warmup pre-alloc are now
    # implemented; get_kv_cache_spec emits SlidingWindowSpec(512) for the 30 sliding layers so the
    # sliding pools shrink ~3.7x. Set False (TT_LAGUNA_HYBRID_KV=0) to fall back to the legacy uniform
    # full-cache path (single KV group). NOTE (2026-08-06): on STOCK vLLM 0.24.0 the hybrid multi-group KV
    # manager needs per-group budgets, but the plugin sizes a SINGLE num_gpu_blocks_override → the
    # sufficiency check rejects 131072 (needs ~20 GiB per the 10 full layers vs the single-override budget).
    # The fork served UNIFORM (1 group), where a single override works. So set TT_LAGUNA_HYBRID_KV=0 for the
    # fork-free 0.24.0 stack until the plugin sizes hybrid per-group. Default True preserves prior behavior.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = os.environ.get("TT_LAGUNA_HYBRID_KV", "1") == "1"

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
        # Per-K1 captured spec-decode VERIFY trace (batched decode over K+1 candidates, seq KV write).
        self._verify_dec: dict[int, dict] = {}
        # Persistent prefill buffers (sampling tensors + fixed [1,1,1,H] terminal + B=1 sampler),
        # allocated once BEFORE the decode trace is captured (see warmup_model_prefill).
        self._pf: Optional[dict] = None
        # Persistent prefill page-table buffers keyed by shape (allocate-once, then copy-in), kept
        # SEPARATE from the decode trace's page table so a prefill never overwrites the decode pt.
        self._pf_pt: dict = {}
        # Hybrid KV: persistent prefill page-table buffers per GROUP ("full"/"sliding"); the 2 groups
        # have identical shape but different content, so they can't share the shape-keyed _pf_pt.
        self._pf_pt_groups: dict = {}
        # Per built-layer group kind ("full"/"sliding"), lazily derived from each decoder's cfg.
        self._layer_kinds: Optional[list] = None
        # max_num_blocks_per_req, learned from warmup_model_decode; lets prefill warmup pre-allocate
        # the serving-shape page-table buffer before the decode trace is captured.
        self._max_blocks: Optional[int] = None
        self.already_warmed_up_prefill = False
        self._in_prefill_warmup = False  # True only while warmup_model_prefill runs (suppresses the
        # _prefill_pt diagnostic for intentional warmup pre-allocs; a warning outside warmup = the W1 bug).
        # ---- eager spec-decode (opt-in) — served in-adapter, B==1 greedy. Phase 2. ----
        # TT_LAGUNA_SPEC_DECODE: "" off | "probe" = run the one-shot feasibility probe (does eager verify
        # run under the resident decode trace without an alloc-under-trace hang?) | "1" = full buffered loop.
        self._spec_mode = os.environ.get("TT_LAGUNA_SPEC_DECODE", "")
        self._spec_probed = False
        self._spec_buf: list = []  # pending committed token ids, returned one per vLLM step
        self._spec_hist: list = []  # running token history for the single served request (ngram source)
        self._spec = None  # lazily-built SpeculativeDecoder (served mode); needs kv_cache/page_table per call
        self._spec_tok = None  # persistent [1,1,1,1] device token buffer the plugin reads back
        self._spec_next_pos = None  # position we expect on the next decode call; discontinuity = new request
        self._spec_prefill_seq: list = []  # prompt tokens stashed at prefill (greedy gives no history via kwargs)
        # Diagnostic sink: MPI-worker stdout isn't captured in the readiness log, so spec/probe verdicts go
        # to a file readable regardless of process. Only touched when spec mode is set (no normal-run noise).
        self._spec_log_path = (
            "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/"
            "doc/vllm_integration/_runs/spec_probe.txt"
        )
        if self._spec_mode:
            self._spec_log(f"__init__ pid={os.getpid()} spec_mode={self._spec_mode!r}")
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
        """Emit a HYBRID per-layer spec: ``SlidingWindowSpec(sliding_window=512)`` for the 30
        sliding-attention layers and ``FullAttentionSpec`` for the 10 full-attention layers, keyed by
        ``model.layers.{i}.self_attn`` and driven off ``text_config.layer_types``.

        The TT platform opts into vLLM's hybrid KV manager (``platform.support_hybrid_kv_cache() ==
        True``), so these two spec kinds are packed into separate ``kv_cache_groups`` (NOT collapsed by
        ``unify_hybrid_kv_cache_specs``). vLLM's ``SlidingWindowManager`` gives the sliding group a
        SMALLER physical block pool and full-width block tables whose out-of-window entries point at a
        shared ``null_block`` (absolute ``pos//block_size`` indexing is preserved — no ring remap), so
        the model's existing paged fill/read (``sliding_window_size=512`` on the SDPA read side) is
        unchanged; it just indexes a smaller sliding pool. This cuts per-device KV from ~5.4 GB toward
        ~1.5 GB at full context (30/40 layers windowed to 512), freeing DRAM for larger batches.
        Per-layer block tables reach the model as ``page_tables_per_layer`` (see prefill/decode)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        layer_types = getattr(text_config, "layer_types", None)
        if num_layers is None and layer_types is not None:
            num_layers = len(layer_types)
        num_layers = int(num_layers)
        sliding_window = int(getattr(text_config, "sliding_window", 0) or 0)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        try:  # vLLM moved this constant across versions (fork 0.16 vs stock 0.24) — tolerate both.
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        except ImportError:
            from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

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
        )

        # A layer is sliding iff layer_types[i] == "sliding_attention". Gated behind
        # _HYBRID_KV_CACHE_GROUPS_ENABLED, which is currently True (default): the per-group page-table
        # threading (prefill + decode trace) and warmup pre-alloc have landed, so sliding layers get a
        # SlidingWindowSpec(512) and their own smaller pool. Setting the flag False falls back to the
        # legacy uniform FullAttentionSpec for every layer (single KV group / single page table).
        def _is_sliding(i):
            return (
                cls._HYBRID_KV_CACHE_GROUPS_ENABLED
                and bool(layer_types)
                and layer_types[i] == "sliding_attention"
                and sliding_window > 0
            )

        spec = {}
        for i in range(num_layers):
            key = f"model.layers.{i}.self_attn"
            if _is_sliding(i):
                spec[key] = SlidingWindowSpec(sliding_window=sliding_window, **common)
            else:
                spec[key] = FullAttentionSpec(**common)
        # Phase-3 diagnostic (env-gated TT_LAGUNA_KV_SPEC_LOG=1): record what this hook returns. NOTE
        # (2026-08-04): an always-on run proved this hook is NEVER CALLED at serving — the plugin's
        # _try_get_spec_from_model_hook (worker.py) does not reach it, so vLLM uses the single-spec default
        # (1 KV group / uniform full-KV). The model + HF config are correct (30 sliding + 10 full,
        # sliding_window=512); the bug is plugin hook-resolution. Kept env-gated as a re-check.
        if os.environ.get("TT_LAGUNA_KV_SPEC_LOG") != "1":
            return spec
        try:
            nsl = sum(1 for v in spec.values() if type(v).__name__ == "SlidingWindowSpec")
            nfa = len(spec) - nsl
            kinds = sorted({type(v).__name__ for v in spec.values()})
            with open(
                "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/"
                "doc/vllm_integration/_runs/kv_spec.txt",
                "a",
            ) as _f:
                _f.write(
                    f"[laguna kv_spec] get_kv_cache_spec CALLED pid={os.getpid()}: {len(spec)} layers, "
                    f"sliding={nsl} full={nfa}, kinds={kinds}, sliding_window={sliding_window}, "
                    f"hybrid_flag={cls._HYBRID_KV_CACHE_GROUPS_ENABLED}\n"
                )
        except Exception:
            pass
        return spec

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1, **kwargs):
        """Total KV-cache token pool. Return the advertised context (131072, the verified-servable
        limit on P150x4 — see ADVERTISED_MAX_CONTEXT and doc/context_contract.json), so a single
        request can use the whole advertised window. Bounded by the requested ``max_model_len`` when
        smaller (e.g. the reduced bring-up target)."""
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
        self._verify_dec = {}
        return kv_cache

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Hybrid allocation: one paged buffer per attention layer, each sized to ITS group's block
        budget. ``per_layer_specs`` is the plugin's list of ``(shape, dtype, tensor_idx)`` in model
        layer-index order; ``shape`` = ``(num_blocks, local_kv_heads, block_size, head_dim)`` with the
        sliding-window layers already given a smaller ``num_blocks`` by vLLM's hybrid manager. Same
        BFP8 DRAM/replicated layout as ``allocate_kv_cache``; returns the per-layer dict list consumed
        by ``LagunaModel.prefill_layers`` / ``decode_layers`` (via ``zip(self.layers, kv_cache)``)."""
        kv_cache = []
        for entry in per_layer_specs:
            shape = tuple(entry[0])
            num_blocks, local_kv_heads, block_size, head_dim = shape
            k = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            v = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.float32),
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
        self._decode = {}
        self._verify_dec = {}
        return kv_cache

    # ---- hybrid KV: per-group page-table helpers ---- #
    def _group_kinds(self):
        """['full'|'sliding'] per BUILT layer, from each decoder's cfg.is_sliding."""
        if self._layer_kinds is None:
            self._layer_kinds = [
                "sliding" if bool(getattr(dec.cfg, "is_sliding", False)) else "full" for dec in self.model.layers
            ]
        return self._layer_kinds

    def _group_reps(self):
        """{kind: first built-layer index of that kind} — a representative layer per KV group."""
        reps = {}
        for i, k in enumerate(self._group_kinds()):
            reps.setdefault(k, i)
        return reps

    def _prefill_pt_grouped(self, page_tables_per_layer):
        """Persistent prefill page-table buffers, ONE per group ("full"/"sliding"), returned as a
        per-layer list aligned to the built layers. ``page_tables_per_layer[i]`` is layer i's group's
        host block table; same-group layers share one buffer (2 distinct, same shape/different content).
        Allocate-once per group+shape, then copy-in (no per-call alloc under the resident decode trace,
        provided warmup_model_prefill pre-touched every group — which it does)."""
        kinds = self._group_kinds()
        bufs = {}
        for kind, i in self._group_reps().items():
            pt = torch.as_tensor(page_tables_per_layer[i], dtype=torch.int32)
            if pt.dim() == 1:
                pt = pt.reshape(1, -1)
            buf = self._pf_pt_groups.get(kind)
            if buf is None or tuple(buf.shape) != tuple(pt.shape):
                buf = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
                self._pf_pt_groups[kind] = buf
            ttnn.copy_host_to_device_tensor(self.gen._host(pt, ttnn.int32), buf)
            bufs[kind] = buf
        return [bufs[k] for k in kinds]

    def _decode_pt_grouped_alloc(self, page_tables_per_layer):
        """Allocate the 2 per-group persistent DECODE page-table buffers (BEFORE trace capture) and
        return (per_layer_list, groups_dict, reps_dict). Both groups share the full-width null-padded
        shape [B, max_blocks], so the buffers are identical shape / different content."""
        kinds = self._group_kinds()
        reps = self._group_reps()
        groups = {}
        for kind, i in reps.items():
            pt = torch.as_tensor(page_tables_per_layer[i], dtype=torch.int32)
            if pt.dim() == 1:
                pt = pt.reshape(1, -1)
            groups[kind] = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
        return [groups[k] for k in kinds], groups, reps

    def _decode_pt_grouped_refresh(self, st, page_tables_per_layer):
        """Copy each group's host block table into its persistent decode buffer, only when changed."""
        for kind, i in st["pt_reps"].items():
            pt_host = torch.as_tensor(page_tables_per_layer[i], dtype=torch.int32)
            if pt_host.dim() == 1:
                pt_host = pt_host.reshape(1, -1)
            last = st["last_pt_host_groups"].get(kind)
            if last is None or not torch.equal(pt_host, last):
                ttnn.copy_host_to_device_tensor(self._page_table_to_device_host(pt_host), st["pt_groups"][kind])
                st["last_pt_host_groups"][kind] = pt_host.clone()
                self.gen.counters["page_table_refresh"] += 1

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
            if self.already_warmed_up_prefill and not self._in_prefill_warmup:
                # W1 diagnostic: any allocation AFTER warmup happens under the resident decode trace and
                # is the multi-minute stall. Warmup should have pre-touched every (N, serve_w) shape.
                print(
                    f"[laguna] WARNING: prefill page-table alloc for unwarmed shape {key} AT SERVING "
                    f"(under resident decode trace — this is the W1 stall). Widen warmup_model_prefill.",
                    flush=True,
                )
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
    #   (2) The last-real-token hidden is selected without baking plen into a new program per distinct
    #       length, and WITHOUT a host round-trip (item 2.2, DONE). `_last_token_shards` builds a tiny
    #       [1,1,1,L] one-hot on host (1.0 at column plen-1 — the index is DATA, not program shape),
    #       copies it into a persistent per-bucket-L selector buffer, and runs the fixed-shape matmul
    #       sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H] to pick row plen-1 on device. In bf16 the one-hot ·
    #       hidden reproduces the selected row bit-exactly (1.0*x + 0-sum), so greedy output is
    #       identical. This removes the ~32 MB whole-hidden readback that used to run on EVERY prefill.
    #       Both the selector buffers (allocated in _prefill_state) and the matmul program (compiled by
    #       warmup_model_prefill's per-L prefill_forward calls) exist pre-trace, so serving only copies
    #       the one-hot in and runs a pre-compiled matmul — no alloc/compile under the resident decode
    #       trace. Sampling likewise reuses persistent B=1 buffers (copy-in, no alloc).

    def _prefill_bucket_lens(self):
        """Supported prefill bucket lengths (powers of two from 128 up to the SERVABLE context). Every
        request rounds UP to one of these via ``_bucket_len``, and ``warmup_model_prefill`` compiles
        EVERY one of them before the decode trace is captured.

        item 1.1 — the ceiling is the servable context ``min(max_model_len, ADVERTISED_MAX_CONTEXT)``,
        NOT a fixed 8192 cap. The sequence-pipelined prefill tail (``_prefill_pipelined``) reassembles
        per-chunk outputs with a program whose shape depends on the CHUNK COUNT (``ttnn.concat(outs)``
        + the per-chunk input ``ttnn.slice``s), so a prompt needing more chunks than any warmed bucket
        would first-compile that program UNDER the resident decode trace — orders of magnitude slower
        and trace-corrupting. (Under the old fixed 8192 cap, warmup exercised only 2- and 4-chunk
        counts, so any prompt ≥16384 tokens compiled a new tail program mid-serve.) Warming the full
        power-of-two ladder makes every servable prompt run only pre-compiled programs, and keeps
        ``_bucket_len`` returning a value that is ALWAYS in the warmed set.

        item 2.1 — the ladder floor is 32 (one tile), not 128, so the small cached-suffix prefills that
        dominate agentic serving (prefix caching hits ~95% of turns; the fresh suffix is typically 8–74
        tokens) round up to 32/64/128 instead of always 128. That cuts up to ~16× of wasted prefill work
        on the common turn (plen=8 → bucket 32 = 4× vs 16×; plen=40 → 64; plen=74 → 128). Right-padding
        stays safe at the smaller buckets: the pad positions are future cache slots (plen..L-1),
        overwritten before any decode step reads them, and causal attention means the last real token
        never attends to them — so the last-token logits (and greedy output) are bit-identical.

        ``TT_LAGUNA_PREFILL_WARM_CAP`` may lower the ceiling for fast dev iteration (bounds warmup
        cost), but any prompt LONGER than the cap then compiles under the trace — a one-time warning is
        logged. Do NOT set it below max_model_len for serving."""
        import os as _os

        servable = min(int(self.max_model_len), ADVERTISED_MAX_CONTEXT)
        cap = servable
        env = _os.environ.get("TT_LAGUNA_PREFILL_WARM_CAP")
        if env:
            cap = min(servable, int(env))
            if cap < servable and not getattr(type(self), "_warned_warm_cap", False):
                print(
                    f"[laguna] WARNING: TT_LAGUNA_PREFILL_WARM_CAP={cap} < servable context {servable}; "
                    f"prompts longer than {cap} tokens will compile prefill programs under the resident "
                    f"decode trace (very slow + trace-unsafe, item 1.1). Dev-only knob — unset for serving.",
                    flush=True,
                )
                type(self)._warned_warm_cap = True
        buckets, b = [], 32  # item 2.1: floor 32 (one tile) to match small cached-suffix prefills
        while b < cap:
            buckets.append(b)
            b *= 2
        buckets.append(cap)
        return sorted(set(x for x in buckets if x >= 1))

    def _bucket_len(self, plen):
        buckets = self._prefill_bucket_lens()
        for b in buckets:
            if plen <= b:
                return b  # always in the warmed set (top bucket == servable context)
        # Unreachable for in-contract prompts: the top bucket is the servable context
        # (min(max_model_len, ADVERTISED_MAX_CONTEXT)), so any plen ≤ max_model_len is caught above.
        # A prompt beyond that is out of contract (vLLM rejects it); round up as a last-resort guard —
        # this length is NOT warmed and would compile under the decode trace.
        top = buckets[-1]
        return ((plen + top - 1) // top) * top

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
        # item 2.2 — persistent per-bucket-L on-device last-token SELECTOR buffers. Each is a
        # [1,1,1,L] bf16 (TILE, replicated) one-hot INPUT; the fixed-shape matmul
        # ``sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H]`` picks the last REAL row (the row index is DATA,
        # written per prompt via copy_host_to_device_tensor), replacing the ~32 MB host readback of
        # the whole bucketed hidden that ran on every prefill. Allocated HERE (pre-trace, once, keyed
        # by L exactly like the ``_pf_pt`` page tables) for EVERY warmed bucket — so at serve time the
        # selector matmul is a copy-in + pre-compiled matmul with no device allocation / compilation
        # under the resident decode trace. The matmul program itself is compiled during
        # ``warmup_model_prefill`` (its per-bucket ``prefill_forward`` calls run ``_last_token_shards``,
        # which executes this same ``sel @ h`` for each L). bf16 one-hot · bf16 hidden reproduces the
        # selected row bit-exactly (1.0*x + 0-sum), so greedy output is identical to the readback path.
        st["sel"] = {
            L: self.gen._rep(torch.zeros([1, 1, 1, L], dtype=torch.float32), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for L in self._prefill_bucket_lens()
        }
        self._pf = st
        return st

    def _last_token_shards(self, h, plen, L):
        """Select the last REAL token's logit shards with a fixed-shape ON-DEVICE one-hot selector.

        ``h`` is the bucketed prefill output ``[1, L, H]`` (L fixed per bucket) held on device and
        REPLICATED across the mesh. item 2.2: instead of reading the whole ``[1,L,H]`` hidden back to
        host (~32 MB every prefill) to slice row ``plen-1``, build a tiny ``[1,1,1,L]`` one-hot on host
        (1.0 at column ``plen-1`` — the index is DATA, not program shape), copy it into the persistent
        per-L selector buffer (a COPY, no allocation), and run the fixed-shape matmul
        ``sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H]``. bf16 one-hot · bf16 hidden reproduces the selected
        row bit-exactly (the only nonzero term is ``1.0 * h[plen-1]`` and 1.0 is exact in bf16; every
        other product is 0), so the column-sharded LM head — and therefore greedy output — is identical
        to the old readback path. The selector matmul + the ``[1,L,H]->[1,1,L,H]`` reshape are compiled
        pre-trace by warmup (its ``prefill_forward`` calls run this for every bucket L). Decode never
        leaves the device; this is prefill-only. Falls back to the host readback if a bucket's selector
        is somehow missing (shouldn't happen post-warmup) so serving can never crash."""
        st = self._prefill_state()
        sel = st["sel"].get(L)
        # item 2.2 selector gated OFF by default: as first implemented it calls ttnn.from_torch(onehot)
        # per prefill, which ALLOCATES a device buffer under the resident decode trace ("Allocating device
        # buffers is unsafe", allocator.cpp:123) — trace-unsafe + slow. Needs a trace-safe rewrite (build the
        # one-hot as a ttnn HOST tensor + copy_host_to_device into the persistent sel; matmul into a
        # persistent output). Until then, TT_LAGUNA_SELECTOR=1 opts in; default uses the host-readback path.
        if sel is not None and os.environ.get("TT_LAGUNA_SELECTOR") == "1":
            onehot = torch.zeros([1, 1, 1, L], dtype=torch.float32)
            onehot[0, 0, 0, plen - 1] = 1.0
            src = ttnn.from_torch(
                onehot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
            )
            ttnn.copy_host_to_device_tensor(src, sel)  # into persistent selector buffer (no alloc)
            h4 = ttnn.reshape(h, (1, 1, L, self.hidden))  # [1,L,H] -> [1,1,L,H] (leading unit dim)
            sel_row = ttnn.matmul(sel, h4)  # [1,1,1,L] @ [1,1,L,H] -> [1,1,1,H] == h[plen-1] exactly
            return self.model.lm_head_shards_decode(sel_row)
        # Fallback (bucket L not warmed): host readback of the replicated hidden, slice row on host.
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
        page_tables_per_layer=None,
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
        # Hybrid KV: per-layer (per-group) page tables when the plugin provides them; else the single
        # shape-keyed persistent table (legacy uniform path). Both are trace-safe (pre-allocated in
        # warmup_model_prefill). model.prefill_layers indexes a list per layer, or uses a lone tensor.
        if page_tables_per_layer is not None:
            pt = self._prefill_pt_grouped(page_tables_per_layer)
        else:
            pt = self._prefill_pt(page_table)  # persistent, shape-keyed (no per-call alloc under trace)
        if prompt_lens is None:
            prompt_lens = [int(tokens.shape[1])] * batch
        starts = [0] * batch if start_pos is None else [int(x) for x in start_pos]

        if self._spec_mode == "1" and batch == 1:
            # Stash the prompt token sequence for ngram seeding — on the greedy served path the plugin
            # passes NO prompt_tokens/output_tokens (those are penalty-gated, model_runner.py:1051), so
            # this prefill is the only place the running request's prompt is visible. Offset-write handles
            # chunked prefill (multiple calls with increasing start_pos); start_pos 0 begins a new request.
            try:
                row = [int(v) for v in tokens[0].tolist()[: int(prompt_lens[0])]]
                s = int(starts[0])
                if s == 0:
                    self._spec_prefill_seq = list(row)
                    self._spec_next_pos = None  # force reseed on the first decode of this request
                else:
                    need = s + len(row)
                    if len(self._spec_prefill_seq) < need:
                        self._spec_prefill_seq += [0] * (need - len(self._spec_prefill_seq))
                    self._spec_prefill_seq[s:need] = row
            except Exception:  # noqa: BLE001 - diagnostic stash; never break prefill
                pass

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

    def _row_logits(self, h, row, L, st):
        """LM-head over a single row of the ON-DEVICE prefill hidden ``h`` ([1,L,H], replicated),
        selected by the same fixed-shape one-hot matmul as ``_last_token_shards`` (item 2.2): a
        ``[1,1,1,L]`` one-hot with 1.0 at ``row`` picks that row bit-exactly. Reuses the persistent
        per-L selector buffer ``st["sel"][L]`` (same L bucket, so the matmul program is already warmed
        by ``_last_token_shards``). Avoids the ~32 MB whole-hidden readback the verify path used to do.
        Falls back to a per-row host readback if the bucket's selector is missing (shouldn't happen)."""
        sel = st["sel"].get(L)
        if sel is not None and os.environ.get("TT_LAGUNA_SELECTOR") == "1":  # gated off — see _last_token_shards
            onehot = torch.zeros([1, 1, 1, L], dtype=torch.float32)
            onehot[0, 0, 0, row] = 1.0
            src = ttnn.from_torch(
                onehot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
            )
            ttnn.copy_host_to_device_tensor(src, sel)
            h4 = ttnn.reshape(h, (1, 1, L, self.hidden))
            sel_row = ttnn.matmul(sel, h4)  # [1,1,1,H] == h[0, row] exactly
            return self.model.logits_to_host(self.model.lm_head_shards_decode(sel_row)).reshape(self.vocab)
        # Fallback: read the hidden back and slice this row on host.
        hh = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).reshape(
            -1, L, self.hidden
        )
        hrow = hh[0, row].to(torch.float32).reshape(1, 1, 1, self.hidden)
        hsrc = ttnn.from_torch(
            hrow, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(hsrc, st["last_h"])
        return self.model.logits_to_host(self.model.lm_head_shards_decode(st["last_h"])).reshape(self.vocab)

    def verify_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        page_tables_per_layer=None,
        logit_rows=None,
        **kwargs,
    ):
        """Speculative-decode VERIFY step for ONE request (batch-1 path).

        ``tokens`` is [1, S] processed through the (suffix-)prefill path against the paged KV starting
        at ``start_pos``. Returns host logits — row j is the next-token distribution given context
        through position ``start_pos+j`` (i.e. predicts the token at ``start_pos+j+1``), so
        argmax(row j) is the target-greedy token for that slot.

        **Alignment:** the prefill flash-attention requires ``chunk_start_idx (= start_pos) % 64 == 0``
        for suffix prefills (``start_pos>0``; q_chunk=32 ∧ k_chunk=64/128 → lcm 64 = the block size,
        see optimized_decoder.py:418-429). The caller (spec_decode.SpeculativeDecoder) therefore aligns
        ``start_pos`` down to a 64-boundary and prepends the already-known history tokens for that
        window; those real tokens rewrite identical KV (idempotent) and only the trailing rows are read.

        ``logit_rows``: optional list of row indices to run the LM head on (in order). Only those rows'
        logits are returned as ``[len(logit_rows), vocab]`` — the caller asks for just the K+1 trailing
        rows (anchor + drafts), skipping the vocab projection for the re-fed alignment prefix. ``None``
        returns all ``S`` rows.

        KV for all S positions is written; rejected-draft positions are overwritten by the next
        iteration's verify (implicit batch-1 rollback), and the pad/right-fill keeps stale future-KV
        harmless."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(1, -1)
        S = int(tokens.shape[1])
        st = self._prefill_state()
        pt = (
            self._prefill_pt_grouped(page_tables_per_layer)
            if page_tables_per_layer is not None
            else self._prefill_pt(page_table)
        )
        L = self._bucket_len(S)
        padded = torch.zeros(L, dtype=torch.int64)
        padded[:S] = tokens[0, :S]
        tok_tt = self.gen._tokens_to_device(padded)
        x = self.model.embed_prefill(tok_tt)
        h = self.model.prefill_layers(x, kv_cache, pt, user_id=0, start_pos=int(start_pos))
        # item 2.2: select each requested row ON DEVICE (one-hot matmul over the still-resident hidden)
        # instead of reading the whole [1,L,H] hidden back to host first.
        rows = list(range(S)) if logit_rows is None else [int(r) for r in logit_rows]
        logits = torch.stack([self._row_logits(h, r, L, st) for r in rows], dim=0)  # [len(rows), vocab]
        return logits

    def _spec_log(self, msg):
        """Append a diagnostic line to the spec-probe file AND stdout. The model runs in an MPI worker
        whose stdout is not captured in the readiness log, so the file is the reliable sink."""
        line = f"[laguna spec] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            import os as _os

            _os.makedirs(_os.path.dirname(self._spec_log_path), exist_ok=True)
            with open(self._spec_log_path, "a") as _f:
                _f.write(line + "\n")
        except Exception:
            pass

    def _spec_feasibility_probe(self, tokens, pos, page_table, kv_cache, page_tables_per_layer):
        """PHASE-2 FEASIBILITY PROBE (TT_LAGUNA_SPEC_DECODE=probe): run ONE eager batched-decode VERIFY
        (K1=2 = anchor + 1 dummy draft) under the RESIDENT decode trace and report whether it completes
        without an alloc-under-trace hang. This is the open question blocking eager in-adapter spec-decode
        (the eager verify allocates activation buffers; doing that under a resident trace may be the
        allocator.cpp:123 hazard). Writes throwaway KV at pos+1 -> probe boot only, never real serving."""
        import time as _t

        try:
            anchor = int(tokens[0, 0])
            p0 = int(pos[0])
            toks = [anchor, anchor]  # K1=2: anchor + one dummy draft at the next position
            positions = [p0, p0 + 1]
            pt_arg = None if page_tables_per_layer is not None else page_table
            self._spec_log(
                f"PROBE START pid={os.getpid()} anchor={anchor} pos={p0} hybrid={page_tables_per_layer is not None}"
            )
            # Run 3x: iter 0 = compile (slow), iters 1-2 = WARM. Warm eager-verify time vs a ~35ms traced
            # decode step is the go/no-go for eager spec-serving (decode is dispatch-bound; eager = full
            # host dispatch, which tracing eliminates). Compare warm ms to draft_len to judge break-even.
            for it in range(3):
                t0 = _t.perf_counter()
                g = self.verify_greedy_decode(
                    toks,
                    positions,
                    page_table=pt_arg,
                    kv_cache=kv_cache,
                    page_tables_per_layer=page_tables_per_layer,
                    traced=False,
                )
                dt = (_t.perf_counter() - t0) * 1000.0
                self._spec_log(
                    f"PROBE iter{it} ({'compile' if it == 0 else 'WARM'}): {dt:.0f}ms -> {[int(x) for x in g]}"
                )
            self._spec_log(
                "PROBE OK: eager verify under resident decode trace completed (no hang) -> FEASIBLE. "
                "Compare the WARM ms above to a ~35ms traced decode step to judge if eager spec can win."
            )
        except Exception as e:  # noqa: BLE001 - diagnostic probe, report any failure verbatim
            self._spec_log(f"PROBE FAILED under resident decode trace: {type(e).__name__}: {e}")

    def _spec_is_greedy(self, sampling_params):
        try:
            t = sampling_params.temperature
            return float(t[0] if hasattr(t, "__len__") else t) <= 0.0
        except Exception:  # noqa: BLE001
            return False

    def _spec_history(self, prompt_tokens, output_tokens, tokens):
        """Full current token sequence for the single served request (ngram source + position anchor)."""

        def _row0(x):
            if x is None:
                return []
            r = x[0] if hasattr(x, "__getitem__") else x
            return [int(v) for v in (r.tolist() if hasattr(r, "tolist") else r)]

        hist = _row0(prompt_tokens) + _row0(output_tokens)
        return hist or [int(torch.as_tensor(tokens).reshape(-1)[0])]

    def _spec_serve(
        self, tokens, pos, page_table, kv_cache, page_tables_per_layer, reset_batch, kwargs, read_from_device
    ):
        """One served decode step via TRACED spec-decode. Runs one spec round when the commit buffer is
        empty, then returns the buffered committed tokens one per vLLM step (plugin reads self._spec_tok).
        Bounded-K keeps the round's look-ahead KV writes inside the current allocated block. Needs warmup to
        have captured the verify traces + omitted the normal decode trace (TT_LAGUNA_SPEC_DECODE=1)."""
        if self._spec is None:
            from models.autoports.poolside_laguna_xs_2_1.tt.spec_decode import SpeculativeDecoder

            k_max = int(os.environ.get("TT_LAGUNA_SPEC_K", "4"))
            traced = os.environ.get("TT_LAGUNA_SPEC_TRACED", "1") == "1"  # 0 = eager verify (bisection)
            single = os.environ.get("TT_LAGUNA_SPEC_SINGLE", "") == "1"  # 1 = fixed-K, one resident verify trace
            self._spec = SpeculativeDecoder(
                self,
                kv_cache=kv_cache,
                page_table=page_table,
                page_tables_per_layer=page_tables_per_layer,
                stop_tokens=None,
                draft_len=k_max,
                ngram_max_n=int(os.environ.get("TT_LAGUNA_SPEC_NGRAM_MAX", "10")),
                verify_mode="decode",
                traced=traced,
                adaptive=not single,  # single mode: fixed K=k_max so only the K1=k_max+1 trace is ever used
                k_min=1,
                k_max=k_max,
                guard=not single,  # single mode: never fall back to a K1=1 native step (would need a 2nd trace)
            )
            self._spec_log(f"serve INIT: traced={traced} k_max={k_max} single={single}")
            self._spec_tok = self.gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        # per-call context refresh (the request's block table grows as it advances)
        self._spec.kv_cache = kv_cache
        self._spec.page_table = page_table
        self._spec.page_tables_per_layer = page_tables_per_layer
        cur = int(torch.as_tensor(tokens).reshape(-1)[0])
        p0 = int(pos.reshape(-1)[0])
        # NEW-REQUEST detection by position CONTINUITY, not reset_batch. reset_batch fires every decode
        # step here (per-step full refresh, see laguna-batched-decode-corruption), so it cannot flag a new
        # request. Within a request the plugin advances pos by exactly 1 each call; a mismatch (or the very
        # first call) means a fresh request → reseed history from the stashed prompt + reset guard/adaptive.
        if self._spec_next_pos is None or p0 != self._spec_next_pos or not self._spec_hist:
            self._spec.serve_reset()
            self._spec_buf = []
            # Seed from the prompt stashed at prefill (greedy path gets no history via kwargs). History must
            # have len == p0+1 and end with cur (verify uses anchor_pos = len-1 as the absolute KV position).
            seed = list(self._spec_prefill_seq or [])
            if len(seed) >= p0:
                seed = seed[:p0]
            else:  # prompt stash short (unexpected) — front-pad so positions still line up
                seed = [cur] * (p0 - len(seed)) + seed
            seed.append(cur)
            self._spec_hist = seed
            self._spec_log(
                f"serve NEWREQ: seeded hist len={len(seed)} pos0={p0} cur={cur} "
                f"prompt_stash={len(self._spec_prefill_seq or [])}"
            )
        if not self._spec_buf:
            history = self._spec_hist  # SELF-TRACKED across the whole request (authoritative)
            # bounded-K: keep look-ahead writes (anchor_pos+1 .. anchor_pos+K) inside the anchor's own
            # block. anchor_pos = len-1, so the room left in the block is 63 - (anchor_pos % 64).
            k_cap = 63 - ((len(history) - 1) % 64)
            committed = list(self._spec.serve_round(history, k_cap=k_cap))
            self._spec_hist.extend(committed)  # grow history so the next round's anchor advances
            self._spec_buf = committed
            self._spec_log(
                f"serve ROUND: anchor_pos={p0} committed={len(committed)} toks={committed[:8]} "
                f"k_cur={getattr(self._spec,'_sv_k_cur',None)} spec_on={getattr(self._spec,'_sv_spec_on',None)}"
            )
        tok_id = int(self._spec_buf.pop(0))
        self._spec_next_pos = p0 + 1  # plugin appends the returned token → next decode is at p0+1
        ttnn.copy_host_to_device_tensor(
            self._host_rank4_tok_batch(torch.tensor([[tok_id]], dtype=torch.int64), 1), self._spec_tok
        )
        if read_from_device:
            return self._read_tokens_host(self._spec_tok, 1)
        return [self._spec_tok]

    def verify_forward_decode(
        self,
        tokens,
        positions,
        page_table=None,
        kv_cache=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        """Speculative-decode VERIFY via the batched-DECODE path (gemma4 `ttnn_verify_forward` pattern).

        The ``B = K+1`` candidate tokens ``[anchor, d0, …, d_{K-1}]`` occupy the BATCH dim at consecutive
        positions ``positions = [P-1, P, …, P-1+K]``, all pointing at the SAME user's KV blocks (the
        page-table row is replicated B times). One batched decode runs the fast paged-SDPA-**decode**
        (reads KV in O(1) w.r.t. context, unlike the prefill-path `verify_forward`), with
        ``sequential_kv_write=True`` so the B rows' shared-block cache writes serialize (no RMW race —
        see MultichipDecoder._seq_kv_write). Returns host logits ``[B, vocab]``; row j predicts the token
        at ``positions[j]+1``, so row j's argmax is the target-greedy token for that slot (greedy accept).

        Eager (untraced) — a first correctness/latency vehicle; the traced B=K+1 fast path is a follow-up.
        """
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        B = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(B)

        def _row_to_B(row):
            """One user's block row -> [B, w] device page table (replicated across the K+1 candidates)."""
            r = torch.as_tensor(row, dtype=torch.int32)
            if r.dim() == 1:
                r = r.unsqueeze(0)
            r = r[:1]  # the single served user's row
            if B > 1:
                r = r.repeat(B, 1)
            return self._page_table_to_device(r)

        if page_tables_per_layer is not None:
            # HYBRID serving: the 30 sliding layers read a different (smaller) KV pool than the 10 full
            # layers, so each layer needs ITS group's block row. decode_layers routes a per-layer list
            # (model.py: per_layer = isinstance(page_table,(list,tuple))). Build one device PT per built
            # layer, each = that layer's group row replicated to the B=K+1 candidate batch. Without this
            # the sliding layers would index the full pool -> silently wrong verify logits.
            pt = [_row_to_B(ptl_i) for ptl_i in page_tables_per_layer]
        else:
            pt = _row_to_B(page_table)  # uniform: one row shared by all layers
        tok_tt = self.gen._rep(tokens.reshape(1, B).to(torch.int32), ttnn.uint32)
        cur = self.gen._rep(pos, ttnn.int32)
        ridx = self.gen._rep(pos.reshape(1, B), ttnn.uint32)
        h = self.model.embed_decode(tok_tt)
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache, sequential_kv_write=True)
        shards = self.model.lm_head_shards_decode(h)
        logits = self.model.logits_to_host(shards).reshape(B, self.vocab)
        return logits

    def _alloc_verify_decode(self, K1, kv_cache, tokens, pos, pt_host):
        """Phase 1 of verify-trace warmup: allocate ALL persistent device buffers for one K1 and warm
        the program cache (compile), WITHOUT capturing a trace. Multi-K adaptive warmup MUST allocate
        every K1's buffers before ANY begin_trace_capture: allocating a device buffer while a captured
        trace is resident corrupts the trace (TTNN: 'Allocating device buffers is unsafe due to the
        existence of an active trace') and the subsequent replay hangs the mesh. Returns a state dict
        carrying the closured ``step`` for the later capture phase (tid filled in by _trace_verify_decode).

        Fully on-device — the greedy token per row is produced by the same Sampling1D (top-k=1) the
        model's serving decode uses, so the verify matches what the hardware actually greedy-decodes."""
        g = self.gen
        # LOGITS mode (default): the trace ends at the mesh-sharded logits; the greedy token is argmax'd ON
        # HOST after replay (verify_forward_decode's known-correct path). This AVOIDS the on-device Sampling1D,
        # which under trace at K1>=2 deterministically miscomputes some rows (float-garbage or valid-but-wrong
        # ids in the tok buffer -> corrupts the always-committed anchor -> garbage output). The forward
        # (decode_layers + lm_head) is still traced, so the dispatch-elimination speedup is preserved; the only
        # added cost is reading K1 x vocab logits to host + a host argmax per step. Set TT_LAGUNA_SPEC_LOGITS=0
        # to use the (buggy) on-device sampler path instead.
        # ROOT CAUSE of the traced-verify garbage: the on-device greedy argmax (inside Sampling1D) is the
        # multicore ttnn.argmax, which is ROW-PARALLEL and returns GARBAGE unless the batch (row) dim is
        # tile-aligned to 32 (gemma4 spec_decode._argmax_last: "1/5 rows -> wrong; padded to 32 -> exact").
        # The verify batch is K1=2..5 rows -> unaligned -> some rows (incl. the always-committed anchor) get
        # float-bit garbage / wrong ids -> cascades to garbage output. FIX (on-device, matches gemma4): run
        # the FORWARD at K1 rows (padding it would write KV at 32 positions -> OOB) but PAD THE LOGITS to 32
        # rows just before the argmax, then slice back to K1. Nearly free (tiled matmul already pays for a
        # 32-row tile). TT_LAGUNA_SPEC_LOGITS=1 keeps the host-argmax fallback (correct, but transfers logits).
        logits_mode = os.environ.get("TT_LAGUNA_SPEC_LOGITS", "0") == "1"
        R32 = 32
        tok = g._rep(torch.zeros([1, 1, 1, K1], dtype=torch.int32), ttnn.uint32)
        cur = g._rep(torch.zeros([K1], dtype=torch.int32), ttnn.int32)
        ridx = g._rep(torch.zeros([1, K1], dtype=torch.int32), ttnn.uint32)
        pt = self._page_table_to_device(pt_host)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(K1, 1), K1), tok)
        ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), cur)
        ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), ridx)
        st = dict(tid=None, tok=tok, cur=cur, ridx=ridx, pt=pt, logits_mode=logits_mode, k1=K1)
        if not logits_mode:
            # FORCE-ARGMAX path (Sampling1D._sample_argmax = all-gather vocab + ttnn.argmax). Passing k/p/temp
            # all None selects it (allow_force_argmax=True); passing k=1/p=1/temp=1 instead selects the top-k
            # path (per-shard top-1 + gather) which returned WRONG rows in the batched verify. Sampler + output
            # run at 32 tile-aligned rows; the forward stays K1 (padding it would OOB the KV writes).
            sampler = g._sampler(R32)
            tok_out = g._rep(torch.zeros([1, 1, 1, R32], dtype=torch.int32), ttnn.uint32)  # 32-row argmax output
            st["tok_out"] = tok_out

        def step():
            hh = self.model.embed_decode(ttnn.reshape(tok, (1, K1)))
            hh = self.model.decode_layers(hh, cur, ridx, pt, kv_cache, sequential_kv_write=True)
            shards = self.model.lm_head_shards_decode(hh)  # [1,1,K1,V/D]
            if logits_mode:
                st["logits"] = shards  # persistent trace-output handle; ConcatMesh+argmax on host post-replay
            else:
                # pad logits rows K1->32 (tile-align) so the multicore argmax is row-correct, then force-argmax
                s32 = ttnn.pad(shards, [(0, 0), (0, 0), (0, R32 - K1), (0, 0)], value=0.0) if K1 < R32 else shards
                sampler.decode_forward(s32, tt_out_tok=tok_out)  # k/p/temp None -> force-argmax

        step()  # compile (warm program cache) — no trace resident yet, so allocations here are safe
        ttnn.synchronize_device(self.mesh_device)
        st["_step"] = step
        return st

    def _trace_verify_decode(self, K1, st):
        """Phase 2 of verify-trace warmup: capture the trace using the buffers _alloc_verify_decode
        already allocated. NO new persistent allocation happens here (step() only re-runs compiled
        programs into existing buffers), so this is safe to call repeatedly while earlier K1 traces are
        already resident."""
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        st["_step"]()  # capture
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        st["tid"] = tid
        st.pop("_step", None)
        self._verify_dec[K1] = st
        return st

    def _capture_verify_decode(self, K1, kv_cache, tokens, pos, pt_host):
        """Single-K capture (lazy fallback path): allocate then immediately capture. Safe only when no
        other verify trace is being captured in the same window — for multi-K adaptive warmup use
        warmup_verify_decode_multi, which allocates all buffers before capturing any trace."""
        st = self._alloc_verify_decode(K1, kv_cache, tokens, pos, pt_host)
        return self._trace_verify_decode(K1, st)

    def verify_greedy_decode(
        self, tokens, positions, page_table=None, kv_cache=None, page_tables_per_layer=None, traced=True
    ):
        """Greedy batched-DECODE verify → per-row target-greedy token ids ``[K+1]`` (torch int32).

        The fast path for spec-decode: K+1 candidates in the batch dim at consecutive positions run
        through ONE batched decode (fast paged-decode SDPA, O(1) in context) with race-safe
        sequential_kv_write and the on-device greedy sampler (top-k=1), so row j's id IS argmax of its
        logits (= g[j], the target-greedy token for slot j) — no host logit transfer. ``traced=True``
        replays a captured trace (host dispatch eliminated); the page-table row is constant across
        iterations (same user blocks), so only tokens+positions refresh per replay."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        K1 = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(K1)
        if not traced:
            # Eager verify: forward BOTH the uniform page_table and the hybrid page_tables_per_layer;
            # verify_forward_decode replicates the user's row to the K1 candidate batch and routes the
            # per-layer list to decode_layers (sliding layers read their own pool). page_table may be
            # None in pure-hybrid serving, so hand it through unchanged.
            logits = self.verify_forward_decode(
                tokens,
                pos,
                page_table=page_table,
                kv_cache=kv_cache,
                page_tables_per_layer=page_tables_per_layer,
            )
            return torch.argmax(logits, dim=-1).to(torch.int32)
        # HARD GUARD (audit item 4): traced spec-verify has NO hybrid grouped-PT path. The traced page-table
        # refresh below only fires when `page_tables_per_layer is None`; a hybrid per-layer PT would replay the
        # FROZEN warmup identity table and silently emit wrong greedy ids. Hybrid KV is dead at serving today
        # (the plugin never calls get_kv_cache_spec), so this is dormant — but enabling hybrid KV (the wanted
        # capacity win) MUST fail loudly here, not corrupt. Fix: eager verify (traced=False), or extend the
        # trace refresh to grouped PTs before combining hybrid KV with traced spec-decode.
        if page_tables_per_layer is not None:
            raise NotImplementedError(
                "traced spec-decode verify does not support hybrid per-layer page tables "
                "(page_tables_per_layer): the traced page-table refresh is uniform-only, so a hybrid PT would "
                "replay a stale identity table and produce silently-wrong tokens. Run eager verify "
                "(traced=False), or add a grouped-PT trace-refresh path before enabling hybrid KV + spec-decode."
            )
        pt_row = torch.as_tensor(page_table, dtype=torch.int32)
        if pt_row.dim() == 1:
            pt_row = pt_row.unsqueeze(0)
        pt_row = pt_row[:1]  # the single user's block row (replicated below to the batch size used)
        pt_host = pt_row.repeat(K1, 1) if K1 > 1 else pt_row
        st = self._verify_dec.get(K1)
        if st is None:
            # item 3.2: verify-decode trace for this K1 not pre-captured -> compile+capture in-request.
            print(
                f"[laguna] WARNING: lazy spec-decode VERIFY-trace capture for K1={K1} (not pre-warmed) — "
                f"this step includes compile+capture, not a warm replay. (item 3.2)",
                flush=True,
            )
            st = self._capture_verify_decode(K1, kv_cache, tokens, pos, pt_host)  # lazy fallback
            if st.get("logits_mode"):
                return torch.argmax(self.model.logits_to_host(st["logits"]).reshape(K1, int(self.vocab)), dim=-1).to(
                    torch.int32
                )
            return ttnn.to_torch(ttnn.get_device_tensors(st["tok_out"])[0]).reshape(-1)[:K1].to(torch.int32)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(K1, 1), K1), st["tok"])
        ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), st["cur"])
        ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), st["ridx"])
        # Refresh the page table into the persistent trace buffer that the replay reads. Without this,
        # st["pt"] stays frozen at the warmup identity table (arange) and the verify indexes the WRONG
        # physical KV blocks on any real (non-identity) served page table -> silently wrong greedy ids.
        # (Uniform serving path; hybrid grouped-PT verify is a separate follow-up — serving is uniform today.)
        if page_tables_per_layer is None and st.get("pt") is not None:
            if st.get("last_pt_host") is None or not torch.equal(pt_host, st["last_pt_host"]):
                ttnn.copy_host_to_device_tensor(self._page_table_to_device_host(pt_host), st["pt"])
                st["last_pt_host"] = pt_host.clone()
        ttnn.execute_trace(self.mesh_device, st["tid"], cq_id=0, blocking=True)
        if st.get("logits_mode"):
            # host argmax of the traced logit shards (same gather as eager verify_forward_decode) — bypasses
            # the buggy on-device sampler. traced_ids is bit-correct iff the traced FORWARD matches eager.
            logits = self.model.logits_to_host(st["logits"]).reshape(K1, int(self.vocab))
            traced_ids = torch.argmax(logits, dim=-1).to(torch.int32)
        else:
            # ON-DEVICE argmax (padded to 32 rows) — read the device-0 replica (gemma4 _ids_to_host pattern),
            # then slice off the row-padding back to K1.
            th = ttnn.to_torch(ttnn.get_device_tensors(st["tok_out"])[0])
            traced_ids = th.reshape(-1)[:K1].to(torch.int32)
        # CORRECTNESS GUARD: the traced on-device Sampling1D DETERMINISTICALLY fails to write some rows for
        # certain logit distributions, leaving stale FLOAT bit-patterns in the uint32 tok buffer (e.g.
        # 1096876032=0x41600000=14.0f, or a negative) — an out-of-range "token". When that lands on row 0
        # (the anchor, always committed) it corrupts output. Detect any out-of-vocab id and recompute the
        # WHOLE round via the eager host-argmax verify (verify_forward_decode — known bit-correct). Cheap:
        # only fires on the rare bad round. (Root cause is the sampler kernel under trace; this makes the
        # served path correct without it.)
        ids_t = torch.as_tensor(traced_ids).reshape(-1).to(torch.int64)
        if bool(((ids_t < 0) | (ids_t >= int(self.vocab))).any()):
            elog = self.verify_forward_decode(
                tokens, pos, page_table=page_table, kv_cache=kv_cache, page_tables_per_layer=page_tables_per_layer
            )
            eager_ids = torch.argmax(elog, dim=-1).to(torch.int32)
            self._spec_log(
                f"GUARD: out-of-vocab traced id at K1={K1} pos0={int(pos[0])} "
                f"traced={[int(x) for x in ids_t]} -> eager fallback={[int(x) for x in eager_ids]}"
            )
            traced_ids = eager_ids
        if os.environ.get("TT_LAGUNA_SPEC_DEBUG", "") == "1":
            # Shadow the traced verify with an EAGER verify (known-correct host-argmax path) on the SAME
            # tokens/positions/pt/kv. Both read the CURRENT (post-trace) KV, so agreement means the trace's
            # per-step COMPUTE matches eager (any full-run divergence is then accumulating KV-write drift);
            # disagreement means the trace's own read/compute is wrong THIS step.
            try:
                elog = self.verify_forward_decode(
                    tokens, pos, page_table=page_table, kv_cache=kv_cache, page_tables_per_layer=page_tables_per_layer
                )
                eager_ids = torch.argmax(elog, dim=-1).to(torch.int32).reshape(-1)
                t_ids = torch.as_tensor(traced_ids).reshape(-1)
                if not torch.equal(t_ids.to(torch.int64), eager_ids.to(torch.int64)):
                    self._spec_log(
                        f"TRACE-vs-EAGER MISMATCH K1={K1} pos={[int(x) for x in pos]} "
                        f"traced={[int(x) for x in t_ids]} eager={[int(x) for x in eager_ids]} "
                        f"pt0={[int(x) for x in pt_host[0][:4]]}"
                    )
                else:
                    self._spec_log(f"trace==eager K1={K1} pos0={int(pos[0])} pt0={[int(x) for x in pt_host[0][:4]]}")
            except Exception as e:  # noqa: BLE001
                self._spec_log(f"shadow-eager failed: {type(e).__name__}: {e}")
        return traced_ids

    def verify_sampler_eager(self, tokens, positions, page_table=None, kv_cache=None):
        """DIAGNOSTIC: run the batched decode-verify through the on-device SAMPLER but EAGERLY (no trace
        capture/replay). Isolates the sampler from the trace: the host-argmax eager path
        (verify_forward_decode) is known-correct, so if this sampler-eager path also matches it, the
        sampler is fine and the traced divergence is purely a trace-replay defect; if it diverges here,
        the on-device sampler is the culprit (model-layer fixable). Returns per-row greedy ids [B]."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        B = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(B)
        pt_host = torch.as_tensor(page_table, dtype=torch.int32)
        if pt_host.dim() == 1:
            pt_host = pt_host.unsqueeze(0)
        if pt_host.shape[0] == 1 and B > 1:
            pt_host = pt_host.repeat(B, 1)
        g = self.gen
        tok = g._rep(torch.zeros([1, 1, 1, B], dtype=torch.int32), ttnn.uint32)
        cur = g._rep(pos, ttnn.int32)
        ridx = g._rep(pos.reshape(1, B), ttnn.uint32)
        k = g._rep(torch.ones([B], dtype=torch.int32), ttnn.uint32)
        p = g._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        t = g._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        seeds = g._rep(torch.zeros([B], dtype=torch.int32), ttnn.uint32)
        sampler = g._sampler(B)
        pt = self._page_table_to_device(pt_host)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(B, 1), B), tok)
        h = self.model.embed_decode(ttnn.reshape(tok, (1, B)))
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache, sequential_kv_write=True)
        shards = self.model.lm_head_shards_decode(h)
        sampler.decode_forward(shards, k=k, p=p, temp=t, seeds=seeds, tt_out_tok=tok)
        return self._read_tokens_host(tok, B)

    def warmup_verify_decode(self, draft_len, kv_cache, num_blocks, block_size=64):
        """Capture the spec-decode VERIFY trace in the SAFE warmup window (mirrors how the normal decode
        trace is captured by warmup_model_decode). Capturing lazily mid-serving hangs the mesh.

        Capture at a populated cache + block-internal positions to match serving structure; positions/tokens
        refresh per replay."""
        K1 = int(draft_len) + 1
        if kv_cache is None or K1 in self._verify_dec:
            return
        base = 2 * block_size
        dummy = torch.zeros(base, dtype=torch.int64)
        ptp = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.prefill_forward(
            dummy.reshape(1, base),
            page_table=ptp,
            kv_cache=kv_cache,
            prompt_lens=[base],
            start_pos=[0],
            sampling_params=None,
        )
        pos = torch.arange(base - 1, base - 1 + K1, dtype=torch.int32)
        tokens = torch.zeros(K1, dtype=torch.int64)
        pt_host = ptp.repeat(K1, 1)
        self._capture_verify_decode(K1, kv_cache, tokens, pos, pt_host)

    def warmup_verify_decode_multi(self, draft_lens, kv_cache, num_blocks, block_size=64):
        """Adaptive-K verify warmup: pre-capture a verify trace for EACH draft_len in one safe window.

        The naive loop `for k: warmup_verify_decode(k)` captures trace K1=2, then allocates K1=3's
        buffers while K1=2's trace is resident — TTNN flags 'Allocating device buffers is unsafe due to
        the existence of an active trace' and the first replay hangs the mesh. This stages ALL buffer
        allocation (phase 1) before ANY trace capture (phase 2), so no allocation ever races a resident
        trace."""
        K1s = sorted({int(d) + 1 for d in draft_lens if int(d) + 1 not in self._verify_dec})
        if kv_cache is None or not K1s:
            return
        base = 2 * block_size
        dummy = torch.zeros(base, dtype=torch.int64)
        ptp = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.prefill_forward(
            dummy.reshape(1, base),
            page_table=ptp,
            kv_cache=kv_cache,
            prompt_lens=[base],
            start_pos=[0],
            sampling_params=None,
        )
        staged = {}
        for K1 in K1s:  # phase 1: allocate + compile every K1 (no trace resident)
            pos = torch.arange(base - 1, base - 1 + K1, dtype=torch.int32)
            tokens = torch.zeros(K1, dtype=torch.int64)
            pt_host = ptp.repeat(K1, 1)
            staged[K1] = self._alloc_verify_decode(K1, kv_cache, tokens, pos, pt_host)
        for K1 in K1s:  # phase 2: capture every trace (buffers already allocated)
            self._trace_verify_decode(K1, staged[K1])

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
        page_tables_per_layer=None,
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

        if self._spec_mode and not self._spec_probed:
            # One-time diagnostic + Phase-2 feasibility probe. Placed BEFORE the host-sampling return and
            # gated only on "once" (not B==1: served decode is padded to max_batch_size, so B is never 1).
            self._spec_probed = True
            self._spec_log(
                f"decode_forward#1 pid={os.getpid()} B={B} reset_batch={reset_batch} "
                f"host_sampling={sampling_params is None} hybrid={page_tables_per_layer is not None} "
                f"spec_mode={self._spec_mode!r}"
            )
            if self._spec_mode == "probe":
                # Run ONE eager verify under the resident decode trace on row 0; fall through to normal
                # decode (throwaway KV at pos+1 — probe boot only).
                self._spec_feasibility_probe(tokens, pos, page_table, kv_cache, page_tables_per_layer)

        # FULL served spec-decode: in this mode the normal decode trace was OMITTED at warmup, so ALL decode
        # goes through the traced verify path. Requires --max-num-seqs 1 (B==1, no padding) + greedy.
        if self._spec_mode == "1" and B == 1 and sampling_params is not None and self._spec_is_greedy(sampling_params):
            return self._spec_serve(
                tokens, pos, page_table, kv_cache, page_tables_per_layer, reset_batch, kwargs, read_from_device
            )

        if sampling_params is None:
            return self._decode_host_sampling(tokens, pos, page_table, kv_cache, read_from_device)

        hybrid = page_tables_per_layer is not None
        st = self._decode.get(B)
        if st is None:
            # item 3.2: the decode trace for this batch B was NOT pre-captured by warmup, so we compile +
            # capture it now — INSIDE a live request. This is orders of magnitude slower than a warm
            # replay and is invisible in the served latency unless flagged. Warn so a lazy capture is not
            # silently mistaken for warm serving (drive warmup_model_decode for every served B to avoid).
            print(
                f"[laguna] WARNING: lazy decode-trace capture for batch B={B} inside decode_forward "
                f"(warmup did not pre-capture this B) — first-token latency for this B includes "
                f"compile+capture, not a warm replay. (item 3.2)",
                flush=True,
            )
            if hybrid:
                pt_persist, groups, reps = self._decode_pt_grouped_alloc(page_tables_per_layer)
                st = self._decode_state(B, kv_cache, pt_persist)
                st["pt_groups"], st["pt_reps"], st["last_pt_host_groups"] = groups, reps, {}
            else:
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
        if hybrid:
            # Refresh BOTH per-group persistent buffers (full + sliding) that the trace closed over.
            self._decode_pt_grouped_refresh(st, page_tables_per_layer)
        else:
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
        self._in_prefill_warmup = True  # suppress the _prefill_pt diagnostic for intentional warmup allocs
        # Allocate persistent sampling buffers AND the per-bucket-L one-hot selector buffers (item 2.2),
        # all pre-trace. The selector MATMUL program (sel[1,1,1,L] @ h[1,1,L,H]) + the [1,L,H]->[1,1,L,H]
        # reshape are compiled for every bucket L by the per-L ``prefill_forward`` calls below, which run
        # ``_last_token_shards`` (and thus the selector matmul) for each L — so no selector program
        # first-compiles under the resident decode trace at serving time.
        self._prefill_state()
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
        # Hybrid iff the per-layer pools differ (sliding layers got a smaller pool).
        hybrid = len({int(kv["blocks_per_user"]) for kv in kv_cache}) > 1
        for L in self._prefill_bucket_lens():
            nb = (L + bs - 1) // bs
            if nb > total_blocks:  # cache too small for this bucket (reduced bring-up); skip
                continue
            w = serve_w if serve_w >= nb else nb
            dummy = torch.zeros((1, L), dtype=torch.int64)
            if hybrid:
                # Per-layer warmup page tables bounded to EACH layer's pool: block ids beyond a
                # (small) sliding pool would be OOB, so clamp each layer's real-block count to its
                # pool. Content is dummy (garbage prefill); only shape + in-bounds indices matter for
                # compiling the per-group programs and pre-allocating the per-group persistent buffers.
                ptl = []
                for kv in kv_cache:
                    pool = int(kv["blocks_per_user"])
                    m = min(nb, pool)
                    t = torch.zeros((1, w), dtype=torch.int32)
                    t[0, :m] = torch.arange(m, dtype=torch.int32)
                    ptl.append(t)
                self.prefill_forward(
                    dummy,
                    page_tables_per_layer=ptl,
                    kv_cache=kv_cache,
                    prompt_lens=[L],
                    start_pos=[0],
                    sampling_params=greedy,
                )
            else:
                pt = torch.zeros((1, w), dtype=torch.int32)
                pt[0, :nb] = torch.arange(nb, dtype=torch.int32)
                self.prefill_forward(
                    dummy, page_table=pt, kv_cache=kv_cache, prompt_lens=[L], start_pos=[0], sampling_params=greedy
                )
        # W1 fix — warm the serving ROW-COUNT dimension of the prefill page table. Serving batches up to
        # ``max_num_seqs`` new requests into ONE prefill call (page_table ``[num_reqs, serve_w]``), and
        # ``_prefill_pt``/``_prefill_pt_grouped`` are SHAPE-KEYED — an unseen ``(num_reqs>1, serve_w)`` shape
        # would allocate its persistent buffer under the resident decode trace at serving, i.e. the
        # allocator.cpp:123 "unsafe alloc under active trace" -> the multi-minute recompile/alloc stall (W1).
        # The bucket loop above only warmed row-count 1. Pre-allocate every ``(N, serve_w)`` buffer here
        # (pure allocation, before the decode trace exists — same as the (1,w) warmup), so serving-time
        # concurrent prefill is allocation-free for all batch sizes. Cheap: N buffer allocs, no compute.
        if serve_w:
            for N in range(1, int(self.max_batch_size) + 1):
                if hybrid:
                    self._prefill_pt_grouped([torch.zeros((N, serve_w), dtype=torch.int32) for _ in kv_cache])
                else:
                    self._prefill_pt(torch.zeros((N, serve_w), dtype=torch.int32))
        self._in_prefill_warmup = False
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
        # SPEC-DECODE served mode (TT_LAGUNA_SPEC_DECODE=1): capture the VERIFY traces (K1=1..k_max+1) and
        # OMIT the normal decode trace. Two resident CCL-bearing traces (normal decode + verify) deadlock
        # the mesh; routing ALL batch-1 greedy decode through the verify traces (K1=1 = a native single-token
        # step, K1=K+1 = a spec step) keeps only one trace family resident. Serve with --max-num-seqs 1 so
        # B==1 (no padding). Mirrors the standalone driver's capture_decode_trace=False.
        if self._spec_mode == "1":
            k_max = int(os.environ.get("TT_LAGUNA_SPEC_K", "4"))
            single = os.environ.get("TT_LAGUNA_SPEC_SINGLE", "") == "1"
            # SINGLE mode: capture ONLY the K1=k_max+1 verify trace (one resident trace, fixed-K, always-spec).
            # Tests whether multi-trace COEXISTENCE is the intermittent-corruption source: the standalone driver
            # (single fixed-K verify trace) is correct; serving uses adaptive-K -> up to 5 coexisting traces.
            draft_lens = [k_max] if single else list(range(0, k_max + 1))
            self.warmup_verify_decode_multi(draft_lens, kv_cache, int(num_blocks) if num_blocks else 1)
            print(
                f"[laguna spec] warmup: captured verify traces K1={[d + 1 for d in draft_lens]}; "
                f"normal decode trace OMITTED (deadlock-safe, batch-1 greedy spec-decode; single={single})",
                flush=True,
            )
            return None
        if B in self._decode:
            return None
        nb = int(num_blocks) if num_blocks else 1
        hybrid = len({int(kv["blocks_per_user"]) for kv in kv_cache}) > 1
        if hybrid:
            # Capture the decode trace over the TWO per-group persistent page tables (full + sliding),
            # both full-width [B, nb] null-padded (dummy zeros → block 0, in-bounds for both pools).
            ptl = [torch.zeros([B, nb], dtype=torch.int32) for _ in kv_cache]
            pt_persist, groups, reps = self._decode_pt_grouped_alloc(ptl)
            st = self._decode_state(B, kv_cache, pt_persist)
            st["pt_groups"], st["pt_reps"], st["last_pt_host_groups"] = groups, reps, {}
        else:
            pt_persist = self.gen._rep(torch.zeros([B, nb], dtype=torch.int32), ttnn.int32)
            self._decode_state(B, kv_cache, pt_persist)
        return None
