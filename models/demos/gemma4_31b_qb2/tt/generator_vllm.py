# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Translate vLLM scheduler updates into the Gemma4 TP4 generator's inputs."""

import gemma_history_reservation as _history_reservation
import os
import json
import time

import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.kv_precision import selected_kv_dtypes, tile_bytes
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, build_generator


# Experimental tight full-attention packing. vLLM gives the single full
# attention group 256-token scheduler blocks (two native 128-token pages) while
# sliding groups keep 128-token blocks; every group page is one wide backing
# row. A full block b owns the whole wide row b = native full rows 2b and 2b+1,
# so the second half of each wide row is no longer idle. Ownership remains one
# global scheduler ID per wide row, exactly as in the unpacked alias.
PACKED_POOL_PROFILES = {"c32a32_p2_f412672": 5336, "c32a32_p2_f443392": 5576, "c32a32_p2_f458752": 5696, "c32a32_p2_f626688": 7008}


def spec_ngram_enabled():
    """Opt-in speculative decoding with the host n-gram drafter (GEMMA_SPEC_NGRAM=1); default off."""
    value = os.environ.get("GEMMA_SPEC_NGRAM", "0")
    if value not in ("0", "1"):
        raise RuntimeError("GEMMA_SPEC_NGRAM must be 0 or 1")
    return value == "1"


def full_pack_enabled():
    value = os.environ.get("GEMMA_EXPERIMENT_FULL_PACK2", "0")
    if value not in ("0", "1"):
        raise RuntimeError("GEMMA_EXPERIMENT_FULL_PACK2 must be 0 or 1")
    profile = os.environ.get("GEMMA_SERIAL_PREPARED_POOL_PROFILE")
    if (value == "1") != (profile in PACKED_POOL_PROFILES):
        raise RuntimeError("Full packing and packed pool profiles must be selected together")
    if value == "1" and os.environ.get("GEMMA_EXPERIMENT_NATIVE_ALIAS") != "1":
        raise RuntimeError("Full packing requires the shared native backing alias")
    return value == "1"


class Gemma4ForCausalLM:
    _router = ttnn.FabricRouterConfig()
    _router.max_packet_payload_size_bytes = 8192
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_chunked_prefill": True,
        "max_device_top_k": 32,
        "fabric_config": {
            "config": ttnn.FabricConfig.FABRIC_1D_RING,
            "router_config": _router,
        },
    }
    if spec_ngram_enabled():
        # Verification returns, for every candidate row, a token sampled with that request's own sampling
        # parameters and per-position seed (not an argmax). With deterministic n-gram drafts, the runner's
        # id-equality walk is then exact speculative sampling, so sampled requests may speculate.
        model_capabilities = dict(
            model_capabilities,
            supports_spec_decode=True,
            spec_requirements=["device_propose"],
            supports_async_spec_decode=True,
            spec_sampled_verify=True,
            spec_token_history=True,
            spec_stateless_verify=True,
        )
    supports_explicit_admission_limit = True
    supports_prefill_history_finality = True
    decode_input_update_contract = 1
    MAX_PREFILL_TOKENS = 8192

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None):
        # vLLM inspects this keyword before the TT loader supplies a live mesh.
        if vllm_config is not None:
            raise RuntimeError("Initialize the TT model through initialize_vllm_model")
        self.generator = generator
        self.mesh = generator.mesh
        self.max_batch_size = max_batch_size
        self.admitted_request_limit = max_batch_size
        self.max_seq_len = max_seq_len
        self.entry = None
        self.cache = None
        self.prefill_slot_keys = {}
        self.next_prefill_key = 0
        self._spec_dirty = False
        self.spec_stats = dict(propose_s=0.0, proposals=0, drafted=0, verify_s=0.0, verifies=0, verify_rows=0)
        self._pool_memory_interval = int(os.environ.get("GEMMA_POOL_MEMORY_SAMPLE_INTERVAL", "1"))
        if not 1 <= self._pool_memory_interval <= 65536:
            raise ValueError("GEMMA_POOL_MEMORY_SAMPLE_INTERVAL must be in 1..65536")
        self._pool_memory_attempts = 0
        self._pool_memory_phase_attempts = {}

    def configure_admission_limit(self, limit):
        if type(limit) is not int or (self.max_batch_size, limit) not in ((32, 20), (32, 24)):
            raise ValueError("Experimental serving admission requires C32/A20 or A24")
        if self.cache is not None or self.entry is not None or self.prefill_slot_keys:
            raise RuntimeError("Admission must be configured before KV allocation or requests")
        self.admitted_request_limit = limit

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The TT generator owns the terminal norm and LM head")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("Gemma4 QB2 requires TP4, DP1, and batch 1..32")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        return cls(build_generator(mesh_device, max_seq_len=max_seq_len), max_batch_size, max_seq_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        if vllm_config.max_in_flight_tokens > 2 * cls.MAX_PREFILL_TOKENS:
            raise ValueError("Gemma4 QB2 supports at most 8192 batched tokens with async chunked prefill")
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != 128:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        # Equal accounting page sizes preserve the logical block size across
        # hybrid groups. Different native geometries still use separate pools.
        # The runner divides these global head counts by the TP degree.
        page_bytes = 2 * block * config.num_key_value_heads * config.head_dim * 2
        # Packed full pages hold 2*block tokens in the same accounting page:
        # 2*256*4*512*2 bytes equals the sliding 2*128*16*256*2 bytes, so vLLM
        # needs no padding and hybrid groups resolve to scheduler block 256.
        full_block = 2 * block if full_pack_enabled() else block
        result = {}
        for i, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            if not sliding and full_block != block and (
                    2 * full_block * config.num_global_key_value_heads * config.global_head_dim * 2 != page_bytes):
                raise ValueError("Packed full page must equal the sliding accounting page")
            common = dict(
                block_size=block if sliding else full_block,
                num_kv_heads=config.num_key_value_heads if sliding else config.num_global_key_value_heads,
                head_size=config.head_dim if sliding else config.global_head_dim,
                dtype=torch.bfloat16,
                page_size_padded=page_bytes,
            )
            result[f"model.layers.{i}.self_attn"] = (
                SlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**common)
            )
        return result

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, **kwargs):
        # The shared pool uses one full-attention group and five sliding groups.
        # Each window can straddle nine pages. Reserve both async batches of
        # in-flight prefill tokens globally, in addition to the per-request tails.
        # The plugin separately adds one output page per request.
        admitted = kwargs.get("admitted_request_limit", max_num_seqs)
        if type(admitted) is not int or not 1 <= admitted <= max_num_seqs:
            raise ValueError("Invalid shared admission for pool sizing")
        if admitted != max_num_seqs and (max_num_seqs, admitted) not in ((32, 20), (32, 24)):
            raise ValueError("Only C32/A20 or A24 supports a reduced tail reserve")
        tails = Decoder.SLIDING_WINDOW_PAGES * Decoder.PAGE_SIZE * admitted
        pool_tokens = int(os.environ.get("GEMMA_EXPERIMENT_KV_POOL_TOKENS", "262144"))
        if pool_tokens < 262144 or pool_tokens % Decoder.PAGE_SIZE:
            raise ValueError("Experimental pool must preserve context and page alignment")
        return pool_tokens + 5 * (tails + 2 * cls.MAX_PREFILL_TOKENS)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        history_capacity = _history_reservation.serving_history_capacity(
            batch=self.max_batch_size, admitted=self.admitted_request_limit, capacity=self.max_seq_len)
        if os.environ.get("GEMMA_SERIAL_PREPARED_POOL_PROFILE") == "c32a24_f458752":
            if len(per_layer_specs) != 60 or any(type(shape[0]) is not int or shape[0] != 5336 for shape, dtype, pool_id in per_layer_specs):
                raise RuntimeError("A24/F458752 requires exactly60 allocator specs with n5336 before TT access")
        packed = full_pack_enabled()
        if packed:
            expected_n = PACKED_POOL_PROFILES[os.environ["GEMMA_SERIAL_PREPARED_POOL_PROFILE"]]
            if len(per_layer_specs) != 60 or any(type(shape[0]) is not int or shape[0] != expected_n for shape, dtype, pool_id in per_layer_specs):
                raise RuntimeError(f"Packed profile requires exactly60 allocator specs with n{expected_n} before TT access")
            if self.admitted_request_limit != self.max_batch_size:
                raise RuntimeError("Packed profile uses native physical admission")
        self._kv_full_pack = packed
        before = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.DRAM)
        print("KV_POOL_MEMORY " + json.dumps({"phase":"before","banks":before.num_banks,"allocated_per_bank":before.total_bytes_allocated_per_bank,"free_per_bank":before.total_bytes_free_per_bank}), flush=True)
        if os.environ.get("GEMMA_EXPERIMENT_NATIVE_ALIAS") != "1":
            raise RuntimeError("Experimental backing reinterpretation requires explicit opt-in")
        if len(per_layer_specs) != len(self.generator.model.layers):
            raise ValueError("The allocator must supply exactly one spec per layer")
        grouped = {}
        for i, (layer, (shape, dtype, pool_id)) in enumerate(zip(self.generator.model.layers, per_layer_specs)):
            tokens = 256 if packed and not layer.sliding else 128
            if tuple(shape[1:]) != (layer.kv_heads, tokens, layer.head_dim) or dtype != torch.bfloat16:
                raise ValueError(f"Unexpected allocator geometry or accounting dtype: {shape}, {dtype}")
            grouped.setdefault(pool_id, []).append((i, layer, int(shape[0])))
        pools, full_views, native_factors, scheduler_bounds = {}, {}, {}, {}
        for pool_id, members in grouped.items():
            sizes = {n for _, _, n in members}
            if len(sizes) != 1 or len(members) != 6 or sum(not layer.sliding for _, layer, _ in members) != 1:
                raise ValueError("Alias requires one full and five sliding layers with identical global block bounds")
            n = sizes.pop()
            if n <= 1:
                raise ValueError("Alias pool must contain null and live block IDs")
            wide_layer = next(layer for _, layer, _ in members if layer.sliding)
            if any(layer.kv_dtypes != wide_layer.kv_dtypes for _, layer, _ in members):
                raise ValueError("Aliased layers must share the exact selected KV dtype pair")
            pair = wide_layer.allocate_cache(physical_pages=n + self.max_batch_size)
            pools[pool_id] = pair
            full_views[pool_id] = tuple(self._reinterpret_kv_backing(t) for t in pair)
            if pair[0].buffer_unique_id() == pair[1].buffer_unique_id():
                raise ValueError("K and V root allocations must be distinct")
            for i, layer, _ in members:
                native_factors[i] = 1 if layer.sliding else 2
                scheduler_bounds[i] = n
        # Explicit roots outlive every native view and every trace using cache.
        self._kv_alias_pool_members = {pool: [i for i, _, _ in members] for pool, members in grouped.items()}
        print("KV_FULL_PACK " + json.dumps({"enabled":packed,"full_scheduler_block_tokens":256 if packed else 128,
            "native_full_rows_per_scheduler_id":2 if packed else 1,"scheduler_ids":sorted({n for _,_,n in sum(grouped.values(), [])})}), flush=True)
        print("KV_ALIAS_ROOTS " + json.dumps({"pools":len(pools), "roots":[{"pool":str(pool),"root_id":root.buffer_unique_id(),"view_id":full_views[pool][index].buffer_unique_id(),"address":root.buffer_address(),"wide_rows":int(root.shape[0]),"dtype":str(root.dtype),"tile_bytes":tile_bytes(root.dtype),"logical_backing_bytes":int(root.shape[0])*128*tile_bytes(root.dtype)} for pool,pair in pools.items() for index,root in enumerate(pair)]}), flush=True)
        self._kv_alias_roots = pools
        self._kv_alias_factors = native_factors
        self._kv_alias_bounds = scheduler_bounds
        kv, scratch = [], {}
        for i, (layer, (shape, dtype, pool_id)) in enumerate(zip(self.generator.model.layers, per_layer_specs)):
            factor = native_factors[i]
            kv.append(pools[pool_id] if layer.sliding else full_views[pool_id])
            scratch[i] = [factor * (int(shape[0]) + j) for j in range(self.max_batch_size)]
        tables = {
            i: torch.zeros(self.max_batch_size, self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))
        }
        self.cache = CacheState(
            kv, tables, self.max_batch_size, self.max_seq_len, scratch_pages=scratch, vllm_owned=True
        )
        self.cache.serving_admission_limit = self.admitted_request_limit
        after = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.DRAM)
        print("KV_POOL_MEMORY " + json.dumps({"phase":"after","banks":after.num_banks,"allocated_per_bank":after.total_bytes_allocated_per_bank,"free_per_bank":after.total_bytes_free_per_bank,"unique_pools":len(pools),"pool_full_tokens":int(os.environ.get("GEMMA_EXPERIMENT_KV_POOL_TOKENS", "262144"))}), flush=True)
        self.generator.setup_history_bank(self.cache, capacity=history_capacity)
        if self.cache.history_bank is not None:
            if len(self.cache.history_bank.roots) != history_capacity:
                raise RuntimeError("Allocated history count differs from selected capacity")
            print("GEMMA_HISTORY_RESERVATION_CAPACITY " + json.dumps(dict(
                batch=self.max_batch_size, admitted=self.admitted_request_limit, selected=history_capacity,
                roots=len(self.cache.history_bank.roots))), flush=True)
        self.generator.prepare_serving_prefill(self.cache)
        return self.cache

    @staticmethod
    def _reinterpret_kv_backing(root):
        # Experimental pinned-runtime raw tile alias. This does NOT preserve
        # tensor values under ordinary reshape semantics. Full scheduler IDs
        # must be doubled, and global group ownership must remain exclusive.
        shape = tuple(int(x) for x in root.shape)
        if (len(shape) != 4 or shape[1:] != (4, 128, 256)
                or root.dtype not in selected_kv_dtypes() or root.layout != ttnn.TILE_LAYOUT
                or root.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or root.buffer_aligned_page_size() != tile_bytes(root.dtype)):
            raise ValueError("Unsupported root backing geometry")
        view = ttnn.experimental.view(root, [2 * shape[0], 1, 128, 512])
        if (view.buffer_unique_id() != root.buffer_unique_id()
                or view.buffer_address() != root.buffer_address()
                or view.buffer_aligned_page_size() != root.buffer_aligned_page_size()
                or tuple(view.padded_shape) != (2 * shape[0], 1, 128, 512)):
            raise RuntimeError("Pinned runtime did not produce the required zero-copy tile alias")
        return view

    def _audit_alias_consumers(self, sources, ranges):
        # Diagnostic of actual scheduler-provided pages consumed by this update.
        # It complements global allocator source proof; it is not an independent
        # snapshot of every idle hash or every previously submitted device op.
        began = time.perf_counter()
        if ranges is None:
            raise ValueError("Alias diagnostic requires current active token ranges")
        maximum_full = maximum_sliding = 0
        for members in self._kv_alias_pool_members.values():
            full_ids, sliding_ids = set(), set()
            for i in members:
                full = self._kv_alias_factors[i] == 2
                target = full_ids if full else sliding_ids
                source = sources[i]
                for row, limits in enumerate(ranges):
                    if limits is None:
                        continue
                    sliding_begin, end = limits
                    granule = 256 if full and self._kv_full_pack else 128
                    begin_page = 0 if full else sliding_begin // 128
                    end_page = (end + granule - 1) // granule
                    target.update(int(v) for v in source[row, begin_page:end_page].tolist() if int(v) > 0)
            overlap = full_ids & sliding_ids
            if overlap:
                raise RuntimeError(f"Full/sliding live consumer block collision: {sorted(overlap)[:16]}")
            maximum_full = max(maximum_full, len(full_ids))
            maximum_sliding = max(maximum_sliding, len(sliding_ids))
        self._alias_audit_count = getattr(self, "_alias_audit_count", 0) + 1
        self._alias_audit_seconds = getattr(self, "_alias_audit_seconds", 0.0) + time.perf_counter() - began
        if self._alias_audit_count <= 3 or self._alias_audit_count % 64 == 0:
            print("KV_ALIAS_CONSUMER_AUDIT " + json.dumps({"calls":self._alias_audit_count,
                "full_live_ids":maximum_full,"sliding_live_ids":maximum_sliding,
                "cross_geometry_overlap":0,"host_seconds":self._alias_audit_seconds}), flush=True)

    def _allocation_probe_begin(self):
        limit = int(os.environ.get("GEMMA_ALLOCATION_EVENT_PROBES", "0"))
        if not limit or getattr(self, "_allocation_probe_count", 0) >= limit:
            return
        histories = getattr(self.cache, "prefill_history", {})
        required = int(os.environ.get("GEMMA_PEAK_MIN_HISTORIES", str(self.max_batch_size - 1)))
        if len(histories) < required:
            return
        view = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.DRAM)
        self._allocation_probe_active = {"before_bytes":view.total_bytes_allocated_per_bank * view.num_banks,
            "banks":view.num_banks,"before_block_table":view.block_table,
            "history_count":len(histories),"started":time.perf_counter()}
        ttnn._ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    def _allocation_probe_end(self):
        state = getattr(self, "_allocation_probe_active", None)
        if state is None:
            return
        self._allocation_probe_active = None
        graph = ttnn._ttnn.graph.end_graph_capture()
        if isinstance(graph, str):
            graph = json.loads(graph)
        events = [e for e in graph if e.get("node_type") in ("buffer_allocate", "buffer_deallocate")]
        view = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.DRAM)
        after = view.total_bytes_allocated_per_bank * view.num_banks
        delta, peak = {}, {}
        for event in events:
            p = event["params"]
            if int(p["buffer_type"]) != 0:  # Exclude TRACE (4) from ordinary DRAM.
                continue
            device = str(p["device_id"])
            size = int(p["max_size_per_bank"]) * state["banks"]
            sign = 1 if event["node_type"] == "buffer_allocate" else -1
            delta[device] = delta.get(device, 0) + sign * size
            peak[device] = max(peak.get(device, 0), delta[device])
        self._allocation_probe_count = getattr(self, "_allocation_probe_count", 0) + 1
        probe = self._allocation_probe_count
        result = {"probe":probe,"before_bytes":state["before_bytes"],"after_bytes":after,
            "history_count":state["history_count"],"delta_by_virtual_device":delta,
            "peak_extra_by_virtual_device":peak,"event_count":len(events),
            "boundary_reconciles":bool(delta) and all(v == after-state["before_bytes"] for v in delta.values()),
            "scope":"one adapter prefill including sampling; common mesh allocator, not independent physical-chip samples",
            "instrumented_host_seconds":time.perf_counter()-state["started"]}
        # Preserve raw allocation events and the allocator endpoints for an
        # independent reconstruction. Chunked lines avoid log line truncation.
        import base64, hashlib, zlib
        raw = json.dumps({"summary":result,"before_block_table":state["before_block_table"],
            "after_block_table":view.block_table,"events":events},separators=(",",":")).encode()
        encoded = base64.b64encode(zlib.compress(raw)).decode()
        chunks = [encoded[i:i+12000] for i in range(0,len(encoded),12000)]
        result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw),chunks=len(chunks))
        print("KV_ALLOCATION_EVENT_PEAK " + json.dumps(result),flush=True)
        for index, chunk in enumerate(chunks):
            print("KV_ALLOCATION_EVENT_DATA " + json.dumps({"probe":probe,"index":index,"count":len(chunks),"base64_zlib":chunk}),flush=True)

    def _sample_pool_memory(self, phase):
        # Sampled host allocator boundaries: never a device/lifetime peak.
        self._pool_memory_attempts += 1
        attempt = self._pool_memory_phase_attempts.get(phase, 0) + 1
        self._pool_memory_phase_attempts[phase] = attempt
        # Preserve the first boundary of each kind and every active explicit
        # allocation-event probe. Trace allocation tracking is independent.
        force = getattr(self, "_allocation_probe_active", None) is not None
        if not force and (attempt - 1) % self._pool_memory_interval:
            return
        started = time.perf_counter()
        view = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.DRAM)
        allocated = view.total_bytes_allocated_per_bank
        previous_high = getattr(self, "_pool_observed_high", 0)
        high = max(previous_high, allocated)
        self._pool_observed_high = high
        self._pool_memory_samples = getattr(self, "_pool_memory_samples", 0) + 1
        self._pool_memory_host_s = getattr(self, "_pool_memory_host_s", 0.0) + time.perf_counter() - started
        if allocated > previous_high or self._pool_memory_samples % 64 == 0:
            trace_view = ttnn.get_memory_view(self.generator.mesh, ttnn.BufferType.TRACE)
            print("KV_POOL_MEMORY_BOUNDARY " + json.dumps({"phase": phase, "banks": view.num_banks,
                "allocated_per_bank": allocated, "free_per_bank": view.total_bytes_free_per_bank,
                "trace_allocated_per_bank": trace_view.total_bytes_allocated_per_bank, "trace_banks": trace_view.num_banks,
                "trace_free_per_bank": trace_view.total_bytes_free_per_bank,
                "largest_contiguous_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
                "observed_high_per_bank": high, "samples": self._pool_memory_samples,
                "query_host_s_total": self._pool_memory_host_s,
                "boundary_attempts": self._pool_memory_attempts,
                "sample_interval_per_phase": self._pool_memory_interval,
                "scope": "sampled host boundaries, not device or lifetime peak"}), flush=True)

    def _tables(self, cache, page_tables_per_layer, *, slots=None, audit_ranges=None):
        if cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(cache.kv):
            raise ValueError("Hybrid attention requires a page table for every layer")
        if os.environ.get("GEMMA_EXPERIMENT_NATIVE_ALIAS_AUDIT") == "1":
            self._audit_alias_consumers(page_tables_per_layer, audit_ranges)
        result, translated = {}, {}
        for i, source in enumerate(page_tables_per_layer):
            # Share immutable host rows only within this update: vLLM mutates
            # its source tensors in place between scheduler steps.
            factor = self._kv_alias_factors[i]
            bound = self._kv_alias_bounds[i]
            key = (id(source), factor, bound)
            if key not in translated:
                dest = torch.zeros_like(cache.page_tables[i])
                rows, cols = source.shape
                if rows > cache.batch or cols > dest.shape[1]:
                    raise ValueError("Scheduler page table exceeds the configured cache")
                indices = list(range(rows)) if slots is None else slots
                if len(indices) > rows:
                    raise ValueError("Page table has fewer rows than prefill requests")
                # Prefill rows are compact, then padded to the decode batch.
                # Scatter only the live prefix into its assigned state slots.
                live = source[: len(indices)].to(torch.int32).clamp_min(0)
                if live.numel() and int(live.max()) >= bound:
                    raise ValueError("Scheduler ID enters private warmup storage")
                if factor == 1:
                    dest[indices, :cols] = live
                elif not self._kv_full_pack:
                    dest[indices, :cols] = live * 2
                else:
                    # Scheduler block b (256 tokens) -> native pages 2b, 2b+1
                    # (128 tokens each) of the [2P,1,128,512] full view.
                    width = min(cols, (dest.shape[1] + 1) // 2)
                    if cols > width and live.numel() and bool((live[:, width:] > 0).any()):
                        raise ValueError("Packed full table has live blocks beyond the context")
                    packed_ids = live[:, :width]
                    expanded = torch.stack((packed_ids * 2, packed_ids * 2 + 1), dim=2).reshape(len(indices), 2 * width)
                    # The null block (0) stays native null page 0 in both halves.
                    expanded[expanded == 1] = 0
                    span = min(2 * width, dest.shape[1])
                    dest[indices, :span] = expanded[:, :span]
                translated[key] = dest
            result[i] = translated[key]
        return result

    def _sampling(self, params, batch, positions, *, reset_seed=True):
        def rows(name):
            value = getattr(params, name)
            value = value.tolist() if isinstance(value, torch.Tensor) else value
            return list(value)[:batch] if isinstance(value, (list, tuple)) else [value] * batch

        for name, default in (
            ("presence_penalty", 0),
            ("frequency_penalty", 0),
            ("repetition_penalty", 1),
            ("enable_log_probs", False),
        ):
            if any(x != default for x in rows(name)):
                raise ValueError(f"The shared host sampler must handle {name}")
        seeds = [0] * batch
        if reset_seed:
            modulus = 2**31 - self.generator.max_seq_len
            seeds = [
                int(torch.randint(0, modulus, ()).item()) if seed is None else int(seed) % modulus
                for seed in rows("seed")
            ]
            seeds = [seed + max(0, int(pos)) for seed, pos in zip(seeds, positions)]
        return dict(
            top_k=rows("top_k"), top_p=rows("top_p"), temperature=rows("temperature"), seed=seeds, reset_seed=reset_seed
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        sampling_params=None,
        page_tables_per_layer=None,
        start_pos=None,
        empty_slots=None,
        retain_prefill_history=None,
        **kwargs,
    ):
        starts = [0] * len(prompt_lens) if start_pos is None else [int(x) for x in start_pos]
        lengths = [int(x) for x in prompt_lens]
        slots = list(range(len(lengths))) if empty_slots is None else list(empty_slots)
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots, audit_ranges=[(max(0, (start // 128) * 128 - 1024), end) for start, end in zip(starts, lengths)])
        self.entry = None
        # Prefix pages can be shared by several live requests. History belongs
        # to a request's stable state slot, never to a shared physical page.
        keys = []
        for slot in slots:
            if slot not in self.prefill_slot_keys:
                self.prefill_slot_keys[slot] = self.next_prefill_key
                self.next_prefill_key += 1
            keys.append(self.prefill_slot_keys[slot])
        reservation = _history_reservation.context()
        if reservation is not None and reservation.scheduler is not None:
            reservation.prepare_adapter(self, keys, slots, starts, lengths, retain_prefill_history)
        self._allocation_probe_begin()
        try:
            self._sample_pool_memory("before_prefill")
            outputs = self.generator.prefill_forward(
                tokens,
                page_table=tables,
                kv_cache=kv_cache,
                prompt_lens=lengths,
                slots=slots,
                return_device=sampling_params is not None,
                start_positions=starts,
                history_keys=keys,
                retain_prefill_history=retain_prefill_history,
            )
            self._sample_pool_memory("after_prefill")
            if sampling_params is None:
                return outputs
            params = self._sampling(sampling_params, len(lengths), [n - 1 for n in lengths])
            return self.generator.sample_prefill_outputs(outputs, kv_cache, params)
        finally:
            self._allocation_probe_end()

    def set_kv_layer_groups(self, layer_to_group):
        """vLLM KV-cache group of every layer (plugin, before allocation). Layers of one group share a block table,
        so the verify trace keeps one row table per group."""
        groups = {}
        for layer, group in enumerate(layer_to_group):
            groups.setdefault(int(group), []).append(layer)
        self._spec_groups = [groups[g] for g in sorted(groups)]

    def _check_spec_groups(self):
        groups = getattr(self, "_spec_groups", None)
        if not groups or sorted(i for g in groups for i in g) != list(range(len(self.generator.model.layers))):
            raise RuntimeError("Speculative verify needs the KV-cache layer groups from the plugin")
        for g in groups:
            kinds = {self.generator.model.layers[i].kind for i in g}
            factors = {self._kv_alias_factors[i] for i in g}
            bounds = {self._kv_alias_bounds[i] for i in g}
            if len(kinds) != 1 or len(factors) != 1 or len(bounds) != 1:
                raise RuntimeError("A KV-cache group mixes layer kinds or table translations")
        return groups

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, k):
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        from models.demos.gemma4_31b_qb2.tt.generator import Gemma4Generator

        if not spec_ngram_enabled():
            return SpecReject("GEMMA_SPEC_NGRAM=1 is required for speculative decoding")
        if max_num_seqs > 32:
            return SpecReject("the verify step holds 32 token rows")
        limit = Gemma4Generator.SPEC_MAX_CANDIDATES - 1
        return SpecPlan(
            effective_k=max(1, min(int(k), limit)),
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=True,
        )

    def propose_draft_tokens(self, num_drafts, committed, positions, accepted_counts, hidden=None,
                             token_history=None):
        """Host n-gram drafts from each request's full token history, cost-aware lengths (spec_ngram.py)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        from models.demos.gemma4_31b_qb2.tt import spec_ngram

        rows = int(committed.shape[0])
        histories = [None] * rows
        if token_history is not None:
            for row in range(rows):
                if int(positions[row, 0]) < 0 or int(accepted_counts[row]) < 1:
                    continue
                histories[row] = token_history[row]
        t = time.perf_counter()
        drafts, valid = spec_ngram.propose(histories, int(num_drafts))
        self.spec_stats["propose_s"] += time.perf_counter() - t
        self.spec_stats["proposals"] += 1
        self.spec_stats["drafted"] += int(sum(valid))
        return DraftOutput(draft_token_ids=torch.tensor(drafts, dtype=torch.int32).reshape(rows, int(num_drafts)),
                           num_valid=torch.tensor(valid, dtype=torch.int32))

    def _verify(self, tokens, start_pos, kv_cache, page_tables_per_layer, num_valid_drafts, sampling_params,
                slot_remap):
        """One verify step: the [B, 1+K] candidate block becomes 32 token rows (see generator.spec_decode)."""
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, PLACEHOLDER_TOKEN_ID, VerifyOutput

        t = time.perf_counter()
        batch, width = tokens.shape
        first = start_pos[:, 0]
        tables = self._tables(kv_cache, page_tables_per_layer, audit_ranges=[
            (max(0, int(pos) - 1023), int(pos) + width) if int(pos) >= 0 else None for pos in first])
        for row, position in enumerate(first):
            if int(position) >= 0:
                self.release_request(row if slot_remap is None else int(slot_remap[row]))
        if sampling_params is None:
            raise ValueError("Speculative verify samples on device; host sampling is not supported here")
        # Per-request seeds follow the plain path: seed + position of the input token (_sampling).
        params = self._sampling(sampling_params, self.max_batch_size, first.tolist(), reset_seed=True)
        row_tokens = torch.zeros(32, dtype=torch.int64)
        row_pos = torch.full((32,), -1, dtype=torch.int32)
        row_slot = torch.zeros(32, dtype=torch.int32)
        row_cand = torch.zeros(32, dtype=torch.int32)
        row_params = {name: [params[name][0]] * 32 for name in ("top_k", "top_p", "temperature")}
        row_params["seed"] = [0] * 32
        where = {}
        r = 0
        for b in range(batch):
            if int(first[b]) < 0:
                continue
            n = 1 + int(num_valid_drafts[b])
            n = min(n, 32 - r)
            for j in range(n):
                row_tokens[r] = int(tokens[b, j])
                row_pos[r] = int(start_pos[b, j])
                row_slot[r] = b
                row_cand[r] = j
                for name in ("top_k", "top_p", "temperature"):
                    row_params[name][r] = params[name][b]
                row_params["seed"][r] = (int(params["seed"][b]) + j) % (2**31)
                where[(b, j)] = r
                r += 1
            if r >= 32:
                break
        groups = self._check_spec_groups()
        sampled = self.generator.spec_decode(kv_cache, tokens=row_tokens, positions=row_pos, slots=row_slot,
                                             candidates=row_cand, host_tables=tables, groups=groups,
                                             sampling_params=row_params)
        ids = torch.full((batch, width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for (b, j), row in where.items():
            ids[b, j] = int(sampled[row])
        # The plain trace's resident inputs and sampler state are stale after a verify.
        self._spec_dirty = True
        self.spec_stats["verify_s"] += time.perf_counter() - t
        self.spec_stats["verifies"] += 1
        self.spec_stats["verify_rows"] += r
        return VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        slot_remap=None,
        **kwargs,
    ):
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid_drafts = kwargs.pop("num_valid_drafts", None)
        kwargs.pop("accepted_counts", None)
        if spec_mode is not None:
            if kv_cache is not self.cache:
                raise ValueError("Serving must use the cache allocated by vLLM")
            return self._verify(tokens, start_pos, kv_cache, page_tables_per_layer, num_valid_drafts,
                                sampling_params, slot_remap)
        if self._spec_dirty:
            # A verify replaced the sampler's per-row parameters/seeds and advanced every request past the plain
            # trace's resident tokens and positions: reload all of them.
            reload_inputs = True
            reset_sampling_state = True
            reload_sampling_params = True
            self._spec_dirty = False
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if not enable_trace:
            raise ValueError("Gemma4 QB2 requires traced decode")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        sample = sampling_params is not None
        if reload_inputs:
            tables = self._tables(kv_cache, page_tables_per_layer, audit_ranges=[(max(0, int(pos) - 1023), int(pos) + 1) if int(pos) >= 0 else None for pos in start_pos])
            # A row entering decode no longer needs its private prefill tail.
            # The plugin notifies slot moves after submission succeeds, so
            # histories still use source slots when a decode row is gathered.
            for row, position in enumerate(start_pos):
                if int(position) >= 0:
                    slot = row if slot_remap is None else int(slot_remap[row])
                    self.release_request(slot)
            self.entry = self.generator.prepare_decode(tokens, start_pos, page_table=tables, kv_cache=kv_cache)
            self._sample_pool_memory("after_prepare_decode")
        elif self.entry is None or self.generator.traces.get(id(kv_cache)) is not self.entry:
            raise ValueError("A retired decode trace requires current scheduler inputs")
        elif reload_page_table:
            tables = self._tables(kv_cache, page_tables_per_layer, audit_ranges=[(max(0, int(pos) - 1023), int(pos) + 1) if int(pos) >= 0 else None for pos in start_pos])
            self.generator._tables(kv_cache, tables)
        if sample and (reload_sampling_params or reset_sampling_state):
            params = self._sampling(sampling_params, self.max_batch_size, start_pos, reset_seed=reset_sampling_state)
            self.generator.configure_sampling(batch=self.max_batch_size, **params)
        self.generator.replay(self.entry, sample=sample)
        if not sample:
            return self.generator._host_logits(self.entry["logits"])[0, 0, : self.max_batch_size].unsqueeze(1)
        output = self.entry["tokens"]
        return self.process_decode_output_host(output, is_tokens=True) if read_from_device else output

    def read_decode_output(self, output, async_read=False):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        # Tokens are replicated. Read one rank instead of gathering the vocabulary.
        host = ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=True):
        if not is_tokens:
            return output.reshape(self.max_batch_size, 1, -1)
        if output.storage_type() != ttnn.StorageType.HOST:
            output = self.read_decode_output(output)
        return ttnn.to_torch(output).reshape(-1)[: self.max_batch_size].long().reshape(-1, 1)

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        # Each unseen logical prefill shape retires traces before compilation.
        pass

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        if spec_ngram_enabled():
            # The generator captures the verify trace together with every plain decode trace capture.
            self.generator.spec_groups = self._check_spec_groups()
        self.generator._tables(kv_cache, kv_cache.page_tables)
        self.generator.configure_sampling(batch=self.max_batch_size, temperature=0)
        self.entry = self.generator._entry(kv_cache)

    def note_state_slots_moved(self, moves):
        self.prefill_slot_keys = {moves.get(slot, slot): key for slot, key in self.prefill_slot_keys.items()}

    def release_request(self, slot):
        key = self.prefill_slot_keys.pop(slot, None)
        if key is not None and self.cache is not None:
            self.cache.release_history(key)

    def release_persistent_capture(self):
        if self.cache is not None:
            self.cache.clear_histories()
        self.prefill_slot_keys.clear()
        self.generator.close()
        self.entry = None
