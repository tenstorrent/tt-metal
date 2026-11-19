# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit request state and split-sampling driver for the TP4 text model."""

# Readiness loads this file with spec_from_file_location under a synthetic
# top-level name. Preserve the same package-relative dependencies in that path.
if not __package__:
    __package__ = "models.autoports.qwen_qwen3_8_flash_next.tt"

import os
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from readiness_check.contract import Generator
from transformers import AutoTokenizer

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

from .cache_budget import KNOWN_STATIC_BYTES, plan_shared_kv_cache
from .cache_pool import SharedKVPool, normalize_specs
from .chunk_output import write_token_chunk
from .device_seed import DeviceSeedState
from .factored_decode import FactoredDecode
from .host_sampling import HostSampler, HostSamplingParams
from .model import QwenModel
from .ngram_hash import EOS
from .pending_prefill import plan_pending_prefill
from .precision import load_policy, static_weight_delta
from .prefill_checkpoint import PrefillCheckpoints
from .slot_state import SlotStateMover, remap_slot_mapping, remap_token_buffer, validate_remap
from .token_feedback import DeviceTokenFeedback

DRAM = ttnn.DRAM_MEMORY_CONFIG
PREFILL_BUCKETS = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class DecodeSubmission:
    tensor: object
    slots: tuple
    active_rows: tuple
    is_tokens: bool


class QwenGenerator(Generator):
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": False,  # recurrent_state excludes KV-only reuse.
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "max_device_top_k": 40,
        "supports_async_decode": False,
        "output_tokens_per_step": 1,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
    }

    @staticmethod
    def get_max_tokens_all_users(
        model_name=None, num_devices=4, tt_data_parallel=1, max_model_len=262144, max_num_seqs=1
    ):
        """Pre-worker-padding pool budget; conservative arithmetic, not hardware acceptance.

        The pinned worker adds one output page per slot. The pure planning helper
        reserves that addition, full non-KV slot/scratch state and a null page;
        it rejects infeasible plans rather than shrinking the advertised context.
        """
        if num_devices != 4 or tt_data_parallel != 1:
            raise ValueError("The accepted cache/memory plan requires TP4 on one 1x4 mesh")
        policy = load_policy(Path(__file__).resolve().parents[1])
        return plan_shared_kv_cache(
            max_model_len,
            max_num_seqs,
            known_static_bytes=KNOWN_STATIC_BYTES + static_weight_delta(policy),
            kv_cache_dtype=policy["kv_cache_dtype"],
        ).plugin_max_tokens_all_users()

    def __init__(self, model, tokenizer, *, max_seq_len=None, max_batch_size=1, kv_cache=None, defer_kv_cache=False):
        self.model, self.mesh, self.tokenizer = model, model.mesh, tokenizer
        self.event_sink = None
        self.capacity = max_seq_len or model.config.max_position_embeddings
        if not 1 <= max_batch_size <= 32:
            raise ValueError("The fixed-slot contract supports 1..32 slots")
        self.max_batch_size = max_batch_size
        self.shared_pool = None
        self.cache_setup_error = None
        if defer_kv_cache and kv_cache is not None:
            raise ValueError("Deferred allocation cannot replace caller-owned cache")
        self.cache_binding_pending = bool(defer_kv_cache)
        self.owns_cache_storage = kv_cache is None
        self.caches = (
            list(kv_cache)
            if kv_cache is not None
            else [
                model.create_cache(capacity=self.capacity, defer_kv_cache=defer_kv_cache) for _ in range(max_batch_size)
            ]
        )
        if len(self.caches) != max_batch_size or any(c.capacity < self.capacity for c in self.caches):
            raise ValueError("Caller-owned cache must cover every fixed slot and the configured context")
        for cache in self.caches:
            for state in cache.layers:
                if state.attention is not None:
                    for tensor in (state.attention.k_cache, state.attention.v_cache):
                        if tensor is not None and tensor.dtype != getattr(ttnn, model.precision["kv_cache_dtype"]):
                            raise ValueError("Caller-owned KV cache dtype must match the constructed precision policy")
        scratch = (
            model.create_cache(capacity=self.capacity, defer_kv_cache=defer_kv_cache) if max_batch_size > 1 else None
        )
        self.slot_mover = SimpleNamespace(scratch=scratch) if defer_kv_cache else SlotStateMover(self.caches, scratch)
        self.prefill_checkpoints = PrefillCheckpoints(self.caches, self.mesh)
        self.decode_tokens = self._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32))
        self.saved_tokens = self._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32))
        self.word_suffix = self._upload(torch.tensor([1] + [0] * 14, dtype=torch.int32).reshape(1, 1, 15))
        self.prefill_inputs = {n: self._upload(torch.zeros(1, n, dtype=torch.int32)) for n in PREFILL_BUCKETS}
        self.prefill_words = {n: self._upload(torch.zeros(1, n, 16, dtype=torch.int32)) for n in PREFILL_BUCKETS}
        self.sample_logits = ttnn.full(
            (1, 1, 32, model.config.vocab_size // 4),
            -100.0,
            device=self.mesh,
            dtype=getattr(ttnn, model.precision["sampling_dtype"]),
            layout=ttnn.TILE_LAYOUT,
        )
        # Dedicated persistent output for all six canonical prefill traces.
        # It must not alias the sampler canvas, which accumulates completed rows.
        self.prefill_output = ttnn.empty(
            (1, 1, 32, model.config.vocab_size // 4),
            device=self.mesh,
            dtype=getattr(ttnn, model.precision["logits_dtype"]),
            layout=ttnn.TILE_LAYOUT,
        )
        self.prefill_traces = {}
        # One stable prefill binding across scheduler slots. The existing
        # migration scratch owns private recurrent/index state; shared KV stays
        # in the caller's pool and is selected by its copied page table.
        self.prefill_cache = self.caches[0] if max_batch_size == 1 else scratch
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            max_batch_size=32,
            max_top_k=64,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 1, "topology": ttnn.Topology.Ring}
            },
        )
        self.sampler = SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=model.ccl)
        # Pinned serving stack routes penalties to its explicit host sampler.
        # The common count updater otherwise mutates every physical row, including inactive slots.
        self.sampler.tt_sampling._allow_penalties_sampling = False
        self.seed_state = DeviceSeedState(self.mesh, self.sampler.tt_sampling.seeds_tt_tensor)
        self.token_feedback = DeviceTokenFeedback(self.decode_tokens, self.saved_tokens, self.seed_state.active)
        self.request_seeds = [None] * 32
        self.seed_skip = self._upload(torch.full((32,), 2**32 - 1, dtype=torch.int64))
        self.traces = {}
        self.sampling_traces = {}
        self.prepared = False
        self.page_tables = {}
        self.pending_prefill = {}
        self.counters = dict(
            prefill_replays=0,
            model_replays=0,
            sampling_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            readbacks=0,
            synchronizations=0,
            cache_resets=0,
            captures=0,
            releases=0,
        )
        self.set_sampling_params()
        # All persistent decode workspaces precede every capture. Cache binding
        # remains setup-only; the owner reads the final bound caches in prepare.
        self.factored_decode = FactoredDecode(self) if max_batch_size > 1 else None

    def _check_cache_setup(self):
        if getattr(self, "cache_setup_error", None) is not None:
            raise RuntimeError(f"Generator cache setup failed; reconstruct the generator: {self.cache_setup_error}")
        if getattr(self, "cache_binding_pending", False):
            raise RuntimeError("Deferred KV storage must be allocated and bound before preparation or execution")

    def _resolve_caches(self, kv_cache, slots):
        self._check_cache_setup()
        if isinstance(kv_cache, SharedKVPool):
            if kv_cache is not self.shared_pool:
                raise ValueError("KV pool belongs to another generator")
            return [self.caches[slot] for slot in slots]
        return [self.caches[slot] for slot in slots] if kv_cache is None else kv_cache

    def allocate_kv_cache(self, shape, dtype, num_layers):
        return self.allocate_kv_cache_per_layer([(shape, dtype, index) for index in range(num_layers)])

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Pinned allocator contract; setup only, returning opaque fixed-slot storage.

        Free owned private KV before allocating global storage to avoid doubling
        full-context DRAM. After the first free, failure poisons this instance;
        callers must reconstruct it, never fall back to the old private cache.
        """
        if self.cache_setup_error is not None:
            raise RuntimeError(f"Generator cache setup failed; reconstruct the generator: {self.cache_setup_error}")
        if self.prepared or self.traces or self.prefill_traces or self.sampling_traces:
            raise RuntimeError("KV allocation must precede preparation and every capture")
        if self.shared_pool is not None or not self.owns_cache_storage:
            raise ValueError("Cache storage is already externally bound; allocation cannot replace it")
        per_layer_specs = tuple(per_layer_specs)
        specs = normalize_specs(per_layer_specs)
        if any(spec.dtype != self.model.precision["kv_cache_dtype"] for spec in specs):
            raise ValueError("Shared KV allocation dtype must match the constructed precision policy")
        all_caches = self.caches + ([] if self.slot_mover.scratch is None else [self.slot_mover.scratch])
        for cache in all_caches:
            if cache.position != 0:
                raise ValueError("KV allocation requires empty request slots")
            states = [layer.attention for layer in cache.layers if layer.attention is not None]
            if len(states) != len(specs):
                raise ValueError("Allocator specs must enumerate indexed-attention layers only, in decoder order")
        required = {}
        for spec in specs:
            required[spec.tensor_idx] = required.get(spec.tensor_idx, 0) + 2 * self.max_batch_size
            if required[spec.tensor_idx] > spec.shape[0]:
                raise ValueError("Pool must fit disjoint temporary warmup pages for every fixed slot and group")
        old = {
            id(tensor): tensor
            for cache in all_caches
            for layer in cache.layers
            if layer.attention is not None
            for tensor in (layer.attention.k_cache, layer.attention.v_cache)
            if tensor is not None
        }
        ttnn.synchronize_device(self.mesh)
        self.cache_setup_error = "shared-pool replacement is incomplete"
        try:
            for cache in all_caches:
                cache.kv_bound = False
            for tensor in old.values():
                ttnn.deallocate(tensor)
            for cache in all_caches:
                for layer in cache.layers:
                    if layer.attention is not None:
                        layer.attention.k_cache = layer.attention.v_cache = None
            pool = SharedKVPool(per_layer_specs, self.mesh)
            pool.bind(all_caches)
            tables = pool.warmup_tables(batch_size=self.max_batch_size, capacity=((self.capacity + 127) // 128) * 128)
            for slot, cache in enumerate(self.caches):
                states = [layer.attention for layer in cache.layers if layer.attention is not None]
                for state, table in zip(states, tables, strict=True):
                    self._refresh(state.page_table, table[slot : slot + 1])
            self.slot_mover = SlotStateMover(self.caches, self.slot_mover.scratch)
            self.shared_pool = pool
            self.page_tables.clear()  # Warmup reservations are not scheduler ownership.
            self.cache_setup_error = None
            self.cache_binding_pending = False
            return pool
        except Exception as error:
            self.cache_setup_error = f"{type(error).__name__}: {error}"
            raise

    def _upload(self, value, dtype=ttnn.uint32):
        return ttnn.from_torch(value.contiguous(), device=self.mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _refresh(self, destination, value):
        host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
        ttnn.copy_host_to_device_tensor(host, destination)

    def set_sampling_params(self, params=None, *, host_sampling=False, slots=None):
        params = params or SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        host_params = HostSamplingParams.from_object(params)
        if host_sampling:
            self.host_sampling_params = host_params
            self.sampling_host_only = True
            return
        defaults = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        params = SamplingParams(
            **{f.name: getattr(params, f.name, getattr(defaults, f.name)) for f in fields(defaults)}
        )
        if slots is not None:
            try:
                slots = tuple(slot.__index__() for slot in slots)
            except (AttributeError, TypeError) as error:
                raise ValueError("Sampling destination slots must be integers") from error
            if len(set(slots)) != len(slots) or any(not 0 <= slot < self.max_batch_size for slot in slots):
                raise ValueError("Invalid sampling destination slots")
        raw = {}
        for field in fields(params):
            value = getattr(params, field.name)
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Sampling parameters must be host metadata")
                value = value.tolist()
            if isinstance(value, tuple):
                value = list(value)
            raw[field.name] = value
        count = (
            len(raw["temperature"])
            if isinstance(raw["temperature"], list)
            else (len(slots) if slots is not None else 1)
        )
        if not 1 <= count <= 32 or (slots is not None and len(slots) != count):
            raise ValueError("Sampling parameters must describe every submitted request")

        def rows(name, default):
            value = raw[name]
            if value is None:
                value = default
            if not isinstance(value, list):
                return [value] * count
            if len(value) != count:
                raise ValueError(f"sampling_params.{name} must have one value per submitted request")
            return value

        temperatures = torch.as_tensor(rows("temperature", 0.0), dtype=torch.float64)
        top_k = torch.as_tensor(rows("top_k", 1), dtype=torch.float64)
        top_p = torch.as_tensor(rows("top_p", 1.0), dtype=torch.float64)
        if not torch.isfinite(temperatures).all() or (temperatures < 0).any():
            raise ValueError("Invalid sampling temperature")
        if not torch.isfinite(top_p).all() or ((top_p < 0) | (top_p > 1)).any():
            raise ValueError("Invalid sampling top-p")
        if not torch.isfinite(top_k).all() or (top_k != top_k.trunc()).any() or (top_k < -1).any():
            raise ValueError("Invalid sampling top-k")
        if (((top_k < 1) | (top_k > 40)) & (temperatures != 0)).any():
            raise ValueError("Unbounded or top-k above 40 requires explicit host_sampling=True")
        for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            if any(value != default for value in rows(name, default)):
                raise ValueError("Penalties require explicit host_sampling=True")
        if any(rows("enable_log_probs", False)):
            raise ValueError("Logprobs require explicit host_sampling=True on this mesh")
        seeds = raw["seed"] if isinstance(raw["seed"], list) else [raw["seed"]]
        if isinstance(raw["seed"], list) and len(seeds) != count:
            raise ValueError("Sampling seeds must have one value per submitted request")
        for seed in seeds:
            if seed is not None:
                try:
                    seed.__index__()
                except (AttributeError, TypeError) as error:
                    raise ValueError("Sampling seeds must be integers or None") from error
        # Validate before the common formatter, which clamps k/p and inverts
        # temperature. Scalars broadcast only over the submitted request rows.
        raw.update(temperature=temperatures.tolist(), top_k=top_k.to(torch.int64).tolist(), top_p=top_p.tolist())
        params = replace(params, **raw)
        formatted = format_sampling_params(params, 32, max_top_k=40)
        if slots is not None:
            merged = {}
            for field in fields(formatted):
                previous = getattr(self.formatted_sampling_params, field.name)
                incoming = getattr(formatted, field.name)
                # The common formatter intentionally preserves num_logprobs=None.
                # With logprobs disabled its per-slot neutral representation is0.
                neutral = 0 if field.name == "num_logprobs" else None
                values = (
                    list(previous) if isinstance(previous, list) else [neutral if previous is None else previous] * 32
                )
                incoming = incoming if isinstance(incoming, list) else [neutral if incoming is None else incoming] * 32
                for row, slot in enumerate(slots):
                    values[slot] = incoming[row]
                merged[field.name] = values
            formatted = replace(formatted, **merged)
        # Keep authoritative slot metadata on the host. The next sampling
        # submission installs only its active rows, once per scheduler change.
        # Native sampling executes all 32 physical rows, so stale inactive k40
        # must never reach the narrow graph. Validation-only objects use this
        # same setter without allocating or uploading anything.
        if formatted != getattr(self, "formatted_sampling_params", None):
            self._sampling_params_version = getattr(self, "_sampling_params_version", 0) + 1
        self.host_sampling_params = host_params
        self.sampling_host_only = False
        self.formatted_sampling_params = formatted
        self.request_seeds = list(formatted.seed)

    def _configure_sampling(self, slots):
        """Refresh fixed-address sampler parameters only after metadata changes."""
        slots = tuple(sorted(slots))
        key = (slots, self._sampling_params_version)
        if key == getattr(self, "_sampling_configuration", None):
            return
        active = set(slots)
        formatted = self.formatted_sampling_params
        effective = replace(
            formatted,
            top_k=[value if row in active else 1 for row, value in enumerate(formatted.top_k)],
            top_p=[value if row in active else 0.0 for row, value in enumerate(formatted.top_p)],
            temperature=[value if row in active else 1.0 for row, value in enumerate(formatted.temperature)],
        )
        if effective != getattr(self, "_effective_sampling_params", None):
            self.sampler.reset_sampling_params(effective)
            self._effective_sampling_params = effective
        self._sampling_configuration = key

    def _warm_sampling(self, *, all_configs=True):
        """Compile both candidate families before retaining any live trace."""
        self._configure_sampling(tuple(range(self.max_batch_size)))
        saved = self._effective_sampling_params
        narrow = replace(saved, top_k=[1] * 32, top_p=[0.0] * 32, temperature=[1.0] * 32)
        self.sampler.reset_sampling_params(narrow)
        self.sampler.precompile(
            self.sample_logits,
            tt_out_tok=self.decode_tokens,
            all_configs=all_configs,
            compile_token_update=True,
        )
        if self.sampler.tt_sampling.max_top_k > 32:
            wide = replace(saved, top_k=[40] * 32, top_p=[0.95] * 32, temperature=[1.0] * 32)
            self.sampler.reset_sampling_params(wide)
            self.sampler.precompile(
                self.sample_logits,
                tt_out_tok=self.decode_tokens,
                all_configs=False,
                compile_token_update=True,
            )
        self.sampler.reset_sampling_params(saved)

    def _decode_body(self, slots):
        tokens, words = [], []
        for slot in slots:
            token = ttnn.reshape(self.decode_tokens[:, :, :, slot : slot + 1], (1, 1))
            tokens.append(token)
            words.append(ttnn.concat([ttnn.reshape(token, (1, 1, 1)), self.word_suffix], dim=2))
        outputs = self.model.decode_batch(tuple(tokens), tuple(words), caches=[self.caches[i] for i in slots])
        # Sampler slots retain scheduler slot indices, including inactive rows.
        if len(slots) <= 24:
            # Native BF16 in-place fill, then exact tiled row copies. This avoids
            # 32 untilizes and needs no additional physical32 DRAM template.
            # Measured crossover: row writes win through24; concat wins at28/32.
            ttnn.full_like(self.sample_logits, -100.0, optional_tensor=self.sample_logits)
            for slot, output in zip(slots, outputs, strict=True):
                write_token_chunk(output, self.sample_logits, slot)
        else:
            rows = {slot: output for slot, output in zip(slots, outputs, strict=True)}
            combined = ttnn.concat([rows.get(i, self.inactive_logits) for i in range(32)], dim=2)
            ttnn.copy(combined, self.sample_logits)

    def _reset_cache(self, cache):
        self._check_cache_setup()
        self.prefill_checkpoints.invalidate(slot for slot, owned in enumerate(self.caches) if owned is cache)
        self.counters["cache_resets"] += 1
        for state in cache.layers:
            for key in ("conv", "recurrent", "ple_conv"):
                value = getattr(state, key, None)
                if value is not None:
                    ttnn.multiply(value, 0.0, output_tensor=value)
            if state.token_history is not None:
                history = torch.zeros(1, 16, dtype=torch.int32)
                history[:, :2] = EOS
                self._refresh(state.token_history, history)
            if state.attention is not None:
                attn = state.attention
                for key in ("k_cache", "v_cache", "raw_history", "compressed_keys"):
                    if cache.shared_kv and key in ("k_cache", "v_cache"):
                        continue
                    value = getattr(attn, key)
                    ttnn.multiply(value, 0.0, output_tensor=value)
                self._refresh(attn.position, torch.zeros(1, dtype=torch.int32))
                if not cache.shared_kv:
                    self._refresh(
                        attn.page_table, torch.arange(attn.page_table.shape[-1], dtype=torch.int32).reshape(1, -1)
                    )
        cache.position = 0
        for slot, bound in enumerate(self.caches):
            if bound is cache:
                self.page_tables.pop(slot, None)
                self.pending_prefill.pop(slot, None)

    def reset(self):
        for cache in self.caches:
            self._reset_cache(cache)
        self.sampler.reset_output_state()
        self.seed_state.reset_slots(
            self.request_seeds[: self.max_batch_size], [0] * self.max_batch_size, list(range(self.max_batch_size))
        )

    def prepare(self):
        """Compile all physical prefill buckets and decode widths before capture."""
        self._check_cache_setup()
        if self.shared_pool is not None:
            self.shared_pool.freeze()
        if self.prepared:
            return
        preparation_start = time.perf_counter()
        self.inactive_logits = ttnn.full(
            (1, 1, 1, self.model.config.vocab_size // 4),
            -100.0,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        for n in PREFILL_BUCKETS:
            warmup_start = time.perf_counter()
            print(f"PREPARE_PREFILL {n}", flush=True)
            self._reset_cache(self.caches[0])
            if self.prefill_cache is not self.caches[0]:
                self.slot_mover._copy(self.caches[0], self.prefill_cache)
            self._prefill_body(n)
            ttnn.full_like(self.sample_logits, -100.0, optional_tensor=self.sample_logits)
            self._write_prefill_last(n, 0)
            self._event("warmup", signature=("prefill", n), host_submission_seconds=time.perf_counter() - warmup_start)
        if self.factored_decode is None:
            self._decode_body((0,))
        else:
            self.factored_decode.warmup()
        self.seed_state.prepare()
        self.token_feedback.prepare()
        self.seed_state.remap(tuple(range(self.max_batch_size)), warmup=True)
        self.slot_mover.warmup()
        self.prefill_checkpoints.warmup(self.caches)
        remap_token_buffer(self.decode_tokens, tuple(range(self.max_batch_size)), warmup=True)
        ttnn.copy(self.decode_tokens, self.saved_tokens)
        self._restore_inactive_tokens(())
        print("PREPARE_SAMPLING", flush=True)
        warmup_start = time.perf_counter()
        self._warm_sampling()
        self._event(
            "warmup", signature=("sampling", "all_configs"), host_submission_seconds=time.perf_counter() - warmup_start
        )
        ttnn.copy(self.seed_skip, self.sampler.tt_sampling.seeds_tt_tensor)
        self.reset()
        ttnn.synchronize_device(self.mesh)
        self.counters["synchronizations"] += 1
        self.slot_mover.prepare_copy_traces(self._capture_state_copy)
        if self.factored_decode is not None:
            # Warm every shape, reset state, then retain the complete bounded
            # capture set before serving the first request.
            self.factored_decode.capture()
        self.prepared = True
        self._event("prepared", seconds=time.perf_counter() - preparation_start, counters=dict(self.counters))
        print("PREPARED", flush=True)

    def _capture_state_copy(self, body, signature):
        trace = self._record_trace(body, signature=signature)
        self.traces[signature] = trace
        self.counters["captures"] += 1
        return lambda: ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)

    def _prefill_body(self, count, slot=None):
        # Explicit slots remain available to standalone diagnostic warmups.
        cache = self.prefill_cache if slot is None else self.caches[slot]
        output = self.prefill_output
        logits = self.model.forward_chunk(self.prefill_inputs[count], self.prefill_words[count], cache=cache)
        physical = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - count), (0, 0)], value=0.0) if count != 32 else logits
        ttnn.copy(physical, output)

    def _write_prefill_last(self, count, slot):
        # A nonaligned tiled slice may allocate a physical tile row. Its copy
        # completes in CQ order before workspace reuse; drop its Python owner
        # before the next model or sampling replay can reuse reserved addresses.
        last = self.prefill_output[:, :, count - 1 : count, :]
        write_token_chunk(last, self.sample_logits, slot)
        del last

    def _prefill_chunk(self, count, slot):
        key = count
        if key not in self.prefill_traces:
            trace = self._record_trace(lambda: self._prefill_body(count), signature=("prefill", count))
            self.prefill_traces[key] = trace
            self.counters["captures"] += 1
        ttnn.execute_trace(self.mesh, self.prefill_traces[key], cq_id=0, blocking=False)
        self.counters["prefill_replays"] += 1

    def _capture(self, slots):
        if self.max_batch_size != 1 or tuple(slots) != (0,):
            raise ValueError("Whole-model capture is reserved for standalone B1; use _replay_decode")
        # A consumed decode token ends the prefill-only rollback opportunity.
        self.prefill_checkpoints.invalidate(slots)
        # Scheduler row order does not change physical cache/token bindings.
        # Public submission/output order stays outside this canonical trace key.
        slots = tuple(sorted(slots))
        if slots not in self.traces:
            trace = self._record_trace(lambda: self._decode_body(slots), signature=("decode", slots))
            self.traces[slots] = trace
            self.counters["captures"] += 1
        return self.traces[slots]

    def _replay_decode(self, slots):
        if self.factored_decode is None:
            ttnn.execute_trace(self.mesh, self._capture(slots), cq_id=0, blocking=False)
        else:
            slots = tuple(sorted(slots))
            # Invalidate rollback only for requests that consume a decode token.
            self.prefill_checkpoints.invalidate(slots)
            self.factored_decode.replay(slots)
        self.counters["model_replays"] += 1

    def _event(self, kind, **fields):
        sink = getattr(self, "event_sink", None)
        if sink is not None:
            sink(dict(kind=kind, time=time.monotonic(), **fields))

    def _record_trace(self, body, signature=None):
        started = time.perf_counter()
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            body()
        except BaseException:
            # Close the capture without replaying its partial command stream.
            # Otherwise mesh teardown can wait forever on unfinished capture.
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
            ttnn.release_trace(self.mesh, trace)
            self._event("failed_capture_release", signature=signature, trace_id=str(trace))
            raise
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self._event("capture", signature=signature, trace_id=str(trace), seconds=time.perf_counter() - started)
        return trace

    def _restore_inactive_tokens(self, slots):
        # Membership upload happens only on a scheduler transition. Inside
        # sampling capture it is already current, so this enqueues only the
        # fixed-address UINT32 select; replay consumes the live device mask.
        self.seed_state.set_active_slots(slots)
        self.token_feedback.forward()

    def _sample(self, slots=(0,)):
        slots = tuple(sorted(slots))
        self._configure_sampling(slots)
        self.seed_state.set_active_slots(slots)
        key = (
            self.sampler._penalties_active,
            self.sampler.tt_sampling.force_argmax_sampling,
            self.sampler.tt_sampling.candidate_width,
        )
        if self.sampler._log_probs_active:
            raise ValueError("Use explicit host logits for logprob requests on this mesh")
        if key not in self.sampling_traces:

            def body():
                ttnn.copy(self.decode_tokens, self.saved_tokens)
                self.seed_state.forward()
                self.sampler._run_sampling(
                    self.sample_logits, penalties_on=key[0], grammar_on=False, tt_out_tok=self.decode_tokens
                )
                self._restore_inactive_tokens(slots)
                ttnn.copy(self.seed_skip, self.sampler.tt_sampling.seeds_tt_tensor)

            trace = self._record_trace(body, signature=("sampling", key))
            self.sampling_traces[key] = trace
            self.counters["captures"] += 1
        ttnn.execute_trace(self.mesh, self.sampling_traces[key], cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1

    def _read_tokens(self):
        self.counters["readbacks"] += 1
        self.counters["synchronizations"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.decode_tokens)[0]).flatten().tolist()

    def _read_logits(self, output):
        self.counters["readbacks"] += 1
        self.counters["synchronizations"] += 1
        return (
            ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3))
            .reshape(-1, self.model.config.vocab_size)
            .float()
        )

    def _table_plan(
        self,
        page_table,
        caches,
        slots,
        lengths,
        *,
        page_tables_per_layer=None,
        remap=None,
        table_rows=None,
        submission_rows=None,
        reuse_decode_snapshot=False,
        prefill_starts=None,
    ):
        """Validate tables and all retained slots before any state mutation.

        Input table rows follow submission order (including padded runner rows),
        not physical slot indices. Per-layer routing overrides group0's common
        table, exactly as the pinned runner passes both arguments.
        """
        pool = getattr(self, "shared_pool", None)
        table_rows = list(range(len(slots))) if table_rows is None else list(table_rows)
        submission_rows = len(slots) if submission_rows is None else submission_rows
        states = [[layer.attention for layer in cache.layers if layer.attention is not None] for cache in caches]
        count = len(states[0]) if states else len([s for s in self.caches[0].layers if s.attention is not None])
        provided = (
            page_tables_per_layer
            if page_tables_per_layer is not None
            else (None if page_table is None else [page_table] * count)
        )
        tables = None
        if provided is not None:
            if len(provided) != count:
                raise ValueError("Expected one page table per indexed-attention layer")
            for table in provided:
                if not isinstance(table, torch.Tensor) or table.device.type != "cpu":
                    raise ValueError("Page tables must be host tensors")
                if table.ndim != 2 or not submission_rows <= table.shape[0] <= self.max_batch_size:
                    raise ValueError("Page tables must cover submitted rows, optionally padded to fixed-slot capacity")
                if table.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
                    raise ValueError("Page tables must have integer dtype")
            # Host sampling reloads tokens every step, but usually supplies the
            # same full-width scheduler tables. Reuse only our cloned, validated
            # snapshots: caller tensors can be views mutated between calls.
            # Prefill must return tables even when identical: a request reset
            # after planning clears its snapshot before _bind restores it.
            identity = tuple(range(self.max_batch_size))
            if (
                reuse_decode_snapshot
                and pool is not None
                and len(pool.specs) == count
                and (remap is None or remap == identity)
                and 1 <= len(slots) <= 32
                and len(slots) == len(caches) == len(lengths) == len(table_rows)
                and len(set(slots)) == len(slots)
                and all(type(n) is int and 0 <= n <= self.capacity for n in lengths)
                and all(type(row) is int and 0 <= row < submission_rows for row in table_rows)
                and all(type(slot) is int and 0 <= slot < self.max_batch_size for slot in slots)
                and all(cache is self.caches[slot] for cache, slot in zip(caches, slots, strict=True))
                and all(
                    getattr(cache, "shared_kv_pool", None) is pool and (not cache.position or slot in self.page_tables)
                    for slot, cache in enumerate(self.caches)
                )
                and all(
                    slot in self.page_tables and (length + 31) // 32 == (self.caches[slot].position + 31) // 32
                    for slot, length in zip(slots, lengths, strict=True)
                )
            ):
                # Retain the original remap/key checks even for an identity
                # mapping; equal-valued floats are not valid integer indices.
                snapshot = remap_slot_mapping(self.page_tables, identity if remap is None else remap)
                unchanged = True
                for row, slot in zip(table_rows, slots, strict=True):
                    previous = snapshot[slot]
                    previous = (previous,) * count if isinstance(previous, torch.Tensor) else previous
                    if len(previous) != count or any(
                        not torch.equal(table[row : row + 1], old)
                        for table, old in zip(provided, previous, strict=True)
                    ):
                        unchanged = False
                        break
                if unchanged:
                    return None
            rows = [table[table_rows] for table in provided]
            if pool is not None:
                tables = pool.normalize_tables(
                    batch_size=len(slots), capacity=self.capacity, page_tables_by_layer=rows, lengths=lengths
                )
            else:
                for index, table in enumerate(rows):
                    for row, per_cache in enumerate(states):
                        state = per_cache[index]
                        if tuple(table[row : row + 1].shape) != tuple(state.page_table.shape):
                            raise ValueError("Page-table physical shape differs from the configured cache")
                        if ((table[row] < 0) | (table[row] >= state.k_cache.shape[0])).any():
                            raise ValueError("Page-table entry is outside the allocated KV pool")
                tables = tuple(table.int().contiguous() for table in rows)
        identity = tuple(range(self.max_batch_size))
        if prefill_starts is not None:
            remap = (
                plan_pending_prefill(
                    tables=tables,
                    shadows=self.page_tables,
                    positions=[cache.position for cache in self.caches],
                    pending=self.pending_prefill,
                    slots=slots,
                    starts=prefill_starts,
                )
                if pool is not None
                else identity
            )
        if pool is None:
            return (tables, remap) if prefill_starts is not None else tables
        remap = identity if remap is None else remap
        shadow = remap_slot_mapping(self.page_tables, remap)
        retained_lengths = [self.caches[source].position for source in remap]
        # An unchanged within-page replay needs no table reconstruction or upload.
        boundary = any(
            (length + 31) // 32 != (retained_lengths[slot] + 31) // 32
            for slot, length in zip(slots, lengths, strict=True)
        )
        if tables is None and not boundary and remap == tuple(range(self.max_batch_size)):
            return (None, remap) if prefill_starts is not None else None
        full = [
            torch.zeros(self.max_batch_size, ((self.capacity + 127) // 128) * 4, dtype=torch.int32)
            for _ in range(count)
        ]
        for slot, length in enumerate(retained_lengths):
            previous = shadow.get(slot)
            if previous is None and length:
                raise ValueError("Retained shared-cache slot has no authoritative page-table mapping")
            if previous is not None:
                previous = (previous,) * count if isinstance(previous, torch.Tensor) else previous
                for index, value in enumerate(previous):
                    full[index][slot] = value[0]
        for row, (slot, length) in enumerate(zip(slots, lengths, strict=True)):
            if tables is None and slot not in shadow:
                raise ValueError("Shared-cache requests require explicit scheduler page tables before first use")
            if tables is not None:
                for index, table in enumerate(tables):
                    full[index][slot] = table[row]
            retained_lengths[slot] = length
        pool.validate_active_pages(full, retained_lengths)
        return (tables, remap) if prefill_starts is not None else tables

    def _bind(self, page_table, caches, slots, positions):
        # page_table has already been normalized and fully validated by _table_plan.
        for row, slot in enumerate(slots):
            if caches[row] is not self.caches[slot]:
                raise ValueError("Cache ownership must be bound to fixed slots before preparation")
            if page_table is not None:
                values = tuple(table[row : row + 1] for table in page_table)
                previous = self.page_tables.get(slot)
                if isinstance(previous, torch.Tensor):
                    previous = (previous,) * len(values)
                changed = False
                states = [layer.attention for layer in caches[row].layers if layer.attention is not None]
                for index, (state, value) in enumerate(zip(states, values, strict=True)):
                    if previous is None or not torch.equal(previous[index], value):
                        self._refresh(state.page_table, value)
                        changed = True
                self.page_tables[slot] = tuple(value.clone() for value in values)
                self.counters["page_table_refreshes"] += int(changed)
            if positions is not None:
                for state in caches[row].layers:
                    if state.attention is not None:
                        self._refresh(state.attention.position, torch.tensor([positions[row]], dtype=torch.int32))
                caches[row].position = int(positions[row])
                self.counters["position_refreshes"] += 1

    @staticmethod
    def prefill_spans(start, end):
        """Cover valid rows with six reusable shapes, retaining the absolute32 grid.

        Logical tails and resumed prefixes select dyadic physical buckets;
        padding never advances recurrent state or fills extra cache positions.
        """
        while start < end:
            remaining = min(end, (start // 32 + 1) * 32) - start
            stop = start + (1 << (remaining.bit_length() - 1))
            yield start, stop
            start = stop

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        return_all_logits=False,
        slots=None,
        empty_slots=None,
        return_device=False,
        sampling_params=None,
        page_tables_per_layer=None,
        prefill_sample_mask=None,
        **kwargs,
    ):
        # Validate the entire submission before any reset or upload: a bad
        # later row must not discard another request's already resident state.
        if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu":
            raise ValueError("Prefill tokens must be a host tensor")
        if tokens.ndim != 2:
            raise ValueError("Prefill tokens must have shape [batch, prompt width]")
        integer_dtypes = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
        if tokens.dtype not in integer_dtypes:
            raise ValueError("Prefill tokens must have integer dtype")
        if slots is None and empty_slots is not None:
            slots = empty_slots
        for metadata in (prompt_lens, start_pos, slots):
            if getattr(getattr(metadata, "device", None), "type", "cpu") != "cpu":
                raise ValueError("Prefill lengths, positions, and slots must be host metadata")
        try:
            prompt_lens = tuple(end.__index__() for end in prompt_lens)
            slots = tuple(range(len(prompt_lens))) if slots is None else tuple(slot.__index__() for slot in slots)
            starts = (0,) * len(slots) if start_pos is None else tuple(start.__index__() for start in start_pos)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Prefill lengths, positions, and slots must be integer rows") from error
        if not slots or len(slots) != tokens.shape[0] or len(prompt_lens) != len(slots) or len(starts) != len(slots):
            raise ValueError("Prefill tokens, lengths, positions, and slots must have the same nonzero row count")
        if len(set(slots)) != len(slots) or any(not 0 <= s < self.max_batch_size for s in slots):
            raise ValueError("Invalid fixed slots")
        if prefill_sample_mask is None:
            sample_mask = (True,) * len(slots)
        else:
            mask = torch.as_tensor(prefill_sample_mask)
            if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != (len(slots),):
                raise ValueError("prefill_sample_mask must contain one host Boolean per submitted row")
            sample_mask = tuple(mask.tolist())
        if return_all_logits and not all(sample_mask):
            raise ValueError("Masked prefill supports last-token outputs only")
        sample_slots = tuple(slot for slot, emit in zip(slots, sample_mask, strict=True) if emit)
        caches = self._resolve_caches(kv_cache, slots)
        if len(caches) != len(slots):
            raise ValueError("Prefill cache list must have one entry per submitted row")
        if any(cache is not self.caches[slot] for cache, slot in zip(caches, slots, strict=True)):
            raise ValueError("Cache ownership must be bound to fixed slots before preparation")
        for row, (start, end, cache) in enumerate(zip(starts, prompt_lens, caches, strict=True)):
            if not 0 <= start < end <= min(tokens.shape[1], self.capacity, cache.capacity):
                raise ValueError("Expected 0 <= absolute start < prompt end <= supported context")
            # Prefix tokens also initialize sampler history on resumed prefill;
            # padding beyond this row's logical end is deliberately ignored.
            prompt = tokens[row, :end]
            if ((prompt < 0) | (prompt >= self.model.config.vocab_size)).any():
                raise ValueError("Prefill token is outside the model vocabulary")
        tables, prefill_remap = self._table_plan(
            page_table,
            caches,
            slots,
            prompt_lens,
            page_tables_per_layer=page_tables_per_layer,
            prefill_starts=starts,
        )
        if sampling_params is not None and not all(sample_mask):
            # Intermediate rows can carry host-only request features. They do
            # not sample now and must not change any sampler slot metadata.
            if not sample_slots:
                sampling_params = None
            else:
                defaults = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
                selected = {}
                for field in fields(defaults):
                    value = getattr(sampling_params, field.name, getattr(defaults, field.name))
                    if isinstance(value, torch.Tensor):
                        if value.device.type != "cpu":
                            raise ValueError("Sampling parameters must be host metadata")
                        value = value.tolist()
                    if not isinstance(value, (list, tuple)):
                        # Preserve the common formatter's scalar-seed contract:
                        # only submitted row zero receives a scalar seed.
                        value = [value] + [None] * (len(slots) - 1) if field.name == "seed" else [value] * len(slots)
                    if len(value) != len(slots):
                        raise ValueError(f"sampling_params.{field.name} must have one value per submitted request")
                    selected[field.name] = [item for item, emit in zip(value, sample_mask, strict=True) if emit]
                sampling_params = SamplingParams(**selected)
        if sampling_params is not None:
            if return_all_logits:
                raise ValueError("Device-sampled prefill cannot return all logits")
            validation = SimpleNamespace(
                sampler=SimpleNamespace(reset_sampling_params=lambda _: None),
                max_batch_size=self.max_batch_size,
                formatted_sampling_params=self.formatted_sampling_params,
            )
            type(self).set_sampling_params(validation, sampling_params, slots=sample_slots)
        self.prepare()
        if prefill_remap != tuple(range(self.max_batch_size)):
            moved = [
                {"source": source, "destination": destination, "position": self.caches[source].position}
                for destination, source in enumerate(prefill_remap)
                if destination != source and source in self.pending_prefill
            ]
            self.slot_mover.apply(prefill_remap)
            self.prefill_checkpoints.remap(prefill_remap)
            self.page_tables = remap_slot_mapping(self.page_tables, prefill_remap)
            self.pending_prefill = remap_slot_mapping(self.pending_prefill, prefill_remap)
            self._event("pending_prefill_relocation", moves=moved)
        # All submission/table/sampling validation has completed before any
        # checkpoint copy or reset. Decide against the committed cache position,
        # before _bind applies the caller's authoritative position metadata.
        restore_tail = tuple(
            (
                self.prefill_checkpoints.can_restore(slot, start, int(tokens[row, start - 1]), cache.position)
                if start
                else False
            )
            for row, (slot, start, cache) in enumerate(zip(slots, starts, caches, strict=True))
        )
        if sampling_params is not None:
            self.set_sampling_params(sampling_params, slots=sample_slots)
            return_device = True
        for row, (slot, start, end) in enumerate(zip(slots, starts, prompt_lens, strict=True)):
            if start == 0:
                self._reset_cache(caches[row])
                self.sampler.reset_output_state(slots=[slot])
        self._bind(tables, caches, slots, starts)
        if sample_slots:
            if self.sampler._penalties_active:
                prompt_history = torch.full((32, tokens.shape[1]), -1, dtype=torch.int64)
                for row, slot in enumerate(slots):
                    if sample_mask[row]:
                        prompt_history[slot, : prompt_lens[row]] = tokens[row, : prompt_lens[row]]
                self.sampler.reset_prompt_tokens(prompt_history, slots=list(sample_slots))
            self.seed_state.reset_slots(
                [self.request_seeds[slot] for slot in sample_slots],
                [end for end, emit in zip(prompt_lens, sample_mask, strict=True) if emit],
                list(sample_slots),
            )
        results = []
        if return_device and sample_slots:
            ttnn.full_like(self.sample_logits, -100.0, optional_tensor=self.sample_logits)
        for row, (start, end, cache) in enumerate(zip(starts, prompt_lens, caches, strict=True)):
            if cache is not self.prefill_cache:
                self.slot_mover._copy(cache, self.prefill_cache)
            replay_start = start
            if restore_tail[row]:
                self.prefill_checkpoints.restore(slot=slots[row], cache=self.prefill_cache)
                replay_start -= 1
            self.prefill_checkpoints.invalidate((slots[row],))
            chunks = []
            for offset, stop in self.prefill_spans(replay_start, end):
                self._event("prefill_chunk", slot=slots[row], start=offset, end=stop, logical_end=end)
                logical = tokens[row : row + 1, offset:stop].int()
                n = logical.shape[1]
                words = torch.zeros(1, n, 16, dtype=torch.int32)
                words[:, :, 0], words[:, :, 1] = logical, 1
                self._refresh(self.prefill_inputs[n], logical)
                self._refresh(self.prefill_words[n], words)
                if n == 1 and stop == end and end % 2:
                    self.prefill_checkpoints.save(slots[row], self.prefill_cache, end=end, token=int(logical[0, 0]))
                self._prefill_chunk(n, slots[row])
                if sample_mask[row] and return_device and offset + n == end:
                    self._write_prefill_last(n, slots[row])
                elif sample_mask[row] and not return_device and (return_all_logits or offset + n == end):
                    chunks.append(self._read_logits(self.prefill_output)[max(0, start - offset) : n])
            self.prefill_cache.position = end
            if cache is not self.prefill_cache:
                self.slot_mover._copy(self.prefill_cache, cache)
            cache.position = end
            if sample_mask[row]:
                self.pending_prefill.pop(slots[row], None)
            else:
                self.pending_prefill[slots[row]] = end
            if not return_device:
                if sample_mask[row]:
                    value = torch.cat(chunks)
                    results.append(value if return_all_logits else value[-1])
                else:
                    results.append(torch.zeros(self.model.config.vocab_size))
        if not sample_slots:
            return None
        if return_device:
            if sampling_params is not None:
                self._sample(sample_slots)
                values = self._read_tokens()
                return torch.tensor(
                    [[values[slot] if emit else -1] for slot, emit in zip(slots, sample_mask, strict=True)],
                    dtype=torch.int64,
                )
            return self.sample_logits
        if return_all_logits:
            return torch.nn.utils.rnn.pad_sequence(results, batch_first=True)
        return torch.stack(results).unsqueeze(1)

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        slots=None,
        sample_on_device=False,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        sampling_params=None,
        read_from_device=True,
        slot_remap=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise ValueError("reset_batch is incompatible with the explicit reload contract")
        if reload_inputs and reload_page_table:
            raise ValueError("Input reload already includes page tables")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Sampling reset requires authoritative inputs")
        # Reject malformed submissions before preparation, slot migration, or any
        # persistent upload. The scheduler commits ownership only on acceptance.
        if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu":
            raise ValueError("Decode tokens must be a host tensor")
        if tokens.ndim not in (1, 2) or (tokens.ndim == 2 and tokens.shape[1] != 1):
            raise ValueError("Decode expects one token per submitted row")
        if tokens.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Decode tokens must have integer dtype")
        if getattr(getattr(start_pos, "device", None), "type", "cpu") != "cpu":
            raise ValueError("Decode positions must be host metadata")
        try:
            positions_host = tuple(pos.__index__() for pos in start_pos)
            slots = tuple(range(len(tokens))) if slots is None else tuple(slot.__index__() for slot in slots)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Decode positions and slots must be integer rows") from error
        if len(slots) != len(tokens) or len(positions_host) != len(slots):
            raise ValueError("Decode tokens, positions, and slots must have the same row count")
        if len(set(slots)) != len(slots) or any(not 0 <= s < self.max_batch_size for s in slots):
            raise ValueError("Invalid fixed slots")
        caches = self._resolve_caches(kv_cache, slots)
        if len(caches) != len(slots):
            raise ValueError("Decode cache list must have one entry per submitted row")
        active_rows = [i for i, pos in enumerate(positions_host) if pos >= 0]
        active_slots = tuple(slots[i] for i in active_rows)
        active_caches = [caches[i] for i in active_rows]
        if any(cache is not self.caches[slot] for cache, slot in zip(active_caches, active_slots, strict=True)):
            raise ValueError("Bind caller-owned cache at construction before preparation")
        remap = tuple(range(self.max_batch_size))
        if slot_remap is not None:
            remap = validate_remap(slot_remap, self.max_batch_size)
            nonidentity = remap != tuple(range(self.max_batch_size))
            if nonidentity and sample_on_device and not (reload_sampling_params and reset_sampling_state):
                raise ValueError("Slot migration requires an authoritative sampling-state reload")
            # Validate the shadow mapping before the first device copy as well.
            remap_slot_mapping(self.page_tables, remap)
        for row, slot, cache in zip(active_rows, active_slots, active_caches, strict=True):
            position = positions_host[row] if reload_inputs else self.caches[remap[slot]].position
            if not 0 <= position < min(self.capacity, cache.capacity):
                raise ValueError("Decode position exceeds the configured context")
            if reload_inputs and not 0 <= tokens[row].item() < self.model.config.vocab_size:
                raise ValueError("Decode token is outside the model vocabulary")
        tables = None
        if active_slots:
            lengths = [
                (positions_host[row] if reload_inputs else self.caches[remap[slot]].position) + 1
                for row, slot in zip(active_rows, active_slots, strict=True)
            ]
            tables = self._table_plan(
                page_table if reload_inputs or reload_page_table else None,
                active_caches,
                active_slots,
                lengths,
                page_tables_per_layer=page_tables_per_layer if reload_inputs or reload_page_table else None,
                remap=remap,
                table_rows=active_rows,
                submission_rows=len(slots),
                reuse_decode_snapshot=True,
            )
        if sample_on_device:
            self.sampler.validate_decode_state_commands(
                reload_sampling_params=reload_sampling_params, reset_sampling_state=reset_sampling_state
            )
        if reload_sampling_params:
            # Reuse the public setter's complete normalization/validation on a
            # host-only sink. Its real sampler upload happens after acceptance.
            validation = SimpleNamespace(
                sampler=SimpleNamespace(reset_sampling_params=lambda _: None),
                max_batch_size=self.max_batch_size,
                formatted_sampling_params=self.formatted_sampling_params,
            )
            type(self).set_sampling_params(validation, sampling_params)
        if not active_slots and slot_remap is None and not (reload_sampling_params or reset_sampling_state):
            return (
                torch.full((len(slots),), -1, dtype=torch.long)
                if sample_on_device
                else torch.zeros(len(slots), self.model.config.vocab_size)
            )
        self.prepare()
        if slot_remap is not None:
            self.slot_mover.apply(remap)
            self.prefill_checkpoints.remap(remap)
            self.pending_prefill = remap_slot_mapping(self.pending_prefill, remap)
            self.seed_state.remap(remap)
            self.request_seeds = [self.request_seeds[i] for i in remap] + self.request_seeds[self.max_batch_size :]
            remap_token_buffer(self.decode_tokens, remap)
            self.page_tables = remap_slot_mapping(self.page_tables, remap)
            self.sampler.apply_slot_remap(remap + tuple(range(self.max_batch_size, 32)))
        if reload_sampling_params:
            previous_seeds = list(self.request_seeds)
            self.set_sampling_params(sampling_params)
            if not reset_sampling_state:
                changed = [slot for slot in active_slots if self.request_seeds[slot] != previous_seeds[slot]]
                if changed:
                    next_counters = {
                        slot: (int(start_pos[row]) if reload_inputs else self.caches[slot].position) + 1
                        for row, slot in zip(active_rows, active_slots, strict=True)
                    }
                    self.seed_state.reset_slots(
                        [self.request_seeds[slot] for slot in changed],
                        [next_counters[slot] for slot in changed],
                        changed,
                    )
        if reset_sampling_state:
            self.sampler.reset_output_state(slots=list(active_slots))
            self.seed_state.reset_slots(
                [self.request_seeds[slot] for slot in active_slots],
                [int(start_pos[row]) + 1 for row in active_rows],
                list(active_slots),
            )
        if reset_sampling_state and reload_sampling_params:
            self.sampler.commit_decode_state_commands(
                reload_sampling_params=True, reset_sampling_state=True, sampling_state_slots=None
            )
        if active_slots and (reload_inputs or reload_page_table):
            positions = [int(start_pos[i]) for i in active_rows] if reload_inputs else None
            self._bind(tables, active_caches, active_slots, positions)
        if active_slots and reload_inputs:
            host_tokens = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            for row, slot in zip(active_rows, active_slots, strict=True):
                host_tokens[..., slot] = int(tokens[row].item())
            ttnn.copy(self.decode_tokens, self.saved_tokens)
            self._refresh(self.decode_tokens, host_tokens)
            self._restore_inactive_tokens(active_slots)
            self.counters["token_refreshes"] += 1
        if active_slots:
            self._replay_decode(active_slots)
            for cache in active_caches:
                cache.position += 1
        if sample_on_device and active_slots:
            self._sample(active_slots)
        if not active_slots:
            # Explicit ownership/state commands above still apply to an empty
            # launch; no forward, sample, or device read is needed for sentinels.
            return (
                torch.full((len(slots),), -1, dtype=torch.long)
                if sample_on_device
                else torch.zeros(len(slots), self.model.config.vocab_size)
            )
        submission = DecodeSubmission(
            self.decode_tokens if sample_on_device else self.sample_logits,
            slots,
            tuple(active_rows),
            sample_on_device,
        )
        if not read_from_device:
            return submission
        return self.process_decode_output_host(self.read_decode_output(submission)).squeeze(1)

    def prefill_logits(self, prompt_token_ids):
        self.prepare()
        self.reset()
        result = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=None,
            kv_cache=[self.caches[0]],
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )
        return result[0].unsqueeze(0)

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        self.prepare()
        self.reset()
        tokens = torch.tensor([prompt_token_ids])
        if start and prefill_prefix:
            self.prefill_forward(tokens, page_table=None, kv_cache=[self.caches[0]], prompt_lens=[start])
        result = self.prefill_forward(
            tokens,
            page_table=None,
            kv_cache=[self.caches[0]],
            prompt_lens=[len(prompt_token_ids)],
            start_pos=[start],
            return_all_logits=True,
        )[0]
        positions = list(range(start, len(prompt_token_ids))) if positions is None else list(positions)
        return result[[p - start for p in positions]].unsqueeze(0)

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        sampling_params=None,
        **kwargs,
    ):
        if sampling_params is not None:
            self.set_sampling_params(sampling_params, host_sampling=host_sampling)
        if self.sampling_host_only and not host_sampling:
            raise ValueError("This sampling configuration requires host_sampling=True")
        if not enable_trace:
            raise ValueError("Generation requires traced decode")
        if len(prompt_token_ids) + max_new_tokens - 1 > self.capacity:
            raise ValueError("Generation exceeds supported context")
        if not prompt_token_ids:
            raise ValueError("Prompt cannot be empty")
        self.prepare()
        self.reset()
        if max_new_tokens <= 0:
            return []
        self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=None,
            kv_cache=[self.caches[0]],
            prompt_lens=[len(prompt_token_ids)],
            return_device=True,
        )
        compatibility = (
            HostSampler(self.host_sampling_params, prompt_token_ids=prompt_token_ids) if host_sampling else None
        )
        output = []
        for step in range(max_new_tokens):
            if step:
                self._replay_decode((0,))
                self.caches[0].position += 1
            if host_sampling:
                prediction = compatibility.sample(self._read_logits(self.sample_logits)[0], commit=False).token_id
            else:
                self._sample()
                prediction = int(self._read_tokens()[0])
            output.append(prediction)
            if next_input is not None or host_sampling:
                forced = next_input(step, prediction) if next_input is not None else prediction
                if compatibility is not None:
                    compatibility.append_token(forced)
                value = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                value[..., 0] = int(forced)
                self._refresh(self.decode_tokens, value)
                self.counters["token_refreshes"] += 1
        return output

    def cache_spec(self):
        """Describe the opaque cache that an external scheduler can preallocate."""
        result = []
        for layer_index, state in zip(self.model.layer_indices, self.caches[0].layers, strict=True):
            tensors = {}
            for name in ("conv", "recurrent", "ple_conv", "token_history"):
                tensor = getattr(state, name, None)
                if tensor is not None:
                    tensors[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            if state.attention is not None:
                for name in ("k_cache", "v_cache", "page_table", "position", "raw_history", "compressed_keys"):
                    tensor = getattr(state.attention, name)
                    tensors[name] = (
                        {
                            "shape": None,
                            "dtype": self.model.precision["kv_cache_dtype"],
                            "allocation": "pending global pool",
                        }
                        if tensor is None
                        else {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                    )
            result.append({"layer": layer_index, "tensors": tensors})
        return {
            "context": self.capacity,
            "slots": self.max_batch_size,
            "page_size": 32,
            "layers": result,
            "kv_ownership": (
                "shared_global_pool"
                if self.shared_pool is not None
                else ("pending_global_pool" if self.cache_binding_pending else "private_per_slot")
            ),
            "indexed_attention_allocation_order": [
                index
                for index, state in zip(self.model.layer_indices, self.caches[0].layers, strict=True)
                if state.attention is not None
            ],
            "paged_kv_dtype": self.model.precision["kv_cache_dtype"],
            "local_kv_heads": 1,
            "head_dim": 256,
        }

    def release_request(self, slot):
        self._reset_cache(self.caches[slot])
        self.sampler.reset_output_state(slots=[slot])
        self.seed_state.reset_slots([None], [0], [slot])
        self.request_seeds[slot] = None

    def warmup_model_prefill(self, *, kv_cache=None, **kwargs):
        caches = self._resolve_caches(kv_cache, tuple(range(self.max_batch_size)))
        if len(caches) != len(self.caches) or any(a is not b for a, b in zip(caches, self.caches, strict=True)):
            raise ValueError("Warmup must use the cache storage bound to this generator")
        self.prepare()

    def warmup_model_decode(self, *, kv_cache=None, **kwargs):
        self.warmup_model_prefill(kv_cache=kv_cache, **kwargs)

    def read_decode_output(self, output, async_read=False, **kwargs):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        self.counters["readbacks"] += 1
        if not async_read:
            self.counters["synchronizations"] += 1
        host = DecodeSubmission(
            output.tensor.cpu(blocking=not async_read), output.slots, output.active_rows, output.is_tokens
        )
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, **kwargs):
        if isinstance(output, torch.Tensor):
            return output.unsqueeze(1)
        if output.is_tokens:
            values = ttnn.to_torch(ttnn.get_device_tensors(output.tensor)[0]).flatten()
            result = torch.full((len(output.slots), 1), -1, dtype=torch.long)
            for row in output.active_rows:
                result[row, 0] = values[output.slots[row]]
        else:
            values = (
                ttnn.to_torch(output.tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3))
                .reshape(-1, self.model.config.vocab_size)
                .float()
            )
            result = torch.zeros(len(output.slots), 1, self.model.config.vocab_size)
            for row in output.active_rows:
                result[row, 0] = values[output.slots[row]]
        return result

    def release_persistent_capture(self):
        self.close()

    def teardown(self):
        """Readiness P1 releases live traces before dropping the generator."""
        self.close()

    def close(self):
        self.prefill_checkpoints.invalidate(range(self.max_batch_size))
        self.sampler.reset_trace()
        for trace in (
            list(self.traces.values()) + list(self.sampling_traces.values()) + list(self.prefill_traces.values())
        ):
            ttnn.release_trace(self.mesh, trace)
            self._event("shutdown_release", trace_id=str(trace))
            self.counters["releases"] += 1
        self.prefill_traces.clear()
        self.traces.clear()
        self.sampling_traces.clear()
        # Reusing the warmed owner after explicit close may use eager state
        # copies, but must never call closures bound to released trace IDs.
        getattr(self.slot_mover, "copy_replays", {}).clear()
        if self.factored_decode is not None:
            self.factored_decode.invalidate_plan()


def build_generator(model_dir, mesh_device, **kwargs):
    checkpoint = kwargs.pop(
        "checkpoint",
        os.environ.get(
            "QWEN_CHECKPOINT",
            "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
        ),
    )
    packed = kwargs.pop(
        "packed_cache",
        os.environ.get("QWEN_PLE_CACHE", "/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide"),
    )
    layer_indices = kwargs.pop("layer_indices", None)
    precision = load_policy(model_dir, kwargs.pop("precision_config", None))
    model = QwenModel(
        checkpoint,
        mesh_device,
        packed_cache=packed,
        layer_indices=layer_indices,
        precision=precision,
        hf_config=kwargs.pop("hf_config", None),
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return QwenGenerator(model, tokenizer, **kwargs)
