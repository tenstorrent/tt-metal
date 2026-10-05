"""Serving-ready state ownership and canonical split sampling for K2 TP4."""

from __future__ import annotations

import math
import time
from collections import Counter
from contextlib import contextmanager

import torch

try:
    from readiness_check.contract import Generator
except ImportError:  # Serving images (e.g. TTI/CI) do not ship the bring-up runtime.
    from models.autoports.ifm_k2_horizon_7b.tt.readiness_contract import Generator

import ttnn
from models.autoports.ifm_k2_horizon_7b.tt.model import K2Model
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.seed_manager_1d import _hash_request_seed_to_device_seed
from models.common.sampling.token_history import history_program


class K2Generator(Generator):
    def __init__(self, mesh_device, **kwargs):
        unknown = kwargs.keys() - {
            "override_num_layers",
            "head_dtype",
            "head_fidelity",
            "head_split_size",
            "head_workers",
            "head_k",
            "decoder_policies",
            "precision_config",
        }
        if unknown:
            raise TypeError(f"Unsupported generator construction options: {sorted(unknown)}")
        self.mesh = mesh_device
        self.model = K2Model(
            mesh_device,
            override_num_layers=kwargs.get("override_num_layers"),
            head_dtype=kwargs.get("head_dtype"),
            head_fidelity=kwargs.get("head_fidelity"),
            head_split_size=kwargs.get("head_split_size", 8192),
            head_workers=kwargs.get("head_workers", 1),
            head_k=kwargs.get("head_k", 4),
            decoder_policies=kwargs.get("decoder_policies"),
            precision_config=kwargs.get("precision_config"),
        )
        self.tokenizer = self.model.tokenizer
        self.sampling_dtype = getattr(ttnn, self.model.precision_config["dtypes"]["sampling_values"])
        self.token_dtype = getattr(ttnn, self.model.precision_config["dtypes"]["token_ids"])
        self.kv_cache = None
        self.page_table = None
        self.capacity = 0
        self.batch_size = 0
        self.state = None
        self.counters = Counter()
        self.last_perf = {}
        self.prepared_prefills = set()
        self.prefill_state = None
        self.token_history = None
        self.history_index = self.model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.sampler = Sampling1D(
            self.model.padded_vocab,
            mesh_device,
            valid_vocab_size=self.model.vocab_size,
            max_batch_size=32,
            max_top_k=32,
            tt_ccl=self.model.layers[0].ccl,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
            num_gather_links=1,
        )
        self.sampler.load_device_buffers()
        self.k = self.model.upload(
            torch.ones(32, dtype=torch.int32), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.p = self.model.upload(torch.zeros(32), dtype=self.sampling_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.temp = self.model.upload(torch.ones(32), dtype=self.sampling_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.seed_values = torch.arange(32, dtype=torch.int32) + 123
        self.seeds = self.model.upload(self.seed_values, dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.prefill_tokens = self.model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def _refresh(self, dst, tensor, counter):
        src = ttnn.from_torch(
            tensor.contiguous(), dtype=dst.dtype, layout=dst.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(src, dst)
        self.counters[counter] += 1

    def _release_traces(self, *, drop_state=True, next_batch=1):
        # B1 generation interleaves prepared prefill (rows1/32) and decode.
        # External B2..32 decode needs only its fixed batch after prefill has
        # completed. Retaining all three buckets needlessly consumes L1 and
        # can overlap the head's static working buffers.
        pool = self.model.pool.tensors
        keep_rows = (1, 32) if next_batch == 1 else (next_batch,)
        obsolete = [key for key in pool if key[1][-2] not in keep_rows]
        if self.state is not None or self.prefill_state is not None or obsolete:
            ttnn.synchronize_device(self.mesh)
            self.counters["release_synchronizations"] += 1
        if self.prefill_state is not None and self.prefill_state["trace"] is not None:
            ttnn.release_trace(self.mesh, self.prefill_state["trace"])
            self.prefill_state["trace"] = None
        if self.state is not None:
            for trace in self.state["sample_traces"].values():
                ttnn.release_trace(self.mesh, trace)
            if self.state.get("trace") is not None:
                ttnn.release_trace(self.mesh, self.state["trace"])
            self.state["sample_traces"] = {}
            self.state["trace"] = None
            self.state.pop("logits", None)
            if drop_state:
                self.state = None
        # All layers retain this same pool object. Only release its obsolete
        # owners after both model and sampler traces have been retired. Native
        # cached CCL programs rebind buffer addresses on the next eager warmup.
        for key in obsolete:
            del pool[key]
        if obsolete:
            # Program-cache readiness does not preserve the persistent scratch
            # owned by a prepared prefill. Batch changes can retire its gather
            # buckets, including last-token-only buckets with prefill geometry.
            # Rewarm those paths before admitting another live decode trace.
            self.prepared_prefills.clear()
            if self.prefill_state is not None:
                self.prefill_state["prepared"] = False
        self.counters["collective_buckets_retired"] += len(obsolete)

    def close(self):
        self._release_traces()
        self.prefill_state = None
        self.sampler.release()

    def _ensure_owned_cache(self, batch, capacity):
        if self.kv_cache is None or self.batch_size != batch or self.capacity < capacity:
            self._release_traces(next_batch=batch)
            self.prefill_state = None
            self.kv_cache = None
            self.kv_cache, self.page_table = self.model.allocate_cache(batch_size=batch, capacity=capacity)
            self.capacity = self.page_table.shape[1] * self.model.page_size
            self.batch_size = batch

    def reset(self):
        # Reset only generator-owned cache. External cache ownership is never
        # inferred from a low-level call and is never silently zeroed or freed.
        if self.kv_cache is not None:
            for pair in self.kv_cache:
                for tensor in pair:
                    ttnn.mul(tensor, 0.0, output_tensor=tensor)
        if self.state:
            self._refresh(self.state["tokens"], torch.zeros(1, 1, 1, 32, dtype=torch.int32), "token_refreshes")
            self._refresh(self.state["positions"], torch.full((32,), -1, dtype=torch.int32), "position_refreshes")
            self._refresh(self.state["rope"], torch.zeros(1, 32, dtype=torch.int32), "rope_refreshes")
        self.seed_values = torch.arange(32, dtype=torch.int32) + 123
        self._refresh(self.seeds, self.seed_values, "seed_refreshes")
        self._refresh(self.history_index, torch.zeros(1, 1, 1, 32, dtype=torch.int32), "history_index_refreshes")

    def _ensure_token_history(self, count):
        """Allocate before capture. Row-major UINT32 preserves exact token IDs."""
        if self.token_history is None or self.token_history.shape[2] < count:
            self._release_traces(drop_state=False)
            self.token_history = ttnn.empty(
                (1, 1, max(128, math.ceil(count / 128) * 128), 32),
                dtype=self.token_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def _prepare_prefill_inputs(self, prompt, *, page_table=None, kv_cache=None, sample=False):
        """One bounded, exact-length prefill bucket; allocate before all captures.

        Longer/chunked and mixed-slot prefills retain the public
        eager path. Arbitrary logical lengths in this bucket use the same
        padding/attention implementation as that path.
        """
        page_table = self.page_table if page_table is None else page_table
        kv_cache = self.kv_cache if kv_cache is None else kv_cache
        cache = tuple(tuple(pair) for pair in kv_cache)
        key = (len(prompt), tuple(page_table.shape), tuple(tuple(id(t) for t in pair) for pair in cache), sample)
        if self.prefill_state is not None and self.prefill_state["key"] == key and self.prefill_state["prepared"]:
            return False
        self._release_traces(drop_state=False)
        self.prefill_state = None
        s = {
            "key": key,
            "trace": None,
            "prepared": False,
            "sample": sample,
            "tokens": self.model.upload(torch.tensor([prompt]), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
            "indices": self.model.upload(
                torch.arange(len(prompt)).reshape(1, -1), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "table": self.model.upload(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            "table_host": page_table.clone(),
            "plan": self.model.layers[0].prepare_prefill(seq_len=len(prompt)),
            "cache": cache,
        }
        self.prefill_state = s
        warm = self._prefill_step()
        # Both trace families share a persistent output owner. A retained
        # prefill-capture output would otherwise overlap decode's workspace.
        s["logits"] = ttnn.empty_like(warm)
        ttnn.copy(warm, s["logits"])
        ttnn.synchronize_device(self.mesh)
        s["prepared"] = True
        return True

    def _prefill_step(self):
        s = self.prefill_state
        try:
            return self.model.prefill_from_device(
                s["tokens"], s["indices"], plan=s["plan"], page_table=s["table"], kv_cache=s["cache"], last_only=True
            )
        finally:
            # AGMM scratch is produced and consumed within this prefill. In
            # particular layer1 switches BF16 QKV -> BFP8 MLP scratch. Retain
            # no capture-time owner into the separate decode trace; captured
            # commands own the workspace lifetime, just as for eager prefill.
            for layer in self.model.layers:
                layer._prefill_gather_buffer = None
                layer._prefill_gather_shape = None

    def _capture_prefill(self):
        with self._capture() as trace:
            transient = self._prefill_step()
            ttnn.copy(transient, self.prefill_state["logits"])
            del transient
            if self.prefill_state["sample"]:
                self._sample(
                    self.model.sampler_logits(self.prefill_state["logits"]),
                    strategy="split",
                    out_tokens=self.prefill_tokens,
                )
        self.prefill_state["trace"] = trace
        self.counters["prefill_captures"] += 1

    def _replay_prefill(self, prompt, *, page_table=None):
        s = self.prefill_state
        page_table = self.page_table if page_table is None else page_table
        self._refresh(s["tokens"], torch.tensor([prompt]), "prefill_token_refreshes")
        if not torch.equal(s["table_host"], page_table):
            self._refresh(s["table"], page_table, "prefill_page_table_refreshes")
            s["table_host"] = page_table.clone()
        if s["trace"] is None:
            # Prefill-only traffic has no real decode with which to prepare
            # compatible scratch. Keep it eager until that boundary exists.
            transient = self._prefill_step()
            ttnn.copy(transient, s["logits"])
            del transient
        else:
            ttnn.execute_trace(self.mesh, s["trace"], cq_id=0, blocking=False)
            self.counters["prefill_replays"] += 1
        return s["logits"]

    def _record_token(self):
        ttnn.generic_op([self.state["tokens"], self.history_index, self.token_history], self.state["history_program"])

    def _read_logits(self, logits):
        self.counters["logit_readbacks"] += 1
        if logits.shape[-2] > 1:
            # The generic mesh composer is costly for wide vocabulary rows.
            # Preserve the same TP4 shard order and
            # BF16 values using PyTorch's contiguous row copies instead.
            # Decode supplies an already-completed host tensor. Other callers
            # retain the same single blocking mesh read as ttnn.to_torch.
            if ttnn.is_tensor_storage_on_device(logits):
                logits = logits.cpu()
            shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(logits)]
            return torch.cat(shards, dim=-1)[..., : self.model.vocab_size]
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))[
            ..., : self.model.vocab_size
        ]

    def _read_tokens(self, *, batch):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.state["tokens"])[0]).reshape(-1)[:batch].long()

    def _validate_cache(self, table, cache):
        if not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[1] % 4:
            raise ValueError("Host page table must be rank 2 and cover whole K128 read windows (4 pages)")
        if table.dtype not in (torch.int32, torch.int64) or table.numel() == 0:
            raise ValueError("Page table must contain integer page indices")
        if len(cache) != self.model.num_layers:
            raise ValueError("One K/V pair is required per layer")
        pages = cache[0][0].shape[0]
        if (table < 0).any() or (table >= pages).any():
            raise ValueError("Page-table indices exceed the supplied physical cache")
        for layer, pair in zip(self.model.layers, cache):
            if len(pair) != 2 or any(tuple(t.shape) != (pages, 2, 32, 128) or t.dtype != layer.kv_dtype for t in pair):
                raise ValueError("External cache must preserve TP4 local-head shape and the selected KV dtype")

    def read_decode_output(self, *, async_read=False):
        """Queue token output on CQ0 after sampling; no token feedback through host."""
        self.counters["token_readbacks"] += 1
        return ttnn.get_device_tensors(self.state["tokens"])[0].cpu(blocking=not async_read)

    @staticmethod
    def process_decode_output_host(output, *, batch_size=1):
        """For async reads, caller must synchronize CQ0 before consuming output."""
        return ttnn.to_torch(output).reshape(-1)[:batch_size].long()

    def configure_sampling(self, *, top_k=1, top_p=0.0, temperature=1.0, seed=None):
        """Refresh persistent per-slot sampler state once at a request boundary."""
        if not isinstance(top_k, int) or not 1 <= top_k <= 32:
            raise ValueError("Common sampler supports integer top_k 1..32")
        if not math.isfinite(top_p) or not 0 <= top_p <= 1:
            raise ValueError("top_p must be finite and in [0, 1]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if temperature == 0:
            top_k, temperature, top_p = 1, 1.0, 0.0
        self._refresh(self.k, torch.full((32,), top_k, dtype=torch.int32), "sampling_param_refreshes")
        self._refresh(self.p, torch.full((32,), top_p), "sampling_param_refreshes")
        self._refresh(self.temp, torch.full((32,), 1.0 / temperature), "sampling_param_refreshes")
        if seed is not None:
            # Use the common bounded request hash: excludes the native sentinel
            # and reserves signed-positive increment headroom for full context.
            self.seed_values = torch.arange(32, dtype=torch.int32) + _hash_request_seed_to_device_seed(seed, 0)
            self._refresh(self.seeds, self.seed_values, "seed_refreshes")
        self.sampling_is_greedy = top_k == 1 or top_p == 0

    def configure_sampling_batch(self, *, top_k, top_p, temperature, seeds, positions):
        """Configure the same split sampler for independent serving slots.

        Absolute positions align the request RNG stream on scheduler resets;
        steady replay advances both position and seed exactly once on device.
        """
        count = len(positions)

        def rows(value):
            return list(value) if isinstance(value, (list, tuple, torch.Tensor)) else [value] * count

        ks, ps, ts, ss = map(rows, (top_k, top_p, temperature, seeds))
        if any(len(v) != count for v in (ks, ps, ts, ss)) or not 1 <= count <= 32:
            raise ValueError("Sampling parameters must match the serving slots")
        k = torch.ones(32, dtype=torch.int32)
        p, temp = torch.zeros(32), torch.ones(32)
        seed_values = torch.arange(32, dtype=torch.int32) + 123
        for i, (ki, pi, ti, si, pos) in enumerate(zip(ks, ps, ts, ss, positions)):
            if ti == 0 or ki == 1:
                ki, pi, ti = 1, 0.0, 1.0
            if not 1 <= ki <= 32 or not 0 <= pi <= 1 or not math.isfinite(ti) or ti <= 0:
                raise ValueError("Device sampler requires top_k 1..32, top_p 0..1, finite positive temperature")
            k[i], p[i], temp[i] = ki, pi, 1.0 / ti
            request_seed = torch.randint(0, 2**31 - 1, ()).item() if si is None else int(si)
            seed_values[i] = _hash_request_seed_to_device_seed(request_seed, 0) + max(0, pos)
        for dst, value in ((self.k, k), (self.p, p), (self.temp, temp)):
            self._refresh(dst, value, "sampling_param_refreshes")
        self.seed_values = seed_values
        self._refresh(self.seeds, seed_values, "seed_refreshes")
        self.sampling_is_greedy = bool(((k == 1) | (p == 0)).all())

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        start_pos=None,
        active_slots=None,
        sampling_mode="device",
        strategy="split",
        read_from_device=True,
        trace_prefill=False,
        **kwargs,
    ):
        """Mixed prompts and caller-owned pages/caches; arbitrary logical lengths.

        Slots index rows of the supplied page table; caller padding is ignored.
        All-logits is an explicit diagnostic/host-sampling readback boundary.
        Default returns sampled tokens [batch] in request row order. The explicit
        device_logits mode exposes a list of TP-sharded terminal tensors for
        advanced device consumers; host mode returns [batch,1,vocab] logits.
        """
        if not isinstance(page_table, torch.Tensor):
            raise TypeError("prefill page_table must be a host int tensor; decode also caches a stable device copy")
        if kwargs or sampling_mode not in ("device", "host", "device_logits"):
            raise ValueError("Unsupported prefill options or sampling mode")
        self._validate_cache(page_table, kv_cache)
        batch = len(prompt_lens)
        if not 1 <= batch <= 32:
            raise ValueError("Prefill supports 1..32 request rows")
        starts = [0] * batch if start_pos is None else list(start_pos)
        slots = list(range(batch)) if active_slots is None else list(active_slots)
        if tokens.ndim != 2 or tokens.shape[0] != batch or len(starts) != batch or len(slots) != batch:
            raise ValueError("tokens, prompt_lens, start_pos and active_slots must agree")
        if len(set(slots)) != len(slots) or any(s < 0 or s >= page_table.shape[0] for s in slots):
            raise ValueError("active_slots must be distinct valid page-table rows")
        for row, length in enumerate(prompt_lens):
            start = int(starts[row])
            if length < 1 or start < 0 or start + length > self.model.context:
                raise ValueError("Prompt must fit the HF context")
            if tokens.shape[1] < length or (start + length + 31) // 32 > page_table.shape[1]:
                raise ValueError("Prompt token/page-table capacity is too short")
            if (tokens[row, :length] < 0).any() or (tokens[row, :length] >= self.model.vocab_size).any():
                raise ValueError("Prompt contains an invalid token ID")
        if sampling_mode == "device" and (
            strategy not in ("split", "argmax")
            or strategy == "argmax"
            and not getattr(self, "sampling_is_greedy", True)
        ):
            raise ValueError("Prefill sampling strategy must agree with configured sampling parameters")

        if (
            trace_prefill
            and batch == 1
            and starts[0] == 0
            and prompt_lens[0] <= 4096
            and not return_all_logits
            and sampling_mode == "device"
            and strategy == "split"
        ):
            prompt = tokens[0, : prompt_lens[0]].tolist()
            table = page_table[slots[0] : slots[0] + 1].int().contiguous()
            incompatible = any(key[1][-2] not in (1, 32) for key in self.model.pool.tensors)
            if incompatible or self.state is not None and self.state["batch"] != 1:
                self._release_traces(drop_state=False)
            prepared_now = self._prepare_prefill_inputs(prompt, page_table=table, kv_cache=kv_cache, sample=True)
            s = self.prefill_state
            logits = s["logits"] if prepared_now else self._replay_prefill(prompt, page_table=table)
            if s["trace"] is None:
                self._sample(self.model.sampler_logits(logits), strategy="split", out_tokens=self.prefill_tokens)
                self.counters["prefill_prepared_eager"] += 1
            if not read_from_device:
                return self.prefill_tokens
            self.counters["token_readbacks"] += 1
            return ttnn.to_torch(ttnn.get_device_tensors(self.prefill_tokens)[0]).reshape(-1)[:1].long()
        signature = self._prefill_signature(
            prompt_lens, starts, slots, page_table, kv_cache, return_all_logits, sampling_mode
        )
        incompatible_scratch = any(key[1][-2] not in (1, 32) for key in self.model.pool.tensors)
        batched_decode = self.state is not None and self.state["batch"] != 1
        if signature not in self.prepared_prefills or incompatible_scratch or batched_decode:
            # New eager programs can allocate immutable DRAM kernel binaries.
            # Finish and invalidate trace handles before admitting that variant;
            # keep stable inputs, cache and weights when their shapes still fit.
            # Program readiness does not imply L1 workspace availability: a
            # later B2 decode can add scratch that overlaps a prepared prefill's
            # static buffers. Retire it even on a known-signature request.
            # Also bound short all-logits buckets when no trace is live.
            self._release_traces(drop_state=False)
        with self._prefill_program_guard(signature):
            try:
                outputs = []
                for row, (length, start, slot) in enumerate(zip(prompt_lens, starts, slots)):
                    length, start = int(length), int(start)
                    if length < 1 or start < 0 or start + length > self.model.context:
                        raise ValueError("Prompt must fit the HF context")
                    if tokens.shape[1] < length or (start + length + 31) // 32 > page_table.shape[1]:
                        raise ValueError("Prompt token/page-table capacity is too short")
                    table = self.model.upload(
                        page_table[slot : slot + 1].int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    parts = []
                    step = self.model.layers[0].chunk_size
                    for offset in range(0, length, step):
                        end = min(offset + step, length)
                        # Early chunks fill cache only. Terminal logits for them are not
                        # materialized unless the caller explicitly requests all logits.
                        logits = self.model.prefill_chunk(
                            tokens[row, offset:end].tolist(),
                            start_pos=start + offset,
                            page_table=table,
                            kv_cache=kv_cache,
                            last_only=not return_all_logits,
                            cache_only=not return_all_logits and end < length,
                        )
                        if return_all_logits:
                            parts.append(self._read_logits(logits).reshape(1, end - offset, -1))
                    outputs.append(torch.cat(parts, dim=1) if return_all_logits else logits)
            finally:
                # Fused prefill gather scratch is never consumed by decode. Drop its
                # Python owner after eager consumers have been enqueued, so no changing
                # 4096/tail allocation survives into trace replay. The CQ owns enqueued
                # commands; model weights/cache/collective decode pools remain stable.
                for layer in self.model.layers:
                    layer._prefill_gather_buffer = None
                    layer._prefill_gather_shape = None
            self.prepared_prefills.add(signature)
            if return_all_logits:
                width = max(prompt_lens)
                return torch.cat(
                    [torch.nn.functional.pad(out, (0, 0, 0, width - out.shape[1])) for out in outputs], dim=0
                )
            if sampling_mode == "host":
                return torch.cat([self._read_logits(out).reshape(1, 1, -1) for out in outputs], 0)
            if sampling_mode == "device":
                if strategy not in ("split", "argmax") or (
                    strategy == "argmax" and not getattr(self, "sampling_is_greedy", True)
                ):
                    raise ValueError("Prefill sampling strategy must agree with configured sampling parameters")
                logits = ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
                self._sample(self.model.sampler_logits(logits), strategy=strategy, out_tokens=self.prefill_tokens)
                if not read_from_device:
                    return self.prefill_tokens
                self.counters["token_readbacks"] += 1
                return ttnn.to_torch(ttnn.get_device_tensors(self.prefill_tokens)[0]).reshape(-1)[:batch].long()
            return outputs

    @staticmethod
    def _prefill_signature(lengths, starts, slots, table, cache, all_logits, mode="device_logits"):
        return (
            tuple(map(int, lengths)),
            tuple(map(int, starts)),
            tuple(slots),
            tuple(table.shape),
            tuple((tuple(t.shape), str(t.dtype)) for pair in cache for t in pair),
            all_logits,
            mode,
        )

    def prefill_logits(self, prompt_token_ids):
        self._ensure_owned_cache(1, len(prompt_token_ids))
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def _sample(self, logits, *, strategy, seeds=None, out_tokens=None, record_history=False):
        seeds = self.seeds if seeds is None else seeds
        kwargs = {} if strategy == "argmax" else {"k": self.k, "p": self.p, "temp": self.temp, "seeds": seeds}
        result = self.sampler.decode_forward(
            logits, tt_out_tok=self.state["tokens"] if out_tokens is None else out_tokens, **kwargs
        )
        ttnn.plus_one(seeds)
        if record_history:
            self._record_token()
        return result

    def _model_step(self):
        s = self.state
        logits = self.model.decode(
            s["tokens"], s["positions"], s["rope"], page_table=s["table"], kv_cache=s["cache"], batch_size=s["batch"]
        )
        ttnn.plus_one(s["positions"], skip_negative_entries=True)
        ttnn.plus_one(s["rope"])
        return logits

    def _prepare_state(self, tokens, positions, *, page_table, kv_cache, strategy="split"):
        self._validate_cache(page_table, kv_cache)
        batch = len(positions)
        if not 1 <= batch <= 32:
            raise ValueError("Decode supports 1..32 fixed slots")
        if (positions >= self.model.context).any() or (positions < -1).any():
            raise ValueError("Decode positions must be -1 (inactive) or within context")
        if tokens.numel() != batch or page_table.shape[0] != batch:
            raise ValueError("Decode tokens, positions and page-table rows must match fixed batch slots")
        if (tokens < 0).any() or (tokens >= self.model.vocab_size).any():
            raise ValueError("Decode requires valid token IDs, including inactive row placeholders")
        if (positions >= page_table.shape[1] * self.model.page_size).any():
            raise ValueError("Decode position exceeds the supplied page-table capacity")
        table = page_table.int().contiguous()
        key = (batch, tuple(table.shape), tuple(id(t) for pair in kv_cache for t in pair))
        new = self.state is None or self.state["key"] != key
        token_host = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        token_host.reshape(-1)[:batch] = tokens.reshape(-1).int()
        pos_host = torch.full((32,), -1, dtype=torch.int32)
        pos_host[:batch] = positions.int()
        rope_host = pos_host.clamp_min(0).reshape(1, 32)
        if new:
            self._release_traces(next_batch=batch)
            self.state = {
                "key": key,
                "batch": batch,
                "cache": kv_cache,
                "sample_traces": {},
                "trace": None,
                "tokens": self.model.upload(token_host, dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "positions": self.model.upload(pos_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                "rope": self.model.upload(rope_host, dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "table": self.model.upload(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                "table_host": table.clone(),
            }
            self.counters.update(
                {"token_refreshes": 1, "position_refreshes": 1, "rope_refreshes": 1, "page_table_refreshes": 1}
            )
        else:
            self._refresh(self.state["tokens"], token_host, "token_refreshes")
            self._refresh(self.state["positions"], pos_host, "position_refreshes")
            self._refresh(self.state["rope"], rope_host, "rope_refreshes")
            self.refresh_page_table(table)
        if self.state["trace"] is None:
            # Same-owner requests may follow eager all-logits work after a
            # trace-only invalidation. Retire its tail buckets before capture.
            self._release_traces(drop_state=False, next_batch=batch)
            print("K2 WARM MODEL", flush=True)
            warm = self._model_step()
            print("K2 WARM SAMPLES", flush=True)
            # Both graph variants are ready before either trace becomes live.
            warm_seeds = self.model.upload(self.seed_values, dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if self.token_history is not None:
                self.state["history_program"] = history_program(
                    self.state["tokens"], self.history_index, self.token_history
                )
                self._record_token()
            self._sample(warm, strategy="split", seeds=warm_seeds)
            self._sample(warm, strategy="argmax", seeds=warm_seeds)
            ttnn.synchronize_device(self.mesh)
            del warm, warm_seeds
            self._refresh(self.state["tokens"], token_host, "token_refreshes")
            self._refresh(self.state["positions"], pos_host, "position_refreshes")
            self._refresh(self.state["rope"], rope_host, "rope_refreshes")
            print("K2 CAPTURE MODEL", flush=True)
            with self._capture() as trace:
                logits = self._model_step()
            self.state.update(trace=trace, logits=logits)
            print("K2 CAPTURE SAMPLES", flush=True)
            self._capture_sampler("split")
            self._capture_sampler("argmax")
            if self.token_history is not None:
                self._capture_sampler("split", record_history=True)
                self._capture_sampler("argmax", record_history=True)

        # Bind/capture only after a real caller-owned decode has warmed its
        # scratch and restored authoritative inputs. Capture does not execute
        # the previous prompt or sample, and all retained owners predate traces.
        prefill = self.prefill_state
        if (
            prefill is not None
            and prefill["prepared"]
            and prefill["trace"] is None
            and batch == 1
            and prefill["key"][2] == tuple(tuple(id(t) for t in pair) for pair in kv_cache)
        ):
            self.mesh.set_program_cache_misses_allowed(False)
            try:
                self._capture_prefill()
            except BaseException:
                prefill["prepared"] = False
                self._release_traces(drop_state=False)
                raise
            finally:
                self.mesh.set_program_cache_misses_allowed(True)

    def refresh_page_table(self, table):
        if self.state["table_host"] is None or not torch.equal(table, self.state["table_host"]):
            self._refresh(self.state["table"], table.int(), "page_table_refreshes")
            self.state["table_host"] = table.clone()

    @contextmanager
    def _capture(self):
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            try:
                yield trace
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        except BaseException:
            ttnn.release_trace(self.mesh, trace)
            raise

    @contextmanager
    def _prefill_program_guard(self, signature):
        live = self.state is not None and self.state["trace"] is not None
        if live:
            self.mesh.set_program_cache_misses_allowed(False)
        try:
            yield
        except BaseException:
            # Partial eager work may have modified KV. Abort the request and
            # invalidate handles before allowing any later preparation/retry.
            self.prepared_prefills.discard(signature)
            if live:
                self._release_traces(drop_state=False)
            raise
        finally:
            if live:
                self.mesh.set_program_cache_misses_allowed(True)

    def _capture_sampler(self, strategy, *, record_history=False):
        with self._capture() as trace:
            self._sample(self.state["logits"], strategy=strategy, record_history=record_history)
        self.state["sample_traces"][strategy + ("_history" if record_history else "")] = trace

    def replay(self, *, strategy="split", sample=True, record_history=False):
        if strategy not in ("split", "argmax"):
            raise ValueError("strategy must be split or argmax")
        if sample and strategy == "argmax" and not getattr(self, "sampling_is_greedy", True):
            raise ValueError("argmax requires greedy sampling parameters")
        ttnn.execute_trace(self.mesh, self.state["trace"], cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if sample:
            key = strategy + ("_history" if record_history else "")
            ttnn.execute_trace(self.mesh, self.state["sample_traces"][key], cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            if record_history:
                self.counters["history_writes"] += 1

    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        *,
        page_table,
        kv_cache,
        sampling_mode="device",
        strategy="split",
        read_from_device=True,
        **kwargs,
    ):
        """Explicit scheduler inputs, or continue the established device state.

        Omit both tokens/start_pos for steady-state token feedback. Supplying
        them is an explicit scheduler refresh, including fixed inactive rows -1.
        """
        if kwargs or sampling_mode not in ("device", "host"):
            raise ValueError("Unsupported decode options or sampling mode")
        if tokens is None and start_pos is None:
            if self.state is None or self.state["trace"] is None:
                raise ValueError("Establish decode state with explicit tokens and positions first")
            if (
                tuple(page_table.shape) != self.state["key"][1]
                or tuple(id(t) for pair in kv_cache for t in pair) != self.state["key"][2]
            ):
                raise ValueError("Changed cache/slot shape requires explicit scheduler inputs")
            self._validate_cache(page_table, kv_cache)
            self.refresh_page_table(page_table)
        elif tokens is None or start_pos is None:
            raise ValueError("Supply both tokens and start_pos, or neither")
        else:
            self._prepare_state(tokens, start_pos, page_table=page_table, kv_cache=kv_cache, strategy=strategy)
        self.replay(strategy=strategy, sample=sampling_mode == "device")
        if not read_from_device:
            return self.state["tokens"] if sampling_mode == "device" else self.state["logits"]
        if sampling_mode == "host":
            return self._read_logits(self.state["logits"]).reshape(32, -1)[: self.state["batch"]]
        return self._read_tokens(batch=self.state["batch"])

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        sampling_mode="device",
        strategy="split",
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=123,
        token_output="buffered",
        trace_prefill=True,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("K2 generation requires traced decode")
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be device or explicit host compatibility")
        if strategy not in ("split", "argmax"):
            raise ValueError("strategy must be split or argmax")
        if token_output not in ("buffered", "per_token"):
            raise ValueError("token_output must be buffered or per_token")
        if max_new_tokens < 1:
            return []
        if len(prompt_token_ids) + max_new_tokens - 1 > self.model.context:
            raise ValueError("Prompt plus generation exceeds HF context")
        if kwargs:
            raise TypeError(f"Unsupported generation options: {sorted(kwargs)}")
        if not prompt_token_ids:
            raise ValueError("Prompt must contain at least one token")
        if any(token < 0 or token >= self.model.vocab_size for token in prompt_token_ids):
            raise ValueError("Prompt token ID is outside the model vocabulary")
        if strategy == "argmax" and temperature != 0 and top_k != 1 and top_p != 0:
            raise ValueError("argmax requires greedy sampling parameters")
        if sampling_mode == "host" and temperature != 0 and top_k != 1 and top_p != 0:
            raise ValueError("Host compatibility supports greedy sampling; use low-level logits for other samplers")
        setup_start = time.perf_counter()
        record_history = token_output == "buffered" and sampling_mode == "device" and next_input is None
        if record_history:
            self._ensure_token_history(max_new_tokens)
        self._ensure_owned_cache(1, min(self.model.context, len(prompt_token_ids) + max_new_tokens))
        use_prefill_trace = trace_prefill and len(prompt_token_ids) <= 4096
        signature = self._prefill_signature([len(prompt_token_ids)], [0], [0], self.page_table, self.kv_cache, False)
        key = (1, tuple(self.page_table.shape), tuple(id(t) for pair in self.kv_cache for t in pair))
        needs_setup = self.state is None or self.state["key"] != key or self.state["trace"] is None
        if use_prefill_trace:
            self._prepare_prefill_inputs(prompt_token_ids)
            needs_setup = needs_setup or self.state is None or self.state["trace"] is None
        if needs_setup or signature not in self.prepared_prefills:
            # Compile cache reset and exact prefill variants before tracing.
            self.reset()
            self.configure_sampling(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
            if signature not in self.prepared_prefills and not use_prefill_trace:
                warm_prefill = self.prefill_forward(
                    torch.tensor([prompt_token_ids]),
                    page_table=self.page_table,
                    kv_cache=self.kv_cache,
                    prompt_lens=[len(prompt_token_ids)],
                    sampling_mode="device_logits",
                )
                ttnn.synchronize_device(self.mesh)
                del warm_prefill
            self._prepare_state(
                torch.zeros(1, 1, dtype=torch.int32),
                torch.tensor([min(len(prompt_token_ids), self.model.context - 1)]),
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                strategy=strategy,
            )
            self.prepared_prefills.add(signature)
        if use_prefill_trace and self.prefill_state["trace"] is None:
            self._capture_prefill()
        ttnn.synchronize_device(self.mesh)
        start = time.perf_counter()
        setup_seconds = start - setup_start
        # Warmed TTFT includes request cache reset, parameter/input refresh,
        # prefill, final projection, sampling and first-token readback.
        self.reset()
        self.configure_sampling(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        self._prepare_state(
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([min(len(prompt_token_ids), self.model.context - 1)]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            strategy=strategy,
        )
        if use_prefill_trace:
            logits = self._replay_prefill(prompt_token_ids)
        else:
            logits = self.prefill_forward(
                torch.tensor([prompt_token_ids]),
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                prompt_lens=[len(prompt_token_ids)],
                sampling_mode="device_logits",
            )[0]
        if sampling_mode == "host":
            token = int(self._read_logits(logits).reshape(-1).argmax())
            self._refresh(
                self.state["tokens"], torch.tensor([token] + [0] * 31).reshape(1, 1, 1, 32), "token_refreshes"
            )
        else:
            # Prefill sampling is outside the steady-state decode traces.
            logits = self.model.sampler_logits(logits)
            self._sample(logits, strategy=strategy, record_history=record_history)
            token = int(self._read_tokens(batch=1)[0])
        del logits
        first_sampled_token = token
        result = [token]
        ttft = time.perf_counter() - start
        before = self.counters.copy()
        begin = time.perf_counter()
        pending = []
        for step in range(max_new_tokens - 1):
            if next_input is not None:
                forced = next_input(step, result[-1])
                self._refresh(
                    self.state["tokens"], torch.tensor([forced] + [0] * 31).reshape(1, 1, 1, 32), "token_refreshes"
                )
            self.replay(strategy=strategy, sample=sampling_mode == "device", record_history=record_history)
            if sampling_mode == "host":
                token = int(self._read_logits(self.state["logits"])[0, 0, 0].argmax())
                self._refresh(
                    self.state["tokens"], torch.tensor([token] + [0] * 31).reshape(1, 1, 1, 32), "token_refreshes"
                )
            else:
                if next_input is None:
                    if not record_history:
                        pending.append(self.read_decode_output(async_read=True))
                    continue
                token = int(self._read_tokens(batch=1)[0])
            result.append(token)
        if pending:
            ttnn.synchronize_device(self.mesh)
            self.counters["output_synchronizations"] += 1
            result.extend(int(self.process_decode_output_host(item)[0]) for item in pending)
        if record_history:
            self.counters["history_readbacks"] += 1
            # One caller-visible boundary after the final replay. Read the
            # already bounded allocation without compiling a new slice op.
            history = ttnn.to_torch(ttnn.get_device_tensors(self.token_history)[0])
            result = history.reshape(-1, 32)[:max_new_tokens, 0].long().tolist()
        if next_input is not None:
            next_input(max_new_tokens - 1, result[-1])
        elapsed = time.perf_counter() - begin
        self.last_perf = {
            "prompt_len": len(prompt_token_ids),
            "generation_len": max_new_tokens,
            "first_sampled_token": first_sampled_token,
            "preparation_seconds": setup_seconds,
            "ttft_seconds": ttft,
            "decode_seconds": elapsed,
            "decode_tokens_per_second_per_user": (max_new_tokens - 1) / elapsed,
            "sampling_mode": sampling_mode,
            "strategy": strategy,
            "teacher_forcing": next_input is not None,
            "token_output": "buffered" if record_history else "per_token",
            "prefill_execution": "trace" if use_prefill_trace else "eager_chunks",
            "steady_state_counters": dict(self.counters - before),
        }
        return result


def build_generator(model_dir, mesh_device, **kwargs):
    return K2Generator(mesh_device, **kwargs)
