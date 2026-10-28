# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the selected full-model Muse generator."""

import json
import os

import torch

import ttnn
from models.demos.muse_glimmer_30b_qb2.tt.generator import MuseGlimmerGenerator
from models.demos.muse_glimmer_30b_qb2.tt.model import REVISION, MuseGlimmerModel


class MuseGlimmerForConditionalGeneration:
    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": True,
        "supports_async_decode": True,
        "supports_device_penalties": False,
        "max_device_top_k": 32,
    }

    def __init__(self, model=None, batch_size=32, *, vllm_config=None, prefix=""):
        self.model = model
        self.batch_size = batch_size
        self.generator = None
        self.page_table_groups = None
        # Requests the device sampler cannot serve (top_k above 32, penalties,
        # logprobs, structured outputs) and prefill steps that carry an
        # intermediate chunk arrive without device sampling params and sample on
        # the host. MUSE_VLLM_ALLOW_HOST_SAMPLING=0 refuses them instead.
        self.allow_host_sampling = os.environ.get("MUSE_VLLM_ALLOW_HOST_SAMPLING", "1") == "1"
        event_path = os.environ.get("MUSE_VLLM_EVENTS")
        self.events = open(event_path, "a", buffering=1) if event_path else None
        self.audit_path = os.environ.get("MUSE_VLLM_AUDIT_JSON")
        self.audit = {"host_sampling_calls": 0, "device_sampling_calls": 0}
        self._sampling = dict(
            top_k=[1] * batch_size,
            top_p=[0.0] * batch_size,
            temperature=[1.0] * batch_size,
            seeds=list(range(batch_size)),
        )
        self._seed_bases = list(range(batch_size))

    # Registry inspection requires the upstream interface, while execution is
    # exclusively through TTModelLoader and its prefill/decode contract.
    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The full-model generator owns the LM head")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or max_seq_len != 131072 or not 1 <= max_batch_size <= 32:
            raise ValueError("Require TP4, 131072 context, and 1..32 sequences")
        if optimizations is not None:
            raise ValueError("Serving uses the datatype-sweep selected policy")
        indices = os.environ.get("MUSE_VLLM_LAYER_INDICES")
        indices = [int(i) for i in indices.split(",")] if indices else None
        model = MuseGlimmerModel(mesh_device, layer_indices=indices)
        obj = cls(model, max_batch_size)
        if os.environ.get("MUSE_PHASE_EVENTS"):
            from models.demos.muse_glimmer_30b_qb2.tt.benchmark_timing import install

            install(
                obj,
                dict(
                    module=__file__,
                    revision=REVISION,
                    layers=model.layer_indices,
                    precision=model.precision_config,
                    max_num_seqs=max_batch_size,
                    max_model_len=max_seq_len,
                ),
            )
        obj._event(
            dict(
                event="adapter_loaded",
                pid=os.getpid(),
                module=__file__,
                revision=REVISION,
                layers=model.layer_indices,
                precision=model.precision_config,
            )
        )
        print(f"MUSE_VLLM_ADAPTER {__file__} pid={os.getpid()} layers={model.layer_indices}", flush=True)
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        # Shared pooled capacity; the plugin adds per-sequence and sliding-group
        # reservations. This is not a per-request cap or a batch-1 assumption.
        return 131072

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        config = vllm_config.model_config.hf_config.text_config
        block_size = vllm_config.cache_config.block_size
        if block_size != 32:
            raise ValueError(f"Muse uses 32-token KV pages, got {block_size}")
        common = dict(
            block_size=32,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            dtype=vllm_config.model_config.dtype,
        )
        result = {}
        for i, kind in enumerate(config.layer_types):
            spec = (
                SlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if kind == "sliding_attention"
                else FullAttentionSpec(**common)
            )
            result[f"model.layers.{i}.self_attn"] = spec
        return result

    def _event(self, event):
        if self.events is not None:
            self.events.write(json.dumps(event, default=str) + "\n")

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        buffers = {}
        cache = []
        for idx in self.model.layer_indices:
            shape, _, tensor_idx = per_layer_specs[idx]
            if tuple(shape[1:]) != (1, 32, 128):
                raise ValueError(f"Unexpected serving KV shape {shape}")
            if tensor_idx not in buffers:
                buffers[tensor_idx] = [
                    ttnn.zeros(
                        shape,
                        device=self.model.mesh,
                        dtype=self.model.layers[len(cache)].kv_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                ]
            cache.append(buffers[tensor_idx])
        pages = cache[0][0].shape[0]
        self.generator = MuseGlimmerGenerator(
            self.model,
            batch_size=self.batch_size,
            num_pages=pages,
            kv_cache=cache,
            prepare=False,
            trace_reset=False,
            per_layer_page_tables=True,
            page_table_groups=self.page_table_groups,
            event_sink=self._event if self.events is not None else None,
        )
        self._event(
            dict(
                event="cache_allocated",
                specs=per_layer_specs,
                unique_buffers=len(buffers),
                physical_pages=pages,
                ownership="vllm",
                page_table_groups=self.page_table_groups,
            )
        )
        self.audit.update(
            physical_pages=pages,
            unique_cache_pairs=len(buffers),
            layers=self.model.layer_indices,
            precision=self.model.precision_config,
            page_table_groups=self.page_table_groups,
        )
        return cache

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, **kwargs):
        self.generator._check_cache(kv_cache)
        self.generator.prepare()

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device, **kwargs):
        self.warmup_model_prefill(kv_cache, enable_trace, can_sample_on_device)

    def _tables(self, page_table, page_tables_per_layer, slots=None):
        if page_tables_per_layer is None:
            raise ValueError("Hybrid serving requires per-layer page tables")
        tables = []
        for local_idx, idx in enumerate(self.model.layer_indices):
            value = page_tables_per_layer[idx]
            representative = self.generator._page_table_representatives[local_idx]
            if representative != local_idx:
                original = page_tables_per_layer[self.model.layer_indices[representative]]
                if value is not original and not torch.equal(value, original):
                    raise ValueError(f"Native page mappings disagree within group at layer {idx}")
                tables.append(tables[representative])
                continue
            old = self.generator._layer_page_hosts[local_idx]
            out = (
                old.clone()
                if slots is not None and old is not None
                else torch.zeros((self.batch_size, 4096), dtype=torch.int32)
            )
            if value.shape[1] > 4096:
                raise ValueError("Page table exceeds the supported context")
            if slots is not None:
                out[slots, : value.shape[1]] = value[: len(slots)].to(torch.int32)
            else:
                out[: value.shape[0], : value.shape[1]] = value.to(torch.int32)
            tables.append(out)
        return tables

    def _sampling_params(self, params, slots=None, positions=None):
        slots = list(range(self.batch_size)) if slots is None else slots

        def row(value, i):
            return value[i] if isinstance(value, (list, tuple, torch.Tensor)) else value

        for i, slot in enumerate(slots):
            temp, k, p = (row(getattr(params, key), i) for key in ("temperature", "top_k", "top_p"))
            greedy = temp == 0
            seed = row(params.seed, i)
            if seed is None:
                # Keep an unseeded stream's base through parameter-only updates.
                seed = self._seed_bases[slot] if positions is None else int(torch.randint(0, 2**30, ()).item())
            base = int(seed) % (2**31 - 131073)
            # Sampling1D increments its explicit seed once per emitted token.
            # On layout resets position provides a deterministic offset.
            offset = (
                max(0, int(positions[i]))
                if positions is not None
                else self._sampling["seeds"][slot] - self._seed_bases[slot]
            )
            self._sampling["top_k"][slot] = 1 if greedy else int(k)
            self._sampling["top_p"][slot] = 0.0 if greedy else float(p)
            self._sampling["temperature"][slot] = 1.0 if greedy else float(temp)
            self._sampling["seeds"][slot] = base + offset
            self._seed_bases[slot] = base
        return self._sampling

    def _remap_slots(self, remap):
        indices = torch.as_tensor(remap).flatten().tolist()
        if sorted(indices) != list(range(self.batch_size)):
            raise ValueError(f"Invalid slot remap {indices}")
        if indices == list(range(self.batch_size)):
            return
        self._sampling = {name: [values[i] for i in indices] for name, values in self._sampling.items()}
        self._seed_bases = [self._seed_bases[i] for i in indices]
        # Exact seed ledger: this copies known RNG state, never sampled tokens.
        self.generator._write(torch.tensor(self._sampling["seeds"], dtype=torch.int32), self.generator.sampler._seeds)

    def _check_sampling(self, params):
        if params is None and not self.allow_host_sampling:
            raise ValueError("Host sampling is disabled (MUSE_VLLM_ALLOW_HOST_SAMPLING=0)")
        if params is None:
            self.audit["host_sampling_calls"] += 1
            self._event(dict(event="host_sampling_compatibility"))
        else:
            self.audit["device_sampling_calls"] += 1
        return params is not None

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        enable_trace=True,
        page_tables_per_layer=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill options: {sorted(kwargs)}")
        slots = list(range(len(prompt_lens))) if empty_slots is None else list(empty_slots)
        # The plugin samples any step that carries an intermediate chunk on the
        # host, so a device-sampled prefill only holds final rows. Host-sampled
        # steps return logits for every row; the plugin drops intermediate ones.
        sample = self._check_sampling(sampling_params)
        starts = [0] * len(slots) if start_pos is None else torch.as_tensor(start_pos).flatten().tolist()
        # The plugin supplies the full prefix and absolute chunk end positions;
        # the generator API consumes only this chunk and its logical lengths.
        lengths = [int(end) - int(start) for end, start in zip(prompt_lens, starts)]
        chunks = torch.zeros((len(slots), max(lengths)), dtype=tokens.dtype)
        for row, (start, length) in enumerate(zip(starts, lengths)):
            chunks[row, :length] = tokens[row, start : start + length]
        tables = self._tables(page_table, page_tables_per_layer, slots)
        params = (
            self._sampling_params(sampling_params, slots, [int(end) - 1 for end in prompt_lens]) if sample else None
        )
        out = self.generator.prefill_forward(
            chunks,
            page_table=tables[0],
            page_tables_per_layer=tables,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=slots,
            start_pos=starts,
            sample=sample,
            sampling_params=params,
            sample_slots=slots,
            read_from_device=True,
        )
        self.generator._event("prefill_complete", slots=slots, sampled_slots=slots, sample=sample)
        if sample:
            for slot in slots:
                self._sampling["seeds"][slot] += 1
        return (out[slots].reshape(len(slots), 1), None) if sample else (out[slots], None)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported decode options: {sorted(kwargs)}")
        if not enable_trace:
            raise ValueError("Serving decode requires tracing")
        if reload_inputs and reload_page_table or reset_sampling_state and not reload_inputs:
            raise ValueError("Invalid v1 decode reload commands")
        sample = self._check_sampling(sampling_params)
        if slot_remap is not None and not reload_inputs:
            raise ValueError("A slot remap requires authoritative input reconstruction")
        if slot_remap is not None:
            self._remap_slots(slot_remap)
        if sample and reload_sampling_params:
            self._sampling_params(sampling_params, positions=start_pos if reset_sampling_state else None)
        elif sample and reset_sampling_state:
            self._sampling["seeds"] = [base + max(0, int(pos)) for base, pos in zip(self._seed_bases, start_pos)]
        if sample and (reload_sampling_params or reset_sampling_state):
            self.generator.set_sampling_params(**self._sampling)
        tables = self._tables(page_table, page_tables_per_layer) if reload_inputs or reload_page_table else None
        out = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=tables[0] if tables is not None else None,
            page_tables_per_layer=tables,
            kv_cache=kv_cache,
            sample=sample,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            read_from_device=False,
        )
        self.generator._event(
            "decode_submit", reload_inputs=reload_inputs, reload_page_table=reload_page_table, sample=sample
        )
        if sample:
            self._sampling["seeds"] = [seed + 1 for seed in self._sampling["seeds"]]
        result = (out, sample)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(result), is_tokens=sample)
        return result

    def read_decode_output(self, tt_out, async_read=False):
        tensor, is_tokens = tt_out
        # Token output is replicated. Transfer a single replica of the small
        # token vector; compatibility logits retain their TP vocabulary shards.
        tensor = ttnn.get_device_tensors(tensor)[0] if is_tokens else tensor
        host = tensor.cpu(blocking=not async_read, cq_id=0)
        self.generator.counters["token_readbacks" if is_tokens else "full_logits_readbacks"] += 1
        result = (host, is_tokens)
        return (result, [ttnn.record_event(self.model.mesh, 0)]) if async_read else result

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if ttnn.is_tensor_storage_on_device(tt_out[0]):
            # Synchronous decode hands over the device output unread.
            tt_out = self.read_decode_output(tt_out)
        host, tokens = tt_out
        if tokens:
            out = (
                ttnn.to_torch(ttnn.get_device_tensors(host)[0])
                .flatten()[: self.batch_size]
                .to(torch.int64)
                .reshape(-1, 1)
            )
        else:
            out = ttnn.to_torch(host, mesh_composer=ttnn.ConcatMeshToTensor(self.model.mesh, dim=-1))
            out = out.reshape(self.batch_size, 1, -1)[:, :, : self.model.vocab_size]
        return out, None

    def release_persistent_capture(self):
        if self.generator is not None:
            if self.audit_path:
                from pathlib import Path

                Path(self.audit_path).write_text(
                    json.dumps(
                        dict(
                            self.audit,
                            counters=dict(self.generator.counters),
                            capabilities=self.model_capabilities,
                            batch_size=self.batch_size,
                            allow_host_sampling=self.allow_host_sampling,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            self.generator.close()
        if self.events is not None:
            self.events.close()
            self.events = None
