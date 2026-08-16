# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM adapter for ``google/gemma-4-26B-A4B-it``.

vLLM owns paged attention K/V and block tables.  Model execution, precision,
traced decode, and split on-device sampling remain owned by ``Gemma4Generator``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator, build_generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT, SLIDING_CACHE_TOKENS, FullModelState
from models.common.sampling import format_sampling_params

MODEL_ROOT = Path(__file__).resolve().parents[1]
PRECISION_CONFIG = MODEL_ROOT / "doc" / "datatype_sweep" / "selected_precision_config.json"
MAX_MODEL_LEN = 262_144
MAX_TOKENS_ALL_USERS = 262_144


class Gemma4ForCausalLM(nn.Module):
    """vLLM interface translation over the full-model generator."""

    decode_input_update_contract = 1
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "state_slots_are_stateless": True,
    }

    def __init__(
        self,
        vllm_config: Any = None,
        prefix: str = "",
        *,
        generator: Gemma4Generator | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del vllm_config, prefix
        self.generator = generator
        self.model = None if generator is None else generator.model
        self.mesh_device = None if generator is None else generator.mesh_device
        self.max_batch_size = 0 if generator is None else generator.model.max_batch_size
        self.max_seq_len = 0 if generator is None else generator.model.max_seq_len
        self._serving_state: FullModelState | None = None
        self._last_page_tables: list[torch.Tensor] | None = None
        self._page_table_refreshes = 0
        self._decode_ready = False
        self._unseeded_epoch = 0

    # These three methods declare vLLM's standard text-generation protocol for
    # registry introspection. The TT worker calls the explicit prefill/decode
    # interface below, so entering them would be a backend wiring error.
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TT Gemma4 uses the TT worker prefill/decode interface")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        raise NotImplementedError("TT Gemma4 uses the TT worker prefill/decode interface")

    def compute_logits(self, hidden_states: torch.Tensor):
        raise NotImplementedError("TT Gemma4 terminal logits are produced by Gemma4Generator")

    @classmethod
    def get_max_tokens_all_users(cls, **_: Any) -> int:
        # The context contract proves one aggregate full-attention budget of
        # 262,144 tokens. Concurrency shares that pool without reducing the
        # per-request max_model_len advertised by vLLM.
        return MAX_TOKENS_ALL_USERS

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        hf_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        specs = {}
        for layer_idx, layer_type in enumerate(hf_config.layer_types):
            name = f"model.layers.{layer_idx}.self_attn"
            if layer_type == "sliding_attention":
                specs[name] = SlidingWindowSpec(
                    block_size=64,
                    num_kv_heads=8,
                    head_size=256,
                    dtype=dtype,
                    sliding_window=SLIDING_CACHE_TOKENS,
                )
            elif layer_type == "full_attention":
                specs[name] = FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=2,
                    head_size=512,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"unsupported Gemma4 layer type {layer_type!r} at layer {layer_idx}")
        return specs

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **_: Any,
    ) -> "Gemma4ForCausalLM":
        del optimizations
        if tt_data_parallel != 1:
            raise NotImplementedError("Gemma4 autoport supports one TP4 model replica")
        if not 1 <= int(max_batch_size) <= DECODE_SLOT_COUNT:
            raise ValueError(f"max_num_seqs must be in [1, {DECODE_SLOT_COUNT}]")
        if not 1 <= int(max_seq_len) <= MAX_MODEL_LEN:
            raise ValueError(f"max_model_len must be in [1, {MAX_MODEL_LEN}]")
        model_path = getattr(hf_config, "_name_or_path", None) or "google/gemma-4-26B-A4B-it"
        generator = build_generator(
            MODEL_ROOT,
            mesh_device,
            model_path=model_path,
            max_seq_len=int(max_seq_len),
            max_batch_size=int(max_batch_size),
            num_layers=n_layers,
            sampling_mode="device",
            # Preserve the selected autoport policy by default while honoring
            # the same documented override as non-vLLM generator construction.
            precision_config_path=os.getenv("GEMMA4_PRECISION_CONFIG") or PRECISION_CONFIG,
            create_kv_cache=False,
        )
        return cls(generator=generator)

    def _require_generator(self) -> Gemma4Generator:
        if self.generator is None:
            raise RuntimeError("Gemma4 vLLM model has not been initialized")
        return self.generator

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate the exact vLLM-sized per-layer K/V tensors and bind them."""
        gen = self._require_generator()
        if len(per_layer_specs) != gen.model.num_layers:
            raise ValueError(f"vLLM requested {len(per_layer_specs)} layers; TT built {gen.model.num_layers}")
        cache = []
        page_tables = []
        for model_spec, (shape, _torch_dtype, _tensor_idx) in zip(gen.model.cache_specs, per_layer_specs):
            shape = tuple(int(value) for value in shape)
            expected = (model_spec.local_kv_heads, model_spec.block_size, model_spec.head_dim)
            if shape[1:] != expected:
                raise ValueError(f"layer {model_spec.layer_idx} cache geometry {shape[1:]} != {expected}")
            cache.append(
                tuple(
                    ttnn.zeros(
                        shape,
                        dtype=gen.model.kv_cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                )
            )
            table_width = math.ceil(
                (SLIDING_CACHE_TOKENS if model_spec.layer_type == "sliding_attention" else self.max_seq_len)
                / model_spec.block_size
            )
            page_tables.append(
                ttnn.from_torch(
                    torch.zeros((DECODE_SLOT_COUNT, table_width), dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        self._serving_state = FullModelState(
            kv_cache=cache,
            page_tables=page_tables,
            cache_specs=gen.model.cache_specs,
            max_batch_size=self.max_batch_size,
            slot_context_lengths=[self.max_seq_len] * DECODE_SLOT_COUNT,
            prompt_lens=[0] * DECODE_SLOT_COUNT,
            positions=torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32),
            active_mask=torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bool),
        )
        return cache

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self.allocate_kv_cache_per_layer([(kv_cache_shape, dtype, i) for i in range(num_layers)])

    @staticmethod
    def _sampling_values(sampling_params, batch_size: int, *, unseeded_epoch: int = 0) -> dict[str, Any]:
        if sampling_params is None:
            return {}
        params = format_sampling_params(sampling_params, DECODE_SLOT_COUNT)
        return {
            "top_k": tuple(int(v) for v in params.top_k[:batch_size]),
            "top_p": tuple(float(v) for v in params.top_p[:batch_size]),
            "temperature": tuple(float(v) for v in params.temperature[:batch_size]),
            # Unseeded stochastic requests need independent row streams;
            # explicit seeds retain their exact cross-batch semantics.
            "seeds": tuple(
                ((unseeded_epoch * 104_729 + row) % 2_147_483_647) if v is None else int(v)
                for row, v in enumerate(params.seed[:batch_size])
            ),
        }

    @staticmethod
    def _normalize_page_tables(page_table, page_tables_per_layer, num_layers: int) -> list[torch.Tensor]:
        tables = page_tables_per_layer if page_tables_per_layer is not None else [page_table] * num_layers
        if tables is None or len(tables) != num_layers or any(table is None for table in tables):
            raise ValueError("Gemma4 serving requires one scheduler block table per layer")
        return [torch.as_tensor(table, dtype=torch.int32) for table in tables]

    def _page_tables_changed(self, host_tables: Sequence[torch.Tensor]) -> bool:
        return self._last_page_tables is None or any(
            not torch.equal(current, previous) for current, previous in zip(host_tables, self._last_page_tables)
        )

    def _refresh_page_tables(self, host_tables: Sequence[torch.Tensor]) -> None:
        state = self._serving_state
        if state is None:
            raise RuntimeError("vLLM KV cache has not been allocated")
        for source, target in zip(host_tables, state.page_tables):
            full = torch.zeros(tuple(target.shape), dtype=torch.int32)
            rows = min(full.shape[0], source.shape[0])
            cols = min(full.shape[1], source.shape[1])
            full[:rows, :cols] = source[:rows, :cols]
            host = ttnn.from_torch(
                full,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host, target)
        self._last_page_tables = [table.clone() for table in host_tables]
        self._page_table_refreshes += 1

    def _chunk_page_tables(
        self, host_tables: Sequence[torch.Tensor], start_pos: torch.Tensor, prompt_lens: Sequence[int]
    ) -> list[ttnn.Tensor] | None:
        # Gemma4's local-attention cache uses a 1024-token position modulo.
        # Paged fill therefore needs the scheduler's full table width even when
        # the current prefill chunk spans only a few blocks; narrowing the table
        # makes the modulo larger than its addressable block span.
        del host_tables, start_pos, prompt_lens
        return None

    def _release_decode_traces(self) -> None:
        gen = self._require_generator()
        for trace in gen._trace_cache.values():
            ttnn.release_trace(self.mesh_device, trace.model_trace_id)
            if trace.sampling_trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace.sampling_trace_id)
        gen._trace_cache.clear()

    def _host_prefill_logits(self, logits, batch_size: int) -> torch.Tensor:
        gen = self._require_generator()
        items = logits if isinstance(logits, list) else [logits]
        gathered = [gen._gather_logits_to_torch(item).reshape(1, 1, -1) for item in items]
        result = torch.cat(gathered, dim=0)
        if result.shape[0] != batch_size:
            raise ValueError(f"expected {batch_size} prefill outputs, got {result.shape[0]}")
        return result

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        page_tables_per_layer=None,
        **_: Any,
    ):
        gen = self._require_generator()
        state = self._serving_state
        if state is None or kv_cache is not state.kv_cache:
            raise ValueError("prefill must pass the vLLM-owned cache returned by this adapter")
        tables = self._normalize_page_tables(page_table, page_tables_per_layer, gen.model.num_layers)
        if self._page_tables_changed(tables):
            # Staging tensors must be allocated only after captured addresses
            # are unpinned; otherwise TT-Metal correctly warns that the new
            # buffers can overlap the active trace allocation.
            self._release_decode_traces()
            self._refresh_page_tables(tables)
        cumulative_ends = [int(value) for value in prompt_lens]
        self._unseeded_epoch += 1
        starts = [0] * len(cumulative_ends)
        if start_pos is not None:
            raw_positions = torch.as_tensor(start_pos, dtype=torch.int32)
            if raw_positions.ndim == 1:
                raw_positions = raw_positions.reshape(len(cumulative_ends), -1)
            starts = [int(raw_positions[row, 0]) for row in range(len(cumulative_ends))]
        lengths = [end - start for start, end in zip(starts, cumulative_ends)]
        if any(length < 1 for length in lengths):
            raise ValueError(f"prefill cumulative ends {cumulative_ends} must exceed starts {starts}")
        if empty_slots is not None:
            slots = [int(slot) for slot in empty_slots]
            if len(slots) != len(lengths) or len(set(slots)) != len(slots):
                raise ValueError("empty_slots must contain one unique state slot per prefill row")
            if any(slot < 0 or slot >= self.max_batch_size for slot in slots):
                raise ValueError("empty_slots contains a state slot outside the serving batch")
            # vLLM's external page tables are already packed in execution-row
            # order. Gemma has no recurrent state beyond that cache, so state
            # slot IDs must not be substituted for cache user-row IDs.
        max_chunk = max(lengths)
        chunk_tokens = torch.zeros((len(lengths), max_chunk), dtype=tokens.dtype)
        positions = torch.zeros((len(lengths), max_chunk), dtype=torch.int32)
        for row, (start, end, length) in enumerate(zip(starts, cumulative_ends, lengths)):
            chunk_tokens[row, :length] = tokens[row, start:end]
            positions[row, :length] = torch.arange(start, end, dtype=torch.int32)
        logits = gen.prefill_forward(
            chunk_tokens,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=lengths,
            start_pos=positions,
            chunk_page_tables=self._chunk_page_tables(tables, positions, lengths),
        )
        self._decode_ready = False
        if sampling_params is None:
            return self._host_prefill_logits(logits, len(lengths))
        if isinstance(logits, list):
            values = self._sampling_values(sampling_params, len(lengths), unseeded_epoch=self._unseeded_epoch)
            tokens = []
            for row, row_logits in enumerate(logits):
                row_values = {name: (value[row],) for name, value in values.items()}
                sampled = gen.sample_device_logits(row_logits, batch_size=1, **row_values)
                tokens.append(gen._read_tokens(sampled, 1))
            return torch.cat(tokens)
        sampled = gen.sample_device_logits(
            logits,
            batch_size=len(lengths),
            **self._sampling_values(sampling_params, len(lengths), unseeded_epoch=self._unseeded_epoch),
        )
        return gen._read_tokens(sampled, len(lengths))

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        page_tables_per_layer=None,
        slot_remap=None,
        **_: Any,
    ):
        if not enable_trace:
            raise ValueError("Gemma4 vLLM decode requires tracing")
        gen = self._require_generator()
        state = self._serving_state
        if state is None or kv_cache is not state.kv_cache:
            raise ValueError("decode must pass the vLLM-owned cache returned by this adapter")
        tables = self._normalize_page_tables(page_table, page_tables_per_layer, gen.model.num_layers)
        page_changed = self._page_tables_changed(tables)
        if page_changed:
            # Captured decode reads these stable device tensors by address.
            # Updating their contents in place does not invalidate the trace
            # and preserves the device-feedback token/current-position state.
            self._refresh_page_tables(tables)
        remap_changed = False
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap, dtype=torch.int64).reshape(-1)
            if remap.numel() != self.max_batch_size:
                raise ValueError(f"slot_remap has {remap.numel()} entries; expected {self.max_batch_size}")
            if sorted(remap.tolist()) != list(range(self.max_batch_size)):
                raise ValueError("slot_remap must be a permutation of the serving slots")
            remap_changed = not torch.equal(remap, torch.arange(self.max_batch_size))
        # The TT worker has already gathered tokens, positions, block tables,
        # and sampling parameters into decode-row order. Sampling1D consumes
        # explicit per-call seeds and therefore owns no hidden per-slot RNG
        # state to remap. A non-identity mapping still invalidates captured
        # row state, so recapture before executing the newly aligned batch.
        if reset_batch or remap_changed or not self._decode_ready:
            self._release_decode_traces()
            self._decode_ready = True
        flat_positions = start_pos.reshape(-1)[: self.max_batch_size]
        logical_batch = int((flat_positions >= 0).sum().item())
        if (
            logical_batch < 1
            or not bool((flat_positions[:logical_batch] >= 0).all())
            or bool((flat_positions[logical_batch:] >= 0).any())
        ):
            raise ValueError("vLLM decode rows must pack active requests before inactive slots")
        # Model compute must use only scheduler-active rows. Sampling1D pads
        # logits and parameters to its canonical 32 lanes internally; padding
        # the model itself made greedy trajectories depend on concurrency and
        # let inactive rows participate in decode collectives.
        execution_batch = logical_batch
        active = flat_positions[:execution_batch] >= 0
        mode = "host" if sampling_params is None else "device"
        output = gen.decode_forward(
            tokens.reshape(-1)[:execution_batch].reshape(execution_batch, 1),
            flat_positions[:execution_batch],
            page_table=state.page_tables,
            kv_cache=state,
            sampling_mode=mode,
            enable_trace=True,
            active_mask=active,
            **self._sampling_values(sampling_params, execution_batch, unseeded_epoch=self._unseeded_epoch),
        )
        if sampling_params is None:
            # Explicit compatibility mode for vLLM features unsupported by the
            # device sampler. Keep the full-logits gather delegated to the
            # canonical generator and never enter this branch in performance
            # runs (`sample_on_device_mode=all`).
            return gen._gather_logits_to_torch(output).reshape(execution_batch, 1, -1)
        if read_from_device:
            return self.process_decode_output_host(
                self.read_decode_output(output), is_tokens=sampling_params is not None
            )
        return output

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if async_read:
            host = tt_out.cpu(blocking=False)
            return host, [ttnn.record_event(self.mesh_device, 0)]
        return tt_out.cpu()

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if not isinstance(tt_out, torch.Tensor):
            tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        if is_tokens:
            return tt_out.reshape(-1)[: self.max_batch_size].to(torch.int32), None
        return tt_out, None

    def warmup_model_prefill(self, *args, **kwargs):
        del args, kwargs

    def warmup_model_decode(self, *args, **kwargs):
        del args, kwargs

    def teardown(self):
        self._release_decode_traces()


__all__ = ["Gemma4ForCausalLM"]
