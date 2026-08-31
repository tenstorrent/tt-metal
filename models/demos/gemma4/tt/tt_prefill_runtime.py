# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common-runner runtime for Gemma 4 31B prefill and global KV staging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.demo.prefill_runtime import (
    PAGE_BLOCK_SIZE,
    _cache_completion_state,
    _chunk_page_table_row,
    _cp_chunk_valid_lengths,
    _device_page_table,
    _fixed_cache_slot_blocks,
    _host_mesh_scalars,
    _host_page_table,
    _host_tensor,
    _page_table_row,
)
from models.demos.gemma4.tt.common import create_tt_model


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int
    max_seq_len: int
    mesh_shape: tuple = (8, 4)
    chunk_size: int = 8192
    num_users: int = 8
    sp_axis: int = 0
    tp_axis: int = 1
    weight_cache_path: Optional[Path] = None
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0
    use_trace: bool = False
    sliding_cache_len: int = 262144

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Run the existing Gemma chunked prefill path under the common engine."""

    def __init__(self, mesh_device, hf_config, model_path: str, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.model_path = model_path
        self.config = config
        self.compiled = False
        self._on_layer_complete = None
        self._layer_completion_sink = None

        if config.use_trace:
            raise NotImplementedError("Gemma 4 migration staging is initially supported only with PREFILL_USE_TRACE=0")
        if not (
            config.is_first_rank and config.is_last_rank and config.first_layer_idx == 0 and config.num_layers == 60
        ):
            raise NotImplementedError("initial Gemma 4 common-prefill support requires one rank owning all 60 layers")
        if config.mesh_shape != (8, 4):
            raise ValueError(f"Gemma 4 migration requires CP8/TP4 mesh (8,4), got {config.mesh_shape}")
        if config.max_seq_len % config.chunk_size:
            raise ValueError(f"max_seq_len={config.max_seq_len} must be divisible by chunk_size={config.chunk_size}")
        # Sliding computation currently needs a full-history ring cache in
        # addition to the decode-ordered migration cache. Refuse profiles where
        # those two families alone exceed a conservative per-device budget,
        # before model construction fails deep inside DRAM allocation.
        sliding_one_family = (
            config.num_users * 50 * 4 * (config.sliding_cache_len // config.sp_factor) * 256 * 2 * 1088 // 1024
        )
        sliding_total = 2 * sliding_one_family
        if sliding_total > 24 * 1024**3:
            raise MemoryError(
                "Gemma 4 sliding ring + migration caches require "
                f"{sliding_total / 1024**3:.2f} GiB/device before weights/global KV. "
                "Lower PREFILL_GEMMA4_SLIDING_CACHE_LEN; the current full-history ring "
                "implementation cannot fit this profile."
            )

        self.mesh_config = MeshConfig(
            config.mesh_shape,
            decode=ModeConfig(tp=config.tp_factor),
            prefill=ModeConfig(tp=config.tp_factor, sp=config.sp_factor),
        )
        max_blocks = config.num_users * config.max_seq_len // PAGE_BLOCK_SIZE
        from models.tt_transformers.tt.common import PagedAttentionConfig

        self.paged_config = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=max_blocks)
        model_args, self.model, _, _ = create_tt_model(
            mesh_device=mesh_device,
            # Requests may be chunk-interleaved by the common producer. Ring
            # history therefore needs one scratch slot per scheduler slot.
            max_batch_size=config.num_users,
            max_seq_len=config.max_seq_len,
            dtype=ttnn.bfloat16,
            state_dict=_cache_completion_state(model_path),
            num_layers=config.num_layers,
            mesh_config=self.mesh_config,
            paged_attention_config=self.paged_config,
            create_kv_cache=False,
            model_path=model_path,
            bounded_sliding_kv_cache=False,
            prefill_chunk_size=config.chunk_size,
            sliding_ring_max_seq_len=config.sliding_cache_len,
        )
        self.hf_config = model_args
        self.cp = config.sp_factor
        self._page_table_width = config.max_seq_len // PAGE_BLOCK_SIZE // self.cp
        self._chunk_page_table_width = config.chunk_size // PAGE_BLOCK_SIZE // self.cp
        self._sliding_layers = tuple(layer_type == "sliding_attention" for layer_type in model_args.layer_types)
        self._slot_blocks = [_fixed_cache_slot_blocks(slot, self._page_table_width) for slot in range(config.num_users)]

        identity = torch.arange(self._page_table_width, dtype=torch.int32).reshape(1, -1)
        self.page_table = _device_page_table(mesh_device, identity)
        self.chunk_page_table = _device_page_table(mesh_device, identity[:, : self._chunk_page_table_width])
        self.model._active_page_tables_per_layer = self._layer_page_tables(0)
        self.model.update_persistent_per_layer_page_tables(self.model._active_page_tables_per_layer)

        zeros = torch.zeros((1, config.chunk_size), dtype=torch.int32)
        self.device_positions = ttnn.to_device(
            _host_tensor(
                mesh_device,
                zeros,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            device=mesh_device,
        )
        self.model.set_prefill_rope_positions(self.device_positions)
        self.model._ring_metadata_external = True

    def _resolve_kv(self, kv_caches):
        from models.demos.gemma4.tt.runners.adapters.gemma4 import Gemma4KvCaches

        if not isinstance(kv_caches, Gemma4KvCaches):
            raise TypeError(f"expected Gemma4KvCaches, got {type(kv_caches).__name__}")
        return kv_caches

    def _layer_page_tables(self, slot: int) -> list[torch.Tensor]:
        full = _page_table_row(self._page_table_width, self._slot_blocks[slot])
        # Sliding layers have no paged cache in the migration runtime. Keep a
        # shape-compatible table in the per-layer list; it is never consumed.
        return [full for _is_sliding in self._sliding_layers]

    def _stage_metadata(self, *, slot_id: int, actual_start: int, actual_end: int) -> None:
        layer_tables = self._layer_page_tables(slot_id)
        ttnn.copy_host_to_device_tensor(_host_page_table(self.mesh_device, layer_tables[0]), self.page_table)
        self.model.update_persistent_per_layer_page_tables(layer_tables)

        chunk_idx = actual_start // self.config.chunk_size
        chunk_row = _chunk_page_table_row(self._slot_blocks[slot_id], chunk_idx, self._chunk_page_table_width)
        ttnn.copy_host_to_device_tensor(_host_page_table(self.mesh_device, chunk_row), self.chunk_page_table)

        valid = actual_end - actual_start
        valid_lengths = _cp_chunk_valid_lengths(valid, self.config.chunk_size, self.cp, self.config.tp_factor)
        if self.model.prefill_valid_len_dev is not None:
            ttnn.copy_host_to_device_tensor(
                _host_mesh_scalars(self.mesh_device, valid_lengths), self.model.prefill_valid_len_dev
            )
        positions = torch.arange(actual_start, actual_start + self.config.chunk_size, dtype=torch.int32).unsqueeze(0)
        ttnn.copy_host_to_device_tensor(
            _host_tensor(
                self.mesh_device,
                positions,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            self.device_positions,
        )
        self.model.ccl_manager.set_ring_metadata(slot_idx=slot_id, kv_actual_global=actual_start)
        for semaphore in self.model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(semaphore, 0)

    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor:
        if len(token_ids) != self.config.chunk_size:
            raise ValueError(f"expected {self.config.chunk_size} token ids, got {len(token_ids)}")
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(1, self.config.chunk_size)
        return ttnn.to_device(
            _host_tensor(
                self.mesh_device,
                tokens,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            device=self.mesh_device,
        )

    def compile(self, kv_caches) -> None:
        self.prefill_chunk(
            self.make_chunk_input([0] * self.config.chunk_size),
            kv_caches,
            slot_id=0,
            actual_start=0,
            actual_end=self.config.chunk_size,
        )
        if self.config.sliding_cache_len > self.config.chunk_size:
            self.prefill_chunk(
                self.make_chunk_input([0] * self.config.chunk_size),
                kv_caches,
                slot_id=0,
                actual_start=self.config.chunk_size,
                actual_end=2 * self.config.chunk_size,
            )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev=None,
        **_kwargs,
    ):
        del record_dev
        if d2h_service is not None:
            raise NotImplementedError("Gemma 4 currently uses host layer-completion callbacks")
        if not 0 <= slot_id < self.config.num_users:
            raise ValueError(f"slot_id {slot_id} outside [0, {self.config.num_users})")
        if actual_start % self.config.chunk_size:
            raise ValueError(f"actual_start={actual_start} must be chunk aligned")
        if not actual_start < actual_end <= actual_start + self.config.chunk_size:
            raise ValueError(f"invalid chunk range [{actual_start}, {actual_end})")
        if actual_start + self.config.chunk_size > self.config.max_seq_len:
            raise ValueError("chunk exceeds the configured cache")

        kv = self._resolve_kv(kv_caches)
        if actual_end > kv.sliding_migration.max_seq_len:
            raise ValueError(
                f"request end {actual_end} exceeds PREFILL_GEMMA4_SLIDING_CACHE_LEN="
                f"{kv.sliding_migration.max_seq_len}"
            )
        self._stage_metadata(slot_id=slot_id, actual_start=actual_start, actual_end=actual_end)
        embeds, page_table, chunk_page_table, _ = self.model.transform_and_embed_prefill_inputs_device(
            input_tensor, self.page_table, self.chunk_page_table, None
        )
        input_tensor.deallocate(True)

        if self._layer_completion_sink is not None:
            sink = self._layer_completion_sink

            def on_layer_complete(layer_idx: int) -> None:
                sink(layer_idx, request_id)

        else:
            on_layer_complete = self._on_layer_complete

        out = self.model.ttnn_prefill_forward(
            x=embeds,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=actual_start,
            kv_cache=kv.paged,
            get_last_token=-1,
            user_id=0,
            migration_slot_id=slot_id,
            migration_actual_end=actual_end,
            global_migration_cache=kv.migration,
            sliding_migration_cache=kv.sliding_migration,
            on_layer_complete=on_layer_complete,
        )
        if out is not None:
            out.deallocate(True)
        return None

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        if not self.compiled:
            raise RuntimeError("compile must finish before layer-ack wiring")

        def on_layer_complete(_layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    def set_layer_completion_sink(self, sink) -> None:
        if not self.compiled:
            raise RuntimeError("compile must finish before layer-completion wiring")
        self._layer_completion_sink = sink

    def kv_migration_base_address(self, kv_caches) -> int:
        return int(self._resolve_kv(kv_caches).migration.kv.buffer_address())

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
    ) -> str:
        del first_layer_idx, num_my_layers, stage_layout
        from models.demos.gemma4.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            cache=self._resolve_kv(kv_caches).migration,
            seq_len=self.config.max_seq_len,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            chunk_size=self.config.chunk_size,
            global_layers=tuple(range(5, 60, 6)),
            path=path,
        )

    def release_trace(self) -> None:
        return None
