# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common-prefill runtime for Gemma 4 CP8/TP4."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.demo.prefill_runtime import _cache_completion_state, _host_tensor, _lm_head_deferred
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.runners.kv_caches import Gemma4KvCaches


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
    use_trace: bool = True

    @property
    def sp_factor(self):
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self):
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Run one serialized chunk at a time into engine-owned durable KV slots."""

    def __init__(self, mesh_device, model_path: str, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.model_path = model_path
        self.config = config
        self.model = None
        self.compiled = False
        self._trace_controller = None
        self._trace_captured = False
        self._trace_input = None
        self._trace_output = None
        self._on_layer_complete = None
        self._layer_completion_sink = None
        self._trace_request_id = 0
        if not (
            config.is_first_rank and config.is_last_rank and config.first_layer_idx == 0 and config.mesh_shape == (8, 4)
        ):
            raise NotImplementedError("Gemma 4 common prefill currently supports one CP8/TP4 rank")
        if config.max_seq_len % config.chunk_size:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if config.chunk_size % (config.sp_factor * 1024):
            raise ValueError("chunk_size must give every CP rank at least one 1024-token sliding window")

    def _resolve_kv(self, kv_caches):
        if not isinstance(kv_caches, Gemma4KvCaches):
            raise TypeError(f"expected Gemma4KvCaches, got {type(kv_caches).__name__}")
        return kv_caches

    def _build_model(self, kv_caches):
        self.mesh_config = MeshConfig(
            self.config.mesh_shape,
            decode=ModeConfig(tp=self.config.tp_factor),
            prefill=ModeConfig(tp=self.config.tp_factor, sp=self.config.sp_factor),
        )
        _, self.model, _, _ = create_tt_model(
            mesh_device=self.mesh_device,
            max_batch_size=1,
            max_seq_len=self.config.max_seq_len,
            dtype=ttnn.bfloat16,
            state_dict=_cache_completion_state(self.model_path),
            num_layers=self.config.num_layers,
            mesh_config=self.mesh_config,
            create_kv_cache=False,
            model_path=self.model_path,
            prefill_chunk_size=self.config.chunk_size,
            ring_kv_caches=kv_caches.layers,
        )
        self.model._ring_metadata_external = True
        self.model._prefill_trace_mode = True
        positions = torch.arange(self.config.chunk_size, dtype=torch.int32).reshape(1, -1)
        self.device_positions = ttnn.to_device(
            _host_tensor(
                self.mesh_device,
                positions,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            device=self.mesh_device,
        )
        self.model.set_prefill_rope_positions(self.device_positions)
        self._trace_input = self.make_chunk_input([0] * self.config.chunk_size)

    def make_chunk_input(self, token_ids: list[int]):
        if len(token_ids) != self.config.chunk_size:
            raise ValueError(f"expected {self.config.chunk_size} token ids, got {len(token_ids)}")
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(1, -1)
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

    def _normalize_input(self, input_tensor):
        if tuple(input_tensor.shape) == (1, self.config.chunk_size):
            return input_tensor
        if input_tensor.shape[-1] * self.config.sp_factor != self.config.chunk_size:
            raise ValueError(f"unexpected Gemma 4 chunk tensor shape {tuple(input_tensor.shape)}")
        return ttnn.reshape(input_tensor, (1, self.config.chunk_size))

    def _stage_metadata(self, slot_id: int, actual_start: int):
        positions = torch.arange(actual_start, actual_start + self.config.chunk_size, dtype=torch.int32).reshape(1, -1)
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

    def _forward(self, input_tensor, chunk_start: int):
        with _lm_head_deferred(self.model):
            embeds, _, _, _ = self.model.transform_and_embed_prefill_inputs_device(input_tensor, None, None, None)
            return self.model.ttnn_prefill_forward(
                x=embeds,
                chunk_start_idx=chunk_start,
                kv_cache=None,
                get_last_token=-1,
                user_id=0,
                on_layer_complete=self._on_layer_complete,
            )

    def compile(self, kv_caches):
        kv = self._resolve_kv(kv_caches)
        started = time.perf_counter()
        self._build_model(kv)
        self._stage_metadata(0, 0)
        output = self._forward(self._trace_input, 0)
        ttnn.synchronize_device(self.mesh_device)
        output.deallocate(True)
        self.compiled = True
        logger.info(f"Gemma 4 runtime compiled in {time.perf_counter() - started:.1f}s")

    def capture_trace(self, kv_caches):
        self._resolve_kv(kv_caches)
        if not self.config.use_trace or self._trace_captured:
            return
        if not self.compiled:
            raise RuntimeError("compile must run before capture_trace")
        controller = SubDeviceTraceController(self.mesh_device)
        if self._on_layer_complete is not None:
            controller.set_layer_ack_callback(self._on_layer_complete)
        self.model.set_prefill_trace_controller(controller)
        self._stage_metadata(0, 0)
        controller.begin_capture()
        self._trace_output = self._forward(self._trace_input, 0)
        controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)
        self._trace_controller = controller
        self._trace_captured = True
        logger.info(
            f"Gemma 4 prefill trace captured: segments={controller.num_segments}, " f"bytes={controller.trace_bytes()}"
        )

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
        metadata_msg=None,
        **_kwargs,
    ):
        del metadata_msg
        if d2h_service is not None:
            raise NotImplementedError("Gemma 4 D2H layer acks are not wired; use the host ack channel")
        kv = self._resolve_kv(kv_caches)
        if not 0 <= slot_id < kv.num_users:
            raise ValueError(f"slot_id {slot_id} outside [0, {kv.num_users})")
        if actual_start % self.config.chunk_size:
            raise ValueError("actual_start must be chunk aligned")
        if not actual_start < actual_end <= actual_start + self.config.chunk_size:
            raise ValueError(f"invalid chunk range [{actual_start}, {actual_end})")
        if actual_start + self.config.chunk_size > self.config.max_seq_len:
            raise ValueError("chunk exceeds the configured cache")
        self._stage_metadata(slot_id, actual_start)
        source = self._normalize_input(input_tensor)
        if self.config.use_trace:
            if not self._trace_captured:
                raise RuntimeError("capture_trace must run before the first traced request")
            self._trace_request_id = request_id
            ttnn.copy(source, self._trace_input)
            self._trace_controller.replay()
            ttnn.deallocate(input_tensor)
            return None
        output = self._forward(source, actual_start)
        ttnn.deallocate(input_tensor)
        if output is not None:
            output.deallocate(True)
        return None

    def set_layer_ack_channel(self, channel):
        if not self.compiled:
            raise RuntimeError("compile must finish before layer-ack wiring")
        self._on_layer_complete = lambda _layer_idx: channel.inject(1)

    def set_layer_completion_sink(self, sink):
        if not self.compiled:
            raise RuntimeError("compile must finish before completion wiring")
        self._layer_completion_sink = sink
        self._on_layer_complete = lambda layer_idx: sink(layer_idx, self._trace_request_id)

    def kv_migration_base_address(self, kv_caches):
        first = self._resolve_kv(kv_caches)[0]
        tensor = first.kv if hasattr(first, "kv") else first[0]
        return int(tensor.buffer_address())

    def build_kv_chunk_table(self, kv_caches, path: str, **_kwargs):
        from models.demos.gemma4.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_caches=self._resolve_kv(kv_caches),
            chunk_size=self.config.chunk_size,
            sp_axis=self.config.sp_axis,
            path=path,
        )

    def release_trace(self):
        if self._trace_controller is not None:
            self._trace_controller.release()
            self.model.set_prefill_trace_controller(None)
            self._trace_controller = None
            self._trace_captured = False
