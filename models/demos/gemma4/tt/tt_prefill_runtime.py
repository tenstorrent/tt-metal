# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common-prefill runtime for Gemma 4 context-parallel prefill.

One instance per pipeline rank. A rank owns GLOBAL layers ``[first_layer_idx,
first_layer_idx + num_layers)``; ``is_first_rank`` decides whether the chunk input is
token IDs (embedded here) or a hidden-state activation arriving over the D2D socket, and
``is_last_rank`` decides whether the output is the populated KV cache (nothing to return)
or the residual stream to hand downstream. Single-rank is the all-defaults case.
"""

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
        self._trace_d2h_service = None
        self._trace_metadata_msg = None
        if config.sp_factor * config.tp_factor != mesh_device.get_num_devices():
            raise ValueError(
                f"mesh_shape {config.mesh_shape} (sp={config.sp_factor} x tp={config.tp_factor}) does not "
                f"match the {mesh_device.get_num_devices()} devices opened"
            )
        if config.first_layer_idx < 0 or config.num_layers <= 0:
            raise ValueError(f"invalid layer window [{config.first_layer_idx}, +{config.num_layers})")
        if config.max_seq_len % config.chunk_size:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if config.chunk_size % (config.sp_factor * 1024):
            raise ValueError("chunk_size must give every CP rank at least one 1024-token sliding window")
        # Needed before the model exists, to size a non-first rank's placeholder activation.
        # Read from the checkpoint's config rather than carried in the config object so it
        # cannot disagree with the model the next rank actually built.
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        _hf = Gemma4ModelArgs.load_hf_config(model_path)
        self.hidden_size = int(getattr(_hf, "text_config", _hf).hidden_size)

    def _host_state_dict(self):
        """Host weights for THIS rank: the cache-completion pair, or a cold layer window.

        Warm cache (the normal path): only what ttnn.as_tensor cannot serve from a tensorbin --
        the per-layer scalars (read as Python floats) and the embedding table.

        Cold build (GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1): the real weights, but only for the layers
        this rank owns. Reading the whole 62 GiB checkpoint per rank is what makes four concurrent
        ranks a host-RAM problem rather than an NFS one; a window is ~1/PP of that.
        """
        from models.demos.gemma4.demo.prefill_runtime import _load_full_weights
        from models.demos.gemma4.utils.partial_weights import load_layer_window_state

        if not _load_full_weights():
            return _cache_completion_state(self.model_path)
        return load_layer_window_state(
            self.model_path,
            self.config.first_layer_idx,
            self.config.num_layers,
            # First rank only. This service builds the model with prefill_weights_only=True (see
            # _build_model), so no rank builds an LM head -- the KV cache IS the product -- and the
            # 2.8 GiB embedding table is needed nowhere but where tokens are embedded.
            with_embedding=self.config.is_first_rank,
        )

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
            state_dict=self._host_state_dict(),
            num_layers=self.config.num_layers,
            mesh_config=self.mesh_config,
            create_kv_cache=False,
            prefill_weights_only=True,  # This service produces KV without computing logits.
            model_path=self.model_path,
            prefill_chunk_size=self.config.chunk_size,
            ring_kv_caches=kv_caches.layers,
            first_layer_idx=self.config.first_layer_idx,
            is_first_rank=self.config.is_first_rank,
            is_last_rank=self.config.is_last_rank,
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
        self._trace_metadata_msg = self._make_metadata_msg((0, 0, self.config.chunk_size))

    def make_placeholder_activation(self):
        """A non-first rank's stand-in for the hidden state it will receive over D2D.

        Same per-device shape and layout as ``activation_global_spec(chunk, hidden)``
        mapped ``[Shard(2), Replicate()]`` by the runner: sequence CP-sharded, embedding
        whole (Gemma 4 sets ``pipeline_activation_emb_tp_sharded = False``). Used to
        compile and to capture the trace before any real activation exists.
        """
        chunk_per_chip = self.config.chunk_size // self.config.sp_factor
        return ttnn.from_torch(
            torch.zeros(1, 1, chunk_per_chip, self.hidden_size, dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def make_chunk_input(self, token_ids: list[int]):
        """Create the same CP-major local shape produced by H2DStreamService.

        First rank only: a later pipeline rank never sees token IDs, it receives an
        already-embedded activation, so it gets a placeholder of that shape instead.
        """
        if not self.config.is_first_rank:
            return self.make_placeholder_activation()
        if len(token_ids) != self.config.chunk_size:
            raise ValueError(f"expected {self.config.chunk_size} token ids, got {len(token_ids)}")
        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(
            self.config.sp_factor, 1, self.config.chunk_size // self.config.sp_factor
        )
        mapper = ttnn.create_mesh_mapper(
            self.mesh_device,
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
        )
        return ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def _normalize_input(self, input_tensor):
        expected_local = self.config.chunk_size // self.config.sp_factor
        if not self.config.is_first_rank:
            # Hidden state from upstream: [1, 1, chunk/cp, hidden] per device. Check the
            # two dims that actually carry meaning -- a wrong sequence split or a
            # TP-sharded embedding would otherwise surface as a matmul shape error 60
            # layers later.
            shape = tuple(int(d) for d in input_tensor.shape)
            if shape[-2:] != (expected_local, self.hidden_size):
                raise ValueError(
                    f"unexpected Gemma 4 pipeline activation shape {shape}; expected "
                    f"(..., {expected_local}, {self.hidden_size}) on a non-first rank"
                )
            return input_tensor
        local_tokens = 1
        for dim in input_tensor.shape:
            local_tokens *= int(dim)
        if local_tokens != expected_local:
            raise ValueError(
                f"unexpected Gemma 4 local chunk shape {tuple(input_tensor.shape)}; "
                f"expected {expected_local} tokens per CP rank"
            )
        return input_tensor

    def _make_metadata_msg(self, values):
        """Allocate the fixed-address metadata record read by traced D2H acknowledgements."""
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, 1, 1, 3),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    @property
    def trace_metadata_msg(self):
        return self._trace_metadata_msg

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

    def _forward(self, input_tensor, chunk_start: int, *, d2h_service=None, metadata_msg=None):
        with _lm_head_deferred(self.model):
            if self.config.is_first_rank:
                embeds, _, _, _ = self.model.transform_and_embed_prefill_inputs_device(input_tensor, None, None, None)
            else:
                # Already the residual stream: the upstream stage embedded it and ran its own
                # layers. Embedding again would be nonsense (these are not token ids).
                #
                # Clone rather than pass it straight through: Gemma4DecoderLayer DEALLOCATES its
                # input (it is the residual it adds into, freed after the add), so handing it
                # self._trace_input would free the persistent buffer the captured trace reads
                # from and that prefill_chunk copies each chunk into -- a use-after-free that
                # surfaces as a segfault inside the next forward's first rms_norm, not as an
                # error here. The first rank gets this for free because ttnn.embedding produces
                # a fresh tensor and leaves the token buffer alone. One 11 MB device copy per
                # chunk, inside the trace.
                embeds = ttnn.clone(input_tensor)
            return self.model.ttnn_prefill_forward(
                x=embeds,
                chunk_start_idx=chunk_start,
                kv_cache=None,
                get_last_token=-1,
                user_id=0,
                on_layer_complete=self._on_layer_complete,
                d2h_service=d2h_service,
                metadata_msg=metadata_msg,
            )

    def compile(self, kv_caches):
        kv = self._resolve_kv(kv_caches)
        started = time.perf_counter()
        self._build_model(kv)
        self._stage_metadata(0, 0)
        output = self._forward(self._trace_input, 0)
        ttnn.synchronize_device(self.mesh_device)
        if output is not None:
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
        if self._trace_d2h_service is not None:
            # The D2H op is wired after the ordinary compile pass. Warm its programs before
            # capture; trace capture cannot absorb a program-cache miss. This emits one real
            # record per layer, which the runner drains via warmup_ack_count().
            warm_output = self._forward(
                self._trace_input,
                0,
                d2h_service=self._trace_d2h_service,
                metadata_msg=self._trace_metadata_msg,
            )
            ttnn.synchronize_device(self.mesh_device)
            if warm_output is not None:
                warm_output.deallocate(True)
        controller.begin_capture()
        out = self._forward(
            self._trace_input,
            0,
            d2h_service=self._trace_d2h_service,
            metadata_msg=self._trace_metadata_msg if self._trace_d2h_service is not None else None,
        )
        controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)
        # Non-last rank: this is the persistent activation buffer every replay refreshes in
        # place, and it is what prefill_chunk hands back for the driver to push over D2D.
        # Last/single rank: the populated KV cache is the output, so the post-norm hidden is
        # dead -- but it stays allocated because the captured trace writes to its address.
        self._trace_output = out
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
            if d2h_service is not None and d2h_service is not self._trace_d2h_service:
                raise ValueError("traced D2H service must be registered with set_d2h_ack_service before capture")
            if self._trace_d2h_service is not None:
                if metadata_msg is None:
                    raise ValueError("metadata_msg is required for traced D2H layer acknowledgements")
                ttnn.copy(metadata_msg, self._trace_metadata_msg)
            ttnn.copy(source, self._trace_input)
            self._trace_controller.replay()
            if input_tensor is not self._trace_input:
                ttnn.deallocate(input_tensor)
            # Non-last rank: the replay just refreshed the persistent output buffer; hand it
            # back for the driver to forward over D2D. It must NOT be deallocated -- the
            # captured trace writes to that exact address on every subsequent chunk, which is
            # why the runner sends it with deallocate=False under use_trace.
            return None if self.config.is_last_rank else self._trace_output
        if d2h_service is not None and metadata_msg is None:
            raise ValueError("metadata_msg is required for D2H layer acknowledgements")
        output = self._forward(source, actual_start, d2h_service=d2h_service, metadata_msg=metadata_msg)
        if input_tensor is not output:
            ttnn.deallocate(input_tensor)
        if self.config.is_last_rank:
            if output is not None:
                output.deallocate(True)
            return None
        return output

    def set_layer_ack_channel(self, channel):
        if not self.compiled:
            raise RuntimeError("compile must finish before layer-ack wiring")
        if self._trace_d2h_service is not None:
            raise RuntimeError("D2H and host layer acknowledgements are mutually exclusive")
        self._on_layer_complete = lambda _layer_idx: channel.inject(1)

    def set_layer_completion_sink(self, sink):
        if not self.compiled:
            raise RuntimeError("compile must finish before completion wiring")
        if self._trace_d2h_service is not None:
            raise RuntimeError("D2H and host layer acknowledgements are mutually exclusive")
        self._layer_completion_sink = sink
        self._on_layer_complete = lambda layer_idx: sink(layer_idx, self._trace_request_id)

    def set_d2h_ack_service(self, d2h_service):
        """Bind a traceable device-side layer-ack service before trace capture."""
        if not self.config.use_trace:
            raise RuntimeError("eager prefill passes d2h_service per call")
        if not self.compiled:
            raise RuntimeError("compile must finish before D2H acknowledgement wiring")
        if self._trace_captured:
            raise RuntimeError("D2H service must be registered before trace capture")
        if self._on_layer_complete is not None:
            raise RuntimeError("D2H and host layer acknowledgements are mutually exclusive")
        self._trace_d2h_service = d2h_service

    def warmup_ack_count(self):
        if not self.config.use_trace or self._trace_d2h_service is None:
            return 0
        return self.config.num_layers

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
        if self._trace_output is not None:
            self._trace_output.deallocate(True)
            self._trace_output = None
