# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-request and packed multi-request prefill into engine-owned KV slots."""

from dataclasses import replace

import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.common import create_tt_model


class Gemma4PrefillRuntime:
    def __init__(self, *, mesh_device, hf_model_id, tt_cache_path, config):
        self.mesh_device = mesh_device
        self.mesh_config = MeshConfig(mesh_device)
        self.hf_model_id = hf_model_id
        self.tt_cache_path = tt_cache_path
        self.config = config
        self.trace_id = None
        self.d2h_service = None
        self.layer_completion_sink = None
        self.slot_ends = [0] * config.num_users
        self.slot_requests = [None] * config.num_users
        self.ragged_variants = {}
        self.output_generation = 0
        self.batch_result = None
        self._single_trace_ready = False
        self._captured_d2h_service = None
        self.next_completion_id = 0

    def _host_tokens(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, self.config.chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.config.mesh_shape, dims=(1, None)),
        )

    def make_chunk_input(self, token_ids):
        return ttnn.to_device(self._host_tokens(token_ids), self.mesh_device)

    def _stage_positions(self, slot_id, actual_start):
        self.model.prefill_metadata.update(slot_idx=slot_id, kv_actual_global=actual_start)
        positions = range(actual_start, actual_start + self.config.chunk_size)
        ttnn.copy_host_to_device_tensor(self._host_tokens(positions), self.positions)

    def _forward(self):
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.input_tokens)
        return self.model(
            hidden_states=embeddings,
            d2h_service=self.d2h_service,
            metadata_msg=self.metadata,
        )

    def compile(self, kv_cache):
        _, self.model, _, _ = create_tt_model(
            mesh_config=self.mesh_config,
            hf_model_id=self.hf_model_id,
            prefill_chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            max_batch_size=1,
            ring_kv_caches=kv_cache,
            tt_cache_path=self.tt_cache_path,
        )
        self.input_tokens = self.make_chunk_input([0] * self.config.chunk_size)
        self.positions = self.make_chunk_input(range(self.config.chunk_size))
        self.metadata = ttnn.from_torch(
            torch.tensor([0, 0, self.config.chunk_size], dtype=torch.int64).reshape(1, 1, 1, 3),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.model.set_prefill_rope_positions(self.positions)
        self.model._prefill_metadata_external = True
        self._stage_positions(0, 0)
        output = self._forward()
        ttnn.synchronize_device(self.mesh_device)
        output.deallocate(True)

    def set_d2h_ack_service(self, service):
        self.d2h_service = service

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def warmup_ack_count(self):
        return self.config.num_layers if self.d2h_service is not None else 0

    def capture_trace(self, kv_cache):
        self._check_cache(kv_cache)
        self._invalidate_batch_output()
        self._release_ragged_variants()
        self._release_single_trace()
        # Compile staging copies before capture too: their program-cache buffers
        # must not be allocated into trace transient memory on the first replay.
        ttnn.copy(self.input_tokens, self.input_tokens)
        if self.metadata is not None:
            ttnn.copy(self.metadata, self.metadata)
        if self.d2h_service is not None:
            output = self._forward()
            ttnn.synchronize_device(self.mesh_device)
            output.deallocate(True)
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.output = self._forward()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self._single_trace_ready = True
        self._captured_d2h_service = self.d2h_service

    def _check_cache(self, kv_cache):
        if len(kv_cache.layers) != len(self.model.layers) or any(
            layer.self_attn.ring_kv_cache is not cache for layer, cache in zip(self.model.layers, kv_cache.layers)
        ):
            raise ValueError("The traced runtime requires the caches supplied to compile")

    def validate_chunk(self, slot_id, actual_start, actual_end):
        if not 0 <= slot_id < self.config.num_users:
            raise ValueError(f"KV slot {slot_id} is outside [0, {self.config.num_users})")
        if actual_start < 0 or actual_start % self.config.chunk_size:
            raise ValueError(f"Chunk start must be a nonnegative multiple of {self.config.chunk_size}")
        if not actual_start < actual_end <= min(actual_start + self.config.chunk_size, self.config.max_seq_len):
            raise ValueError(f"Chunk must contain 1 to {self.config.chunk_size} real tokens within the context")
        if actual_start != 0 and actual_start != self.slot_ends[slot_id]:
            raise ValueError(f"Slot {slot_id} expects position {self.slot_ends[slot_id]}, got {actual_start}")

    def validate_batch(self, requests):
        """Validate the entire boundary before staging or writing any request."""
        if not requests or len(requests) > self.config.num_users:
            raise ValueError("A batch must contain 1 to num_users active requests")
        if len({r.slot_id for r in requests}) != len(requests):
            raise ValueError("A batch may contain only one chunk per KV slot")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Batch request IDs must be unique")
        completion_ids = [r.completion_id for r in requests if r.completion_id is not None]
        if len(set(completion_ids)) != len(completion_ids) or any(i < 0 for i in completion_ids):
            raise ValueError("Completion IDs must be distinct nonnegative chunk sequence numbers")
        for request in requests:
            self.validate_chunk(request.slot_id, request.actual_start, request.actual_end)
            owner = self.slot_requests[request.slot_id]
            if request.actual_start and owner is not None and owner != request.request_id:
                raise ValueError(f"Slot {request.slot_id} belongs to request {owner}")
            if any(token < 0 or token >= self.model.vocab_size for token in request.token_ids):
                raise ValueError("Token IDs must lie within the model vocabulary")

    def prefill_batch(self, requests, kv_cache, *, use_trace=True):
        """Run an ordered boundary of independent request chunks.

        Requests contain host token IDs (only valid tokens), stable request IDs,
        slots and absolute starts. A start of zero refills a slot. Subsequent
        chunks must follow that request's previous full chunk. Partial chunks
        are final. Copy result.to_torch() before the next runtime call to retain
        outputs; the device result is a lease on this batch's output buffer.

        Shape changes select a tile-rounded trace variant. One graph is resident:
        a different shape or the legacy path releases it before allocation. Set
        use_trace=False for the eager fallback. No inactive lane writes a cache.
        Socket/callback acknowledgements occur after the batch completes, once
        per request per layer, so compilation never sends duplicate completions.
        """
        from models.demos.gemma4_d_p.tt.ragged_prefill import RaggedPrefillPlan
        from models.demos.gemma4_d_p.tt.runners.ragged_runtime import RaggedPrefillResult, RaggedTraceVariant

        requests = tuple(requests)
        next_id = max(
            self.next_completion_id,
            max((r.completion_id + 1 for r in requests if r.completion_id is not None), default=0),
        )
        assigned = []
        for request in requests:
            if request.completion_id is None:
                request = replace(request, completion_id=next_id)
                next_id += 1
            assigned.append(request)
        requests = tuple(assigned)
        self.validate_batch(requests)
        self._check_cache(kv_cache)
        if self.d2h_service is not None and self._captured_d2h_service is not self.d2h_service:
            raise RuntimeError("Batched socket acknowledgements require the service warmup/capture_trace handshake")
        plan = RaggedPrefillPlan.for_requests(
            requests, chunk_size=self.config.chunk_size, cp=self.mesh_config.cp_degree, tp=self.mesh_config.tp_degree
        )
        self._invalidate_batch_output()
        self._release_single_trace()
        variant = self.ragged_variants.get(plan) if use_trace else None
        if variant is None:
            self._release_ragged_variants()
            variant = RaggedTraceVariant(self, plan)
            if use_trace:
                self.ragged_variants[plan] = variant
        try:
            variant.stage(requests)
            output = variant.run(use_trace=use_trace)
            ttnn.synchronize_device(self.mesh_device)
        except Exception:
            if use_trace:
                self.ragged_variants.pop(plan, None)
                variant.release()
            raise
        finally:
            if not use_trace:
                variant.release()
        for request in requests:
            self.slot_ends[request.slot_id] = request.actual_end
            self.slot_requests[request.slot_id] = request.request_id
        self.next_completion_id = max(self.next_completion_id, next_id)
        # Keep each chunk's layer acknowledgements contiguous, as required by
        # the existing socket/migration consumer's implicit layer counter.
        for request in requests:
            for layer_idx in range(self.config.num_layers):
                if self.layer_completion_sink is not None:
                    self.layer_completion_sink(layer_idx, request.completion_id)
                if self.d2h_service is not None:
                    metadata = self._batch_ack_metadata(request)
                    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(self.d2h_service, metadata=metadata)
                    metadata.deallocate(True)
        self.batch_result = RaggedPrefillResult(plan, requests, output, self, self.output_generation, not use_trace)
        return self.batch_result

    def _invalidate_batch_output(self):
        self.output_generation += 1
        if self.batch_result is not None:
            self.batch_result.deallocate()
            self.batch_result = None

    def _release_single_trace(self):
        if self.trace_id is not None:
            ttnn.synchronize_device(self.mesh_device)
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None
            self.output.deallocate(True)

    def _release_ragged_variants(self):
        if self.ragged_variants:
            ttnn.synchronize_device(self.mesh_device)
            for variant in self.ragged_variants.values():
                variant.release()
            self.ragged_variants.clear()

    def _batch_ack_metadata(self, request):
        return ttnn.from_torch(
            torch.tensor([request.slot_id, request.actual_start, request.actual_end], dtype=torch.int64).reshape(
                1, 1, 1, 3
            ),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prefill_requests(self, prompts, kv_cache, **batch_options):
        """Continuously refill slots from ``(request_id, token_ids)`` prompts.

        Yield one output lease per completed batch. This host scheduler
        is also the integration point for serving queues with explicit batch
        boundaries; the legacy blocking socket protocol retains batch one.
        """
        from models.demos.gemma4_d_p.tt.ragged_prefill import iter_request_batches

        for requests in iter_request_batches(
            prompts, num_slots=self.config.num_users, chunk_size=self.config.chunk_size
        ):
            yield self.prefill_batch(requests, kv_cache, **batch_options)

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        self.validate_chunk(slot_id, actual_start, actual_end)
        self._check_cache(kv_cache)
        self._invalidate_batch_output()
        if self.trace_id is None and not self._single_trace_ready:
            raise RuntimeError("capture_trace must run before serving chunks")
        if d2h_service is not self.d2h_service:
            raise ValueError("The D2H service must match the captured service")
        if self.trace_id is None and self.d2h_service is not None:
            raise RuntimeError("Switching back to the socket trace requires its warmup handshake and capture_trace")
        tokens = ttnn.reshape(input_tensor, (1, self.config.chunk_size // self.mesh_config.cp_degree))
        ttnn.copy(tokens, self.input_tokens)
        if metadata_msg is not None:
            ttnn.copy(metadata_msg, self.metadata)
        elif self.d2h_service is not None:
            raise ValueError("D2H acknowledgments require request metadata")
        self._stage_positions(slot_id, actual_start)
        # Staging inputs are no longer needed, and must not survive a replay
        # that can reuse the transient addresses at which they were allocated.
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)
        if self.trace_id is None:
            self.capture_trace(kv_cache)
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        self.slot_ends[slot_id] = actual_end
        # The legacy socket loop supplies a chunk counter as request_id, not a
        # stable request identity. A batch scheduler can adopt its prefix.
        self.slot_requests[slot_id] = None
        self.next_completion_id = max(self.next_completion_id, request_id + 1)
        if self.layer_completion_sink is not None:
            for layer_idx in range(self.config.num_layers):
                self.layer_completion_sink(layer_idx, request_id)

    def kv_migration_stages(self, kv_cache, first_layer_idx, num_my_layers):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        self._check_cache(kv_cache)
        if first_layer_idx != 0 or num_my_layers != self.config.num_layers:
            raise ValueError("Gemma4 migration requires all layers on one rank")
        stages = []
        for layer_idx, cache in enumerate(kv_cache.layers):
            tensors = (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)
            stages.extend(KvCacheStage(int(tensor.buffer_address()), layer_idx, 1) for tensor in tensors)
        return stages

    def build_kv_chunk_table(self, kv_cache, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        self._check_cache(kv_cache)
        stages = self.kv_migration_stages(
            kv_cache, first_layer_idx, self.config.num_layers if num_my_layers is None else num_my_layers
        )
        if stage_layouts is not None:
            if len(stage_layouts) != len(stages):
                raise ValueError("Gemma4 migration requires one stage per cache tensor")
            for gathered, stage in zip(stage_layouts, stages):
                if len(gathered) != 1 or any(
                    gathered[0][key] != value
                    for key, value in (
                        ("rank", 0),
                        ("base_addr", stage.base_addr),
                        ("first_layer", stage.first_layer),
                        ("count", stage.count),
                    )
                ):
                    raise ValueError("Gemma4 migration stage does not match its single-rank cache")
        return build_and_serialize_kv_chunk_table(
            path=path,
            mesh_device=self.mesh_device,
            kv_caches=kv_cache,
            chunk_size=self.config.chunk_size,
        )

    def release_trace(self):
        self._invalidate_batch_output()
        self._release_ragged_variants()
        self._release_single_trace()
        self._single_trace_ready = False
        self._captured_d2h_service = None
        self.d2h_service = None
        self.layer_completion_sink = None
