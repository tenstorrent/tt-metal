# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed projection/MLP traces with eager per-request striped attention on CQ0."""

from dataclasses import dataclass

import ttnn
from models.demos.gemma4_d_p.tt.attention.striped_prefill import (
    striped_prefill_attention,
    striped_prefill_program_config,
)
from models.demos.gemma4_d_p.tt.ccl import ccl_allgather


@dataclass
class _AttentionBuffers:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor | None
    output: ttnn.Tensor
    stats: ttnn.Tensor

    def tensors(self):
        return tuple(tensor for tensor in (self.q, self.k, self.v, self.output, self.stats) if tensor is not None)


class Gemma4HybridPrefill:
    """Own two traces per layer, independent of the staged request count and spans.

    Tokens and model RoPE positions are caller-owned buffers with fixed addresses.
    Update their contents and the model's packed metadata on CQ0 before run().
    The returned output belongs to this object and is overwritten on each run.
    """

    @ttnn.command_queue(0)
    def __init__(self, model, tokens):
        if model.stripe_size != 32 or model._rope_prefill_positions is None:
            raise ValueError("Hybrid prefill requires a striped model with staged RoPE positions")
        local_tokens = model.prefill_chunk_size // model.mesh_config.cp_degree
        if tuple(tokens.shape) != (1, local_tokens):
            raise ValueError("Token buffer must match the model's fixed packed chunk size")
        self.model = model
        self.device = model.mesh_device
        self.tokens = tokens
        self.positions = model._rope_prefill_positions
        self.traces = []
        self._closed = False
        self._configs = {}
        self.hidden = self._empty((1, 1, local_tokens // model.mesh_config.tp_degree, model.hidden_size))
        self.output = self._empty((1, 1, local_tokens, model.hidden_size))
        self.rope = {}
        self.buffers = {}
        for layer in model.layers:
            if layer.layer_type in self.buffers:
                continue
            self.rope[layer.layer_type] = model.lookup_packed_rope(layer.layer_type)
            attention = layer.self_attn
            cache = attention.ring_kv_cache
            keys = cache.k if attention.config.is_sliding else cache.kv
            q_shape = (
                1,
                attention.config.num_attention_heads // model.mesh_config.tp_degree,
                local_tokens,
                attention.config.head_dim,
            )
            self.buffers[layer.layer_type] = _AttentionBuffers(
                q=self._empty(q_shape),
                k=self._empty((1, keys.shape[1], local_tokens, keys.shape[3]), keys.dtype),
                v=(
                    self._empty((1, cache.v.shape[1], local_tokens, cache.v.shape[3]), cache.v.dtype)
                    if attention.config.is_sliding
                    else None
                ),
                output=self._empty(q_shape),
                stats=self._empty((1, q_shape[1], 2 * local_tokens, 1)),
            )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _empty(self, shape, dtype=ttnn.bfloat16):
        return ttnn.empty(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _requests(self):
        if self._closed:
            raise RuntimeError("Hybrid prefill has been closed")
        if self.model._rope_prefill_positions is not self.positions:
            raise ValueError("Update the existing RoPE position buffer; its address is captured")
        layout = self.model.packed_prefill_metadata.layout
        if layout is None:
            raise ValueError("Stage packed request metadata before capture or replay")
        return self.model.prefill_requests(layout)

    def _plan(self, requests):
        plans = {}
        for layer in self.model.layers:
            kind = layer.layer_type
            if kind in plans:
                continue
            attention = layer.self_attn
            buffers = self.buffers[kind]
            cache = attention.ring_kv_cache
            keys = cache.k if attention.config.is_sliding else cache.kv
            plans[kind] = {}
            for length, _ in requests:
                key = (kind, length)
                if key not in self._configs:
                    self._configs[key] = striped_prefill_program_config(
                        attention, length, buffers.q.shape[1], keys.shape[2]
                    )
                plans[kind][length] = self._configs[key]
        return plans

    def _project(self, index):
        if index == 0:
            embeddings = self.model.transform_and_embed_prefill_inputs_device(self.tokens)
            ttnn.copy(embeddings, self.hidden)
            embeddings.deallocate(True)
            for kind, persistent in self.rope.items():
                current = self.model.lookup_packed_rope(kind)
                for source, target in zip(current[:-1], persistent[:-1]):
                    ttnn.copy(source, target)
                    source.deallocate(True)
        layer = self.model.layers[index]
        rope = self.rope[layer.layer_type]
        q, k, v = layer.prepare_attention(
            self.hidden,
            packed_global_rope=rope if layer.layer_type == "full_attention" else None,
            packed_sliding_rope=rope if layer.layer_type == "sliding_attention" else None,
        )
        buffers = self.buffers[layer.layer_type]
        for source, target in ((q, buffers.q), (k, buffers.k), (v, buffers.v)):
            if source is not None:
                if source.dtype == target.dtype:
                    ttnn.copy(source, target)
                else:
                    ttnn.typecast(source, target.dtype, output_tensor=target)
                source.deallocate(True)
        ttnn.zeros_like(buffers.output, optional_tensor=buffers.output)

    def _attend(self, index, requests, plans):
        layer = self.model.layers[index]
        buffers = self.buffers[layer.layer_type]
        striped_prefill_attention(
            layer.self_attn,
            buffers.q,
            buffers.k,
            buffers.v,
            requests,
            self.compute_config,
            output_tensor=buffers.output,
            stats_tensor=buffers.stats,
            program_configs=plans[layer.layer_type],
        )

    def _finish(self, index):
        layer = self.model.layers[index]
        attention_output = layer.self_attn.project_output(self.buffers[layer.layer_type].output)
        result = layer.finish_attention(self.hidden, attention_output, preserve_input=True)
        if result.is_sharded():
            interleaved = ttnn.sharded_to_interleaved(result, ttnn.DRAM_MEMORY_CONFIG)
            result.deallocate(True)
            result = interleaved
        if index == len(self.model.layers) - 1:
            result = ccl_allgather(result, self.model.mesh_config, self.model.ccl_manager, dim=2)
            ttnn.copy(result, self.output)
        else:
            ttnn.copy(result, self.hidden)
        result.deallocate(True)

    def _capture(self, function, index):
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.traces.append(trace)
        try:
            function(index)
        finally:
            ttnn.end_trace_capture(self.device, trace, cq_id=0)

    @ttnn.command_queue(0)
    def capture(self):
        if self.traces:
            raise RuntimeError("Hybrid traces are already captured")
        requests = self._requests()
        plans = self._plan(requests)
        for index in range(len(self.model.layers)):
            self._project(index)
            self._attend(index, requests, plans)
            self._finish(index)
        ttnn.synchronize_device(self.device)
        try:
            for index in range(len(self.model.layers)):
                self._capture(self._project, index)
                self._capture(self._finish, index)
        except Exception:
            self.close()
            raise

    @ttnn.command_queue(0)
    def run(self, *, blocking=False):
        requests = self._requests()
        if len(self.traces) != 2 * len(self.model.layers):
            raise RuntimeError("Capture hybrid traces before replay")
        plans = self._plan(requests)
        for index in range(len(self.model.layers)):
            ttnn.execute_trace(self.device, self.traces[2 * index], cq_id=0, blocking=False)
            self._attend(index, requests, plans)
            ttnn.execute_trace(self.device, self.traces[2 * index + 1], cq_id=0, blocking=False)
        if blocking:
            ttnn.synchronize_device(self.device)
        return self.output

    def close(self):
        if self._closed:
            return
        ttnn.synchronize_device(self.device)
        for trace in self.traces:
            ttnn.release_trace(self.device, trace)
        self.traces.clear()
        for buffers in self.buffers.values():
            for tensor in buffers.tensors():
                tensor.deallocate(True)
        for rope in self.rope.values():
            for tensor in rope[:-1]:
                tensor.deallocate(True)
        self.hidden.deallocate(True)
        self.output.deallocate(True)
        self._closed = True
