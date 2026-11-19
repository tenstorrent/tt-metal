# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Finite-shape prefill preparation around the accepted TP4 decoder.

Logical positions are runtime data. Public calls retain FP32 recurrent state
across all internal chunks and publish BF16 exactly once. Each request row runs
against its selected external cache slot; only diagnostics assemble host logits.
"""

import json

import torch

import ttnn

from .flat_prefill import _can_flatten, _finish_flat, _flat_chunk
from .multichip_decoder import PagedState, RecurrentState


def chunk_intervals(start, end, chunk_size=512):
    """Preserve the accepted GDN chunk boundaries and exact logical tails."""
    while start < end:
        count = min(end - start, chunk_size - start % chunk_size)
        yield start, count
        start += count


def disjoint_active_write_pages(row_table, first_page, length, physical_pages):
    """Certify the exact tile-copy write footprint, excluding unused virtual pages."""
    count = (length + 31) // 32
    if length < 1 or first_page < 0 or first_page + count > row_table.shape[1]:
        raise ValueError("Active cache write range exceeds the virtual page table")
    active = row_table[:, first_page : first_page + count]
    if bool(((active < 0) | (active >= physical_pages)).any()):
        raise ValueError("Active cache write page exceeds the physical allocation")
    return active.numel() == active.unique().numel()


def warmup_page_tables(batch, pages, physical_pages):
    """Own disjoint real writes; skip excess synthetic fill pages on small pools.

    The full attention table always contains valid page IDs. Its first page per
    row is distinct for the prepared decode-style head helper. Native fill's -1
    sentinel suppresses writes for synthetic pages beyond the physical pool.
    """
    if physical_pages < batch:
        raise ValueError("Prefill preparation needs at least one physical cache page per active row")
    first = torch.arange(batch * 16, dtype=torch.int32).reshape(16, batch).T.contiguous()
    fill = torch.where(first < physical_pages, first, -1)
    full = torch.zeros(batch, pages, dtype=torch.int32)
    full[:, :16] = fill.clamp_min(0)
    return full, fill


class BoundedPrefill:
    chunk_size = 512

    def __init__(self, model, cache, page_table, *, prefill_batch_size=1):
        self.model, self.mesh, self.cache = model, model.mesh, cache
        self.batch_size, self.pages = page_table.shape
        self.prefill_batch_size = prefill_batch_size
        if prefill_batch_size not in (1, self.batch_size):
            raise ValueError("Prepare single-row or the configured full-batch prefill variant")
        batch = prefill_batch_size
        if self.pages < 16 or self.pages % 16:
            raise ValueError("Bounded prefill needs the decoder's 512-token padded page capacity")
        self.capacity = min(self.pages * 32, model.config.max_position_embeddings)
        self.max_chunk = min(self.chunk_size, self.capacity)
        self.prepared = False
        self._preparing = False
        self._bulk_fill_enabled = False
        self.prepared_lengths = []
        self.counters = {
            "chunks": 0,
            "unaligned_attention_steps": 0,
            "page_table_refreshes": 0,
            "bulk_fill_chunks": 0,
            "ordered_fill_chunks": 0,
        }
        self.physical_pages = min(
            (
                int(tensor.shape[0])
                for state in cache
                if not isinstance(state, RecurrentState)
                for tensor in (state.key, state.value)
            ),
            default=batch * 16,
        )
        self._page_host = page_table[:batch].to(torch.int32).clone()
        self._fill_page_host = self._page_host[:, :16].clone()
        self.full_table = self._upload(self._page_host, ttnn.int32)
        self.fill_table = self._upload(self._fill_page_host, ttnn.int32)
        self.start = self._upload(torch.zeros(1, dtype=torch.int32), ttnn.int32)
        self.head_positions = self._upload(torch.zeros(batch, dtype=torch.int32), ttnn.int32)
        self.batch_indices = self._upload(torch.arange(batch, dtype=torch.int32), ttnn.int32)
        self.single_position = self._upload(torch.zeros((batch, 1), dtype=torch.int32), ttnn.uint32)
        # Runtime gather removes scalar slice offsets from unaligned heads.
        self.row_index = model.io._upload(torch.zeros(batch, 1, 1280, dtype=torch.int32), ttnn.uint32)
        self.ids, self.positions = {}, {}
        for length in range(1, self.max_chunk + 1):
            self.ids[length] = self._upload(torch.zeros((batch, length), dtype=torch.int32), ttnn.uint32)
            self.positions[length] = self._upload(
                torch.arange(length, dtype=torch.int32)[None].repeat(batch, 1), ttnn.uint32
            )
        self.states = []
        for state in cache:
            if isinstance(state, RecurrentState):
                self.states.append(
                    RecurrentState(
                        model.io._upload(torch.zeros(batch, 3, 2560), layout=ttnn.ROW_MAJOR_LAYOUT),
                        model.io._upload(torch.zeros(batch, 12, 128, 128), ttnn.float32),
                    )
                )
            else:
                self.states.append(PagedState(state.key, state.value, self.full_table))

    def _upload(self, value, dtype):
        return self.model.io._upload(value, dtype, ttnn.ROW_MAJOR_LAYOUT)

    def _write(self, value, target):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def _import_state(self, slot):
        for original, work in zip(self.cache, self.states):
            if isinstance(original, RecurrentState):
                full = self.prefill_batch_size == self.batch_size
                conv = original.conv if full else original.conv[slot : slot + 1]
                recurrent = original.recurrent if full else original.recurrent[slot : slot + 1]
                ttnn.copy(conv, work.conv)
                ttnn.typecast(recurrent, ttnn.float32, output_tensor=work.recurrent)

    def _publish_state(self, slot):
        for original, work in zip(self.cache, self.states):
            if not isinstance(original, RecurrentState):
                continue
            if self.prefill_batch_size == self.batch_size:
                ttnn.copy(work.conv, original.conv)
                ttnn.typecast(work.recurrent, original.recurrent.dtype, output_tensor=original.recurrent)
            else:
                recurrent = ttnn.typecast(work.recurrent, original.recurrent.dtype)
                for source, target in ((work.conv, original.conv), (recurrent, original.recurrent)):
                    begins = [slot] + [0] * (len(target.shape) - 1)
                    ends = [slot + 1] + list(target.shape)[1:]
                    if target.layout == ttnn.TILE_LAYOUT:
                        # slice_write's interleaved TILE route reassigns its
                        # output Tensor& while untilizing/retilizing. Never pass
                        # a trace-bound persistent tensor to that route.
                        staging = ttnn.to_layout(target, ttnn.ROW_MAJOR_LAYOUT)
                        source = ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT)
                        ttnn.experimental.slice_write(source, staging, begins, ends, [1] * len(target.shape))
                        ttnn.copy(ttnn.to_layout(staging, ttnn.TILE_LAYOUT), target)
                    else:
                        ttnn.experimental.slice_write(source, target, begins, ends, [1] * len(target.shape))

    def _attention_prefill(self, layer, hidden, state, rope, *, logical_shape=None):
        query, key, value, gate = layer._qkv(hidden, rope, prefill=True, logical_shape=logical_shape)
        batch, length = tuple(hidden.shape)[:2] if logical_shape is None else logical_shape
        padded = (length + 31) // 32 * 32
        if padded != length:
            query, key, value = (
                ttnn.pad(t, ((0, 0), (0, 0), (0, padded - length), (0, 0)), value=0.0) for t in (query, key, value)
            )
        # Fill copies raw cache tiles; decode update performs its own repack.
        key, value = (ttnn.typecast(t, state.key.dtype) if t.dtype != state.key.dtype else t for t in (key, value))
        bulk = batch > 1 and layer.policy.get("batched_prefill_cache", False) and self._bulk_fill_enabled
        if bulk:
            from .batched_cache_fill_candidate import batched_fill_pair

            batched_fill_pair(
                state.key,
                state.value,
                key,
                value,
                self.fill_table,
                self.batch_indices,
                disjoint_write_pages=True,
            )
        if not bulk or self._preparing:
            # Prepare both existing ownership branches; later page mappings can
            # select ordered writes without compiling after decode capture.
            for row in range(batch):
                ttnn.experimental.paged_fill_cache(state.key, key[row : row + 1], self.fill_table, batch_idx=row)
                ttnn.experimental.paged_fill_cache(state.value, value[row : row + 1], self.fill_table, batch_idx=row)
        output = ttnn.transformer.chunked_scaled_dot_product_attention(
            query,
            state.key,
            state.value,
            self.full_table,
            chunk_start_idx_tensor=self.start,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*layer.policy.get("sdpa_grid", [8, 4])),
                q_chunk_size=32,
                k_chunk_size=layer._attention_chunk(state),
                exp_approx_mode=False,
            ),
            compute_kernel_config=layer.ckc,
        )
        output = output[:, :, :length, :]
        if logical_shape is None:
            return layer._attention_out(hidden, output, gate)
        output = ttnn.reshape(ttnn.transformer.concatenate_heads(output), (1, batch * length, 1536))
        output = ttnn.multiply(
            layer._linear(hidden, "gate") if gate is None else gate,
            output,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        return layer._linear(output, "self_attn.o_proj.weight")

    def _layer_chunk(self, layer, value, state, rope):
        batch, length, width = tuple(value.shape)
        if _can_flatten(layer, batch, length):
            if layer.kind == "linear_attention":
                # Reuse the accepted packed correction, tiled convolution,
                # flat GDN and fused finish implementation without reimplementing it.
                return _flat_chunk(layer, value, state, start_pos=0, rope=None)
            flat = ttnn.reshape(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), (1, batch * length, width))
            normalized = layer._norm(flat, "input_layernorm.weight", logical_batch=batch)
            attention = self._attention_prefill(layer, normalized, state, rope, logical_shape=(batch, length))
            if layer.policy.get("norm_projection_l1", True) and normalized.is_sharded():
                ttnn.deallocate(normalized)
            return ttnn.reshape(_finish_flat(layer, flat, attention, batch, length), (batch, length, width))
        normalized = layer._norm(value, "input_layernorm.weight")
        attention = (
            layer._gdn(normalized, state)
            if layer.kind == "linear_attention"
            else self._attention_prefill(layer, normalized, state, rope)
        )
        if layer.kind == "full_attention" and layer.policy.get("norm_projection_l1", True) and normalized.is_sharded():
            ttnn.deallocate(normalized)
        return layer._finish(value, attention, prefill=True)

    def _attention_head_step(self, layer, value, state, rope):
        normalized = layer._norm(value, "input_layernorm.weight")
        attention = layer._attention_decode(normalized, state, self.head_positions, rope)
        if layer.policy.get("norm_projection_l1", True) and normalized.is_sharded():
            ttnn.deallocate(normalized)
        return layer._finish(value, attention, prefill=True)

    def _attention_head(self, layer, value, state, absolute):
        outputs = []
        batch = self.prefill_batch_size
        for index in range(value.shape[1]):
            self._write(torch.full((batch, 1, 1280), index, dtype=torch.int32), self.row_index)
            row = ttnn.gather(value, 1, self.row_index)
            self._write(torch.full((batch,), absolute + index, dtype=torch.int32), self.head_positions)
            self._write(torch.full((batch, 1), absolute + index, dtype=torch.int32), self.single_position)
            rope = self.model.rotations(self.single_position, batch=batch, length=1)
            outputs.append(self._attention_head_step(layer, row, state, rope))
            self.counters["unaligned_attention_steps"] += 1
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)

    @staticmethod
    def _signature(layer, state):
        # Native program signatures exclude weight addresses, but include specs.
        def tensor_signature(tensor):
            return tuple(tensor.shape), str(tensor.dtype), str(tensor.layout), str(tensor.memory_config())

        weights = tuple(
            (group, name, tensor_signature(tensor))
            for group in ("weights", "row_parallel_weights", "shard_norms")
            for name, tensor in sorted(getattr(layer, group).items())
        )
        state_tensors = (state.conv, state.recurrent) if isinstance(state, RecurrentState) else (state.key, state.value)
        return (
            layer.kind,
            json.dumps(layer.policy, sort_keys=True),
            weights,
            tuple(tensor_signature(t) for t in state_tensors),
            tuple((name, str(config)) for name, config in sorted(layer.matmul_configs.items())),
        )

    @staticmethod
    def _prepare_layer_buffers(layer, representative):
        # Each layer must OWN equivalent persistent buffers: sharing them would
        # couple unrelated traces/resources. Shapes were discovered by warmup.
        for signature, buffers in representative._ccl_buffers.items():
            if signature not in layer._ccl_buffers:
                layer._ccl_buffers[signature] = tuple(
                    (
                        None
                        if tensor is None
                        else ttnn.empty(
                            tuple(tensor.shape),
                            dtype=tensor.dtype,
                            layout=tensor.layout,
                            device=layer.mesh,
                            memory_config=tensor.memory_config(),
                        )
                    )
                    for tensor in buffers
                )

    def prepare(self, *, _diagnostic_lengths=None):
        """Compile every bounded shape before any trace; caller resets its KV.

        Preparation intentionally mutates cache contents. The generator must run
        its prepared reset and restore RNG/request inputs before first capture.
        This method never releases or captures traces itself.
        """
        if self.prepared:
            return
        lengths = (
            list(range(1, self.max_chunk + 1)) if _diagnostic_lengths is None else sorted(set(_diagnostic_lengths))
        )
        if not lengths or any(not 1 <= length <= self.max_chunk for length in lengths):
            raise ValueError("Diagnostic preparation lengths must lie within the physical chunk capacity")
        runtime_full, runtime_fill = self._page_host.clone(), self._fill_page_host.clone()
        previous_bulk = self._bulk_fill_enabled
        warm_full, warm_fill = warmup_page_tables(self.prefill_batch_size, self.pages, self.physical_pages)
        self._write(warm_full, self.full_table)
        self._write(warm_fill, self.fill_table)
        self._preparing = True
        # Synthetic real destinations are disjoint; -1 entries skip writes in
        # the native kernel. This safely warms both branches even for a pool
        # smaller than batch*16, without restricting valid shorter requests.
        self._bulk_fill_enabled = True
        try:
            batch = self.prefill_batch_size
            representatives = {}
            concat_inputs = None
            layer_representatives = []
            for layer, state in zip(self.model.layers, self.states):
                signature = self._signature(layer, state)
                representative = representatives.setdefault(signature, (layer, state))
                layer_representatives.append(representative[0])
            for slot in range(self.batch_size if batch == 1 else 1):
                self._import_state(slot)
                self._publish_state(slot)
            self._write(torch.zeros(1, dtype=torch.int32), self.start)
            for length in lengths:
                value = self.model.embeddings(self.ids[length])
                rope = self.model.rotations(self.positions[length], batch=batch, length=length)
                for layer, state in representatives.values():
                    result = self._layer_chunk(layer, value, state, rope)
                    if layer.kind == "full_attention":
                        if length == 32 and batch == 1:
                            # Exercise a vector start inside a512-key SDPA block.
                            # The tensor start is runtime data; all shapes and
                            # program signatures match the aligned path above.
                            self._write(torch.tensor([32], dtype=torch.int32), self.start)
                            shifted_fill = torch.nn.functional.pad(warm_fill[:, 1:], (0, 1), value=-1)
                            self._write(shifted_fill, self.fill_table)
                            result = self._layer_chunk(layer, value, state, rope)
                            self._write(torch.zeros(1, dtype=torch.int32), self.start)
                            self._write(warm_fill, self.fill_table)
                        # Warm the head helpers without executing O(length) attention
                        # calls for every shape. Only runtime index data varies.
                        row = ttnn.gather(value, 1, self.row_index)
                        one_rope = self.model.rotations(self.single_position, batch=batch, length=1)
                        one = self._attention_head_step(layer, row, state, one_rope)
                        if length > 1:
                            if concat_inputs is None:
                                # Concat hashes the input alias partition. Runtime
                                # head outputs are DISTINCT buffers, so [one]*N is
                                # a different program signature and is not warmup.
                                concat_inputs = [ttnn.clone(one) for _ in range(self.max_chunk)]
                            result = ttnn.concat(concat_inputs[:length], dim=1)
                self.model.terminal(result)
                self.model.terminal(result[:, -1:, :])
                self.prepared_lengths.append(length)
                if length % 32 == 0:
                    print(f"PREPARE prefill {length}/{self.max_chunk}", flush=True)
            for layer, representative in zip(self.model.layers, layer_representatives):
                self._prepare_layer_buffers(layer, representative)
        finally:
            self._write(runtime_full, self.full_table)
            self._write(runtime_fill, self.fill_table)
            self._page_host, self._fill_page_host = runtime_full, runtime_fill
            self._bulk_fill_enabled = previous_bulk
            self._preparing = False
        self.prepared = True

    def run(self, tokens, *, slot=0, start_pos=0, page_table, return_all_logits=False, allow_unprepared=False):
        """Yield (absolute_start, logical_length, device_logits) for uniform rows.

        ``tokens`` is host token data for exactly the segment starting at
        ``start_pos``. Consume the iterator completely to publish recurrent state.
        Only the final chunk yields logits when ``return_all_logits`` is false.
        ``allow_unprepared`` is an explicit eager diagnostic path permitted only
        before the generator has captured its first trace. The generator owns
        that lifetime check; this helper never retires traces to permit a call.
        """
        if not self.prepared and not allow_unprepared:
            raise RuntimeError("Prepare bounded prefill before capturing any generator traces")
        if not self.prepared and ttnn.is_trace_capture_active(self.mesh):
            raise RuntimeError("Unprepared prefill cannot execute during capture")
        batch = self.prefill_batch_size
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        if tokens.ndim == 1 and batch == 1:
            tokens = tokens[None]
        if tokens.ndim != 2 or tokens.shape[0] != batch:
            raise ValueError("Token rows must match the prepared physical prefill batch")
        length = tokens.shape[1]
        end = start_pos + length
        if not 0 <= slot < self.batch_size or not 0 <= start_pos < end <= self.capacity:
            raise ValueError("Prefill slot or logical interval is outside configured capacity")
        if batch > 1 and slot != 0:
            raise ValueError("Uniform full-batch prefill uses all configured slots in slot order")
        if tuple(page_table.shape) != (self.batch_size, self.pages):
            raise ValueError("Prefill page table must retain its configured shape")
        row_table = page_table[slot : slot + batch].to(torch.int32)
        if self._page_host is None or not torch.equal(row_table, self._page_host):
            self._write(row_table, self.full_table)
            self._page_host = row_table.clone()
            self.counters["page_table_refreshes"] += 1
        self._import_state(slot)
        for absolute, count in chunk_intervals(start_pos, end, self.chunk_size):
            if self.prepared and count not in self.prepared_lengths:
                raise RuntimeError("Diagnostic preparation does not cover this chunk; public preparation covers 1..512")
            relative = absolute - start_pos
            print(
                "QWEN_PREFILL_CHUNK "
                + json.dumps(
                    dict(
                        batch=batch,
                        slot=slot,
                        start=absolute,
                        length=count,
                        end=absolute + count,
                        aligned=absolute % self.chunk_size == 0,
                        tile_aligned=absolute % 32 == 0,
                        cache_pages=self.physical_pages,
                        mode="eager",
                        prepared=self.prepared,
                    )
                ),
                flush=True,
            )
            self._write(tokens[:, relative : relative + count], self.ids[count])
            self._write(
                torch.arange(absolute, absolute + count, dtype=torch.int32)[None].repeat(batch, 1),
                self.positions[count],
            )
            self._write(torch.tensor([absolute], dtype=torch.int32), self.start)
            # The tile-offset vector candidate passed real-weight validation
            # for single rows. Preserve the accepted multi-row arithmetic.
            aligned = absolute % self.chunk_size == 0 or (batch == 1 and absolute % 32 == 0)
            if aligned:
                first_page = absolute // 32
                self._bulk_fill_enabled = disjoint_active_write_pages(row_table, first_page, count, self.physical_pages)
                fill = row_table[:, first_page : first_page + 16]
                # A tile-aligned continuation can start inside the final512
                # region. Fill-only -1 entries preserve the fixed16-page shape
                # without granting SDPA access beyond the actual context.
                self._fill_page_host = torch.nn.functional.pad(fill, (0, 16 - fill.shape[1]), value=-1)
                self._write(self._fill_page_host, self.fill_table)
                bulk = (
                    batch > 1
                    and self._bulk_fill_enabled
                    and any(
                        layer.kind == "full_attention" and layer.policy.get("batched_prefill_cache", False)
                        for layer in self.model.layers
                    )
                )
                self.counters["bulk_fill_chunks" if bulk else "ordered_fill_chunks"] += 1
            value = self.model.embeddings(self.ids[count])
            rope = self.model.rotations(self.positions[count], batch=batch, length=count)
            for layer, state in zip(self.model.layers, self.states):
                if layer.kind == "full_attention" and not aligned:
                    value = self._attention_head(layer, value, state, absolute)
                else:
                    value = self._layer_chunk(layer, value, state, rope)
            self.counters["chunks"] += 1
            if return_all_logits or absolute + count == end:
                logits = self.model.terminal(value if return_all_logits else value[:, -1:, :])
                yield absolute, count, logits
        self._publish_state(slot)
