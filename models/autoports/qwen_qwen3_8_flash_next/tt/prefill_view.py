# SPDX-License-Identifier: Apache-2.0
"""Serial real-row prefill sharing a fixed-batch serving model's public state.

Only execution geometry and private scratch differ. All weights, request banks,
physical KV pages, precision, context capacity and sampling belong to the owner.
Both paths execute serially on CQ0; independent command queues are unsupported.
"""

import copy
from collections import Counter

import torch

import ttnn

from .gdn_prefill_resident import PrefillResidentGDN
from .multichip_decoder import AttentionOutputProjection, MeshOps, MultichipDecoder
from .prefill_hybrid import forward1024


class _PrefillOps(MeshOps):
    def norm(self, value, weight, eps, *, reuse_output=False):
        shape = tuple(value.shape)
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] not in (128, 256, 512)
            or shape[2:] != (4, 640)
            or value.dtype != ttnn.bfloat16
        ):
            if len(shape) > 1:
                self.norm_unpacked_sequences.add(shape[1])
            return self.norm_unpacked(value, weight, eps, reuse_output=reuse_output)
        # Remove only implicit tile rows. Reduction width, communicated FP32
        # statistics, compute fidelity and stream-specific weights are unchanged.
        flat = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), (1, 1, shape[1] * 4, 640)),
            ttnn.TILE_LAYOUT,
        )
        stats = ttnn.rms_norm_pre_all_gather(flat, dtype=ttnn.float32, compute_kernel_config=self.compute)
        gathered = self.gather(stats)
        if reuse_output:
            del stats
        normalized = ttnn.rms_norm_post_all_gather(flat, gathered, epsilon=eps, compute_kernel_config=self.compute)
        restored = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT
        )
        # This is warm/capture metadata, not a device read or synchronization.
        self.norm_compaction[str(shape[1])] = dict(
            sequence=shape[1],
            input_shape=list(shape),
            input_padded_shape=list(value.padded_shape),
            compact_shape=list(flat.shape),
            compact_padded_shape=list(flat.padded_shape),
            restored_shape=list(restored.shape),
            input_memory_config=str(value.memory_config()),
            compact_memory_config=str(flat.memory_config()),
            restored_memory_config=str(restored.memory_config()),
            input_dtype=str(value.dtype),
            statistics_dtype=str(gathered.dtype),
            weight_shape=list(weight.shape),
            weight_dtype=str(weight.dtype),
            compute_kernel_config={
                key: str(getattr(self.compute, key))
                for key in (
                    "math_fidelity",
                    "math_approx_mode",
                    "fp32_dest_acc_en",
                    "packer_l1_acc",
                    "dst_full_sync_en",
                )
            },
            restore_before_stream_weight=True,
        )
        return ttnn.multiply(restored, weight, output_tensor=restored if reuse_output else None)

    def norm_unpacked(self, value, weight, eps, *, reuse_output=False):
        # Preserve fixed-batch prefill's FP32 communicated statistics, including
        # singleton tails. The standalone B1 singleton optimization uses BF16.
        stats = ttnn.rms_norm_pre_all_gather(value, dtype=ttnn.float32, compute_kernel_config=self.compute)
        gathered = self.gather(stats)
        if reuse_output:
            del stats
        normalized = ttnn.rms_norm_post_all_gather(value, gathered, epsilon=eps, compute_kernel_config=self.compute)
        return ttnn.multiply(normalized, weight, output_tensor=normalized if reuse_output else None)


class _RowOutputProjection(AttentionOutputProjection):
    def __call__(self, x, *, concatenated=False):
        # The fixed-batch owner omits the B1-only column weight representation.
        # Keep its row weight/reduction and BF16 output for singleton tails too.
        return self.ops.scatter_reduce(self.ops.linear(x, self.row_weight, compute_kernel_config=self.compute))


class _PrefillDecoder(MultichipDecoder):
    def prefill_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if self.prefill_hybrid_enabled and hidden_states.shape[2] == 1024:
            if slots is None or (self.ple is not None and tokens is None):
                raise ValueError("Hybrid serving prefill requires request slots and PLE token IDs")
            return forward1024(
                self, hidden_states, positions=current_positions, slots=slots, tokens=tokens, page_table=page_table
            )
        return super().prefill_forward(
            hidden_states, current_positions=current_positions, slots=slots, tokens=tokens, page_table=page_table
        )

    def decode_forward(self, *args, **kwargs):
        raise RuntimeError("Serial prefill view cannot decode; use the fixed-batch owner")


def _clone(owner, ops, cls=None):
    result = copy.copy(owner)
    if cls is not None:
        result.__class__ = cls
    result.ops = ops
    return result


def _ops(owner):
    result = copy.copy(owner)
    result.__class__ = _PrefillOps
    result.gather_outputs = {}
    result.reduce_outputs = {}
    result.large_prefill = False
    result.l1_prefill = False
    result.norm_compaction = {}
    result.norm_unpacked_sequences = set()
    return result


def _shared_zero(pool, name, reference, shape, mesh):
    """Reuse only equivalent scratch, allocated once before any trace exists.

    These are the same serialized lifetimes as model.py's serving helpers:
    GDN gathers/scatters complete work, MoE rewrites live dispatch rows before
    reading, and QSA gather fills/masks scoring scratch before the next reader.
    Mutable work and immutable zero sources have distinct names/allocations.
    """
    shape = tuple(shape)
    for old_mesh, tensor in pool.setdefault(name, []):
        if (
            old_mesh is mesh
            and tuple(tensor.shape) == shape
            and tensor.dtype == reference.dtype
            and tensor.layout == reference.layout
            and tensor.memory_config() == reference.memory_config()
        ):
            return tensor
    tensor = ttnn.zeros(
        shape,
        dtype=reference.dtype,
        layout=reference.layout,
        memory_config=reference.memory_config(),
        device=mesh,
    )
    pool[name].append((mesh, tensor))
    return tensor


def _decoder(owner, pool, lengths):
    hybrid_enabled = owner.batch_size == 32 and 1024 in lengths
    # Attention/PLE/norm geometry stays512. Only an exact1024 call can join
    # independent MoE rows; every other length retains original512 chunking.
    lengths = sorted({min(512, n - start) for n in lengths for start in range(0, n, 512)})
    chunk_size = max(lengths)
    ops = _ops(owner.ops)
    result = _clone(owner, ops, _PrefillDecoder)
    result.batch_size = 1
    result.chunk_size = chunk_size
    result.lengths = {}
    result.prefill_hybrid_enabled = hybrid_enabled
    result.prefill_hybrid_signatures = {}
    result.attn_hyper = _clone(owner.attn_hyper, ops)
    result.mlp_hyper = _clone(owner.mlp_hyper, ops)
    result.mlp = _clone(owner.mlp, ops)
    result.mlp.capacity = _shared_zero(pool, "moe_capacity", owner.mlp.capacity, owner.mlp.capacity.shape, owner.mesh)
    result.mlp.prefill_group_size = 512
    result.mlp.offsets256 = ops.upload(torch.arange(128, dtype=torch.int32) * 256, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    result.mlp.offsets512 = ops.upload(torch.arange(128, dtype=torch.int32) * 512, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    if hybrid_enabled:
        result.mlp.offsets1024 = ops.upload(
            torch.arange(128, dtype=torch.int32) * 1024, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
    attention = _clone(owner.attention, ops)
    result.attention = attention
    attention.batch_size = 1
    attention.chunk_size = chunk_size
    attention.output_projection = _clone(owner.attention.output_projection, ops, _RowOutputProjection)
    if owner.kind == "linear_attention":
        attention.__class__ = PrefillResidentGDN
        attention.prefill_resident_enabled = owner.batch_size == 32
        attention.prefill_resident_signatures = {}
        attention.prefill_qk_rm_enabled = True
        attention.prefill_qk_rm_signatures = {}
        for name in ("recurrent_state", "zero_recurrent", "conv_state", "zero_conv"):
            original = getattr(owner.attention, name)
            setattr(attention, name, _shared_zero(pool, name, original, (1, *tuple(original.shape)[1:]), owner.mesh))
    else:
        if not attention.paged_pooled:
            raise ValueError("Serial prefill requires the serving paged QSA cache")
        attention._decode_phase = False
        for name in ("_physical_update_indices", "_index_core", "_pooled_gather_partition"):
            if hasattr(attention, name):
                delattr(attention, name)
        attention.chunk_offsets = ops.upload(torch.arange(chunk_size).float().reshape(1, 1, -1, 1), ttnn.float32)
        attention.indexer_prefix = ops.upload(torch.zeros(1, 4, 31, 128))
        attention.indexer_weights = ops.upload(torch.ones(1, 1, chunk_size + 32, 4))
        attention.pooled_keys = _shared_zero(
            pool, "pooled_keys", owner.attention.pooled_keys, owner.attention.pooled_keys.shape, owner.mesh
        )
    if owner.ple is not None:
        result.ple = _clone(owner.ple, ops)
        result.ple.workspaces = {}
        result.ple.flat_norm_weights = {}
    result.prepare(lengths)
    return result


def _model(owner, lengths):
    if not owner.serving_pool or owner.batch_size <= 1:
        raise ValueError("Serial prefill view requires a multirow serving owner")
    result = _clone(owner, _ops(owner.ops))
    result.batch_size = 1
    result.terminal = _clone(owner.terminal, result.ops)
    pool = {}
    result.layers = [_decoder(layer, pool, lengths) for layer in owner.layers]
    # Work shape differs from the owner, so bind an independent mapping whose
    # public banks and paged caches still have the owner's identical addresses.
    result.cache = result.export_cache()
    result._bound_cache = result.cache
    for original, view in zip(owner.layers, result.layers):
        shared = (
            ("recurrent_bank", "conv_bank")
            if original.kind == "linear_attention"
            else ("packed_cache", "raw_key_cache", "pooled_cache")
        )
        for name in shared:
            if getattr(original.attention, name) is not getattr(view.attention, name):
                raise RuntimeError(f"Serial prefill lost public ownership of {name}")
        if original.ple is not None and original.ple_cache is not view.ple_cache:
            raise RuntimeError("Serial prefill lost public PLE ownership")
    return result


class SerialPrefillView:
    """Prefill logits only; one original-owner sampler call follows all rows."""

    def __init__(self, owner, output):
        self.owner, self.mesh, self.output = owner, owner.mesh, output
        self.buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        if owner.batch_size == 32:
            self.buckets.append(1024)
        self.model = _model(owner, self.buckets)
        self.inputs = {
            n: self.model.ops.upload(torch.zeros(1, n, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            for n in self.buckets
        }
        self.positions = self.model.ops.upload(
            torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        slot = torch.zeros(1, 32, dtype=torch.int32)
        slot[0, 0] = owner.max_requests
        self.slots = self.model.ops.upload(slot, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        scratch = (owner.scratch_page_base + torch.arange(owner.context // 32) % 16).reshape(1, -1).int()
        self.pages = self.model.ops.upload(scratch, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        width = owner.cfg.vocab_size // 4
        self.logits = ttnn.empty((1, 1, 1, width), dtype=owner.logits_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        self.rows = ttnn.zeros(
            (1, 1, owner.batch_size, width), dtype=owner.logits_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
        )
        self.counters = Counter()
        self.last_physical_chunks = []

    def _copy(self, value, target):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def _forward(self, n):
        logits = self.model.forward(
            self.inputs[n],
            positions=self.positions,
            slots=self.slots,
            page_table=self.pages,
            kv_cache=self.model.cache,
            last_token_only=True,
        )
        ttnn.copy(logits, self.logits)

    def _store_row(self, row):
        # CQ0 orders consumption of this conversion temporary before the next
        # replay. Its Python reference does not survive this method, and it has
        # no persistent alias; device completion remains asynchronous.
        logits = ttnn.to_layout(self.logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            logits, self.rows, [0, 0, row, 0], [1, 1, row + 1, self.rows.shape[-1]], [1, 1, 1, 1]
        )

    def _finish_logits(self):
        ttnn.copy(ttnn.to_layout(self.rows, ttnn.TILE_LAYOUT), self.output)

    def _begin_logits(self):
        # Owner decode may have changed inactive logits since the last prefill.
        # Refresh all rows from their current owner before replacing real rows.
        ttnn.copy(ttnn.to_layout(self.output, ttnn.ROW_MAJOR_LAYOUT), self.rows)

    def warm(self):
        self._begin_logits()
        for n in self.buckets:
            print("PREPARE_PREFILL_VIEW", n, flush=True)
            self._forward(n)
        # Row offsets are runtime arguments for interleaved slice_write. Warm
        # every destination anyway, including the last row, before any capture.
        for row in range(self.owner.batch_size):
            self._store_row(row)
        self._finish_logits()

    def capture(self, traces, event):
        for n in self.buckets:
            name = f"prefill_view_{n}"
            print("CAPTURE", name, flush=True)
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._forward(n)
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
            traces[name] = trace
            event("capture", signature=name, trace_id=trace, physical_batch=1)

    def forward(self, tokens, lengths, starts, slots, pages, active, *, replay, event):
        self.last_physical_chunks = []
        self._begin_logits()
        for row, enabled in enumerate(active):
            if not enabled:
                continue
            slot = torch.zeros(1, 32, dtype=torch.int32)
            slot[0, 0] = slots[row]
            self._copy(slot, self.slots)
            self._copy(pages[row : row + 1], self.pages)
            offset = 0
            while offset < lengths[row]:
                n = max(size for size in self.buckets if size <= lengths[row] - offset)
                position = torch.zeros(1, 32, dtype=torch.int32)
                position[0, 0] = int(starts[row]) + offset
                self._copy(position, self.positions)
                self._copy(tokens[row : row + 1, offset : offset + n], self.inputs[n])
                replay(f"prefill_view_{n}")
                self.counters[f"chunk_{n}"] += 1
                physical_chunk = dict(
                    chunk_length=n,
                    start_positions=[int(position[0, 0])],
                    active=[True],
                    batch=1,
                    logical_row=row,
                    physical_slot=slots[row],
                )
                if n == 1024:
                    physical_chunk.update(
                        non_moe_chunks=[
                            dict(
                                chunk_length=min(512, n - start),
                                start_positions=[int(position[0, 0]) + start],
                                active=[True],
                                batch=1,
                            )
                            for start in range(0, n, 512)
                        ],
                        moe_group_rows=n,
                        lm_head_invocations=1,
                    )
                self.last_physical_chunks.append(physical_chunk)
                event(
                    "prefill_view_chunk",
                    logical_row=row,
                    physical_slot=slots[row],
                    physical_batch=1,
                    chunk_length=n,
                    start_position=int(position[0, 0]),
                )
                offset += n
            self._store_row(row)
        self._finish_logits()

    def runtime_summary(self):
        layers = []
        for index, owner, view in zip(self.owner.layer_indices, self.owner.layers, self.model.layers):
            names = (
                ("recurrent_bank", "conv_bank")
                if owner.kind == "linear_attention"
                else ("packed_cache", "raw_key_cache", "pooled_cache")
            )
            layers.append(
                dict(
                    layer=index,
                    public_state={
                        name: dict(
                            shared=getattr(owner.attention, name) is getattr(view.attention, name),
                            buffer_address=getattr(view.attention, name).buffer_address(),
                        )
                        for name in names
                    },
                    ple_cache_shared=owner.ple is None or owner.ple_cache is view.ple_cache,
                    recurrence_broadcast={
                        "owner": getattr(owner.attention, "recurrence_broadcast", False),
                        "view": getattr(view.attention, "recurrence_broadcast", False),
                    },
                )
            )
        return dict(
            execution_batch=1,
            last_token_only=True,
            server_slots=self.owner.batch_size,
            buckets=self.buckets,
            context=self.model.context,
            public_pool_tokens=self.model.public_pool_tokens,
            layer_indices=self.model.layer_indices,
            precision=self.model.precision_policy,
            layers=layers,
            hybrid1024=[
                {
                    "layer": index,
                    "enabled": layer.prefill_hybrid_enabled,
                    "owner_enabled": getattr(original, "prefill_hybrid_enabled", False),
                    "signatures": list(layer.prefill_hybrid_signatures.values()),
                }
                for index, original, layer in zip(self.model.layer_indices, self.owner.layers, self.model.layers)
            ],
            terminal_norm_1024_compact="1024" in self.model.ops.norm_compaction,
            terminal_norm_1024_unpacked=1024 in self.model.ops.norm_unpacked_sequences,
            gdn_recurrence_broadcast=[
                {"layer": index, "enabled": layer.attention.recurrence_broadcast}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind == "linear_attention"
            ],
            gdn_prefill_resident=[
                {
                    "layer": index,
                    "enabled": layer.attention.prefill_resident_enabled,
                    "owner_enabled": getattr(original.attention, "prefill_resident_enabled", False),
                    "signatures": list(layer.attention.prefill_resident_signatures.values()),
                }
                for index, original, layer in zip(self.model.layer_indices, self.owner.layers, self.model.layers)
                if layer.kind == "linear_attention"
            ],
            gdn_prefill_qk_rm=[
                {
                    "layer": index,
                    "enabled": layer.attention.prefill_qk_rm_enabled,
                    "owner_enabled": getattr(original.attention, "prefill_qk_rm_enabled", False),
                    "signatures": list(layer.attention.prefill_qk_rm_signatures.values()),
                }
                for index, original, layer in zip(self.model.layer_indices, self.owner.layers, self.model.layers)
                if layer.kind == "linear_attention"
            ],
            pooled_gather_partitions=[
                {"layer": index, **getattr(layer.attention, "_pooled_gather_partition", {})}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind != "linear_attention"
            ],
            private_work_allocations={
                name: len(
                    {
                        id(getattr(layer.attention, name))
                        for layer in self.model.layers
                        if hasattr(layer.attention, name)
                    }
                )
                for name in ("recurrent_state", "conv_state", "zero_recurrent", "zero_conv", "pooled_keys")
            },
            moe_capacity_allocations=len({id(layer.mlp.capacity) for layer in self.model.layers}),
            norm_compaction=dict(
                layers=[
                    {"layer": index, "signatures": list(layer.ops.norm_compaction.values())}
                    for index, layer in zip(self.model.layer_indices, self.model.layers)
                ],
                terminal_signatures=list(self.model.ops.norm_compaction.values()),
            ),
            moe_prefill_group_size=[
                {
                    "layer": index,
                    "group_size": layer.mlp.prefill_group_size,
                    "capacity_rows": layer.mlp.prefill_group_size,
                }
                for index, layer in zip(self.model.layer_indices, self.model.layers)
            ],
            counters=dict(self.counters),
        )
