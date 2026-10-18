# SPDX-License-Identifier: Apache-2.0
"""Exactly1024 serial prefill: original512 non-MoE geometry, joined expert group.

Weights, precision, cache banks and sampling remain owned by the original model.
This helper is enabled only by the fixed32-slot owner's serial prefill view.
"""

import math

import ttnn

from .multichip_decoder import _EXPERT_COMPACT_PACKED, _EXPERT_DISPATCH_PACKED, _expert_capacity_source, _expert_kernel

DISPATCH1024 = _expert_capacity_source(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*1024+count")
COMPACT1024 = _expert_capacity_source(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<1024)")
COMPACT1024 = _expert_capacity_source(COMPACT1024, "(e%128)*4+packed/32", "(e%128)*32+packed/32")


def group1024(self, x):
    if tuple(x.shape) != (1, 1, 1024, 2560):
        raise ValueError("Hybrid expert group requires exactly1024 serial rows")
    n = x.shape[2]
    padded = math.ceil(n / 32) * 32
    router = self.ops.linear(x, self.router, compute_kernel_config=self.router_compute)
    logits = router[..., :512]
    values, indices = ttnn.topk(
        ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
    )
    selected = ttnn.softmax(
        ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
    )
    ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
    counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
    mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
    capacity = ttnn.empty(
        (1, 1, 128 * 1024, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
    )
    self.hybrid_capacity_rows = int(capacity.shape[2]) // 128
    counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
    source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    _expert_kernel(
        self.ops.mesh,
        [source, ids, scores, self.rank, counts, mapping, capacity],
        DISPATCH1024,
        [n],
        workers=32,
        cb_size=5120 + n * 192 + 64,
    )
    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        capacity,
        self.offsets1024,
        counts,
        self.ids,
        self.gates,
        self.ups,
        self.downs,
        1024,
        compute_kernel_config=self.routed_compute,
    )
    compact = ttnn.empty(
        (1, 10, padded, 2560),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=self.ops.mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    _expert_kernel(
        self.ops.mesh,
        [down, ids, scores, self.rank, mapping, compact],
        COMPACT1024,
        [n],
        workers=80,
        cb_size=2176 + n * 320 + 64,
    )
    scores = (
        ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0) if n < padded else selected[..., :10]
    )
    weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
    result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
    shared = self.ops.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
    shared = self.ops.linear(
        ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
        self.shared_down,
        compute_kernel_config=self.shared_compute,
    )
    return ttnn.add(
        result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    )


def forward1024(self, x, *, positions, slots, tokens, page_table):
    if tuple(x.shape) != (1, 1, 1024, 2560) or not self.prefill_hybrid_enabled:
        raise ValueError("Hybrid prefill requires the enabled serial1024 view")
    length = 1024
    old_large, old_l1 = self.ops.large_prefill, self.ops.l1_prefill
    pieces, mixtures, injections = [], [], []
    observed_pieces, observed_l1 = [], []
    retained_memory, retained_shapes = [], []
    try:
        for start in range(0, length, 512):
            stop = min(start + 512, length)
            size = stop - start
            self.ops.large_prefill, self.ops.l1_prefill = size >= 256, True
            value = x[:, :, start:stop, :]
            observed_pieces.append(int(value.shape[2]))
            observed_l1.append(bool(self.ops.l1_prefill))
            position = positions if start == 0 else ttnn.add(positions, start)
            ids = None if tokens is None else tokens[:, start:stop]
            if self.ple is not None:
                value = ttnn.add(
                    value,
                    self.ple.forward(
                        value,
                        tokens=ids,
                        positions=position,
                        slots=slots,
                        valid_lengths=self.lengths[size],
                        cache=self.ple_cache,
                    ),
                )
            mixed, inject = self.attn_hyper.read(value, decode=False)
            if self.kind == "linear_attention":
                update = self.attention.prefill_forward(mixed, current_positions=position, slots=slots)
            else:
                update = self.attention.prefill_forward(mixed, current_pos=position, page_table=page_table, slots=slots)
            value = self.attn_hyper.write(value, update, inject)
            mixed, inject = self.mlp_hyper.read(value, decode=False)
            # A completed HC residual is L1-backed. Retaining it while the
            # next512 attention compiles/executes can collide with its static
            # kernel buffers in the full48 stack. Stage completed pieces just
            # as the original chunk loop does; CQ0 orders the copies without
            # introducing a device synchronization or changing arithmetic.
            pieces.append(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG))
            mixtures.append(ttnn.to_memory_config(mixed, ttnn.DRAM_MEMORY_CONFIG))
            injections.append(ttnn.to_memory_config(inject, ttnn.DRAM_MEMORY_CONFIG))
            retained = {"value": pieces[-1], "mixed": mixtures[-1], "inject": injections[-1]}
            retained_memory.append({name: str(tensor.memory_config()) for name, tensor in retained.items()})
            retained_shapes.append({name: list(tensor.shape) for name, tensor in retained.items()})
            del retained, value, mixed, inject, update
        # The already-qualified standalone1024 component uses DRAM matmul
        # intermediates. Keep that policy;512 non-MoE work remains in L1.
        self.ops.large_prefill, self.ops.l1_prefill = True, False
        joined = ttnn.concat(mixtures, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        observed_group = int(joined.shape[2])
        observed_moe_l1 = bool(self.ops.l1_prefill)
        update = self.ops.scatter_reduce(group1024(self.mlp, joined))
        del joined, mixtures
        outputs = []
        for start, value, injection in zip(range(0, length, 512), pieces, injections, strict=True):
            stop = min(start + 512, length)
            self.ops.large_prefill, self.ops.l1_prefill = stop - start >= 256, True
            out = self.mlp_hyper.write(value, update[:, :, start:stop, :], injection)
            outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
            del out
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.prefill_hybrid_signatures["1024"] = dict(
            sequence=int(x.shape[2]),
            non_moe_chunks=observed_pieces,
            moe_group_rows=observed_group,
            capacity_rows=self.mlp.hybrid_capacity_rows,
            non_moe_l1=observed_l1,
            moe_l1=observed_moe_l1,
            attention_chunk_size=self.attention.chunk_size,
            retained_piece_memory_configs=retained_memory,
            retained_piece_shapes=retained_shapes,
        )
        return result
    finally:
        self.ops.large_prefill, self.ops.l1_prefill = old_large, old_l1
