# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 TP4 attention and hybrid EP4 / indexed-TP4 routed decoder.

The optimized decoder is the single-chip numerical baseline. Construction packs
all weights; forwards use device tensors only. The selected residual is replicated
[1,S,2048] on a 1x4 Blackhole mesh; hidden-sharded512 remains a measured candidate.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

from .fused_decoder import (
    DECAY_LOOKUP_EMPTY,
    DECAY_LOOKUP_READER,
    DECODE_CONV_COMPUTE,
    DECODE_CONV_READER,
    DECODE_CONV_WRITER,
    DRAM,
    LOCAL_SILU_COMPUTE_SOURCE,
    LOCAL_SILU_READER_SOURCE,
    SILU_COMPACT_ENTRIES,
    Chunk64MKLNegativeExp,
    Chunk64OutputGraph,
    DirectCoefficientStreamingScan,
    WeightedNormTail,
    compact_silu_math_table,
    local_silu_math_table,
    reshape,
)
from .mesh_compensated_norm import CompensatedNormEpilogue, normalized_singleton
from .mesh_fp32_topk import MeshFP32TopK
from .optimized_decoder import OptimizedDecoder
from .precision import actual_compute, assert_compute, device_compute, device_dtype, resolve_precision
from .slot_state import copy_if, masked_convolution_reader


def _bounded_concat(tensors, *, dim, fan_in):
    """Preserve logical order without exceeding native concat input limits."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("concat requires at least one tensor")
    while len(tensors) > 1:
        tensors = [
            ttnn.concat(tensors[i : i + fan_in], dim=dim) if len(tensors[i : i + fan_in]) > 1 else tensors[i]
            for i in range(0, len(tensors), fan_in)
        ]
    return tensors[0]


class MeshCCL(CCLManager):
    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


@dataclass(frozen=True)
class MeshPolicy:
    residual_layout: str = "replicated"
    num_links: int = 2
    moe_chunk: int = 2048


# This edit changes the loop bound, never its BF16-code arithmetic.
TP4_DECAY_LOOKUP_READER = DECAY_LOOKUP_READER.replace("h<32", "h<8")
# Debug temporary has only 64 qkv tiles per chip.
TP4_DECODE_CONV_WRITER = DECODE_CONV_WRITER.replace("j*256+tile", "j*64+tile")


class TP4WeightedNormTail(WeightedNormTail):
    def __call__(self, core, sums):
        assert tuple(core.shape) == (1, 8, 128)
        assert tuple(sums.shape) == (1, 8, 1)
        # The physical tile still contains32 rows. Native arithmetic remains
        # unchanged; the extra24 rows cannot affect a valid row.
        core = ttnn.pad(core, [(0, 0), (0, 24), (0, 0)], 0.0)
        sums = ttnn.pad(sums, [(0, 0), (0, 24), (0, 0)], 0.0)
        value, debug = super().__call__(core, sums)
        # Retain physical32 rows for the exact interleaved head-row reader.
        return value, debug


class TP4LocalSiluMultiply:
    """Eight workers, compact immutable LUT, no runtime tensor conversions.

    Weighted FP32 carriers [1,32,128] have already been rounded to BF16.
    Z is BF16 TILE [1,1,1024]. Output directly matches local head concat:
    BF16 TILE [1,1,1024], padded [1,32,1024], eight width shards [32,128].
    """

    def __init__(self, t, mesh, debug=False):
        self.t, self.mesh, self.debug = (t, mesh, debug)
        grid = mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= 8
        self.coordinates = [(i % grid.x, i // grid.x) for i in range(8)]
        self.cores = t.num_cores_to_corerangeset(8, grid, row_wise=True)
        self.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        table_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (1, SILU_COMPACT_ENTRIES), t.ShardOrientation.ROW_MAJOR),
        )
        self.table = t.from_torch(
            compact_silu_math_table(local_silu_math_table()).expand(8, SILU_COMPACT_ENTRIES).contiguous(),
            dtype=t.float32,
            layout=t.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=table_memory,
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes

    def __call__(self, weighted, z):
        t = self.t
        assert tuple(weighted.shape) == (1, 32, 128) and weighted.dtype == t.float32
        assert tuple(z.shape) == (1, 1, 1024) and z.dtype == t.bfloat16
        assert weighted.layout == z.layout == t.TILE_LAYOUT
        output = t.empty(
            (1, 1, 1024), dtype=t.bfloat16, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
        )
        bindings = [(16, output)]
        gate = None
        if self.debug:
            gate = t.empty(
                (1, 1, 1024), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
            )
            bindings.append((17, gate))
        runtime = t.RuntimeArgs()
        for head, (x, y) in enumerate(self.coordinates):
            runtime[x][y] = [weighted.buffer_address(), z.buffer_address(), self.table.buffer_address(), head]
        accessor_args = t.TensorAccessorArgs(weighted).get_compile_time_args()
        accessor_args += t.TensorAccessorArgs(z).get_compile_time_args()

        def kernel(source, config, compile_args, runtime_args):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=config,
                compile_time_args=compile_args,
                runtime_args=runtime_args,
            )

        cbs = [t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings]
        for index, dtype, size, pages in ((0, t.float32, 4096, 2), (1, t.float32, 4096, 2), (2, t.bfloat16, 2048, 1)):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, size)],
                )
            )
        program = t.ProgramDescriptor(
            kernels=[
                kernel(LOCAL_SILU_READER_SOURCE, t.ReaderConfigDescriptor(), accessor_args, runtime),
                kernel(LOCAL_SILU_COMPUTE_SOURCE, self.config, [int(self.debug)], []),
            ],
            cbs=cbs,
        )
        t.generic_op([weighted, z, self.table, *[v for _, v in bindings]], program)
        return (output, gate)


class TP4DecodeConvolution:
    def __init__(self, t, mesh, taps, max_cores=64):
        self.t, self.mesh, self.taps, self.max_cores = (t, mesh, taps, max_cores)

    def __call__(self, qkv, state, debug=False, *, activity=None):
        t = self.t
        assert tuple(qkv.shape) == (1, 1, 2048) and qkv.dtype == t.bfloat16 and (qkv.layout == t.TILE_LAYOUT)
        assert tuple(state.shape) == (1, 3, 2048) and state.dtype == t.bfloat16 and (state.layout == t.ROW_MAJOR_LAYOUT)
        assert state.memory_config() == t.DRAM_MEMORY_CONFIG
        assert qkv.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert all(
            (tuple(w.shape) == (1, 1, 2048) and w.dtype == t.float32 and (w.layout == t.TILE_LAYOUT) for w in self.taps)
        )
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(64, grid.x * grid.y, self.max_cores)
        partitions = []
        start = 0
        for h in range(active):
            count = 64 // active + (h < 64 % active)
            partitions.append((h % grid.x, h // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            (1, 1, 2048), device=self.mesh, dtype=t.bfloat16, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        partial = (
            t.empty(
                (4, 1, 1, 2048),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            if debug
            else None
        )
        rd, cr, wr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partitions:
            rd[x][y] = [
                qkv.buffer_address(),
                state.buffer_address(),
                *[w.buffer_address() for w in self.taps],
                start,
                count,
            ] + ([activity.buffer_address()] if activity is not None else [])
            cr[x][y] = [count]
            wr[x][y] = [out.buffer_address(), partial.buffer_address() if debug else 0, start, count]
        accessors = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (qkv, state, *self.taps)]
        offsets = []
        cursor = 5
        for a in accessors:
            offsets.append(cursor)
            cursor += len(a)
        rdargs = offsets[1:] + sum(accessors, [])
        reader_source = DECODE_CONV_READER
        if activity is not None:
            # Only fixed-slot decode uses this predicate. Keep the selected
            # unmasked source and argument sequence unchanged for solo decode.
            reader_source = masked_convolution_reader(reader_source, len(rdargs))
            rdargs += list(t.TensorAccessorArgs(activity).get_compile_time_args())
        oa = list(t.TensorAccessorArgs(out).get_compile_time_args())
        pa = list(t.TensorAccessorArgs(partial if debug else out).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        cb_specs = [
            (0, 2, 4096, t.float32),
            (1, 2, 4096, t.float32),
            (2, 1, 2048, t.bfloat16),
            (3, 1, 4096, t.float32),
            (24, 2, 4096, t.float32),
            (16, 1, 2048, t.bfloat16),
        ]
        if debug:
            cb_specs.append((17, 1, 4096, t.float32))
        program = t.ProgramDescriptor(
            kernels=[
                kernel(reader_source, t.ReaderConfigDescriptor(), rdargs, rd),
                kernel(DECODE_CONV_COMPUTE, cfg, [int(debug)], cr),
                kernel(TP4_DECODE_CONV_WRITER, t.WriterConfigDescriptor(), [int(debug), 2 + len(oa)] + oa + pa, wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, dtype, size)],
                )
                for i, pages, size, dtype in cb_specs
            ],
        )
        t.generic_op(
            [qkv, state, *self.taps, out] + ([partial] if debug else []) + ([activity] if activity is not None else []),
            program,
        )
        return (out, partial)


class TP4DecodeDecayLookup:
    def __init__(self, t, mesh, table, lo=None, hi=None, max_cores=32):
        self.t, self.mesh, self.table = (t, mesh, table)
        self.lo, self.hi, self.max_cores = (lo, hi, max_cores)
        assert tuple(table.shape) in ((8, 16384), (8, 65536))
        assert table.dtype == t.float32 and table.layout == t.ROW_MAJOR_LAYOUT
        assert table.memory_config() == t.DRAM_MEMORY_CONFIG
        assert (lo is not None and hi is not None) == (table.shape[1] == 16384)

    def __call__(self, projection_a):
        t, table = (self.t, self.table)
        assert projection_a.dtype == t.bfloat16 and projection_a.layout == t.TILE_LAYOUT
        assert len(projection_a.shape) == 3 and projection_a.shape[0] == 1 and (projection_a.shape[2] == 8)
        assert projection_a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        rows = projection_a.shape[1]
        tiles = (rows + 31) // 32
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(tiles, grid.x * grid.y, self.max_cores)
        partitions, start = ([], 0)
        for i in range(active):
            count = tiles // active + (i < tiles % active)
            partitions.append((i % grid.x, i // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        output = t.empty(
            tuple(projection_a.shape),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        ia, ta, oa = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (projection_a, table, output)]
        args = [5 + len(ia), 5 + len(ia) + len(ta), int(self.lo is not None), self.lo or 0, self.hi or 0] + ia + ta + oa
        runtime = t.RuntimeArgs()
        for x, y, start, count in partitions:
            runtime[x][y] = [
                projection_a.buffer_address(),
                table.buffer_address(),
                output.buffer_address(),
                start,
                count,
                rows,
            ]
        program = t.ProgramDescriptor(
            kernels=[
                t.KernelDescriptor(
                    kernel_source=TP4_DECAY_LOOKUP_READER,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ReaderConfigDescriptor(),
                    compile_time_args=args,
                    runtime_args=runtime,
                ),
                t.KernelDescriptor(
                    kernel_source=DECAY_LOOKUP_EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.WriterConfigDescriptor(),
                ),
                t.KernelDescriptor(
                    kernel_source=DECAY_LOOKUP_EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ComputeConfigDescriptor(),
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=12288, core_ranges=cores, format_descriptors=[t.CBFormatDescriptor(0, t.float32, 4096)]
                )
            ],
        )
        t.generic_op([projection_a, table, output], program)
        return output


class LinearTP4Mixin:
    def _recurrent_sum(self, products):
        direct = getattr(self, "full_model_recurrent_sum", None)
        if direct is not None and direct.supports(products):
            return direct(products)
        head_rows = ttnn.permute(products, (0, 2, 1, 3))
        chunks = ttnn.reshape(head_rows, (8, 16, 8, 128))
        total = ttnn.experimental.fast_reduce_nc(
            chunks, dims=[0, 1], compute_kernel_config=self.compute, memory_config=products.memory_config()
        )
        return ttnn.permute(total[:, :, :8, :], (0, 2, 1, 3))

    def _recurrent_sum_terminal(self, products):
        direct = self.full_model_recurrent_sum
        if direct is not None and direct.supports(products):
            # Write directly in the next consumer's head-row layout; preserve
            # all FP32 add ordering while omitting inverse physical permutes.
            return direct(products, terminal_rows=True)
        return ttnn.permute(self._recurrent_sum(products), (0, 2, 1, 3))

    def _recurrent_step(self, q, k, v, g, beta, state, *, decay_value=None, round_state=False, output_state=None):
        """FP32 SFPU state update; native FDR narrows FP32 state through TF32 FPU inputs."""
        state_slab = getattr(self, "full_model_state_slab", None)
        if state_slab is not None and state_slab.supports_native(
            q, k, v, beta, state, decay_value, output_state=output_state
        ):
            return state_slab.from_native(
                q, k, v, beta, state, decay_value, round_state=round_state, output_state=output_state
            )
        q = ttnn.reshape(ttnn.permute(q, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        k = ttnn.reshape(ttnn.permute(k, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = reshape(beta, (1, 8, 1, 1))
        state_activations = self.round_widen_chain if round_state else []
        if decay_value is None:
            decay = reshape(g, (1, 8, 1, 1))
            decayed = ttnn.multiply(
                state,
                decay,
                input_tensor_a_activations=state_activations,
                input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            decayed = ttnn.multiply(
                state,
                reshape(decay_value, (1, 8, 1, 1)),
                input_tensor_a_activations=state_activations,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        product_sum = getattr(self, "full_model_product_sum", None)
        if product_sum is not None and product_sum.supports(decayed, k):
            memory = product_sum(decayed, k)
        else:
            decayed_heads = ttnn.reshape(decayed, (1, 4, 2, 128, 128))
            memory = self._recurrent_sum(ttnn.reshape(ttnn.multiply(decayed_heads, k), (1, 8, 128, 128)))
        delta = ttnn.multiply(ttnn.subtract(v, memory, memory_config=ttnn.L1_MEMORY_CONFIG), beta)
        update = ttnn.multiply(k, ttnn.reshape(delta, (1, 4, 2, 1, 128)), memory_config=ttnn.L1_MEMORY_CONFIG)
        new = ttnn.add(decayed, ttnn.reshape(update, (1, 8, 128, 128)), output_tensor=output_state)
        if product_sum is not None and product_sum.supports(new, q):
            return (product_sum(new, q, scale_q=True, terminal_rows=True), new)
        output_products = ttnn.reshape(
            ttnn.multiply(
                ttnn.reshape(new, (1, 4, 2, 128, 128)),
                q,
                input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 128 ** (-0.5))],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            (1, 8, 128, 128),
        )
        if getattr(self, "full_model_optimized", False):
            return (self._recurrent_sum_terminal(output_products), new)
        output = self._recurrent_sum(output_products)
        return (ttnn.permute(output, (0, 2, 1, 3)), new)

    def _linear_attention(self, x, state, *, decode, cache_boundary=True, projection_k=512, activity=None):
        s = x.shape[1]
        if self.pack_linear:
            projected = self.linear(x, "attention_combined", projection_k=projection_k)
            state_projected, output_projected = projected[:, :, :1600], projected[:, :, 1600:]
        else:
            state_projected = self.linear(
                x, "attention_packed", ttnn.bfloat16, projection_k=projection_k, terminal_scan=decode and cache_boundary
            )
            output_projected = self.linear(
                x, "output_only_packed", projection_k=projection_k, terminal_scan=decode and cache_boundary
            )
        qkv = ttnn.concat([output_projected[:, :, :512], state_projected[:, :, :1536]], dim=-1)
        projection_a = state_projected[:, :, 1536:1544]
        projection_b = state_projected[:, :, 1568:1576]
        projection_z = output_projected[:, :, 512:]
        if decode and cache_boundary:
            conv, _conv_partial = self.decode_convolution(qkv, state["conv"], debug=False, activity=activity)
            q, k, v = (conv[:, :, :512], conv[:, :, 512:1024], conv[:, :, 1024:])
        else:
            history = ttnn.concat([state["conv"], ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)], dim=1)
            tap_input = ttnn.to_layout(history[:, :s, :], ttnn.TILE_LAYOUT)
            accum = ttnn.multiply(tap_input, self.taps_fp32[0], dtype=ttnn.float32, fast_and_approximate_mode=False)
            history_rm = history
            for j in range(1, 4):
                tap_input = ttnn.to_layout(history_rm[:, j : j + s, :], ttnn.TILE_LAYOUT, dtype=ttnn.float32)
                accum = ttnn.addcmul(accum, tap_input, self.taps_fp32[j])
            conv = ttnn.unary_chain(accum, self.conv_post_chain)
            q, k, v = (conv[:, :, :512], conv[:, :, 512:1024], conv[:, :, 1024:])
            ttnn.slice(history_rm, (0, s, 0), (1, s + 3, 2048), output_tensor=state["conv"])
        v = reshape(v[:, :s, :], (1, s, 8, 128)) if decode else v

        def l2(a):
            def rounded(x):
                return ttnn.unary_chain(x, self.round_widen_chain)

            wide = ttnn.typecast(a, ttnn.float32)
            squares = ttnn.multiply(wide, wide, activations=self.round_widen_chain)
            sums = rounded(ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute))
            shifted = ttnn.add(sums, 1e-06, activations=self.round_widen_chain)
            inverse = ttnn.unary_chain(shifted, self.rsqrt_round_chain)
            return ttnn.multiply(wide, inverse, activations=self.round_widen_chain)

        qk = reshape(ttnn.concat([q[:, :s, :], k[:, :s, :]], dim=0), (2, s, 4, 128))
        qk = l2(qk)
        q, k = (qk[:1, :, :, :], qk[1:, :, :, :])
        b = ttnn.typecast(projection_b, ttnn.float32)
        if decode and cache_boundary:
            g = None
            decay_value = self.decode_decay_lookup(projection_a)
        else:
            raw_indices = ttnn.bitcast(projection_a, ttnn.uint16)
            if decode and self.compact_decay_table is not None:
                raw_indices = ttnn.typecast(raw_indices, ttnn.uint32)
                magnitude_indices = ttnn.add(
                    raw_indices,
                    raw_indices,
                    dtype=ttnn.uint32,
                    input_tensor_a_activations=self.decay_finite_index_chain,
                    input_tensor_b_activations=self.decay_nan_index_chain,
                )
                raw_indices = ttnn.add(
                    magnitude_indices,
                    raw_indices,
                    dtype=ttnn.uint32,
                    input_tensor_b_activations=self.decay_sign_index_chain,
                )
            indices = ttnn.reshape(ttnn.transpose(raw_indices, -1, -2), (8, s))
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)

            def lookup(table):
                values = ttnn.transpose(ttnn.gather(table, 1, indices), -1, -2)
                return ttnn.to_layout(ttnn.reshape(values, (1, s, 8)), ttnn.TILE_LAYOUT)

            g = None if decode else lookup(self.gate_table)
            decay_value = None
            if decode:
                decay_value = lookup(
                    self.compact_decay_table if self.compact_decay_table is not None else self.decay_table
                )
        beta = ttnn.unary_chain(b, self.beta_chain)
        if decode:
            initial = state["recurrent"]
            state_slab = getattr(self, "full_model_state_slab", None)
            conditional_commit = (
                getattr(self, "fuse_recurrent_state_commit", False)
                and activity is not None
                and state_slab is not None
                and state_slab.supports_native(
                    q, k, v, beta, initial, decay_value, output_state=initial, activity=activity
                )
            )
            if conditional_commit:
                # Only the state writer changes; every ordered arithmetic and
                # output operation remains the existing native state slab.
                o, new = state_slab.from_native(
                    q,
                    k,
                    v,
                    beta,
                    initial,
                    decay_value,
                    round_state=cache_boundary,
                    output_state=initial,
                    activity=activity,
                    native_qk_reader=getattr(self, "native_qk_reader", False),
                )
            else:
                o, new = self._recurrent_step(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial,
                    decay_value=decay_value,
                    round_state=cache_boundary,
                    output_state=state["recurrent"] if activity is None else None,
                )
                if activity is not None:
                    # The candidate is ordinary transient output, not a snapshot
                    # of persistent state. Inactive FP32 bytes are never written.
                    copy_if(ttnn.to_memory_config(new, DRAM), state["recurrent"], activity, self.mesh_device)
        else:
            qq = ttnn.permute(ttnn.repeat_interleave(q, 2, dim=2), (0, 2, 1, 3))
            kk = ttnn.permute(ttnn.repeat_interleave(k, 2, dim=2), (0, 2, 1, 3))
            vv = ttnn.permute(ttnn.typecast(reshape(v, (1, s, 8, 128)), ttnn.float32), (0, 2, 1, 3))
            bb = ttnn.permute(reshape(beta, (1, s, 8, 1)), (0, 2, 1, 3))
            gg = ttnn.permute(reshape(g, (1, s, 8, 1)), (0, 2, 1, 3))
            o, new, _ = self.chunk64_state(qq, kk, vv, gg, bb, state["recurrent"])
        if not decode:
            ttnn.copy(reshape(new, (1, 8, 128, 128)), state["recurrent"])
        if not decode:
            core = reshape(ttnn.typecast(o, ttnn.bfloat16), (1, s * 8, 128))
            wide = ttnn.typecast(core, ttnn.float32, memory_config=DRAM)
            squares = ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM)
            sums = ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=DRAM)
            variance = ttnn.multiply(
                sums, 1 / 128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM
            )
            shifted = ttnn.add(variance, self.config.rms_norm_eps, memory_config=DRAM)
            inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=DRAM)
            normalized_fp32 = ttnn.multiply(
                wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM
            )
            normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=DRAM)
            weighted = ttnn.multiply(
                normalized,
                self.w["linear_attn.norm.weight"],
                dtype=ttnn.float32,
                fast_and_approximate_mode=False,
                activations=self.round_widen_chain,
                memory_config=DRAM,
            )
            silu_indices = ttnn.bitcast(projection_z, ttnn.uint16)
            silu_indices = ttnn.reshape(ttnn.to_layout(silu_indices, ttnn.ROW_MAJOR_LAYOUT), (1, s * 1024))
            gate_rows = ttnn.gather(self.silu_table, 1, silu_indices)
            gate_rows = ttnn.to_layout(ttnn.reshape(gate_rows, (1, s * 8, 128)), ttnn.TILE_LAYOUT)
            gated = ttnn.multiply(
                weighted, gate_rows, dtype=ttnn.bfloat16, fast_and_approximate_mode=False, memory_config=DRAM
            )
            return self.linear(reshape(gated, (1, s, 1024)), "linear_attn.out_proj.weight")
        core_bf16 = ttnn.typecast(o, ttnn.bfloat16)
        core_rows = reshape(core_bf16, (1, 8, 128))
        wide = ttnn.typecast(core_rows, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        squares = ttnn.multiply(
            wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sums = ttnn.sum(
            squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        if cache_boundary:
            weighted, _ = self.decode_weighted_norm_tail(wide, sums)
            merged, _ = self.decode_local_silu(weighted, projection_z)
            return self.linear(
                merged,
                "linear_attn.out_proj.weight",
                ordered_out=cache_boundary,
                terminal_scan=decode and cache_boundary,
            )
        else:
            variance = ttnn.multiply(
                sums, 1 / 128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            shifted = ttnn.add(variance, self.config.rms_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG)
            normalized_fp32 = ttnn.multiply(
                wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            weighted = ttnn.multiply(
                normalized,
                self.w["linear_attn.norm.weight"],
                dtype=ttnn.float32,
                fast_and_approximate_mode=False,
                activations=self.round_widen_chain,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_rows = reshape(ttnn.typecast(projection_z, ttnn.float32), (1, 8, 128))
            gated = ttnn.multiply(
                weighted,
                gate_rows,
                input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                fast_and_approximate_mode=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        head_shard = ttnn.create_sharded_memory_config(
            (32, 128),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        gated = ttnn.to_memory_config(ttnn.reshape(gated, (1, 1, 8, 128)), head_shard)
        merged = ttnn.experimental.nlp_concat_heads_decode(gated, num_heads=8)
        merged = ttnn.reshape(merged, (1, 1, 1024), (1, 32, 1024))
        return self.linear(
            merged, "linear_attn.out_proj.weight", ordered_out=cache_boundary, terminal_scan=decode and cache_boundary
        )

    def new_recurrent_state(self):
        """One request; local conv [1,3,2048], state [1,8,128,128]."""
        return {
            "conv": ttnn.from_torch(
                torch.zeros(1, 3, 8192),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=2),
            ),
            "recurrent": ttnn.from_torch(
                torch.zeros(1, 32, 128, 128),
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            ),
        }

    def linear(
        self, x, key, dtype=ttnn.bfloat16, *, projection_k=512, ordered_out=False, terminal_scan=False, **kwargs
    ):
        if self.kind != "linear_attention" or key not in (
            "attention_packed",
            "output_only_packed",
            "attention_combined",
            "linear_attn.out_proj.weight",
        ):
            return super().linear(
                x, key, dtype, projection_k=projection_k, ordered_out=ordered_out, terminal_scan=terminal_scan, **kwargs
            )
        is_output = key == "linear_attn.out_proj.weight"
        # Prefill output uses the baseline's native projection. It emits a
        # local FP32 partial; parent must reduce before BF16 cast/residual add.
        if is_output and not ordered_out:
            return ttnn.linear(
                x, self.w[key], dtype=ttnn.float32, compute_kernel_config=self.compute, memory_config=DRAM
            )
        if x.shape[-2] != 1:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            rows = [
                ttnn.to_layout(
                    self.linear(x[:, r : r + 1, :], key, dtype, projection_k=1024),
                    ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=DRAM,
                )
                for r in range(x.shape[-2])
            ]
            return ttnn.to_layout(ttnn.concat(rows, dim=1), ttnn.TILE_LAYOUT, memory_config=DRAM)
        if is_output:
            total_k, width, projection_k = 1024, 2048, 512
            weight = self.scan_attention_out_weight
        else:
            total_k = 2048
            width = {"attention_packed": 1600, "output_only_packed": 1536, "attention_combined": 3136}[key]
            weight = (
                self.scan_combined_weight
                if key == "attention_combined"
                else self.scan_attention_weight
                if key == "attention_packed"
                else self.scan_output_weight
            )
        weight = ttnn.reshape(weight, (total_k // projection_k, 1, projection_k, width))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=DRAM)
        # DirectCoefficientStreamingScan pair-swaps x internally. Corresponding
        # immutable weight rows were pair-swapped at load time.
        partial = self.direct_coefficient_scan(weight, x)
        total = ttnn.experimental.fast_reduce_nc(
            partial, dims=[0], compute_kernel_config=self.compute, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        total = ttnn.reshape(total, (1, 1, width), (1, 32, width))
        return ttnn.to_memory_config(total, DRAM) if is_output else ttnn.typecast(total, dtype, memory_config=DRAM)


def install_linear_tp4(self, sd, upload):
    """Load-time replacement. upload(t,dtype,dim) shards the stated dimension.

    Use class MultichipDecoder(LinearTP4Mixin, OptimizedDecoder), or bind these
    methods explicitly. Parent collects output-projection FP32 partials and
    casts to BF16 only after reduction. No all-reduce is inserted here.
    """
    if self.kind != "linear_attention":
        return
    mesh = self.mesh_device
    qkv = sd["linear_attn.in_proj_qkv.weight"]
    q, k, v = qkv.split((2048, 2048, 4096), dim=0)
    z = sd["linear_attn.in_proj_z.weight"]
    a, b = [sd[f"linear_attn.in_proj_{name}.weight"] for name in ("a", "b")]
    order = torch.arange(2048).reshape(-1, 2).flip(1).reshape(-1)
    state_pieces, output_pieces, conv_pieces = [], [], []
    taps = sd["linear_attn.conv1d.weight"]
    tq, tk, tv = taps.split((2048, 2048, 4096), dim=0)
    for rank in range(4):
        qs, vs = slice(rank * 512, (rank + 1) * 512), slice(rank * 1024, (rank + 1) * 1024)
        hs = slice(rank * 8, (rank + 1) * 8)
        state_pieces.append(
            torch.cat(
                (
                    k[qs],
                    v[vs],
                    torch.nn.functional.pad(a[hs], (0, 0, 0, 24)),
                    torch.nn.functional.pad(b[hs], (0, 0, 0, 24)),
                ),
                dim=0,
            ).T[order]
        )
        output_pieces.append(torch.cat((q[qs], z[vs]), dim=0).T[order])
        conv_pieces.append(torch.cat((tq[qs], tk[qs], tv[vs]), dim=0))
    if self.pack_linear:
        combined = torch.cat([torch.cat((a, b), dim=1) for a, b in zip(state_pieces, output_pieces)], dim=1)
        self.scan_combined_weight = upload(combined.reshape(1, 1, 2048, 12544), ttnn.bfloat16, dim=3)
        # The parent constructed replicated single-chip scans. Only the local
        # packed scan is consumed by this branch; release superseded storage.
        del self.scan_attention_weight, self.scan_output_weight
    else:
        self.scan_attention_weight = upload(
            torch.cat(state_pieces, dim=1).reshape(1, 1, 2048, 6400), ttnn.bfloat16, dim=3
        )
        self.scan_output_weight = upload(
            torch.cat(output_pieces, dim=1).reshape(1, 1, 2048, 6144), ttnn.bfloat16, dim=3
        )
    output_weight = sd["linear_attn.out_proj.weight"].T
    self.w["linear_attn.out_proj.weight"] = upload(output_weight, ttnn.bfloat16, dim=0)
    output_order = torch.arange(4096).reshape(-1, 2).flip(1).reshape(-1)
    self.scan_attention_out_weight = upload(output_weight[output_order].reshape(1, 1, 4096, 2048), ttnn.bfloat16, dim=2)
    for key in ("attention_packed", "output_only_packed"):
        self.w.pop(key, None)
    for key in ("linear_attn.out_proj.weight", "output_only_packed"):
        self.decode_linear.pop(key, None)
    conv = torch.cat(conv_pieces, dim=0)
    self.taps_fp32 = [upload(conv[:, 0, j].reshape(1, 1, 8192), ttnn.float32, dim=2) for j in range(4)]

    def row_upload(value, dim=None):
        mapper = ttnn.ReplicateTensorToMesh(mesh) if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim)
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=mapper,
        )

    codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
    gates = -sd["linear_attn.A_log"].float().exp().reshape(32, 1) * torch.nn.functional.softplus(
        codes.reshape(1, -1) + sd["linear_attn.dt_bias"].float().reshape(32, 1)
    )
    decays = gates.exp()
    self.gate_table, self.decay_table = row_upload(gates, 0), row_upload(decays, 0)
    # Retaining uncompressed exact LUT avoids per-device clamp-window divergence.
    # 8x65536 FP32 is2MiB/chip; optimize compact-table storage after correctness.
    self.compact_decay_table = None
    self.decode_decay_lookup = TP4DecodeDecayLookup(ttnn, mesh, self.decay_table)
    self.decode_convolution = TP4DecodeConvolution(ttnn, mesh, self.taps_fp32)
    self.decode_weighted_norm_tail = TP4WeightedNormTail(ttnn, mesh, sd["linear_attn.norm.weight"])
    if self.decode_tile_silu:
        # Local import avoids the helper's dependency on TP4LocalSiluMultiply
        # during this module's initialization.
        from .mesh_decode_silu import TP4TileSilu

        self.decode_local_silu = TP4TileSilu(ttnn, mesh)
    else:
        self.decode_local_silu = TP4LocalSiluMultiply(ttnn, mesh)
    self.direct_coefficient_scan = DirectCoefficientStreamingScan(ttnn, mesh)
    self.chunk64_state = Chunk64OutputGraph(
        ttnn, mesh, lambda x: upload(x, ttnn.float32), variant="sfpu_fma", cumsum="compensated"
    )
    self.chunk64_state.mkl_negative_exp = Chunk64MKLNegativeExp(
        ttnn, mesh, lambda x: upload(x, ttnn.float32), DRAM, max_shape=(1, 8, 64)
    )


class CollectiveWorkspace:
    """Serial CQ0 scratch shared across a decoder stack; prepare before capture.

    Every layer's combine consumes the EP output before the next layer writes it.
    The backing allocations and CCL semaphores outlive all traces using them.
    Concurrent forwards using this workspace are unsupported.
    """

    def __init__(self, mesh_device, num_links=2):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.ccl = MeshCCL(mesh_device, num_links, ttnn.Topology.Linear)
        self.ep_outputs = {}
        self.direct_ar_buffers = {}
        self.direct_ar_semaphore = None


def dispatch_capacity(tokens, expert_count=64):
    # Bound persistent workspace variants without padding the logical input.
    # A 1024-token input keeps its1024 capacity bucket, never the2048 bucket.
    bucket = max(32, 1 << (tokens - 1).bit_length())
    return ((8 * bucket + 31 * expert_count + 31) // 32) * 32


class MultichipDecoder(LinearTP4Mixin, OptimizedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        residual_layout="replicated",
        num_links=2,
        linear_norm="ordered",
        full_norm="ordered",
        decode_projection="dram",
        attention_collective="separate",
        gather_projection="separate",
        replicated_collective=None,
        gather_rows=64,
        direct_ar_fp32=True,
        local_expert_dispatch=True,
        sparse_geometry=None,
        workspace=None,
        batch_attention_reduce=True,
        batch_direct_ar=True,
        batch_full_projections=True,
        pack_linear=True,
        batch_shared_projections=False,
        split_rows_once=True,
        projection_precision=None,
        ccl_dtype="bf16",
        kv_dtype="bf16",
        full_projection_split=False,
        decode_fp32_selector=True,
        decode_compensated_norm=True,
        decode_tile_silu=True,
        decode_full_input_shard=True,
        full_model_optimized=False,
        precision_config=None,
    ):
        import torch

        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Qwen multichip decoder requires the target 1x4 mesh")
        if residual_layout not in ("sharded", "replicated"):
            raise ValueError("unknown residual contract")
        self = super().from_state_dict(state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device)
        self.full_model_optimized = full_model_optimized
        self.full_model_recurrent_sum = None
        self.full_model_product_sum = None
        self.full_model_state_slab = None
        self.policy = MeshPolicy(residual_layout, num_links)
        self.local_expert_dispatch = local_expert_dispatch
        self.batch_attention_reduce = batch_attention_reduce
        self.batch_direct_ar = batch_direct_ar
        self.batch_full_projections = batch_full_projections
        self.pack_linear = pack_linear
        self.batch_shared_projections = batch_shared_projections
        self.split_rows_once = split_rows_once
        self.decode_fp32_selector = decode_fp32_selector
        self.decode_compensated_norm = decode_compensated_norm
        self.decode_tile_silu = decode_tile_silu
        self.decode_full_input_shard = decode_full_input_shard
        self.mesh_fp32_selector = MeshFP32TopK(mesh_device) if decode_fp32_selector else None
        self.compensated_norm_epilogue = (
            CompensatedNormEpilogue(ttnn, mesh_device, self.config.rms_norm_eps)
            if decode_compensated_norm and self.kind == "linear_attention"
            else None
        )
        # Standalone decoder callers retain their legacy switches; full-model
        # construction supplies a complete, already layer-resolved policy.
        self.runtime_precision = resolve_precision(precision_config) if precision_config is not None else None
        if self.runtime_precision is not None:
            if projection_precision:
                raise ValueError("Use precision_config or legacy projection_precision, not both")
            ccl_dtype = self.runtime_precision["ccl_dtype"]
            kv_dtype = self.runtime_precision["kv_cache_dtype"]
        self.projection_precision = projection_precision or {}
        if ccl_dtype not in ("bf16", "bfp8"):
            raise ValueError("unknown collective dtype")
        self.ccl_dtype = ccl_dtype
        if kv_dtype not in ("bf16", "bfp8"):
            raise ValueError("unknown KV dtype")
        self.kv_dtype = ttnn.bfloat16 if kv_dtype == "bf16" else ttnn.bfloat8_b
        self.full_projection_split = full_projection_split
        self.sparse_geometry = {"gu": {"block_w": 64}} if sparse_geometry is None else sparse_geometry
        self.residual_layout = residual_layout
        if linear_norm not in ("ordered", "distributed"):
            raise ValueError("unknown linear normalization policy")
        self.linear_norm = linear_norm
        if full_norm not in ("ordered", "distributed"):
            raise ValueError("unknown full normalization policy")
        if decode_projection not in ("interleaved", "dram"):
            raise ValueError("unknown decode projection policy")
        self.full_norm = full_norm
        self.decode_projection = decode_projection
        if attention_collective not in ("separate", "fused_rs"):
            raise ValueError("unknown attention collective policy")
        if attention_collective == "fused_rs" and residual_layout != "sharded":
            raise ValueError("fused attention RS requires hidden-sharded residuals")
        self.attention_collective = attention_collective
        if gather_projection not in ("separate", "fused"):
            raise ValueError("unknown gather projection policy")
        if gather_projection == "fused" and (residual_layout != "sharded" or full_norm != "distributed"):
            raise ValueError("fused gather projection requires distributed norm and sharded residual")
        self.gather_projection = gather_projection
        if gather_rows not in (32, 64):
            raise ValueError("gather projection row carrier must be32 or64")
        self.gather_rows = gather_rows
        if replicated_collective is None:
            replicated_collective = "direct" if residual_layout == "replicated" else "rs_ag"
        if replicated_collective not in ("rs_ag", "direct"):
            raise ValueError("unknown replicated collective policy")
        if replicated_collective == "direct" and residual_layout != "replicated":
            raise ValueError("direct all-reduce requires replicated residuals")
        self.replicated_collective = replicated_collective
        self.direct_ar_fp32 = direct_ar_fp32
        self.fused_projection_compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False
        )
        self.hidden_width = 512 if residual_layout == "sharded" else 2048
        self.collective_workspace = workspace if workspace is not None else CollectiveWorkspace(mesh_device, num_links)
        if self.collective_workspace.mesh_device is not mesh_device or self.collective_workspace.num_links != num_links:
            raise ValueError("workspace must match mesh and CCL policy")
        self.ccl = self.collective_workspace.ccl
        self.ep_outputs = self.collective_workspace.ep_outputs
        self.grouped_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def weight_dtype(group):
            return (
                device_dtype(self.runtime_precision["weight_groups"][group])
                if self.runtime_precision
                else ttnn.bfloat8_b
            )

        if self.runtime_precision:
            kernels = {
                group: device_compute(mesh_device, spec)
                for group, spec in self.runtime_precision["compute_fidelities"].items()
            }
            self.expert_compute = kernels["expert_gu"]
            self.expert_down_compute = kernels["expert_down"]
            self.grouped_compute = kernels["expert_ep"]
            self.shared_compute = kernels["shared_packed"]
            self.projection_computes = {
                "attention_packed": kernels["attention_input"],
                "self_attn.o_proj.weight": kernels["attention_output"],
                "shared_packed": kernels["shared_packed"],
                "mlp.shared_expert.down_proj.weight": kernels["shared_down"],
            }
        else:
            self.expert_down_compute = self.expert_compute
            self.projection_computes = {}

        def upload(x, dtype=ttnn.bfloat16, dim=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh_device) if dim is None else ttnn.ShardTensorToMesh(mesh_device, dim=dim)
            )
            return ttnn.from_torch(
                x.contiguous(), dtype=dtype, layout=layout, device=mesh_device, memory_config=DRAM, mesh_mapper=mapper
            )

        self.shard_norms = {
            key: upload((state_dict[key].float() + 1).reshape(1, 1, 1, 2048), dim=3)
            for key in ("input_layernorm.weight", "post_attention_layernorm.weight")
        }
        # Discard superseded single-chip projection/expert copies before installing
        # the two explicit hybrid representations used across the layer stack.
        self.candidate_weights.clear()
        self.candidate_decode.clear()
        self.grouped_gu, self.grouped_down = [], []
        del self.egu, self.ed
        self.mesh_weights = {}

        def columns(parts):
            return torch.cat([torch.cat([p.chunk(4, dim=-1)[r] for p in parts], dim=-1) for r in range(4)], dim=-1)

        gu = state_dict["mlp.experts.gate_up_proj"]
        ed = state_dict["mlp.experts.down_proj"]
        gate, up = gu.chunk(2, dim=1)
        self.egu = upload(
            columns([gate.transpose(-1, -2), up.transpose(-1, -2)]).unsqueeze(0), weight_dtype("expert_gu"), 3
        )
        self.ed = upload(ed.transpose(-1, -2).unsqueeze(0), weight_dtype("expert_down"), 2)
        self.ep_weights = {"gate": [], "up": [], "down": []}
        for local in range(64):
            for name, values in (("gate", gate), ("up", up), ("down", ed)):
                self.ep_weights[name].append(
                    upload(torch.cat([values[r * 64 + local].T for r in range(4)], dim=0), weight_dtype("expert_ep"), 0)
                )
        self.expert_ids = upload(torch.arange(256, dtype=torch.int32), ttnn.uint32, 0, ttnn.ROW_MAJOR_LAYOUT)
        table = torch.full((4, 256), -1, dtype=torch.int32)
        for r in range(4):
            table[r, r * 64 : (r + 1) * 64] = 0
        self.ep_reduce_table = upload(table, ttnn.int32, 0, ttnn.ROW_MAJOR_LAYOUT)
        self.dispatch_table = (
            self.ep_reduce_table
            if local_expert_dispatch
            else upload(torch.zeros(1, 256, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        )
        self.local_expert_mask = (
            upload((table >= 0).to(torch.float32).reshape(4, 1, 1, 256), ttnn.float32, 0)
            if local_expert_dispatch
            else None
        )
        self.mask_zero = upload(torch.zeros(1, 1, 2048, 256), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.mask_one = upload(torch.ones(1, 1, 2048, 8), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.expert_prefix_sum = upload(torch.triu(torch.ones(256, 256)))
        shared_gate = state_dict["mlp.shared_expert.gate_proj.weight"].T
        shared_up = state_dict["mlp.shared_expert.up_proj.weight"].T
        scalar = torch.nn.functional.pad(state_dict["mlp.shared_expert_gate.weight"].T, (0, 31))
        packed = torch.cat(
            [
                torch.cat(
                    [shared_gate[:, r * 128 : (r + 1) * 128], shared_up[:, r * 128 : (r + 1) * 128], scalar], dim=-1
                )
                for r in range(4)
            ],
            dim=-1,
        )
        self.mesh_weights["shared_packed"] = upload(packed, weight_dtype("shared_packed"), 1)
        self.shared_scalar_weight = None
        if self.runtime_precision and (
            self.runtime_precision["weight_groups"]["shared_packed"] != "bfp8"
            or self.runtime_precision["compute_fidelities"]["shared_packed"]
            != dict(math_fidelity="HiFi2", fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=False)
        ):
            # Packed gate/up also contains an expert weighting scalar. Preserve
            # its baseline quantization/arithmetic when sweeping the MLP group.
            # Replication matches the identical scalar on all four TP ranks.
            self.shared_scalar_weight = upload(scalar, ttnn.bfloat8_b)
            self.shared_scalar_compute = device_compute(
                mesh_device,
                dict(math_fidelity="HiFi2", fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=False),
            )
        self.mesh_weights["mlp.shared_expert.down_proj.weight"] = upload(
            state_dict["mlp.shared_expert.down_proj.weight"].T, weight_dtype("shared_down"), 0
        )
        if self.kind == "full_attention":
            qg = state_dict["self_attn.q_proj.weight"].reshape(16, 2, 256, 2048)
            k = state_dict["self_attn.k_proj.weight"].reshape(2, 256, 2048)
            v = state_dict["self_attn.v_proj.weight"].reshape(2, 256, 2048)
            packed = torch.cat(
                [
                    torch.cat(
                        [
                            qg[r * 4 : (r + 1) * 4, 0].reshape(1024, 2048),
                            k[r // 2],
                            v[r // 2],
                            qg[r * 4 : (r + 1) * 4, 1].reshape(1024, 2048),
                        ],
                        dim=0,
                    )
                    for r in range(4)
                ],
                dim=0,
            )
            self.mesh_weights["attention_packed"] = upload(packed.T, weight_dtype("attention_input"), 1)
            self.mesh_weights["self_attn.o_proj.weight"] = upload(
                state_dict["self_attn.o_proj.weight"].T, weight_dtype("attention_output"), 0
            )
        else:
            install_linear_tp4(self, state_dict, upload)
        dtypes = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        for group, keys in (
            ("attention", ("attention_packed", "self_attn.o_proj.weight")),
            ("shared", ("shared_packed", "mlp.shared_expert.down_proj.weight")),
        ):
            spec = self.projection_precision.get(group, {})
            for key in keys:
                if key in self.mesh_weights and "dtype" in spec:
                    self.mesh_weights[key] = ttnn.typecast(self.mesh_weights[key], dtypes[spec["dtype"]])
            if "fidelity" in spec:
                kernel = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, spec["fidelity"]),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
                if group == "attention":
                    self.candidate_compute["attention_packed"] = kernel
                else:
                    self.shared_compute = kernel
        if full_projection_split and self.kind == "full_attention":
            packed = self.mesh_weights["attention_packed"]
            for name, start, end in (("q", 0, 1024), ("k", 1024, 1280), ("v", 1280, 1536), ("gate", 1536, 2560)):
                self.mesh_weights["attention_" + name] = packed[:, start:end]
        self.mesh_decode_weights = {}
        if decode_projection == "dram":
            for key, weight in self.mesh_weights.items():
                role = {
                    "self_attn.o_proj.weight": "attention_out",
                    "mlp.shared_expert.down_proj.weight": "shared_down",
                }.get(key, key)
                self.mesh_decode_weights[key] = self._make_dram_projection(weight, role)
        if self.kind == "full_attention" and gather_projection == "fused":
            self.gather_projection_buffer = upload(torch.zeros(1, 1, gather_rows, 2048))
            self.gather_projection_weight = ttnn.reshape(self.mesh_weights["attention_packed"], (1, 1, 2048, 2560))
            self.gather_projection_program = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=8,
                N_block_size=10,
                subblock_h=1,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 2),
            )
            self.gather_projection_semaphores = [
                ttnn.create_global_semaphore(mesh_device, self.ccl.ccl_cores, 0) for _ in range(2)
            ]
        if replicated_collective == "direct" or (
            decode_full_input_shard and self.kind == "full_attention" and residual_layout == "replicated"
        ):
            self.direct_ar_memory = ttnn.create_sharded_memory_config(
                (32, 64),
                ttnn.CoreGrid(x=8, y=4),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        if replicated_collective == "direct":
            scratch_memory = ttnn.create_sharded_memory_config(
                (32, 256),
                ttnn.CoreGrid(x=8, y=4),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            workspace = self.collective_workspace
            buffer_dtypes = [("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)]
            if ccl_dtype == "bfp8":
                buffer_dtypes.append(("bfp8", ttnn.bfloat8_b))
            for name, dtype in buffer_dtypes:
                if name == "fp32" and not direct_ar_fp32:
                    continue
                if name not in workspace.direct_ar_buffers:
                    workspace.direct_ar_buffers[name] = ttnn.empty(
                        (1, 1, 32, 8192),
                        device=mesh_device,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=scratch_memory,
                    )
            if workspace.direct_ar_semaphore is None:
                workspace.direct_ar_semaphore = ttnn.create_global_semaphore(mesh_device, self.ccl.ccl_cores, 0)
            self.direct_ar_scratch = workspace.direct_ar_buffers["bf16"]
            if ccl_dtype == "bfp8":
                self.direct_ar_scratch_bfp8 = workspace.direct_ar_buffers["bfp8"]
            if direct_ar_fp32:
                self.direct_ar_scratch_fp32 = workspace.direct_ar_buffers["fp32"]
            self.direct_ar_semaphore = workspace.direct_ar_semaphore
        self.precision_policy.update(
            mesh="TP4/EP4",
            residual=residual_layout,
            router="FP32",
            experts="BFP8/LoFi indexed TP B1 and grouped EP B>1",
            mesh_decode_projection=decode_projection,
            mesh_full_norm=full_norm,
            mesh_linear_norm=linear_norm,
            mesh_attention_collective=attention_collective,
            mesh_gather_projection=gather_projection,
            mesh_replicated_collective=replicated_collective,
            mesh_direct_ar_fp32=direct_ar_fp32,
            mesh_gather_rows=gather_rows,
            mesh_local_expert_dispatch=local_expert_dispatch,
            mesh_sparse_geometry=self.sparse_geometry,
            batch_attention_reduce=batch_attention_reduce,
            batch_direct_ar=batch_direct_ar,
            batch_full_projections=batch_full_projections,
            pack_linear=pack_linear,
            batch_shared_projections=batch_shared_projections,
            split_rows_once=split_rows_once,
            projection_precision=self.projection_precision,
            ccl_dtype=ccl_dtype,
            kv=kv_dtype,
            full_projection_split=full_projection_split,
            decode_fp32_selector=decode_fp32_selector,
            decode_compensated_norm=decode_compensated_norm,
            decode_tile_silu=decode_tile_silu,
            decode_full_input_shard=decode_full_input_shard,
            decode_selector_ties=(
                "lower expert ID for equal FP32 keys; NaNs above numbers" if decode_fp32_selector else "native TTNN"
            ),
        )
        # Execution geometry is separate from numerical policy. Standalone
        # decoder construction retains the established baseline by default.
        self.execution_geometry = {
            "full_model_optimized": full_model_optimized,
            "direct_ar_logical_rows": bool(full_model_optimized and replicated_collective == "direct"),
            "scan_activation_read_bytes": 32 if full_model_optimized else 2048,
            "scan_arithmetic": "unchanged ordered FP32 FPU/cumsum",
            "recurrent_contraction": (
                "exact FP32 direct L1 tiles with terminal head-row output"
                if full_model_optimized and self.kind == "linear_attention"
                else "baseline"
            ),
            "unsupported_geometry": "baseline helper",
        }
        if full_model_optimized:
            from .mesh_cached_scan import CachedCoefficientScan

            self.direct_coefficient_scan = CachedCoefficientScan(ttnn, mesh_device)
            self.execution_geometry["coefficient_cache"] = self.direct_coefficient_scan.execution_geometry.copy()
            if self.kind == "linear_attention":
                from .mesh_direct_recurrent_sum import DirectRecurrentSum
                from .mesh_product_recurrent_sum import ProductRecurrentSum
                from .mesh_state_slab_recurrence import StateSlabRecurrence

                self.full_model_recurrent_sum = DirectRecurrentSum(ttnn, mesh_device)
                self.full_model_product_sum = ProductRecurrentSum(ttnn, mesh_device)
                self.full_model_state_slab = StateSlabRecurrence(ttnn, mesh_device)
                self.execution_geometry["state_slab"] = self.full_model_state_slab.execution_geometry.copy()
                self.execution_geometry[
                    "recurrent_contraction"
                ] = "exact FP32 full state slab, terminal head-row output; guarded product/direct-sum/baseline fallback"
        if self.runtime_precision:
            weights = self.runtime_precision["weight_groups"]
            fidelities = self.runtime_precision["compute_fidelities"]
            self.precision_policy.update(
                expert_dtype=weights["expert_gu"],
                expert_down_dtype=weights["expert_down"],
                grouped_prefill_dtype=weights["expert_ep"],
                grouped_prefill_down_dtype=weights["expert_ep"],
                expert_fidelity=fidelities["expert_gu"]["math_fidelity"],
                shared_dtype=weights["shared_packed"],
                shared_fidelity=fidelities["shared_packed"]["math_fidelity"],
                experts="Explicit indexed GU/down and homogeneous grouped EP policy; see resolved_precision",
                attention="Explicit full-attention input/output policy; ordered BF16/FP32 linear exceptions",
            )
            self.precision_policy["resolved_precision"] = self.runtime_precision
            self.precision_policy["runtime_precision"] = self.precision_summary()
        return self

    def precision_summary(self):
        policy = self.runtime_precision
        if policy is None:
            return None
        weights = {
            "expert_gu": [self.egu],
            "expert_down": [self.ed],
            "expert_ep": [w for group in self.ep_weights.values() for w in group],
        }
        roles = {
            "attention_packed": "attention_input",
            "self_attn.o_proj.weight": "attention_output",
            "shared_packed": "shared_packed",
            "mlp.shared_expert.down_proj.weight": "shared_down",
        }
        for key, group in roles.items():
            if key in self.mesh_weights:
                weights[group] = [self.mesh_weights[key]]
                if key in self.mesh_decode_weights:
                    weights[group].append(self.mesh_decode_weights[key][0])
        kernels = {
            "expert_gu": self.expert_compute,
            "expert_down": self.expert_down_compute,
            "expert_ep": self.grouped_compute,
        }
        kernels.update(
            {group: self.projection_computes[key] for key, group in roles.items() if key in self.mesh_weights}
        )
        for group, tensors in weights.items():
            assert all(t.dtype == device_dtype(policy["weight_groups"][group]) for t in tensors), group
            assert_compute(kernels[group], policy["compute_fidelities"][group])
        if self.shared_scalar_weight is not None:
            assert self.shared_scalar_weight.dtype == ttnn.bfloat8_b
            assert self.shared_scalar_compute.math_fidelity == ttnn.MathFidelity.HiFi2
        assert self.w["mlp.gate.weight"].dtype == ttnn.float32
        assert self.compute.math_fidelity == ttnn.MathFidelity.HiFi4
        for key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            assert self.w[key].dtype == ttnn.float32
            assert self.shard_norms[key].dtype == ttnn.bfloat16
        assert self.kv_dtype == device_dtype(policy["kv_cache_dtype"])
        assert self.ccl_dtype == policy["ccl_dtype"]
        if self.kind == "linear_attention":
            assert self.scan_router_weight.dtype == ttnn.bfloat16
            assert self.scan_attention_out_weight.dtype == ttnn.bfloat16
            assert (
                self.scan_combined_weight.dtype == ttnn.bfloat16
                if self.pack_linear
                else self.scan_attention_weight.dtype == self.scan_output_weight.dtype == ttnn.bfloat16
            )
        return dict(
            kind=self.kind,
            weights={
                group: {"dtype": str(tensors[0].dtype), "tensor_count": len(tensors)}
                for group, tensors in weights.items()
            },
            compute={group: actual_compute(kernel) for group, kernel in kernels.items()},
            kv_dtype=str(self.kv_dtype),
            ccl_dtype=self.ccl_dtype,
            ccl_scope="BF16 reduction partials only; FP32 partials preserved; BF16 residual output",
            linear_attention="BF16 ordered coefficient scan / FP32 state" if self.kind == "linear_attention" else None,
            active_experts=8,
            shared_scalar=dict(
                dtype="bfp8",
                fidelity="HiFi2",
                path="separate_preserved" if self.shared_scalar_weight is not None else "baseline_packed",
            ),
        )

    def linear(self, x, key, dtype=ttnn.bfloat16, **kwargs):
        if hasattr(self, "mesh_weights") and key in self.mesh_weights:
            if key == "attention_packed" and self.full_projection_split:
                return ttnn.concat(
                    [self.linear(x, "attention_" + part, dtype, **kwargs) for part in ("q", "k", "v", "gate")],
                    dim=-1,
                    memory_config=DRAM,
                )
            kernel = (
                self.shared_compute
                if key.startswith("mlp.") or key == "shared_packed"
                else self.candidate_compute.get("attention_packed", self.compute)
            )
            precision_key = "attention_packed" if key.startswith("attention_") else key
            kernel = self.projection_computes.get(precision_key, kernel)
            if key == "attention_packed" and x.shape[-1] == 512:
                rows = x.shape[-2]
                local = reshape(x, (1, 1, rows, 512))
                if rows < self.gather_rows:
                    local = ttnn.pad(local, [(0, 0), (0, 0), (0, self.gather_rows - rows), (0, 0)], 0.0)
                _, projected = ttnn.experimental.strided_all_gather_minimal_matmul_async(
                    local,
                    self.gather_projection_weight,
                    persistent_output_buffer=self.gather_projection_buffer,
                    dim=3,
                    multi_device_global_semaphore=self.gather_projection_semaphores,
                    strided_all_gather_core_grid_offset=ttnn.CoreCoord(0, 2),
                    num_links=1,
                    topology=self.ccl.topology,
                    memory_config_ag=DRAM,
                    memory_config_mm=DRAM,
                    config=self.gather_projection_program,
                    compute_kernel_config=kernel,
                    cluster_axis=1,
                    num_workers_per_link=1,
                    num_buffers_per_channel=1,
                    read_local_slice_from_input=False,
                    chunks=1,
                    mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.Off,
                )
                return reshape(projected[:, :, :rows, :], (1, rows, 2560))
            if (
                x.shape[-2] == 1
                or (
                    self.batch_shared_projections
                    if key in ("shared_packed", "mlp.shared_expert.down_proj.weight")
                    else self.batch_full_projections
                )
                and kwargs.get("batch_decode", False)
                and x.shape[-2] <= 32
            ) and key in self.mesh_decode_weights:
                weight, im, om, pc = self.mesh_decode_weights[key]
                result = ttnn.linear(
                    ttnn.to_memory_config(x, im),
                    weight,
                    dtype=dtype,
                    program_config=pc,
                    compute_kernel_config=kernel,
                    memory_config=om,
                )
                return ttnn.to_memory_config(result, DRAM)
            return ttnn.linear(x, self.mesh_weights[key], dtype=dtype, compute_kernel_config=kernel, memory_config=DRAM)
        return super().linear(x, key, dtype, **kwargs)

    def _gather(self, x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=x.memory_config(),
        )

    def _reduce(self, x, *, batched=False):
        if self.ccl_dtype == "bfp8" and x.dtype == ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        if (
            self.replicated_collective == "direct"
            and (x.shape[-2] == 1 or self.batch_direct_ar and batched and x.shape[-2] <= 32)
            and (x.dtype in (ttnn.bfloat16, ttnn.bfloat8_b) or self.direct_ar_fp32)
        ):
            rows = x.shape[-2]
            local = reshape(ttnn.to_memory_config(x, DRAM), (1, 1, rows, 2048))
            logical_rows = (
                self.full_model_optimized
                and x.dtype in (ttnn.bfloat16, ttnn.float32)
                and tuple(local.padded_shape) == (1, 1, 32, 2048)
            )
            if rows < 32 and not logical_rows:
                local = ttnn.pad(local, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
            local = ttnn.to_memory_config(local, self.direct_ar_memory)
            total = ttnn.experimental.all_reduce_async(
                local,
                (
                    self.direct_ar_scratch_fp32
                    if x.dtype == ttnn.float32
                    else self.direct_ar_scratch_bfp8
                    if x.dtype == ttnn.bfloat8_b
                    else self.direct_ar_scratch
                ),
                cluster_axis=1,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=self.direct_ar_semaphore,
                memory_config=self.direct_ar_memory,
                dtype=ttnn.bfloat16,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
                fp32_dest_acc=True,
            )
            if logical_rows:
                return reshape(total, (1, rows, 2048))
            return reshape(total[:, :, :rows, :], (1, rows, 2048))
        x = reshape(ttnn.to_memory_config(x, DRAM), (1, 1, x.shape[-2], 2048))
        y = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=DRAM,
        )
        if self.residual_layout == "replicated":
            y = self._gather(y)
        return reshape(ttnn.typecast(y, ttnn.bfloat16), (1, x.shape[-2], self.hidden_width))

    def norm(self, x, key):
        if (
            getattr(self, "decode_compensated_norm", False)
            and self.kind == "linear_attention"
            and x.shape[-2] == 1
            and key in ("input_layernorm.weight", "post_attention_layernorm.weight")
        ):
            return normalized_singleton(self, x, key, self.compensated_norm_epilogue)
        return super().norm(x, key)

    def _normalized_input(self, x, key):
        if self.residual_layout == "replicated":
            if (
                self.decode_full_input_shard
                and self.kind == "full_attention"
                and x.shape[-2] == 1
                and key == "input_layernorm.weight"
            ):
                x = ttnn.to_memory_config(x, self.direct_ar_memory)
            if self.kind == "full_attention" and x.is_sharded():
                return ttnn.rms_norm(
                    x,
                    weight=self.w[key],
                    memory_config=x.memory_config(),
                    epsilon=self.config.rms_norm_eps,
                    compute_kernel_config=self.compute,
                )
            return self.norm(x, key)
        local = reshape(ttnn.to_memory_config(x, DRAM), (1, 1, x.shape[-2], 512))
        norm_policy = self.linear_norm if self.kind == "linear_attention" else self.full_norm
        if norm_policy == "ordered":
            return self.norm(reshape(self._gather(local), (1, x.shape[-2], 2048)), key)
        stats = ttnn.rms_norm_pre_all_gather(local, compute_kernel_config=self.compute, dtype=ttnn.bfloat16)
        stats = self._gather(stats)
        normalized = ttnn.rms_norm_post_all_gather(
            local,
            stats,
            epsilon=self.config.rms_norm_eps,
            weight=self.shard_norms[key],
            compute_kernel_config=self.compute,
            memory_config=DRAM,
        )
        if (
            self.kind == "full_attention"
            and self.gather_projection == "fused"
            and key == "input_layernorm.weight"
            and x.shape[-2] <= self.gather_rows
        ):
            return reshape(normalized, (1, x.shape[-2], 512))
        return reshape(self._gather(normalized), (1, x.shape[-2], 2048))

    def _shared(self, x, *, batched=False):
        both = self.linear(x, "shared_packed", batch_decode=batched)
        if self.shared_scalar_weight is not None:
            scalar = ttnn.linear(
                x,
                self.shared_scalar_weight,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.shared_scalar_compute,
                memory_config=DRAM,
            )
            both = ttnn.concat([both[:, :, :256], scalar], dim=-1, memory_config=DRAM)
        middle = ttnn.multiply(
            both[:, :, :128], both[:, :, 128:256], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        down = self.linear(middle, "mlp.shared_expert.down_proj.weight", batch_decode=batched)
        return ttnn.multiply(down, both[:, :, 256:257], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

    def _moe(
        self,
        x,
        *,
        ordered_router=False,
        batched=False,
        route_activity_mask=None,
        route_slots=None,
        fuse_moe_route_plan=False,
    ):
        # Ordered RMSNorm can return intra-chip L1 shards. Dispatch and its
        # grouped expert consumer require DRAM-interleaved activations.
        tokens = x.shape[-2]
        x = ttnn.to_memory_config(
            x, ttnn.L1_MEMORY_CONFIG if tokens == 1 and self.sparse_geometry.get("input_l1", False) else DRAM
        )
        if tokens > 1 and tokens % 32:
            # The grouped combine tilizer needs complete tiles. Padding stays
            # inside MoE, never reaches attention/cache, and is sliced away.
            padded = ttnn.pad(x, [(0, 0), (0, -tokens % 32), (0, 0)], 0.0)
            return self._moe(
                padded,
                ordered_router=ordered_router,
                batched=batched,
                route_activity_mask=route_activity_mask,
                route_slots=route_slots,
                fuse_moe_route_plan=fuse_moe_route_plan,
            )[:, :tokens, :]
        logits = reshape(
            self.linear(
                x, "mlp.gate.weight", ttnn.float32, ordered_router=ordered_router, terminal_scan=ordered_router
            ),
            (1, 1, tokens, 256),
        )
        values, ids = (
            self.mesh_fp32_selector(logits)
            if tokens == 1 and self.decode_fp32_selector
            else ttnn.topk(logits, k=8, dim=-1, sorted=True)
        )
        scores = ttnn.typecast(
            ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute), ttnn.bfloat16
        )
        if tokens == 1:
            ids = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)

            def sparse(a, w, sparse_a=False):
                n = w.shape[-1]
                geometry = self.sparse_geometry.get("gu" if n == 256 else "down", {})
                cores = geometry.get("cores", 8 if n == 256 else 64)
                block_w = geometry.get("block_w", 16 if n == 256 else 4)
                per_n = n // (32 * cores)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8),
                    in0_block_w=block_w,
                    out_subblock_h=1,
                    out_subblock_w=geometry.get("subblock_w", 1),
                    out_block_h=1,
                    out_block_w=per_n,
                    per_core_M=1,
                    per_core_N=per_n,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=True,
                )
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=self.indexed_sparsity,
                    indices=ids,
                    is_input_a_sparse=sparse_a,
                    is_input_b_sparse=True,
                    program_config=pc,
                    compute_kernel_config=self.expert_down_compute if sparse_a else self.expert_compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            gu = reshape(sparse(reshape(x, (1, 1, 1, 2048)), self.egu), (1, 8, 1, 256))
            act = ttnn.multiply(
                gu[:, :, :, :128], gu[:, :, :, 128:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
            down = reshape(sparse(act, self.ed, True), (1, 8, 1, 2048))
            weighted = ttnn.multiply(down, reshape(scores, (1, 8, 1, 1)))
            total = ttnn.experimental.fast_reduce_nc(
                weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
            )
            routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
        else:
            routed = self._grouped_moe(
                x,
                ids,
                scores,
                stochastic_rounding=not batched,
                route_activity_mask=route_activity_mask,
                route_slots=route_slots,
                fuse_moe_route_plan=fuse_moe_route_plan,
            )
        return ttnn.add(routed, self._shared(x, batched=batched))

    def _grouped_moe(
        self,
        x,
        ids,
        scores,
        *,
        stochastic_rounding=True,
        route_activity_mask=None,
        route_slots=None,
        fuse_moe_route_plan=False,
    ):
        tokens = x.shape[-2]
        if fuse_moe_route_plan and route_activity_mask is None:
            raise ValueError("MoE route-plan fusion requires persistent route activity")
        if route_activity_mask is not None:
            if not self.local_expert_dispatch or stochastic_rounding:
                raise ValueError("Inactive route suppression requires local deterministic grouped decode")
            if fuse_moe_route_plan:
                from .moe_route_mask import plan_inactive_routes

                ids, counts, offsets = plan_inactive_routes(
                    ids, route_activity_mask, self.ep_reduce_table, route_slots, self.mesh_device
                )
            else:
                from .moe_route_mask import mask_inactive_routes

                ids = mask_inactive_routes(
                    ids, route_activity_mask, self.ep_reduce_table, route_slots, self.mesh_device
                )
        # Both paths feed identical masked IDs to dispatch and post-combine.
        if not fuse_moe_route_plan:
            zero = self.mask_zero[:, :, :tokens, :]
            one = self.mask_one[:, :, :tokens, :]
            selected = ttnn.to_layout(
                ttnn.scatter(zero, dim=-1, index=ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT), src=one), ttnn.TILE_LAYOUT
            )
            counts = ttnn.sum(ttnn.typecast(selected, ttnn.float32), dim=2, keepdim=True)
            if self.local_expert_dispatch:
                # Native dispatch skips table==-1 before reading offsets or writing
                # payload. Keep global IDs, but compact only this rank's 64 experts.
                counts = ttnn.multiply(counts, self.local_expert_mask)
            aligned = ttnn.multiply(ttnn.ceil(ttnn.multiply(counts, 1 / 32)), 32)
            prefix = ttnn.matmul(
                ttnn.typecast(aligned, ttnn.bfloat16),
                self.expert_prefix_sum,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute,
            )
            offsets = ttnn.subtract(prefix, aligned)

            def ints(t, shape):
                return ttnn.reshape(ttnn.to_layout(ttnn.typecast(t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), shape)

            counts = ints(counts, (1, 256))
            offsets = ints(offsets, (1, 256))
        capacity = dispatch_capacity(tokens, 64 if self.local_expert_dispatch else 256)
        dispatch_ids = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (1, tokens, 8)
        )
        routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            x,
            dispatch_ids,
            offsets,
            self.dispatch_table,
            dispatch_group_size=1,
            experts_per_chip=256,
            num_routed_experts=256,
            num_experts_per_tok=8,
            metadata_len=3,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=0,
        )
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            ttnn.reshape(routed, (capacity, 2048)),
            self.ep_weights["gate"],
            self.ep_weights["up"],
            self.ep_weights["down"],
            ttnn.reshape(counts, (256,)),
            self.expert_ids,
            input_m_tiles=(tokens + 31) // 32,
            compute_kernel_config=self.grouped_compute,
            core_grid=ttnn.CoreCoord(8, 8),
            output=self.ep_outputs[capacity],
            expert_region_offsets=ttnn.reshape(offsets, (256,)),
            read_x_at_offset=True,
            # A launch-global stochastic stream couples independent request
            # rows through their expert counts. Fixed-slot decode needs the
            # same rounding for a row regardless of other active requests.
            # Prefill retains its selected stochastic rounding contract.
            stochastic_rounding=stochastic_rounding,
        )
        combined = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(down, (1, 1, capacity, 2048)),
            metadata,
            ttnn.reshape(counts, (1, 1, 256)),
            ttnn.reshape(offsets, (1, 1, 256)),
            dispatch_group_size=1,
            experts_per_chip=256,
            num_experts_per_tok=8,
            seq_len_per_chip=tokens,
            cluster_axis=0,
            init_zeros=True,
        )
        # Both supported native layouts run the same ordered reduction. Decode
        # may pack eight weights per token page instead of one scalar per page;
        # prefill retains its original representation and rounding contract.
        weight_shape = (1, 1, tokens, 8, 1)
        if not stochastic_rounding and getattr(self, "packed_postreduce_weights", False):
            weight_shape = (1, 1, tokens, 8)
        valid_token_rows = 0
        if getattr(self, "prune_postreduce_padding", False) and route_activity_mask is not None:
            # The coherent route mask makes every synthetic padding row off-rank.
            # Native combine initializes these rows to +0. Keep actual bucket rows,
            # including inactive holes, in the original exact reduction path.
            assert not stochastic_rounding and tokens == 32 and route_slots is not None
            assert 1 <= len(route_slots) <= 32
            valid_token_rows = len(route_slots)
        routed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined,
            ttnn.reshape(ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT), weight_shape),
            dispatch_ids,
            self.ep_reduce_table,
            expert_dim=3,
            output_memory_config=DRAM,
            **({"valid_token_rows": valid_token_rows} if valid_token_rows else {}),
        )
        return reshape(routed, (1, tokens, 2048))

    def prepare_moe(self, tokens):
        if tokens > 1:
            capacity = dispatch_capacity(tokens, 64 if self.local_expert_dispatch else 256)
            if capacity not in self.ep_outputs:
                self.ep_outputs[capacity] = ttnn.zeros(
                    (capacity, 2048),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=DRAM,
                )

    def prepare_prefill(self, *, seq_len, start_pos=0, chunk_size=4096):
        plan = super().prepare_prefill(seq_len=seq_len, start_pos=start_pos, chunk_size=chunk_size)
        for offset in range(0, seq_len, self.policy.moe_chunk):
            self.prepare_moe(min(self.policy.moe_chunk, seq_len - offset))
        return plan

    def _attention_residual(
        self, x, *, state, decode, positions=None, start_pos=0, cache_boundary=True, projection_k=512
    ):
        norm = self._normalized_input(x, "input_layernorm.weight")
        if self.kind == "linear_attention":
            attention = self._linear_attention(
                norm,
                state,
                decode=decode,
                cache_boundary=cache_boundary,
                projection_k=projection_k,
                activity=positions.get("activity") if positions is not None else None,
            )
        else:
            attention = self._full_attention(norm, state, positions, decode=decode, start_pos=start_pos)
            if self.attention_collective == "fused_rs":
                return ttnn.add(x, attention, fast_and_approximate_mode=False)
        return ttnn.add(x, self._reduce(attention), fast_and_approximate_mode=False)

    def _forward(
        self, hidden_states, *, state, decode, positions=None, start_pos=0, cache_boundary=True, projection_k=512
    ):
        x = self._attention_residual(
            hidden_states,
            state=state,
            decode=decode,
            positions=positions,
            start_pos=start_pos,
            cache_boundary=cache_boundary,
            projection_k=projection_k,
        )
        return ttnn.add(
            x,
            self._reduce(
                self._moe(
                    self._normalized_input(x, "post_attention_layernorm.weight"),
                    ordered_router=decode and cache_boundary,
                )
            ),
        )

    def prefill_forward(self, hidden_states, *, state, plan):
        if plan[-1][0] + plan[-1][1] != hidden_states.shape[1]:
            raise ValueError("plan length does not match input")
        if self.kind == "full_attention" and plan[-1][2] + plan[-1][1] > state["context_length"]:
            raise ValueError("prefill exceeds allocated context")
        if self.kind == "linear_attention":
            ttnn.copy(ttnn.typecast(ttnn.typecast(state["recurrent"], ttnn.bfloat16), ttnn.float32), state["recurrent"])
        chunks = []
        for offset, n, absolute, positions in plan:
            chunks.append(
                self._attention_residual(
                    hidden_states[:, offset : offset + n, :],
                    state=state,
                    decode=n == 1 and (self.kind == "full_attention" or hidden_states.shape[1] == 1),
                    positions=positions,
                    start_pos=absolute,
                    cache_boundary=False,
                    projection_k=1024 if hidden_states.shape[1] > 1 else 512,
                )
            )
        # Singleton boundaries may use direct L1 all-reduce while an aligned
        # interior returns DRAM. Normalize only at this prefill concatenation.
        # Bound DRAM concat too: maximum linear prefill has 4096 chunks,
        # exceeding the native reader's runtime-argument budget in one op.
        attended = (
            _bounded_concat([ttnn.to_memory_config(t, DRAM) for t in chunks], dim=1, fan_in=64)
            if len(chunks) > 1
            else chunks[0]
        )
        outputs = []
        for offset in range(0, hidden_states.shape[1], self.policy.moe_chunk):
            x = attended[:, offset : min(offset + self.policy.moe_chunk, hidden_states.shape[1]), :]
            outputs.append(
                ttnn.add(x, self._reduce(self._moe(self._normalized_input(x, "post_attention_layernorm.weight"))))
            )
        return (
            _bounded_concat([ttnn.to_memory_config(t, DRAM) for t in outputs], dim=1, fan_in=64)
            if len(outputs) > 1
            else outputs[0]
        )

    def decode_forward(self, hidden_states, *, state, positions=None):
        if tuple(hidden_states.shape) != (1, 1, self.hidden_width):
            raise ValueError("decode row does not match residual contract")
        return self._forward(hidden_states, state=state, decode=True, positions=positions)

    def prepare_decode_batch(self, batch):
        self.prepare_moe(batch)

    def decode_batch(
        self,
        hidden_states,
        *,
        states,
        positions,
        hidden_row_mask=None,
        hidden_row_slots=None,
        route_activity_mask=None,
        route_slots=None,
        fuse_moe_route_plan=False,
        solo_slot0_attention=False,
        solo_zero_partial=None,
    ):
        if not len(hidden_states) == len(states) == len(positions):
            raise ValueError("row counts differ")
        if hidden_row_mask is not None and (
            len(states) not in (2, 4, 32) or hidden_row_slots is None or len(hidden_row_slots) != len(states)
        ):
            raise ValueError("Fused hidden extraction requires a supported batch and slot mapping")
        if solo_slot0_attention:
            if (
                len(states) != 2
                or route_activity_mask is None
                or tuple(route_slots) != (0, 1)
                or not self.batch_attention_reduce
                or not self.batch_full_projections
                or self.attention_collective != "separate"
                or self.residual_layout != "replicated"
            ):
                raise ValueError("Solo attention only supports the prepared masked replicated B2 graph")
            width, dtype = (2048, ttnn.float32) if self.kind == "linear_attention" else (1024, ttnn.bfloat16)
            if (
                solo_zero_partial is None
                or tuple(solo_zero_partial.shape) != (1, 1, width)
                or tuple(solo_zero_partial.padded_shape) != (1, 32, width)
                or solo_zero_partial.dtype != dtype
                or solo_zero_partial.layout != ttnn.TILE_LAYOUT
                or solo_zero_partial.memory_config() != DRAM
            ):
                raise ValueError("Solo attention requires its immutable correctly typed zero partial")
        if len(states) == 1:
            return (self.decode_forward(hidden_states[0], state=states[0], positions=positions[0]),)
        if self.batch_attention_reduce:
            partials = []
            if self.batch_full_projections and self.kind == "full_attention":
                residual = _bounded_concat(hidden_states, dim=1, fan_in=16)
                norm = self._normalized_input(residual, "input_layernorm.weight")
                projected = self.linear(norm, "attention_packed", batch_decode=True)
                for i, (state, position) in enumerate(zip(states, positions)):
                    if solo_slot0_attention and i == 1:
                        partials.append(solo_zero_partial)
                        continue
                    partials.append(
                        ttnn.to_memory_config(
                            self._full_attention(
                                None,
                                state,
                                position,
                                decode=True,
                                start_pos=0,
                                projected=projected[:, i : i + 1, :],
                                defer_output=True,
                            ),
                            DRAM,
                        )
                    )
                # These same copies formerly ran after all rows completed,
                # retaining up to32 sharded head outputs in L1 concurrently.
                # Drain each row before the next SDPA needs its static CBs.
                merged = _bounded_concat([ttnn.to_memory_config(t, DRAM) for t in partials], dim=1, fan_in=16)
                attention = self._full_output(merged, batch_decode=True)
            else:
                for i, (row, state, position) in enumerate(zip(hidden_states, states, positions)):
                    if solo_slot0_attention and i == 1:
                        # This independent norm/attention result is unused; keep
                        # the zero partial and every downstream B2 carrier.
                        partials.append(solo_zero_partial)
                        continue
                    norm = self._normalized_input(row, "input_layernorm.weight")
                    partials.append(
                        self._linear_attention(
                            norm,
                            state,
                            decode=True,
                            activity=position.get("activity") if position is not None else None,
                        )
                        if self.kind == "linear_attention"
                        else self._full_attention(norm, state, position, decode=True, start_pos=0)
                    )
                attention = _bounded_concat([ttnn.to_memory_config(t, DRAM) for t in partials], dim=1, fan_in=16)
            if self.attention_collective != "fused_rs":
                attention = self._reduce(attention, batched=True)
            residual = _bounded_concat(hidden_states, dim=1, fan_in=16)
            x = ttnn.add(residual, attention, fast_and_approximate_mode=False)
        else:
            attended = [
                self._attention_residual(x, state=s, decode=True, positions=p)
                for x, s, p in zip(hidden_states, states, positions)
            ]
            # Bound native sharded concat's circular-buffer count.
            x = _bounded_concat(attended, dim=1, fan_in=16)
        result = ttnn.add(
            x,
            self._reduce(
                self._moe(
                    self._normalized_input(x, "post_attention_layernorm.weight"),
                    batched=True,
                    route_activity_mask=route_activity_mask,
                    route_slots=route_slots,
                    fuse_moe_route_plan=fuse_moe_route_plan,
                ),
                batched=True,
            ),
        )
        result = ttnn.to_memory_config(result, DRAM)
        if hidden_row_mask is not None:
            from .hidden_row_mask import split_masked_hidden_rows

            return split_masked_hidden_rows(result, hidden_row_mask, hidden_row_slots, self.mesh_device)
        if self.split_rows_once:
            result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
            return tuple(ttnn.to_layout(result[:, i : i + 1, :], ttnn.TILE_LAYOUT) for i in range(len(states)))
        return tuple(result[:, i : i + 1, :] for i in range(len(states)))

    def new_kv_state(self, *, context_length=262144, page_order=None, num_physical_pages=None, kv_cache=None):
        """Allocate finite paged storage. Page size is 32; order names owned pages.

        State allocation and page-table upload are setup, before any live trace.
        kv_cache can share a physical pool across rows with disjoint owned pages.
        The table is padded to eight entries by aliasing its final owned page.
        These entries are read-only causal padding, never writable logical pages.
        """
        import torch

        if not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("context_length is outside the HF contract")
        blocks = (context_length + 31) // 32
        padded_blocks = (blocks + 7) // 8 * 8
        pages = torch.arange(blocks, dtype=torch.int32) if page_order is None else page_order.to(torch.int32)
        physical = padded_blocks if num_physical_pages is None else num_physical_pages
        if physical < padded_blocks:
            raise ValueError(
                "physical pool must contain at least round_up(logical_pages, 8) pages for TTNN table validation"
            )
        if pages.shape != (blocks,) or len(pages.unique()) != blocks or pages.min() < 0 or (pages.max() >= physical):
            raise ValueError("page_order must name distinct owned pages within the physical pool")
        pages = torch.cat((pages, pages[-1:].expand(padded_blocks - blocks)))
        table = ttnn.from_torch(pages.reshape(1, -1), dtype=ttnn.int32, device=self.mesh_device)
        shape = (physical, 1, 32, 256)
        if kv_cache is None:
            kv_cache = tuple(
                (
                    ttnn.zeros(shape, device=self.mesh_device, dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT)
                    for _ in range(2)
                )
            )
        if len(kv_cache) != 2 or any(
            (tuple(t.shape) != shape or t.dtype != self.kv_dtype or t.layout != ttnn.TILE_LAYOUT for t in kv_cache)
        ):
            raise ValueError("shared KV pool must match the configured dtype and tiled geometry")
        return {"k": kv_cache[0], "v": kv_cache[1], "page_table": table, "context_length": context_length}

    def _full_attention(self, x, state, positions, *, decode, start_pos, projected=None, defer_output=False):
        if projected is None:
            projected = self.linear(x, "attention_packed")
        n = projected.shape[1]
        packed_qkv = projected[:, :, :1536]
        gate = reshape(projected[:, :, 1536:], (1, n, 4, 256))
        shard = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if decode:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(packed_qkv, (1, 1, 1, 1536)), num_heads=4, num_kv_heads=1, memory_config=shard
            )
            norm_shard = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard.shard_spec)
            q, k = [ttnn.to_memory_config(t, norm_shard) for t in (q, k)]
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                packed_qkv, num_heads=4, num_kv_heads=1, transpose_key=False
            )
            gate = ttnn.permute(gate, (0, 2, 1, 3))
        q, k = (self.norm(q, "self_attn.q_norm.weight"), self.norm(k, "self_attn.k_norm.weight"))
        if decode:
            q, k = [ttnn.to_memory_config(t, shard) for t in (q, k)]
        cos = ttnn.permute(positions["cos"], (0, 2, 1, 3))
        sin = ttnn.permute(positions["sin"], (0, 2, 1, 3))

        def rope(a):
            r = ttnn.experimental.rotary_embedding(a[:, :, :, :64], cos, sin, token_index=0 if decode else None)
            r = r[:, : a.shape[1], : a.shape[2], :]
            return ttnn.concat([r, a[:, :, :, 64:]], dim=-1)

        q, k = (rope(q), rope(k))
        if decode:
            k, v = [ttnn.to_memory_config(a, shard) for a in (k, v)]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_update_cache(
                    dest, src, update_idxs_tensor=positions["current_pos"], page_table=state["page_table"]
                )
            # Cache updates and SDPA share the ordered command queue. Release
            # dead head buffers (including the loop's V alias) before SDPA's
            # page-table workspace competes for L1 at maximum context.
            del k, v, src
            q = ttnn.to_memory_config(q, shard)
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state["k"],
                state["v"],
                cur_pos_tensor=positions["current_pos"],
                page_table_tensor=state["page_table"],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
                memory_config=DRAM,
            )
        else:
            pages = positions.get("fill_pages")
            if pages is None:
                pages = state["page_table"][:, start_pos // 32 :]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_fill_cache(dest, ttnn.typecast(src, dest.dtype), pages, batch_idx=0)
            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state["k"],
                state["v"],
                state["page_table"],
                **(
                    {"chunk_start_idx_tensor": positions["current_pos"]}
                    if "fill_pages" in positions
                    else {"chunk_start_idx": start_pos}
                ),
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
            )
        if decode:
            if positions.get("activity") is not None:
                # SDPA's -1 path returns without writing its output. Zero an
                # inactive row before any arithmetic consumes that storage.
                copy_if(o, o, positions["activity"], self.mesh_device, zero_if_inactive=True)
            o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            o = ttnn.to_memory_config(o, shard)
            merged = ttnn.experimental.nlp_concat_heads_decode(o, num_heads=4)
            merged = ttnn.reshape(merged, (1, 1, 1024), (1, 32, 1024))
        else:
            o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            merged = ttnn.transformer.concatenate_heads(o)
        return merged if defer_output else self._full_output(merged)

    def _full_output(self, merged, *, batch_decode=False):
        if self.attention_collective == "fused_rs":
            merged = ttnn.to_memory_config(merged, DRAM)
            rows = merged.shape[-2]
            if rows % 32:
                merged = ttnn.pad(merged, [(0, 0), (0, -rows % 32), (0, 0)], 0.0)
            reduced = ttnn.experimental.matmul_reduce_scatter_prefill(
                merged,
                self.mesh_weights["self_attn.o_proj.weight"],
                cluster_axis=1,
                scatter_dim=-1,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
                compute_kernel_config=self.fused_projection_compute,
                memory_config=DRAM,
            )
            return reduced[:, :rows, :]
        return self.linear(merged, "self_attn.o_proj.weight", batch_decode=batch_decode)
