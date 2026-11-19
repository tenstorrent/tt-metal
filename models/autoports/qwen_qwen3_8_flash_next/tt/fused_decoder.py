# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-only fused decoder; acceptance and experiments are recorded in stage docs.

The public API and state lifecycle are inherited unchanged. Components share the
functional setup weights; runtime dispatch enters FusedSparseMoe explicitly.
"""

import math
from copy import copy

import ttnn

from .decoder_components import GatedDeltaNet, GatedResidual, SparseMoe, linear
from .functional_decoder import FunctionalDecoder
from .gdn_recurrence import _KERNEL_ROOT, fp32_recurrent_gated_delta_rule
from .ple import PLE
from .precise_gdn_projection import PreciseGDNProjection
from .precise_projection import precise_indexer_rope
from .qsa import QSA
from .qsa_chunked import ChunkedQSA
from .router_topk import aten_router_topk


class FusedPreciseProjection(PreciseGDNProjection):
    """Fuse the FP32 product tree into the dedicated accurate SFPU reduction.

    Weight storage and BF16 boundaries are unchanged. The reduction's summation
    order differs from the explicit tree, so integrated routing checks are required.
    """

    def __init__(
        self, original, compute, decode_group=1, *, balanced_tree=False, widen_in_unpack=False, host_weight=None
    ):
        self.balanced_tree = balanced_tree
        self.widen_in_unpack = widen_in_unpack
        self.weights = original.weights
        self.input_width = self.weights[0].shape[-1]
        self.parent_block = self.weights[0].shape[-2]
        self.output_width = getattr(original, "output_width", len(self.weights) * self.parent_block)
        self.chunk_prefill = hasattr(original, "parent_block")
        self.prefill_block = 32 if self.chunk_prefill else self.parent_block
        if decode_group > 1 and host_weight is not None:
            if tuple(host_weight.shape) != (self.output_width, self.input_width):
                raise ValueError("Packed setup requires the original checkpoint projection matrix")
            device, dtype = self.weights[0].device(), self.weights[0].dtype
            # Setup owns this replaced projector. Release its original contiguous
            # blocks before uploading their grouped replacement. This avoids a
            # device concat and simultaneous original/grouped weight storage.
            # A comparison harness retaining a functional model must copy the
            # projector object before handing it to this ownership-taking path.
            original.weights = []
            self.weights = []
            block = self.parent_block * decode_group
            self.weights = [
                ttnn.from_torch(
                    host_weight[start : start + block].reshape(1, 1, -1, self.input_width).contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for start in range(0, self.output_width, block)
            ]
        elif decode_group > 1:
            self.weights = [
                (
                    ttnn.concat(self.weights[start : start + decode_group], dim=2)
                    if len(self.weights[start : start + decode_group]) > 1
                    else self.weights[start]
                )
                for start in range(0, len(self.weights), decode_group)
            ]
        self.compute = compute

    def __call__(self, value):
        count = value.shape[-2]
        if count > 32:
            return ttnn.concat([self(value[:, :, start : start + 32, :]) for start in range(0, count, 32)], dim=2)
        expanded = ttnn.reshape(ttnn.typecast(value, ttnn.float32), (1, count, 1, self.input_width))
        columns = []
        for parent in self.weights:
            parent_block = parent.shape[-2]
            block = self.prefill_block if count > 1 else parent_block
            for offset in range(0, parent_block, block):
                weight = parent if block == parent_block else parent[:, :, offset : offset + block, :]
                # Device typecast allocates even when its dtype is unchanged.
                # GR weights already reside in FP32; keep their existing storage.
                precise_weight = (
                    weight
                    if weight.dtype == ttnn.float32 or self.widen_in_unpack
                    else ttnn.typecast(weight, ttnn.float32)
                )
                products = ttnn.multiply(expanded, precise_weight)
                del precise_weight
                if self.balanced_tree:
                    # QKV state feeds tied router cutoffs. Preserve the exact
                    # original FP32 addition order while merging output blocks.
                    width, tails = self.input_width, []
                    while width > 1:
                        half = width // 2
                        if width % 2:
                            tails.append(products[..., -1:])
                        left, right = products[..., :half], products[..., half : 2 * half]
                        del products
                        products = ttnn.add(left, right)
                        del left, right
                        width = half
                    for tail in tails:
                        products = ttnn.add(products, tail)
                    reduced = products
                else:
                    reduced = ttnn.sum(
                        products,
                        dim=-1,
                        keepdim=True,
                        compute_kernel_config=self.compute,
                        fast_and_approximate_mode=False,
                    )
                    # The sum owns separate storage. Release its consumed full
                    # product before allocating the next output-column block.
                    del products
                columns.append(ttnn.reshape(reduced, (1, 1, count, block)))
        result = ttnn.concat(columns, dim=-1)
        return ttnn.typecast(result[..., : self.output_width], ttnn.bfloat16)


def fused_rms(x, weight, eps, compute):
    """FP32 unweighted normalization followed by the per-stream learned scale."""
    value = ttnn.typecast(x, ttnn.float32)
    normalized = ttnn.rms_norm(value, epsilon=eps, compute_kernel_config=compute)
    return ttnn.typecast(ttnn.multiply(normalized, weight), x.dtype)


def merged_rms(x, weight, eps, compute=None, rhs=False):
    """Preserve the functional reduction; merge its epsilon add and rsqrt."""
    value = ttnn.typecast(x, ttnn.float32)
    variance = ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True)
    if rhs:
        normalized = ttnn.multiply(
            value,
            variance,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, eps),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0),
            ],
        )
        return ttnn.typecast(ttnn.multiply(normalized, weight), x.dtype)
    scale = ttnn.add(variance, eps, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)])
    return ttnn.typecast(ttnn.multiply(ttnn.multiply(value, scale), weight), x.dtype)


class FusedGatedResidual(GatedResidual):
    def read(self, x):
        tokens = x.shape[-2]
        grouped = ttnn.permute(ttnn.reshape(x, (1, tokens, self.streams, self.width)), (0, 2, 1, 3))
        normalize = merged_rms if getattr(self, "merge_rsqrt", False) else fused_rms
        normalized = (
            merged_rms(grouped, self.weight, self.eps, self.compute, rhs=True)
            if getattr(self, "merge_rhs", False)
            else normalize(grouped, self.weight, self.eps, self.compute)
        )
        flattened = ttnn.reshape(ttnn.permute(normalized, (0, 2, 1, 3)), (1, 1, tokens, self.streams * self.width))
        # BF16 SFPU nonlinear/reduction intermediates lose bits before the
        # final rounding. Evaluate in FP32, preserving each HF BF16 boundary.
        scaled = ttnn.multiply(self.precise_down(flattened), 1 / self.streams)
        mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(scaled, ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(self.precise_up(mixing), ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.permute(ttnn.reshape(mixing, (1, tokens, self.streams, self.width)), (0, 2, 1, 3))
        mixed = ttnn.typecast(
            ttnn.mean(ttnn.typecast(ttnn.multiply(mixing, normalized), ttnn.float32), dim=1, keepdim=True),
            ttnn.bfloat16,
        )
        scaled_injection = ttnn.multiply(linear(flattened, self.inject, self.compute), 1 / self.streams)
        injection = ttnn.multiply(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(scaled_injection, ttnn.float32)), ttnn.bfloat16), 2
        )
        injection = ttnn.permute(ttnn.reshape(injection, (1, 1, tokens, self.streams)), (0, 3, 2, 1))
        return mixed, injection


_RESIDUAL_WRITE_READER = r"""

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ba = TensorAccessorArgs<0>();
    constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto block = TensorAccessor(ba, get_arg_val<uint32_t>(0), 2048);
    const auto inject = TensorAccessor(ia, get_arg_val<uint32_t>(1), 2048);
    const auto residual = TensorAccessor(ra, get_arg_val<uint32_t>(2), 2048);
    const uint32_t start = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    const uint32_t mt = get_arg_val<uint32_t>(5);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        const uint32_t row = tile / 320, col = tile % 320, stream = col / 80;
        cb_reserve_back(0, 1); cb_reserve_back(1, 1); cb_reserve_back(2, 1);
        noc_async_read(block.get_noc_addr(row * 80 + col % 80), get_write_ptr(0), 2048);
        noc_async_read(inject.get_noc_addr(stream * mt + row), get_write_ptr(1), 2048);
        noc_async_read(residual.get_noc_addr(tile), get_write_ptr(2), 2048);
        noc_async_read_barrier();
        cb_push_back(0, 1); cb_push_back(1, 1); cb_push_back(2, 1);
    }
}

"""


_RESIDUAL_WRITE_COMPUTE = r"""

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 1, 3);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(0, 1); cb_wait_front(1, 1); cb_wait_front(2, 1);
        tile_regs_acquire();
        copy_init(0); copy_tile(0, 0, 0);
        copy_init(1); copy_tile(1, 0, 1);
        copy_init(2); copy_tile(2, 0, 2);
        sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(0, 1);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        add_binary_tile_init(); add_binary_tile(0, 2, 0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        cb_reserve_back(3, 1);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, 3); tile_regs_release();
        cb_push_back(3, 1);
        cb_pop_front(0, 1); cb_pop_front(1, 1); cb_pop_front(2, 1);
    }
}

"""


_RESIDUAL_WRITE_WRITER = r"""

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        cb_wait_front(3, 1);
        noc_async_write(get_read_ptr(3), output.get_noc_addr(tile), 2048);
        noc_async_write_barrier(); cb_pop_front(3, 1);
    }
}

"""


def fused_residual_write(residual, block, injection):
    tokens = residual.shape[-2]
    if (
        tuple(residual.shape) != (1, 1, tokens, 10240)
        or tuple(block.shape) != (1, 1, tokens, 2560)
        or tuple(injection.shape) != (1, 4, tokens, 1)
        or tokens < 1
        or any(
            t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            for t in (residual, block, injection)
        )
        or any(t.device() != residual.device() for t in (block, injection))
    ):
        raise ValueError("Residual write requires matching BF16 TILE DRAM four-stream inputs")
    padded_tokens = (tokens + 31) // 32 * 32
    for tensor, expected in (
        (residual, (1, 1, padded_tokens, 10240)),
        (block, (1, 1, padded_tokens, 2560)),
        (injection, (1, 4, padded_tokens, 32)),
    ):
        tile = tensor.get_tile()
        if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
            raise ValueError("Residual write requires standard 32x32 tiles")
        if tuple(tensor.padded_shape) != expected:
            raise ValueError("Residual write received unexpected physical padding")
    mesh = residual.device()
    output = ttnn.empty_like(residual)
    mt = (tokens + 31) // 32
    total = mt * 320
    grid = mesh.compute_with_storage_grid_size()
    cores = min(total, grid.x * grid.y)
    ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
            for i in range(cores)
        ]
    )
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(cores):
        x, y = i % grid.x, i // grid.x
        start, end = total * i // cores, total * (i + 1) // cores
        reader[x][y] = [
            block.buffer_address(),
            injection.buffer_address(),
            residual.buffer_address(),
            start,
            end - start,
            mt,
        ]
        compute[x][y] = [end - start]
        writer[x][y] = [output.buffer_address(), start, end - start]
    ct = []
    for t in (block, injection, residual):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=args,
            runtime_args=rt,
            config=config,
        )
        for source, args, rt, config in [
            (_RESIDUAL_WRITE_READER, ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                _RESIDUAL_WRITE_COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=False,
                ),
            ),
            (
                _RESIDUAL_WRITE_WRITER,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op(
        [block, injection, residual, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output


class FusedResidualWriter(GatedResidual):
    def write(self, residual, block, injection):
        return fused_residual_write(residual, block, injection)


class FusedNormResidualWriter(FusedGatedResidual):
    write = FusedResidualWriter.write


_COMPACT_ROUTE_PACK_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(0), 4096);
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(1), 2048);
    const auto out_ids = TensorAccessor(oa, get_arg_val<uint32_t>(2), 20);
    const auto out_scores = TensorAccessor(pa, get_arg_val<uint32_t>(3), 20);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(ids.get_noc_addr(0), scratch, 64);
    noc_async_read(scores.get_noc_addr(0), scratch + 64, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* input_ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint16_t* input_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 64);
    volatile tt_l1_ptr uint16_t* sorted_ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 128);
    volatile tt_l1_ptr uint16_t* sorted_scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 192);
    for (uint32_t i = 0; i < 10; ++i) {
        const uint16_t id = input_ids[i];
        ASSERT(id < 512);
        uint32_t j = i;
        while (j > 0 && sorted_ids[j-1] > id) {
            sorted_ids[j] = sorted_ids[j-1]; sorted_scores[j] = sorted_scores[j-1]; --j;
        }
        sorted_ids[j] = id; sorted_scores[j] = input_scores[i];
    }
    noc_async_write(scratch + 128, out_ids.get_noc_addr(0), 20);
    noc_async_write(scratch + 192, out_scores.get_noc_addr(0), 20);
    noc_async_write_barrier();
}
"""


def fused_compact_routes(indices, scores):
    assert tuple(indices.shape) == tuple(scores.shape) == (1, 1, 1, 10)
    assert indices.dtype == ttnn.uint32 and scores.dtype == ttnn.bfloat16
    device = indices.device()
    ids = ttnn.empty((1, 1, 1, 10), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    weights = ttnn.empty((1, 1, 1, 10), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tensors = [indices, scores, ids, weights]
    core = ttnn.CoreCoord(0, 0)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [x.buffer_address() for x in tensors]
    ct = []
    for x in tensors:
        ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_COMPACT_ROUTE_PACK_SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=256,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op(tensors, program)
    return ids, weights


_COMPACT_UNION_ROUTES_KERNEL = r"""

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const uint32_t tokens = get_arg_val<uint32_t>(5), slots = tokens * 10;
    const uint32_t stride = ((slots*2+63)/64)*32; // aligned uint16 row stride
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(0), 4096);
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(1), 2048);
    const auto out_ids = TensorAccessor(oa, get_arg_val<uint32_t>(2), slots*2);
    const auto out_scores = TensorAccessor(wa, get_arg_val<uint32_t>(3), slots*2);
    const auto reduce_ids = TensorAccessor(ra, get_arg_val<uint32_t>(4), slots*2);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(ids.get_noc_addr(0), scratch, 4096);
    noc_async_read(scores.get_noc_addr(0), scratch + 4096, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ip = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 4096);
    volatile tt_l1_ptr uint16_t* op = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 6144);
    volatile tt_l1_ptr uint16_t* map = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 7168);
    volatile tt_l1_ptr uint16_t* wp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 8192);
    for (uint32_t e=0;e<512;++e) map[e]=0;
    for (uint32_t t=0;t<tokens;++t) {
        uint32_t row = t<16 ? t*16 : 512+(t-16)*16;
        for (uint32_t j=0;j<10;++j) { ASSERT(ip[row+j]<512);map[ip[row+j]]=1; }
    }
    uint32_t used=0;
    for (uint32_t e=0;e<512;++e) if(map[e]) { op[used]=e;map[e]=++used; }
    ASSERT(used<=slots);
    for(uint32_t j=used;j<slots;++j)op[j]=0;
    for(uint32_t j=0;j<tokens*stride;++j)wp[j]=0;
    for(uint32_t t=0;t<tokens;++t) {
        uint32_t row=t<16?t*16:512+(t-16)*16;
        for(uint32_t j=0;j<10;++j)wp[t*stride+map[ip[row+j]]-1]=sp[row+j];
        noc_async_write(scratch+6144,reduce_ids.get_noc_addr(t),slots*2);
        noc_async_write(scratch+8192+t*stride*2,out_scores.get_noc_addr(t),slots*2);
    }
    noc_async_write(scratch+6144,out_ids.get_noc_addr(0),slots*2);
    noc_async_write_barrier();
}

"""


def fused_compact_union_routes(indices, scores):
    tokens = indices.shape[-2]
    slots = 10 * tokens
    if (
        tuple(indices.shape) != (1, 1, tokens, 10)
        or tuple(scores.shape) != tuple(indices.shape)
        or not 1 <= tokens <= 32
        or indices.dtype != ttnn.uint32
        or scores.dtype != ttnn.bfloat16
        or any(t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG for t in [indices, scores])
        or scores.device() != indices.device()
    ):
        raise ValueError("Compact union requires matching TILE top10 IDs/scores for 1..32 logical rows")
    device = indices.device()
    ids = ttnn.empty((1, 1, 1, slots), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    weights = ttnn.empty((tokens, 1, 1, slots), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reduction = ttnn.empty((tokens, 1, 1, slots), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tensors = [indices, scores, ids, weights, reduction]
    core = ttnn.CoreCoord(0, 0)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in tensors] + [tokens]
    ct = []
    for t in tensors:
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_COMPACT_UNION_ROUTES_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=32768,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=32768)],
            )
        ],
        semaphores=[],
    )
    ttnn.generic_op(tensors, program)
    return ids, weights, reduction


_PACKED_SILU_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    const uint32_t half_tiles = get_arg_val<uint32_t>(3);
    for (uint32_t out = start; out < start + count; ++out) {
        const uint32_t row = out / half_tiles, col = out % half_tiles;
        cb_reserve_back(0, 1); cb_reserve_back(1, 1);
        noc_async_read(input.get_noc_addr(row * 2 * half_tiles + col), get_write_ptr(0), 2048);
        noc_async_read(input.get_noc_addr(row * 2 * half_tiles + half_tiles + col), get_write_ptr(1), 2048);
        noc_async_read_barrier(); cb_push_back(0, 1); cb_push_back(1, 1);
    }
}
"""
_PACKED_SILU_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 1, 2);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t t = 0; t < count; ++t) {
        cb_wait_front(0, 1); cb_reserve_back(3, 1);
        copy_init(0); tile_regs_acquire(); copy_tile(0, 0, 0);
        silu_tile_init(); silu_tile(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, 3); tile_regs_release();
        cb_push_back(3, 1); cb_pop_front(0, 1);
        cb_wait_front(3, 1); cb_wait_front(1, 1); cb_reserve_back(2, 1);
        tile_regs_acquire(); copy_init(3); copy_tile(3, 0, 0);
        copy_init(1); copy_tile(1, 0, 1);
        mul_binary_tile_init(); mul_binary_tile(0, 1, 0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, 2); tile_regs_release();
        cb_push_back(2, 1); cb_pop_front(3, 1); cb_pop_front(1, 1);
    }
}
"""
_PACKED_SILU_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t t = start; t < start + count; ++t) {
        cb_wait_front(2, 1); noc_async_write(get_read_ptr(2), output.get_noc_addr(t), 2048);
        noc_async_write_barrier(); cb_pop_front(2, 1);
    }
}
"""


def fused_packed_silu(x, output_tensor=None):
    if (
        x.dtype != ttnn.bfloat16
        or x.layout != ttnn.TILE_LAYOUT
        or x.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        or len(x.shape) < 2
        or x.shape[-1] % 64
    ):
        raise ValueError("Packed SiLU requires interleaved BF16 tiles with tile-aligned half width")
    device = x.device()
    shape = list(x.shape)
    shape[-1] //= 2
    out = output_tensor
    if out is None:
        out = ttnn.empty(
            shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    elif (
        list(out.shape) != shape
        or out.dtype != ttnn.bfloat16
        or out.layout != ttnn.TILE_LAYOUT
        or out.device() != device
        or out.memory_config() != ttnn.DRAM_MEMORY_CONFIG
    ):
        raise ValueError("Packed activation destination must match the output shape, dtype, layout and device")
    tiles = math.prod(shape[:-2]) * ((shape[-2] + 31) // 32) * (shape[-1] // 32)
    half_tiles = shape[-1] // 32
    grid = device.compute_with_storage_grid_size()
    n = min(grid.x * grid.y, tiles)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for i, c in enumerate(cores):
        count = tiles // n + int(i < tiles % n)
        reader[c.x][c.y] = [x.buffer_address(), start, count, half_tiles]
        compute[c.x][c.y] = [count]
        writer[c.x][c.y] = [out.buffer_address(), start, count]
        start += count
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for source, ct, rt, config in [
            (
                _PACKED_SILU_READER,
                ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                reader,
                ttnn.ReaderConfigDescriptor(),
            ),
            (_PACKED_SILU_COMPUTE, [], compute, config),
            (
                _PACKED_SILU_WRITER,
                ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out


_QKV_TREE_READER = r"""

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto wa=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto w=TensorAccessor(wa,get_arg_val<uint32_t>(1),2048);
    const uint32_t first=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3), nt=get_arg_val<uint32_t>(4);
    const uint32_t cache=get_write_ptr(6);
    for(uint32_t k=0;k<80;++k)noc_async_read(x.get_noc_addr(k),cache+k*2048,2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* input=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cache);
    // Both alternating query buffers start at zero; only column zero is consumed.
    volatile tt_l1_ptr uint32_t* qs=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for(uint32_t i=0;i<1024;++i)qs[i]=0;
    for(uint32_t task=first;task<first+count;++task) {
        const uint32_t q=task/nt, outcol=task%nt;
        for(uint32_t j=0;j<5;++j)for(uint32_t b=0;b<16;++b) {
            cb_reserve_back(0,1);cb_reserve_back(1,1);
            noc_async_read(w.get_noc_addr((j*16+b)*nt+outcol),get_write_ptr(0),2048);
            volatile tt_l1_ptr uint16_t* query=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(1));
            for(uint32_t n=0;n<32;++n) {
                const uint32_t k=j+5*(b+16*n);
                const uint32_t offset=(k/32)*1024+(q/16)*512+((k%32)/16)*256+(q%16)*16+k%16;
                query[(n/16)*512+(n%16)*16]=input[offset];
            }
            noc_async_read_barrier();cb_push_back(0,1);cb_push_back(1,1);
        }
    }
}

"""

_QKV_TREE_COMPUTE = r"""

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"

#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"
void balanced_column_reduce_tile0() {
    MATH((::_llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>()));
    MATH((ckernel::sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
    MATH((::_llk_math_eltwise_sfpu_start_(0)));
    MATH(([&]() {
        constexpr uint32_t upper[4] = {0, 0, 16, 16};
        constexpr uint32_t lower[4] = {32, 32, 48, 48};
        constexpr uint32_t column[4] = {0, 2, 0, 2};
        for (uint32_t i=0;i<4;++i) {
            ckernel::sfpu::load_face_data<InstrModLoadStore::FP32, false>(upper[i], lower[i], column[i]);
            // Width32 ->16: corresponding upper/lower rows first.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG4,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG5,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG2,p_sfpu::LCONST_1,p_sfpu::LREG6,p_sfpu::LREG2,0);
            TTI_SFPADD(p_sfpu::LREG3,p_sfpu::LCONST_1,p_sfpu::LREG7,p_sfpu::LREG3,0);
            // Width16 ->8 ->4 across the four row-group registers.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG2,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG3,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG1,p_sfpu::LREG0,0);
            TTI_SFPTRANSP(0,0,0,0);
            // Width4 ->2 ->1 across the four positions of the remaining row group.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG2,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG3,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG1,p_sfpu::LREG0,0);
            TT_SFPSTORE(p_sfpu::LREG0,static_cast<uint32_t>(InstrModLoadStore::FP32),ADDR_MOD_7,upper[i]+column[i]);
        }
    }()));
    MATH((::_llk_math_eltwise_sfpu_done_()));
}

void load(uint32_t cb,uint32_t index,uint32_t dst) {
    reconfig_data_format_srca(cb);copy_init(cb);copy_tile(cb,index,dst);
}
void pair_sum(uint32_t input,uint32_t left,uint32_t right,uint32_t output,uint32_t index) {
    tile_regs_acquire();load(input,left,0);load(input,right,1);
    add_binary_tile_init();add_binary_tile(0,1,0);
    tile_regs_commit();tile_regs_wait();pack_tile<true>(0,output,index);tile_regs_release();
}
void kernel_main() {
    compute_kernel_hw_startup(0,2);
    const uint32_t tasks=get_arg_val<uint32_t>(0);
    for(uint32_t task=0;task<tasks;++task) {
        pack_reconfig_data_format(2);
        cb_reserve_back(4,5);
        for(uint32_t j=0;j<5;++j) {
            cb_reserve_back(2,16);
            for(uint32_t b=0;b<16;++b) {
                cb_wait_front(0,1);cb_wait_front(1,1);
                tile_regs_acquire();load(0,0,0);load(1,0,1);
                sfpu_mul_bcast_col_init();sfpu_mul_bcast_col(0,1);
                balanced_column_reduce_tile0();
                tile_regs_commit();tile_regs_wait();pack_tile<true>(0,2,b);tile_regs_release();
                cb_pop_front(0,1);cb_pop_front(1,1);
            }
            cb_push_back(2,16);
            uint32_t input=2,output=3;
            for(uint32_t width=16;width>1;width/=2) {
                cb_wait_front(input,16);cb_reserve_back(output,16);
                for(uint32_t i=0;i<width/2;++i)pair_sum(input,i,i+width/2,output,i);
                cb_pop_front(input,16);cb_push_back(output,16);
                uint32_t swap=input;input=output;output=swap;
            }
            cb_wait_front(2,16);
            tile_regs_acquire();load(2,0,0);tile_regs_commit();tile_regs_wait();
            pack_tile<true>(0,4,j);tile_regs_release();cb_pop_front(2,16);
        }
        cb_push_back(4,5);cb_wait_front(4,5);cb_reserve_back(5,2);
        pair_sum(4,0,2,5,0);pair_sum(4,1,3,5,1);cb_push_back(5,2);cb_wait_front(5,2);
        tile_regs_acquire();load(5,0,0);load(5,1,1);
        add_binary_tile_init();add_binary_tile(0,1,0);load(4,4,1);add_binary_tile_init();add_binary_tile(0,1,0);
        typecast_tile_init<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b>();typecast_tile<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();tile_regs_wait();cb_reserve_back(7,1);
        pack_reconfig_data_format(7);pack_tile<true>(0,7);tile_regs_release();cb_push_back(7,1);
        cb_pop_front(4,5);cb_pop_front(5,2);
    }
}

"""

_QKV_TREE_WRITER = r"""

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(a,get_arg_val<uint32_t>(0),2048);
    const uint32_t first=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2), nt=get_arg_val<uint32_t>(3), tokens=get_arg_val<uint32_t>(4);
    const uint32_t zeros=get_write_ptr(8);
    volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for(uint32_t i=0;i<16;++i)z[i]=0;
    for(uint32_t task=first;task<first+count;++task) {
        const uint32_t q=task/nt,col=task%nt;
        cb_wait_front(7,1);const uint32_t src=get_read_ptr(7);
        const uint32_t offset=(q/16)*1024+(q%16)*32;
        noc_async_write(src,output.get_noc_addr(col,offset),32);
        noc_async_write(src+512,output.get_noc_addr(col,offset+512),32);
        if(q==0)for(uint32_t r=tokens;r<32;++r) {
            const uint32_t pad=(r/16)*1024+(r%16)*32;
            noc_async_write(zeros,output.get_noc_addr(col,pad),32);
            noc_async_write(zeros,output.get_noc_addr(col,pad+512),32);
        }
        noc_async_write_barrier();cb_pop_front(7,1);
    }
}

"""


class FusedExactQKV(PreciseGDNProjection):
    """Fuse BF16 products and the original FP32 half-tree, including the odd tail."""

    def __init__(self, original, weight):
        import torch

        if weight is None or tuple(weight.shape) != (10240, 2560):
            raise ValueError("Exact QKV fusion requires the original checkpoint matrix")
        device = original.weights[0].device()
        if original.weights[0].dtype != ttnn.bfloat16:
            raise ValueError("Exact QKV fusion preserves BF16 resident weights")
        # Ownership is transferred by the factory; no duplicate device matrix.
        original.weights = []
        self.balanced_tree = True
        self.widen_in_unpack = True
        self.output_width, self.input_width = weight.shape
        assert self.input_width == 2560 and self.output_width % 32 == 0
        order = torch.tensor([j + 5 * (b + 16 * n) for j in range(5) for b in range(16) for n in range(32)])
        self.weight = ttnn.from_torch(
            weight[:, order].T.contiguous().reshape(1, 1, 2560, -1),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.weights = [self.weight]

    def __call__(self, x):
        count = x.shape[-2]
        if count > 32:
            return ttnn.concat([self(x[:, :, start : start + 32, :]) for start in range(0, count, 32)], dim=2)
        if (
            tuple(x.shape) != (1, 1, count, 2560)
            or not 1 <= count <= 32
            or x.dtype != ttnn.bfloat16
            or x.layout != ttnn.TILE_LAYOUT
        ):
            raise ValueError("Exact QKV expects logical [1,1,1..32,2560] BF16 TILE input")
        out = ttnn.empty(
            (1, 1, count, self.output_width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device()
        )
        grid = x.device().compute_with_storage_grid_size()
        nt = self.output_width // 32
        tasks = count * nt
        used = min(grid.x * grid.y, tasks)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            start = tasks * i // used
            size = tasks * (i + 1) // used - start
            reader[c.x][c.y] = [x.buffer_address(), self.weight.buffer_address(), start, size, nt]
            compute[c.x][c.y] = [size]
            writer[c.x][c.y] = [out.buffer_address(), start, size, nt, count]
        config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False)
        config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=src,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            )
            for src, ct, rt, cfg in [
                (
                    _QKV_TREE_READER,
                    ttnn.TensorAccessorArgs(x).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.weight).get_compile_time_args(),
                    reader,
                    ttnn.ReaderConfigDescriptor(),
                ),
                (_QKV_TREE_COMPUTE, [], compute, config),
                (
                    _QKV_TREE_WRITER,
                    ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    writer,
                    ttnn.WriterConfigDescriptor(),
                ),
            ]
        ]
        cbs = []
        for index, tiles, dtype, page in [
            (0, 2, ttnn.bfloat16, 2048),
            (1, 2, ttnn.bfloat16, 2048),
            (2, 16, ttnn.float32, 4096),
            (3, 16, ttnn.float32, 4096),
            (4, 5, ttnn.float32, 4096),
            (5, 2, ttnn.float32, 4096),
            (6, 80, ttnn.bfloat16, 2048),
            (7, 2, ttnn.bfloat16, 2048),
            (8, 1, ttnn.bfloat16, 64),
        ]:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=tiles * page,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        ttnn.generic_op([x, self.weight, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out


class FusedSparseMoe(SparseMoe):
    def prepare_fused_reduce(self, mesh_device):
        import torch

        self.reduce_indices = {
            n: ttnn.from_torch(
                torch.arange(self.experts, dtype=torch.int32).expand(n, 1, 1, self.experts).contiguous(),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            for n in range(1, 33)
        }
        self.reduce_mapping = ttnn.from_torch(
            torch.zeros(1, self.experts, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )

        if getattr(self, "compact_experts", False):
            self.compact_sparsity = ttnn.from_torch(
                torch.ones(1, 1, 1, self.experts, dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def pack_projections(self, *, shared=False, experts=False, host_experts=None, packed_gate_cores=None):
        # Setup only; no packing or host conversion is performed by forward.
        if shared:
            self.shared_packed = ttnn.concat([self.shared_gate, self.shared_up, self.shared_route], dim=-1)
            self.shared_width = self.shared_gate.shape[-1]
            del self.shared_gate, self.shared_up, self.shared_route
        if experts:
            if host_experts is None:
                self.expert_packed = ttnn.concat([self.gate, self.up], dim=-1)
                del self.gate, self.up
            else:
                if tuple(host_experts.shape) != (self.experts, 2 * self.intermediate, self.width):
                    raise ValueError("Packed expert setup requires original gate/up checkpoint weights")
                device, dtype = self.gate.device(), self.gate.dtype
                del self.gate, self.up
                self.expert_packed = ttnn.from_torch(
                    host_experts.transpose(-1, -2).unsqueeze(0).contiguous(),
                    device=device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if packed_gate_cores is not None:
                self.program = copy(self.program)
                self.program.decode_gate_up_cores = tuple(packed_gate_cores)

    def router_logits(self, x):
        """FP32 accumulation, HF BF16 output boundary, then FP32 routing."""
        return ttnn.typecast(self.precise_router(x), ttnn.float32)

    def __call__(self, x):
        tokens = x.shape[-2]
        if tokens > 32:
            raise ValueError("SparseMoe physical chunk is at most one tile; caller must chunk")
        logits = self.router_logits(x)
        scores, indices = aten_router_topk(logits)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        compact = getattr(self, "compact_experts", False) and tokens <= getattr(self, "compact_prefill_limit", 1)
        if compact:
            if tokens == 1:
                compact_ids, compact_scores = fused_compact_routes(indices, ttnn.typecast(scores, ttnn.bfloat16))
                compact_reduce_ids = compact_ids
            else:
                compact_ids, compact_scores, compact_reduce_ids = fused_compact_union_routes(
                    indices, ttnn.typecast(scores, ttnn.bfloat16)
                )
            active = self.compact_sparsity
        else:
            if getattr(self, "fuse_moe_cleanup", False):
                # zeros_like with a dtype override may upload small tensors.
                # Explicit device storage/fill is legal inside trace capture.
                zero_scores = ttnn.empty_like(logits, dtype=ttnn.bfloat16)
                zero_scores = ttnn.fill(zero_scores, 0.0, output_tensor=zero_scores)
            else:
                zero_scores = ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16))
            dense = ttnn.scatter(
                zero_scores,
                dim=3,
                index=indices,
                src=ttnn.typecast(scores, ttnn.bfloat16),
            )
            del zero_scores
            # Sparse matmul chooses experts per token tile. Union the selected experts
            # across rows, retaining the independent per-token scores for reduction.
            active = ttnn.to_layout(ttnn.max(dense, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(
            sparsity=active,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
        )
        if compact:
            kwargs["indices"] = compact_ids
        if hasattr(self, "expert_packed"):
            activation_destination = None
            if getattr(self, "early_packed_activation", False):
                # Keep the surviving activation in the first allocation, before
                # the wider temporary. This changes placement, not live bytes.
                shape = [*list(x.shape)[:-2], *list(self.expert_packed.shape)[:-2], tokens, self.intermediate]
                if compact:
                    shape[-3] = compact_ids.shape[-1]
                activation_destination = ttnn.empty(
                    shape,
                    device=x.device(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            program = self.program.get_decode_gate_up_config(tokens, 2 * self.intermediate, k=self.width)
            packed = ttnn.sparse_matmul(x, self.expert_packed, program_config=program, **kwargs)
            if getattr(self, "fuse_packed_activation", False):
                activated = fused_packed_silu(packed, output_tensor=activation_destination)
                del activation_destination
            else:
                gate, up = packed[..., : self.intermediate], packed[..., self.intermediate :]
            del packed
        else:
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            if getattr(self, "matmul_expert_activation", False):
                program.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            gate = ttnn.sparse_matmul(x, self.gate, program_config=program, **kwargs)
            program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
            up = ttnn.sparse_matmul(x, self.up, program_config=program, **kwargs)
        if not (hasattr(self, "expert_packed") and getattr(self, "fuse_packed_activation", False)):
            if getattr(self, "matmul_expert_activation", False):
                activated = ttnn.multiply(gate, up, output_tensor=gate if self.reuse_private_buffers else None)
            elif self.fuse_expert_activation:
                activated = ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    output_tensor=gate if self.reuse_private_buffers else None,
                )
            elif self.reuse_private_buffers:
                # The expert gate has no other consumer. Preserve both BF16
                # boundaries while avoiding two private 20 MiB temporaries.
                activated = ttnn.multiply(ttnn.silu(gate, output_tensor=gate), up, output_tensor=gate)
            else:
                activated = ttnn.multiply(ttnn.silu(gate), up)
            # These private intermediates have no further consumers. Release their
            # references before the 80 MiB per-expert output allocation on PLE.
            del gate, up
        down = ttnn.sparse_matmul(
            activated,
            self.down,
            is_input_a_sparse=True,
            program_config=self.program.get_decode_down_config(tokens, self.width, k=self.intermediate),
            **kwargs,
        )
        del activated
        if hasattr(self, "reduce_indices"):
            scores_rm = (
                compact_scores
                if compact
                else (
                    ttnn.reshape(ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT), (tokens, 1, 1, self.experts))
                    if getattr(self, "fuse_moe_cleanup", False)
                    else ttnn.to_layout(ttnn.permute(dense, (2, 0, 1, 3)), ttnn.ROW_MAJOR_LAYOUT)
                )
            )
            reduction_ids = compact_reduce_ids if compact else self.reduce_indices[tokens]
            routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                down,
                reduction_ids,
                self.reduce_mapping,
                reduce_dim=1,
                split_size=self.width,
                cluster_axis=0,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
        else:
            weights = ttnn.permute(ttnn.typecast(dense, ttnn.bfloat16), (0, 3, 2, 1))
            # The private per-expert output has no other consumer. Weight it in
            # place so the resident PLE table need not coexist with an 80 MiB copy.
            ttnn.multiply(down, weights, output_tensor=down)
            if self.reuse_private_buffers and tokens < 32:
                # Generic sum clones a padded input before zeroing its padding.
                # This weighted down buffer is private, so zero it in place and
                # expose its existing physical height to the same reduction.
                # Explicit padded shapes make both reshapes metadata-only views.
                ttnn.fill_implicit_tile_padding(down, 0.0)
                down = ttnn.reshape(down, (1, self.experts, 32, self.width), (1, self.experts, 32, self.width))
                routed = ttnn.sum(down, dim=1, keepdim=True)
                routed = ttnn.reshape(routed, (1, 1, tokens, self.width), (1, 1, 32, self.width))
            else:
                routed = ttnn.sum(down, dim=1, keepdim=True)
        del down
        if hasattr(self, "shared_packed"):
            packed = linear(x, self.shared_packed, self.compute)
            gate = packed[..., : self.shared_width]
            up = packed[..., self.shared_width : 2 * self.shared_width]
            route = packed[..., 2 * self.shared_width :]
            del packed
        else:
            gate = (
                ttnn.linear(
                    x,
                    self.shared_gate,
                    activation="silu",
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if getattr(self, "matmul_shared_activation", False)
                else linear(x, self.shared_gate, self.compute)
            )
            up = linear(x, self.shared_up, self.compute)
            route = linear(x, self.shared_route, self.compute)
        shared = (
            ttnn.multiply(gate, up)
            if getattr(self, "matmul_shared_activation", False)
            else (
                ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                if self.fuse_shared_activation
                else ttnn.multiply(ttnn.silu(gate), up)
            )
        )
        shared = linear(shared, self.shared_down, self.compute)
        shared = (
            ttnn.multiply(shared, route, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            if getattr(self, "fuse_moe_cleanup", False)
            else ttnn.multiply(shared, ttnn.sigmoid(route))
        )
        return ttnn.add(routed, shared)


def fused_indexer_scores(owner, query, compressed_keys, *, merge_heads=False, accurate_sum=False):
    """Merge peer score graphs, optionally replacing their FP32 trees by SFPU sum."""
    t = ttnn
    count, key_count = query.shape[-2], compressed_keys.shape[-2]
    if tuple(query.shape) != (1, 4, count, 128) or not 1 <= count <= 32:
        raise ValueError("Expected indexer queries [1,4,1..32,128]")
    if tuple(compressed_keys.shape) != (1, 1, key_count, 128) or key_count % 32:
        raise ValueError("Expected tile-aligned compressed keys [1,1,K,128]")
    query = t.typecast(query, t.float32)
    keys = t.typecast(compressed_keys, t.float32)
    group = 4 if merge_heads else 1
    # Bound the largest prefill product to the original 128MiB. Decode can
    # merge all keys/heads into that same bound without changing arithmetic.
    block = (65536 if count == 1 else 2048) if merge_heads else 8192
    heads = []
    for head in range(0, 4, group):
        q = t.reshape(query[:, head : head + group], (group, count, 1, 128))
        blocks = []
        for start in range(0, key_count, block):
            end = min(start + block, key_count)
            products = t.multiply(q, keys[:, :, start:end])
            if accurate_sum:
                products = t.sum(
                    products,
                    dim=-1,
                    keepdim=True,
                    compute_kernel_config=owner.compute,
                    fast_and_approximate_mode=False,
                )
            else:
                width = 128
                while width > 1:
                    half = width // 2
                    products = t.add(products[..., :half], products[..., half:width])
                    width = half
            blocks.append(t.reshape(products, (1, group, count, end - start)))
        heads.append(t.concat(blocks, dim=-1) if len(blocks) > 1 else blocks[0])
    dots = t.concat(heads, dim=1) if len(heads) > 1 else heads[0]
    return t.multiply(t.sum(t.relu(dots), dim=1, keepdim=True), 128**-0.5)


class FusedQSA(QSA):
    def indexer_scores(self, query, compressed_keys):
        return fused_indexer_scores(
            self,
            query,
            compressed_keys,
            merge_heads=self.merge_score_heads,
            accurate_sum=self.fuse_score_sum,
        )


_QSA_SELECTION_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
struct SelectionItem { uint32_t bits; uint32_t index; };
inline bool worse(SelectionItem a, SelectionItem b) {
    return a.bits < b.bits || (a.bits == b.bits && a.index > b.index);
}
inline uint32_t selection_face(uint32_t row, uint32_t column) {
    return (row >> 4) * 512 + (column >> 4) * 256 + (row & 15u) * 16 + (column & 15u);
}
inline void sift_down(SelectionItem* heap, uint32_t size) {
    uint32_t parent = 0;
    while (parent * 2 + 1 < size) {
        uint32_t child = parent * 2 + 1;
        if (child + 1 < size && worse(heap[child + 1], heap[child])) ++child;
        if (!worse(heap[child], heap[parent])) break;
        SelectionItem temporary = heap[parent]; heap[parent] = heap[child]; heap[child] = temporary;
        parent = child;
    }
}
void kernel_main() {
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(0), 4096);
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(1), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(2), 4096);
    const uint32_t row = get_arg_val<uint32_t>(3), keys = get_arg_val<uint32_t>(4);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(positions.get_noc_addr(0) + selection_face(row, 0) * 4, scratch, 64);
    noc_async_read_barrier();
    const uint32_t position_bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    // Positions are exact, nonnegative FP32 integers below 262144.
    uint32_t position = 0;
    if (position_bits != 0) {
        const uint32_t exponent = (position_bits >> 23) & 255u;
        ASSERT((position_bits >> 31) == 0 && exponent >= 127 && exponent <= 144);
        position = ((position_bits & 0x7fffffu) | 0x800000u) >> (150u - exponent);
    }
    ASSERT(position < keys * 4);
    const uint32_t complete = (position + 1) / 4;
    volatile tt_l1_ptr uint32_t* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128);
    SelectionItem* heap = reinterpret_cast<SelectionItem*>(scratch + 8320);
    uint32_t* result = reinterpret_cast<uint32_t*>(scratch + 12416);
    uint32_t size = 0;
    for (uint32_t start = 0; start < complete; start += 2048) {
        const uint32_t remaining = complete - start;
        const uint32_t length = remaining < 2048 ? remaining : 2048;
        for (uint32_t tile = 0; tile < (length + 31) / 32; ++tile) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(scores.get_noc_addr(start / 32 + tile) + selection_face(row, half * 16) * 4,
                               scratch + 128 + tile * 128 + half * 64, 64);
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < length; ++i) {
            uint32_t bits = values[i];
            if ((bits & 0x7fffffffu) == 0) bits = 0;
            ASSERT(bits < 0x7f800000u); // finite nonnegative score after ReLU/head sum
            const SelectionItem item = {bits, start + i};
            if (size < 512) {
                uint32_t child = size++;
                heap[child] = item;
                while (child > 0) {
                    const uint32_t parent = (child - 1) / 2;
                    if (!worse(heap[child], heap[parent])) break;
                    SelectionItem temporary = heap[parent]; heap[parent] = heap[child]; heap[child] = temporary;
                    child = parent;
                }
            } else if (worse(heap[0], item)) {
                heap[0] = item;
                sift_down(heap, size);
            }
        }
    }
    for (uint32_t i = 0; i < 2080; ++i) result[i] = 0xffffffffu;
    while (size > 0) {
        const uint32_t group = heap[0].index;
        --size;
        for (uint32_t offset = 0; offset < 4; ++offset) result[size * 4 + offset] = group * 4 + offset;
        if (size > 0) { heap[0] = heap[size]; sift_down(heap, size); }
    }
    for (uint32_t offset = 0; offset < 32; ++offset) {
        const uint32_t token = complete * 4 + offset;
        if (token <= position) result[2048 + offset] = token;
    }
    for (uint32_t tile = 0; tile < 65; ++tile) {
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_write(scratch + 12416 + tile * 128 + half * 64,
                            output.get_noc_addr(tile) + selection_face(row, half * 16) * 4, 64);
        }
    }
    noc_async_write_barrier();
}
"""


def fused_selected_token_ids(scores, positions):
    """FP32 top512 + causal mask + four-token expansion, entirely on device.

    Scores are nonnegative and finite. Ties use ascending key index, within
    the original non-stable TopK contract. All 2080 logical slots are written.
    """
    count, width = scores.shape[-2], scores.shape[-1]
    device = scores.device()
    if tuple(scores.shape) != (1, 1, count, width) or not 1 <= count <= 32 or width % 32:
        raise ValueError("Expected scores [1,1,1..32,tile-aligned keys]")
    if tuple(positions.shape) != (1, 1, count, 1):
        raise ValueError("Expected positions [1,1,Q,1]")
    for value in (scores, positions):
        if (
            value.dtype != ttnn.float32
            or value.layout != ttnn.TILE_LAYOUT
            or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or value.device() != device
        ):
            raise ValueError("Selection inputs require FP32 tiled DRAM on the same device")
    output = ttnn.empty(
        (count, 2080), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < count:
        raise ValueError("Need one selection worker per logical row")
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    arguments = ttnn.RuntimeArgs()
    for row, core in enumerate(cores):
        arguments[core.x][core.y] = [
            scores.buffer_address(),
            positions.buffer_address(),
            output.buffer_address(),
            row,
            width,
        ]
    compile_args = []
    for value in (scores, positions, output):
        compile_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    ttnn.generic_op(
        [scores, positions, output],
        ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=_QSA_SELECTION_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=arguments,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=32768,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=32768)
                    ],
                )
            ],
            semaphores=[],
        ),
    )
    return output


class FusedChunkedQSA(ChunkedQSA):
    def _selected_ids(self, query, *, state, positions):
        if getattr(self, "fuse_selector", False):
            return fused_selected_token_ids(self.attention.indexer_scores(query, state.compressed_keys), positions)
        return super()._selected_ids(query, state=state, positions=positions)

    @classmethod
    def from_component(
        cls,
        original,
        *,
        gate=True,
        concat=True,
        pack=True,
        split=False,
        rope=True,
        decode_rope=False,
        score_sum=False,
        merge_score_heads=False,
        valid_rows=False,
        row_compute=False,
        copy_once=False,
        selector=False,
    ):
        component = cls.__new__(cls)
        component.__dict__.update(original.__dict__)
        if score_sum or merge_score_heads or valid_rows or row_compute or copy_once:
            attention_type = FusedQSA if score_sum or merge_score_heads else type(original.attention)
            attention = attention_type.__new__(attention_type)
            attention.__dict__.update(original.attention.__dict__)
            component.attention = attention
            if score_sum or merge_score_heads:
                attention.fuse_score_sum = score_sum
                attention.merge_score_heads = merge_score_heads
            if valid_rows or row_compute or copy_once:
                attention.project_index_qk = FusedOrderedIndexerProjection(
                    attention.project_index_qk, valid_row_compute=row_compute, copy_once=copy_once
                )
        component.fuse_selector = selector
        component.fuse_gate = gate
        component.fuse_concat = concat
        component.fuse_rope = rope
        component.decode_rope = decode_rope
        if pack or split:
            attention = component.attention
            component.packed_qkv = ttnn.concat([attention.q, attention.k, attention.v], dim=-1)
            component.split_heads = split
        return component

    def _rope(self, value, positions):
        """value [1,H,T,D], positions [1,1,T,1]; only first 64 dims rotate."""
        t = self.t
        count = value.shape[-2]
        indices = t.reshape(t.to_layout(t.typecast(positions, t.uint32), t.ROW_MAJOR_LAYOUT), (1, count))
        cos = t.reshape(
            t.to_layout(t.embedding(indices, self.attention.cos, layout=t.ROW_MAJOR_LAYOUT), t.TILE_LAYOUT),
            (1, 1, count, 64),
        )
        sin = t.reshape(
            t.to_layout(t.embedding(indices, self.attention.sin, layout=t.ROW_MAJOR_LAYOUT), t.TILE_LAYOUT),
            (1, 1, count, 64),
        )
        if value.shape[-1] == 128:
            return precise_indexer_rope(value, cos, sin)
        if getattr(self, "fuse_rope", False):
            rotary = value[..., :64]
            if count == 1 and getattr(self, "decode_rope", False):
                rotary = t.permute(rotary, (2, 0, 1, 3))
                result = t.permute(
                    t.experimental.rotary_embedding(
                        rotary, cos, sin, token_index=0, compute_kernel_config=self.attention.compute
                    ),
                    (1, 2, 0, 3),
                )
            else:
                result = t.experimental.rotary_embedding(rotary, cos, sin, compute_kernel_config=self.attention.compute)
        else:
            rotated = t.concat([t.neg(value[..., 32:64]), value[..., :32]], dim=-1)
            result = t.add(t.multiply(value[..., :64], cos), t.multiply(rotated, sin))
        result = result[:, : value.shape[1], :count, :]
        return t.concat([result, value[..., 64:]], dim=-1)

    def forward_chunk(self, hidden_states, *, state, return_selected_ids=False):
        """Consume 1..32 rows, preserving state.position for caller advancement."""
        t, a = self.t, self.attention
        count = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
            raise ValueError("Expected [1,1,1..32,2560] valid token rows")
        pos = t.to_layout(t.reshape(t.typecast(state.position, t.float32), (1, 1, 1, 1)), t.TILE_LAYOUT)
        positions = t.add(pos, self.query_offsets[:, :, :count, :])
        group = t.floor(t.multiply(positions, 0.25))
        group_starts = t.multiply(group, 4)
        write_groups = t.where(t.eq(t.subtract(positions, group_starts), 3), group, -1.0)

        if count == 1 and hasattr(self, "decode_packed_qkv"):
            packed = a.linear(hidden_states, self.decode_packed_qkv, a.compute)
            query, key, value = t.experimental.nlp_create_qkv_heads_decode(
                packed[..., :7168],
                num_heads=24,
                num_kv_heads=2,
                memory_config=t.MemoryConfig(t.TensorMemoryLayout.HEIGHT_SHARDED, t.BufferType.L1),
            )
            query, key, value = [
                t.permute(t.to_memory_config(head, t.DRAM_MEMORY_CONFIG), (0, 2, 1, 3)) for head in (query, key, value)
            ]
            gate = t.permute(t.reshape(packed[..., 7168:], (1, 1, 24, 256)), (0, 2, 1, 3))
            q_gate = t.concat([query, gate], dim=-1)
        elif hasattr(self, "packed_qkv"):
            packed = a.linear(hidden_states, self.packed_qkv, a.compute)
            if getattr(self, "split_heads", False):
                q_heads, key, value = t.transformer.split_query_key_value_and_split_heads(
                    t.reshape(packed, (1, count, 13312)),
                    num_heads=48,
                    num_kv_heads=2,
                    transpose_key=False,
                )
                q_gate = t.permute(t.reshape(t.permute(q_heads, (0, 2, 1, 3)), (1, count, 24, 512)), (0, 2, 1, 3))
            else:
                q_gate = t.permute(t.reshape(packed[..., :12288], (1, count, 24, 512)), (0, 2, 1, 3))
                key = t.permute(t.reshape(packed[..., 12288:12800], (1, count, 2, 256)), (0, 2, 1, 3))
                value = t.permute(t.reshape(packed[..., 12800:], (1, count, 2, 256)), (0, 2, 1, 3))
            del packed
        else:
            q_gate = t.permute(t.reshape(a.linear(hidden_states, a.q, a.compute), (1, count, 24, 512)), (0, 2, 1, 3))
            key = t.permute(t.reshape(a.linear(hidden_states, a.k, a.compute), (1, count, 2, 256)), (0, 2, 1, 3))
            value = t.permute(t.reshape(a.linear(hidden_states, a.v, a.compute), (1, count, 2, 256)), (0, 2, 1, 3))
        query, gate = q_gate[..., :256], q_gate[..., 256:]
        query = self._rope(a.rms(query, a.q_norm, a.eps), positions)
        key = self._rope(a.rms(key, a.k_norm, a.eps), positions)

        index_qk = a.project_index_qk(hidden_states)
        index_query = t.permute(t.reshape(index_qk[..., :512], (1, count, 4, 128)), (0, 2, 1, 3))
        index_query = self._rope(a.rms(index_query, a.index_q_norm, a.eps), positions)
        raw = index_qk[..., 512:]
        history = t.concat([state.raw_history, raw], dim=2)
        # Sliding four-row sums must accumulate FP32, then round to BF16 before
        # the HF k-layernorm. Both arbitrary continuation phase and short tails
        # use exactly the same device expression.
        history_fp32 = t.typecast(history, t.float32)
        if getattr(self, "pool_mean", False):
            pooled = t.typecast(
                t.mean(
                    t.concat([history_fp32[:, :, offset : offset + count, :] for offset in range(4)], dim=0),
                    dim=0,
                    keepdim=True,
                    compute_kernel_config=a.compute,
                ),
                t.bfloat16,
            )
        else:
            pooled = t.add(history_fp32[:, :, :count, :], history_fp32[:, :, 1 : count + 1, :])
            pooled = t.add(pooled, history_fp32[:, :, 2 : count + 2, :])
            pooled = t.add(pooled, history_fp32[:, :, 3 : count + 3, :])
            pooled = t.typecast(t.multiply(pooled, 0.25), t.bfloat16)
        pooled = self._rope(a.rms(pooled, a.index_k_norm, a.eps), group_starts)

        for row in range(count):
            position = t.reshape(
                t.to_layout(t.typecast(positions[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            group_position = t.reshape(
                t.to_layout(t.typecast(write_groups[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            a._update(state.k_cache, t.permute(key[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.v_cache, t.permute(value[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.compressed_keys, pooled[:, :, row : row + 1, :], group_position)
        t.copy(history[:, :, -3:, :], state.raw_history)

        selected_ids = self._selected_ids(index_query, state=state, positions=positions)
        selected_k, selected_v = self.gather(state.k_cache, state.v_cache, state.page_table, selected_ids)
        sentinel = t.reshape(self.invalid_ids[:, :, :count, :], (count, 2080))
        allowed = t.ne(selected_ids, sentinel)
        mask = t.reshape(t.typecast(t.where(allowed, 0.0, -1.0e9), t.bfloat16), (count, 1, 1, 2080))
        attention = t.transformer.scaled_dot_product_attention(
            t.permute(query, (2, 1, 0, 3)),
            selected_k,
            selected_v,
            is_causal=False,
            attn_mask=mask,
            scale=256**-0.5,
            program_config=self.sdpa_program,
            compute_kernel_config=a.compute,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        attention = t.permute(attention, (2, 1, 0, 3))
        attention = (
            t.multiply(attention, gate, input_tensor_b_activations=[t.UnaryOpType.SIGMOID])
            if self.fuse_gate
            else t.multiply(attention, t.sigmoid(gate))
        )
        if count == 1 and getattr(self, "decode_concat", False):
            layout = t.MemoryConfig(
                t.TensorMemoryLayout.HEIGHT_SHARDED,
                t.BufferType.L1,
                t.ShardSpec(
                    t.CoreRangeSet({t.CoreRange(t.CoreCoord(0, 0), t.CoreCoord(0, 0))}),
                    [32, 256],
                    t.ShardOrientation.ROW_MAJOR,
                ),
            )
            decode_heads = t.to_memory_config(t.permute(attention, (0, 2, 1, 3)), layout)
            attention = t.to_memory_config(
                t.experimental.nlp_concat_heads_decode(decode_heads, num_heads=24), t.DRAM_MEMORY_CONFIG
            )[..., :1, :]
        else:
            attention = (
                t.reshape(t.transformer.concatenate_heads(attention), (1, 1, count, 6144))
                if self.fuse_concat
                else t.reshape(t.permute(attention, (0, 2, 1, 3)), (1, 1, count, 6144))
            )
        output = a.linear(attention, a.out, a.compute)
        return (output, selected_ids) if return_selected_ids else output


_COMPACT_GDN_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
inline uint32_t face(uint32_t r, uint32_t c) {
    return ((r/16)*2+c/16)*256+(r%16)*16+c%16;
}
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<3>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da=TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(2),4096);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(3),4096);
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(4),4096);
    const auto d=TensorAccessor(da,get_arg_val<uint32_t>(5),4096);
    const auto b=TensorAccessor(ba,get_arg_val<uint32_t>(6),4096);
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(7),4096);
    const uint32_t h=get_arg_val<uint32_t>(0), T=get_arg_val<uint32_t>(1);
    cb_reserve_back(14,1);
    const uint32_t scratch=get_write_ptr(14);
    cb_reserve_back(5,16);
    for(uint32_t i=0;i<16;++i)noc_async_read(s.get_noc_addr(h*16+i),get_write_ptr(5)+i*4096,4096);
    noc_async_read_barrier();cb_push_back(5,16);
    auto vector_row = [&](const auto& a,uint32_t cb,uint32_t token,uint32_t head,uint32_t htiles) {
        cb_reserve_back(cb,4);
        const uint32_t dest=get_write_ptr(cb);
        if(token==0) {
            volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dest);
            for(uint32_t i=0;i<4096;++i)z[i]=0;
        }
        for(uint32_t col=0;col<4;++col)for(uint32_t half=0;half<2;++half) {
            const uint32_t tile=(token*htiles+head/32)*4+col;
            noc_async_read(a.get_noc_addr(tile)+face(head%32,half*16)*4,
                dest+col*4096+half*1024,64);
        }
        noc_async_read_barrier();cb_push_back(cb,4);
    };
    auto scalar = [&](const auto& a,uint32_t cb,uint32_t token) {
        cb_reserve_back(cb,1);
        const uint32_t dest=get_write_ptr(cb);
        volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dest);
        if(token<(T>1?2:1))for(uint32_t i=0;i<1024;++i)z[i]=0;
        noc_async_read(a.get_noc_addr((token/32)*2+h/32)+face(token%32,((h%32)/16)*16)*4,scratch,64);
        noc_async_read_barrier();
        z[0]=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[h%16];
        cb_push_back(cb,1);
    };
    for(uint32_t token=0;token<T;++token) {
        vector_row(q,0,token,h/3,1);vector_row(k,1,token,h/3,1);vector_row(v,2,token,h,2);
        scalar(d,3,token);scalar(b,4,token);
    }
    cb_push_back(14,1);cb_wait_front(14,1);cb_pop_front(14,1);
}
"""

_COMPACT_GDN_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
inline uint32_t face(uint32_t r,uint32_t c) {
    return ((r/16)*2+c/16)*256+(r%16)*16+c%16;
}
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<3>();
    constexpr auto sa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(2),4096);
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(3),4096);
    const uint32_t h=get_arg_val<uint32_t>(0),T=get_arg_val<uint32_t>(1);
    cb_reserve_back(15,1);
    const uint32_t zero=get_write_ptr(15);
    volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
    for(uint32_t i=0;i<16;++i)z[i]=0;
    for(uint32_t token=0;token<T;++token) {
        cb_wait_front(6,4);
        const uint32_t src=get_read_ptr(6);
        for(uint32_t col=0;col<4;++col)for(uint32_t half=0;half<2;++half) {
            const uint32_t tile=(token*2+h/32)*4+col;
            noc_async_write(src+col*4096+half*1024,o.get_noc_addr(tile)+face(h%32,half*16)*4,64);
            if(h>=32)noc_async_write(zero,o.get_noc_addr(tile)+face(h%32+16,half*16)*4,64);
        }
        noc_async_write_barrier();cb_pop_front(6,4);
    }
    cb_wait_front(7,16);
    for(uint32_t i=0;i<16;++i)noc_async_write(get_read_ptr(7)+i*4096,s.get_noc_addr(h*16+i),4096);
    noc_async_write_barrier();cb_pop_front(7,16);
    cb_push_back(15,1);cb_wait_front(15,1);cb_pop_front(15,1);
}
"""


def fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    *,
    initial_state,
    output_final_state=True,
    compute_kernel_config=None,
    precise_dot_products=True,
    compact_io=False,
):
    import ttnn as t

    count = q.shape[1]
    if tuple(q.shape) != (1, count, 16, 128) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Expected target q/k [1,T,16,128]")
    if tuple(v.shape) != (1, count, 48, 128) or tuple(g.shape) != (1, count, 48) or tuple(beta.shape) != tuple(g.shape):
        raise ValueError("Expected target v [1,T,48,128], g/beta [1,T,48]")
    if tuple(initial_state.shape) != (1, 48, 128, 128) or initial_state.dtype != t.float32:
        raise ValueError("A persistent FP32 [1,48,128,128] initial state is mandatory")
    if count < 1:
        raise ValueError("At least one actual token is required")
    device = q.device()

    def fp32(x):
        return x if x.dtype == t.float32 else t.typecast(x, t.float32)

    def reshape_rm(x, shape):
        return t.reshape(x, shape)

    def heads(x, number):
        return t.reshape(t.permute(fp32(x), (0, 2, 1, 3)), (number, count, 128))

    if compact_io:
        qh = t.multiply(fp32(q), 128**-0.5)
        kh, vh = fp32(k), fp32(v)
        decay, betah = t.exp(fp32(g)), fp32(beta)
    else:
        qh = heads(t.multiply(fp32(q), 128**-0.5), 16)
        kh = heads(k, 16)
        qh = t.repeat_interleave(qh, repeats=3, dim=0)
        kh = t.repeat_interleave(kh, repeats=3, dim=0)
        vh = heads(v, 48)
        if count > 1:
            qh, kh, vh = (reshape_rm(x, (48 * count, 1, 128)) for x in (qh, kh, vh))
        decay = t.exp(t.reshape(t.permute(fp32(g), (0, 2, 1)), (48, count)))
        betah = t.reshape(t.permute(fp32(beta), (0, 2, 1)), (48, count))
        decay = reshape_rm(decay, (48 * count, 1, 1))
        betah = reshape_rm(betah, (48 * count, 1, 1))
    state = t.reshape(initial_state, (48, 128, 128))
    output = t.empty(
        (1, count, 48, 128) if compact_io else (48 * count, 1, 128),
        dtype=t.float32,
        layout=t.TILE_LAYOUT,
        device=device,
        memory_config=t.DRAM_MEMORY_CONFIG,
    )
    final = t.empty(
        (48, 128, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
    )
    tensors = [qh, kh, vh, decay, betah, state, output, final]
    for tensor in tensors:
        if (
            tensor.layout != t.TILE_LAYOUT
            or tensor.dtype != t.float32
            or tensor.memory_config() != t.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Recurrence requires interleaved DRAM FP32 TILE operands")
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 48:
        raise ValueError("Target recurrence requires48 compute cores")
    cores = [t.CoreCoord(h // grid.y, h % grid.y) for h in range(48)]
    ranges = t.CoreRangeSet([t.CoreRange(core, core) for core in cores])
    reader_rt, writer_rt, compute_rt = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    addresses = [value.buffer_address() for value in tensors]
    for h, core in enumerate(cores):
        reader_rt[core.x][core.y] = [h, count, *addresses[:6], 0, 0, 48]
        writer_rt[core.x][core.y] = [h, count, *addresses[6:], 48]
        compute_rt[core.x][core.y] = [count]
    reader_ct = [4, 4, 0]
    for value in tensors[:6]:
        reader_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    writer_ct = [4, 4, 0]
    for value in tensors[6:]:
        writer_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    compute_config = t.ComputeConfigDescriptor(
        math_fidelity=t.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=False, math_approx_mode=False
    )
    modes = [t.UnpackToDestMode.Default] * 64
    # Direct copies in decay, residual and rank1 seeding must retain all FP32
    # mantissa bits. q/k/matmul operands remain native unless explicitly copied.
    # State CBs8/9/10 are also FPU operands: direct-DST modes cannot be
    # applied to those indices. Separate aliases share SRAM but expose an
    # exact-copy view; their read pointers are synchronized in the kernel.
    exact_aliases = {8: 17, 9: 18, 10: 19}
    for cb in (2, 3, 4, 5, 11, *exact_aliases.values()):
        modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    if precise_dot_products:
        # The SFPU variant has no FPU source consumers: all arithmetic inputs
        # load FP32 directly into DST, including q/k, state and delta.
        exact_aliases = {}
        for cb in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
            modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    specs = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 2 if count > 1 else 1),
        (4, 2 if count > 1 else 1),
        (5, 16),
        (6, 8),
        (7, 32),
        (8, 16),
        (9, 16),
        (10, 16),
        (11, 4),
        (12, 4),
        (13, 4),
    ]
    if compact_io:
        specs += [(14, 1), (15, 1)]
    cbs = [
        t.CBDescriptor(
            total_size=tiles * 4096,
            core_ranges=ranges,
            format_descriptors=[
                t.CBFormatDescriptor(buffer_index=view, data_format=t.float32, page_size=4096)
                for view in ([index, exact_aliases[index]] if index in exact_aliases else [index])
            ],
        )
        for index, tiles in specs
    ]
    descriptors = []
    for filename, ct, rt, config in (
        ("gdn_reader.cpp", reader_ct, reader_rt, t.ReaderConfigDescriptor()),
        ("gdn_writer.cpp", writer_ct, writer_rt, t.WriterConfigDescriptor()),
        ("gdn_compute_sfpu.cpp" if precise_dot_products else "gdn_compute.cpp", [4, 4, 0], compute_rt, compute_config),
    ):
        descriptors.append(
            t.KernelDescriptor(
                kernel_source=(
                    _COMPACT_GDN_READER
                    if compact_io and filename == "gdn_reader.cpp"
                    else (
                        _COMPACT_GDN_WRITER
                        if compact_io and filename == "gdn_writer.cpp"
                        else str(_KERNEL_ROOT / filename)
                    )
                ),
                source_type=(
                    t.KernelDescriptor.SourceType.SOURCE_CODE
                    if compact_io and filename in ("gdn_reader.cpp", "gdn_writer.cpp")
                    else t.KernelDescriptor.SourceType.FILE_PATH
                ),
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )
        )
    program = t.ProgramDescriptor(kernels=descriptors, cbs=cbs, semaphores=[])
    t.generic_op(tensors, program)
    result = output if compact_io else t.permute(t.reshape(output, (1, 48, count, 128)), (0, 2, 1, 3))
    return result, t.reshape(final, (1, 48, 128, 128)) if output_final_state else None


_CONV_EXACT_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    const auto x0 = TensorAccessor(a0, get_arg_val<uint32_t>(0), 2048);
    const auto x1 = TensorAccessor(a1, get_arg_val<uint32_t>(1), 2048);
    const auto x2 = TensorAccessor(a2, get_arg_val<uint32_t>(2), 2048);
    const auto x3 = TensorAccessor(a3, get_arg_val<uint32_t>(3), 2048);
    const auto w0 = TensorAccessor(a4, get_arg_val<uint32_t>(4), 4096);
    const auto w1 = TensorAccessor(a5, get_arg_val<uint32_t>(5), 4096);
    const auto w2 = TensorAccessor(a6, get_arg_val<uint32_t>(6), 4096);
    const auto w3 = TensorAccessor(a7, get_arg_val<uint32_t>(7), 4096);
    const uint32_t start = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    for (uint32_t col = start; col < start + count; ++col) {
        for (uint32_t tap = 0; tap < 4; ++tap) {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            const uint32_t xp = get_write_ptr(0), wp = get_write_ptr(1);
            switch (tap) {
                case 0: noc_async_read(x0.get_noc_addr(col), xp, 2048); noc_async_read(w0.get_noc_addr(col), wp, 4096); break;
                case 1: noc_async_read(x1.get_noc_addr(col), xp, 2048); noc_async_read(w1.get_noc_addr(col), wp, 4096); break;
                case 2: noc_async_read(x2.get_noc_addr(col), xp, 2048); noc_async_read(w2.get_noc_addr(col), wp, 4096); break;
                case 3: noc_async_read(x3.get_noc_addr(col), xp, 2048); noc_async_read(w3.get_noc_addr(col), wp, 4096); break;
            }
            noc_async_read_barrier();
            cb_push_back(0, 1);
            cb_push_back(1, 1);
        }
    }
}
"""

_CONV_EXACT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 1, 2);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t col = 0; col < count; ++col) {
        tile_regs_acquire();
        for (uint32_t tap = 0; tap < 4; ++tap) {
            // Keep the first pair in tile0; the second pair uses tiles1/2.
            const uint32_t dst = tap == 0 ? 0 : (tap == 3 ? 2 : 1);
            cb_wait_front(0, 1);
            cb_wait_front(1, 1);
            reconfig_data_format_srca(0);
            copy_init(0);
            copy_tile(0, 0, dst);
            reconfig_data_format_srca(1);
            copy_init(1);
            copy_tile(1, 0, 3);
            sfpu_mul_bcast_row_init();
            sfpu_mul_bcast_row(dst, 3);
            if (tap == 1) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);
        }
        add_binary_tile_init();
        add_binary_tile(1, 2, 1);
        add_binary_tile(0, 1, 0);
        // Exact functional BF16 boundary, represented as masked FP32 bits.
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        // Typecast changes SFPU constants: restore the accurate FP32 SiLU init.
        silu_tile_init();
        silu_tile<true>(0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        cb_reserve_back(2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 2);
        tile_regs_release();
        cb_push_back(2, 1);
    }
}
"""

_CONV_EXACT_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t col = start; col < start + count; ++col) {
        cb_wait_front(2, 1);
        noc_async_write(get_read_ptr(2), output.get_noc_addr(col), 2048);
        noc_async_write_barrier();
        cb_pop_front(2, 1);
    }
}
"""


def fused_exact_conv_silu(history, taps, count):
    """Four exact FP32 products/pairwise adds, BF16 RNE, FP32 SiLU, BF16 RNE.

    Native SFPU primitives are fused within one generic program. Four shifted
    BF16 slices remain device operations; the compute kernel never uses the FPU.
    """
    if not 1 <= count <= 32 or tuple(history.shape) != (1, 1, count + 3, 10240):
        raise ValueError("Expected BF16 GDN history [1,1,logical_count+3,10240], count1..32")
    device = history.device()
    if history.dtype != ttnn.bfloat16 or history.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Exact fused convolution requires BF16 tiled history")
    windows = [history[:, :, i : i + count, :] for i in range(4)]
    for tap in taps:
        if tuple(tap.shape) != (1, 1, 1, 10240) or tap.dtype != ttnn.float32 or tap.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Exact fused convolution requires four FP32 tiled taps")
    if len(taps) != 4 or any(
        value.memory_config() != ttnn.DRAM_MEMORY_CONFIG or value.device() != device for value in [*windows, *taps]
    ):
        raise ValueError("Exact fused convolution requires same-device interleaved DRAM")
    output = ttnn.empty(
        (1, 1, count, 10240),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = device.compute_with_storage_grid_size()
    cores = min(grid.x * grid.y, 320)
    locations = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in locations])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addresses = [value.buffer_address() for value in [*windows, *taps]]
    start = 0
    for i, core in enumerate(locations):
        number = 320 // cores + int(i < 320 % cores)
        reader[core.x][core.y] = [*addresses, start, number]
        compute[core.x][core.y] = [number]
        writer[core.x][core.y] = [output.buffer_address(), start, number]
        start += number
    read_ct = []
    for value in [*windows, *taps]:
        read_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for source, ct, rt, config in (
            (_CONV_EXACT_READER, read_ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                _CONV_EXACT_COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    math_approx_mode=False,
                ),
            ),
            (
                _CONV_EXACT_WRITER,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * size,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )
        for index, dtype, size in ((0, ttnn.bfloat16, 2048), (1, ttnn.float32, 4096), (2, ttnn.bfloat16, 2048))
    ]
    ttnn.generic_op([*windows, *taps, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return output


class FusedGatedDeltaNet(GatedDeltaNet):
    def prepare_conv(self, mesh_device):
        import torch

        self.conv_start = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        self.conv_taps = [ttnn.typecast(tap, ttnn.bfloat16) for tap in self.conv]
        self.conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128)
        self.conv_compute = self.compute

    def __call__(self, x, *, conv_state, recurrent_state):
        tokens = x.shape[-2]
        if tokens > 32 and (hasattr(self, "conv_taps") or getattr(self, "fuse_conv_exact", False)):
            return ttnn.concat(
                [
                    self(
                        x[:, :, start : min(start + 32, tokens), :],
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                    )
                    for start in range(0, tokens, 32)
                ],
                dim=2,
            )
        packed = linear(x, self.zab, self.compute)
        z = packed[..., : self.value_width]
        a = packed[..., self.value_width : self.value_width + self.hvalue]
        b = packed[..., self.value_width + self.hvalue :]
        del packed
        projected = linear(x, self.qkv, self.compute)
        history = ttnn.concat((conv_state, projected), dim=2)
        if hasattr(self, "conv_taps"):
            # Consume only logical output rows; physical padding is private to the op.
            physical = ttnn.reshape(projected, (1, 32, self.conv_width), (1, 32, self.conv_width))
            carry = ttnn.reshape(ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT), (1, 3, self.conv_width))
            q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                physical,
                carry,
                *self.conv_taps,
                self.key_width,
                self.key_width,
                self.value_width,
                actual_start=self.conv_start,
                predecessor_carry=carry,
                program_config=self.conv_program,
                compute_kernel_config=self.conv_compute,
            )
            q = ttnn.reshape(q[:, :tokens, :], (1, tokens, self.hkey, self.dkey))
            k = ttnn.reshape(k[:, :tokens, :], (1, tokens, self.hkey, self.dkey))
            v = ttnn.reshape(v[:, :tokens, :], (1, tokens, self.hvalue, self.dvalue))
            ttnn.copy(history[:, :, -3:, :], conv_state)
        else:
            # HF Conv1d accumulates all four BF16 products before its BF16 output
            # rounding. Intermediate BF16 products/sums perturb the downstream gate.
            if getattr(self, "fuse_conv_exact", False):
                convolved = fused_exact_conv_silu(history, self.conv, tokens)
            else:
                conv_input = ttnn.typecast(history, ttnn.float32)
                parts = [ttnn.multiply(conv_input[:, :, lane : lane + tokens, :], self.conv[lane]) for lane in range(4)]
                convolved = ttnn.typecast(
                    ttnn.add(ttnn.add(parts[0], parts[1]), ttnn.add(parts[2], parts[3])), ttnn.bfloat16
                )
                convolved = ttnn.typecast(ttnn.silu(ttnn.typecast(convolved, ttnn.float32)), ttnn.bfloat16)
            ttnn.copy(history[:, :, -3:, :], conv_state)
            q = ttnn.reshape(convolved[:, :, :, : self.key_width], (1, tokens, self.hkey, self.dkey))
            k = ttnn.reshape(convolved[:, :, :, self.key_width : 2 * self.key_width], (1, tokens, self.hkey, self.dkey))
            v = ttnn.reshape(convolved[:, :, :, 2 * self.key_width :], (1, tokens, self.hvalue, self.dvalue))

        def normalize(value):
            value = ttnn.typecast(value, ttnn.float32)
            return ttnn.multiply(
                value, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(value, value), dim=-1, keepdim=True), 1e-6))
            )

        q, k = normalize(q), normalize(k)
        decay = ttnn.multiply(self.A, ttnn.softplus(ttnn.add(ttnn.typecast(a, ttnn.float32), self.dt)))
        beta = ttnn.typecast(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(b, ttnn.float32)), ttnn.bfloat16),
            ttnn.float32,
        )
        recurrent = (
            fused_recurrent_gated_delta_rule
            if getattr(self, "fuse_recurrence_layout", False) or getattr(self, "compact_recurrence_io", False)
            else fp32_recurrent_gated_delta_rule
        )
        output, state = recurrent(
            q,
            k,
            ttnn.typecast(v, ttnn.float32),
            ttnn.reshape(decay, (1, tokens, self.hvalue)),
            ttnn.reshape(beta, (1, tokens, self.hvalue)),
            initial_state=recurrent_state,
            output_final_state=True,
            compute_kernel_config=self.compute,
            **(
                {"compact_io": getattr(self, "compact_recurrence_io", False)}
                if recurrent is fused_recurrent_gated_delta_rule
                else {}
            ),
        )
        ttnn.copy(state, recurrent_state)
        output = ttnn.typecast(output, ttnn.bfloat16)
        value = ttnn.typecast(output, ttnn.float32)
        scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True), self.eps))
        output = ttnn.multiply(ttnn.typecast(ttnn.multiply(value, scale), ttnn.bfloat16), self.norm)
        gate = ttnn.reshape(z, (1, tokens, self.hvalue, self.dvalue))
        output = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(output, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32))),
            ttnn.bfloat16,
        )
        return linear(ttnn.reshape(output, (1, 1, tokens, self.value_width)), self.out, self.compute)


class FusedPLE(PLE):
    def normalize(self, x, weight, eps):
        from .decoder_components import rms

        return merged_rms(x, weight, eps, rhs=True) if self.merge_norm else rms(x, weight, eps)

    def __call__(self, hidden, embeddings, *, conv_state, conv_mask=None):
        tokens = hidden.shape[-2]
        key = self.normalize(self._group(linear(embeddings, self.key, self.compute)), self.norm_key, self.eps)
        query = self.normalize(self._group(hidden), self.norm_query, self.eps)
        value = linear(embeddings, self.value, self.compute)
        # Match HF: its multiplication/reduction is BF16, nonlinear gate is
        # signed sqrt(|gate|). clamp applies to magnitude, sign(0) remains 0.
        gate = ttnn.multiply(ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True), 1 / math.sqrt(self.width))
        gate = ttnn.multiply(ttnn.sqrt(ttnn.clamp(ttnn.abs(gate), min=1e-6)), ttnn.sign(gate))
        gated = ttnn.multiply(gate, value, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        normalized = self._flatten(self.normalize(gated, self.norm_conv, self.eps))
        gated = self._flatten(gated)
        if conv_mask is not None:
            gated = ttnn.multiply(gated, conv_mask)
            normalized = ttnn.multiply(normalized, conv_mask)
        history = ttnn.concat((conv_state, normalized), dim=2)
        convolution = None
        for tap, weight in enumerate(self.conv):
            start = tap * self.dilation
            contribution = ttnn.multiply(history[:, :, start : start + tokens, :], weight)
            convolution = contribution if convolution is None else ttnn.add(convolution, contribution)
        ttnn.copy(history[:, :, -self.state_len :, :], conv_state)
        return ttnn.add(gated, convolution, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])


_ORDERED_VALID_ROWS_READER = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Native BF16 tile face layout: TL, TR, BL, BR; sixteen rows per face.
constexpr uint32_t tile_element(uint32_t row, uint32_t column) {
    return (row >> 4) * 512 + (column >> 4) * 256 + (row & 15u) * 16 + (column & 15u);
}

void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(xa, get_arg_val<uint32_t>(0), 2048);
    const auto weight = TensorAccessor(wa, get_arg_val<uint32_t>(1), 2048);
    const uint32_t output_column_tile = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_left = 2, cb_right = 3;
    // These immutable scratch reservations belong only to the reader. They
    // are never handed to compute and are not recycled during this program.
    cb_reserve_back(cb_x, 80);
    cb_reserve_back(cb_w, 80);
    const uint32_t x_address = get_write_ptr(cb_x);
    const uint32_t w_address = get_write_ptr(cb_w);
    for (uint32_t tile = 0; tile < 80; ++tile) {
        noc_async_read(input.get_noc_addr(tile), x_address + tile * 2048, 2048);
        noc_async_read(weight.get_noc_addr(tile * 20 + output_column_tile), w_address + tile * 2048, 2048);
    }
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint16_t* x = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(x_address);
    const volatile tt_l1_ptr uint16_t* w = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(w_address);
    // Initialize both ping-pong tiles once. Only valid output rows need
    // per-K broadcast materialization; padded products remain exact zero.
    cb_reserve_back(cb_left, 2);
    cb_reserve_back(cb_right, 2);
    volatile tt_l1_ptr uint32_t* zero_left = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_left));
    volatile tt_l1_ptr uint32_t* zero_right = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_right));
    for (uint32_t i = 0; i < 1024; ++i) {
        zero_left[i] = 0;
        zero_right[i] = 0;
    }
    for (uint32_t pair = 0; pair < 1280; ++pair) {
        // Intel VDPBF16PS consumes the high member first. Do not pre-sum pairs.
        for (uint32_t member = 0; member < 2; ++member) {
            const uint32_t k = pair * 2 + (1 - member);
            const uint32_t k_tile = k >> 5;
            const uint32_t k_local = k & 31u;
            cb_reserve_back(cb_left, 1);
            cb_reserve_back(cb_right, 1);
            volatile tt_l1_ptr uint16_t* left = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_left));
            volatile tt_l1_ptr uint16_t* right =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_right));
            for (uint32_t row = 0; row < count; ++row) {
                const uint16_t value = x[k_tile * 1024 + tile_element(row, k_local)];
                for (uint32_t column = 0; column < 32; ++column) {
                    const uint32_t target = tile_element(row, column);
                    left[target] = value;
                    right[target] = w[k_tile * 1024 + tile_element(k_local, column)];
                }
            }
            cb_push_back(cb_left, 1);
            cb_push_back(cb_right, 1);
        }
    }
}
"""


_ORDERED_VALID_ROWS_COMPUTE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reconfig_data_format.h"


// Same SFPU operations over only the physically necessary top faces.
// One SFPU iteration covers two rows; use the established <=4-row policy.
template <int Iterations>
ALWI void multiply_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, Iterations, DST_ACCUM_MODE),
        a, b, out, VectorMode::R)));
}
template <int Iterations>
ALWI void add_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, Iterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a, b, out, VectorMode::R)));
}
ALWI void multiply_valid(uint32_t a, uint32_t b, uint32_t out, uint32_t rows) {
    if (rows <= 4) multiply_rows<2>(a, b, out);
    else if (rows <= 8) multiply_rows<4>(a, b, out);
    else if (rows <= 16) multiply_rows<8>(a, b, out);
    else mul_binary_tile(a, b, out);
}
ALWI void add_valid(uint32_t a, uint32_t b, uint32_t out, uint32_t rows) {
    if (rows <= 4) add_rows<2>(a, b, out);
    else if (rows <= 8) add_rows<4>(a, b, out);
    else if (rows <= 16) add_rows<8>(a, b, out);
    else add_binary_tile(a, b, out);
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr bool copy_once = get_compile_time_arg_val(0);
    constexpr bool valid_compute = get_compile_time_arg_val(1);
    constexpr uint32_t cb_left = 2, cb_right = 3, cb_out = 4;
    compute_kernel_hw_startup(cb_left, cb_right, cb_out);
    if constexpr (copy_once) copy_init(cb_left);
    tile_regs_acquire();
    // Tile2 is partial0, then the total; tile3 holds independently zeroed
    // later partials. No partial is packed/reloaded or routed through TF32 SrcA.
    const uint32_t block_width = get_arg_val<uint32_t>(0);
    const uint32_t blocks = (2560 + block_width - 1) / block_width;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t accumulator = block == 0 ? 2 : 3;
        fill_tile_init();
        fill_tile(accumulator, 0.0f);
        const uint32_t remaining = 2560 - block * block_width;
        const uint32_t width = remaining < block_width ? remaining : block_width;
        for (uint32_t k = 0; k < width; ++k) {
            cb_wait_front(cb_left, 1);
            cb_wait_front(cb_right, 1);
            if constexpr (!copy_once) {
                reconfig_data_format_srca(cb_left);
                copy_init(cb_left);
            }
            copy_tile(cb_left, 0, 0);
            if constexpr (!copy_once) {
                reconfig_data_format_srca(cb_right);
                copy_init(cb_right);
            }
            copy_tile(cb_right, 0, 1);
            // BF16 products are exact FP32 for the normal finite target
            // operands. Keep each product and accumulator addition separate:
            // the reader sends high,low; there is no pair sum or tree reduce.
            mul_binary_tile_init();
            if constexpr (valid_compute) multiply_valid(0, 1, 0, rows);
            else mul_binary_tile(0, 1, 0);
            add_binary_tile_init();
            if constexpr (valid_compute) add_valid(accumulator, 0, accumulator, rows);
            else add_binary_tile(accumulator, 0, accumulator);
            cb_pop_front(cb_left, 1);
            cb_pop_front(cb_right, 1);
        }
        if (block != 0) {
            // Pinned brgemm: each new partial + accumulated earlier partials.
            add_binary_tile_init();
            add_binary_tile(3, 2, 2);
        }
    }
    cb_reserve_back(cb_out, 1);
    tile_regs_commit();
    tile_regs_wait();
    // One final BF16 boundary. fp32_dest_acc_en retains the partial mantissas;
    // normal non-stochastic pack RNE must be checked by the exact-output probe.
    pack_tile(2, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
"""


def fused_ordered_indexer_projection(
    hidden_states,
    transposed_weight,
    *,
    output=None,
    block_size=None,
    fp32_output=False,
    direct_bf16_pack=False,
    valid_row_compute=False,
    copy_once=False,
):
    import ttnn as t

    if not isinstance(hidden_states, t.Tensor) or not isinstance(transposed_weight, t.Tensor):
        raise ValueError("Expected device tensors")
    count = hidden_states.shape[-2]
    if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
        raise ValueError("Expected BF16 input [1,1,1..32,2560]")
    if tuple(transposed_weight.shape) != (1, 1, 2560, 640):
        raise ValueError("Expected pretransposed BF16 weight [1,1,2560,640]")
    block_size = (512 if count == 1 else 1024) if block_size is None else block_size
    if type(block_size) is not int or block_size not in (512, 1024):
        raise ValueError("Candidate block_size must be512 or1024")
    if type(fp32_output) is not bool or type(direct_bf16_pack) is not bool:
        raise ValueError("Diagnostic mode flags must be bool")
    if fp32_output and direct_bf16_pack:
        raise ValueError("FP32 output and direct BF16 packing are mutually exclusive")
    output_dtype = t.float32 if fp32_output else t.bfloat16
    kernel_dtype = t.bfloat16 if direct_bf16_pack else t.float32
    output_tile_bytes = 2048 if direct_bf16_pack else 4096
    device = hidden_states.device()

    def validate(value, padded, dtype=t.bfloat16):
        if (
            not t.is_tensor_storage_on_device(value)
            or not value.is_allocated()
            or value.dtype != dtype
            or value.layout != t.TILE_LAYOUT
            or value.memory_config() != t.DRAM_MEMORY_CONFIG
            or tuple(value.padded_shape) != padded
            or value.device() != device
        ):
            raise ValueError("Ordered projection requires exact interleaved DRAM BF16 tile geometry on one device")

    validate(hidden_states, (1, 1, 32, 2560))
    validate(transposed_weight, (1, 1, 2560, 640))
    if output is None:
        output = t.empty(
            (1, 1, count, 640),
            dtype=output_dtype,
            layout=t.TILE_LAYOUT,
            device=device,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
    if not isinstance(output, t.Tensor) or tuple(output.shape) != (1, 1, count, 640):
        raise ValueError("Output shape must be [1,1,Q,640]")
    validate(output, (1, 1, 32, 640), output_dtype)
    if output.buffer_address() in (hidden_states.buffer_address(), transposed_weight.buffer_address()):
        raise ValueError("Output may not alias an input")
    kernel_output = output
    if not fp32_output and not direct_bf16_pack:
        kernel_output = t.empty(
            (1, 1, count, 640),
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            device=device,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 20:
        raise ValueError("Ordered projection requires20 compute cores")
    cores = [t.CoreCoord(i % grid.x, i // grid.x) for i in range(20)]
    ranges = t.CoreRangeSet([t.CoreRange(core, core) for core in cores])
    reader, writer, compute = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    for column, core in enumerate(cores):
        reader[core.x][core.y] = [hidden_states.buffer_address(), transposed_weight.buffer_address(), column, count]
        writer[core.x][core.y] = [kernel_output.buffer_address(), column]
        compute[core.x][core.y] = [block_size, count]
    read_ct = []
    for value in (hidden_states, transposed_weight):
        read_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    write_ct = [output_tile_bytes, *t.TensorAccessorArgs(kernel_output).get_compile_time_args()]
    configs = [
        ("fused_reader", read_ct, reader, t.ReaderConfigDescriptor()),
        ("ordered_projection_writer.cpp", write_ct, writer, t.WriterConfigDescriptor()),
        (
            "fused_compute" if valid_row_compute or copy_once else "ordered_projection_compute.cpp",
            [int(copy_once), int(valid_row_compute)],
            compute,
            t.ComputeConfigDescriptor(
                math_fidelity=t.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                math_approx_mode=False,
            ),
        ),
    ]
    # Scratch0/1 cache input and this core's weight tiles. Stream2/3 feed the
    # exact BF16 operands, and output4 is packed once after the FP32 sum.
    cbs = [
        t.CBDescriptor(
            total_size=tiles * 2048,
            core_ranges=ranges,
            format_descriptors=[t.CBFormatDescriptor(buffer_index=index, data_format=t.bfloat16, page_size=2048)],
        )
        for index, tiles in ((0, 80), (1, 80), (2, 2), (3, 2))
    ]
    cbs.append(
        t.CBDescriptor(
            total_size=output_tile_bytes,
            core_ranges=ranges,
            format_descriptors=[
                t.CBFormatDescriptor(buffer_index=4, data_format=kernel_dtype, page_size=output_tile_bytes)
            ],
        )
    )
    kernels = [
        t.KernelDescriptor(
            kernel_source=(
                _ORDERED_VALID_ROWS_READER
                if filename == "fused_reader"
                else _ORDERED_VALID_ROWS_COMPUTE
                if filename == "fused_compute"
                else str(_KERNEL_ROOT / filename)
            ),
            source_type=(
                t.KernelDescriptor.SourceType.SOURCE_CODE
                if filename in ("fused_reader", "fused_compute")
                else t.KernelDescriptor.SourceType.FILE_PATH
            ),
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for filename, ct, rt, config in configs
    ]
    t.generic_op(
        [hidden_states, transposed_weight, kernel_output], t.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    if not fp32_output and not direct_bf16_pack:
        # Direct BF16 pack rounds exact midpoints differently on this device.
        # Typecast explicitly rounds the FP32 bits to nearest-even before pack.
        # Supplying output_tensor preserves the caller's traced output address.
        t.typecast(kernel_output, t.bfloat16, output_tensor=output)
    return output


class FusedOrderedIndexerProjection:
    """Same ordered FP32 arithmetic; materialize broadcasts only for valid rows."""

    def __init__(self, original, *, valid_row_compute=False, copy_once=False):
        self.weight = original.weight
        self.valid_row_compute = valid_row_compute
        self.copy_once = copy_once

    def __call__(self, hidden_states):
        return fused_ordered_indexer_projection(
            hidden_states, self.weight, valid_row_compute=self.valid_row_compute, copy_once=self.copy_once
        )


class FusedDecoder(FunctionalDecoder):
    @classmethod
    def from_state_dict(
        cls,
        *args,
        fuse_expert_activation=True,
        fuse_shared_activation=True,
        fuse_residual_norm=False,
        fuse_residual_write=True,
        fuse_qsa_gate=True,
        fuse_qsa_concat=True,
        pack_shared=True,
        pack_experts=True,
        packed_expert_gate_cores=(10, 4),
        fuse_packed_activation=True,
        early_packed_activation=True,
        fuse_reduction=True,
        fuse_moe_cleanup=True,
        pack_gdn=True,
        pack_qsa=True,
        split_qsa_heads=False,
        merge_norm=False,
        merge_norm_rhs=False,
        fuse_ple=False,
        fuse_qsa_rope=True,
        decode_qsa_rope=False,
        fuse_precise_reductions=True,
        fuse_gdn_conv=False,
        fuse_gdn_conv_exact=True,
        fuse_gdn_layout=True,
        fuse_gdn_io=True,
        compact_experts=True,
        compact_prefill_limit=2,
        fuse_projection_cast=True,
        direct_packed_weights=True,
        projection_decode_group=64,
        residual_decode_group=None,
        preserve_qkv_tree=True,
        fuse_exact_qkv=True,
        qkv_decode_group=None,
        fuse_indexer_valid_rows=True,
        fuse_indexer_row_compute=False,
        fuse_indexer_copy_init=False,
        fuse_indexer_score_sum=True,
        merge_indexer_score_heads=True,
        fuse_qsa_selector=True,
        **kwargs,
    ):
        if not 1 <= compact_prefill_limit <= 32:
            raise ValueError("Compact prefill limit must be 1..32 logical rows")
        if compact_prefill_limit > 1 and not (compact_experts and pack_experts and fuse_packed_activation):
            raise ValueError("Compact prefill requires compact routes and packed expert activation")
        if fuse_exact_qkv and not (direct_packed_weights and fuse_precise_reductions and preserve_qkv_tree):
            raise ValueError("Exact QKV fusion requires direct checkpoint setup and preserved FP32 tree")
        if fuse_gdn_conv and fuse_gdn_conv_exact:
            raise ValueError("Choose one convolution fusion")
        if early_packed_activation and not fuse_packed_activation:
            raise ValueError("Early activation placement requires packed activation fusion")
        if fuse_packed_activation and (not pack_experts or not fuse_expert_activation):
            raise ValueError("Packed activation requires packed experts and expert activation fusion")
        if compact_experts and not fuse_reduction:
            raise ValueError("Compact experts require the fused weighted reduction")
        state_dict = args[0] if args else kwargs.get("state_dict")
        if direct_packed_weights and state_dict is None:
            raise ValueError("Direct packed setup requires checkpoint state_dict")
        model = super().from_state_dict(*args, **kwargs)
        prefix = f"model.language_model.layers.{model.layer_idx}."
        checkpoint = (
            ({k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict)
            if direct_packed_weights
            else None
        )

        def host_weight(name):
            return checkpoint[name] if checkpoint is not None else None

        fused = FusedSparseMoe.__new__(FusedSparseMoe)
        fused.__dict__.update(model.mlp.__dict__)
        fused.compact_experts = compact_experts
        fused.compact_prefill_limit = compact_prefill_limit
        fused.fuse_moe_cleanup = fuse_moe_cleanup
        fused.fuse_packed_activation = fuse_packed_activation
        fused.early_packed_activation = early_packed_activation
        fused.fuse_expert_activation = fuse_expert_activation
        fused.fuse_shared_activation = fuse_shared_activation
        # Transfer ownership before replacing expert buffers; no old component
        # may retain them during the direct checkpoint upload on capacity-bound PLE.
        model.mlp = fused
        fused.pack_projections(
            shared=pack_shared,
            experts=pack_experts,
            host_experts=host_weight("mlp.experts.gate_up_proj") if pack_experts else None,
            packed_gate_cores=packed_expert_gate_cores,
        )
        if fuse_reduction:
            fused.prepare_fused_reduce(model.mesh_device)
        model.mlp = fused
        if (
            pack_gdn or fuse_gdn_conv or fuse_gdn_conv_exact or fuse_gdn_layout or fuse_gdn_io
        ) and model.kind == "linear_attention":
            component = FusedGatedDeltaNet.__new__(FusedGatedDeltaNet)
            component.__dict__.update(model.attn.__dict__)
            component.zab = ttnn.concat([component.z, component.a, component.b], dim=-1)
            del component.z, component.a, component.b
            component.compact_recurrence_io = fuse_gdn_io
            component.fuse_recurrence_layout = fuse_gdn_layout
            component.fuse_conv_exact = fuse_gdn_conv_exact
            if fuse_gdn_conv:
                component.prepare_conv(model.mesh_device)
                component.conv_compute = ttnn.init_device_compute_kernel_config(
                    model.mesh_device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
            model.attn = component
        if fuse_residual_norm or merge_norm or merge_norm_rhs:
            for name in ("attn_mix", "mlp_mix"):
                component = FusedGatedResidual.__new__(FusedGatedResidual)
                component.__dict__.update(getattr(model, name).__dict__)
                component.merge_rsqrt = merge_norm
                component.merge_rhs = merge_norm_rhs
                setattr(model, name, component)
        if fuse_residual_write:
            for name in ("attn_mix", "mlp_mix"):
                original = getattr(model, name)
                cls = FusedNormResidualWriter if isinstance(original, FusedGatedResidual) else FusedResidualWriter
                component = cls.__new__(cls)
                component.__dict__.update(original.__dict__)
                setattr(model, name, component)
                del original
        if model.ple is not None and fuse_ple:
            component = FusedPLE.__new__(FusedPLE)
            component.__dict__.update(model.ple.__dict__)
            component.merge_norm = merge_norm_rhs
            model.ple = component
        if model.chunked_attn is not None and (
            fuse_qsa_gate
            or fuse_qsa_concat
            or pack_qsa
            or split_qsa_heads
            or fuse_qsa_rope
            or fuse_indexer_score_sum
            or merge_indexer_score_heads
            or fuse_indexer_valid_rows
            or fuse_indexer_row_compute
            or fuse_indexer_copy_init
            or fuse_qsa_selector
        ):
            model.chunked_attn = FusedChunkedQSA.from_component(
                model.chunked_attn,
                gate=fuse_qsa_gate,
                concat=fuse_qsa_concat,
                rope=fuse_qsa_rope,
                decode_rope=decode_qsa_rope,
                pack=pack_qsa,
                split=split_qsa_heads,
                score_sum=fuse_indexer_score_sum,
                merge_score_heads=merge_indexer_score_heads,
                valid_rows=fuse_indexer_valid_rows,
                row_compute=fuse_indexer_row_compute,
                copy_once=fuse_indexer_copy_init,
                selector=fuse_qsa_selector,
            )
            model.attn = model.chunked_attn.attention
        if fuse_precise_reductions:
            for name, prefix in (("attn_mix", "attn_hyper_connection."), ("mlp_mix", "mlp_hyper_connection.")):
                component = getattr(model, name)
                component.precise_down = FusedPreciseProjection(
                    component.precise_down,
                    component.compute,
                    projection_decode_group if residual_decode_group is None else residual_decode_group,
                    widen_in_unpack=fuse_projection_cast,
                    host_weight=host_weight(prefix + "input_mix_weight_down.weight"),
                )
                component.precise_up = FusedPreciseProjection(
                    component.precise_up,
                    component.compute,
                    projection_decode_group if residual_decode_group is None else residual_decode_group,
                    widen_in_unpack=fuse_projection_cast,
                    host_weight=host_weight(prefix + "input_mix_weight_up.weight"),
                )
            model.mlp.precise_router = FusedPreciseProjection(
                model.mlp.precise_router,
                model.mlp.compute,
                projection_decode_group,
                widen_in_unpack=fuse_projection_cast,
                host_weight=host_weight("mlp.gate.weight"),
            )
            if model.kind == "linear_attention":
                if fuse_exact_qkv:
                    model.attn.qkv = FusedExactQKV(model.attn.qkv, host_weight("linear_attn.in_proj_qkv.weight"))
                else:
                    model.attn.qkv = FusedPreciseProjection(
                        model.attn.qkv,
                        model.attn.compute,
                        projection_decode_group if qkv_decode_group is None else qkv_decode_group,
                        balanced_tree=preserve_qkv_tree,
                        widen_in_unpack=fuse_projection_cast,
                        host_weight=host_weight("linear_attn.in_proj_qkv.weight"),
                    )
                model.attn.out = FusedPreciseProjection(
                    model.attn.out,
                    model.attn.compute,
                    projection_decode_group,
                    widen_in_unpack=fuse_projection_cast,
                    host_weight=host_weight("linear_attn.out_proj.weight"),
                )
        return model
