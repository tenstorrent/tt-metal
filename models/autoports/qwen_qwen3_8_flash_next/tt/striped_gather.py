# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct selected-cell NoC reads from a 32-row tiled striped table.

`striped_gather(table, columns, stripes)` returns a TILE tensor with the same
logical shape as columns/stripes and table's dtype. Inputs are rank-2 device
TILE tensors: table [32,C], columns/stripes [N,W], UINT32 indices. Each result
is table[stripes[n,w], columns[n,w]]. Supported table dtypes are UINT32, BF16
and FLOAT32. This operation never scans unselected table tiles.

`StripedGather(table=..., columns=..., stripes=..., output=...)` uses four
caller-owned persistent buffers. Call prepare() before trace capture, then
forward() allocates nothing. The convenience function allocates only its
output and builds a descriptor using current device buffer addresses; warm
every physical shape/dtype before capturing it. GenericOp hashes the compile
signature, not runtime argument values, so new addresses reuse the program.

No imports of torch/TTNN occur at module load, permitting a CPU-only proof of
the exact shared tile address helper. Runtime methods do not inspect tensor
contents or convert to/from host memory.
"""

_ADDRESS_SOURCE = r"""
inline uint32_t striped_tile_element(uint32_t stripe, uint32_t column) {
    const uint32_t tile_column = column & 31u;
    const uint32_t face = ((stripe >> 4) << 1) | (tile_column >> 4);
    return (face << 8) | ((stripe & 15u) << 4) | (tile_column & 15u);
}
"""


def kernel_source():
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _ADDRESS_SOURCE
        + r"""
void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t VALUE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t INPUT_WT = (W + 31u) / 32u;
    constexpr uint32_t OUTPUT_BYTES = 1024u * VALUE_BYTES;
    constexpr auto ta = TensorAccessorArgs<4>();
    constexpr auto ca = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(0), OUTPUT_BYTES);
    const auto columns = TensorAccessor(ca, get_arg_val<uint32_t>(1), 4096);
    const auto stripes = TensorAccessor(sa, get_arg_val<uint32_t>(2), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(3), OUTPUT_BYTES);
    const uint32_t begin = get_arg_val<uint32_t>(4);
    const uint32_t end = get_arg_val<uint32_t>(5);
    // All four regions start on 64-byte boundaries, including the selected
    // source line. A 64-byte read never crosses a 2048/4096-byte tile page.
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t out_ptr = scratch + 8192;
    const uint32_t line_ptr = scratch + 12288;
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 4096);
    volatile tt_l1_ptr uint32_t* out32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ptr);
    volatile tt_l1_ptr uint16_t* out16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_ptr);
    volatile tt_l1_ptr uint32_t* line32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(line_ptr);
    volatile tt_l1_ptr uint16_t* line16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(line_ptr);
    for (uint32_t tile = begin; tile < end; ++tile) {
        noc_async_read(columns.get_noc_addr(tile), scratch, 4096);
        noc_async_read(stripes.get_noc_addr(tile), scratch + 4096, 4096);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < OUTPUT_BYTES / 4; ++i) out32[i] = 0;
        const uint32_t tile_row = tile / INPUT_WT;
        const uint32_t tile_col = tile % INPUT_WT;
        for (uint32_t element = 0; element < 1024; ++element) {
            const uint32_t face = element >> 8;
            const uint32_t local_row = ((face >> 1) << 4) + ((element & 255u) >> 4);
            const uint32_t local_col = ((face & 1u) << 4) + (element & 15u);
            if (tile_row * 32 + local_row >= H || tile_col * 32 + local_col >= W) continue;
            const uint32_t column = c[element];
            const uint32_t stripe = s[element];
            ASSERT(column < C && stripe < 32);
            const uint32_t byte_offset = striped_tile_element(stripe, column) * VALUE_BYTES;
            const uint32_t aligned_offset = byte_offset & ~63u;
            // Tile id is column/32 because the source has exactly one tile row.
            // TensorAccessor handles DRAM bank interleaving even for >4GB tables.
            noc_async_read(table.get_noc_addr(column >> 5) + aligned_offset, line_ptr, 64);
            noc_async_read_barrier();
            if constexpr (VALUE_BYTES == 4) out32[element] = line32[(byte_offset & 63u) >> 2];
            else out16[element] = line16[(byte_offset & 63u) >> 1];
        }
        noc_async_write(out_ptr, output.get_noc_addr(tile), OUTPUT_BYTES);
        // Do not reuse the output scratch until its DMA write completes.
        noc_async_write_barrier();
    }
}
"""
    )


class StripedGather:
    """Prepared direct-gather operation with explicitly owned stable buffers."""

    def __init__(self, *, table, columns, stripes, output, kernel_source_fn=None):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [table, columns, stripes, output]
        if len(table.shape) != 2 or table.shape[0] != 32 or table.shape[1] < 32 or table.shape[1] % 32:
            raise ValueError("table must be TILE [32,C], with C divisible by 32")
        if table.dtype not in (ttnn.uint32, ttnn.bfloat16, ttnn.float32):
            raise ValueError("Only UINT32/BF16/FLOAT32 table values are supported")
        if len(columns.shape) != 2 or min(columns.shape) < 1 or tuple(columns.shape) != tuple(stripes.shape):
            raise ValueError("columns and stripes must share nonempty rank-2 [N,W] shapes")
        if columns.dtype != ttnn.uint32 or stripes.dtype != ttnn.uint32:
            raise ValueError("columns and stripes must be exact UINT32")
        if tuple(output.shape) != tuple(columns.shape) or output.dtype != table.dtype:
            raise ValueError("output must match index logical shape and table dtype")
        for tensor in self.tensors:
            if tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("All buffers must be tiled interleaved DRAM")
            tile = tensor.get_tile()
            if tuple(tile.tile_shape) != (32, 32) or tile.transpose_within_face or tile.transpose_of_faces:
                raise ValueError("Expected standard 32x32 tiles")
            if tuple(tensor.padded_shape) != tuple((value + 31) // 32 * 32 for value in tensor.shape):
                raise ValueError("Unexpected physical padding")
            if tensor.device() != table.device():
                raise ValueError("All buffers must share the same mesh")
        if output.buffer_address() in [tensor.buffer_address() for tensor in self.tensors[:-1]]:
            raise ValueError("Output must not alias any input")
        rows, width = tuple(columns.shape)
        tile_count = ((rows + 31) // 32) * ((width + 31) // 32)
        grid = table.device().compute_with_storage_grid_size()
        core_count = min(tile_count, grid.x * grid.y)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(core_count)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        args = ttnn.RuntimeArgs()
        addresses = [tensor.buffer_address() for tensor in self.tensors]
        for number, core in enumerate(cores):
            args[core.x][core.y] = [
                *addresses,
                tile_count * number // core_count,
                tile_count * (number + 1) // core_count,
            ]
        compile_args = [rows, width, table.shape[1], 2 if table.dtype == ttnn.bfloat16 else 4]
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=(kernel_source if kernel_source_fn is None else kernel_source_fn)(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=16384,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=16384)
                    ],
                )
            ],
            semaphores=[],
        )

    def prepare(self):
        """Compile/cache before trace capture without reading source values."""
        self._ttnn.experimental.prepare_generic_op(self.tensors, self.program)

    def forward(self):
        """Execute direct indexed reads into the caller-provided output."""
        return self._ttnn.generic_op(self.tensors, self.program)


def striped_gather(table, columns, stripes, *, output=None, kernel_source_fn=None):
    """Functional form: returns [N,W], rather than a flattened [N*W,1].

    Optional output must satisfy the stable-buffer class contract. A newly
    allocated output is ordinary transient device memory, just like TTNN ops;
    warm the signature before capture and consume capture output before reuse.
    """
    import ttnn

    if output is None:
        output = ttnn.empty(
            tuple(columns.shape),
            dtype=table.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=table.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    operation = StripedGather(
        table=table, columns=columns, stripes=stripes, output=output, kernel_source_fn=kernel_source_fn
    )
    return operation.forward()
