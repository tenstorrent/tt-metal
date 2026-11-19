# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Non-default BFP8 local KV storage with selected-row BF16 decompression.

Reads aligned exponent/mantissa lines for selected rows only; no full-cache
conversion, no changed selection, no logical sequence alignment requirement.
The indexer compressed-key cache and raw history retain their exact BF16 policy.
"""
from pathlib import Path


class LocalSelectedPagedKVBFP8:
    def __init__(self, *, k_cache, v_cache, page_table, selected_ids, keys, values):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [k_cache, v_cache, page_table, selected_ids, keys, values]
        shape = tuple(k_cache.shape)
        if len(shape) != 4 or shape[1] != 1 or shape[3] != 256 or shape[2] % 32:
            raise ValueError("Expected paged caches [physical_pages,1,page_size,256], page_size divisible by32")
        if tuple(v_cache.shape) != shape or min(shape) < 1:
            raise ValueError("K/V cache shapes must agree and be nonempty")
        if len(page_table.shape) != 2 or page_table.shape[0] != 1:
            raise ValueError("Page table must be [1,P]")
        if page_table.shape[1] < 1 or page_table.dtype not in (ttnn.int32, ttnn.uint32):
            raise ValueError("Page table must have INT32/UINT32 entries")
        if page_table.shape[1] % 16 and k_cache.device().arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Short unaligned page-table sticks currently require Blackhole DRAM alignment")
        # Blackhole NOC_DRAM_READ_ALIGNMENT_BYTES is64. Buffer::aligned_size()
        # reserves every RM page rounded to that alignment, and TensorAccessorArgs
        # serializes aligned_page_size. Thus the final64-byte page-table read is
        # within the physical allocation even for logical P=1 or4. Other targets
        # are required to provide a full64-byte-multiple logical stick above.
        if page_table.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Page table must be row-major")
        if len(selected_ids.shape) != 2 or selected_ids.shape[1] != 2080 or not 1 <= selected_ids.shape[0] <= 32:
            raise ValueError("selected_ids must be [Q,2080], 1<=Q<=32")
        if selected_ids.dtype != ttnn.uint32 or selected_ids.layout != ttnn.TILE_LAYOUT:
            raise ValueError("selected_ids must be tiled UINT32")
        queries = selected_ids.shape[0]
        out_shape = (queries, 1, 2080, 256)
        if tuple(keys.shape) != out_shape or tuple(values.shape) != out_shape:
            raise ValueError("Output shapes must be [Q,1,2080,256]")
        for tensor in (k_cache, v_cache):
            if tensor.dtype != ttnn.bfloat8_b or tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError("K/V cache must be BFLOAT8_B TILE")
        for tensor in (keys, values):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError("K/V output must be BF16 TILE")
        for tensor in self.tensors:
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("All tensors must be interleaved DRAM")
        addresses = [tensor.buffer_address() for tensor in self.tensors]
        if addresses[4] in addresses[:4] or addresses[5] in addresses[:5]:
            raise ValueError("Output buffers must not alias input buffers or each other")
        group_count = queries * 65
        grid = k_cache.device().compute_with_storage_grid_size()
        core_count = min(group_count, grid.x * grid.y)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(core_count)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        runtime_args = ttnn.RuntimeArgs()
        for number, core in enumerate(cores):
            runtime_args[core.x][core.y] = [
                *addresses,
                group_count * number // core_count,
                group_count * (number + 1) // core_count,
            ]
        compile_args = [shape[2], shape[0], page_table.shape[1], queries]
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(Path(__file__).with_name("kernels") / "local_selected_paged_kv_bfp8.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=runtime_args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=40960,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=40960)
                    ],
                )
            ],
            semaphores=[],
        )

    def prepare(self):
        self._ttnn.experimental.prepare_generic_op(self.tensors, self.program)

    def forward(self):
        output = self._ttnn.generic_op(self.tensors, self.program)
        return self.tensors[-2], output


def local_selected_paged_kv_bfp8(k_cache, v_cache, page_table, selected_ids, *, keys=None, values=None):
    """Functional device-only gather; optional outputs must be stable owned buffers."""
    import ttnn

    shape = (selected_ids.shape[0], 1, 2080, 256)
    if keys is None:
        keys = ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=k_cache.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if values is None:
        values = ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=k_cache.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    operation = LocalSelectedPagedKVBFP8(
        k_cache=k_cache, v_cache=v_cache, page_table=page_table, selected_ids=selected_ids, keys=keys, values=values
    )
    return operation.forward()
