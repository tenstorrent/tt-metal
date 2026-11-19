# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct BF16 paged K/V reads for a bounded QSA selected-token set.

Public device API:
    selected_paged_kv(k_cache, v_cache, page_table, selected_ids) -> (keys, values)

Caches: BF16 TILE [physical_pages,2,page_size,256], page_size divisible by32.
Page table: INT32/UINT32 ROW_MAJOR [1,P], P positive. Blackhole DRAM rounds
the physical stick to64 bytes; other architectures require P divisible by16.
IDs: UINT32 TILE [Q,2080], 1<=Q<=32. Each ID is an absolute logical token;
UINT32_MAX is invalid and yields an all-zero row. The caller must encode all
future, incomplete-block and padding entries as UINT32_MAX. The validity mask
is device-derivable from selected_ids != UINT32_MAX; this module does not infer
current positions or select blocks. Outputs: BF16 TILE [Q,2,2080,256].

The dataflow kernel visits selected IDs only, reads their page-table entries,
and issues aligned 64-byte NoC cache reads. It never scans the full K/V cache.
It owns eight channel tiles per work group; this avoids races between adjacent
selected-token output rows. All invalid rows and padding are explicitly zero.

For trace use, warm the functional signature before capture, or construct
SelectedPagedKV with persistent output buffers and call prepare()/forward().
Both cache buffers and the page table are read-only. No host read/conversion
occurs in forward. CPU address proofs can import this file without TTNN.
"""

_ADDRESS_SOURCE = r"""
inline uint32_t qsa_face_element(uint32_t row, uint32_t column) {
    return (((row >> 4) * 2u + (column >> 4)) * 256u) + (row & 15u) * 16u + (column & 15u);
}
inline uint32_t qsa_cache_tile(uint32_t physical_page, uint32_t head,
                               uint32_t page_size, uint32_t token_in_page, uint32_t channel_tile) {
    return (((physical_page * 2u + head) * (page_size / 32u) + token_in_page / 32u) * 8u) + channel_tile;
}
"""


def kernel_source():
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _ADDRESS_SOURCE
        + r"""
void kernel_main() {
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t PHYSICAL_PAGES = get_compile_time_arg_val(1);
    constexpr uint32_t LOGICAL_PAGES = get_compile_time_arg_val(2);
    constexpr uint32_t QUERIES = get_compile_time_arg_val(3);
    constexpr uint32_t SLOT_TILES = 65; // 2080 selected slots
    constexpr auto ka = TensorAccessorArgs<4>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto koa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto voa = TensorAccessorArgs<koa.next_compile_time_args_offset()>();
    const auto keys = TensorAccessor(ka, get_arg_val<uint32_t>(0), 2048);
    const auto values = TensorAccessor(va, get_arg_val<uint32_t>(1), 2048);
    const auto pages = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(3), 4096);
    const auto kout = TensorAccessor(koa, get_arg_val<uint32_t>(4), 2048);
    const auto vout = TensorAccessor(voa, get_arg_val<uint32_t>(5), 2048);
    const uint32_t begin = get_arg_val<uint32_t>(6);
    const uint32_t end = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t kp = scratch;
    const uint32_t vp = scratch + 16384;
    const uint32_t ip = scratch + 32768;
    const uint32_t pp = scratch + 36864;
    const uint32_t lp = scratch + 36928;
    volatile tt_l1_ptr uint32_t* kw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp);
    volatile tt_l1_ptr uint32_t* vw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vp);
    volatile tt_l1_ptr uint16_t* kh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp);
    volatile tt_l1_ptr uint16_t* vh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(vp);
    volatile tt_l1_ptr uint32_t* selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    volatile tt_l1_ptr uint32_t* page_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp);
    volatile tt_l1_ptr uint16_t* lines = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lp);
    uint64_t previous_line[32];
    for (uint32_t group = begin; group < end; ++group) {
        const uint32_t query = group / (2 * SLOT_TILES);
        const uint32_t head = (group / SLOT_TILES) % 2;
        const uint32_t slot_tile = group % SLOT_TILES;
        ASSERT(query < QUERIES);
        for (uint32_t i = 0; i < 4096; ++i) { kw[i] = 0; vw[i] = 0; }
        for (uint32_t i = 0; i < 32; ++i) previous_line[i] = ~uint64_t(0);
        uint32_t previous_page_line = 0xffffffffu;
        // Q<=32, therefore selected_ids has exactly one physical tile row.
        noc_async_read(ids.get_noc_addr(slot_tile), ip, 4096);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t token = selected[qsa_face_element(query, row)];
            if (token == 0xffffffffu) continue;
            ASSERT(token < LOGICAL_PAGES * PAGE_SIZE);
            const uint32_t logical_page = token / PAGE_SIZE;
            const uint32_t page_line = (logical_page * 4u) & ~63u;
            if (page_line != previous_page_line) {
                noc_async_read(pages.get_noc_addr(0) + page_line, pp, 64);
                noc_async_read_barrier();
                previous_page_line = page_line;
            }
            const uint32_t physical_page = page_words[logical_page & 15u];
            ASSERT(physical_page < PHYSICAL_PAGES);
            const uint32_t local_token = token % PAGE_SIZE;
            const uint32_t source_row = local_token & 31u;
            // Eight channels and two faces are read in parallel for K and V.
            // Consecutive selected tokens often share their aligned source
            // lines; retain each channel/face line until its address changes.
            for (uint32_t channel = 0; channel < 8; ++channel) {
                const uint32_t tile = qsa_cache_tile(physical_page, head, PAGE_SIZE, local_token, channel);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t offset = qsa_face_element(source_row, half * 16) * 2u;
                    const uint32_t aligned = offset & ~63u;
                    const uint32_t line = channel * 2 + half;
                    const uint64_t kaddr = keys.get_noc_addr(tile) + aligned;
                    const uint64_t vaddr = values.get_noc_addr(tile) + aligned;
                    if (kaddr != previous_line[line]) {
                        noc_async_read(kaddr, lp + line * 64, 64);
                        previous_line[line] = kaddr;
                    }
                    if (vaddr != previous_line[line + 16]) {
                        noc_async_read(vaddr, lp + (line + 16) * 64, 64);
                        previous_line[line + 16] = vaddr;
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t channel = 0; channel < 8; ++channel) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t line = channel * 2 + half;
                    const uint32_t source_offset = (qsa_face_element(source_row, half * 16) * 2u & 63u) / 2u;
                    const uint32_t dest = channel * 1024 + qsa_face_element(row, half * 16);
                    for (uint32_t element = 0; element < 16; ++element) {
                        kh[dest + element] = lines[line * 32 + source_offset + element];
                        vh[dest + element] = lines[(line + 16) * 32 + source_offset + element];
                    }
                }
            }
        }
        const uint32_t first_output_tile = group * 8;
        for (uint32_t channel = 0; channel < 8; ++channel) {
            noc_async_write(kp + channel * 2048, kout.get_noc_addr(first_output_tile + channel), 2048);
            noc_async_write(vp + channel * 2048, vout.get_noc_addr(first_output_tile + channel), 2048);
        }
        noc_async_write_barrier();
    }
}
"""
    )


class SelectedPagedKV:
    def __init__(self, *, k_cache, v_cache, page_table, selected_ids, keys, values):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [k_cache, v_cache, page_table, selected_ids, keys, values]
        shape = tuple(k_cache.shape)
        if len(shape) != 4 or shape[1] != 2 or shape[3] != 256 or shape[2] % 32:
            raise ValueError("Expected paged caches [physical_pages,2,page_size,256], page_size divisible by32")
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
        out_shape = (queries, 2, 2080, 256)
        if tuple(keys.shape) != out_shape or tuple(values.shape) != out_shape:
            raise ValueError("Output shapes must be [Q,2,2080,256]")
        for tensor in (k_cache, v_cache, keys, values):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError("K/V cache and outputs must be BF16 TILE")
        for tensor in self.tensors:
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("All tensors must be interleaved DRAM")
        addresses = [tensor.buffer_address() for tensor in self.tensors]
        if addresses[4] in addresses[:4] or addresses[5] in addresses[:5]:
            raise ValueError("Output buffers must not alias input buffers or each other")
        group_count = queries * 2 * 65
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
                    kernel_source=kernel_source(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
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


def selected_paged_kv(k_cache, v_cache, page_table, selected_ids, *, keys=None, values=None):
    """Functional device-only gather; optional outputs must be stable owned buffers."""
    import ttnn

    shape = (selected_ids.shape[0], 2, 2080, 256)
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
    operation = SelectedPagedKV(
        k_cache=k_cache, v_cache=v_cache, page_table=page_table, selected_ids=selected_ids, keys=keys, values=values
    )
    return operation.forward()
