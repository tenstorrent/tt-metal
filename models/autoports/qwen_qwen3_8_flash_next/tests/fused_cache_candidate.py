# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unselected B1 paired native K/V cache-write experiment.

The indexed op consumes physical token indices, not a page table. One small
device program translates current positions through the live page table before
the native in-place update. Raw index-key writes and all reads remain selected.
B>1 and chunks above 128 retain the existing transfer path explicitly.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

_BASE_QSA = fused.FusedQSA
_PHYSICAL_INDICES = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t seq = get_compile_time_arg_val(0);
    constexpr uint32_t page_rows = get_compile_time_arg_val(1);
    constexpr uint32_t capacity = get_compile_time_arg_val(2);
    constexpr uint32_t cache_pages = get_compile_time_arg_val(3);
    constexpr uint32_t output_offset = 192; // 64-byte position + at most 128-byte page window.
    constexpr auto pa = TensorAccessorArgs<4>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(0));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(1));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(2));
    CircularBuffer scratch(0);
    scratch.reserve_back(1);
    // Blackhole DRAM reads require matching source/destination residues mod64.
    // Reserve alignment slack even if the generic CB allocator returns mod32.
    const uint32_t base = (scratch.get_write_ptr() + 63) & ~63u;
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto pages = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + 64);
    auto physical = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + output_offset);
    Noc noc;
    noc.async_read(positions, CoreLocalMem<uint32_t>(base), 32, {.page_id=0}, {});
    noc.async_read_barrier();
    const uint32_t start = pos[0];
    ASSERT(start <= capacity && seq <= capacity - start);
    for (uint32_t token = 0; token < 128; ++token) physical[token] = -1;
    if (start <= capacity && seq <= capacity - start) {
        // At most five page IDs are needed for 128 rows with page_rows>=32.
        // Align the NoC source/destination offsets to 64 bytes. The final
        // aligned window can include row-major page padding but never crosses
        // the allocated aligned page; only valid logical entries are indexed.
        const uint32_t first_page = start / page_rows;
        const uint32_t last_page = (start + seq - 1) / page_rows;
        const uint32_t first_entry = first_page & ~15u;
        const uint32_t entries_end = (last_page + 16) & ~15u;
        const uint32_t read_bytes = (entries_end - first_entry) * 4;
        ASSERT(read_bytes <= 128);
        noc.async_read(table, CoreLocalMem<uint32_t>(base + 64), read_bytes,
                       {.page_id=0, .offset_bytes=first_entry * 4}, {});
        noc.async_read_barrier();
        for (uint32_t token = 0; token < seq; ++token) {
            const uint32_t absolute = start + token;
            const int32_t page = pages[absolute / page_rows - first_entry];
            ASSERT(page >= 0 && static_cast<uint32_t>(page) < cache_pages);
            if (page >= 0 && static_cast<uint32_t>(page) < cache_pages)
                physical[token] = static_cast<int32_t>(page * page_rows + absolute % page_rows);
        }
    }
    noc.async_write(CoreLocalMem<uint32_t>(base + output_offset), output, 512, {}, {.page_id=0});
    noc.async_write_barrier();
}
"""


def _replace_cache_writes(method, *, decode_only=False, update_method=None):
    source = textwrap.dedent(inspect.getsource(method))
    old = (
        "    for value, cache, count in ((k, self.k_cache, 2), (v, self.v_cache, 2), (raw, self.raw_key_cache, 1)):\n"
        "        self._transfer(value, cache, positions_packed, page_table, mode=0, sequence=seq, heads=count)"
    )
    condition = "batch == 1 and seq == 1" if decode_only else "batch == 1 and seq <= 128"
    update_method = update_method or ("_update_kv_paged" if decode_only else "_update_kv_indexed")
    new = (
        f"    if {condition}:\n"
        f"        self.{update_method}(k, v, positions_packed, page_table)\n"
        "        self._transfer(raw, self.raw_key_cache, positions_packed, page_table, mode=0, sequence=seq, heads=1)\n"
        "    else:\n"
        "        for value, cache, count in ((k, self.k_cache, 2), (v, self.v_cache, 2), (raw, self.raw_key_cache, 1)):\n"
        "            self._transfer(value, cache, positions_packed, page_table, mode=0, sequence=seq, heads=count)"
    )
    if source.count(old) != 1:
        raise RuntimeError("Expected exactly one paired K/V plus raw-key transfer loop in FusedQSA._chunk_forward")
    source = source.replace(old, new, 1)
    namespace = dict(method.__globals__)
    exec(compile(source, f"{__file__}:IndexedCacheQSA._chunk_forward", "exec"), namespace)
    rewritten = namespace[method.__name__]
    rewritten._rewritten_source = source
    return rewritten


class IndexedCacheQSA(_BASE_QSA):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        if obj.batch_size == 1:
            obj._physical_update_indices = ttnn.empty(
                (1, 128),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=obj.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            core = ttnn.CoreCoord(0, 0)
            obj._index_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        return obj

    def _update_kv_indexed(self, k, v, positions, page_table):
        seq = k.shape[2]
        if tuple(k.shape) != (1, 2, seq, 256) or tuple(v.shape) != tuple(k.shape) or not 1 <= seq <= 128:
            raise ValueError("Indexed-cache candidate requires B1/H2/D256 with 1..128 logical rows")
        if tuple(positions.shape) != (1, 32) or tuple(page_table.shape) != (1, self.max_context // self.page_size):
            raise ValueError("Expected packed B1 positions and full request page table")
        output = self._physical_update_indices
        size = 768  # alignment slack64 + position64 + page window128 + indices512.
        tensors = [positions, page_table, output]
        args = [seq, self.page_size, self.max_context, self.k_cache.shape[0]]
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        kernel = ttnn.KernelDescriptor(
            kernel_source=_PHYSICAL_INDICES,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self._index_core,
            compile_time_args=args,
            runtime_args=[((0, 0), [tensor.buffer_address() for tensor in tensors])],
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = ttnn.CBDescriptor(
            total_size=size,
            core_ranges=self._index_core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch]))
        # Cache and input tensors are distinct BF16 TILE interleaved allocations.
        # The operation is in-place, takes no program config, and reads the live
        # device indices on every replay. No layout conversion is needed here.
        ttnn.experimental.indexed_fused_update_cache(self.k_cache, k, self.v_cache, v, output)

    _chunk_forward = _replace_cache_writes(_BASE_QSA._chunk_forward)


class PagedDecodeCacheQSA(_BASE_QSA):
    """B1 S1 alternative using two explicitly disjoint L1 shard placements.

    Mirrors test_paged_fused_update_cache.py's num_users=1 tiled contract.
    Its permutes, placements and index cast are intentionally inside measured
    runtime; only the shard specifications are prepared at construction.
    """

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        if obj.batch_size == 1:
            obj._paged_cache_input_configs = []
            for x in (0, 1):
                core = ttnn.CoreCoord(x, 0)
                grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
                shard = ttnn.ShardSpec(grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
                obj._paged_cache_input_configs.append(
                    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
                )
        return obj

    def _update_kv_paged(self, k, v, positions, page_table):
        if tuple(k.shape) != (1, 2, 1, 256) or tuple(v.shape) != tuple(k.shape):
            raise ValueError("Paged decode-cache candidate requires B1/S1/H2/D256")
        key = ttnn.to_memory_config(ttnn.permute(k, (0, 2, 1, 3)), self._paged_cache_input_configs[0])
        value = ttnn.to_memory_config(ttnn.permute(v, (0, 2, 1, 3)), self._paged_cache_input_configs[1])
        # Packed [1,32] RM contains one batch row. Native validation derives
        # num_indices from dim0 and its reader consumes entry0 for that user.
        update_indices = ttnn.typecast(positions, ttnn.int32)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache,
            key,
            self.v_cache,
            value,
            update_idxs_tensor=update_indices,
            page_table=page_table,
        )

    _chunk_forward = _replace_cache_writes(_BASE_QSA._chunk_forward, decode_only=True)


class IndexedDecodeCacheQSA(IndexedCacheQSA):
    _chunk_forward = _replace_cache_writes(
        _BASE_QSA._chunk_forward, decode_only=True, update_method="_update_kv_indexed"
    )


def install(name="indexed_cache"):
    candidates = {
        "indexed_cache": IndexedCacheQSA,
        "indexed_decode_cache": IndexedDecodeCacheQSA,
        "paged_decode_cache": PagedDecodeCacheQSA,
    }
    if name not in candidates:
        raise ValueError(f"Unknown cache experiment {name!r}; choose one of {tuple(candidates)}")
    fused.FusedQSA = candidates[name]
    return candidates[name]


def snapshot_provenance(directory):
    """Setup-only source snapshots for benchmark evidence, including native code."""
    checkout = Path(__file__).resolve().parents[4]
    native = Path("ttnn/cpp/ttnn/operations/experimental/indexed_page_cache")
    paths = [
        Path(__file__).resolve().relative_to(checkout),
        Path(fused.__file__).resolve().relative_to(checkout),
        native / "indexed_page_cache_nanobind.cpp",
        native / "indexed_page_cache.cpp",
        native / "device/indexed_fused_update_cache/indexed_fused_update_cache_device_operation.cpp",
        native / "device/indexed_fused_update_cache/indexed_fused_update_cache_program_factory.cpp",
        native / "device/kernels/dataflow/indexed_fused_update_cache.cpp",
        Path("ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache_nanobind.cpp"),
        Path(
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp"
        ),
        Path(
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp"
        ),
        Path(
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp"
        ),
        Path(
            "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp"
        ),
        Path("ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp"),
    ]
    output = Path(directory) / "indexed_cache_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for index, relative in enumerate(paths):
        data = (checkout / relative).read_bytes()
        snapshot = output / f"source_{index:02d}.txt"
        snapshot.write_bytes(data)
        manifest[str(relative)] = {"sha256": hashlib.sha256(data).hexdigest(), "snapshot": snapshot.name}
    manifest["embedded_physical_indices"] = {"sha256": hashlib.sha256(_PHYSICAL_INDICES.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
