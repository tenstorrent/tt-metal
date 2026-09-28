// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorAccessor <-> DFB pattern matrix for bringing up the Quasar HW address generator behind
// TensorAccessor.
//
// Every row is a Metal 2.0 program of two DM kernels on one node, connected by a DFB:
//   ReadOnly  : ta_reader_to_dfb (tensor -> DFB, AddrGen src candidate) + ta_writer_from_dfb forced
//               onto the SW path; host checks out == in. Isolates the read side.
//   WriteOnly : ta_fill_dfb (device-generated, page-id-tagged pattern) + ta_writer_from_dfb
//               (AddrGen dst candidate); host checks out == pattern. Isolates the write side and
//               catches a writer that walks pages out of logical order.
//   Copy      : reader + writer both AddrGen candidates; host checks out == in. The common case.
//
// Tensor-side kernels use DFB implicit sync (async_read/async_write<NocOptions::TXN_ID>) by default;
// a few *_Explicit rows keep the reserve/push/wait/pop path covered.
//
// Phase A: every row runs on the SW get_noc_addr path. The rows are the golden for Phase B, which
// flips eligible layouts onto the HW AddrGen path inside Noc::async_read / async_write.
//
// Sized for emu-quasar-2x3 (2 worker Tensix on one row). Rows that need a larger grid or more DRAM
// banks stay in the suite and GTEST_SKIP with the requirement, so they run unchanged on a larger
// emulator or silicon. Run with ATT enabled:
//   TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_NOC_ATT=quasar_aether_2x3 TT_METAL_ATT_PROGRAM_FOR_TEST=1

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal {

namespace unit_tests::dm::ta_addrgen {

namespace m2 = experimental;

constexpr auto kReaderKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_reader_to_dfb.cpp";
constexpr auto kWriterKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_writer_from_dfb.cpp";
constexpr auto kFillKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_fill_dfb.cpp";

// Per-kernel opt-out from the HW AddrGen path. Phase A has no HW path, so this is a no-op until
// Phase B honors it in the Quasar Noc hooks.
constexpr auto kDisableAddrgenDefine = "TT_TA_ADDRGEN_DISABLE";

constexpr uint32_t kNumDfbEntries = 4;

enum class KernelShape { ReadOnly, WriteOnly, Copy };
enum class IterMode : uint32_t { PageIdLoop = 0, PagesIterator = 1 };

// Row-major UINT32 tensors throughout: one page is one row (interleaved / HEIGHT) or one
// shard-width row segment (WIDTH / BLOCK), and the host vector is in logical page order.
// TILE-layout pages are always exactly one tile, regardless of sharding (get_page_shape_tile in
// page_config.cpp ignores shard shape); ROW_MAJOR bundles a whole shard-row segment into one page for
// WIDTH/BLOCK. So a page is 1 row tall under ROW_MAJOR but kTileDim rows tall under TILE.
constexpr uint32_t kTileDim = 32;

// Element byte size for the small set of dtypes this suite uses. All chosen shapes keep
// page_size_bytes a multiple of 4 for every one of these, so the uint32-word golden below stays valid.
uint32_t elem_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8: return 1;
        case DataType::UINT16: return 2;
        case DataType::UINT32: return 4;
        default: TT_THROW("elem_size_bytes: unsupported dtype {}", dtype);
    }
}

// tt::is_data_format_supported(_, ARCH::QUASAR) rejects plain UInt16/UInt32 (see is_supported_quasar
// in tt_backend_api_types.cpp); it accepts the Raw* family, which is also the right semantic match for
// a DM-only DFB that never interprets these bytes as float/int, just moves them.
DataFormat to_data_format(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8: return DataFormat::RawUInt8;
        case DataType::UINT16: return DataFormat::RawUInt16;
        case DataType::UINT32: return DataFormat::RawUInt32;
        default: TT_THROW("to_data_format: unsupported dtype {}", dtype);
    }
}

struct LayoutCase {
    std::string name;
    BufferType buffer_type = BufferType::DRAM;
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;
    Layout layout = Layout::ROW_MAJOR;
    DataType dtype = DataType::UINT32;
    // Logical shape in dtype elements. Rank 2 unless nd_outer > 1 (then [nd_outer, rows, cols]).
    // For Layout::TILE, rows/cols/shard_rows/shard_cols must be multiples of kTileDim.
    uint32_t rows = 16;
    uint32_t cols = 64;
    uint32_t nd_outer = 1;
    // Sharded only.
    uint32_t shard_rows = 0;
    uint32_t shard_cols = 0;
    CoreCoord shard_grid = {1, 1};
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    bool nd_round_robin = false;  // ND spec with ROUND_ROBIN_1D distribution (else legacy 2D ShardSpec)
    // Platform requirements; rows that exceed the platform GTEST_SKIP instead of being dropped.
    uint32_t min_dram_banks = 1;
    // If set, shard_grid/rows/cols above are only the per-bank granule (a safe 1-bank fallback if
    // resolve_grid() is somehow skipped); resolve_grid() rescales them to every worker core / DRAM bank
    // the device actually has, so the row automatically stresses a bigger banking loop -- including the
    // BANK_END wrap-around -- on a bigger emulator or silicon, with no code change.
    bool use_max_grid = false;
    // Only meaningful with use_max_grid: this row claims a genuine 2D split (both shard_grid dims > 1),
    // so it must skip rather than silently degenerate to a 1D split when the device grid is a single row.
    bool require_2d_grid = false;
};

uint32_t page_rows(const LayoutCase& lc) { return lc.layout == Layout::TILE ? kTileDim : 1; }

uint32_t page_cols(const LayoutCase& lc) {
    if (lc.layout == Layout::TILE) {
        return kTileDim;
    }
    return (lc.memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            lc.memory_layout == TensorMemoryLayout::BLOCK_SHARDED)
               ? lc.shard_cols
               : lc.cols;
}

uint32_t page_size_bytes(const LayoutCase& lc) { return page_cols(lc) * page_rows(lc) * elem_size_bytes(lc.dtype); }

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Ceiling, not floor: a dim that isn't an exact multiple of its page granule (e.g. TILE rows/cols not
// a multiple of kTileDim) still gets a final, partially-padding page -- the physical buffer allocates
// it and the golden below writes/reads it as an opaque page, so counting it is correct, not optional.
uint32_t num_pages(const LayoutCase& lc) {
    return lc.nd_outer * ceil_div(lc.rows, page_rows(lc)) * ceil_div(lc.cols, page_cols(lc));
}

bool is_sharded(const LayoutCase& lc) { return lc.memory_layout != TensorMemoryLayout::INTERLEAVED; }

// Rescales a use_max_grid row's shard_grid/rows/cols to the device's actual worker grid (L1) or DRAM
// bank count (DRAM), keeping the per-bank granule (shard_rows/shard_cols) from the case definition.
// A no-op for rows that don't opt in, so it is safe to call on every row.
LayoutCase resolve_grid(distributed::MeshDevice& device, LayoutCase lc) {
    if (!lc.use_max_grid || !is_sharded(lc)) {
        return lc;
    }
    // L1 shards are placed on the physical 2D compute grid; DRAM shards are a linear list of banks (no
    // second dimension), matching how the existing fixed-size Height/Width/BlockDram rows use {N, 1}.
    lc.shard_grid = lc.buffer_type == BufferType::L1
                        ? device.compute_with_storage_grid_size()
                        : CoreCoord{device.allocator()->get_num_banks(BufferType::DRAM), 1};
    const uint32_t num_shards = lc.shard_grid.x * lc.shard_grid.y;
    switch (lc.memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED: lc.rows = num_shards * lc.shard_rows; break;
        case TensorMemoryLayout::WIDTH_SHARDED: lc.cols = num_shards * lc.shard_cols; break;
        case TensorMemoryLayout::BLOCK_SHARDED:
            // get_shape_fits_shard_grid_error (tensor_spec.cpp) swaps which grid axis maps to
            // height/width for COL_MAJOR vs ROW_MAJOR; match it or a non-square grid TT_FATALs.
            if (lc.orientation == ShardOrientation::ROW_MAJOR) {
                lc.rows = lc.shard_grid.y * lc.shard_rows;
                lc.cols = lc.shard_grid.x * lc.shard_cols;
            } else {
                lc.rows = lc.shard_grid.x * lc.shard_rows;
                lc.cols = lc.shard_grid.y * lc.shard_cols;
            }
            break;
        default: break;
    }
    return lc;
}

TensorSpec make_tensor_spec(const LayoutCase& lc) {
    const Shape logical_shape = lc.nd_outer > 1 ? Shape{lc.nd_outer, lc.rows, lc.cols} : Shape{lc.rows, lc.cols};
    std::optional<MemoryConfig> memory_config;
    if (!is_sharded(lc)) {
        memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, lc.buffer_type};
    } else {
        const CoreRangeSet grid(CoreRange({0, 0}, {lc.shard_grid.x - 1, lc.shard_grid.y - 1}));
        if (lc.nd_outer > 1 || lc.nd_round_robin) {
            const Shape shard_shape =
                lc.nd_outer > 1 ? Shape{1, lc.shard_rows, lc.shard_cols} : Shape{lc.shard_rows, lc.shard_cols};
            // GRID_2D requires shard_shape.rank() <= 2 (BufferDistributionSpec::compute_core_list); this
            // branch is only reached when nd_outer > 1 (rank-3 shard_shape above) or nd_round_robin is
            // requested, so ROUND_ROBIN_1D is the only valid strategy here in either case.
            NdShardSpec nd_spec{
                .shard_shape = shard_shape,
                .grid = grid,
                .orientation = lc.orientation,
                .shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D,
            };
            memory_config = MemoryConfig{lc.buffer_type, nd_spec};
        } else {
            ShardSpec shard_spec{grid, {lc.shard_rows, lc.shard_cols}, lc.orientation};
            memory_config = MemoryConfig{lc.memory_layout, lc.buffer_type, shard_spec};
        }
    }
    auto tensor_layout = TensorLayout(lc.dtype, PageConfig(lc.layout), *memory_config);
    return TensorSpec(logical_shape, tensor_layout);
}

uint32_t fill_pattern_word(uint32_t page_id, uint32_t word) { return (page_id << 16) | word; }

std::string skip_reason(distributed::MeshDevice& device, const LayoutCase& lc) {
    const CoreCoord grid = device.compute_with_storage_grid_size();
    if (is_sharded(lc) && lc.buffer_type == BufferType::L1 && (lc.shard_grid.x > grid.x || lc.shard_grid.y > grid.y)) {
        return fmt::format(
            "needs a >= {}x{} worker grid (have {}x{}); run on a larger emulator or silicon",
            lc.shard_grid.x,
            lc.shard_grid.y,
            grid.x,
            grid.y);
    }
    const uint32_t dram_banks = device.allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t needed_dram_banks =
        (is_sharded(lc) && lc.buffer_type == BufferType::DRAM) ? lc.shard_grid.x * lc.shard_grid.y : lc.min_dram_banks;
    if (dram_banks < needed_dram_banks) {
        return fmt::format(
            "needs >= {} DRAM banks (have {}); run on a larger emulator or silicon", needed_dram_banks, dram_banks);
    }
    // resolve_grid() sizes shard_grid to whatever the device has, so the generic oversized-grid check
    // above never trips for a use_max_grid row; a genuine-2D row still needs an explicit floor.
    if (lc.require_2d_grid && lc.shard_grid.y <= 1) {
        return fmt::format(
            "needs a worker grid with > 1 row for a genuine 2D block split (have {}x{})", grid.x, grid.y);
    }
    return {};
}

m2::KernelSpec make_kernel(
    const std::string& name,
    const char* source,
    const std::string& dfb_accessor,
    m2::DFBEndpointType endpoint,
    const std::optional<std::string>& tensor_accessor,
    std::optional<IterMode> iter_mode,
    bool implicit_sync,
    bool disable_addrgen) {
    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{name},
        .source = std::filesystem::path{source},
        .num_threads = 1,
        .hw_config = m2::DataMovementHardwareConfig{},
    };
    // DFB implicit sync (async_read/async_write<TXN_ID>) is the default; only explicit-sync kernels opt out.
    if (!implicit_sync) {
        std::get<m2::DataMovementHardwareConfig>(kernel.hw_config).config_2xx =
            m2::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for = {m2::DFBSpecName{"staging"}},
            };
    }
    kernel.dfb_bindings = {
        {.dfb_spec_name = m2::DFBSpecName{"staging"},
         .accessor_name = dfb_accessor,
         .endpoint_type = endpoint,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    if (tensor_accessor) {
        m2::test_helpers::BindTensorParameterToKernel(kernel, *tensor_accessor, *tensor_accessor);
    }
    if (iter_mode) {
        kernel.compile_time_args = {
            {"iter_mode", static_cast<uint32_t>(*iter_mode)}, {"implicit_sync", implicit_sync ? 1u : 0u}};
    }
    kernel.runtime_arg_schema = {.runtime_arg_names = {"start_page", "num_pages"}};
    if (disable_addrgen) {
        kernel.compiler_options.defines.emplace(kDisableAddrgenDefine, "1");
    }
    return kernel;
}

void run_case(
    distributed::MeshDevice& device, const LayoutCase& lc, KernelShape shape, IterMode iter_mode, bool implicit_sync) {
    const m2::NodeCoord node{0, 0};
    const uint32_t page_size = page_size_bytes(lc);
    const uint32_t pages = num_pages(lc);
    // The golden below moves data as raw uint32 words regardless of the tensor's logical dtype (it
    // never interprets the bytes), so this only needs page_size to be word-aligned, not 4 bytes/elem.
    ASSERT_EQ(page_size % sizeof(uint32_t), 0u) << "page size must be uint32-word-aligned for the golden";
    const uint32_t words_per_page = page_size / sizeof(uint32_t);

    const TensorSpec tensor_spec = make_tensor_spec(lc);
    ASSERT_EQ(tensor_spec.compute_page_size_bytes(), page_size) << "page size assumption broken";

    MeshTensor out_tensor = MeshTensor::allocate_on_device(device, tensor_spec);
    std::optional<MeshTensor> in_tensor;
    if (shape != KernelShape::WriteOnly) {
        in_tensor = MeshTensor::allocate_on_device(device, tensor_spec);
    }

    // The fill kernel writes DFB entries with CPU stores, so it always syncs explicitly.
    m2::KernelSpec producer = shape == KernelShape::WriteOnly ? make_kernel(
                                                                    "producer",
                                                                    kFillKernel,
                                                                    "out",
                                                                    m2::DFBEndpointType::PRODUCER,
                                                                    std::nullopt,
                                                                    std::nullopt,
                                                                    /*implicit_sync=*/false,
                                                                    /*disable_addrgen=*/false)
                                                              : make_kernel(
                                                                    "producer",
                                                                    kReaderKernel,
                                                                    "out",
                                                                    m2::DFBEndpointType::PRODUCER,
                                                                    "src",
                                                                    iter_mode,
                                                                    implicit_sync,
                                                                    /*disable_addrgen=*/false);
    m2::KernelSpec consumer = make_kernel(
        "consumer",
        kWriterKernel,
        "in",
        m2::DFBEndpointType::CONSUMER,
        "dst",
        iter_mode,
        implicit_sync,
        /*disable_addrgen=*/shape == KernelShape::ReadOnly);

    auto dfb = m2::test_helpers::MakeMinimalDFB("staging", page_size, kNumDfbEntries);
    dfb.data_format_metadata = to_data_format(lc.dtype);

    m2::ProgramSpec spec{
        .name = "ta_addrgen_" + lc.name,
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"producer", "consumer"})},
    };
    spec.tensor_parameters.push_back({.unique_id = m2::TensorParamName{"dst"}, .spec = tensor_spec});
    if (in_tensor) {
        spec.tensor_parameters.push_back({.unique_id = m2::TensorParamName{"src"}, .spec = tensor_spec});
    }

    Program program = m2::MakeProgramFromSpec(device, spec);

    const auto rtas = m2::MakeRuntimeArgsForSingleNode(node, {{"start_page", 0u}, {"num_pages", pages}});
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"producer"}, .runtime_arg_values = rtas},
        {.kernel = m2::KernelSpecName{"consumer"}, .runtime_arg_values = rtas},
    };
    params.tensor_args = {{m2::TensorParamName{"dst"}, std::cref(out_tensor)}};
    if (in_tensor) {
        params.tensor_args.emplace(m2::TensorParamName{"src"}, std::cref(*in_tensor));
    }
    m2::SetProgramRunArgs(program, params);

    // Zero the destination so a skipped page cannot pass by leftover data.
    std::vector<uint32_t> zeros(pages * words_per_page, 0);
    slow_dispatch::WriteToBuffer(out_tensor.mesh_buffer(), zeros);

    std::vector<uint32_t> expected(pages * words_per_page);
    if (in_tensor) {
        for (uint32_t i = 0; i < expected.size(); ++i) {
            expected[i] = 0x5A000000u ^ (i * 2654435761u);  // unique, non-trivial per word
        }
        slow_dispatch::WriteToBuffer(in_tensor->mesh_buffer(), expected);
    } else {
        for (uint32_t p = 0; p < pages; ++p) {
            for (uint32_t w = 0; w < words_per_page; ++w) {
                expected[(p * words_per_page) + w] = fill_pattern_word(p, w);
            }
        }
    }

    // Quasar emulator: host writes are not ordered with launch (see m2_writeshard_barrier_uint32 in
    // dfb_test_common.hpp). Read back until the staged data is visible.
    std::vector<uint32_t> readback;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), readback);
    ASSERT_EQ(readback, zeros) << "destination zero-fill not visible before launch";
    if (in_tensor) {
        slow_dispatch::ReadFromBuffer(in_tensor->mesh_buffer(), readback);
        ASSERT_EQ(readback, expected) << "source staging not visible before launch";
    }

    LaunchProgram(device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    ASSERT_EQ(output.size(), expected.size());
    for (uint32_t p = 0; p < pages; ++p) {
        for (uint32_t w = 0; w < words_per_page; ++w) {
            const uint32_t idx = (p * words_per_page) + w;
            ASSERT_EQ(output[idx], expected[idx]) << "first mismatch at page " << p << " word " << w;
        }
    }
}

// Layout rows. Names are gtest-safe (alphanumeric + underscore).
std::vector<LayoutCase> layout_cases() {
    std::vector<LayoutCase> cases;

    // Interleaved: HW recipe = BANK_INNER over the ATT DRAM / worker selectors.
    cases.push_back({.name = "InterleavedDram", .buffer_type = BufferType::DRAM});
    cases.push_back({.name = "InterleavedL1", .buffer_type = BufferType::L1});
    cases.push_back({.name = "InterleavedDram4Banks", .buffer_type = BufferType::DRAM, .min_dram_banks = 4});

    // Single-bank shard (recipe 2a: no banking).
    cases.push_back(
        {.name = "SingleBankHeightL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 16,
         .shard_cols = 64,
         .shard_grid = {1, 1}});
    cases.push_back(
        {.name = "SingleBankHeightDram",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 16,
         .shard_cols = 64,
         .shard_grid = {1, 1}});

    // Multi-bank, outer-dim-only split (recipe 2b: BANK_MIDDLE, page-id order is bank-monotone).
    cases.push_back(
        {.name = "HeightL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "HeightL1ColMajor",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});
    cases.push_back(
        {.name = "HeightL1EdgeShard",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .rows = 12,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "HeightDram",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "HeightDramColMajor",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});

    // Multi-bank, inner-dim split (recipe 3/4: bank id wraps every row in page-id order; this is
    // the 2D banking workaround case).
    cases.push_back(
        {.name = "WidthL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "WidthDram",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "WidthL1ColMajor",  // orientation matters even on a 2x1 grid: see HeightL1ColMajor
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});
    cases.push_back(
        {.name = "WidthDramColMajor",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});

    // True 2D BLOCK and 2D-grid orientation: need >= 2x2 workers (skipped on emu-quasar-2x3).
    cases.push_back(
        {.name = "Block2x2L1RowMajor",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .rows = 16,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 2}});
    cases.push_back(
        {.name = "Block2x2L1ColMajor",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .rows = 16,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 2},
         .orientation = ShardOrientation::COL_MAJOR});
    cases.push_back(
        {.name = "Height2x2L1ColMajor",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .rows = 32,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 2},
         .orientation = ShardOrientation::COL_MAJOR});

    // ND sharding: rank-3 outer split, and round-robin shard placement (multi-shard-per-bank).
    cases.push_back(
        {.name = "NdRank3L1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,  // ND spec; HEIGHT only selects page = full row
         .rows = 8,
         .cols = 64,
         .nd_outer = 4,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "NdRoundRobinL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .rows = 32,
         .shard_rows = 4,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .nd_round_robin = true});

    // TILE layout: a parallel axis over the same recipe families, but a page is always exactly one
    // 32x32 tile (get_page_shape_tile ignores shard shape), unlike ROW_MAJOR where WIDTH/BLOCK bundle
    // a whole shard-row segment into one page. Shapes here are tile-aligned (multiples of kTileDim).
    cases.push_back(
        {.name = "InterleavedDramTile",
         .buffer_type = BufferType::DRAM,
         .layout = Layout::TILE,
         .rows = 128,
         .cols = 64});
    cases.push_back(
        {.name = "InterleavedL1Tile", .buffer_type = BufferType::L1, .layout = Layout::TILE, .rows = 128, .cols = 64});

    // Single-bank shard (recipe 2a), tile-aligned.
    cases.push_back(
        {.name = "SingleBankHeightL1Tile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 64,
         .shard_cols = 64,
         .shard_grid = {1, 1}});

    // Multi-bank outer-dim split (recipe 2b: BANK_MIDDLE), tile-aligned.
    cases.push_back(
        {.name = "HeightL1Tile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 32,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "HeightL1ColMajorTile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 32,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});

    // Multi-bank inner-dim split (recipe 3/4: the 2D banking workaround), tile-aligned. Each shard is
    // exactly 1 tile wide so the bank sequence alternates every tile in page-id order: page-id order is
    // B0,B1,B0,B1 across the 2 tile-rows, but storage order (all of bank0's tiles, then bank1's) is
    // B0,B0,B1,B1 -- the same divergence WidthL1/WidthDram exercise under ROW_MAJOR, one tile at a time
    // instead of one row at a time.
    cases.push_back(
        {.name = "WidthL1Tile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 64,
         .shard_cols = 32,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "WidthL1ColMajorTile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 64,
         .shard_cols = 32,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});

    // DRAM + Tile, parity with the ROW_MAJOR Height/WidthDram(ColMajor) rows above.
    cases.push_back(
        {.name = "HeightDramTile",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 32,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "HeightDramColMajorTile",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 32,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});
    cases.push_back(
        {.name = "WidthDramTile",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 64,
         .shard_cols = 32,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "WidthDramColMajorTile",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 64,
         .shard_rows = 64,
         .shard_cols = 32,
         .shard_grid = {2, 1},
         .orientation = ShardOrientation::COL_MAJOR});

    // Max-grid rows: resolve_grid() rescales these to every worker core / DRAM bank the device actually
    // has (see the LayoutCase comment), so they automatically exercise the full BANK_END wrap-around and
    // a deeper 2D-banking-workaround stress as bigger emulators (5x4, 9x4, ...) or silicon become
    // available, with no changes here. shard_rows/shard_cols below are the fixed per-bank granule; rows
    // /cols are only the 1-bank fallback used if resolve_grid() is bypassed.
    cases.push_back(
        {.name = "HeightL1MaxGrid",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .rows = 8,
         .cols = 64,
         .shard_rows = 8,
         .shard_cols = 64,
         .use_max_grid = true});
    cases.push_back(
        {.name = "HeightDramMaxGrid",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .rows = 8,
         .cols = 64,
         .shard_rows = 8,
         .shard_cols = 64,
         .use_max_grid = true});
    cases.push_back(
        {.name = "WidthL1MaxGrid",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 64,
         .shard_rows = 8,
         .shard_cols = 64,
         .use_max_grid = true});
    cases.push_back(
        {.name = "WidthDramMaxGrid",
         .buffer_type = BufferType::DRAM,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .rows = 8,
         .cols = 64,
         .shard_rows = 8,
         .shard_cols = 64,
         .use_max_grid = true});
    cases.push_back(
        {.name = "BlockL1MaxGrid",  // genuine 2D: skips unless the device grid has > 1 row
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .rows = 8,
         .cols = 64,
         .shard_rows = 8,
         .shard_cols = 64,
         .use_max_grid = true,
         .require_2d_grid = true});
    cases.push_back(
        {.name = "HeightL1MaxGridTile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .layout = Layout::TILE,
         .rows = 32,
         .cols = 64,
         .shard_rows = 32,
         .shard_cols = 64,
         .use_max_grid = true});
    cases.push_back(
        {.name = "WidthL1MaxGridTile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .layout = Layout::TILE,
         .rows = 64,
         .cols = 32,
         .shard_rows = 64,
         .shard_cols = 32,
         .use_max_grid = true});
    cases.push_back(
        {.name = "BlockL1MaxGridTile",  // genuine 2D: skips unless the device grid has > 1 row
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .layout = Layout::TILE,
         .rows = 32,
         .cols = 32,
         .shard_rows = 32,
         .shard_cols = 32,
         .use_max_grid = true,
         .require_2d_grid = true});

    // dtype diversity: every row above is UINT32 (4-byte elements, always word-aligned page sizes).
    // These reuse existing recipe families with a smaller element size, stressing page-size math that
    // a 4-byte-only suite can't: a 1- or 2-byte element still yields correct addrgen page/bank strides.
    cases.push_back({.name = "InterleavedDramUInt16", .buffer_type = BufferType::DRAM, .dtype = DataType::UINT16});
    cases.push_back({.name = "InterleavedDramUInt8", .buffer_type = BufferType::DRAM, .dtype = DataType::UINT8});
    cases.push_back(
        {.name = "HeightL1UInt16",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
         .dtype = DataType::UINT16,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "WidthL1UInt8",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
         .dtype = DataType::UINT8,
         .rows = 8,
         .cols = 128,
         .shard_rows = 8,
         .shard_cols = 64,
         .shard_grid = {2, 1}});
    cases.push_back(
        {.name = "InterleavedL1TileUInt16",
         .buffer_type = BufferType::L1,
         .layout = Layout::TILE,
         .dtype = DataType::UINT16,
         .rows = 64,
         .cols = 64});

    // Tile padding: rows/cols are not multiples of kTileDim, so the physical/padded tensor has a final
    // tile-row and tile-column that are only partially valid (real tile alignment always rounds a
    // logical dim up to the tile size -- see create_default_alignment_tile). num_pages()'s ceil_div
    // must count that trailing padded tile, not silently drop it.
    cases.push_back(
        {.name = "InterleavedDramTileRagged",
         .buffer_type = BufferType::DRAM,
         .layout = Layout::TILE,
         .rows = 100,
         .cols = 48});

    return cases;
}

struct MatrixParam {
    LayoutCase layout;
    KernelShape shape;
    IterMode iter_mode;
    bool implicit_sync = true;
};

std::string shape_name(KernelShape s) {
    switch (s) {
        case KernelShape::ReadOnly: return "ReadOnly";
        case KernelShape::WriteOnly: return "WriteOnly";
        case KernelShape::Copy: return "Copy";
    }
    return "Unknown";
}

void PrintTo(const MatrixParam& p, std::ostream* os) {
    *os << p.layout.name << "/" << shape_name(p.shape) << "/"
        << (p.iter_mode == IterMode::PageIdLoop ? "PageId" : "Pages") << (p.implicit_sync ? "" : "/Explicit");
}

std::vector<MatrixParam> matrix() {
    std::vector<MatrixParam> params;
    for (const auto& lc : layout_cases()) {
        for (auto shape : {KernelShape::ReadOnly, KernelShape::WriteOnly, KernelShape::Copy}) {
            for (auto mode : {IterMode::PageIdLoop, IterMode::PagesIterator}) {
                params.push_back({lc, shape, mode});
            }
            // Explicit-sync smoke rows: keep the generic (non-TXN_ID) Noc path covered on one layout per
            // HW recipe family, since Phase B hooks both paths.
            if (lc.name == "InterleavedDram" || lc.name == "HeightL1" || lc.name == "WidthL1") {
                params.push_back({lc, shape, IterMode::PageIdLoop, /*implicit_sync=*/false});
            }
        }
    }
    return params;
}

// Randomized layer. The curated matrix() above is a fixed, hand-picked set of recipe families; this
// generates additional cases by randomly combining the same axes (memory_layout, Layout, buffer_type,
// dtype, orientation, shard geometry, raggedness) that a hand-written list can only sample sparsely.
// Deterministic by default (fixed seed) so a failure is reproducible; TT_TA_ADDRGEN_FUZZ_SEED /
// TT_TA_ADDRGEN_FUZZ_COUNT env vars override the seed / case count for exploratory fuzzing.
LayoutCase random_layout_case(std::mt19937& rng, uint32_t index) {
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_int_distribution<int> die3(0, 2);
    std::uniform_int_distribution<int> die4(0, 3);
    std::uniform_int_distribution<int> grid_dim(1, 3);   // shard_grid.x / .y
    std::uniform_int_distribution<int> tile_mult(1, 3);  // granule = kTileDim * {1,2,3}
    std::uniform_int_distribution<int> rm_granule_idx(0, 3);
    std::uniform_int_distribution<int> unsharded_dim(1, 4);       // *32, for INTERLEAVED rows/cols
    std::uniform_int_distribution<int> ragged_deficit_pct(0, 3);  // 0 = exact, 1..3 = shrink last shard

    static constexpr TensorMemoryLayout kMemLayouts[] = {
        TensorMemoryLayout::INTERLEAVED,
        TensorMemoryLayout::HEIGHT_SHARDED,
        TensorMemoryLayout::WIDTH_SHARDED,
        TensorMemoryLayout::BLOCK_SHARDED,
    };
    static constexpr DataType kDtypes[] = {DataType::UINT32, DataType::UINT16, DataType::UINT8};
    static constexpr uint32_t kRmGranules[] = {4, 8, 16, 32};

    LayoutCase lc;
    lc.buffer_type = coin(rng) ? BufferType::L1 : BufferType::DRAM;
    lc.memory_layout = kMemLayouts[die4(rng)];
    lc.layout = coin(rng) ? Layout::TILE : Layout::ROW_MAJOR;
    lc.dtype = kDtypes[die3(rng)];
    lc.orientation = coin(rng) ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    const uint32_t granule =
        lc.layout == Layout::TILE ? kTileDim * static_cast<uint32_t>(tile_mult(rng)) : kRmGranules[rm_granule_idx(rng)];

    if (lc.memory_layout == TensorMemoryLayout::INTERLEAVED) {
        // No shard geometry; rows/cols alone drive page count. Occasionally not a multiple of the tile
        // size when Layout::TILE, which exercises the ceil_div tile-padding path for free.
        const uint32_t unit = lc.layout == Layout::TILE ? kTileDim : 4;
        lc.rows = unit * static_cast<uint32_t>(unsharded_dim(rng));
        lc.cols = unit * static_cast<uint32_t>(unsharded_dim(rng));
        if (lc.layout == Layout::TILE && coin(rng)) {
            lc.rows += kTileDim / 2;  // ragged: not a multiple of kTileDim
        }
    } else {
        lc.shard_rows = granule;
        lc.shard_cols = granule;
        lc.shard_grid = {static_cast<uint32_t>(grid_dim(rng)), static_cast<uint32_t>(grid_dim(rng))};
        // BLOCK_SHARDED must not degenerate to a single grid dimension of 1 in a way that makes it
        // identical to HEIGHT/WIDTH -- but a degenerate case is still a *valid* one to fuzz, so no
        // adjustment here; skip_reason()/resolve_grid() are not involved, this is a fixed small grid.
        switch (lc.memory_layout) {
            case TensorMemoryLayout::HEIGHT_SHARDED: {
                const uint32_t num_shards = lc.shard_grid.x * lc.shard_grid.y;
                lc.cols = granule;
                lc.rows = num_shards * lc.shard_rows;
                const int deficit_pct = ragged_deficit_pct(rng);
                if (deficit_pct > 0 && lc.shard_rows > 1) {
                    // Shrink only the logical row count (ragged last shard), matching HeightL1EdgeShard;
                    // never below 1 valid row in that shard.
                    const uint32_t deficit =
                        std::min(lc.shard_rows - 1, lc.shard_rows * static_cast<uint32_t>(deficit_pct) / 4);
                    lc.rows -= deficit;
                }
                // Round-robin (multi-shard-per-bank) fuzzing, HEIGHT only -- see random_layout_case's
                // caller comment on why this axis is restricted to HEIGHT_SHARDED.
                lc.nd_round_robin = (die4(rng) == 0);
                break;
            }
            case TensorMemoryLayout::WIDTH_SHARDED: {
                const uint32_t num_shards = lc.shard_grid.x * lc.shard_grid.y;
                lc.rows = granule;
                lc.cols = num_shards * lc.shard_cols;
                break;
            }
            case TensorMemoryLayout::BLOCK_SHARDED:
                // Same height/width <-> grid.y/grid.x swap for COL_MAJOR as resolve_grid() above --
                // see get_shape_fits_shard_grid_error in tensor_spec.cpp.
                if (lc.orientation == ShardOrientation::ROW_MAJOR) {
                    lc.rows = lc.shard_grid.y * lc.shard_rows;
                    lc.cols = lc.shard_grid.x * lc.shard_cols;
                } else {
                    lc.rows = lc.shard_grid.x * lc.shard_rows;
                    lc.cols = lc.shard_grid.y * lc.shard_cols;
                }
                break;
            default: break;
        }
    }

    lc.name = fmt::format(
        "Fuzz{:03d}_{}_{}_{}_{}",
        index,
        lc.memory_layout == TensorMemoryLayout::INTERLEAVED      ? "Interleaved"
        : lc.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? "Height"
        : lc.memory_layout == TensorMemoryLayout::WIDTH_SHARDED  ? "Width"
                                                                 : "Block",
        lc.buffer_type == BufferType::DRAM ? "Dram" : "L1",
        lc.layout == Layout::TILE ? "Tile" : "Rm",
        lc.dtype == DataType::UINT32   ? "U32"
        : lc.dtype == DataType::UINT16 ? "U16"
                                       : "U8");
    return lc;
}

std::vector<MatrixParam> fuzz_cases() {
    const char* seed_env = std::getenv("TT_TA_ADDRGEN_FUZZ_SEED");
    const char* count_env = std::getenv("TT_TA_ADDRGEN_FUZZ_COUNT");
    const uint32_t seed = seed_env ? static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 0)) : 0xA55AF00Du;
    const uint32_t count = count_env ? static_cast<uint32_t>(std::strtoul(count_env, nullptr, 0)) : 40u;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_int_distribution<int> shape_die(0, 2);
    static constexpr KernelShape kShapes[] = {KernelShape::ReadOnly, KernelShape::WriteOnly, KernelShape::Copy};

    std::vector<MatrixParam> params;
    params.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        params.push_back(
            {random_layout_case(rng, i),
             kShapes[shape_die(rng)],
             coin(rng) ? IterMode::PageIdLoop : IterMode::PagesIterator,
             /*implicit_sync=*/true});
    }
    return params;
}

}  // namespace unit_tests::dm::ta_addrgen

class TensorAccessorAddrgen : public QuasarMeshDeviceSingleCardFixture,
                              public ::testing::WithParamInterface<unit_tests::dm::ta_addrgen::MatrixParam> {};

TEST_P(TensorAccessorAddrgen, TensorDfbRoundTrip) {
    using namespace unit_tests::dm::ta_addrgen;
    const auto& p = GetParam();
    auto& device = *devices_.at(0);
    const LayoutCase lc = resolve_grid(device, p.layout);
    if (const auto reason = skip_reason(device, lc); !reason.empty()) {
        GTEST_SKIP() << lc.name << ": " << reason;
    }
    run_case(device, lc, p.shape, p.iter_mode, p.implicit_sync);
}

INSTANTIATE_TEST_SUITE_P(
    PatternMatrix,
    TensorAccessorAddrgen,
    ::testing::ValuesIn(unit_tests::dm::ta_addrgen::matrix()),
    [](const ::testing::TestParamInfo<unit_tests::dm::ta_addrgen::MatrixParam>& info) {
        using namespace unit_tests::dm::ta_addrgen;
        return info.param.layout.name + "_" + shape_name(info.param.shape) + "_" +
               (info.param.iter_mode == IterMode::PageIdLoop ? "PageId" : "Pages") +
               (info.param.implicit_sync ? "" : "_Explicit");
    });

// Randomized fuzz layer: same fixture/body as PatternMatrix, params drawn by fuzz_cases() instead of
// the curated matrix(). Deterministic by default; see fuzz_cases() for the seed/count env overrides.
INSTANTIATE_TEST_SUITE_P(
    Fuzz,
    TensorAccessorAddrgen,
    ::testing::ValuesIn(unit_tests::dm::ta_addrgen::fuzz_cases()),
    [](const ::testing::TestParamInfo<unit_tests::dm::ta_addrgen::MatrixParam>& info) {
        using namespace unit_tests::dm::ta_addrgen;
        return info.param.layout.name + "_" + shape_name(info.param.shape) + "_" +
               (info.param.iter_mode == IterMode::PageIdLoop ? "PageId" : "Pages");
    });

}  // namespace tt::tt_metal
