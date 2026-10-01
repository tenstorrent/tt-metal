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
// Every NoC transfer address comes from tensor_accessor::transfer_noc_addr (via noc_traits_t), which uses the HW
// AddrGen for layouts with a recipe on ATT builds and software otherwise. Tensor-side kernels report how each
// address was produced (TT_TA_ADDRGEN_STATS) and run_case checks that against the layout, so a row can't silently
// pass on the software path. Without TT_METAL_NOC_ATT every row is software and the rows are the golden.
//
// Sized for emu-quasar-2x3 (2 worker Tensix on one row). Rows that need a larger grid or more DRAM
// banks stay in the suite and GTEST_SKIP with the requirement, so they run unchanged on a larger
// emulator or silicon. Run with ATT enabled:
//   TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_NOC_ATT=quasar_aether_2x3 TT_METAL_ATT_PROGRAM_FOR_TEST=1

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"
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

// Per-kernel opt-out from the HW AddrGen path (api/tensor/transfer_noc_addr.h): the kernel's NoC transfers use the
// software TensorAccessor addresses.
constexpr auto kDisableAddrgenDefine = "TT_TA_ADDRGEN_DISABLE";
// Test instrumentation: each tensor-side kernel counts how its transfer addresses were produced and writes
// {hw, sw_ineligible, sw_unsupported, seeks, transfers issued, skips, restores, write seeks, write restores} to its
// report_addr RTA. See TransferStats in transfer_noc_addr.h.
constexpr auto kAddrgenStatsDefine = "TT_TA_ADDRGEN_STATS";
constexpr uint32_t kNumStatsWords = 10;  // word 9: stack bytes the kernel never used (ta_multi/mixed/reader)
constexpr uint32_t kStatsStride = 64;    // bytes reserved per kernel's report
static_assert(kNumStatsWords * sizeof(uint32_t) <= kStatsStride);

constexpr uint32_t kNumDfbEntries = 4;

enum class KernelShape { ReadOnly, WriteOnly, Copy };
// Must match iter_mode in ta_reader_to_dfb.cpp / ta_writer_from_dfb.cpp.
// ShardView / ShardPages walk shard by shard (not page-id order), so both sides of a row must use them together:
// Copy rows only, sharded layouts only. ShardView moves every page slot of every shard, padding included.
// Strided: page ids 0, 2, 4, ... then 1, 3, 5, ... (a steady stride the hardware skips along, and one jump back).
enum class IterMode : uint32_t {
    PageIdLoop = 0,
    PagesIterator = 1,
    PageView = 2,
    Wrapper = 3,
    ShardView = 4,
    ShardPages = 5,
    Strided = 6,
};

std::string iter_mode_name(IterMode m) {
    switch (m) {
        case IterMode::PageIdLoop: return "PageId";
        case IterMode::PagesIterator: return "Pages";
        case IterMode::PageView: return "PageView";
        case IterMode::Wrapper: return "Wrapper";
        case IterMode::ShardView: return "ShardView";
        case IterMode::ShardPages: return "ShardPages";
        case IterMode::Strided: return "Strided";
    }
    return "Unknown";
}

// The HW AddrGen transfer path is compiled in only for ATT builds (qa_hal.cpp defines NOC_ATT_ENABLED iff
// TT_METAL_NOC_ATT is set); without it every transfer address is software.
bool att_enabled() { return std::getenv("TT_METAL_NOC_ATT") != nullptr; }

// Bring-up check, on the host: can one BankingConfig walk this device's interleaved `type` banks in page-id order?
// Needs bank i's ATT selector == bank 0's + i and no per-bank offset (see interleaved_walkable in
// tensor_accessor_addrgen.h). The device assumes yes; kernels for a device where it's no are built with
// TT_TA_ADDRGEN_INTERLEAVED_{DRAM,L1}_SW (see make_kernel). Only the map this suite targets is known here.
bool interleaved_banks_walkable(distributed::MeshDevice& device, BufferType type) {
    const char* map_name = std::getenv("TT_METAL_NOC_ATT");
    if (map_name == nullptr || std::string_view(map_name) != "quasar_aether_2x3") {
        return false;
    }
    const noc_att::MapData& map = quasar_aether_2x3_att_config::MAP;
    const auto& allocator = *device.allocator();
    std::optional<uint32_t> first;
    for (uint32_t bank = 0; bank < allocator.get_num_banks(type); ++bank) {
        uint32_t selector = 0;
        if (type == BufferType::DRAM) {
            if (bank >= map.dram_selectors.size()) {
                return false;
            }
            selector = map.dram_selectors[bank];
        } else {
            const CoreCoord core = device.worker_core_from_logical_core(allocator.get_logical_core_from_bank_id(bank));
            const noc_att::ResolvedTile tile = noc_att::resolve(map, noc_att::Address::worker(core.x, core.y, 0));
            if (!tile.valid) {
                return false;
            }
            selector = tile.selector;
        }
        if (!first) {
            first = selector;
        }
        if (selector != *first + bank || allocator.get_bank_offset(type, bank) != 0) {
            return false;
        }
    }
    return true;
}

// Defines that route interleaved transfers to software on a device whose banks the recipe can't walk.
std::map<std::string, std::string> addrgen_bringup_defines(distributed::MeshDevice& device) {
    std::map<std::string, std::string> defines;
    if (!interleaved_banks_walkable(device, BufferType::DRAM)) {
        defines.emplace("TT_TA_ADDRGEN_INTERLEAVED_DRAM_SW", "1");
    }
    if (!interleaved_banks_walkable(device, BufferType::L1)) {
        defines.emplace("TT_TA_ADDRGEN_INTERLEAVED_L1_SW", "1");
    }
    return defines;
}

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
    if (tensor_accessor) {
        kernel.runtime_arg_schema.runtime_arg_names.push_back("report_addr");
        kernel.compiler_options.defines.emplace(kAddrgenStatsDefine, "1");
    }
    if (disable_addrgen) {
        kernel.compiler_options.defines.emplace(kDisableAddrgenDefine, "1");
    }
    return kernel;
}

// Layouts with a hardware AddrGen recipe (tt_addrgen::has_hw_recipe in tensor_accessor_addrgen.h): every
// TensorAccessor layout -- interleaved, and sharded of any rank / distribution / buffer type.
bool has_hw_recipe(const LayoutCase& /*lc*/) { return true; }

// Single-page L1 buffer. Interleaved allocation reserves the same range in every L1 bank, so its address is
// usable on any worker core, including the program's node (0,0).
std::shared_ptr<distributed::MeshBuffer> make_l1_region(distributed::MeshDevice& device, uint32_t size_bytes) {
    distributed::DeviceLocalBufferConfig local{.page_size = size_bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig replicated{.size = size_bytes};
    return distributed::MeshBuffer::create(replicated, local, &device);
}

// Checks one tensor-side kernel's {hw, sw_ineligible, sw_unsupported, seeks, transfers, skips, restores} report.
void expect_transfer_stats(
    const std::string& kernel,
    const std::vector<uint32_t>& stats,
    const LayoutCase& lc,
    uint32_t pages,
    IterMode iter_mode,
    bool addrgen_allowed,
    bool interleaved_walkable) {
    ASSERT_EQ(stats.size(), static_cast<size_t>(kNumStatsWords));
    ASSERT_NE(stats[0], 0xDEADBEEFu) << kernel << " never wrote its transfer stats";
    const uint32_t hw = stats[0];
    const uint32_t ineligible = stats[1];
    const uint32_t unsupported = stats[2];
    const uint32_t seeks = stats[3];
    const uint32_t transfers = stats[4];
    const uint32_t skips = stats[5];
    const uint32_t restores = stats[6];
    if (stats[9] != 0xDEADBEEFu) {
        log_info(tt::LogTest, "{}: {} bytes of stack never used", kernel, stats[9]);
    }
    const std::string counts = fmt::format(
        "{}: hw {} ({} seeks, {} skips, {} restores), sw_ineligible {}, sw_unsupported {} of {} transfers",
        kernel,
        hw,
        seeks,
        skips,
        restores,
        ineligible,
        unsupported,
        transfers);
    log_info(tt::LogTest, "{}", counts);
    // ShardView also moves the padding slots of edge shards; every other mode moves each logical page once.
    if (iter_mode == IterMode::ShardView) {
        EXPECT_GE(transfers, pages) << counts;
    } else {
        EXPECT_EQ(transfers, pages) << counts;
    }
    pages = transfers;
    ASSERT_EQ(hw + ineligible + unsupported, pages) << counts << " -- some transfer bypassed transfer_noc_addr";
    if (!addrgen_allowed || !att_enabled() || !has_hw_recipe(lc)) {
        EXPECT_EQ(unsupported, pages) << counts << " -- expected the software path only";
        return;
    }
    EXPECT_EQ(unsupported, 0u) << counts << " -- layout has a HW recipe but the transfer path didn't try it";
    // Walkability is a property of the device, so a kernel is all-HW or all-fallback, never mixed.
    EXPECT_TRUE(hw == pages || ineligible == pages) << counts;
    EXPECT_LE(seeks, hw) << counts;
    EXPECT_LE(skips, hw) << counts;
    if (hw > 0) {
        EXPECT_GE(seeks, 1u) << counts << " -- a walk must be programmed before its first pop";
    }
    // Only the interleaved recipe depends on the device's bank tables (host-checked, see interleaved_banks_walkable);
    // sharded layouts always have a recipe.
    if (lc.memory_layout == TensorMemoryLayout::INTERLEAVED && !interleaved_walkable) {
        EXPECT_EQ(ineligible, pages) << counts << " -- banks not walkable: expected the software fallback";
    } else {
        EXPECT_EQ(hw, pages) << counts << " -- expected every transfer address from the HW AddrGen";
    }
}

// Op-to-op R/W inference: what a kernel's ELF says it reads and writes (Paul's .tt.BUF_RW notes, emitted by the NoC
// APIs' endpoint address helpers). Slots are per-binding ids, so only the set structure is compared.
ll_api::BufRwInfo query_buf_rw(
    distributed::MeshWorkload& workload, distributed::MeshDevice& device, const std::string& kernel) {
    auto& programs = workload.get_programs();
    EXPECT_EQ(programs.size(), 1u);
    auto k = programs.begin()->second.impl().get_kernel_by_spec_name(kernel);
    EXPECT_NE(k, nullptr) << "no kernel " << kernel;
    return k ? k->query_buf_rw(*device.get_devices()[0]) : ll_api::BufRwInfo{};
}

// Which bound tensors a kernel touches, as the notes must record them.
enum class TensorRole { None, Reads, Writes };

void expect_buf_rw(
    const std::string& kernel, const ll_api::BufRwInfo& rw, TensorRole role, size_t num_tensors, IterMode mode) {
    EXPECT_FALSE(rw.opaque) << kernel << ": only Noc APIs on bound tensors -- analyzable";
    const std::set<uint32_t>& touched = role == TensorRole::Writes ? rw.writes : rw.reads;
    const std::set<uint32_t>& other = role == TensorRole::Writes ? rw.reads : rw.writes;
    if (role == TensorRole::None) {
        EXPECT_TRUE(rw.reads.empty() && rw.writes.empty()) << kernel << ": touches no bound tensor";
        return;
    }
    // Distinct slots per binding: this is what tells three tensors of one kernel apart.
    EXPECT_EQ(touched.size(), num_tensors) << kernel << ": one record per bound tensor";
    if (mode == IterMode::Wrapper) {
        // The type-erasing wrapper notes its binding as both read and written (see AbstractTensorAccessorWrapper).
        EXPECT_EQ(other, touched) << kernel << ": wrapper over-approximates its binding as read + written";
    } else {
        EXPECT_TRUE(other.empty()) << kernel << ": the DFB side is not a bound tensor";
    }
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

    // Bring-up: interleaved banks this device can't walk go to software (host-checked; the device assumes walkable).
    const auto bringup_defines = addrgen_bringup_defines(device);
    for (m2::KernelSpec* kernel : {&producer, &consumer}) {
        if (kernel->compiler_options.defines.contains(kAddrgenStatsDefine)) {
            for (const auto& [name, value] : bringup_defines) {
                kernel->compiler_options.defines.emplace(name, value);
            }
        }
    }
    const bool walkable = interleaved_banks_walkable(device, lc.buffer_type);

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

    // Transfer-stats reports: producer's words at +0, consumer's at +kStatsStride.
    auto stats_region = make_l1_region(device, 2 * kStatsStride);
    const uint32_t producer_report = static_cast<uint32_t>(stats_region->address());
    const uint32_t consumer_report = producer_report + kStatsStride;
    std::vector<uint32_t> stats_init(2 * kStatsStride / sizeof(uint32_t), 0xDEADBEEF);  // "never written"
    slow_dispatch::WriteToL1(device, node, producer_report, stats_init);

    auto rtas_for = [&](bool with_report, uint32_t report_addr) {
        return with_report ? m2::MakeRuntimeArgsForSingleNode(
                                 node, {{"start_page", 0u}, {"num_pages", pages}, {"report_addr", report_addr}})
                           : m2::MakeRuntimeArgsForSingleNode(node, {{"start_page", 0u}, {"num_pages", pages}});
    };
    const bool producer_is_ta = shape != KernelShape::WriteOnly;
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"producer"}, .runtime_arg_values = rtas_for(producer_is_ta, producer_report)},
        {.kernel = m2::KernelSpecName{"consumer"}, .runtime_arg_values = rtas_for(true, consumer_report)},
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

    auto workload = LaunchProgram(device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    ASSERT_EQ(output.size(), expected.size());
    for (uint32_t p = 0; p < pages; ++p) {
        for (uint32_t w = 0; w < words_per_page; ++w) {
            const uint32_t idx = (p * words_per_page) + w;
            ASSERT_EQ(output[idx], expected[idx]) << "first mismatch at page " << p << " word " << w;
        }
    }

    std::vector<uint32_t> stats;
    if (producer_is_ta) {
        slow_dispatch::ReadFromL1(device, node, producer_report, kNumStatsWords * sizeof(uint32_t), stats);
        expect_transfer_stats("reader", stats, lc, pages, iter_mode, /*addrgen_allowed=*/true, walkable);
    }
    slow_dispatch::ReadFromL1(device, node, consumer_report, kNumStatsWords * sizeof(uint32_t), stats);
    expect_transfer_stats(
        "writer", stats, lc, pages, iter_mode, /*addrgen_allowed=*/shape != KernelShape::ReadOnly, walkable);

    expect_buf_rw(
        "producer",
        query_buf_rw(workload, device, "producer"),
        producer_is_ta ? TensorRole::Reads : TensorRole::None,
        1,
        iter_mode);
    expect_buf_rw("consumer", query_buf_rw(workload, device, "consumer"), TensorRole::Writes, 1, iter_mode);
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

    // Blocked (split in both dims), round-robin over 2 cores: runs on emu-quasar-2x3, unlike the true 2D-grid BLOCK
    // rows below. Page-id order is not monotonic in bank address: each tensor row visits shard 0 (bank 0) then shard 1
    // (bank 1); after a shard's rows, the next band's shards are the second slot of the same banks. The walk must be
    // reprogrammed at each band (2 seeks for the tensor).
    cases.push_back(
        {.name = "NdBlockRoundRobinL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,  // ND spec; BLOCK selects page = shard-row segment
         .rows = 8,
         .cols = 128,
         .shard_rows = 4,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .nd_round_robin = true});
    // Same, but 3 shards per band on 2 banks: a band's shards land on banks 0, 1, 0 (the third one slot deeper), which
    // no single banking loop can walk, so each shard-row segment is its own software seek. Correct, not fast.
    cases.push_back(
        {.name = "NdBlockRoundRobin3WideL1",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .rows = 8,
         .cols = 192,
         .shard_rows = 4,
         .shard_cols = 64,
         .shard_grid = {2, 1},
         .nd_round_robin = true});
    cases.push_back(
        {.name = "NdBlockRoundRobinL1Tile",
         .buffer_type = BufferType::L1,
         .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
         .layout = Layout::TILE,
         .rows = 128,
         .cols = 128,
         .shard_rows = 64,
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
    *os << p.layout.name << "/" << shape_name(p.shape) << "/" << iter_mode_name(p.iter_mode)
        << (p.implicit_sync ? "" : "/Explicit");
}

std::vector<MatrixParam> matrix() {
    std::vector<MatrixParam> params;
    for (const auto& lc : layout_cases()) {
        for (auto shape : {KernelShape::ReadOnly, KernelShape::WriteOnly, KernelShape::Copy}) {
            for (auto mode : {IterMode::PageIdLoop, IterMode::PagesIterator}) {
                params.push_back({lc, shape, mode});
            }
            // PageView / AbstractTensorAccessorWrapper route through their own noc_traits_t specializations;
            // one Copy row per layout covers both sides of each.
            if (shape == KernelShape::Copy) {
                params.push_back({lc, shape, IterMode::PageView});
                params.push_back({lc, shape, IterMode::Wrapper});
                params.push_back({lc, shape, IterMode::Strided});
                if (is_sharded(lc)) {
                    params.push_back({lc, shape, IterMode::ShardView});
                    params.push_back({lc, shape, IterMode::ShardPages});
                }
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

// matrix() rows with interleaved L1 tensors, every iteration mode, for the identity-remap fixture.
std::vector<MatrixParam> interleaved_l1_matrix() {
    std::vector<MatrixParam> params;
    for (const auto& lc : layout_cases()) {
        if (lc.memory_layout != TensorMemoryLayout::INTERLEAVED || lc.buffer_type != BufferType::L1) {
            continue;
        }
        for (auto shape : {KernelShape::ReadOnly, KernelShape::WriteOnly, KernelShape::Copy}) {
            for (auto mode : {IterMode::PageIdLoop, IterMode::PagesIterator, IterMode::PageView, IterMode::Wrapper}) {
                params.push_back({lc, shape, mode});
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

// Opens the device with an identity l1_bank_remap. By default the L1 banking allocator shuffles bank ids across
// cores (fixed-seed std::shuffle in l1_banking_allocator.cpp), so L1 bank i -> ATT worker selector is a
// permutation that a single BankingConfig can't walk and interleaved L1 falls back to software. With the identity
// remap L1 bank i is the i-th compute core in row-major order, which is the ATT worker selectors' order, so the
// interleaved recipe applies to L1 too.
class IdentityL1RemapFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override {
        const ChipId id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().resolve_dispatch_core_config();
        // The remap must list exactly one entry per L1 bank; ask a default-opened device how many there are.
        uint32_t num_l1_banks = 0;
        {
            auto probe = distributed::MeshDevice::create_unit_meshes(
                {id}, l1_small_size_, trace_region_size_, num_command_queues(), dispatch_core_config);
            num_l1_banks = probe.at(id)->allocator()->get_num_banks(BufferType::L1);
        }
        std::vector<uint32_t> identity(num_l1_banks);
        std::iota(identity.begin(), identity.end(), 0u);
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            {id}, l1_small_size_, trace_region_size_, num_command_queues(), dispatch_core_config, identity);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
    }
};

class TensorAccessorAddrgen : public QuasarMeshDeviceSingleCardFixture,
                              public ::testing::WithParamInterface<unit_tests::dm::ta_addrgen::MatrixParam> {};

// Same rows on an identity-remap device, where interleaved L1 must take the HW path too.
class TensorAccessorAddrgenIdentityL1 : public IdentityL1RemapFixture,
                                        public ::testing::WithParamInterface<unit_tests::dm::ta_addrgen::MatrixParam> {
};

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
               iter_mode_name(info.param.iter_mode) + (info.param.implicit_sync ? "" : "_Explicit");
    });

TEST_P(TensorAccessorAddrgenIdentityL1, TensorDfbRoundTrip) {
    using namespace unit_tests::dm::ta_addrgen;
    const auto& p = GetParam();
    auto& device = *devices_.at(0);
    const LayoutCase lc = resolve_grid(device, p.layout);
    if (const auto reason = skip_reason(device, lc); !reason.empty()) {
        GTEST_SKIP() << lc.name << ": " << reason;
    }
    ASSERT_TRUE(interleaved_banks_walkable(device, BufferType::L1))
        << "with an identity l1_bank_remap the L1 banks should be an ascending stride-1 selector run";
    run_case(device, lc, p.shape, p.iter_mode, p.implicit_sync);
}

INSTANTIATE_TEST_SUITE_P(
    PatternMatrix,
    TensorAccessorAddrgenIdentityL1,
    ::testing::ValuesIn(unit_tests::dm::ta_addrgen::interleaved_l1_matrix()),
    [](const ::testing::TestParamInfo<unit_tests::dm::ta_addrgen::MatrixParam>& info) {
        using namespace unit_tests::dm::ta_addrgen;
        return info.param.layout.name + "_" + shape_name(info.param.shape) + "_" +
               iter_mode_name(info.param.iter_mode) + (info.param.implicit_sync ? "" : "_Explicit");
    });

// ============================================================================
// Walk contention: three tensors per DM core, two address generators
// ============================================================================
//
// ta_multi_reader_to_dfb reads page p of src0, src1, src2 round-robin; ta_multi_writer_from_dfb writes dst0..2 the same
// way. Each core has 3 walks for 2 address generators, so after warm-up every transfer finds its walk parked. With
// spill/restore (default) that is a register restore and the software seeks stay what one tensor alone would need;
// built with TT_TA_ADDRGEN_NO_SPILL an evicted walk is dropped, so every transfer is a software seek.
namespace contention {

constexpr uint32_t kNumTensors = 3;
constexpr auto kMultiReader =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_multi_reader_to_dfb.cpp";
constexpr auto kMultiWriter =
    "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/ta_multi_writer_from_dfb.cpp";

struct Param {
    std::string layout;         // a layout_cases() name
    uint32_t seeks_per_tensor;  // software seeks one tensor alone needs in page-id order
    bool spill;
};

void PrintTo(const Param& p, std::ostream* os) { *os << p.layout << (p.spill ? "/Spill" : "/NoSpill"); }

std::vector<Param> params() {
    std::vector<Param> out;
    for (const auto& [layout, seeks] : std::vector<std::pair<std::string, uint32_t>>{
             {"InterleavedDram", 1}, {"WidthL1", 1}, {"HeightL1", 1}, {"NdBlockRoundRobinL1", 2}}) {
        out.push_back({layout, seeks, true});
        out.push_back({layout, seeks, false});
    }
    return out;
}

}  // namespace contention

class TensorAccessorAddrgenContention : public QuasarMeshDeviceSingleCardFixture,
                                        public ::testing::WithParamInterface<contention::Param> {};

TEST_P(TensorAccessorAddrgenContention, ThreeTensorsTwoAddrgens) {
    using namespace unit_tests::dm::ta_addrgen;
    const auto& p = GetParam();
    auto& device = *devices_.at(0);
    const auto cases = layout_cases();
    const auto it = std::find_if(cases.begin(), cases.end(), [&](const LayoutCase& c) { return c.name == p.layout; });
    ASSERT_NE(it, cases.end()) << p.layout;
    const LayoutCase lc = resolve_grid(device, *it);
    if (const auto reason = skip_reason(device, lc); !reason.empty()) {
        GTEST_SKIP() << lc.name << ": " << reason;
    }

    const m2::NodeCoord node{0, 0};
    const uint32_t page_size = page_size_bytes(lc);
    const uint32_t pages = num_pages(lc);
    const uint32_t words_per_page = page_size / sizeof(uint32_t);
    const TensorSpec tensor_spec = make_tensor_spec(lc);

    std::vector<MeshTensor> ins, outs;
    for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
        ins.push_back(MeshTensor::allocate_on_device(device, tensor_spec));
        outs.push_back(MeshTensor::allocate_on_device(device, tensor_spec));
    }

    auto make = [&](const std::string& name,
                    const char* source,
                    const std::string& dfb_accessor,
                    m2::DFBEndpointType endpoint,
                    const std::string& tensor_prefix) {
        m2::KernelSpec kernel{
            .unique_id = m2::KernelSpecName{name},
            .source = std::filesystem::path{source},
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "report_addr"}},
            .hw_config = m2::DataMovementHardwareConfig{},
        };
        kernel.dfb_bindings = {
            {.dfb_spec_name = m2::DFBSpecName{"staging"},
             .accessor_name = dfb_accessor,
             .endpoint_type = endpoint,
             .access_pattern = m2::DFBAccessPattern::STRIDED}};
        for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
            const std::string tensor = tensor_prefix + std::to_string(t);
            m2::test_helpers::BindTensorParameterToKernel(kernel, tensor, tensor);
        }
        kernel.compiler_options.defines.emplace(kAddrgenStatsDefine, "1");
        for (const auto& [define, value] : addrgen_bringup_defines(device)) {
            kernel.compiler_options.defines.emplace(define, value);
        }
        if (!p.spill) {
            kernel.compiler_options.defines.emplace("TT_TA_ADDRGEN_NO_SPILL", "1");
        }
        if (std::getenv("TT_TA_ADDRGEN_TRACE") != nullptr) {  // debug: per-transfer address trace (device print)
            kernel.compiler_options.defines.emplace("TT_TA_ADDRGEN_TRACE", "1");
        }
        return kernel;
    };
    m2::KernelSpec producer = make("producer", contention::kMultiReader, "out", m2::DFBEndpointType::PRODUCER, "src");
    m2::KernelSpec consumer = make("consumer", contention::kMultiWriter, "in", m2::DFBEndpointType::CONSUMER, "dst");

    auto dfb = m2::test_helpers::MakeMinimalDFB("staging", page_size, kNumDfbEntries);
    dfb.data_format_metadata = to_data_format(lc.dtype);
    m2::ProgramSpec spec{
        .name = "ta_addrgen_contention_" + lc.name,
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"producer", "consumer"})},
    };
    for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
        spec.tensor_parameters.push_back(
            {.unique_id = m2::TensorParamName{"src" + std::to_string(t)}, .spec = tensor_spec});
        spec.tensor_parameters.push_back(
            {.unique_id = m2::TensorParamName{"dst" + std::to_string(t)}, .spec = tensor_spec});
    }
    Program program = m2::MakeProgramFromSpec(device, spec);

    auto stats_region = make_l1_region(device, 2 * kStatsStride);
    const uint32_t producer_report = static_cast<uint32_t>(stats_region->address());
    const uint32_t consumer_report = producer_report + kStatsStride;
    std::vector<uint32_t> stats_init(2 * kStatsStride / sizeof(uint32_t), 0xDEADBEEF);
    slow_dispatch::WriteToL1(device, node, producer_report, stats_init);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"producer"},
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"num_pages", pages}, {"report_addr", producer_report}})},
        {.kernel = m2::KernelSpecName{"consumer"},
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"num_pages", pages}, {"report_addr", consumer_report}})},
    };
    for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
        params.tensor_args.emplace(m2::TensorParamName{"src" + std::to_string(t)}, std::cref(ins[t]));
        params.tensor_args.emplace(m2::TensorParamName{"dst" + std::to_string(t)}, std::cref(outs[t]));
    }
    m2::SetProgramRunArgs(program, params);

    std::vector<std::vector<uint32_t>> expected(contention::kNumTensors);
    std::vector<uint32_t> zeros(pages * words_per_page, 0);
    std::vector<uint32_t> readback;
    for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
        expected[t].resize(pages * words_per_page);
        for (uint32_t i = 0; i < expected[t].size(); ++i) {
            expected[t][i] = ((t + 1) << 28) ^ (i * 2654435761u);  // distinct per tensor
        }
        slow_dispatch::WriteToBuffer(ins[t].mesh_buffer(), expected[t]);
        slow_dispatch::WriteToBuffer(outs[t].mesh_buffer(), zeros);
        slow_dispatch::ReadFromBuffer(ins[t].mesh_buffer(), readback);
        ASSERT_EQ(readback, expected[t]) << "source " << t << " staging not visible before launch";
    }

    auto workload = LaunchProgram(device, std::move(program));

    for (uint32_t t = 0; t < contention::kNumTensors; ++t) {
        slow_dispatch::ReadFromBuffer(outs[t].mesh_buffer(), readback);
        ASSERT_EQ(readback.size(), expected[t].size());
        for (uint32_t i = 0; i < readback.size(); ++i) {
            ASSERT_EQ(readback[i], expected[t][i]) << "tensor " << t << ": first mismatch at page "
                                                   << i / words_per_page << " word " << i % words_per_page;
        }
    }

    const uint32_t transfers = contention::kNumTensors * pages;
    for (const auto& [kernel, report_addr] :
         {std::pair<std::string, uint32_t>{"reader", producer_report},
          std::pair<std::string, uint32_t>{"writer", consumer_report}}) {
        std::vector<uint32_t> stats;
        slow_dispatch::ReadFromL1(device, node, report_addr, kNumStatsWords * sizeof(uint32_t), stats);
        expect_transfer_stats(
            kernel,
            stats,
            lc,
            transfers,
            IterMode::PageIdLoop,
            /*addrgen_allowed=*/true,
            interleaved_banks_walkable(device, lc.buffer_type));
        const uint32_t seeks = stats[3];
        const uint32_t restores = stats[6];
        if (p.spill) {
            // Each walk is seeked as it would be alone; every transfer after the first three restores a parked walk.
            EXPECT_EQ(seeks, contention::kNumTensors * p.seeks_per_tensor) << kernel;
            EXPECT_EQ(restores, transfers - contention::kNumTensors) << kernel;
        } else {
            // Round-robin over 3 walks with 2 address generators evicts the walk needed next: every transfer seeks.
            EXPECT_EQ(seeks, transfers) << kernel;
            EXPECT_EQ(restores, 0u) << kernel;
        }
    }

    // Op-to-op R/W inference: three bound tensors per kernel must be three distinct records.
    expect_buf_rw(
        "producer",
        query_buf_rw(workload, device, "producer"),
        TensorRole::Reads,
        contention::kNumTensors,
        IterMode::PageIdLoop);
    expect_buf_rw(
        "consumer",
        query_buf_rw(workload, device, "consumer"),
        TensorRole::Writes,
        contention::kNumTensors,
        IterMode::PageIdLoop);
}

INSTANTIATE_TEST_SUITE_P(
    Walks,
    TensorAccessorAddrgenContention,
    ::testing::ValuesIn(contention::params()),
    [](const ::testing::TestParamInfo<contention::Param>& info) {
        return info.param.layout + (info.param.spill ? "_Spill" : "_NoSpill");
    });

// ============================================================================
// Reads and writes in one kernel: source sides vs destination sides
// ============================================================================
//
// ta_mixed_reader_writer reads three tensors and writes one, page by page. Reads walk on the address generators'
// source sides and writes on their destination sides, so:
//   - the three read walks share two source slots: 3 seeks, then every read after the first three restores a parked
//   walk;
//   - the write walk keeps its destination slot: 1 seek and no restores, however the reads behave.
// (With one shared pool of slots, the reads would evict the write walk too.)
// Sources and destinations may have different layouts (even different memories: an L1-sharded source with a DRAM
// destination walks two different ATT windows on the two sides of one address generator). The copied page size and
// page count must match.
struct MixedParam {
    std::string src_layout;
    std::string dst_layout;
    uint32_t seeks_per_src;  // software seeks one source tensor needs alone
    uint32_t seeks_per_dst;
};

void PrintTo(const MixedParam& p, std::ostream* os) { *os << p.src_layout << "->" << p.dst_layout; }

class TensorAccessorAddrgenMixed : public QuasarMeshDeviceSingleCardFixture,
                                   public ::testing::WithParamInterface<MixedParam> {};

TEST_P(TensorAccessorAddrgenMixed, ReadsAndWritesDoNotCompete) {
    using namespace unit_tests::dm::ta_addrgen;
    const auto& param = GetParam();
    auto& device = *devices_.at(0);
    const auto cases = layout_cases();
    auto find_layout = [&](const std::string& name) {
        const auto it = std::find_if(cases.begin(), cases.end(), [&](const LayoutCase& c) { return c.name == name; });
        EXPECT_NE(it, cases.end()) << name;
        return resolve_grid(device, *it);
    };
    const LayoutCase lc = find_layout(param.src_layout);
    const LayoutCase lc_dst = find_layout(param.dst_layout);
    for (const LayoutCase* c : {&lc, &lc_dst}) {
        if (const auto reason = skip_reason(device, *c); !reason.empty()) {
            GTEST_SKIP() << c->name << ": " << reason;
        }
    }
    ASSERT_EQ(page_size_bytes(lc), page_size_bytes(lc_dst)) << "copy needs equal page sizes";
    ASSERT_EQ(num_pages(lc), num_pages(lc_dst)) << "copy needs equal page counts";
    constexpr uint32_t kNumSrc = 3;
    constexpr uint32_t kNumDst = 1;

    const m2::NodeCoord node{0, 0};
    const uint32_t page_size = page_size_bytes(lc);
    const uint32_t pages = num_pages(lc);
    const uint32_t words_per_page = page_size / sizeof(uint32_t);
    const TensorSpec tensor_spec = make_tensor_spec(lc);
    const TensorSpec dst_tensor_spec = make_tensor_spec(lc_dst);

    std::vector<MeshTensor> ins, outs;
    for (uint32_t t = 0; t < kNumSrc; ++t) {
        ins.push_back(MeshTensor::allocate_on_device(device, tensor_spec));
    }
    for (uint32_t t = 0; t < kNumDst; ++t) {
        outs.push_back(MeshTensor::allocate_on_device(device, dst_tensor_spec));
    }

    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{"mixed"},
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/"
                                        "ta_mixed_reader_writer.cpp"},
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "scratch_addr", "report_addr"}},
        .hw_config = m2::DataMovementHardwareConfig{},
    };
    m2::ProgramSpec spec{
        .name = "ta_addrgen_mixed_" + lc.name,
        .kernels = {kernel},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"mixed"})},
    };
    auto add_tensor = [&](const std::string& name, const TensorSpec& tensor_spec_for) {
        m2::test_helpers::BindTensorParameterToKernel(spec.kernels[0], name, name);
        spec.tensor_parameters.push_back({.unique_id = m2::TensorParamName{name}, .spec = tensor_spec_for});
    };
    for (uint32_t t = 0; t < kNumSrc; ++t) {
        add_tensor("src" + std::to_string(t), tensor_spec);
    }
    for (uint32_t t = 0; t < kNumDst; ++t) {
        add_tensor("dst" + std::to_string(t), dst_tensor_spec);
    }
    spec.kernels[0].compiler_options.defines.emplace(kAddrgenStatsDefine, "1");
    for (const auto& [define, value] : addrgen_bringup_defines(device)) {
        spec.kernels[0].compiler_options.defines.emplace(define, value);
    }
    if (std::getenv("TT_TA_ADDRGEN_TRACE") != nullptr) {  // debug: per-transfer address trace (device print)
        spec.kernels[0].compiler_options.defines.emplace("TT_TA_ADDRGEN_TRACE", "1");
    }
    Program program = m2::MakeProgramFromSpec(device, spec);

    auto scratch = make_l1_region(device, kNumSrc * page_size);
    auto stats_region = make_l1_region(device, kStatsStride);
    std::vector<uint32_t> stats_init(kStatsStride / sizeof(uint32_t), 0xDEADBEEF);
    slow_dispatch::WriteToL1(device, node, stats_region->address(), stats_init);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"mixed"},
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
             node,
             {{"num_pages", pages},
              {"scratch_addr", static_cast<uint32_t>(scratch->address())},
              {"report_addr", static_cast<uint32_t>(stats_region->address())}})}};
    for (uint32_t t = 0; t < kNumSrc; ++t) {
        params.tensor_args.emplace(m2::TensorParamName{"src" + std::to_string(t)}, std::cref(ins[t]));
    }
    for (uint32_t t = 0; t < kNumDst; ++t) {
        params.tensor_args.emplace(m2::TensorParamName{"dst" + std::to_string(t)}, std::cref(outs[t]));
    }
    m2::SetProgramRunArgs(program, params);

    std::vector<std::vector<uint32_t>> expected(kNumSrc);
    std::vector<uint32_t> zeros(pages * words_per_page, 0);
    std::vector<uint32_t> readback;
    for (uint32_t t = 0; t < kNumSrc; ++t) {
        expected[t].resize(pages * words_per_page);
        for (uint32_t i = 0; i < expected[t].size(); ++i) {
            expected[t][i] = ((t + 1) << 28) ^ (i * 2654435761u);
        }
        slow_dispatch::WriteToBuffer(ins[t].mesh_buffer(), expected[t]);
        slow_dispatch::ReadFromBuffer(ins[t].mesh_buffer(), readback);
        ASSERT_EQ(readback, expected[t]) << "source " << t << " staging not visible before launch";
    }
    for (uint32_t t = 0; t < kNumDst; ++t) {
        slow_dispatch::WriteToBuffer(outs[t].mesh_buffer(), zeros);
        slow_dispatch::ReadFromBuffer(outs[t].mesh_buffer(), readback);
        ASSERT_EQ(readback, zeros) << "destination " << t << " zero-fill not visible before launch";
    }

    auto workload = LaunchProgram(device, std::move(program));

    for (uint32_t t = 0; t < kNumDst; ++t) {  // dst t is a copy of src t
        slow_dispatch::ReadFromBuffer(outs[t].mesh_buffer(), readback);
        ASSERT_EQ(readback, expected[t]) << "destination " << t << " differs from source " << t;
    }

    std::vector<uint32_t> stats;
    slow_dispatch::ReadFromL1(device, node, stats_region->address(), kNumStatsWords * sizeof(uint32_t), stats);
    const uint32_t transfers = (kNumSrc + kNumDst) * pages;
    expect_transfer_stats(
        "mixed",
        stats,
        lc,
        transfers,
        IterMode::PageIdLoop,
        /*addrgen_allowed=*/true,
        interleaved_banks_walkable(device, lc.buffer_type));
    EXPECT_EQ(stats[0], transfers) << "every read and write address from the HW AddrGen (both layouts walkable here)";
    const uint32_t seeks = stats[3];
    const uint32_t restores = stats[6];
    const uint32_t write_seeks = stats[7];
    const uint32_t write_restores = stats[8];
    const uint32_t read_seeks = seeks - write_seeks;
    const uint32_t read_restores = restores - write_restores;
    EXPECT_EQ(write_seeks, kNumDst * param.seeks_per_dst)
        << "each write walk seeks once and keeps its destination slot";
    EXPECT_EQ(write_restores, 0u) << "reads must not evict writes";
    EXPECT_EQ(read_seeks, kNumSrc * param.seeks_per_src);
    EXPECT_EQ(read_restores, kNumSrc * pages - kNumSrc) << "three read walks share two source slots";

    // Every tensor is a distinct record, reads and writes separate (dst0/dst1 never appear as read).
    const ll_api::BufRwInfo rw = query_buf_rw(workload, device, "mixed");
    EXPECT_FALSE(rw.opaque);
    EXPECT_EQ(rw.reads.size(), kNumSrc);
    EXPECT_EQ(rw.writes.size(), kNumDst);
}

INSTANTIATE_TEST_SUITE_P(
    Walks,
    TensorAccessorAddrgenMixed,
    ::testing::Values(
        MixedParam{"InterleavedDram", "InterleavedDram", 1, 1},
        MixedParam{"WidthL1", "WidthL1", 1, 1},
        MixedParam{"HeightL1", "HeightL1", 1, 1},
        MixedParam{"NdBlockRoundRobinL1", "NdBlockRoundRobinL1", 2, 2},
        // Different memories on the two sides: read L1 banks while writing DRAM banks, and the reverse.
        MixedParam{"HeightL1", "InterleavedDram", 1, 1},
        MixedParam{"InterleavedDram", "HeightL1", 1, 1}),
    [](const ::testing::TestParamInfo<MixedParam>& info) {
        return info.param.src_layout + "_to_" + info.param.dst_layout;
    });

// Randomized fuzz layer: same fixture/body as PatternMatrix, params drawn by fuzz_cases() instead of
// the curated matrix(). Deterministic by default; see fuzz_cases() for the seed/count env overrides.
INSTANTIATE_TEST_SUITE_P(
    Fuzz,
    TensorAccessorAddrgen,
    ::testing::ValuesIn(unit_tests::dm::ta_addrgen::fuzz_cases()),
    [](const ::testing::TestParamInfo<unit_tests::dm::ta_addrgen::MatrixParam>& info) {
        using namespace unit_tests::dm::ta_addrgen;
        return info.param.layout.name + "_" + shape_name(info.param.shape) + "_" + iter_mode_name(info.param.iter_mode);
    });

// ============================================================================
// Phase B, step B1: real HW AddrGen addresses (interleaved DRAM / L1)
// ============================================================================
//
// Walks an interleaved TensorAccessor with the real overlay address generator, banking parameters
// derived from the live ATT map (not the hardcoded, unverified endpoint IDs in the older
// addrgen_interleaved_example.cpp). The kernel cross-checks every generated address against the software
// TensorAccessor::get_noc_addr() and, in ReadEveryPage, issues the read through the ordinary NoC V3 API at
// the hardware address. See tensor_accessor_addrgen.h and addrgen_hw_interleaved_read.cpp.
namespace b1 {

constexpr uint32_t kRows = 16;
constexpr uint32_t kCols = 64;
constexpr uint32_t kNumPages = kRows;
constexpr uint32_t kPageSize = kCols * sizeof(uint32_t);

// Must match ReportWord in addrgen_hw_interleaved_read.cpp.
enum ReportWord : uint32_t {
    kMismatches = 0,
    kFirstBadPage,
    kFirstBadHwLo,
    kFirstBadHwHi,
    kFirstBadSwLo,
    kFirstBadSwHi,
    kPagesIssued,
    kWalkable,
    kNumBanks,
    kBankSelector0,
    kBankOffset0 = kBankSelector0 + 8,
    kNumReportWords = kBankOffset0 + 8,
};

// "bank i -> selector s (+offset o)" for the banks the kernel reported, for failure messages.
std::string describe_banks(const std::vector<uint32_t>& report) {
    std::string out;
    const uint32_t n = std::min<uint32_t>(report[kNumBanks], 8);
    for (uint32_t b = 0; b < n; ++b) {
        out += fmt::format(
            "{}bank {} -> selector {} (+offset {})",
            b ? ", " : "",
            b,
            report[kBankSelector0 + b],
            static_cast<int32_t>(report[kBankOffset0 + b]));
    }
    return fmt::format("{} bank(s): {}", report[kNumBanks], out);
}

using unit_tests::dm::ta_addrgen::make_l1_region;

struct Result {
    std::vector<uint32_t> report;
    std::vector<uint32_t> dest;
};

Result run(distributed::MeshDevice& device, bool is_dram, bool issue_real_reads, MeshTensor& src_tensor) {
    namespace m2 = experimental;
    const m2::NodeCoord node{0, 0};
    auto dest = make_l1_region(device, kNumPages * kPageSize);
    auto report = make_l1_region(device, kNumReportWords * sizeof(uint32_t));

    std::vector<uint32_t> zeros(kNumPages * kCols, 0);
    slow_dispatch::WriteToL1(device, node, dest->address(), zeros);
    std::vector<uint32_t> report_init(kNumReportWords, 0xDEADBEEF);  // distinguishes "never written"
    slow_dispatch::WriteToL1(device, node, report->address(), report_init);

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/"
                                        "addrgen_hw_interleaved_read.cpp"},
        .num_threads = 1,
        .compile_time_args =
            {{"is_dram", is_dram ? 1u : 0u},
             {"issue_real_reads", issue_real_reads ? 1u : 0u},
             {"walkable",
              unit_tests::dm::ta_addrgen::interleaved_banks_walkable(
                  device, is_dram ? BufferType::DRAM : BufferType::L1)
                  ? 1u
                  : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "dest_addr", "report_addr"}},
        .hw_config = m2::DataMovementHardwareConfig{},
    };
    m2::test_helpers::BindTensorParameterToKernel(reader, "src", "src");

    m2::ProgramSpec spec{
        .name = "hw_addrgen_interleaved",
        .kernels = {reader},
        .tensor_parameters = {{.unique_id = m2::TensorParamName{"src"}, .spec = src_tensor.tensor_spec()}},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"reader"})},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"reader"},
         .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
             node,
             {{"num_pages", kNumPages},
              {"dest_addr", static_cast<uint32_t>(dest->address())},
              {"report_addr", static_cast<uint32_t>(report->address())}})}};
    params.tensor_args = {{m2::TensorParamName{"src"}, std::cref(src_tensor)}};
    m2::SetProgramRunArgs(program, params);

    LaunchProgram(device, std::move(program));

    Result result;
    slow_dispatch::ReadFromL1(device, node, report->address(), kNumReportWords * sizeof(uint32_t), result.report);
    slow_dispatch::ReadFromL1(device, node, dest->address(), kNumPages * kPageSize, result.dest);
    return result;
}

// A layout whose banks the recipe can't walk (host-checked, see interleaved_banks_walkable) stays on the software
// path, so it skips rather than fails; the kernel's device print lists the bank -> selector map.
bool recipe_applies(const std::vector<uint32_t>& report) {
    return report.size() == kNumReportWords && report[kWalkable] == 1u;
}

void expect_addresses_match(const std::vector<uint32_t>& report) {
    ASSERT_EQ(report.size(), static_cast<size_t>(kNumReportWords));
    ASSERT_NE(report[kWalkable], 0xDEADBEEFu) << "kernel never wrote its report";
    const uint64_t hw = (static_cast<uint64_t>(report[kFirstBadHwHi]) << 32) | report[kFirstBadHwLo];
    const uint64_t sw = (static_cast<uint64_t>(report[kFirstBadSwHi]) << 32) | report[kFirstBadSwLo];
    EXPECT_EQ(report[kMismatches], 0u) << report[kMismatches] << " of " << kNumPages
                                       << " addrgen addresses differ from TensorAccessor::get_noc_addr; first at page "
                                       << report[kFirstBadPage] << ": hw 0x" << std::hex << hw << " sw 0x" << sw;
}

// Host's view of the L1 bank -> core map, to compare against the kernel's bank -> ATT selector report.
void log_l1_bank_map(distributed::MeshDevice& device) {
    for (uint32_t bank = 0; bank < device.allocator()->get_num_banks(BufferType::L1); ++bank) {
        const CoreCoord logical = device.allocator()->get_logical_core_from_bank_id(bank);
        log_info(
            tt::LogTest,
            "host: L1 bank {} -> logical core {} -> virtual core {}",
            bank,
            logical.str(),
            device.worker_core_from_logical_core(logical).str());
    }
}

MeshTensor make_src_tensor(distributed::MeshDevice& device, bool is_dram) {
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, is_dram ? BufferType::DRAM : BufferType::L1};
    auto tensor_layout = TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), memory_config);
    return MeshTensor::allocate_on_device(device, TensorSpec(Shape{kRows, kCols}, tensor_layout));
}

}  // namespace b1

class HwAddrgenInterleaved : public QuasarMeshDeviceSingleCardFixture, public ::testing::WithParamInterface<bool> {};

// Address-only: no NoC transaction is issued, so this can't hang. Run it first when bringing up a new
// recipe or platform.
TEST_P(HwAddrgenInterleaved, AddressesMatchSoftware) {
    const bool is_dram = GetParam();
    auto& device = *devices_.at(0);
    if (!is_dram) {
        b1::log_l1_bank_map(device);
    }
    MeshTensor src_tensor = b1::make_src_tensor(device, is_dram);
    const auto result = b1::run(device, is_dram, /*issue_real_reads=*/false, src_tensor);
    if (!b1::recipe_applies(result.report)) {
        GTEST_SKIP() << "interleaved bank selectors on this device are not an ascending stride-1 run (see device "
                        "print); the interleaved recipe doesn't apply and this layout stays on the software path";
    }
    b1::expect_addresses_match(result.report);
}

TEST_P(HwAddrgenInterleaved, ReadEveryPage) {
    const bool is_dram = GetParam();
    auto& device = *devices_.at(0);
    MeshTensor src_tensor = b1::make_src_tensor(device, is_dram);

    std::vector<uint32_t> expected(b1::kNumPages * b1::kCols);
    for (uint32_t i = 0; i < expected.size(); ++i) {
        expected[i] = 0x5A000000u ^ (i * 2654435761u);
    }
    slow_dispatch::WriteToBuffer(src_tensor.mesh_buffer(), expected);
    std::vector<uint32_t> readback;
    slow_dispatch::ReadFromBuffer(src_tensor.mesh_buffer(), readback);
    ASSERT_EQ(readback, expected) << "source staging not visible before launch";

    const auto result = b1::run(device, is_dram, /*issue_real_reads=*/true, src_tensor);
    if (!b1::recipe_applies(result.report)) {
        GTEST_SKIP() << "interleaved bank selectors on this device are not an ascending stride-1 run (see device "
                        "print); the interleaved recipe doesn't apply and this layout stays on the software path";
    }
    b1::expect_addresses_match(result.report);
    ASSERT_EQ(result.report[b1::kPagesIssued], b1::kNumPages) << "not every page was issued";
    ASSERT_EQ(result.dest.size(), expected.size());
    for (uint32_t p = 0; p < b1::kNumPages; ++p) {
        for (uint32_t w = 0; w < b1::kCols; ++w) {
            const uint32_t idx = (p * b1::kCols) + w;
            ASSERT_EQ(result.dest[idx], expected[idx]) << "first mismatch at page " << p << " word " << w;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    B1, HwAddrgenInterleaved, ::testing::Values(true, false), [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Dram" : "L1";
    });

// Interleaved L1 with an identity l1_bank_remap (see IdentityL1RemapFixture): here the recipe must apply, and not
// applying is a failure, not a skip.
class HwAddrgenInterleavedIdentityL1 : public IdentityL1RemapFixture {};

TEST_F(HwAddrgenInterleavedIdentityL1, AddressesMatchSoftware) {
    auto& device = *devices_.at(0);
    b1::log_l1_bank_map(device);
    MeshTensor src_tensor = b1::make_src_tensor(device, /*is_dram=*/false);
    const auto result = b1::run(device, /*is_dram=*/false, /*issue_real_reads=*/false, src_tensor);
    ASSERT_TRUE(b1::recipe_applies(result.report))
        << "with an identity l1_bank_remap the L1 banks should be an ascending stride-1 selector run with no "
           "per-bank offset; got "
        << b1::describe_banks(result.report);
    b1::expect_addresses_match(result.report);
}

TEST_F(HwAddrgenInterleavedIdentityL1, ReadEveryPage) {
    auto& device = *devices_.at(0);
    MeshTensor src_tensor = b1::make_src_tensor(device, /*is_dram=*/false);

    std::vector<uint32_t> expected(b1::kNumPages * b1::kCols);
    for (uint32_t i = 0; i < expected.size(); ++i) {
        expected[i] = 0x5A000000u ^ (i * 2654435761u);
    }
    slow_dispatch::WriteToBuffer(src_tensor.mesh_buffer(), expected);
    std::vector<uint32_t> readback;
    slow_dispatch::ReadFromBuffer(src_tensor.mesh_buffer(), readback);
    ASSERT_EQ(readback, expected) << "source staging not visible before launch";

    const auto result = b1::run(device, /*is_dram=*/false, /*issue_real_reads=*/true, src_tensor);
    ASSERT_TRUE(b1::recipe_applies(result.report))
        << "with an identity l1_bank_remap the L1 banks should be an ascending stride-1 selector run with no "
           "per-bank offset; got "
        << b1::describe_banks(result.report);
    b1::expect_addresses_match(result.report);
    ASSERT_EQ(result.report[b1::kPagesIssued], b1::kNumPages) << "not every page was issued";
    ASSERT_EQ(result.dest.size(), expected.size());
    for (uint32_t p = 0; p < b1::kNumPages; ++p) {
        for (uint32_t w = 0; w < b1::kCols; ++w) {
            const uint32_t idx = (p * b1::kCols) + w;
            ASSERT_EQ(result.dest[idx], expected[idx]) << "first mismatch at page " << p << " word " << w;
        }
    }
}

// Minimal diagnostic (see addrgen_roundtrip_probe.cpp): what does addrgen_1 hand back after reset, after an
// inner-loop offset alone, and after base_start (the command buffer's SRC_BASE) is also set?
class AddrgenRoundtripProbe : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(AddrgenRoundtripProbe, PeekReflectsConfiguration) {
    namespace m2 = experimental;
    auto& device = *devices_.at(0);
    const m2::NodeCoord node{0, 0};
    auto report = b1::make_l1_region(device, 6 * sizeof(uint32_t));
    std::vector<uint32_t> report_init(6, 0xDEADBEEF);
    slow_dispatch::WriteToL1(device, node, report->address(), report_init);

    m2::KernelSpec probe{
        .unique_id = m2::KernelSpecName{"probe"},
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/"
                                        "addrgen_roundtrip_probe.cpp"},
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
        .hw_config = m2::DataMovementHardwareConfig{},
    };
    m2::ProgramSpec spec{
        .name = "addrgen_roundtrip_probe",
        .kernels = {probe},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"probe"})},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"probe"},
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"report_addr", static_cast<uint32_t>(report->address())}})}};
    m2::SetProgramRunArgs(program, params);

    LaunchProgram(device, std::move(program));

    std::vector<uint32_t> raw;
    ASSERT_TRUE(slow_dispatch::ReadFromL1(device, node, report->address(), 6 * sizeof(uint32_t), raw));
    ASSERT_EQ(raw.size(), 6u);
    ASSERT_NE(raw[0], 0xDEADBEEFu) << "kernel never wrote its report";
    auto peek = [&](uint32_t i) { return (static_cast<uint64_t>(raw[2 * i + 1]) << 32) | raw[2 * i]; };
    // Informational: base_start may or may not be folded in by the addrgen itself; record which.
    log_info(
        tt::LogTest, "addrgen probe: reset 0x{:x}, inner-only 0x{:x}, inner+base 0x{:x}", peek(0), peek(1), peek(2));
    EXPECT_EQ(peek(1), 0x1000u) << "inner-loop start offset is not reflected in peek";
}

// Loop-nest semantics probe (addrgen_loop_probe.cpp): pops a fixed number of addresses from addrgen_1 for one
// configuration and compares them with a model of the loop nest. The sharded walkers' cross-bank recipes depend on
// these semantics -- in particular what the inner loop wraps to when a walk is seeked mid-row -- which
// address_generators.md doesn't pin down.
namespace loop_probe {

// Must match addrgen_loop_probe.cpp.
constexpr uint32_t kNumPops = 24;
constexpr uint32_t kBankShift = 26;
constexpr uint64_t kInnerStride = 0x40;
constexpr uint64_t kInnerEnd = 0x100;
constexpr uint64_t kOuterStride = 0x1000;
constexpr uint64_t kOuterEnd = 0x10000;
enum BankOrder : uint32_t { kBankInner = 0, kBankMiddle = 1, kBankOuter = 2 };

struct Config {
    std::string name;
    uint32_t bank_order = kBankInner;
    uint32_t num_banks = 1;
    uint32_t bank_base = 0;  // BankingConfig.base (endpoint id of the first bank)
    uint32_t bank_start = 0;
    uint32_t inner_start = 0;
    uint32_t outer_start = 0;
    bool use_outer = true;
    uint32_t pop_amount = 1;  // addresses each pop advances by (hardware skip)
    bool plain_pop = false;   // use the count-less pop (must equal pop_amount 1)
    uint32_t spill_after = 0;  // save + move the walk to the other addrgen after this many pops (0 = off)
    bool spill_dirty = false;  // the other addrgen runs a different walk before the restore
    bool dest_side = false;    // program/pop the destination side (must match the same loop model)
};

// Model: each loop counts from its programmed start, and on reaching its end wraps to 0 (not to its start) and
// carries into the next loop out. Banks count base + current; the loop order is set by the bank order. An outer loop
// left at its reset value neither advances nor wraps anything.
std::vector<uint64_t> model(const Config& c) {
    uint64_t x = c.inner_start;
    uint64_t y = c.use_outer ? c.outer_start : 0;
    uint32_t bank = c.bank_start;  // relative to c.bank_base
    auto step_x = [&] {
        x += kInnerStride;
        if (x >= kInnerEnd) {
            x = 0;
            return true;
        }
        return false;
    };
    auto step_y = [&] {
        if (!c.use_outer) {
            return true;
        }
        y += kOuterStride;
        if (y >= kOuterEnd) {
            y = 0;
            return true;
        }
        return false;
    };
    auto step_bank = [&] {
        if (++bank == c.num_banks) {
            bank = 0;
            return true;
        }
        return false;
    };
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < kNumPops; ++i) {
        out.push_back(x + y + (static_cast<uint64_t>(c.bank_base + bank) << kBankShift));
        // A pop of N advances through N addresses of the loop nest (N = 0 behaves like 1).
        for (uint32_t n = 0; n < std::max<uint32_t>(c.pop_amount, 1); ++n) {
            switch (c.bank_order) {
                case kBankInner: (void)(step_bank() && step_x() && step_y()); break;
                case kBankMiddle: (void)(step_x() && step_bank() && step_y()); break;
                default: (void)(step_x() && step_y() && step_bank()); break;
            }
        }
    }
    return out;
}

std::string describe(uint64_t a) {
    return fmt::format("bank {} + 0x{:x}", a >> kBankShift, a & ((uint64_t{1} << kBankShift) - 1));
}

std::vector<Config> configs() {
    return {
        {.name = "InnerOnlyMidStart", .inner_start = 0x80, .use_outer = false},
        {.name = "InnerOuterMidStart", .inner_start = 0x80},
        {.name = "BankInner2Outer", .bank_order = kBankInner, .num_banks = 2},
        {.name = "BankMiddle2", .bank_order = kBankMiddle, .num_banks = 2},
        {.name = "BankMiddle2MidStart",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .outer_start = 0x2000},
        {.name = "BankOuter2", .bank_order = kBankOuter, .num_banks = 2},
        // The count-less pop must advance by exactly one (the builtin it wraps once passed a pop amount of 0).
        {.name = "BankMiddle2MidStartPlainPop",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .outer_start = 0x2000,
         .plain_pop = true},
        // Save the walk mid-way, restore it into the other address generator: the sequence must be seamless.
        {.name = "BankInner2OuterSpill", .bank_order = kBankInner, .num_banks = 2, .spill_after = 3},
        {.name = "BankMiddle2MidStartSpill",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .outer_start = 0x2000,
         .spill_after = 5},
        // Restore into an address generator that was just running another walk (no reset), as the walkers do.
        {.name = "BankMiddle2Base2SpillDirty",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_base = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .spill_after = 5,
         .spill_dirty = true},
        {.name = "Bank1Base3SpillDirty", .num_banks = 1, .bank_base = 3, .spill_after = 3, .spill_dirty = true},
        // Nonzero bank base: BANK_CURRENT must save and restore as the same (base-relative) value.
        {.name = "Bank1Base3Spill", .num_banks = 1, .bank_base = 3, .spill_after = 3},
        {.name = "BankMiddle2Base2Spill",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_base = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .spill_after = 5},
        // The destination side has its own registers and bank-order field; it must follow the same loop model, with
        // pops, skips, the count-less pop, and save/restore.
        {.name = "DestBankInner2Outer", .bank_order = kBankInner, .num_banks = 2, .dest_side = true},
        {.name = "DestBankMiddle2MidStart",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .outer_start = 0x2000,
         .dest_side = true},
        {.name = "DestBankOuter2", .bank_order = kBankOuter, .num_banks = 2, .dest_side = true},
        {.name = "DestBankMiddle2Skip5PlainPop",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .plain_pop = true,
         .dest_side = true},
        {.name = "DestBankMiddle2Skip5", .bank_order = kBankMiddle, .num_banks = 2, .pop_amount = 5, .dest_side = true},
        {.name = "DestBankMiddle2Base2SpillDirty",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_base = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .spill_after = 5,
         .spill_dirty = true,
         .dest_side = true},
        // Skips that cross the bank and inner-loop wraps.
        {.name = "BankInner2Skip3", .bank_order = kBankInner, .num_banks = 2, .pop_amount = 3},
        {.name = "BankMiddle2MidStartSkip5",
         .bank_order = kBankMiddle,
         .num_banks = 2,
         .bank_start = 1,
         .inner_start = 0x80,
         .outer_start = 0x2000,
         .pop_amount = 5},
    };
}

}  // namespace loop_probe

class AddrgenLoopProbe : public QuasarMeshDeviceSingleCardFixture,
                         public ::testing::WithParamInterface<loop_probe::Config> {};

TEST_P(AddrgenLoopProbe, MatchesLoopModel) {
    namespace m2 = experimental;
    const auto& c = GetParam();
    auto& device = *devices_.at(0);
    const m2::NodeCoord node{0, 0};
    constexpr uint32_t kNumRegWords = 12;  // one register snapshot (RegSnapshot order), see addrgen_loop_probe.cpp
    constexpr uint32_t kReportBytes = (loop_probe::kNumPops + 2 * kNumRegWords) * sizeof(uint64_t);
    auto report = unit_tests::dm::ta_addrgen::make_l1_region(device, kReportBytes);
    std::vector<uint32_t> report_init(kReportBytes / sizeof(uint32_t), 0xDEADBEEF);
    slow_dispatch::WriteToL1(device, node, report->address(), report_init);

    m2::KernelSpec probe{
        .unique_id = m2::KernelSpecName{"probe"},
        .source =
            std::filesystem::path{
                "tests/tt_metal/tt_metal/data_movement/quasar_examples/quasar_addrgen/kernels/addrgen_loop_probe.cpp"},
        .num_threads = 1,
        .compile_time_args =
            {{"bank_order", c.bank_order},
             {"num_banks", c.num_banks},
             {"bank_base", c.bank_base},
             {"bank_start", c.bank_start},
             {"inner_start", c.inner_start},
             {"outer_start", c.outer_start},
             {"use_outer", c.use_outer ? 1u : 0u},
             {"pop_amount", c.pop_amount},
             {"plain_pop", c.plain_pop ? 1u : 0u},
             {"spill_after", c.spill_after},
             {"spill_dirty", c.spill_dirty ? 1u : 0u},
             {"dest_side", c.dest_side ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
        .hw_config = m2::DataMovementHardwareConfig{},
    };
    m2::ProgramSpec spec{
        .name = "addrgen_loop_probe",
        .kernels = {probe},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("wu", node, {"probe"})},
    };
    Program program = m2::MakeProgramFromSpec(device, spec);
    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = m2::KernelSpecName{"probe"},
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"report_addr", static_cast<uint32_t>(report->address())}})}};
    m2::SetProgramRunArgs(program, params);
    LaunchProgram(device, std::move(program));

    std::vector<uint32_t> raw;
    ASSERT_TRUE(slow_dispatch::ReadFromL1(device, node, report->address(), kReportBytes, raw));
    ASSERT_EQ(raw.size(), kReportBytes / sizeof(uint32_t));
    ASSERT_NE(raw[1], 0xDEADBEEFu) << "kernel never wrote its report";
    const std::vector<uint64_t> expected = loop_probe::model(c);
    std::string table;
    int first_bad = -1;
    for (uint32_t i = 0; i < loop_probe::kNumPops; ++i) {
        const uint64_t got = (static_cast<uint64_t>(raw[2 * i + 1]) << 32) | raw[2 * i];
        if (got != expected[i] && first_bad < 0) {
            first_bad = static_cast<int>(i);
        }
        table += fmt::format(
            "\n  pop {:2}: {:24} model {:24}{}",
            i,
            loop_probe::describe(got),
            loop_probe::describe(expected[i]),
            got == expected[i] ? "" : "  <-- differs");
    }
    // A spill's restore must leave every register of the walk's side as it was before the save -- the trimmed save
    // reads back only the position, so this is what shows a stale program register (one this walk's pops might not
    // exercise).
    std::string reg_diffs;
    if (c.spill_after != 0) {
        static constexpr const char* kRegNames[] = {
            "bank_current",
            "bank_base",
            "bank_size",
            "bank_skip",
            "inner_stride",
            "inner_end",
            "inner_address",
            "outer_stride",
            "outer_end",
            "outer_address",
            "bank_offset",
            "bank_order"};
        auto word = [&](uint32_t i) {
            const uint32_t w = loop_probe::kNumPops + i;
            return (static_cast<uint64_t>(raw[2 * w + 1]) << 32) | raw[2 * w];
        };
        table += fmt::format("\n  registers before save (after pop {}) / after restore:", c.spill_after - 1);
        for (uint32_t i = 0; i < kNumRegWords; ++i) {
            const uint64_t before = word(i);
            const uint64_t after = word(kNumRegWords + i);
            table += fmt::format(
                "\n    {:14} 0x{:x} / 0x{:x}{}", kRegNames[i], before, after, before == after ? "" : "  <-- differs");
            if (before != after) {
                reg_diffs += fmt::format(" {}", kRegNames[i]);
            }
        }
    }
    log_info(tt::LogTest, "addrgen loop probe {}:{}", c.name, table);
    EXPECT_EQ(first_bad, -1) << c.name << ": first difference from the loop model at pop " << first_bad << table;
    EXPECT_TRUE(reg_diffs.empty()) << c.name << ": restore left registers different from before the save:" << reg_diffs
                                   << table;
}

INSTANTIATE_TEST_SUITE_P(
    LoopNest,
    AddrgenLoopProbe,
    ::testing::ValuesIn(loop_probe::configs()),
    [](const ::testing::TestParamInfo<loop_probe::Config>& info) { return info.param.name; });

}  // namespace tt::tt_metal
