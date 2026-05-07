// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>

// UMD: re-exports CoreType (used in SemaphoreDescriptor::core_type member).
#include <umd/device/types/core_coordinates.hpp>

#include <bitset>
#include <optional>
#include <variant>
#include <vector>

/**
 * TODO (#34009): Move to experimental namespace
 * This header contains the structures that describe the program.
 * Their purpose is to allow users to generate a full program description in a lightweight manner without creating the
 * actual `Program` object.
 */

namespace tt::tt_metal {

struct Tile;
class Buffer;
class Program;
namespace experimental {
class GlobalCircularBuffer;
}  // namespace experimental

struct TileDescriptor {
    TileDescriptor() = default;
    TileDescriptor(const Tile& tile);
    TileDescriptor(uint32_t height, uint32_t width, bool transpose) :
        height(height), width(width), transpose(transpose) {}

    uint32_t height = constants::TILE_HEIGHT;
    uint32_t width = constants::TILE_WIDTH;
    bool transpose = false;

    bool operator==(const TileDescriptor& other) const {
        return height == other.height && width == other.width && transpose == other.transpose;
    }
};

struct CBFormatDescriptor {
    uint8_t buffer_index = 0;
    tt::DataFormat data_format = tt::DataFormat::Float32;
    uint32_t page_size = 0;
    std::optional<TileDescriptor> tile;
};

struct CBDescriptor {
    using FormatDescriptors = ttsl::SmallVector<CBFormatDescriptor, 1>;

    uint32_t total_size = 0;
    CoreRangeSet core_ranges;
    FormatDescriptors format_descriptors;
    FormatDescriptors remote_format_descriptors;

    // TODO: Investigate avoiding storing pointers here
    Buffer* buffer = nullptr;
    uint32_t address_offset = 0;
    const experimental::GlobalCircularBuffer* global_circular_buffer = nullptr;
};

struct SemaphoreDescriptor {
    uint32_t id{};
    CoreType core_type = CoreType::WORKER;
    CoreRangeSet core_ranges;
    uint32_t initial_value = 0;
};

struct ReaderConfigDescriptor {};
struct WriterConfigDescriptor {};
struct DataMovementConfigDescriptor {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
};
struct ComputeConfigDescriptor {
    using UnpackToDestModes = std::vector<UnpackToDestMode>;

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    UnpackToDestModes unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
};

// Declares that a specific per-core runtime arg position holds a buffer base address
// that changes on every dispatch.  Populated via KernelDescriptor::emplace_runtime_args().
// The framework resolves these to RuntimeArgsData* pointers on cache miss and
// patches them directly on cache hits, bypassing create_descriptor() entirely.
struct BufferBinding {
    CoreCoord core;
    uint32_t arg_idx;
    Buffer* buffer = nullptr;
};

// Same as BufferBinding but for common (non-per-core) runtime args.
// Populated via KernelDescriptor::emplace_common_runtime_args().
struct CommonBufferBinding {
    uint32_t arg_idx;
    Buffer* buffer = nullptr;
};

struct KernelDescriptor {
    // TODO: investigate using SmallVector here, using std::vector for now to abide size constraint
    // in tt_stl/tt_stl/reflection.hpp:185:23
    using CompileTimeArgs = std::vector<uint32_t>;
    using NamedCompileTimeArgs = std::vector<std::pair<std::string, uint32_t>>;
    using Defines = std::vector<std::pair<std::string, std::string>>;
    using CoreRuntimeArgs = std::vector<uint32_t>;
    using RuntimeArgs = std::vector<std::pair<CoreCoord, CoreRuntimeArgs>>;
    using CommonRuntimeArgs = CoreRuntimeArgs;
    using BufferBindings = ttsl::SmallVector<BufferBinding, 4>;
    using CommonBufferBindings = ttsl::SmallVector<CommonBufferBinding, 2>;
    using ConfigDescriptor = std::
        variant<ReaderConfigDescriptor, WriterConfigDescriptor, DataMovementConfigDescriptor, ComputeConfigDescriptor>;
    enum class SourceType { FILE_PATH, SOURCE_CODE };

    std::string kernel_source;
    SourceType source_type = SourceType::FILE_PATH;

    CoreRangeSet core_ranges;
    CompileTimeArgs compile_time_args;
    NamedCompileTimeArgs named_compile_time_args;
    Defines defines;

    // vector of pairs, where runtime_args[i].first is the core coord and runtime_args[i].second is the vector of
    // runtime args for that core
    RuntimeArgs runtime_args;
    CommonRuntimeArgs common_runtime_args;

    std::optional<KernelBuildOptLevel> opt_level = std::nullopt;

    ConfigDescriptor config;

    // Buffer args declared via emplace_runtime_args() / emplace_common_runtime_args().
    // The framework resolves these to direct pointers into the cached Program on cache miss,
    // enabling O(1) patching on cache hits without calling create_descriptor() again.
    BufferBindings buffer_bindings;
    CommonBufferBindings common_buffer_bindings;

    // Builder for dynamically-constructed runtime arg lists.  Buffer* entries
    // auto-register as buffer bindings; uint32_t entries embed their value.
    // The variant type is hidden — callers push typed values directly.
    struct RTArgList {
        void push_back(uint32_t v) { items_.emplace_back(v); }
        void push_back(Buffer* b) { items_.emplace_back(b); }
        void reserve(size_t n) { items_.reserve(n); }
        // Append a plain uint32_t range (e.g. from fused-op signaler helpers).
        void append(const std::vector<uint32_t>& v) {
            for (uint32_t x : v) {
                items_.emplace_back(x);
            }
        }

    private:
        std::vector<std::variant<uint32_t, Buffer*>> items_;
        friend struct KernelDescriptor;
    };

    // Push a core's runtime args, automatically registering any Buffer* entries
    // as buffer bindings at their position.  Use this instead of
    // runtime_args.emplace_back() when some args are buffer base addresses.
    void emplace_runtime_args(const CoreCoord& core, std::initializer_list<std::variant<uint32_t, Buffer*>> args);
    void emplace_runtime_args(const CoreCoord& core, const RTArgList& args);
    // Vector overload for dynamically-built arg lists.
    void emplace_runtime_args(const CoreCoord& core, const std::vector<std::variant<uint32_t, Buffer*>>& args);

    // Push common runtime args, automatically registering any Buffer* entries
    // as common buffer bindings.  Use this instead of assigning common_runtime_args
    // directly when some args are buffer base addresses.
    void emplace_common_runtime_args(std::initializer_list<std::variant<uint32_t, Buffer*>> args);
    void emplace_common_runtime_args(const RTArgList& args);
};

struct ProgramDescriptor {
    using KernelDescriptors = ttsl::SmallVector<KernelDescriptor, 3>;
    using SemaphoreDescriptors = ttsl::SmallVector<SemaphoreDescriptor, 3>;
    using CBDescriptors = ttsl::SmallVector<CBDescriptor, 5>;

    KernelDescriptors kernels;
    SemaphoreDescriptors semaphores;
    CBDescriptors cbs;
    std::optional<std::uint64_t> custom_program_hash;

    std::optional<uint32_t> find_available_semaphore_id(const CoreCoord& core, CoreType core_type) const;
};

/**
 * Merge multiple ProgramDescriptors into a single one.
 *
 * @param descriptors Vector of ProgramDescriptors to merge.
 * @return A new ProgramDescriptor containing all kernels, CBs, and semaphores.
 * @throws TT_FATAL if any core ranges overlap between any of the descriptors.
 */
ProgramDescriptor merge_program_descriptors(const std::vector<ProgramDescriptor>& descriptors);

/**
 * Apply a descriptor's runtime arguments to a cached Program.
 *
 * Copies all per-core runtime args, common runtime args, and dynamic circular
 * buffer addresses from the descriptor into the Program.  Kernel handles are
 * the descriptor kernel indices (0, 1, 2, ...) which match the sequential
 * assignment made when the Program was originally built from the same
 * descriptor structure.
 */
void apply_descriptor_runtime_args(Program& program, const ProgramDescriptor& desc);

}  // namespace tt::tt_metal

namespace std {

// Hash support for TileDescriptor (needed for reflection system)
template <>
struct hash<tt::tt_metal::TileDescriptor> {
    std::size_t operator()(const tt::tt_metal::TileDescriptor& tile_desc) const noexcept;
};

/**
 * Hash support for ProgramDescriptor.
 *
 * Hashes kernel paths, core ranges, compile args, defines, CB configs, and semaphores.
 * Runtime arg VALUES are excluded (only their count is hashed for structural matching).
 * If custom_program_hash is set, returns that value directly (allows override).
 *
 * Works with ttsl::hash::hash_combine and standard unordered containers.
 */
template <>
struct hash<tt::tt_metal::ProgramDescriptor> {
    std::size_t operator()(const tt::tt_metal::ProgramDescriptor& descriptor) const noexcept;
};

}  // namespace std

// Formatter support for TileDescriptor (needed for reflection/logging)
template <>
struct fmt::formatter<tt::tt_metal::TileDescriptor> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::TileDescriptor& tile_desc, format_context& ctx) const -> format_context::iterator {
        return fmt::format_to(
            ctx.out(), "TileDescriptor({}x{}{})", tile_desc.height, tile_desc.width, tile_desc.transpose ? ",T" : "");
    }
};
