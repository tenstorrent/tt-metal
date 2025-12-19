// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#include <umd/device/types/core_coordinates.hpp>

#include <optional>

/**
 * TODO (#34009): Move to experimental namespace
 * This header contains the structures that describe the program.
 * Their purpose is to allow users to generate a full program description in a lightweight manner without creating the
 * actual `Program` object.
 */

namespace tt::tt_metal {

struct Tile;
class Buffer;
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
    const experimental::GlobalCircularBuffer* global_circular_buffer = nullptr;
};

struct SemaphoreDescriptor {
    uint32_t id{};
    CoreType core_type = CoreType::WORKER;
    CoreRangeSet core_ranges;
    uint32_t initial_value = 0;
};

struct GlobalSemRef {
    uint32_t index;
    constexpr operator uint32_t() const { return index; }
};

// Type trait to detect GlobalSemRef at compile-time
template <typename T>
struct is_global_sem_ref : std::false_type {};
template <>
struct is_global_sem_ref<GlobalSemRef> : std::true_type {};
template <typename T>
inline constexpr bool is_global_sem_ref_v = is_global_sem_ref<std::decay_t<T>>::value;

//==============================================================================
// CoreRuntimeArgs - Runtime args with automatic GlobalSemaphore resolution
//
// Usage:
//   CoreRuntimeArgs args{buf_addr, size, GlobalSemRef{0}, data, GlobalSemRef{1}};
//   // Later, infra calls:
//   args.resolve(semaphores);  // Resolves only positions 2 and 4
//
// How it works:
//   - Variadic constructor captures which positions are GlobalSemRef at compile-time
//   - Stores ref positions in a small vector (typically 0-3 refs per kernel)
//   - resolve() only touches ref positions - O(num_refs) not O(num_args)
//==============================================================================
struct CoreRuntimeArgs {
    std::vector<uint32_t> args;
    ttsl::SmallVector<uint8_t, 4> ref_positions;  // Indices where GlobalSemRef was used

    CoreRuntimeArgs() = default;

    // Variadic constructor - detects GlobalSemRef positions at compile-time
    template <typename... ArgTypes>
    explicit CoreRuntimeArgs(ArgTypes... values) : args{static_cast<uint32_t>(values)...} {
        capture_ref_positions<ArgTypes...>(std::make_index_sequence<sizeof...(ArgTypes)>{});
    }

    // Resolve GlobalSemRef positions to actual addresses
    template <typename SemaphoreContainer>
    void resolve(const SemaphoreContainer& semaphores) {
        for (auto pos : ref_positions) {
            args[pos] = semaphores[args[pos]].address();
        }
    }

    // Check if resolution is needed
    bool needs_resolution() const { return !ref_positions.empty(); }

    // Access underlying args
    std::vector<uint32_t>& get() { return args; }
    const std::vector<uint32_t>& get() const { return args; }

    // Vector-like interface for compatibility with Program constructor
    auto begin() { return args.begin(); }
    auto end() { return args.end(); }
    auto begin() const { return args.begin(); }
    auto end() const { return args.end(); }
    size_t size() const { return args.size(); }
    bool empty() const { return args.empty(); }
    uint32_t& operator[](size_t i) { return args[i]; }
    uint32_t operator[](size_t i) const { return args[i]; }
    uint32_t* data() { return args.data(); }
    const uint32_t* data() const { return args.data(); }

private:
    template <typename... ArgTypes, size_t... Is>
    void capture_ref_positions(std::index_sequence<Is...>) {
        // Fold expression: for each type, if it's GlobalSemRef, record its position
        ((is_global_sem_ref_v<ArgTypes> ? (ref_positions.push_back(static_cast<uint8_t>(Is)), 0) : 0), ...);
    }
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
struct EthernetConfigDescriptor {
    Eth eth_mode = Eth::SENDER;
    NOC noc = NOC::NOC_0;
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
};

struct KernelDescriptor {
    // TODO: investigate using SmallVector here, using std::vector for now to abide size constraint
    // in tt_stl/tt_stl/reflection.hpp:185:23
    using CompileTimeArgs = std::vector<uint32_t>;
    using NamedCompileTimeArgs = std::vector<std::pair<std::string, uint32_t>>;
    using Defines = std::vector<std::pair<std::string, std::string>>;
    // Runtime args: CoreRuntimeArgs auto-detects GlobalSemRef positions
    using RuntimeArgs = std::vector<std::vector<CoreRuntimeArgs>>;
    using CommonRuntimeArgs = CoreRuntimeArgs;
    using ConfigDescriptor = std::variant<
        ReaderConfigDescriptor,
        WriterConfigDescriptor,
        DataMovementConfigDescriptor,
        ComputeConfigDescriptor,
        EthernetConfigDescriptor>;
    enum class SourceType { FILE_PATH, SOURCE_CODE };

    std::string kernel_source;
    SourceType source_type = SourceType::FILE_PATH;

    CoreRangeSet core_ranges;
    CompileTimeArgs compile_time_args;
    NamedCompileTimeArgs named_compile_time_args;
    Defines defines;

<<<<<<< HEAD
    // vector of pairs, where runtime_args[i].first is the core coord and runtime_args[i].second is the vector of
    // runtime args for that core
=======
    // runtime_args[i][j] = CoreRuntimeArgs for core(i, j)
    // Use GlobalSemRef{index} for semaphore addresses - infra resolves automatically
>>>>>>> ae806b43487 (wip templated kernel arg to resolve globalsemaphore address)
    RuntimeArgs runtime_args;
    CommonRuntimeArgs common_runtime_args;

    std::optional<KernelBuildOptLevel> opt_level = std::nullopt;

    ConfigDescriptor config;
};

struct ProgramDescriptor {
    using KernelDescriptors = ttsl::SmallVector<KernelDescriptor, 3>;
    using SemaphoreDescriptors = ttsl::SmallVector<SemaphoreDescriptor, 3>;
    using CBDescriptors = ttsl::SmallVector<CBDescriptor, 5>;

    KernelDescriptors kernels;
    SemaphoreDescriptors semaphores;
    CBDescriptors cbs;
    std::optional<ttsl::hash::hash_t> custom_program_hash;
};

}  // namespace tt::tt_metal

// Hash support for TileDescriptor (needed for reflection system)
namespace std {
template <>
struct hash<tt::tt_metal::TileDescriptor> {
    std::size_t operator()(const tt::tt_metal::TileDescriptor& tile_desc) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(tile_desc.height, tile_desc.width, tile_desc.transpose);
    }
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
