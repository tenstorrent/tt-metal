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

#include <bitset>
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
    using CoreRuntimeArgs = std::vector<uint32_t>;
    using RuntimeArgs = std::vector<std::pair<CoreCoord, CoreRuntimeArgs>>;
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

    // vector of pairs, where runtime_args[i].first is the core coord and runtime_args[i].second is the vector of
    // runtime args for that core
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

    std::optional<uint32_t> find_available_semaphore_id(const CoreCoord& core, CoreType core_type) const;
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
