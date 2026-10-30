// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "umd/device/coordinates/grendel_noc_address_resolver.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/context/mimir_contract.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_gddr_map.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/cce/cce_copy_ops_abi.h"

namespace tt::tt_metal::mimir_test {

// Host mirror of tt_metal/hw/inc/experimental/cce_gddr.h (device-only header).
inline constexpr uint64_t kKiB = 1024;
inline constexpr uint64_t kMiB = 1024 * kKiB;
inline constexpr uint64_t kGiB = 1024 * kMiB;
inline constexpr uint64_t kAnchorBytes = 128 * kKiB;

inline uint64_t host_cce_sram_mla_address(
    const tt::umd::GrendelAddressWindows& windows, uint32_t location, uint64_t offset) {
    return windows.dram_l1_base + location * windows.dram_l1_stride + offset;
}

inline uint32_t cce_hart_count() {
    return MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::DRAM);
}

// TL1 test layout. Offsets are CCE TL1 offsets starting at the DRAM core type's HAL L1 UNRESERVED; MM3's echo slots
// (0x200 B) sit in the first 0x1000 bytes. The host adds the HAL L1 NoC offset, kernels add MEM_L1_UNCACHED_BASE.
inline constexpr uint64_t kTl1EchoSlotBytes = 64;
inline constexpr uint64_t kTl1HartWindowOffset = 0x1000;
inline constexpr uint64_t kTl1HartWindowBytes = 0x40000;
inline constexpr uint64_t kTl1DoneBlockOffset = 0x1000;
inline constexpr uint64_t kTl1DataOffset = 0x2000;
inline constexpr uint64_t kTl1RemoteLandingOffset = 0x27C000;
inline constexpr uint64_t kTl1RemoteLandingSlotBytes = 0x4000;

inline uint64_t tl1_test_base() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
}

inline uint64_t tl1_test_end() {
    return tl1_test_base() +
           MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
}

inline uint64_t tl1_hart_window(uint32_t hart) {
    return tl1_test_base() + kTl1HartWindowOffset + static_cast<uint64_t>(hart) * kTl1HartWindowBytes;
}

inline uint64_t tl1_done_block(uint32_t hart) { return tl1_hart_window(hart) + kTl1DoneBlockOffset; }

inline uint64_t tl1_data(uint32_t hart) { return tl1_hart_window(hart) + kTl1DataOffset; }

inline uint64_t tl1_remote_landing(uint32_t source_cce_index, uint32_t hart) {
    return tl1_test_base() + kTl1RemoteLandingOffset +
           (static_cast<uint64_t>(source_cce_index) * cce_hart_count() + hart) * kTl1RemoteLandingSlotBytes;
}

inline void check_tl1_layout() {
    const uint64_t harts = cce_hart_count();
    TT_FATAL(
        kTl1HartWindowOffset + harts * kTl1HartWindowBytes <= kTl1RemoteLandingOffset,
        "hart windows end at test offset 0x{:x}, past the remote landing offset 0x{:x}",
        kTl1HartWindowOffset + harts * kTl1HartWindowBytes,
        kTl1RemoteLandingOffset);
    TT_FATAL(
        tl1_test_base() + kTl1RemoteLandingOffset +
                num_cces_from_soc(mimir_contract::configured_soc_desc()) * harts * kTl1RemoteLandingSlotBytes <=
            tl1_test_end(),
        "remote landing area does not fit the CCE UNRESERVED TL1");
}

struct MimirCceL1AddressInfo {
    uint32_t cce_index = 0, mimir_index = 0, location = 0;
    ChipId chip = 0;
    CoreCoord translated_core;
    uint64_t tl1_offset = 0;
    uint64_t cluster_address = 0;
    uint64_t host_axi_address = 0;
    uint64_t mla_address = 0;
    uint32_t size = 0;

    std::string to_string() const {
        return fmt::format(
            "CCE{} (M{} location {}, chip {}) TL1 0x{:x} +{} B: translated core ({}, {}), cluster 0x{:x}, "
            "host AXI 0x{:x}, MLA 0x{:x}",
            cce_index,
            mimir_index,
            location,
            chip,
            tl1_offset,
            size,
            translated_core.x,
            translated_core.y,
            cluster_address,
            host_axi_address,
            mla_address);
    }

    std::string host_string() const {
        return fmt::format(
            "CCE{} (M{} location {}, chip {}) TL1 0x{:x} +{} B: core ({}, {}), cluster 0x{:x}, host AXI 0x{:x}",
            cce_index,
            mimir_index,
            location,
            chip,
            tl1_offset,
            size,
            translated_core.x,
            translated_core.y,
            cluster_address,
            host_axi_address);
    }
};

inline MimirCceL1AddressInfo mimir_cce_l1_address(
    const metal_SocDescriptor& soc, ChipId chip, uint32_t cce_index, uint64_t tl1_offset, uint32_t size) {
    TT_FATAL(tl1_offset % 4 == 0, "TL1 offset 0x{:x} is not 4 B aligned", tl1_offset);
    TT_FATAL(size > 0 && size % 4 == 0, "TL1 access size {} is not a positive multiple of 4", size);
    const Hal& hal = MetalContext::instance().hal();
    const uint64_t tl1_begin = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
    const uint64_t tl1_end = tl1_begin + hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE);
    TT_FATAL(
        tl1_offset >= tl1_begin && tl1_offset + size <= tl1_end,
        "TL1 access [0x{:x}, +{}) is outside the CCE TL1 [0x{:x}, 0x{:x})",
        tl1_offset,
        size,
        tl1_begin,
        tl1_end);

    MimirCceL1AddressInfo info;
    info.cce_index = cce_index;
    info.mimir_index = mimir_index_for_cce(cce_index);
    info.location = location_for_cce(cce_index);
    info.chip = chip;
    info.translated_core = soc.get_physical_dram_core_from_logical(logical_cce_core(soc, cce_index));
    const CoreCoord location_translated_core = translated_dram_core(soc, cce_index);
    TT_FATAL(
        location_translated_core == info.translated_core,
        "CCE {} logical CCE core translates to ({}, {}), which does not match translated core ({}, {}) of its "
        "Mimir location",
        cce_index,
        info.translated_core.x,
        info.translated_core.y,
        location_translated_core.x,
        location_translated_core.y);
    info.tl1_offset = tl1_offset;
    info.cluster_address = MetalContext::instance().hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + tl1_offset;
    const tt::umd::GrendelAddressWindows windows = tt::umd::mimir_local_address_windows(soc);
    // The resolver EmuTTDevice installs for this descriptor (umd/device/tt_device/emu_tt_device.cpp:91-92). The
    // device's own instance is a protected member with no accessor (simulation_tt_device.hpp:167), so the test builds
    // the same one.
    const tt::umd::GrendelNocAddressResolver resolver(soc, windows);
    info.host_axi_address = resolver.to_flat_address(
        tt::umd::CoreCoord(info.translated_core.x, info.translated_core.y, CoreType::DRAM, CoordSystem::TRANSLATED),
        info.cluster_address,
        tt::umd::NocId::NOC0);
    const uint64_t cce_window_address = windows.dram_l1_address(cce_index, info.cluster_address);
    TT_FATAL(
        info.host_axi_address == cce_window_address,
        "CCE {} core ({}, {}) resolves to host AXI 0x{:x}, not CCE {}'s SRAM window 0x{:x}",
        cce_index,
        info.translated_core.x,
        info.translated_core.y,
        info.host_axi_address,
        cce_index,
        cce_window_address);
    info.mla_address = host_cce_sram_mla_address(windows, info.location, tl1_offset);
    info.size = size;
    return info;
}

// Which part of CCE TL1 a host access may touch. MM-0/1 and firmware-state reads: WholeTl1 (reserved regions allowed).
// Tests that run kernels (MM-3 onward): Unreserved.
enum class Tl1Range { WholeTl1, Unreserved };

inline void check_cce_tl1_access(uint64_t tl1_offset, uint64_t size_bytes, Tl1Range range) {
    TT_FATAL(tl1_offset % 4 == 0, "TL1 offset 0x{:x} is not 4 B aligned", tl1_offset);
    TT_FATAL(size_bytes > 0 && size_bytes % 4 == 0, "TL1 access size {} is not a positive multiple of 4", size_bytes);
    const Hal& hal = MetalContext::instance().hal();
    const uint64_t range_begin = range == Tl1Range::WholeTl1
                                     ? hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE)
                                     : tl1_test_base();
    const uint64_t range_end =
        range == Tl1Range::WholeTl1
            ? range_begin + hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE)
            : tl1_test_end();
    TT_FATAL(
        tl1_offset >= range_begin && tl1_offset + size_bytes <= range_end,
        "TL1 access [0x{:x}, +{}) is outside the {} range [0x{:x}, 0x{:x})",
        tl1_offset,
        size_bytes,
        range == Tl1Range::WholeTl1 ? "whole-TL1" : "unreserved",
        range_begin,
        range_end);
}

inline void write_cce_tl1(
    const Cluster& cluster,
    ChipId chip,
    uint32_t cce_index,
    uint64_t tl1_offset,
    const std::vector<uint32_t>& data,
    Tl1Range range) {
    const uint32_t size_bytes = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
    check_cce_tl1_access(tl1_offset, size_bytes, range);
    const MimirCceL1AddressInfo info =
        mimir_cce_l1_address(cluster.get_soc_desc(chip), chip, cce_index, tl1_offset, size_bytes);
    cluster.write_core(data.data(), size_bytes, tt_cxy_pair(chip, info.translated_core), info.cluster_address);
}

inline std::vector<uint32_t> read_cce_tl1(
    const Cluster& cluster, ChipId chip, uint32_t cce_index, uint64_t tl1_offset, uint32_t size_bytes, Tl1Range range) {
    check_cce_tl1_access(tl1_offset, size_bytes, range);
    const MimirCceL1AddressInfo info =
        mimir_cce_l1_address(cluster.get_soc_desc(chip), chip, cce_index, tl1_offset, size_bytes);
    std::vector<uint32_t> data(size_bytes / sizeof(uint32_t), 0);
    cluster.read_core(data.data(), size_bytes, tt_cxy_pair(chip, info.translated_core), info.cluster_address);
    return data;
}

// The open device's chip, through the contract suite's CCE SRAM helpers.
inline void write_cce_tl1(
    IDevice* device,
    uint32_t cce_index,
    uint64_t tl1_offset,
    const std::vector<uint32_t>& data,
    Tl1Range range = Tl1Range::Unreserved) {
    const uint32_t size_bytes = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
    check_cce_tl1_access(tl1_offset, size_bytes, range);
    mimir_contract::write_cce_sram(
        device,
        logical_cce_core(soc_desc_of(device), cce_index),
        mimir_contract::cce_sram_noc_addr(tl1_offset),
        data.data(),
        size_bytes);
}

inline std::vector<uint32_t> read_cce_tl1(
    IDevice* device,
    uint32_t cce_index,
    uint64_t tl1_offset,
    uint32_t size_bytes,
    Tl1Range range = Tl1Range::Unreserved) {
    check_cce_tl1_access(tl1_offset, size_bytes, range);
    std::vector<uint32_t> data(size_bytes / sizeof(uint32_t), 0);
    mimir_contract::read_cce_sram(
        device,
        logical_cce_core(soc_desc_of(device), cce_index),
        mimir_contract::cce_sram_noc_addr(tl1_offset),
        data.data(),
        size_bytes);
    return data;
}

// Not allocator()->get_bank_offset: it truncates to int32, zeroes the split descriptor's 4 GiB view offsets.
inline uint64_t chiplet_offset_of(const metal_SocDescriptor& soc, uint32_t view, uint64_t view_offset) {
    return soc.get_address_offset(static_cast<int>(view)) + view_offset;
}

// Passed to cce_args_echo as a compile-time arg.
inline constexpr uint32_t kArgsEchoMagic = 0xEC40EC40;

enum class CopyKind : uint32_t {
    OwnTl1 = cce_copy_ops_abi::kKindOwnTl1,
    Gddr = cce_copy_ops_abi::kKindGddr,
    SramSpa = cce_copy_ops_abi::kKindSramSpa
};

struct CopyEndpoint {
    CopyKind kind;
    uint32_t selector;
    uint64_t offset;
};

inline CopyEndpoint own_tl1(uint64_t tl1_offset) { return {CopyKind::OwnTl1, 0, tl1_offset}; }

inline CopyEndpoint gddr(uint32_t target_mimir, uint64_t chiplet_offset) {
    return {CopyKind::Gddr, target_mimir, chiplet_offset};
}

inline CopyEndpoint sram_spa(uint32_t target_cce_index, uint64_t tl1_offset) {
    return {CopyKind::SramSpa, target_cce_index, tl1_offset};
}

struct CopyOp {
    CopyEndpoint source;
    CopyEndpoint destination;
    uint32_t num_words;
};

class CopyOpTable {
public:
    void add(const CopyOp& op) { ops_.push_back(op); }

    const std::vector<CopyOp>& ops() const { return ops_; }

    std::vector<uint32_t> encode() const {
        namespace abi = cce_copy_ops_abi;
        std::vector<uint32_t> words(ops_.size() * abi::kOpWords, 0);
        for (size_t op_index = 0; op_index < ops_.size(); ++op_index) {
            const CopyOp& op = ops_[op_index];
            uint32_t* record = words.data() + op_index * abi::kOpWords;
            record[abi::kOpSrcKind] = static_cast<uint32_t>(op.source.kind);
            record[abi::kOpSrcSelector] = op.source.selector;
            record[abi::kOpSrcOffsetLo] = static_cast<uint32_t>(op.source.offset & 0xFFFFFFFFu);
            record[abi::kOpSrcOffsetHi] = static_cast<uint32_t>(op.source.offset >> 32);
            record[abi::kOpDstKind] = static_cast<uint32_t>(op.destination.kind);
            record[abi::kOpDstSelector] = op.destination.selector;
            record[abi::kOpDstOffsetLo] = static_cast<uint32_t>(op.destination.offset & 0xFFFFFFFFu);
            record[abi::kOpDstOffsetHi] = static_cast<uint32_t>(op.destination.offset >> 32);
            record[abi::kOpNumWords] = op.num_words;
        }
        return words;
    }

private:
    std::vector<CopyOp> ops_;
};

struct CceCopyLaunch {
    uint32_t cce_index;
    uint32_t hart;
    CopyOpTable table;
};

struct CceCopyResult {
    uint32_t cce_index;
    uint32_t hart;
    bool done;
    uint32_t xor_signature;
};

inline void expect_copy_done(const CceCopyResult& result, uint32_t expected_xor, const std::string& where) {
    EXPECT_TRUE(result.done) << where;
    EXPECT_EQ(result.xor_signature, expected_xor) << where;
}

// Owns one open MeshDevice. The destructor closes it on every path; a throw from close() is reported, not propagated.
class ScopedMesh {
public:
    explicit ScopedMesh(std::shared_ptr<distributed::MeshDevice> mesh) : mesh_(std::move(mesh)) {}
    ScopedMesh(const ScopedMesh&) = delete;
    ScopedMesh& operator=(const ScopedMesh&) = delete;
    ~ScopedMesh() { close(); }

    void close() {
        std::shared_ptr<distributed::MeshDevice> mesh = std::move(mesh_);
        mesh_.reset();
        if (mesh != nullptr) {
            try {
                EXPECT_TRUE(mesh->close());
            } catch (const std::exception& error) {
                ADD_FAILURE() << "MeshDevice::close threw: " << error.what();
            }
        }
    }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_;
};

// Launches `program` through slow dispatch and waits until its cores are idle.
inline void launch_program_and_wait(IDevice* device, Program& program) {
    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/true);
    slow_dispatch::WaitProgramDone(*device, program);
}

// Opens chip 0 as a unit MeshDevice and owns it. mesh() is for host DRAM reads and writes. device() is the physical
// device under it, for host TL1 access and slow-dispatch launches. The destructor closes the mesh on every path;
// close() closes early.
class ScopedDevice {
public:
    ScopedDevice() : mesh_(distributed::MeshDevice::create_unit_mesh(0)), closer_(mesh_) {}

    distributed::MeshDevice& mesh() const { return *mesh_; }
    IDevice* device() const {
        return mesh_->get_device(distributed::MeshCoordinate::zero_coordinate(mesh_->shape().dims()));
    }
    void close() { closer_.close(); }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_;
    ScopedMesh closer_;
};

inline uint64_t go_message_tl1_offset() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::GO_MSG);
}

// The whole GO_MSG word of one CCE.
inline uint32_t read_go_word(IDevice* device, uint32_t cce_index) {
    uint32_t go_word = 0;
    mimir_contract::read_cce_sram(
        device,
        logical_cce_core(soc_desc_of(device), cce_index),
        mimir_contract::cce_sram_noc_addr(go_message_tl1_offset()),
        &go_word,
        sizeof(go_word));
    return go_word;
}

inline uint8_t go_signal_of(uint32_t go_word) {
    dev_msgs::go_msg_t go_message =
        MetalContext::instance().hal().get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::go_msg_t>();
    TT_FATAL(go_message.size() == sizeof(uint32_t), "go_msg_t is {} bytes, not one word", go_message.size());
    std::memcpy(go_message.data(), &go_word, sizeof(go_word));
    return go_message.view().signal();
}

inline uint8_t read_go_signal(IDevice* device, uint32_t cce_index) {
    return go_signal_of(read_go_word(device, cce_index));
}

// The hart a CCE test kernel runs on unless its launch names another.
inline constexpr uint32_t kCceKernelHart = 0;

// One kernel of a CCE Program. Kernels on one CCE run on different harts.
struct CceKernelLaunch {
    uint32_t cce_index = 0;
    std::string kernel_path;
    std::vector<uint32_t> runtime_args;
    uint32_t hart = kCceKernelHart;
};

// A launch onto a CCE that is still running a kernel is lost: when that kernel finishes, firmware overwrites the new GO
// with DONE. So every CCE must read RUN_MSG_DONE first.
inline void require_idle_cces(IDevice* device, const std::vector<CceKernelLaunch>& launches) {
    for (const CceKernelLaunch& launch : launches) {
        const uint8_t signal = read_go_signal(device, launch.cce_index);
        TT_FATAL(
            signal == dev_msgs::RUN_MSG_DONE,
            "CCE {} GO_MSG signal is 0x{:02x}, not RUN_MSG_DONE (0x{:02x}); it is still running a kernel or lost "
            "its firmware",
            launch.cce_index,
            signal,
            dev_msgs::RUN_MSG_DONE);
    }
}

// Adds one kernel per launch to `program`, with its runtime args.
inline void add_cce_kernels(IDevice* device, Program& program, const std::vector<CceKernelLaunch>& launches) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(device);
    for (const CceKernelLaunch& launch : launches) {
        const CoreCoord logical_core = logical_cce_core(soc_desc, launch.cce_index);
        const KernelHandle kernel = CreateKernel(
            program,
            launch.kernel_path,
            logical_core,
            DramConfig{.processor = static_cast<DataMovementProcessor>(launch.hart), .noc = NOC::NOC_0});
        SetRuntimeArgs(program, kernel, logical_core, launch.runtime_args);
    }
}

// Checks that every CCE of `launches` is idle, then launches `program` and returns without waiting. The caller ends it
// with slow_dispatch::WaitProgramDone; a blocking slow-dispatch mesh workload would stall every later host access
// instead.
inline void launch_on_idle_cces(IDevice* device, Program& program, const std::vector<CceKernelLaunch>& launches) {
    require_idle_cces(device, launches);
    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/true);
}

// One Program holding every kernel, launched with launch_on_idle_cces.
inline std::unique_ptr<Program> launch_cce_program(IDevice* device, const std::vector<CceKernelLaunch>& launches) {
    std::unique_ptr<Program> program = std::make_unique<Program>(CreateProgram());
    add_cce_kernels(device, *program, launches);
    launch_on_idle_cces(device, *program, launches);
    return program;
}

// Runs one cce_copy_ops kernel per launch, all in one Program with slow dispatch.
inline ::testing::AssertionResult run_cce_copy_ops(
    IDevice* device, const std::vector<CceCopyLaunch>& launches, std::vector<CceCopyResult>& results) {
    results.clear();
    check_tl1_layout();
    if (launches.empty()) {
        return ::testing::AssertionSuccess();
    }

    const metal_SocDescriptor& soc = soc_desc_of(device);
    Program program = CreateProgram();
    for (const CceCopyLaunch& launch : launches) {
        const uint64_t done_offset = tl1_done_block(launch.hart);
        for (const CopyOp& op : launch.table.ops()) {
            const uint64_t bytes = static_cast<uint64_t>(op.num_words) * sizeof(uint32_t);
            for (const CopyEndpoint* endpoint : {&op.source, &op.destination}) {
                if (endpoint->kind != CopyKind::Gddr) {
                    TT_FATAL(
                        endpoint->offset >= tl1_test_base() && endpoint->offset + bytes <= tl1_test_end(),
                        "CCE {} TL1 endpoint [0x{:x}, +{} B) is outside the test area [0x{:x}, 0x{:x})",
                        launch.cce_index,
                        endpoint->offset,
                        bytes,
                        tl1_test_base(),
                        tl1_test_end());
                }
            }
        }

        const CoreCoord logical_core = logical_cce_core(soc, launch.cce_index);
        const KernelHandle kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/cce/cce_copy_ops.cpp",
            logical_core,
            DramConfig{
                .processor = static_cast<DataMovementProcessor>(launch.hart),
                .noc = NOC::NOC_0,
            });
        std::vector<uint32_t> runtime_args(cce_copy_ops_abi::kHeaderWords);
        runtime_args[cce_copy_ops_abi::kArgDoneOffset] = static_cast<uint32_t>(done_offset);
        runtime_args[cce_copy_ops_abi::kArgNumOps] = static_cast<uint32_t>(launch.table.ops().size());
        const std::vector<uint32_t> encoded = launch.table.encode();
        runtime_args.insert(runtime_args.end(), encoded.begin(), encoded.end());
        SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    }

    for (const CceCopyLaunch& launch : launches) {
        write_cce_tl1(device, launch.cce_index, tl1_done_block(launch.hart), {cce_copy_ops_abi::kDonePoison});
    }

    launch_program_and_wait(device, program);

    for (const CceCopyLaunch& launch : launches) {
        const std::vector<uint32_t> done_words = read_cce_tl1(
            device,
            launch.cce_index,
            tl1_done_block(launch.hart),
            static_cast<uint32_t>(cce_copy_ops_abi::kDoneWords * sizeof(uint32_t)));
        CceCopyResult result{};
        result.cce_index = launch.cce_index;
        result.hart = launch.hart;
        result.done = done_words[cce_copy_ops_abi::kDoneMagicWord] == cce_copy_ops_abi::kDoneMagic;
        result.xor_signature = done_words[cce_copy_ops_abi::kDoneXorWord];
        results.push_back(result);
    }
    return ::testing::AssertionSuccess();
}

// Topology and host accessors.
inline bool mimir_local_package_selected() {
    return mimir_contract::configured_soc_desc().get_arch_descriptor().get_grendel_address_space() ==
           tt::umd::GrendelAddressSpace::MimirLocal;
}

struct MimirTopology {
    uint32_t num_mimirs = 0;
    uint32_t num_cces = 0;
    uint32_t num_views = 0;
    uint64_t view_size = 0;
    std::vector<uint32_t> view_mimir;
    std::vector<uint64_t> view_offset;
};

inline MimirTopology topology_from(const metal_SocDescriptor& soc) {
    MimirTopology topology;
    topology.num_mimirs = static_cast<uint32_t>(soc.get_num_dram_channels());
    topology.num_cces = num_cces_from_soc(soc);
    topology.num_views = static_cast<uint32_t>(soc.get_num_dram_views());
    topology.view_size = soc.dram_view_size;
    for (uint32_t view = 0; view < topology.num_views; ++view) {
        topology.view_mimir.push_back(static_cast<uint32_t>(soc.get_channel_for_dram_view(static_cast<int>(view))));
        topology.view_offset.push_back(static_cast<uint64_t>(soc.get_address_offset(static_cast<int>(view))));
    }
    return topology;
}

// For a case that reads Mimir-local addresses (raw register reads, D2D state, Mimir-local address records). Invoke it
// directly in the test body after skip_unless_programmable_dram_package(): GTEST_SKIP returns from its caller.
#define skip_unless_mimir_local_package()                                                                   \
    do {                                                                                                    \
        if (!::tt::tt_metal::mimir_test::mimir_local_package_selected()) {                                  \
            GTEST_SKIP() << "This case reads Mimir-local addresses; the selected package is full_package."; \
        }                                                                                                   \
    } while (0)

struct MimirDramAddressInfo {
    ChipId chip = 0;
    uint32_t view = 0, mimir_index = 0;
    uint64_t view_offset = 0;
    uint64_t chiplet_offset = 0;
    uint64_t flat_address = 0;
    MimirChannelLocation first_channel;
    uint32_t size = 0;

    std::string to_string() const {
        return fmt::format(
            "view {} (M{}, chip {}) view_offset 0x{:x} +{} B: chiplet offset 0x{:x}, flat 0x{:x}, "
            "first channel {} (tile {} slice {})",
            view,
            mimir_index,
            chip,
            view_offset,
            size,
            chiplet_offset,
            flat_address,
            first_channel.instance,
            first_channel.tile,
            first_channel.slice);
    }

    std::string host_string() const {
        return fmt::format(
            "view {} (M{}, chip {}) view_offset 0x{:x} +{} B: chiplet offset 0x{:x}, flat 0x{:x}, "
            "first channel {} (tile {} slice {})",
            view,
            mimir_index,
            chip,
            view_offset,
            size,
            chiplet_offset,
            flat_address,
            first_channel.instance,
            first_channel.tile,
            first_channel.slice);
    }
};

inline MimirDramAddressInfo mimir_dram_address(
    const metal_SocDescriptor& soc, ChipId chip, uint32_t view, uint64_t view_offset, uint32_t size) {
    TT_FATAL(
        view < soc.get_num_dram_views(),
        "DRAM view {} is outside the {} views of this descriptor",
        view,
        soc.get_num_dram_views());
    TT_FATAL(size > 0, "DRAM access size must be positive");

    MimirDramAddressInfo info;
    info.chip = chip;
    info.view = view;
    info.mimir_index = static_cast<uint32_t>(soc.get_channel_for_dram_view(static_cast<int>(view)));
    info.view_offset = view_offset;
    info.chiplet_offset = static_cast<uint64_t>(soc.get_address_offset(static_cast<int>(view))) + view_offset;
    const tt::umd::GrendelAddressWindows windows = tt::umd::mimir_local_address_windows(soc);
    info.flat_address = windows.dram_address(info.mimir_index, info.chiplet_offset);
    info.first_channel = mimir_channel_location(info.chiplet_offset);
    info.size = size;
    return info;
}

inline ::testing::AssertionResult mimir_host_dram_write(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    uint32_t view,
    uint64_t view_offset,
    const std::vector<uint32_t>& data) {
    const uint32_t size = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
    static_cast<void>(mimir_dram_address(soc, chip, view, view_offset, size));
    cluster.write_dram_vec(data.data(), size, chip, static_cast<int>(view), view_offset);
    return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult mimir_host_dram_read(
    const Cluster& cluster,
    const metal_SocDescriptor& soc,
    ChipId chip,
    uint32_t view,
    uint64_t view_offset,
    uint32_t size_bytes,
    std::vector<uint32_t>& data) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "DRAM read size {} is not a multiple of 4", size_bytes);
    static_cast<void>(mimir_dram_address(soc, chip, view, view_offset, size_bytes));
    data.assign(size_bytes / sizeof(uint32_t), 0);
    // No dram_barrier: EmuAxi runs on SimulationChip, whose dram_membar is a no-op.
    cluster.read_dram_vec(data.data(), size_bytes, chip, static_cast<int>(view), view_offset);
    return ::testing::AssertionSuccess();
}

struct MimirAnchor {
    uint32_t view = 0;
    uint64_t view_offset = 0;
    std::string name;
};

// Anchors may overlap.
inline std::vector<MimirAnchor> mimir_dram_anchors(
    const metal_SocDescriptor& soc, uint32_t view, uint64_t unreserved, bool include_reserved_start) {
    const uint64_t gib_count = soc.dram_view_size / kGiB;
    std::vector<MimirAnchor> anchors;
    if (include_reserved_start) {
        anchors.push_back({view, 0, "start"});
    }
    anchors.push_back({view, unreserved, "unres_g0"});
    for (uint64_t gib = 1; gib < gib_count; ++gib) {
        anchors.push_back({view, unreserved + gib * kGiB, fmt::format("unres_g{}", gib)});
    }
    for (uint64_t gib = 1; gib <= gib_count; ++gib) {
        anchors.push_back({view, gib * kGiB - kAnchorBytes, fmt::format("below_g{}", gib)});
    }
    for (uint64_t gib = 1; gib < gib_count; ++gib) {
        anchors.push_back({view, gib * kGiB, fmt::format("above_g{}", gib)});
    }
    if (soc.dram_view_size % kGiB != 0) {
        anchors.push_back({view, soc.dram_view_size - kAnchorBytes, "view_end"});
    }
    return anchors;
}

struct MimirTransferRecord {
    std::string source;
    std::string destination;
    uint64_t requested_address = 0;
    std::vector<std::pair<std::string, uint64_t>> predicted_stages;
    uint64_t destination_local_address = 0;
    uint32_t length = 0;
    uint32_t alignment = 0;
    bool crosses_d2d = false;
    uint32_t seed = 0;

    std::string to_string() const {
        std::string text = fmt::format(
            "transfer {} -> {}: requested 0x{:x}, destination local 0x{:x}, {} B, alignment {}, crosses D2D {}, "
            "seed 0x{:x}",
            source,
            destination,
            requested_address,
            destination_local_address,
            length,
            alignment,
            crosses_d2d,
            seed);
        for (const std::pair<std::string, uint64_t>& stage : predicted_stages) {
            text += fmt::format(", {}: 0x{:x}", stage.first, stage.second);
        }
        return text;
    }
};

inline MimirTransferRecord make_sram_record(
    const metal_SocDescriptor& soc,
    const CceSpaRoutes& issuer_routes,
    std::string source,
    std::string destination,
    uint32_t dest_cce,
    uint64_t landing_offset,
    uint32_t size_bytes,
    uint32_t seed) {
    const AttEndpointPrediction endpoint = att_endpoint_for_remote_sram(dest_cce);
    MimirTransferRecord record;
    record.source = std::move(source);
    record.destination = std::move(destination);
    record.requested_address = mimir_contract::cce_sram_address(issuer_routes, dest_cce, landing_offset);
    record.destination_local_address = host_cce_sram_mla_address(
        tt::umd::mimir_local_address_windows(soc), location_for_cce(dest_cce), landing_offset);
    record.predicted_stages = {
        {"issued SPA", record.requested_address},
        {fmt::format("ATT entry 2 -> endpoint {} (expected entry word 0x{:x})", endpoint.entry_index, endpoint.value),
         endpoint.entry_index},
        {"after SS_BAR1", record.destination_local_address},
    };
    record.length = size_bytes;
    record.alignment = static_cast<uint32_t>(record.requested_address % 64);
    record.crosses_d2d = true;
    record.seed = seed;
    return record;
}

inline void log_transfer(const MimirTransferRecord& record) { log_info(tt::LogTest, "{}", record.to_string()); }

}  // namespace tt::tt_metal::mimir_test
