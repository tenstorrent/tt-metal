// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <numeric>
#include <string>
#include <string_view>

#include "dev_mem_map.h"  // MEM_LOCAL_BASE
#include "hal_types.hpp"
#include "eth_l1_address_map.h"
#include "llrt/hal.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "wormhole/wh_hal.hpp"
#include "impl/context/metal_context.hpp"
#include "hal_1xx_common.hpp"
#include "impl/dispatch/dispatch_settings.hpp"

namespace {

// Wrap enum definitions in anonymous namespace so as to not clash with other archs.
#include "core_config.h"  // MaxProcessorsPerCoreType

}  // namespace

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been
// committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE =
    ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

constexpr static std::uint32_t DRAM_PROFILER_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE;
constexpr static std::uint32_t get_dram_profiler_size(
    [[maybe_unused]] uint32_t profiler_dram_bank_size_per_risc_bytes) {
#if defined(TRACY_ENABLE)
    constexpr std::uint32_t MAX_NUM_UNHARVESTED_TENSIX_CORES = 80;
    constexpr std::uint32_t MAX_NUM_ETH_CORES = 16;
    constexpr std::uint32_t MAX_NUM_CORES = MAX_NUM_UNHARVESTED_TENSIX_CORES + MAX_NUM_ETH_CORES;
    constexpr std::uint32_t NUM_DRAM_CHANNELS = 12;
    constexpr std::uint32_t CEIL_NUM_CORES_PER_DRAM_CHANNEL =
        (MAX_NUM_CORES + NUM_DRAM_CHANNELS - 1) / NUM_DRAM_CHANNELS;
    return (((profiler_dram_bank_size_per_risc_bytes * MaxProcessorsPerCoreType * CEIL_NUM_CORES_PER_DRAM_CHANNEL) +
             DRAM_ALIGNMENT - 1) /
            DRAM_ALIGNMENT) *
           DRAM_ALIGNMENT;
#else
    return 0;
#endif
}

constexpr static std::uint32_t get_dram_unreserved_base(std::uint32_t dram_profiler_size) {
    return DRAM_PROFILER_BASE + dram_profiler_size;
}
constexpr static std::uint32_t get_dram_unreserved_size(std::uint32_t dram_profiler_size) {
    return MEM_DRAM_SIZE - get_dram_unreserved_base(dram_profiler_size);
}

static constexpr float EPS_WHB0 = 1.19209e-7f;
static constexpr float NAN_WHB0 = 7.0040e+19;
static constexpr float INF_WHB0 = 1.7014e+38;

namespace tt::tt_metal {

class HalJitBuildQueryWormhole : public hal_1xx::HalJitBuildQueryBase {
public:
    std::string linker_flags([[maybe_unused]] const Params& params) const override { return ""; }

    std::vector<std::string> link_objs(const Params& params) const override {
        std::vector<std::string> objs;
        if (params.is_fw and params.core_type != HalProgrammableCoreType::ACTIVE_ETH) {
            objs.push_back("runtime/hw/lib/wormhole/tmu-crt0.o");
        }
        if (params.core_type == HalProgrammableCoreType::TENSIX and
            params.processor_class == HalProcessorClassType::DM and params.processor_id == 1) {
            // ncrisc wormhole kernels have an exciting entry sequence
            if (params.is_fw) {
                objs.push_back("runtime/hw/lib/wormhole/wh-iram-trampoline.o");
                objs.push_back("runtime/hw/lib/wormhole/tdma_xmov.o");
            } else {
                objs.push_back("runtime/hw/lib/wormhole/wh-iram-start.o");
            }
        }
        if ((params.core_type == HalProgrammableCoreType::TENSIX and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0) or
            (params.core_type == HalProgrammableCoreType::IDLE_ETH and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0)) {
            // Brisc and Idle Erisc.
            objs.push_back("runtime/hw/lib/wormhole/noc.o");
        }
        objs.push_back("runtime/hw/lib/wormhole/substitutes.o");
        return objs;
    }

    std::vector<std::string> includes(const Params& params) const override {
        std::vector<std::string> includes;

        // Common includes for all core types
        includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/common");
        includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_io");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/noc");
        includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc");
        includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib");

        switch (params.core_type) {
            case HalProgrammableCoreType::TENSIX:
                switch (params.processor_class) {
                    case HalProcessorClassType::COMPUTE:
                        includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api");
                        includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu");
                        break;
                    case HalProcessorClassType::DM: break;
                }
                includes.push_back("tt_metal/hw/firmware/src/tt-1xx");
                break;
            case HalProgrammableCoreType::ACTIVE_ETH: {
                includes.push_back("tt_metal/hw/inc/ethernet");
                break;
            }
            case HalProgrammableCoreType::IDLE_ETH: includes.push_back("tt_metal/hw/firmware/src/tt-1xx"); break;
            default:
                TT_THROW(
                    "Unsupported programmable core type {} to query includes", enchantum::to_string(params.core_type));
        }
        return includes;
    }

    std::vector<std::string> defines(const Params& params) const override {
        auto defines = HalJitBuildQueryBase::defines(params);
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            // Additional defines on Wormhole for active ETH cores
            if (rtoptions.get_erisc_iram_enabled()) {
                defines.push_back("ENABLE_IRAM");
            }
            defines.push_back("COOPERATIVE_ERISC");
        }
        defines.push_back("ARCH_WORMHOLE");
        return defines;
    }

    std::vector<std::string> srcs(const Params& params) const override {
        auto srcs = HalJitBuildQueryBase::srcs(params);
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            if (params.is_fw) {
                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/erisc.cc");
                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/erisc-crt0.cc");
            } else {
                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/erisck.cc");
            }
        }
        return srcs;
    }

    std::string common_flags(const Params& params) const override {
        std::string cflags = params.core_type == HalProgrammableCoreType::TENSIX &&
                                     params.processor_class == HalProcessorClassType::COMPUTE
                                 ? "-mcpu=tt-wh-tensix "
                                 : "-mcpu=tt-wh ";
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            cflags += "-fno-delete-null-pointer-checks ";
        } else if (
            (params.core_type == HalProgrammableCoreType::TENSIX &&
             params.processor_class == HalProcessorClassType::DM) ||
            params.core_type == HalProgrammableCoreType::IDLE_ETH) {
            cflags += "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
        }
        return cflags;
    }

    std::string linker_script(const Params& params) const override {
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        const std::string_view path = "runtime/hw/toolchain/wormhole";
        std::string_view fork = params.is_fw ? "firmware" : "kernel";
        switch (params.core_type) {
            case HalProgrammableCoreType::TENSIX:
                switch (params.processor_class) {
                    case HalProcessorClassType::DM:
                        return fmt::format("{}/{}_{}risc.ld", path, fork, params.processor_id == 0 ? "b" : "nc");
                    case HalProcessorClassType::COMPUTE:
                        return fmt::format("{}/{}_trisc{}.ld", path, fork, params.processor_id);
                }
                break;
            case HalProgrammableCoreType::ACTIVE_ETH:
                if (params.is_fw) {
                    // Firmware is named 'app' in this case.
                    fork = "app";
                }
                return fmt::format(
                    "{}/erisc-b0-{}{}.ld", path, fork, rtoptions.get_erisc_iram_enabled() ? "_iram" : "");
            case HalProgrammableCoreType::IDLE_ETH:
                if (params.processor_id < 2) {
                    return fmt::format("{}/{}_{}ierisc.ld", path, fork, params.processor_id ? "subordinate_" : "");
                }
                break;
            default: break;
        }
        TT_THROW(
            "Invalid processor id {} of processor class {} in programmable core type {}",
            params.processor_id,
            enchantum::to_string(params.processor_class),
            enchantum::to_string(params.core_type));
    }

    bool firmware_is_kernel_object(const Params&) const override { return false; }
};

void Hal::initialize_wh(bool is_base_routing_fw_enabled, std::uint32_t profiler_dram_bank_size_per_risc_bytes) {
    using namespace wormhole;
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    HalCoreInfoType tensix_mem_map = wormhole::create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = wormhole::create_active_eth_mem_map(is_base_routing_fw_enabled);
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = wormhole::create_idle_eth_mem_map();
    this->core_info_.push_back(idle_eth_mem_map);

    this->dram_bases_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_sizes_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::BARRIER)] = DRAM_BARRIER_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::BARRIER)] = DRAM_BARRIER_SIZE;
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::PROFILER)] = DRAM_PROFILER_BASE;
    const std::uint32_t dram_profiler_size = get_dram_profiler_size(profiler_dram_bank_size_per_risc_bytes);
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::PROFILER)] = dram_profiler_size;
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::UNRESERVED)] =
        get_dram_unreserved_base(dram_profiler_size);
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::UNRESERVED)] =
        get_dram_unreserved_size(dram_profiler_size);

    this->mem_alignments_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::L1)] = L1_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::DRAM)] = DRAM_ALIGNMENT;
    this->mem_alignments_[static_cast<std::size_t>(HalMemType::HOST)] = PCIE_ALIGNMENT;

    this->mem_read_alignments_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_read_alignments_[static_cast<std::size_t>(HalMemType::L1)] = NOC_L1_READ_ALIGNMENT_BYTES;
    this->mem_read_alignments_[static_cast<std::size_t>(HalMemType::DRAM)] = NOC_DRAM_READ_ALIGNMENT_BYTES;
    this->mem_read_alignments_[static_cast<std::size_t>(HalMemType::HOST)] = NOC_PCIE_READ_ALIGNMENT_BYTES;

    this->mem_write_alignments_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_write_alignments_[static_cast<std::size_t>(HalMemType::L1)] = NOC_L1_WRITE_ALIGNMENT_BYTES;
    this->mem_write_alignments_[static_cast<std::size_t>(HalMemType::DRAM)] = NOC_DRAM_WRITE_ALIGNMENT_BYTES;
    this->mem_write_alignments_[static_cast<std::size_t>(HalMemType::HOST)] = NOC_PCIE_WRITE_ALIGNMENT_BYTES;

    this->mem_alignments_with_pcie_.resize(static_cast<std::size_t>(HalMemType::COUNT));
    this->mem_alignments_with_pcie_[static_cast<std::size_t>(HalMemType::L1)] = std::lcm(L1_ALIGNMENT, PCIE_ALIGNMENT);
    this->mem_alignments_with_pcie_[static_cast<std::size_t>(HalMemType::DRAM)] =
        std::lcm(DRAM_ALIGNMENT, PCIE_ALIGNMENT);
    this->mem_alignments_with_pcie_[static_cast<std::size_t>(HalMemType::HOST)] =
        std::lcm(PCIE_ALIGNMENT, PCIE_ALIGNMENT);

    this->relocate_func_ = [](uint64_t addr, uint64_t local_init_addr, bool) {
        if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            // Move addresses in the local memory range to l1 (copied by kernel)
            return (addr & ~MEM_LOCAL_BASE) + local_init_addr;
        } else if ((addr & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
            // Move addresses in the NCRISC memory range to l1 (copied by kernel)
            return (addr & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE_SCRATCH;
        }

        // No relocation needed
        return addr;
    };

    this->erisc_iram_relocate_func_ = [](uint64_t addr) {
        if (addr == static_cast<uint32_t>(eth_iram_mem::address_map::ERISC_IRAM_BASE)) {
            // IRAM enabled program starts from ERISC_IRAM_BASE. This relocation is for where to put the program.
            // At first the program is placed on ERISC_IRAM_BASE, then erisc.cc copies to local IRAM.
            return (uint64_t)eth_l1_mem::address_map::KERNEL_BASE;
        }
        return addr;
    };

    this->valid_reg_addr_func_ = [](uint32_t addr) {
        return (
            ((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            ((addr >= NOC0_REGS_START_ADDR) && (addr < NOC0_REGS_START_ADDR + 0x1000)) ||
            ((addr >= NOC1_REGS_START_ADDR) && (addr < NOC1_REGS_START_ADDR + 0x1000)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0));
    };

    this->noc_xy_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_ENCODING(x, y); };
    this->noc_multicast_encoding_func_ = [](uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) {
        return NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end);
    };
    this->noc_xy_pcie64_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_PCIE_ENCODING(x, y) >> 32; };
    this->noc_mcast_addr_start_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_X(addr); };
    this->noc_mcast_addr_start_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_Y(addr); };
    this->noc_mcast_addr_end_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_X(addr); };
    this->noc_mcast_addr_end_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_Y(addr); };
    this->noc_ucast_addr_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_X(addr); };
    this->noc_ucast_addr_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_Y(addr); };
    this->noc_local_addr_func_ = [](uint64_t addr) -> uint64_t { return NOC_LOCAL_ADDR(addr); };

    this->eth_fw_arg_addr_func_ = [&](int, uint32_t) -> uint32_t { return 0; };

    this->device_features_func_ = [](DispatchFeature feature) -> bool {
        switch (feature) {
            case DispatchFeature::ETH_MAILBOX_API:
            case DispatchFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER: return false;
            case DispatchFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER:
            case DispatchFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER: return true;
            default: TT_THROW("Invalid Wormhole dispatch feature {}", static_cast<int>(feature));
        }
    };

    this->max_processors_per_core_ = MaxProcessorsPerCoreType;
    this->num_nocs_ = NUM_NOCS;
    this->noc_node_id_ = NOC_NODE_ID;
    this->noc_node_id_mask_ = NOC_NODE_ID_MASK;
    this->noc_addr_node_id_bits_ = NOC_ADDR_NODE_ID_BITS;
    this->noc_encoding_reg_ = COORDINATE_VIRTUALIZATION_ENABLED ? NOC_CFG(NOC_ID_LOGICAL) : NOC_NODE_ID;
    this->noc_coord_reg_offset_ = NOC_COORD_REG_OFFSET;
    this->noc_overlay_start_addr_ = NOC_OVERLAY_START_ADDR;
    this->noc_stream_reg_space_size_ = NOC_STREAM_REG_SPACE_SIZE;
    this->noc_stream_remote_dest_buf_size_reg_index_ = STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX;
    this->noc_stream_remote_dest_buf_start_reg_index_ = STREAM_REMOTE_DEST_BUF_START_REG_INDEX;
    this->noc_stream_remote_dest_buf_space_available_reg_index_ = STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX;
    this->noc_stream_remote_dest_buf_space_available_update_reg_index_ =
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX;
    this->coordinate_virtualization_enabled_ = COORDINATE_VIRTUALIZATION_ENABLED;
    this->virtual_worker_start_x_ = VIRTUAL_TENSIX_START_X;
    this->virtual_worker_start_y_ = VIRTUAL_TENSIX_START_Y;
    this->eth_fw_is_cooperative_ = true;
    this->virtualized_core_types_ = {dev_msgs::AddressableCoreType::TENSIX, dev_msgs::AddressableCoreType::ETH};
    this->tensix_harvest_axis_ = static_cast<HalTensixHarvestAxis>(tensix_harvest_axis);

    this->eps_ = EPS_WHB0;
    this->nan_ = NAN_WHB0;
    this->inf_ = INF_WHB0;

    // PCIe address range for Wormhole. Includes the mapping through the outbound iATU. See
    // https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/PCIExpressTile for more details.
    this->pcie_addr_lower_bound_ = 0x8'0000'0000ULL;
    this->pcie_addr_upper_bound_ = 0x8'FFFE'0000ULL - 1ULL;
    this->supports_64_bit_pcie_addressing_ = false;

    this->noc_x_id_translate_table_ = {
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_3)};

    this->noc_y_id_translate_table_ = {
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_3)};

    this->jit_build_query_ = std::make_unique<HalJitBuildQueryWormhole>();

    this->set_iram_text_size_func_ = [](dev_msgs::launch_msg_t::View launch_msg,
                                        HalProgrammableCoreType programmable_core_type,
                                        HalProcessorClassType processor_class,
                                        uint32_t processor_type_idx,
                                        uint32_t iram_text_size) {
        // Only NCRISC on Wormhole needs to set the field ncrisc_kernel_size16 in launch message.
        if (programmable_core_type == HalProgrammableCoreType::TENSIX && processor_class == HalProcessorClassType::DM &&
            processor_type_idx == 1) {
            launch_msg.kernel_config().ncrisc_kernel_size16() = (iram_text_size + 15) >> 4;
        }
    };

    this->verify_eth_fw_version_func_ = [](tt::umd::semver_t /*eth_fw_version*/) {
        // No checks
        return true;
    };

    this->max_pinned_memory_count_ = 12;
    this->total_pinned_memory_size_ =
        4ULL * 1024ULL * 1024ULL * 1024ULL - static_cast<uint64_t>(tt::tt_metal::DispatchSettings::MAX_HUGEPAGE_SIZE);
}

}  // namespace tt::tt_metal
