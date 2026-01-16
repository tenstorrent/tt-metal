// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <numeric>
#include <string>
#include <string_view>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/utils/semver.hpp>

#include "blackhole/bh_hal.hpp"
#include "dev_mem_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "hal_1xx_common.hpp"

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
    constexpr std::uint32_t MAX_NUM_UNHARVESTED_TENSIX_CORES = 140;
    constexpr std::uint32_t MAX_NUM_ETH_CORES = 14;
    constexpr std::uint32_t MAX_NUM_CORES = MAX_NUM_UNHARVESTED_TENSIX_CORES + MAX_NUM_ETH_CORES;
    constexpr std::uint32_t NUM_DRAM_CHANNELS = 8;
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

static constexpr float EPS_BH = 1.19209e-7f;
static constexpr float NAN_BH = 7.0040e+19;
static constexpr float INF_BH = 1.7014e+38;

namespace tt::tt_metal {

class HalJitBuildQueryBlackHole : public hal_1xx::HalJitBuildQueryBase {
private:
    bool enable_2_erisc_mode_;

public:
    HalJitBuildQueryBlackHole(bool enable_2_erisc_mode) : enable_2_erisc_mode_(enable_2_erisc_mode) {}

    std::string linker_flags([[maybe_unused]] const Params& params) const override { return ""; }

    std::vector<std::string> link_objs(const Params& params) const override {
        std::vector<std::string> objs;
        if (params.is_fw) {
            // Needed to setup gp, sp, etc. for all processors which are launched with assert/deassert PC method
            // For 2 erisc, erisc0 is launched from base firmware so it's not needed
            if (!(params.core_type == HalProgrammableCoreType::ACTIVE_ETH && params.processor_id == 0 &&
                  enable_2_erisc_mode_)) {
                objs.push_back("runtime/hw/lib/blackhole/tmu-crt0.o");
            }
        }
        if ((params.core_type == HalProgrammableCoreType::TENSIX and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0) or
            (params.core_type == HalProgrammableCoreType::IDLE_ETH and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0)) {
            // Brisc and Idle Erisc.
            objs.push_back("runtime/hw/lib/blackhole/noc.o");
        }
        objs.push_back("runtime/hw/lib/blackhole/substitutes.o");
        return objs;
    }

    std::vector<std::string> includes(const Params& params) const override {
        std::vector<std::string> includes;

        // Common includes for all core types
        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/common");
        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_io");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines");
        includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/noc");
        includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc");
        includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib");

        switch (params.core_type) {
            case HalProgrammableCoreType::TENSIX:
                switch (params.processor_class) {
                    case HalProcessorClassType::COMPUTE:
                        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_api");
                        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu");
                        break;
                    case HalProcessorClassType::DM: break;
                }
                break;
            case HalProgrammableCoreType::ACTIVE_ETH: {
                includes.push_back("tt_metal/hw/inc/ethernet");
                break;
            }
            case HalProgrammableCoreType::IDLE_ETH: break;
            default:
                TT_THROW(
                    "Unsupported programmable core type {} to query includes", enchantum::to_string(params.core_type));
        }
        includes.push_back("tt_metal/hw/firmware/src/tt-1xx");
        return includes;
    }

    std::vector<std::string> defines(const Params& params) const override {
        auto defines = HalJitBuildQueryBase::defines(params);
        defines.push_back("ARCH_BLACKHOLE");
        // Push back the physical erisc id
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            if (enable_2_erisc_mode_) {
                defines.push_back("ENABLE_2_ERISC_MODE");
                defines.push_back("PHYSICAL_AERISC_ID=" + std::to_string(params.processor_id));
            } else {
                defines.push_back("PHYSICAL_AERISC_ID=1");
            }
        }
        return defines;
    }

    std::vector<std::string> srcs(const Params& params) const override {
        auto srcs = HalJitBuildQueryBase::srcs(params);
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            switch (params.processor_id) {
                case 0:
                    if (params.is_fw) {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc");
                        if (enable_2_erisc_mode_) {
                            // not tmu-crt0
                            srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisc-crt0.cc");
                        }
                    } else {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc");
                    }
                    break;
                case 1:
                    if (params.is_fw) {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/subordinate_erisc.cc");
                    } else {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc");
                    }
                    break;
                default: TT_THROW("Unkown processor id {}", params.processor_id);
            }
        }
        return srcs;
    }

    std::string common_flags(const Params& params) const override {
        std::string cflags = params.core_type == HalProgrammableCoreType::TENSIX &&
                                     params.processor_class == HalProcessorClassType::COMPUTE
                                 ? "-mcpu=tt-bh-tensix "
                                 : "-mcpu=tt-bh ";
        cflags += "-mno-tt-tensix-optimize-replay ";
        if (!(params.core_type == HalProgrammableCoreType::TENSIX &&
              params.processor_class == HalProcessorClassType::COMPUTE)) {
            cflags += "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
        }
        // Unlike other core types, the stack on erisc0 is not dynamic because it's setup by base firmware.
        // Trigger an error for kernels which may exceed the static stack usage to prevent difficult to debug issues
        // 2048 B = stack size taken from the base firmware
        // 64 B = Reserved for base firmware usage
        // 72 B = Approx. stack usage at the time the kernel is launched
        // 2048 B - 64 B - 72 B = 1912 B free for kernel
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH && params.processor_id == 0 &&
            enable_2_erisc_mode_) {
            cflags += "-Werror=stack-usage=1912 ";
        }
        return cflags;
    }

    std::string linker_script(const Params& params) const override {
        const std::string_view fork = params.is_fw ? "firmware" : "kernel";
        const std::string_view path = "runtime/hw/toolchain/blackhole";
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
                if (params.processor_id < 2) {
                    const char* prefix = "";
                    if (params.processor_id) {
                        prefix = "subordinate_";
                    } else if (enable_2_erisc_mode_) {
                        prefix = "main_";
                    }
                    return fmt::format("{}/{}_{}aerisc.ld", path, fork, prefix);
                }
                break;
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

    std::string target_name(const Params& params) const override {
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            // build.cpp used to distinguish "active_erisc" and "erisc" and use
            // that to determine what object files to link.
            // This is no longer necessary, but only to keep the target names unchanged.
            return params.processor_id == 0 ? "active_erisc" : "subordinate_active_erisc";
        }
        return HalJitBuildQueryBase::target_name(params);
    }
};

void Hal::initialize_bh(bool enable_2_erisc_mode, std::uint32_t profiler_dram_bank_size_per_risc_bytes) {
    using namespace blackhole;
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    HalCoreInfoType tensix_mem_map = blackhole::create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = blackhole::create_active_eth_mem_map(enable_2_erisc_mode);
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = blackhole::create_idle_eth_mem_map();
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

    this->relocate_func_ = [](uint64_t addr, uint64_t local_init_addr, bool has_shared_local_mem) {
        if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            // Move addresses in the local memory range to l1 (copied by kernel)
            // For firmware with base fw, __ldm_data is already offset by base fw.
            // So we need to undo that offset here to get the correct relocation address
            // for copying by the kernel to local memory.
            if (has_shared_local_mem) {
                addr -= MEM_ERISC_BASE_FW_LOCAL_SIZE;
            }
            return (addr & ~MEM_LOCAL_BASE) + local_init_addr;
        }

        // Note: Blackhole does not have IRAM

        // No relocation needed
        return addr;
    };

    this->erisc_iram_relocate_func_ = [](uint64_t addr) { return addr; };

    this->valid_reg_addr_func_ = [](uint32_t addr) {
        return (
            ((addr >= NOC_OVERLAY_START_ADDR) &&
             (addr < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) ||
            ((addr >= NOC0_REGS_START_ADDR) && (addr < NOC0_REGS_START_ADDR + 0x1000)) ||
            ((addr >= NOC1_REGS_START_ADDR) && (addr < NOC1_REGS_START_ADDR + 0x1000)) ||
            (addr == RISCV_DEBUG_REG_SOFT_RESET_0) ||
            (addr == IERISC_RESET_PC || addr == SUBORDINATE_IERISC_RESET_PC));  // used to program start addr for eth FW
    };

    this->noc_xy_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_ENCODING(x, y); };
    this->noc_xy_pcie64_encoding_func_ = [](uint32_t x, uint32_t y) {
        // Use non-iATU range for 64-bit inputs.
        return NOC_XY_ENCODING(x, y);
    };
    this->noc_multicast_encoding_func_ = [](uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) {
        return NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end);
    };
    this->noc_mcast_addr_start_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_X(addr); };
    this->noc_mcast_addr_start_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_START_Y(addr); };
    this->noc_mcast_addr_end_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_X(addr); };
    this->noc_mcast_addr_end_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_MCAST_ADDR_END_Y(addr); };
    this->noc_ucast_addr_x_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_X(addr); };
    this->noc_ucast_addr_y_func_ = [](uint64_t addr) -> uint64_t { return NOC_UNICAST_ADDR_Y(addr); };
    this->noc_local_addr_func_ = [](uint64_t addr) -> uint64_t { return NOC_LOCAL_ADDR(addr); };

    this->eth_fw_arg_addr_func_ = [&](int mailbox_index, uint32_t arg_index) -> uint32_t {
        // +1 because of the message
        uint32_t mailbox_base =
            MEM_SYSENG_ETH_MAILBOX_ADDR + (mailbox_index * (MEM_SYSENG_ETH_MAILBOX_NUM_ARGS + 1) * sizeof(uint32_t));
        return mailbox_base + offsetof(blackhole::EthFwMailbox, arg) +
               (arg_index * sizeof(((blackhole::EthFwMailbox*)nullptr)->arg[0]));
    };

    this->device_features_func_ = [](DispatchFeature feature) -> bool {
        switch (feature) {
            case DispatchFeature::ETH_MAILBOX_API:
            case DispatchFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER:
            case DispatchFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER:
            case DispatchFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER: return true;
            default: TT_THROW("Invalid Blackhole dispatch feature {}", static_cast<int>(feature));
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
    this->eth_fw_is_cooperative_ = false;
    this->virtualized_core_types_ = {
        dev_msgs::AddressableCoreType::TENSIX,
        dev_msgs::AddressableCoreType::ETH,
        dev_msgs::AddressableCoreType::PCIE,
        dev_msgs::AddressableCoreType::DRAM};
    this->tensix_harvest_axis_ = static_cast<HalTensixHarvestAxis>(tensix_harvest_axis);

    this->eps_ = EPS_BH;
    this->nan_ = NAN_BH;
    this->inf_ = INF_BH;

    // PCIe address range for Blackhole. Includes both the direct mapping to the IOMMU address range, as well as the
    // mapping through the outbound iATU. See
    // https://github.com/tenstorrent/tt-isa-documentation/tree/main/BlackholeA0/PCIExpressTile for more details.
    this->pcie_addr_lower_bound_ = 0x0000000000000000ULL;
    this->pcie_addr_upper_bound_ = 0x13FF'FFFF'FFFF'FFFFULL;
    this->supports_64_bit_pcie_addressing_ = true;

    this->noc_x_id_translate_table_ = {
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_3),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_4),
        NOC_CFG(NOC_X_ID_TRANSLATE_TABLE_5)};

    this->noc_y_id_translate_table_ = {
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_0),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_1),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_2),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_3),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_4),
        NOC_CFG(NOC_Y_ID_TRANSLATE_TABLE_5)};

    this->jit_build_query_ = std::make_unique<HalJitBuildQueryBlackHole>(enable_2_erisc_mode);

    this->verify_eth_fw_version_func_ = [=](tt::umd::semver_t fw_version) {
        if (enable_2_erisc_mode) {
            tt::umd::semver_t min_version(1, 7, 0);
            if (!(fw_version >= min_version)) {
                log_warning(
                    tt::LogLLRuntime,
                    "Blackhole multi erisc mode requires ethernet firmware version {} or higher, but detected version "
                    "{}. Automatically falling back to single erisc mode for compatibility.",
                    min_version.to_string(),
                    fw_version.to_string());
                return false;
            }
        }
        return true;
    };

    this->max_pinned_memory_count_ = std::numeric_limits<size_t>::max();
    this->total_pinned_memory_size_ = std::numeric_limits<size_t>::max();
}

}  // namespace tt::tt_metal
