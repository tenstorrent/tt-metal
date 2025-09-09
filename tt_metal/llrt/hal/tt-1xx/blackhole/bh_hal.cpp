// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_msgs.h"
#include <cstddef>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <numeric>
#include <string>

#include "blackhole/bh_hal.hpp"
#include "dev_mem_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_overlay_parameters.h"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "hal_1xx_common.hpp"

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been
// committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_BARRIER_SIZE =
    ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

constexpr static std::uint32_t DRAM_PROFILER_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE;
#if defined(TRACY_ENABLE)
constexpr static std::uint32_t MAX_NUM_UNHARVESTED_TENSIX_CORES = 140;
constexpr static std::uint32_t MAX_NUM_ETH_CORES = 14;
constexpr static std::uint32_t MAX_NUM_CORES = MAX_NUM_UNHARVESTED_TENSIX_CORES + MAX_NUM_ETH_CORES;
constexpr static std::uint32_t NUM_DRAM_CHANNELS = 8;
constexpr static std::uint32_t CEIL_NUM_CORES_PER_DRAM_CHANNEL =
    (MAX_NUM_CORES + NUM_DRAM_CHANNELS - 1) / NUM_DRAM_CHANNELS;
constexpr static std::uint32_t DRAM_PROFILER_SIZE =
    (((PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * CEIL_NUM_CORES_PER_DRAM_CHANNEL) +
      DRAM_ALIGNMENT - 1) /
     DRAM_ALIGNMENT) *
    DRAM_ALIGNMENT;
#else
constexpr static std::uint32_t DRAM_PROFILER_SIZE = 0;
#endif

constexpr static std::uint32_t DRAM_UNRESERVED_BASE = DRAM_PROFILER_BASE + DRAM_PROFILER_SIZE;
constexpr static std::uint32_t DRAM_UNRESERVED_SIZE = MEM_DRAM_SIZE - DRAM_UNRESERVED_BASE;

static constexpr float EPS_BH = 1.19209e-7f;
static constexpr float NAN_BH = 7.0040e+19;
static constexpr float INF_BH = 1.7014e+38;

namespace tt {

namespace tt_metal {

namespace blackhole {

// Wrap enum definitions in arch-specific namespace so as to not clash with other archs.
#include "core_config.h"  // ProgrammableCoreType

}

class HalJitBuildQueryBlackHole : public hal_1xx::HalJitBuildQueryBase {
public:
    std::vector<std::string> link_objs(const Params& params) const override {
        std::vector<std::string> objs;
        if (params.is_fw) {
            objs.push_back("runtime/hw/lib/blackhole/tmu-crt0.o");
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
        includes.push_back("tt_metal/hw/inc/blackhole");
        includes.push_back("tt_metal/hw/inc/blackhole/blackhole_defines");
        includes.push_back("tt_metal/hw/inc/blackhole/noc");
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
        return defines;
    }

    std::vector<std::string> srcs(const Params& params) const override {
        auto srcs = HalJitBuildQueryBase::srcs(params);
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            if (params.is_fw) {
                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc");
            } else {
                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc");
            }
        }
        return srcs;
    }

    std::string common_flags(const Params& params) const override {
        std::string cflags = "-mcpu=tt-bh -fno-rvtt-sfpu-replay ";
        if (!(params.core_type == HalProgrammableCoreType::TENSIX &&
              params.processor_class == HalProcessorClassType::COMPUTE)) {
            cflags += "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
        }
        return cflags;
    }

    std::string linker_script(const Params& params) const override {
        switch (params.core_type) {
            case HalProgrammableCoreType::TENSIX:
                switch (params.processor_class) {
                    case HalProcessorClassType::DM: {
                        return fmt::format(
                            "runtime/hw/toolchain/blackhole/{}_{}risc.ld",
                            params.is_fw ? "firmware" : "kernel",
                            params.processor_id == 0 ? "b" : "nc");
                    }
                    case HalProcessorClassType::COMPUTE:
                        return fmt::format(
                            "runtime/hw/toolchain/blackhole/{}_trisc{}.ld",
                            params.is_fw ? "firmware" : "kernel",
                            params.processor_id);
                }
                break;
            case HalProgrammableCoreType::ACTIVE_ETH:
                return params.is_fw ? "runtime/hw/toolchain/blackhole/firmware_aerisc.ld"
                                    : "runtime/hw/toolchain/blackhole/kernel_aerisc.ld";
            case HalProgrammableCoreType::IDLE_ETH:
                switch (params.processor_id) {
                    case 0:
                        return params.is_fw ? "runtime/hw/toolchain/blackhole/firmware_ierisc.ld"
                                            : "runtime/hw/toolchain/blackhole/kernel_ierisc.ld";
                    case 1:
                        return params.is_fw ? "runtime/hw/toolchain/blackhole/firmware_subordinate_ierisc.ld"
                                            : "runtime/hw/toolchain/blackhole/kernel_subordinate_ierisc.ld";
                }
            default:
                TT_THROW(
                    "Unsupported programmable core type {} to query linker script",
                    enchantum::to_string(params.core_type));
        }
        TT_THROW(
            "Invalid processor id {} of processor class {} in programmable core type {}",
            params.processor_id,
            enchantum::to_string(params.processor_class),
            enchantum::to_string(params.core_type));
    }

    std::string target_name(const Params& params) const override {
        if (params.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
            // build.cpp used to distinguish "active_erisc" and "erisc" and use
            // that to determine what object files to link.
            // This is no longer necessary, but only to keep the target names unchanged.
            return "active_erisc";
        }
        return HalJitBuildQueryBase::target_name(params);
    }
};

void Hal::initialize_bh() {
    using namespace blackhole;
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    static_assert(MaxProcessorsPerCoreType <= MAX_RISCV_PER_CORE);

    HalCoreInfoType tensix_mem_map = blackhole::create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = blackhole::create_active_eth_mem_map();
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = blackhole::create_idle_eth_mem_map();
    this->core_info_.push_back(idle_eth_mem_map);

    this->dram_bases_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_sizes_.resize(static_cast<std::size_t>(HalDramMemAddrType::COUNT));
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::BARRIER)] = DRAM_BARRIER_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::BARRIER)] = DRAM_BARRIER_SIZE;
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::PROFILER)] = DRAM_PROFILER_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::PROFILER)] = DRAM_PROFILER_SIZE;
    this->dram_bases_[static_cast<std::size_t>(HalDramMemAddrType::UNRESERVED)] = DRAM_UNRESERVED_BASE;
    this->dram_sizes_[static_cast<std::size_t>(HalDramMemAddrType::UNRESERVED)] = DRAM_UNRESERVED_SIZE;

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
            // For RISC0 we have a shared local memory with base firmware so offset by that
            // if (has_shared_local_mem) {
            //     addr -= MEM_ERISC_BASE_FW_LOCAL_SIZE;
            // }
            // Move addresses in the local memory range to l1 (copied by kernel)
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
               (arg_index * sizeof(((blackhole::EthFwMailbox*)0)->arg[0]));
    };

    this->device_features_func_ = [](DispatchFeature feature) -> bool {
        switch (feature) {
            case DispatchFeature::ETH_MAILBOX_API: return true;
            // Active eth kernel config buffer is not needed until 2 ERISCs
            case DispatchFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER: return false;
            case DispatchFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER: return true;
            case DispatchFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER: return true;
            default: TT_THROW("Invalid Blackhole dispatch feature {}", static_cast<int>(feature));
        }
    };

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
    this->intermesh_eth_links_enabled_ = false;  // Intermesh routing is not enabled on Blackhole
    this->virtualized_core_types_ = {
        AddressableCoreType::TENSIX, AddressableCoreType::ETH, AddressableCoreType::PCIE, AddressableCoreType::DRAM};
    this->tensix_harvest_axis_ = static_cast<HalTensixHarvestAxis>(tensix_harvest_axis);

    this->eps_ = EPS_BH;
    this->nan_ = NAN_BH;
    this->inf_ = INF_BH;

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

    this->jit_build_query_ = std::make_unique<HalJitBuildQueryBlackHole>();
}

}  // namespace tt_metal
}  // namespace tt
