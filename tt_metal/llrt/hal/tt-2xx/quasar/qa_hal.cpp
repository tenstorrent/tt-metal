// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <numeric>
#include <string>

#include "quasar/qa_hal.hpp"
#include "dev_mem_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "hal_2xx_common.hpp"

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

static constexpr float EPS_QA = 1.19209e-7f;  // TODO: verify
static constexpr float NAN_QA = 7.0040e+19;   // TODO: verify
static constexpr float INF_QA = 1.7014e+38;   // TODO: verify

namespace tt::tt_metal {

class HalJitBuildQueryQuasar : public hal_2xx::HalJitBuildQueryBase {
public:
    std::string linker_flags(const Params& params) const override {
        std::string flags;
        if (params.is_fw) {
            flags += fmt::format("-Wl,--defsym=__fw_text={} ", MEM_DM_FIRMWARE_BASE);
            flags += fmt::format("-Wl,--defsym=__text_size={} ", MEM_DM_FIRMWARE_SIZE);
            flags += fmt::format("-Wl,--defsym=__fw_data={} ", MEM_DM_GLOBAL_BASE);
            flags += fmt::format("-Wl,--defsym=__data_size={} ", MEM_DM_GLOBAL_SIZE);
            flags += fmt::format("-Wl,--defsym=__fw_tls={} ", MEM_DM_LOCAL_BASE);
            flags += fmt::format("-Wl,--defsym=__tls_size={} ", MEM_DM_LOCAL_SIZE);
            flags += fmt::format("-Wl,--defsym=__min_stack={} ", MEM_DM_STACK_MIN_SIZE);
            flags += fmt::format("-Wl,--defsym=__local_base={} ", MEM_DM_LOCAL_BASE);
            flags += fmt::format("-Wl,--defsym=__local_stride={} ", MEM_DM_LOCAL_SIZE);
        } else {
            flags += fmt::format("-Wl,--defsym=__kn_text={} ", MEM_DM_KERNEL_BASE);  // this is for legacy kernels
            flags += fmt::format("-Wl,--defsym=__text_size={} ", MEM_DM_KERNEL_SIZE);
            flags += fmt::format(
                "-Wl,--defsym=__fw_data={} ", MEM_DM_GLOBAL_BASE + (params.processor_id * MEM_DM_GLOBAL_SIZE));
            flags += fmt::format("-Wl,--defsym=__data_size={} ", MEM_DM_GLOBAL_SIZE);
            flags +=
                fmt::format("-Wl,--defsym=__fw_tls={} ", MEM_DM_LOCAL_BASE + (params.processor_id * MEM_DM_LOCAL_SIZE));
            flags += fmt::format("-Wl,--defsym=__tls_size={} ", MEM_DM_LOCAL_SIZE);
            flags += fmt::format("-Wl,--defsym=__min_stack={} ", MEM_DM_STACK_MIN_SIZE);
        }
        return flags;
    }

    std::vector<std::string> link_objs(const Params& params) const override {
        std::vector<std::string> objs;
        std::string_view cpu = params.processor_class == HalProcessorClassType::DM ? "tt-qsr64" : "tt-qsr-32";
        std::string_view dir = "runtime/hw/lib/quasar";
        objs.push_back(fmt::format("{}/{}-crt0-tls.o", dir, cpu));
        if (params.is_fw) {
            objs.push_back(fmt::format("{}/{}-crt0.o", dir, cpu));
        }
        if ((params.core_type == HalProgrammableCoreType::TENSIX and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0) or
            (params.core_type == HalProgrammableCoreType::IDLE_ETH and
             params.processor_class == HalProcessorClassType::DM and params.processor_id == 0)) {
            // Brisc and Idle Erisc.
            objs.push_back(fmt::format("{}/{}-noc.o", dir, cpu));
        }
        objs.push_back(fmt::format("{}/{}-substitutes.o", dir, cpu));
        return objs;
    }

    std::vector<std::string> includes(const Params& params) const override {
        std::vector<std::string> includes;

        // Common includes for all core types
        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/common");
        includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_io");
        includes.push_back("tt_metal/hw/inc/internal/tt-2xx");
        includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar");
        includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines");
        includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/noc");
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
        includes.push_back("tt_metal/hw/firmware/src/tt-2xx");
        return includes;
    }

    std::vector<std::string> defines(const Params& params) const override {
        auto defines = HalJitBuildQueryBase::defines(params);
        defines.push_back("ARCH_QUASAR");
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
        // TODO: Use correct tt-qsr cpu options #32893
        std::string cflags =
            params.processor_class == HalProcessorClassType::DM ? "-mcpu=tt-qsr64-rocc " : "-mcpu=tt-qsr32-tensixbh ";
        cflags += "-mno-tt-tensix-optimize-replay ";
        cflags += "-fno-extern-tls-init ";
        cflags += "-ftls-model=local-exec ";
        if (!(params.core_type == HalProgrammableCoreType::TENSIX &&
              params.processor_class == HalProcessorClassType::COMPUTE)) {
            cflags += "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
        }
        return cflags;
    }

    bool firmware_is_kernel_object(const Params&) const override { return true; }
    std::string linker_script(const Params& params) const override {
        switch (params.core_type) {
            case HalProgrammableCoreType::TENSIX:
                switch (params.processor_class) {
                    case HalProcessorClassType::DM: {
                        return fmt::format(
                            "runtime/hw/toolchain/quasar/{}_dm{}.ld",
                            params.is_fw ? "firmware" : "kernel",
                            params.is_fw ? "" : "_lgc");  // hard code legacy kernels for now
                    }
                    case HalProcessorClassType::COMPUTE:
                        return fmt::format(
                            "runtime/hw/toolchain/quasar/{}_trisc{}.ld",
                            params.is_fw ? "firmware" : "kernel",
                            params.processor_id);
                }
                break;
            case HalProgrammableCoreType::ACTIVE_ETH:
                return params.is_fw ? "runtime/hw/toolchain/quasar/firmware_aerisc.ld"
                                    : "runtime/hw/toolchain/quasar/kernel_aerisc.ld";
            case HalProgrammableCoreType::IDLE_ETH:
                switch (params.processor_id) {
                    case 0:
                        return params.is_fw ? "runtime/hw/toolchain/quasar/firmware_ierisc.ld"
                                            : "runtime/hw/toolchain/quasar/kernel_ierisc.ld";
                    case 1:
                        return params.is_fw ? "runtime/hw/toolchain/quasar/firmware_subordinate_ierisc.ld"
                                            : "runtime/hw/toolchain/quasar/kernel_subordinate_ierisc.ld";
                    default: TT_THROW("Invalid processor id {}", params.processor_id);
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

void Hal::initialize_qa(std::uint32_t profiler_dram_bank_size_per_risc_bytes) {
    using namespace quasar;
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(
        static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

    HalCoreInfoType tensix_mem_map = quasar::create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = quasar::create_active_eth_mem_map();
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = quasar::create_idle_eth_mem_map();
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

    this->relocate_func_ = [](uint64_t addr, uint64_t local_init_addr, bool /*has_shared_local_mem*/) {
        if ((addr & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            // For RISC0 we have a shared local memory with base firmware so offset by that
            // if (has_shared_local_mem) {
            //     addr -= MEM_ERISC_BASE_FW_LOCAL_SIZE;
            // }
            // Move addresses in the local memory range to l1 (copied by kernel)
            return (addr & ~MEM_LOCAL_BASE) + local_init_addr;
        }

        // Note: Quasar does not have IRAM

        // No relocation needed
        return addr;
    };

    this->erisc_iram_relocate_func_ = [](uint64_t addr) { return addr; };

    this->valid_reg_addr_func_ = [](uint32_t /*addr*/) {
        return true;  // used to program start addr for eth FW TODO: add correct value
    };

    this->noc_xy_encoding_func_ = [](uint32_t x, uint32_t y) { return NOC_XY_ADDR(x, y, 0); };
    this->noc_multicast_encoding_func_ = [](uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end) {
        return NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, 0);
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
        return mailbox_base + offsetof(quasar::EthFwMailbox, arg) +
               (arg_index * sizeof(((quasar::EthFwMailbox*)nullptr)->arg[0]));
    };

    this->device_features_func_ = [](DispatchFeature feature) -> bool {
        switch (feature) {
            case DispatchFeature::ETH_MAILBOX_API: return true;
            // Active eth kernel config buffer is not needed until 2 ERISCs
            case DispatchFeature::DISPATCH_ACTIVE_ETH_KERNEL_CONFIG_BUFFER: return false;
            case DispatchFeature::DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER:
            case DispatchFeature::DISPATCH_TENSIX_KERNEL_CONFIG_BUFFER: return true;
            default: TT_THROW("Invalid Quasar dispatch feature {}", static_cast<int>(feature));
        }
    };

    this->max_processors_per_core_ = MaxProcessorsPerCoreType;
    this->num_nocs_ = NUM_NOCS;
    this->noc_node_id_ = NOC_NODE_ID;
    this->noc_node_id_mask_ = NOC_NODE_ID_MASK;
    this->noc_addr_node_id_bits_ = NOC_ADDR_NODE_ID_BITS;
    this->noc_encoding_reg_ = NOC_NODE_ID;                                   // TODO: add correct value
    this->noc_coord_reg_offset_ = NOC_COORD_REG_OFFSET;                      // TODO: add correct value
    this->noc_overlay_start_addr_ = 0;                                       // TODO: add correct value
    this->noc_stream_reg_space_size_ = 0;                                    // TODO: add correct value
    this->noc_stream_remote_dest_buf_size_reg_index_ = 0;                    // TODO: add correct value
    this->noc_stream_remote_dest_buf_start_reg_index_ = 0;                   // TODO: add correct value
    this->noc_stream_remote_dest_buf_space_available_reg_index_ = 0;         // TODO: add correct value
    this->noc_stream_remote_dest_buf_space_available_update_reg_index_ = 0;  // TODO: add correct value
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

    this->eps_ = EPS_QA;
    this->nan_ = NAN_QA;
    this->inf_ = INF_QA;

    this->noc_x_id_translate_table_ = {};

    this->noc_y_id_translate_table_ = {};

    this->jit_build_query_ = std::make_unique<HalJitBuildQueryQuasar>();

    this->verify_eth_fw_version_func_ = [](tt::umd::semver_t /*eth_fw_version*/) {
        // No checks
        return true;
    };
}

}  // namespace tt::tt_metal
