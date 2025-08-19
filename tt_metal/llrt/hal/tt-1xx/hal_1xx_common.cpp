// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "hal_1xx_common.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::hal_1xx {

std::vector<std::string> HalJitBuildQueryBase::defines(const HalJitBuildQueryInterface::Params& params) const {
    std::vector<std::string> defines;
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    uint32_t l1_cache_disable_mask = rtoptions.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    switch (params.core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (params.processor_class) {
                case HalProcessorClassType::DM:
                    switch (params.processor_id) {
                        case 0:
                            defines.push_back("COMPILE_FOR_BRISC");
                            if (l1_cache_disable_mask & llrt::DebugHartFlags::RISCV_BR) {
                                defines.push_back("DISABLE_L1_DATA_CACHE");
                            }
                            break;
                        case 1:
                            defines.push_back("COMPILE_FOR_NCRISC");
                            if (l1_cache_disable_mask & llrt::DebugHartFlags::RISCV_NC) {
                                defines.push_back("DISABLE_L1_DATA_CACHE");
                            }
                            break;
                    }
                    break;
                case HalProcessorClassType::COMPUTE: {
                    uint32_t debug_compute_mask =
                        (tt::llrt::DebugHartFlags::RISCV_TR0 | tt::llrt::DebugHartFlags::RISCV_TR1 |
                         tt::llrt::DebugHartFlags::RISCV_TR2);
                    if ((l1_cache_disable_mask & debug_compute_mask) == debug_compute_mask) {
                        defines.push_back("DISABLE_L1_DATA_CACHE");
                    }
                    switch (params.processor_id) {
                        case 0:
                            defines.push_back("UCK_CHLKC_UNPACK");
                            defines.push_back("NAMESPACE=chlkc_unpack");
                            break;
                        case 1:
                            defines.push_back("UCK_CHLKC_MATH");
                            defines.push_back("NAMESPACE=chlkc_math");
                            break;
                        case 2:
                            defines.push_back("UCK_CHLKC_PACK");
                            defines.push_back("NAMESPACE=chlkc_pack");
                            break;
                    }
                    defines.push_back(fmt::format("COMPILE_FOR_TRISC={}", params.processor_id));
                    break;
                }
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH: {
            uint32_t erisc_mask = (tt::llrt::DebugHartFlags::RISCV_ER0 | tt::llrt::DebugHartFlags::RISCV_ER1);
            if ((l1_cache_disable_mask & erisc_mask) == erisc_mask) {
                defines.push_back("DISABLE_L1_DATA_CACHE");
            }
            defines.push_back("COMPILE_FOR_ERISC");
            defines.push_back("ERISC");
            defines.push_back("RISC_B0_HW");
            break;
        }
        case HalProgrammableCoreType::IDLE_ETH: {
            uint32_t erisc_mask = (tt::llrt::DebugHartFlags::RISCV_ER0 | tt::llrt::DebugHartFlags::RISCV_ER1);
            if ((l1_cache_disable_mask & erisc_mask) == erisc_mask) {
                defines.push_back("DISABLE_L1_DATA_CACHE");
            }
            defines.push_back(fmt::format("COMPILE_FOR_IDLE_ERISC={}", params.processor_id));
            defines.push_back("ERISC");
            defines.push_back("RISC_B0_HW");
            break;
        }
        default: TT_ASSERT(false, "Unsupported programmable core type {} to query defines", params.core_type); break;
    }
    return defines;
}

std::vector<std::string> HalJitBuildQueryBase::srcs(const HalJitBuildQueryInterface::Params& params) const {
    std::vector<std::string> srcs;

    switch (params.core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (params.processor_class) {
                case HalProcessorClassType::DM:
                    switch (params.processor_id) {
                        case 0:
                            if (params.is_fw) {
                                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/brisc.cc");
                            } else {
                                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/brisck.cc");
                            }
                            break;
                        case 1:
                            if (params.is_fw) {
                                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc");
                            } else {
                                srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc");
                            }
                            break;
                        default: TT_ASSERT(false, "Invalid processor id {} for TENSIX DM", params.processor_id);
                    }
                    break;
                case HalProcessorClassType::COMPUTE:
                    if (params.is_fw) {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/trisc.cc");
                    } else {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/trisck.cc");
                    }
                    break;
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            // This is different on Wormhole vs Blackhole.
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            switch (params.processor_id) {
                case 0:
                    if (params.is_fw) {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/idle_erisc.cc");
                    } else {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/idle_erisck.cc");
                    }
                    break;
                case 1:
                    if (params.is_fw) {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/subordinate_idle_erisc.cc");
                    } else {
                        srcs.push_back("tt_metal/hw/firmware/src/tt-1xx/idle_erisck.cc");
                    }
                    break;
            }
            break;
        default: TT_ASSERT(false, "Unsupported programmable core type {} to query srcs", params.core_type); break;
    }
    return srcs;
}

std::string HalJitBuildQueryBase::target_name(const HalJitBuildQueryInterface::Params& params) const {
    switch (params.core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (params.processor_class) {
                case HalProcessorClassType::DM: return params.processor_id == 0 ? "brisc" : "ncrisc"; break;
                case HalProcessorClassType::COMPUTE: return fmt::format("trisc{}", params.processor_id);
            }
        case HalProgrammableCoreType::ACTIVE_ETH: return "erisc";
        case HalProgrammableCoreType::IDLE_ETH:
            return params.processor_id == 0 ? "idle_erisc" : "subordinate_idle_erisc";
        default:
            TT_THROW(
                "Unsupported programmable core type {} to query target name", enchantum::to_string(params.core_type));
    }
}

}  // namespace tt::tt_metal::hal_1xx
