#pragma once

#include <stdint.h>
#include <optional>

struct TLB_OFFSETS {
    uint32_t local_offset;
    uint32_t x_end;
    uint32_t y_end;
    uint32_t x_start;
    uint32_t y_start;
    uint32_t noc_sel;
    uint32_t mcast;
    uint32_t ordering;
    uint32_t linked;
    uint32_t static_vc;
    uint32_t static_vc_end;
};

struct TLB_DATA {
    uint64_t local_offset = 0;
    uint64_t x_end = 0;
    uint64_t y_end = 0;
    uint64_t x_start = 0;
    uint64_t y_start = 0;
    uint64_t noc_sel = 0;
    uint64_t mcast = 0;
    uint64_t ordering = 0;
    uint64_t linked = 0;
    uint64_t static_vc = 0;

    // Orderings
    static constexpr uint64_t Relaxed = 0;
    static constexpr uint64_t Strict  = 1;
    static constexpr uint64_t Posted  = 2;

    bool check(const TLB_OFFSETS offset);
    std::optional<uint64_t> apply_offset(const TLB_OFFSETS offset);
};

enum class TensixSoftResetOptions: std::uint32_t {
    NONE = 0,
    BRISC = ((std::uint32_t) 1 << 11),
    TRISC0 = ((std::uint32_t) 1 << 12),
    TRISC1 = ((std::uint32_t) 1 << 13),
    TRISC2 = ((std::uint32_t) 1 << 14),
    NCRISC = ((std::uint32_t) 1 << 18),
    STAGGERED_START = ((std::uint32_t) 1 << 31)
};

std::string TensixSoftResetOptionsToString(TensixSoftResetOptions value);
TensixSoftResetOptions operator|(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs);
TensixSoftResetOptions operator&(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs);
bool operator!=(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs);

static const TensixSoftResetOptions ALL_TRISC_SOFT_RESET = TensixSoftResetOptions::TRISC0 |
                                                           TensixSoftResetOptions::TRISC1 |
                                                           TensixSoftResetOptions::TRISC2;

static const TensixSoftResetOptions ALL_TENSIX_SOFT_RESET = TensixSoftResetOptions::BRISC |
                                                            TensixSoftResetOptions::NCRISC |
                                                            TensixSoftResetOptions::STAGGERED_START |
                                                            ALL_TRISC_SOFT_RESET;

static const TensixSoftResetOptions TENSIX_ASSERT_SOFT_RESET = TensixSoftResetOptions::BRISC |
                                                               TensixSoftResetOptions::NCRISC |
                                                               ALL_TRISC_SOFT_RESET;

static const TensixSoftResetOptions TENSIX_DEASSERT_SOFT_RESET = TensixSoftResetOptions::NCRISC |
                                                                 ALL_TRISC_SOFT_RESET |
                                                                 TensixSoftResetOptions::STAGGERED_START;

static const TensixSoftResetOptions TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER = TensixSoftResetOptions::NCRISC |
                                                                 ALL_TRISC_SOFT_RESET;
