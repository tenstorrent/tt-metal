// Test DST register file capacity based on accumulation mode.
//
// The DST register file has a fixed physical size (16 slots x 1024 elements x 4B = 64KB).
// The number of *active* slots depends on the data format:
//   BF16 mode (DST_ACCUM_MODE == 0): 16 active slots (2B elements packed into 4B storage)
//   FP32 mode (DST_ACCUM_MODE != 0):  8 active slots (4B elements use full storage width)
//
// This test is compiled twice: once with DST_ACCUM_MODE=0 and once with DST_ACCUM_MODE=1.
// Each variant validates the correct active tile count, bounds checking, and tile_regs_acquire.

#include <gtest/gtest.h>

// tt-emule standalone DST register file
#include "tt_emule/dst_register_file.hpp"

// JIT-path DST (requires jit_kernel_stubs.hpp → common.h chain)
// DST_ACCUM_MODE must be defined before this include (via compiler -D flag).
#include "jit_hw/api/compute/common.h"

using tt_emule::DstRegisterFile;

// ============================================================
// Standalone DstRegisterFile tests
// ============================================================

TEST(DstStandalone, PhysicalSizeIs16) {
    EXPECT_EQ(DstRegisterFile::TOTAL_SLOTS, 16u);
}

TEST(DstStandalone, BF16Has16ActiveSlots) {
    DstRegisterFile dst;
    dst.set_fp32_mode(false);
    EXPECT_EQ(dst.active_slots(), 16u);
    EXPECT_EQ(dst.active_slots(), DstRegisterFile::BF16_SLOTS);
}

TEST(DstStandalone, FP32Has8ActiveSlots) {
    DstRegisterFile dst;
    dst.set_fp32_mode(true);
    EXPECT_EQ(dst.active_slots(), 8u);
    EXPECT_EQ(dst.active_slots(), DstRegisterFile::FP32_SLOTS);
}

TEST(DstStandalone, DefaultIsBF16) {
    DstRegisterFile dst;
    EXPECT_FALSE(dst.fp32_mode());
    EXPECT_EQ(dst.active_slots(), 16u);
}

TEST(DstStandalone, ModeSwitchToggles) {
    DstRegisterFile dst;
    dst.set_fp32_mode(true);
    EXPECT_EQ(dst.active_slots(), 8u);
    dst.set_fp32_mode(false);
    EXPECT_EQ(dst.active_slots(), 16u);
}

TEST(DstStandalone, AllPhysicalSlotsWritable) {
    DstRegisterFile dst;
    for (size_t i = 0; i < DstRegisterFile::TOTAL_SLOTS; ++i) {
        dst[i](0, 0) = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < DstRegisterFile::TOTAL_SLOTS; ++i) {
        EXPECT_FLOAT_EQ(dst[i](0, 0), static_cast<float>(i + 1));
    }
}

TEST(DstStandalone, StateMachineFullCycle) {
    DstRegisterFile dst;
    EXPECT_EQ(dst.state(), DstRegisterFile::State::IDLE);
    dst.acquire();
    EXPECT_EQ(dst.state(), DstRegisterFile::State::ACQUIRED);
    dst.commit();
    EXPECT_EQ(dst.state(), DstRegisterFile::State::COMMITTED);
    dst.wait();
    EXPECT_EQ(dst.state(), DstRegisterFile::State::PACKING);
    dst.release();
    EXPECT_EQ(dst.state(), DstRegisterFile::State::IDLE);
}

TEST(DstStandalone, ConstantsMatchJITPath) {
    EXPECT_EQ(DstRegisterFile::BF16_SLOTS, __EMULE_DST_TILES);
    EXPECT_EQ(DstRegisterFile::FP32_SLOTS, __EMULE_DST_TILES_FP32);
    EXPECT_EQ(DstRegisterFile::TOTAL_SLOTS, __EMULE_DST_TILES);
}

TEST(DstStandalone, TotalPhysicalSizeIs64KB) {
    // 16 slots x 1024 elements x 4 bytes = 65536
    constexpr size_t expected = DstRegisterFile::TOTAL_SLOTS * 1024 * sizeof(float);
    EXPECT_EQ(expected, 65536u);
}

// ============================================================
// JIT-path DST tests (compile-time DST_ACCUM_MODE)
// ============================================================

#if DST_ACCUM_MODE == 0

TEST(DstJitBF16, ActiveTilesIs16) {
    EXPECT_EQ(__emule_dst_active_tiles(), 16u);
}

TEST(DstJitBF16, ArrayHas16Slots) {
    EXPECT_EQ(__EMULE_DST_TILES, 16u);
}

TEST(DstJitBF16, CanWriteAllActiveSlots) {
    for (uint32_t s = 0; s < __emule_dst_active_tiles(); ++s) {
        __emule_dst[s][0] = static_cast<float>(s);
    }
    for (uint32_t s = 0; s < __emule_dst_active_tiles(); ++s) {
        EXPECT_FLOAT_EQ(__emule_dst[s][0], static_cast<float>(s));
    }
}

TEST(DstJitBF16, BoundsCheckPassesAtLimit) {
    // Slot 15 is the last valid slot in bf16 mode
    EXPECT_NO_FATAL_FAILURE(__emule_dst_check(15, "test"));
}

TEST(DstJitBF16, BoundsCheckAbortsBeyondLimit) {
    EXPECT_DEATH(__emule_dst_check(16, "test"), "DST out-of-bounds");
}

TEST(DstJitBF16, TileRegsAcquireZerosActiveSlots) {
    // Write sentinel values to all 16 slots
    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t e = 0; e < __EMULE_TILE_ELEMS; ++e)
            __emule_dst[s][e] = 42.0f;
    }
    tile_regs_acquire();
    // All 16 active slots should be zeroed
    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t e = 0; e < __EMULE_TILE_ELEMS; ++e)
            EXPECT_FLOAT_EQ(__emule_dst[s][e], 0.0f)
                << "slot=" << s << " elem=" << e;
    }
}

TEST(DstJitBF16, Int32TypePunRoundTrips) {
    __emule_dst_store_i32(0, 0, 0x12345678);
    EXPECT_EQ(__emule_dst_load_i32(0, 0), 0x12345678);
}

#else  // DST_ACCUM_MODE != 0 (FP32 mode)

TEST(DstJitFP32, ActiveTilesIs8) {
    EXPECT_EQ(__emule_dst_active_tiles(), 8u);
}

TEST(DstJitFP32, ArrayStillHas16PhysicalSlots) {
    // Physical array is always 16 — only active count changes
    EXPECT_EQ(__EMULE_DST_TILES, 16u);
}

TEST(DstJitFP32, CanWriteAllActiveSlots) {
    for (uint32_t s = 0; s < __emule_dst_active_tiles(); ++s) {
        __emule_dst[s][0] = static_cast<float>(s);
    }
    for (uint32_t s = 0; s < __emule_dst_active_tiles(); ++s) {
        EXPECT_FLOAT_EQ(__emule_dst[s][0], static_cast<float>(s));
    }
}

TEST(DstJitFP32, BoundsCheckPassesAtLimit) {
    // Slot 7 is the last valid slot in fp32 mode
    EXPECT_NO_FATAL_FAILURE(__emule_dst_check(7, "test"));
}

TEST(DstJitFP32, BoundsCheckAbortsBeyondLimit) {
    EXPECT_DEATH(__emule_dst_check(8, "test"), "DST out-of-bounds");
}

TEST(DstJitFP32, Slot8AbortsBecauseFP32LimitIs8) {
    // Slot 8 exists physically but is out-of-bounds in fp32 mode
    EXPECT_DEATH(__emule_dst_check(8, "test"), "DST out-of-bounds");
}

TEST(DstJitFP32, TileRegsAcquireZerosOnly8Slots) {
    // Write sentinels to all 16 physical slots
    for (uint32_t s = 0; s < 16; ++s) {
        for (uint32_t e = 0; e < __EMULE_TILE_ELEMS; ++e)
            __emule_dst[s][e] = 42.0f;
    }
    tile_regs_acquire();
    // First 8 slots (active) should be zeroed
    for (uint32_t s = 0; s < 8; ++s) {
        for (uint32_t e = 0; e < __EMULE_TILE_ELEMS; ++e)
            EXPECT_FLOAT_EQ(__emule_dst[s][e], 0.0f)
                << "slot=" << s << " elem=" << e;
    }
    // Slots 8-15 (inactive) should retain sentinel
    for (uint32_t s = 8; s < 16; ++s) {
        EXPECT_FLOAT_EQ(__emule_dst[s][0], 42.0f)
            << "inactive slot " << s << " should not be zeroed";
    }
}

TEST(DstJitFP32, Int32TypePunRoundTrips) {
    __emule_dst_store_i32(0, 0, 0xDEADBEEF);
    EXPECT_EQ(__emule_dst_load_i32(0, 0), static_cast<int32_t>(0xDEADBEEF));
}

#endif
