#ifndef CKERNEL_SFPU_RECIP_TEST_H
#define CKERNEL_SFPU_RECIP_TEST_H

#include "ckernel_sfpu_recip.h"
#include <gtest/gtest.h>

namespace ckernel {

namespace sfpu {

TEST(SfpuRecipTest, FP32Reciprocal) {
    // Test FP32 reciprocal with various inputs
    EXPECT_EQ(reciprocal(1.0f), 1.0f);
    EXPECT_EQ(reciprocal(2.0f), 0.5f);
    EXPECT_EQ(reciprocal(-2.0f), -0.5f);
    EXPECT_EQ(reciprocal(0.5f), 2.0f);
    EXPECT_EQ(reciprocal(-0.5f), -2.0f);
    EXPECT_EQ(reciprocal(0.0f), INFINITY);
    EXPECT_EQ(reciprocal(-0.0f), -INFINITY);
    EXPECT_EQ(reciprocal(INFINITY), 0.0f);
    EXPECT_EQ(reciprocal(-INFINITY), -0.0f);
    EXPECT_TRUE(isnan(reciprocal(NAN)));
}

TEST(SfpuRecipTest, BF16Reciprocal) {
    // Test BF16 reciprocal with various inputs
    EXPECT_EQ(reciprocal(bfloat16(1.0f)), bfloat16(1.0f));
    EXPECT_EQ(reciprocal(bfloat16(2.0f)), bfloat16(0.5f));
    EXPECT_EQ(reciprocal(bfloat16(-2.0f)), bfloat16(-0.5f));
    EXPECT_EQ(reciprocal(bfloat16(0.5f)), bfloat16(2.0f));
    EXPECT_EQ(reciprocal(bfloat16(-0.5f)), bfloat16(-2.0f));
    EXPECT_EQ(reciprocal(bfloat16(0.0f)), bfloat16(INFINITY));
    EXPECT_EQ(reciprocal(bfloat16(-0.0f)), bfloat16(-INFINITY));
    EXPECT_EQ(reciprocal(bfloat16(INFINITY)), bfloat16(0.0f));
    EXPECT_EQ(reciprocal(bfloat16(-INFINITY)), bfloat16(-0.0f));
    EXPECT_TRUE(isnan(reciprocal(bfloat16(NAN)).value()));
    
    // Test correctly rounded BF16 reciprocals
    EXPECT_EQ(reciprocal(bfloat16(1.0f / 3.0f)), bfloat16(3.0f));
    EXPECT_EQ(reciprocal(bfloat16(-1.0f / 3.0f)), bfloat16(-3.0f));
    EXPECT_EQ(reciprocal(bfloat16(1.0f / 10.0f)), bfloat16(10.0f));
    EXPECT_EQ(reciprocal(bfloat16(-1.0f / 10.0f)), bfloat16(-10.0f));
}

} // namespace sfpu

} // namespace ckernel

#endif // CKERNEL_SFPU_RECIP_TEST_H
