
#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeContains){
    EXPECT_TRUE(this->cr1.contains(this->single_core));
    EXPECT_TRUE(this->cr1.contains(this->cr1));
    EXPECT_TRUE(this->cr4.contains(this->cr2));
}

TEST_F(CoreCoordHarness, TestCoreRangeNotContains){
    EXPECT_FALSE( this->single_core.contains(this->cr1));
    EXPECT_FALSE( this->cr1.contains(this->cr2));
    EXPECT_FALSE( this->cr7.contains(this->single_core));
}

}
