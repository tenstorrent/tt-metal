
#include <memory>

#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeIntersects)
{
    EXPECT_EQ ( this->cr2.intersects(this->cr3), ::CoreRange({3,3}, {3,3}));
    EXPECT_EQ ( this->cr1.intersects(this->cr3).value(), ::CoreRange({1,2}, {2,2}));

    EXPECT_EQ ( this->single_core.intersects(this->cr6).value(), this->single_core);
    EXPECT_EQ ( this->cr4.intersects(this->cr5).value(), ::CoreRange({1,0}, {5,4} ));
    EXPECT_EQ ( this->cr1.intersects(this->cr5).value(), ::CoreRange({1,0}, {2,2} ));
    EXPECT_EQ ( this->cr1.intersects(this->cr6).value(), this->cr1 );
}

TEST_F(CoreCoordHarness, TestCoreRangeNotIntersects){
    EXPECT_FALSE ( this->cr1.intersects(this->cr2));
    EXPECT_FALSE ( this->single_core.intersects(this->cr2));
    EXPECT_FALSE ( this->cr1.intersects(this->cr7));
}

}
