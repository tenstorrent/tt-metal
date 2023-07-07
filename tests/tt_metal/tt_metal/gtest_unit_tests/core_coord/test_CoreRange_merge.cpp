

#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeMerge)
{
    EXPECT_EQ ( this->single_core.merge(this->single_core).value(), this->single_core );
    EXPECT_EQ ( this->cr4.merge(this->cr5).value(), this->cr6 );
    EXPECT_EQ ( this->cr4.merge(this->cr6).value(), this->cr6 );
    EXPECT_EQ ( this->cr4.merge(this->cr4).value(), this->cr4 );
}

TEST_F(CoreCoordHarness, TestCoreRangeNotMergeable){
    EXPECT_FALSE ( this->cr1.merge(this->cr3));
    EXPECT_FALSE ( this->cr2.merge(this->cr3));
    EXPECT_FALSE ( this->cr1.merge(this->cr6));
}

}
