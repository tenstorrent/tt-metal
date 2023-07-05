#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"

namespace basic_tests::CoreRangeSet{

TEST_F(CoreCoordHarness, TestCoreRangeSetValidConstruct)
{
    EXPECT_NO_THROW ( ::CoreRangeSet({this->single_core, this->cr2}));
    EXPECT_NO_THROW ( ::CoreRangeSet({this->cr1, this->cr2}) );

    ::CoreRangeSet valid_ranges = ::CoreRangeSet({this->cr1, this->cr2});
    EXPECT_EQ(valid_ranges.ranges().size(), 2);
}

TEST_F(CoreCoordHarness, TestCoreRangeSetInvalidConstruct)
{
    ::CoreRange overlapping_range = {.start={1, 2}, .end={3, 3}};
    EXPECT_ANY_THROW( ::CoreRangeSet({this->cr1, this->cr2, overlapping_range}) );
    EXPECT_ANY_THROW( ::CoreRangeSet({this->single_core, this->cr1}) );
}


}
