
#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"
#include <set>

namespace basic_tests::CoreRangeSet{

TEST_F(CoreCoordHarness, TestCoreRangeSetMerge)
{
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr1}).ranges() , std::set<::CoreRange>( {cr1}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr2}).merge({cr3}).ranges() , std::set<::CoreRange>( {cr1,cr2,cr3}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1, cr2, cr3}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr4}) );

    EXPECT_EQ ( ::CoreRangeSet({cr1, cr2}).merge({cr4}).merge({cr6}).ranges() , std::set<::CoreRange>( {cr6}) );

    EXPECT_EQ ( ::CoreRangeSet({cr7}).merge({cr6}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr8} ) );

    EXPECT_EQ ( ::CoreRangeSet({cr8}).merge({cr7}).merge({cr6}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr8} ) );

}

}
