// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "core_coord_fixture.hpp"
#include <set>

namespace basic_tests::CoreRangeSet{

TEST_F(CoreCoordHarness, TestCoreRangeSetMergeNoSolution)
{
    EXPECT_EQ ( ::CoreRangeSet({sc1}).merge({sc3}).ranges() , std::set<::CoreRange>( {sc1,sc3}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr2}).ranges() , std::set<::CoreRange>( {cr1,cr2}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr1,cr2}).ranges() , std::set<::CoreRange>( {cr1,cr2}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr2}).merge({cr3}).ranges() , std::set<::CoreRange>( {cr1,cr2,cr3}) );
}

TEST_F(CoreCoordHarness, TestCoreRangeSetMergeCoreCoord)
{
    ::CoreRangeSet empty_crs({});
    EXPECT_EQ ( empty_crs.merge({this->sc1}).ranges().size(), 1);
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({sc3, sc4}).ranges() , std::set<::CoreRange>( {cr16}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({sc3}).merge({sc4}).ranges() , std::set<::CoreRange>( {cr16}) );
    CoreRange rect ( {0,0}, {4,2});
    std::set<CoreRange> rect_pts;
    for (unsigned y = rect.start_coord.y; y <= rect.end_coord.y; y++){
        for (unsigned x = rect.start_coord.x; x <= rect.end_coord.x; x++){
            rect_pts.insert ( CoreRange( {x, y}, {x, y} ) );
        }
    }
    EXPECT_EQ ( empty_crs.merge(rect_pts).ranges(), std::set<::CoreRange>( {rect} ));

    // upside-down "T"
    rect_pts.insert ( { CoreRange ( { 2,0}, {3,5} )});
    EXPECT_EQ ( empty_crs.merge(rect_pts).ranges(), std::set<::CoreRange>( {rect, CoreRange( {2,3}, {3,5} ) } ));

    // "H", sub-optimal currently, should be reduced down to 3 CRs instead of 5
    EXPECT_EQ ( empty_crs.merge( { CoreRange { {0,0}, {1,5} }, CoreRange { {3,0}, {4,5}}, CoreRange { {0,2} , {4,3} }  } ).ranges(),
                std::set<::CoreRange>( { CoreRange { {0,0}, {1,1} }, CoreRange { {0,2}, {4,3}}, CoreRange{ {0,4}, {1,5}},
                                       CoreRange { {3,0}, {4,1} }, CoreRange{ {3,4}, {4,5} } } ));
}

TEST_F(CoreCoordHarness, TestCoreRangeSetMergeCoreRange)
{
    EXPECT_EQ ( ::CoreRangeSet({cr1}).merge({cr1}).ranges() , std::set<::CoreRange>( {cr1}) );
    EXPECT_EQ ( ::CoreRangeSet({cr7}).merge({cr6}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr8} ) );
    EXPECT_EQ ( ::CoreRangeSet({cr8}).merge({cr7}).merge({cr6}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr8} ) );
    EXPECT_EQ ( ::CoreRangeSet({cr1, cr2, cr3}).merge({cr4}).ranges() , std::set<::CoreRange>( {cr4}) );
    EXPECT_EQ ( ::CoreRangeSet({cr1, cr2}).merge({cr4}).merge({cr6}).ranges() , std::set<::CoreRange>( {cr6}) );
}

}
