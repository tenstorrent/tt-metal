// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0



#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "core_coord_fixture.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeMerge)
{
    EXPECT_EQ ( this->sc1.merge(this->sc1).value(), this->sc1 );
    EXPECT_EQ ( this->cr4.merge(this->cr5).value(), this->cr6 );
    EXPECT_EQ ( this->cr4.merge(this->cr6).value(), this->cr6 );
    EXPECT_EQ ( this->cr4.merge(this->cr4).value(), this->cr4 );
}

TEST_F(CoreCoordHarness, TestCoreRangeNotMergeable){
    EXPECT_FALSE ( this->cr1.merge(this->cr3));
    EXPECT_FALSE ( this->cr2.merge(this->cr3));
    EXPECT_FALSE ( this->cr1.merge(this->cr6));
    EXPECT_FALSE ( this->sc1.merge(this->sc3));
    EXPECT_FALSE ( this->sc1.merge(this->sc2) );
}

}
