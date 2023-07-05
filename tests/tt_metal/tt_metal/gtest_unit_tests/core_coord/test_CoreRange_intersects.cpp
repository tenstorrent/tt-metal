
#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.h"
#include "../basic_harness.hpp"

namespace basic_tests::CoreRange{

TEST_F(CoreCoordHarness, TestCoreRangeIntersects)
{

    EXPECT_TRUE( this->cr1.intersects(this->cr5).has_value() );
    EXPECT_EQ ( this->cr1.intersects(this->cr5).value(), ::CoreRange({1,0}, {1,1}));

    EXPECT_TRUE( this->single_core.intersects(this->cr6).has_value());
    EXPECT_EQ ( this->single_core.intersects(this->cr6).value(), this->single_core);

    EXPECT_TRUE( this->cr4.intersects(this->cr5).has_value());
    EXPECT_EQ ( this->cr4.intersects(this->cr5).value(), ::CoreRange({1,0}, {5,4} ));

    EXPECT_TRUE( this->cr1.intersects(this->cr6).has_value());
    EXPECT_EQ ( this->cr1.intersects(this->cr6).value(), this->cr1 );

    EXPECT_TRUE( this->cr7.intersects(this->cr8).has_value());
    EXPECT_EQ ( this->cr7.intersects(this->cr8).value(), this->cr7 );
}

TEST_F(CoreCoordHarness, TestCoreRangeNotIntersects){
    EXPECT_FALSE ( this->cr1.intersects(this->cr2));
    EXPECT_FALSE ( this->single_core.intersects(this->cr2));
    EXPECT_FALSE ( this->cr1.intersects(this->cr7));
}

}
