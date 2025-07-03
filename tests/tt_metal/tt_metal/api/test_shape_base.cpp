// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/shape_base.hpp>
#include <exception>
#include <memory>

#include "gtest/gtest.h"
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

TEST(TensorShapeBaseTests, General4D) {
    tt::tt_metal::ShapeBase vec({20, 30, 40, 50});
    EXPECT_EQ(vec.view().size(), vec.view().size());
    EXPECT_EQ(vec.view().size(), 4);
    EXPECT_EQ(vec[0], 20);
    EXPECT_EQ(vec[1], 30);
    EXPECT_EQ(vec[2], 40);
    EXPECT_EQ(vec[3], 50);
    EXPECT_THROW(vec[4], std::exception);
    EXPECT_EQ(vec[-1], vec[3]);
    EXPECT_EQ(vec[-2], vec[2]);
    EXPECT_EQ(vec[-3], vec[1]);
    EXPECT_EQ(vec[-4], vec[0]);
    EXPECT_THROW(vec[-5], std::exception);
}

TEST(TensorVectorBaseTests, General5D) {
    tt::tt_metal::ShapeBase vec({20, 30, 40, 50, 60});
    EXPECT_EQ(vec.view().size(), vec.view().size());
    EXPECT_EQ(vec.view().size(), 5);
    EXPECT_EQ(vec[4], 60);
    EXPECT_THROW(vec[5], std::exception);
    EXPECT_EQ(vec[-1], vec[4]);
    EXPECT_EQ(vec[-2], vec[3]);
    EXPECT_EQ(vec[-3], vec[2]);
    EXPECT_EQ(vec[-4], vec[1]);
    EXPECT_EQ(vec[-5], vec[0]);
    EXPECT_THROW(vec[-6], std::exception);
}

TEST(TensorShapeBaseTests, Empty) {
    tt::tt_metal::ShapeBase vec({});
    EXPECT_EQ(vec.view().size(), vec.view().size());
    EXPECT_EQ(vec.view().size(), 0);
    EXPECT_THROW(vec[0], std::exception);
    EXPECT_THROW(vec[1], std::exception);
    EXPECT_THROW(vec[2], std::exception);
    EXPECT_THROW(vec[3], std::exception);
    EXPECT_THROW(vec[4], std::exception);
    EXPECT_EQ(vec[-1], 1);
    EXPECT_EQ(vec[-2], 1);
    EXPECT_EQ(vec[-3], 1);
    EXPECT_EQ(vec[-4], 1);
    EXPECT_THROW(vec[-5], std::exception);
}

TEST(TensorVectorBaseTests, SingleElement) {
    tt::tt_metal::ShapeBase vec({20});
    EXPECT_EQ(vec.view().size(), vec.view().size());
    EXPECT_EQ(vec.view().size(), 1);
    EXPECT_EQ(vec[0], 20);
    EXPECT_THROW(vec[1], std::exception);
    EXPECT_EQ(vec[-1], vec[0]);
    EXPECT_EQ(vec[-2], 1);
    EXPECT_EQ(vec[-3], 1);
    EXPECT_EQ(vec[-4], 1);
    EXPECT_THROW(vec[-5], std::exception);
}

TEST(TensorShapeBaseTests, TwoElements) {
    tt::tt_metal::ShapeBase vec({20, 30});
    EXPECT_EQ(vec.view().size(), vec.view().size());
    EXPECT_EQ(vec.view().size(), 2);
    EXPECT_EQ(vec[0], 20);
    EXPECT_EQ(vec[1], 30);
    EXPECT_THROW(vec[2], std::exception);
    EXPECT_EQ(vec[-1], vec[1]);
    EXPECT_EQ(vec[-2], vec[0]);
    EXPECT_EQ(vec[-3], 1);
    EXPECT_EQ(vec[-4], 1);
    EXPECT_THROW(vec[-5], std::exception);
}

}  // namespace tt::tt_metal
