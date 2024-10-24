// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <exception>
#include "gtest/gtest.h"

#include "ttnn/tensor/vector_base.hpp"


TEST(TensorVectorBaseTests, General4D) {
    tt::tt_metal::VectorBase vec({20, 30, 40, 50});
    EXPECT_EQ(vec.view().size(), vec.as_vector().size());
    EXPECT_EQ(vec.view().size(), 4);
    EXPECT_EQ(vec[0], 20);
    EXPECT_EQ(vec[1],30);
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

    tt::tt_metal::VectorBase vec({20, 30, 40, 50, 60});
    EXPECT_EQ(vec.view().size(), vec.as_vector().size());
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

TEST(TensorVectorBaseTests, Empty)
{
    tt::tt_metal::VectorBase vec({});
    EXPECT_EQ(vec.view().size(), vec.as_vector().size());
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
    tt::tt_metal::VectorBase vec({20});
    EXPECT_EQ(vec.view().size(), vec.as_vector().size());
    EXPECT_EQ(vec.view().size(), 1);
    EXPECT_EQ(vec[0], 20);
    EXPECT_THROW(vec[1], std::exception);
    EXPECT_EQ(vec[-1], vec[0]);
    EXPECT_EQ(vec[-2], 1);
    EXPECT_EQ(vec[-3], 1);
    EXPECT_EQ(vec[-4], 1);
    EXPECT_THROW(vec[-5], std::exception);
}

TEST(TensorVectorBaseTests, TwoElements) {
    tt::tt_metal::VectorBase vec({20, 30});
    EXPECT_EQ(vec.view().size(), vec.as_vector().size());
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
