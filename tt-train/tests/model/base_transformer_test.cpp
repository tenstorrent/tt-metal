// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "models/base_transformer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

namespace {

// Like the tensor- and pipeline-parallel models, it doesn't implement load_from_safetensors.
class ModelWithoutSafetensorsLoader : public ttml::models::BaseTransformer {};

}  // namespace

TEST(BaseTransformerTest, LoadFromSafetensorsThrowsUnlessTheModelImplementsIt) {
    ModelWithoutSafetensorsLoader model;
    EXPECT_THAT(
        [&model] { model.load_from_safetensors("weights.safetensors"); },
        ::testing::ThrowsMessage<std::logic_error>(::testing::AllOf(
            ::testing::HasSubstr("can't load safetensors weights"), ::testing::HasSubstr("weights.safetensors"))));
}
