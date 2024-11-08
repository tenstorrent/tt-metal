// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <array>
#include <tuple>
#include <vector>

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/event/event.hpp"
#include "tt_metal/impl/sub_device/sub_device.hpp"
#include "tests/tt_metal/test_utils/stimulus.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/command_queue/command_queue_test_utils.hpp"

using namespace tt::tt_metal;

namespace basic_tests {




}  // namespace basic_tests
