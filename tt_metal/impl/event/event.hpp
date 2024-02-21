// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
namespace tt::tt_metal
{
    class Device;
    struct Event
    {
        Device * device = nullptr;
        uint32_t cq_id = -1;
        uint32_t event_id = -1;
    };
}
