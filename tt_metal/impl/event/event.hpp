// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal
{
    class Device;
    struct Event
    {
        Device * device;
        uint32_t cq_id;
        uint32_t event_id;
    };
}
