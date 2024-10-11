// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
These are internal API calls that are used for testing or other internal use only
for the purpose of supporting the the official Metal API located under tt_metal/api.

Note that the directory structure here mirrors that of the Metal API.

*/
#include "types.hpp"
#include "buffer.hpp"
#include "command_queue.hpp"
#include "device.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "program.hpp"
#include "trace.hpp"
