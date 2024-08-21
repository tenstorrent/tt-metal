// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

namespace tt_metal {

// If you are trying to include this file and you aren't hal...you are doing something wrong

HalCoreInfoType create_tensix_mem_map();
HalCoreInfoType create_active_eth_mem_map();
HalCoreInfoType create_idle_eth_mem_map();

}  // namespace tt_metal
}  // namespace tt
