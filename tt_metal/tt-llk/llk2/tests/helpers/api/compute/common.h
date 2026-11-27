// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TEST ONLY: stands in for tt-metal's api/compute/common.h, which a kernel includes for get_arg_val (its run-time
// args). The test that includes the kernel file defines get_arg_val from its own parameters, as it defines
// get_compile_time_arg_val (tt_metal_kernel.h).

#pragma once
