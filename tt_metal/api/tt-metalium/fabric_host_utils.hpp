// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/program_impl.hpp>

namespace tt::tt_fabric {

tt::tt_fabric::FabricEriscDatamoverConfig get_default_fabric_config();

// Used to get the run-time args for estabilishing connection with the fabric router.
// The API appends the connection specific run-time args to the set of exisiting
// run-time args for the worker programs, which allows the workers to conveniently
// build connection management object(s) using the run-time args.
// It is advised to call the API once all the other run-time args for the prgram are
// determined/pushed to keep things clean and avoid any extra arg management.
//
// Inputs:
// src_chip_id: physical chip id/device id of the sender chip
// dst_chip_id: physical chip id/device id of the receiver chip
// routing_plane: the link (0..n) to use b/w the src_chip_id and dst_chip_id. On WH for
//                instance we can have upto 4 active links b/w two chips
// worker_program: program handle
// worker_core: worker core logical coordinates
// worker_args: list of existing run-time args to which the connection args will be appended
//
// Constraints:
// 1. Currently the sender and reciever chip should be physically adjacent
// 2. Currently the sender and reciever chip should be on the same mesh (for 1D)
// 3. When connecting with 1D fabric routers, users are responsible for setting up the
// connection appropriately. The API will not perform any checks to ensure that the
// connection is indeed a 1D connection b/w all the workers.
void append_fabric_connection_rt_args(
    chip_id_t src_chip_id,
    chip_id_t dst_chip_id,
    routing_plane_id_t routing_plane,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args);

}  // namespace tt::tt_fabric
