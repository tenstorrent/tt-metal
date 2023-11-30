// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
* Implements host-side debug print server interface.
*/

#pragma once

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"

enum DebugPrintHartFlags : unsigned int {
    DPRINT_RISCV_NC  = 1,
    DPRINT_RISCV_TR0 = 2,
    DPRINT_RISCV_TR1 = 4,
    DPRINT_RISCV_TR2 = 8,
    DPRINT_RISCV_BR  = 16,
};

constexpr int DPRINT_NRISCVS = 5;

/*
@brief Starts the print server thread - will poll all specified chip/cores/harts of a device for any print data accumulated in thread-local buffers.

thread_mask in this API is using flags from DebugPrintHartFlags to enable per-thread debug server listening.
This call will launch a host thread for each core in cores array and each hart specified by the mask.

Note that this call only works correctly after open_device and start_device calls. (TODO(AP): add runtime checks)

@param filename If filename is null, by default all prints will go to std::cout.
                If filename is specified, the output will go to that newly created file (used for testing).
@param cores    A vector of NOC coordinates of cores for the print server to poll.
                If a given core is not in this list, then it's DPRINT output will not show up on the host.
                Note that these are not logical worker coordinates.
                NOC coordinates start at {1,1} and have gaps, logical start at {0,0}.

This call is not thread safe, and there is only one instance of print server supported at a time.

*/
void tt_start_debug_print_server(
    std::function<CoreCoord ()>get_grid_size,
    std::function<CoreCoord (CoreCoord)>worker_from_logical
);

/*
@brief Stops the print server thread. This call is optional.

Note that this api call is not thread safe at the moment.
*/
void tt_stop_debug_print_server();

/**
@brief Set device side profiler state.

@param profile_device true if profiling, false if not profiling
*/
void tt_set_profiler_state_for_debug_print(bool profile_device);

/**
@brief Return if the instance debug print server is running or not.
*/
bool tt_is_print_server_running();

/**
@brief Set whether the debug print server should be muted.

Note that (1) muting the print server does not disable prints on the device or reading the data back
to the host (the print data is simply discarded instead of emitted), and (2) calling this function
while a kernel is running may result in loss of print data.

@param mute_print_server true to mute the print server, false to unmute
*/
void tt_set_debug_print_server_mute(bool mute_print_server);

/**
@brief Wait until the debug print server is not currently processing data.

Note that this function does not actually check whether the device will continue producing print
data, it only checks whether the print server is currently processing print data.
*/
void tt_await_debug_print_server();

/**
@brief Check whether a print hang has been detected by the print server.

The print server tries to determine if a core is stalled due to the combination of (1) a WAIT
print command and (2) no new print data coming through. An invalid WAIT command and the print
buffer filling up afterwards can cause the core to spin forever. In this case this function will
return true and the print server will be terminated.
*/
bool tt_print_hang_detected();
