/*
* Implements host-side debug print server interface.
*/

#pragma once

#include "tt_cluster.hpp"
#include "tt_xy_pair.h"

enum DebugPrintHartFlags : unsigned int {
    DPRINT_HART_NC  = 1,
    DPRINT_HART_TR0 = 2,
    DPRINT_HART_TR1 = 4,
    DPRINT_HART_TR2 = 8,
    DPRINT_HART_BR  = 16,
};

constexpr int DPRINT_NHARTS = 5;

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
    tt_cluster* cluster,
    const vector<int>& chip_ids,
    const vector<tt_xy_pair>& cores,
    unsigned int thread_mask =
        DPRINT_HART_NC | DPRINT_HART_TR0 | DPRINT_HART_TR1 | DPRINT_HART_TR2 | DPRINT_HART_BR,
    const char* filename = nullptr);

/*
@brief Stops the print server thread. This call is optional.

tt_start_print_server() will register this function call with tt_cluster->on_destroy().

Note that this api call is not thread safe at the moment.
*/
void tt_stop_debug_print_server(tt_cluster* cluster = nullptr);
