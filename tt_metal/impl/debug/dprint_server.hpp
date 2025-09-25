// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Implements host-side debug print server interface.
 */

#pragma once

#include <umd/device/types/cluster_descriptor_types.h>
#include <llrt/rtoptions.hpp>
#include <memory>

namespace tt::tt_metal {

class DPrintServer {
public:
    // Constructor/destructor, reads dprint options from RTOptions.
    DPrintServer(llrt::RunTimeOptions& rtoptions);
    ~DPrintServer();

    // Sets whether the print server is muted. Calling this function while a kernel is running may
    // result in a loss of print data.
    void set_mute(bool mute_print_server);

    // Waits for the print server to finish processing any current print data.
    void await();

    // Attach all enabled devices (via RTOptions) to the print server
    void attach_devices();

    // Detach all devices from the print server
    void detach_devices();

    // Clears the log file of a currently-running print server.
    void clear_log_file();

    // Clears any raised signals (so they can be used again in a later run).
    void clear_signals();

    bool reads_dispatch_cores(chip_id_t device_id);

    // Check whether a print hand has been detected by the server.
    // The print server tries to determine if a core is stalled due to the combination of (1) a WAIT
    // print command and (2) no new print data coming through. An invalid WAIT command and the print
    // buffer filling up afterwards can cause the core to spin forever. In this case this function will
    // return true and the print server will be terminated.
    bool hang_detected();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;  // Pointer to implementation
};

}  // namespace tt::tt_metal
