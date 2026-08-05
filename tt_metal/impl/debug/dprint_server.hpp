// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Implements host-side debug print server interface.
 */

#pragma once

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <memory>
#include <vector>

namespace tt::tt_metal {

class MetalContext;
class MetalEnv;
class DispatchCoreConfig;

struct DPrintBufferInfo {
    uint64_t structure_address;
    uint16_t structure_size;
    uint16_t read_write_pointer_offset;
    uint16_t buffer_offset;
    uint16_t buffer_size;
    uint16_t processor_count;
    uint16_t processor_offset;

    uint64_t get_read_write_pointer_address() const { return structure_address + read_write_pointer_offset; }
};

class DPrintServer {
public:
    // Constructor/destructor, reads dprint options from RTOptions.
    DPrintServer(
        MetalContext* context, MetalEnv& env, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config);
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

    bool reads_dispatch_cores(ChipId device_id);

    // Returns the list of cores the print server polls for the given device.
    std::vector<umd::CoreDescriptor> get_print_cores(ChipId device_id) const;

    // Returns the print buffer info (rw pointer address, buffer offset, and buffer size) for the given core.
    std::vector<DPrintBufferInfo> get_core_buffers(ChipId device_id, const umd::CoreDescriptor& print_core) const;

    // Check whether a print hand has been detected by the server.
    // The print server tries to determine if a core is stalled due to the combination of (1) a WAIT
    // print command and (2) no new print data coming through. An invalid WAIT command and the print
    // buffer filling up afterwards can cause the core to spin forever. In this case this function will
    // return true and the print server will be terminated.
    bool hang_detected();

    class Impl;  // Defined in dprint_server.cpp.

private:
    std::unique_ptr<Impl> impl_;  // Pointer to implementation
};

}  // namespace tt::tt_metal
