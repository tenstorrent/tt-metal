// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
* Implements host-side debug print server interface.
*/

#pragma once

namespace tt {

namespace tt_metal {
    class Device;
}

/*
@brief Attaches a device to be monitored by the print server. If no devices were present on the
    print server, also initializes the print server and launches the thread it runs on.

@param device Pointer to the device to attach to the print server. The cores/harts to be monitored
    on this device are determined by the environment variables read out in RTOptions.

This call is not thread safe, and there is only one instance of print server supported at a time.
*/
void DprintServerAttach(tt::tt_metal::Device* device);

/*
@brief Detach a device so it is no longer monitored by the print server. If no devices are present
    after detatching, also stops the print server.

@param device Pointer to device to detatch, will throw if trying to detatch a device that is not
    currently attached to the server.

Note that this api call is not thread safe at the moment.
*/
void DprintServerDetach(tt::tt_metal::Device* device);

/**
@brief Set device side profiler state.

@param profile_device true if profiling, false if not profiling
*/
void DprintServerSetProfilerState(bool profile_device);

/**
@brief Return if the instance debug print server is running or not.
*/
bool DprintServerIsRunning();

/**
@brief Set whether the debug print server should be muted.

Note that (1) muting the print server does not disable prints on the device or reading the data back
to the host (the print data is simply discarded instead of emitted), and (2) calling this function
while a kernel is running may result in loss of print data.

@param mute_print_server true to mute the print server, false to unmute
*/
void DprintServerSetMute(bool mute_print_server);

/**
@brief Wait until the debug print server is not currently processing data.

Note that this function does not actually check whether the device will continue producing print
data, it only checks whether the print server to finish with any data it is currently processing.
*/
void DprintServerAwait();

/**
@brief Check whether a print hang has been detected by the print server.

The print server tries to determine if a core is stalled due to the combination of (1) a WAIT
print command and (2) no new print data coming through. An invalid WAIT command and the print
buffer filling up afterwards can cause the core to spin forever. In this case this function will
return true and the print server will be terminated.
*/
bool DPrintServerHangDetected();

/**
@brief Clears the print server log file.
*/
void DPrintServerClearLogFile();

/**
@brief Clears any RAISE signals in the print server, so they can be used again in a later run.
*/
void DPrintServerClearSignals();

/**
@brief Returns true if the DPRINT server reads any dispatch cores on a given device.
*/
bool DPrintServerReadsDispatchCores(tt::tt_metal::Device* device);

} // namespace tt
