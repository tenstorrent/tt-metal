#pragma once

namespace tt {

namespace tt_metal {
// Get global host profiling state based on build flag
bool getHostProfilerState ();

// Get global device profiling state based on build flag and environment variables
bool getDeviceProfilerState ();

}  // namespace tt_metal

}  // namespace tt
