#include <string>
#include <fstream>

#include "tt_metal/host_api.hpp"

namespace internal {

void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(const char *DISPATCH_MAP_DUMP, Device* device);
void match_device_program_data_with_host_program_data(const char* host_file, const char* device_file);

} // end namespace internal
