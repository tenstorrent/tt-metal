#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

#include "catch.hpp"

using namespace tt;

// TEST_CASE(
//     "single_core_reader_datacopy_writer", "[dram][single_core][writer][reader]") {
//     const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
//     const int pci_express_slot = 0;
//     auto device = tt_metal::CreateDevice(arch, pci_express_slot);
//     tt_metal::InitializeDevice(device);
//     REQUIRE(
//         read_write_single_core_to_single_dram(
//             device,
//             4,
//             2*32*32,
//             1,
//             0,
//             0,
//             0,
//             UNRESERVED_BASE,
//             tt::DataFormat::Float16_b,
//             UNRESERVED_BASE + 16*32*32,
//             tt::DataFormat::Float16_b,
//             {.x=0, .y=0}
//         )
//     );
//     tt_metal::CloseDevice(device);
// }
