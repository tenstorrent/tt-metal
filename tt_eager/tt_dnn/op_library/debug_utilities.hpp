#pragma once


#include "tt_metal/common/core_coord.h"

#include "tt_metal/host_api.hpp"


#include "tt_metal/llrt/tt_debug_print_server.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

void setup_debug_server(tt::tt_metal::Device *device,
                std::vector<std::pair<size_t, size_t> > logical_coordinates,
                std::vector<int> chips = {0}) {

    std::vector<CoreCoord> physical_coordinates;
    for(auto logical_coord: logical_coordinates){
        physical_coordinates.push_back(device->worker_core_from_logical_core({logical_coord.first, logical_coord.second}));
    }
    tt_start_debug_print_server(device->cluster(), chips, physical_coordinates);
}
