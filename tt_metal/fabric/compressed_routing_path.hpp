#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Template specializations are defined in compressed_routing_path.cpp
template class compressed_routing_path_t<1>;
template class compressed_routing_path_t<2>;

}  // namespace tt::tt_fabric
