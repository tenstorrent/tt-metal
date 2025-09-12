#pragma once

// umd::device
#include <umd/device/tt_cluster_descriptor.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

// enchantum::enchantum
#include <enchantum/enchantum.hpp>

// fmt::fmt-header-only
#include <fmt/base.h>

// TracyClient
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

// nlohmann_json::nlohmann_json
#include <nlohmann/json_fwd.hpp>
#include <nlohmann/json.hpp>

// TT::Metalium::HostDevCommon
#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/fabric_common.h>
#include <hostdevcommon/kernel_structs.h>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

// Reflect::Reflect
#include <reflect>

// TT::STL
#include <tt_stl/aligned_allocator.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/unique_any.hpp>

// tt-logger::tt-logger
#include <tt-logger/tt-logger.hpp>
