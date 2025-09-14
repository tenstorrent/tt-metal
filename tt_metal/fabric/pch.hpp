// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/wire_format.h>
#include <queue>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/edm_fabric_counters.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_metal/common/env_lib.hpp>
#include <tt_metal/impl/context/metal_context.hpp>
#include <tt_metal/impl/dispatch/kernel_config/relay_mux.hpp>
#include <yaml-cpp/yaml.h>
