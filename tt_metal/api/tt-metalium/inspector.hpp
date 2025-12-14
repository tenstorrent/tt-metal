// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before including this file, make sure to compile capnp generated files from rpc.capnp
// and add directory of generated files to include path.
#include <rpc.capnp.h>
#include <string>

namespace tt::tt_metal {

void RegisterInspectorRpcChannel(const std::string& name, inspector::rpc::InspectorChannel::Client channel);

}  // namespace tt::tt_metal
