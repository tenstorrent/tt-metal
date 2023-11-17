// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

#include <string>

namespace tt {

namespace tt_metal {

void dump_tensor(const std::string& file_name, const Tensor& tensor);
Tensor load_tensor(const std::string& file_name);


}  // namespace tt_metalls

}  // namespace tt
