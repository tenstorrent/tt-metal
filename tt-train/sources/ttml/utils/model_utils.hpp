// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "modules/module_base.hpp"

namespace ttml::utils {

using Model = std::shared_ptr<ttml::modules::ModuleBase>;

ttml::serialization::NamedParameters get_model_parameters(const Model &model);
uint64_t get_number_of_parameters(const Model &model, bool tp);

}  // namespace ttml::utils
