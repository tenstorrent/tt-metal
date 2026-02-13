// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "model_utils.hpp"

namespace ttml::utils {

ttml::serialization::NamedParameters get_model_parameters(const Model &model) {
    return model->parameters();
}

uint64_t get_number_of_parameters(const Model &model, bool tp) {
    auto contains = [](const std::string &str, const std::string &substr) {
        return str.find(substr) != std::string::npos;
    };
    auto parameters = get_model_parameters(model);
    uint64_t num_params = 0;
    for (const auto &[name, tensor_ptr] : parameters) {
        auto tensor = tensor_ptr->get_value();
        auto params_in_tensor = tensor.logical_volume();
        if (tp && (contains(name, "fc") || contains(name, "linear") || contains(name, "mlp/w"))) {
            auto tp_size = ttml::autograd::ctx().get_parallelism_context().get_tp_size();
            num_params += params_in_tensor * tp_size;
        } else {
            num_params += params_in_tensor;
        }
    }

    return num_params;
}

}  // namespace ttml::utils
