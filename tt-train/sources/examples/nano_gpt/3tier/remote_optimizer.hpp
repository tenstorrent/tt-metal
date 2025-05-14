// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "optimizers/optimizer_base.hpp"

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;

class RemoteOptimizer : public ttml::optimizers::OptimizerBase {
public:
    RemoteOptimizer(ttml::serialization::NamedParameters parameters, int aggregator_rank) :
        ttml::optimizers::OptimizerBase(std::move(parameters)) {
        m_aggregator_rank = ttml::core::distributed::Rank{aggregator_rank};
        m_sorted_parameters = SortedParameters(m_parameters.begin(), m_parameters.end());
    }

    void zero_grad() override {
        for (auto& [name, tensor_ptr] : m_parameters) {
            if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
                // i don't see a reason why not to set it to empty
                tensor_ptr->set_grad(ttnn::Tensor());
            }
        }
    }

    void step() override {
        m_steps++;
        send_gradients();
        receive_weights();
    }

    [[nodiscard]] ttml::serialization::StateDict get_state_dict() const override {
        ttml::serialization::StateDict dict;
        dict["steps"] = m_steps;
        return dict;
    }

    void set_state_dict(const ttml::serialization::StateDict& dict) override {
        m_steps = ttml::serialization::get_value_type<size_t>(dict, "steps");
    }

    [[nodiscard]] size_t get_steps() const override {
        return m_steps;
    }

    void set_steps(size_t steps) override {
        m_steps = steps;
    }

    SortedParameters get_sorted_parameters() const {
        return m_sorted_parameters;
    }

    void send_gradients() {
        auto& ctx = ttml::autograd::ctx();
        for (auto& [name, tensor_ptr] : m_sorted_parameters) {
            if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
                auto grad = tensor_ptr->get_grad();
                ttml::core::distributed::send_tensor(grad, m_aggregator_rank);
            }
        }
    }

    void receive_weights() {
        for (auto& [name, tensor_ptr] : m_sorted_parameters) {
            auto tensor = tensor_ptr->get_value();
            ttml::core::distributed::recv_tensor(tensor, m_aggregator_rank);
            tensor_ptr->set_value(tensor);
        }
    }

    void set_lr(float lr) override {
    }

    [[nodiscard]] float get_lr() const override {
        return 0.F;
    }

private:
    size_t m_steps{0};
    SortedParameters m_sorted_parameters;
    ttml::core::distributed::Rank m_aggregator_rank{0};
};
