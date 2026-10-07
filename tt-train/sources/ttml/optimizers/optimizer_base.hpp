// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <optional>

#include "serialization/serializable.hpp"

namespace ttml::optimizers {

class OptimizerBase {
public:
    [[nodiscard]] virtual std::string get_name() const = 0;

    explicit OptimizerBase(serialization::NamedParameters&& parameters);
    OptimizerBase(const OptimizerBase&) = delete;
    OptimizerBase& operator=(const OptimizerBase&) = delete;
    OptimizerBase(OptimizerBase&&) = delete;
    OptimizerBase& operator=(OptimizerBase&&) = delete;
    virtual ~OptimizerBase() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual serialization::StateDict get_state_dict() const = 0;
    virtual void set_state_dict(const serialization::StateDict& dict) = 0;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;

    virtual void set_lr(float lr) = 0;
    [[nodiscard]] virtual float get_lr() const = 0;

    // LR recorded the first time this is called, mirroring PyTorch's
    // ``param_group["initial_lr"]`` (set once via ``setdefault``). Schedulers
    // read their base LR from here so that several schedulers attached to the
    // same optimizer (e.g. a warmup/decay chain) share one base even after an
    // earlier scheduler has already scaled ``get_lr()`` at construction.
    [[nodiscard]] float get_initial_lr();

    virtual void print_stats() const;

protected:
    // ``initial_lr`` is always serialized with the optimizer state dict
    // (mirroring PyTorch, where ``param_group["initial_lr"]`` rides along in
    // ``optimizer.state_dict()``) and is required on load. Concrete optimizers
    // call these from their get_state_dict / set_state_dict overrides.
    void save_initial_lr(serialization::StateDict& dict) const;
    void restore_initial_lr(const serialization::StateDict& dict);

    serialization::NamedParameters m_parameters;

private:
    std::optional<float> m_initial_lr;
};

// Puts `topology` back on the value of every tensor in `written`.
//
// A ttnn op labels its output with the UNION of its inputs' mesh topologies
// (ttnn::device_operation::detail::compute_output_placements_and_shape), gradient included, and a composite
// optimizer step stores those outputs in the parameter and its state through Tensor::set_value. A gradient with a
// wrong label -- a CCL output that kept a stale Shard, or a collapsed 1-D label from a default mapper -- would so
// relabel the parameter that the checkpointer later gathers by that label. Capture the parameter's topology BY
// VALUE before the step (Tensor::tensor_topology() returns a reference into attributes the ops overwrite), leave
// the step body as it is, and call this at the end of the iteration over everything the step wrote. Optimizer
// state follows its parameter's distribution, so it takes the same label.
void restore_topology(std::initializer_list<autograd::TensorPtr> written, const tt::tt_metal::TensorTopology& topology);

}  // namespace ttml::optimizers
