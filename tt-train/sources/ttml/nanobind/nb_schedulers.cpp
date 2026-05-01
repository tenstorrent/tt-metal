// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nb_schedulers.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "optimizers/optimizer_base.hpp"
#include "schedulers/cosine_annealing_scheduler.hpp"
#include "schedulers/lambda_scheduler.hpp"
#include "schedulers/linear_scheduler.hpp"
#include "schedulers/scheduler_base.hpp"
#include "schedulers/step_scheduler.hpp"
#include "serialization/serializable.hpp"

// SequentialScheduler is intentionally not exposed here.  It takes
// std::vector<std::unique_ptr<LRSchedulerBase>> which requires transferring
// ownership out of Python-managed objects — non-trivial with nanobind and best
// handled with a dedicated factory when needed.

namespace ttml::nanobind::schedulers {
using namespace ttml::schedulers;

namespace {

// Convert a C++ StateDict (whose values are always ValueType primitives for
// schedulers) to a plain Python dict.  The outer SerializableType variant is
// not registered with nanobind (it contains opaque Tensor / TensorPtr
// alternatives), so we extract the ValueType alternative manually.
nb::dict state_dict_to_py(const serialization::StateDict& dict) {
    nb::dict result;
    for (const auto& [key, serializable] : dict) {
        const auto& vt = std::get<serialization::ValueType>(serializable);
        nb::object py_val = std::visit(
            [](const auto& v) -> nb::object {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, bfloat16>) {
                    return nb::cast(static_cast<float>(v));
                } else {
                    return nb::cast(v);
                }
            },
            vt);
        result[key.c_str()] = py_val;
    }
    return result;
}

// Convert a plain Python dict back to a C++ StateDict.
// Python bool  → bool
// Python int   → size_t  (step counters are always size_t)
// Python float → float
// Python str   → std::string
// Note: check bool before int — Python bool is a subclass of int.
serialization::StateDict py_to_state_dict(const nb::dict& d) {
    serialization::StateDict result;
    for (auto [k, v] : d) {
        auto key = nb::cast<std::string>(k);
        if (nb::isinstance<nb::bool_>(v)) {
            result[key] = serialization::ValueType(nb::cast<bool>(v));
        } else if (nb::isinstance<nb::int_>(v)) {
            result[key] = serialization::ValueType(nb::cast<size_t>(v));
        } else if (nb::isinstance<nb::float_>(v)) {
            result[key] = serialization::ValueType(nb::cast<float>(v));
        } else if (nb::isinstance<nb::str>(v)) {
            result[key] = serialization::ValueType(nb::cast<std::string>(v));
        }
    }
    return result;
}

}  // namespace

void py_module_types(nb::module_& m) {
    nb::class_<LRSchedulerBase>(m, "LRSchedulerBase");
    nb::class_<CosineAnnealingScheduler, LRSchedulerBase>(m, "CosineAnnealingScheduler");
    nb::class_<StepScheduler, LRSchedulerBase>(m, "StepScheduler");
    nb::class_<LinearScheduler, LRSchedulerBase>(m, "LinearScheduler");
    nb::class_<LambdaScheduler, LRSchedulerBase>(m, "LambdaScheduler");
}

void py_module(nb::module_& m) {
    {
        auto py_base = static_cast<nb::class_<LRSchedulerBase>>(m.attr("LRSchedulerBase"));
        py_base.def("step", &LRSchedulerBase::step, "Advance the scheduler by one step");
        py_base.def("get_last_lr", &LRSchedulerBase::get_last_lr, "Return the LR set by the last step() call");
        py_base.def("get_current_lr", &LRSchedulerBase::get_current_lr, "Return the optimizer's current LR");
        py_base.def(
            "get_state_dict",
            [](const LRSchedulerBase& self) { return state_dict_to_py(self.get_state_dict()); },
            "Return scheduler state as a plain Python dict");
        py_base.def(
            "set_state_dict",
            [](LRSchedulerBase& self, const nb::dict& d) { self.set_state_dict(py_to_state_dict(d)); },
            nb::arg("dict"),
            "Restore scheduler state from a plain Python dict");
    }

    {
        auto py_cosine =
            static_cast<nb::class_<CosineAnnealingScheduler, LRSchedulerBase>>(m.attr("CosineAnnealingScheduler"));
        py_cosine.def(
            nb::init<optimizers::OptimizerBase*, size_t, float>(),
            nb::arg("optimizer"),
            nb::arg("T_max"),
            nb::arg("eta_min") = 0.F,
            "Cosine annealing: decays LR from base_lr to eta_min over T_max steps then restarts.\n\n"
            "Args:\n"
            "    optimizer: Optimizer whose LR is managed.\n"
            "    T_max: Steps in one cosine half-cycle.\n"
            "    eta_min: Minimum LR (default 0).");
    }

    {
        auto py_step = static_cast<nb::class_<StepScheduler, LRSchedulerBase>>(m.attr("StepScheduler"));
        py_step.def(
            nb::init<optimizers::OptimizerBase*, size_t, float>(),
            nb::arg("optimizer"),
            nb::arg("step_size"),
            nb::arg("gamma") = 0.1F,
            "Step decay: multiplies LR by gamma every step_size steps.\n\n"
            "Args:\n"
            "    optimizer: Optimizer whose LR is managed.\n"
            "    step_size: Period of LR decay.\n"
            "    gamma: Multiplicative factor (default 0.1).");
    }

    {
        auto py_linear = static_cast<nb::class_<LinearScheduler, LRSchedulerBase>>(m.attr("LinearScheduler"));
        py_linear.def(
            nb::init<optimizers::OptimizerBase*, float, float, size_t>(),
            nb::arg("optimizer"),
            nb::arg("start_factor"),
            nb::arg("end_factor"),
            nb::arg("total_steps"),
            "Linear decay: scales LR by a factor that moves linearly from start_factor to end_factor\n"
            "over total_steps steps.\n\n"
            "Args:\n"
            "    optimizer: Optimizer whose LR is managed.\n"
            "    start_factor: Initial multiplier applied to base_lr.\n"
            "    end_factor: Final multiplier applied to base_lr.\n"
            "    total_steps: Number of steps over which to interpolate.");
    }

    {
        auto py_lambda = static_cast<nb::class_<LambdaScheduler, LRSchedulerBase>>(m.attr("LambdaScheduler"));
        py_lambda.def(
            nb::init<optimizers::OptimizerBase*, std::function<float(int)>>(),
            nb::arg("optimizer"),
            nb::arg("lr_lambda"),
            "Lambda scheduler: sets LR = base_lr * lr_lambda(step) on each step.\n\n"
            "Args:\n"
            "    optimizer: Optimizer whose LR is managed.\n"
            "    lr_lambda: Callable(int) -> float returning the multiplicative factor for the given step.");
    }
}

}  // namespace ttml::nanobind::schedulers
