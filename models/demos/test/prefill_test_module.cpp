// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone, test-only Python extension (`_prefill_test`) for the layer-completion scheduler
// stand-in. Deliberately NOT part of the ttnn module / public API: it is built as its own .so and
// imported via `models.demos.test.prefill_test`. See layer_completion_consumer.hpp for rationale.

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "layer_completion_consumer.hpp"

namespace nb = nanobind;

NB_MODULE(_prefill_test, mod) {
    using tt::tests::prefill_test::LayerCompletionConsumer;

    mod.doc() = "Test-only layer-completion consumer (scheduler stand-in). Not a ttnn API.";

    nb::class_<LayerCompletionConsumer>(mod, "LayerCompletionConsumer")
        .def(
            "__init__",
            [](LayerCompletionConsumer* self,
               const std::string& channel_shm_name,
               uint64_t expected,
               uint32_t connect_timeout_ms,
               uint64_t log_step) {
                new (self) LayerCompletionConsumer(channel_shm_name, expected, connect_timeout_ms, log_step);
            },
            nb::arg("channel_shm_name"),
            nb::arg("expected"),
            nb::arg("connect_timeout_ms") = 30'000u,
            nb::arg("log_step") = 61u,
            "Test/scheduler stand-in: connect to the scheduler counter channel and drain it on a NATIVE "
            "C++ thread (GIL-immune), self-terminating once `expected` completions are drained.")
        .def("stop", &LayerCompletionConsumer::stop, "Idempotent: stop + join + final drain + shutdown channel.")
        .def_prop_ro("total", &LayerCompletionConsumer::total)
        .def_prop_ro("reached_expected", &LayerCompletionConsumer::reached_expected);
}
