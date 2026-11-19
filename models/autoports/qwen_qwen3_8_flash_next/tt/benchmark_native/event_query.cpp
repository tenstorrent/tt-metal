// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <tt-metalium/distributed.hpp>

namespace nb = nanobind;
NB_MODULE(_benchmark_event_query, module) {
    module.def(
        "event_query",
        &tt::tt_metal::distributed::EventQuery,
        nb::arg("event"),
        nb::call_guard<nb::gil_scoped_release>());
}
