// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

Program MakeProgramFromSpec(const ProgramSpec& spec) {
    (void)spec;  // Suppress unused parameter warning for stub implementation
    auto impl = std::make_shared<detail::ProgramImpl>();

    /*
    for (const auto& worker : spec.workers) {
        // TODO: Add semaphores
        for (const auto& semaphore_spec : worker.semaphores) {
            (void)semaphore_spec;  // Placeholder
        }

        // TODO: Add dataflow buffers
        for (const auto& dfb_spec : worker.dataflow_buffers) {
            (void)dfb_spec;  // Placeholder
        }

        // TODO: Add kernels
        for (const auto& kernel_spec : worker.kernels) {
            (void)kernel_spec;  // Placeholder
        }
    }
    */

    return Program(std::move(impl));
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
