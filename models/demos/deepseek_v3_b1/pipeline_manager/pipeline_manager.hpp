// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

class PipelineInterface;

class PipelineManager {
public:
    PipelineManager(PipelineInterface& pipeline, int chunk_size = DEFAULT_CHUNK_SIZE);
    ~PipelineManager();

    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    void start();
    void stop();

    bool push_request(const ISRequest& request);
    bool try_pop_response(PMResponse& response);
    bool try_pop_output(OutputMessage& output);

    // Drain request queue and apply API requests to internal state.
    // Call periodically from the main/API thread.
    void tick();

    // Query the current state of a user slot. Safe to call from any thread.
    UserState get_user_state(int user_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
