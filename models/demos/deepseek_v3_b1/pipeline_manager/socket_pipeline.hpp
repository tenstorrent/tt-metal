// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_interface.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// PipelineInterface backed by H2D/D2H sockets.
// Connects to pre-created sockets exported by a separate launcher process.
// inject() serializes an InjectDescriptor into a 64-byte page and writes over H2D.
// read_result() reads a 64-byte page from D2H and deserializes to ResultDescriptor.
class SocketPipeline : public PipelineInterface {
public:
    SocketPipeline(
        const std::string& h2d_socket_id, const std::string& d2h_socket_id, uint32_t connect_timeout_ms = 30000);

    ~SocketPipeline() override;

    void inject(const InjectDescriptor& desc) override;
    ResultDescriptor read_result() override;
    void reset_kv(int user_id) override;
    void request_stop() override;
    void shutdown() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
