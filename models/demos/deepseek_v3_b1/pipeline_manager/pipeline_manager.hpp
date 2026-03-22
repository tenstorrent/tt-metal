#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

struct PipelineManagerRequest {
    std::string request_id;
    std::vector<uint32_t> prompt_token_ids;
    uint32_t max_new_tokens = 0;
    std::optional<uint32_t> eos_token_id;
};

class PipelineManager {
public:
    PipelineManager(
        std::string h2d_socket_id, std::string d2h_socket_id, uint32_t page_size_bytes, uint32_t connect_timeout_ms);
    ~PipelineManager();

    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    void start();
    void stop();
    void write_token(uint32_t token_id);
    uint32_t read_token();
    void run_one_shot(PipelineManagerRequest& request, std::ostream& output_stream);
    void write_over_socket(uint32_t token_id);
    uint32_t read_over_socket();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
