#include "pipeline_parallel_llama.hpp"

namespace ttml::models::distributed::pipeline_parallel_llama {

void PipelineParallelLlama::verify() const {
    auto total_blocks = std::accumulate(
        pipeline_parallel_config.blocks_per_rank.begin(),
        pipeline_parallel_config.blocks_per_rank.end(),
        0,
        [](int sum, const auto& pair) { return sum + pair.second; });
    if (pipeline_parallel_config.num_blocks != total_blocks) {
        throw std::runtime_error("Number of blocks must match number of blocks per rank");
    }
}

}  // namespace ttml::models::distributed::pipeline_parallel_llama
