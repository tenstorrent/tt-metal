#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"

#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> bert_large_split_fused_qkv(const Tensor& a, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 3072})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    std::vector<Tensor> output = multi_core_split_fused_qkv(a, mem_config, compute_and_storage_grid_size);
    return output;
}

// Q and V heads use transpose_hw=false, while K head requires the additional transpose with transpose_hw=true.
Tensor bert_large_create_q_head(const Tensor& a, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/false);
    return output;
}
Tensor bert_large_create_k_head(const Tensor& a, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/true);
    return output;
}
Tensor bert_large_create_v_head(const Tensor& a, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/false);
    return output;
}

Tensor bert_large_concat_heads(const Tensor& a, const MemoryConfig& mem_config) {
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_concat_heads(a, mem_config, compute_and_storage_grid_size);
    return output;
}

} // namespace tt_metal

} // namespace tt
