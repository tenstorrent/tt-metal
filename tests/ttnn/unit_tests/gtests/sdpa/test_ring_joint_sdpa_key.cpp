// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks that the program key RingJointSDPA derives from its dynamic_key keeps the rules of the hash it
// replaced: logical_n leaves the key only under KV-pad rotation, and the layer fold is keyed only on the metadata path.

#include <cstdint>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"

namespace {

using ttnn::prim::RingJointSDPADeviceOperation;
using ttnn::prim::RingJointSDPAInputs;
using ttnn::prim::RingJointSDPAParams;
using AllGatherParams = ttnn::experimental::prim::RingAttentionAllGatherAsyncParams;

ttnn::Tensor make_tensor(uint32_t seq, uint32_t head_dim = 64) {
    return ttnn::zeros(ttnn::Shape({1, 1, seq, head_dim}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT);
}

ttnn::Tensor make_scalar() { return ttnn::zeros(ttnn::Shape({1}), ttnn::DataType::UINT32, ttnn::ROW_MAJOR_LAYOUT); }

RingJointSDPAParams make_params() {
    return RingJointSDPAParams(
        "rear",
        std::nullopt,
        /*is_causal=*/true,
        /*is_balanced=*/false,
        /*is_cross=*/false,
        /*logical_n=*/256,
        /*logical_l=*/0,
        /*ring_size=*/4,
        tt::tt_metal::MemoryConfig{},
        std::nullopt,
        decltype(RingJointSDPAParams::compute_kernel_config){},
        AllGatherParams(
            /*dim=*/2,
            /*num_links=*/1,
            /*ring_size=*/4,
            tt::tt_metal::MemoryConfig{},
            decltype(AllGatherParams::topology)::Ring,
            {},
            std::nullopt,
            std::nullopt),
        {},
        tt::tt_metal::CoreCoord{0, 0});
}

RingJointSDPAInputs make_inputs() {
    return RingJointSDPAInputs{
        .input_q = make_tensor(64),
        .input_k = make_tensor(64),
        .input_v = make_tensor(64),
        .gathered_k = make_tensor(256)};
}

std::string key(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return ttnn::device_operation::MeshDeviceOperationAdapter<RingJointSDPADeviceOperation>::dynamic_program_key(
        args, tensor_args);
}

TEST(RingJointSDPAKey, KeysLogicalNWithoutRotation) {
    auto args = make_params();
    const auto tensor_args = make_inputs();
    const auto short_key = key(args, tensor_args);
    args.logical_n = 512;
    EXPECT_NE(short_key, key(args, tensor_args));
}

TEST(RingJointSDPAKey, SharesLogicalNUnderRotation) {
    auto args = make_params();
    args.kv_actual_isl = 128;
    const auto tensor_args = make_inputs();
    const auto short_key = key(args, tensor_args);
    args.logical_n = 512;
    args.kv_actual_isl = 384;
    EXPECT_EQ(short_key, key(args, tensor_args));
}

TEST(RingJointSDPAKey, KeysLayerFoldOnlyOnMetadataPath) {
    auto args = make_params();
    args.kv_cache_num_layers = 4;
    auto tensor_args = make_inputs();
    const auto host_key = key(args, tensor_args);
    args.kv_cache_layer_idx = 1;
    EXPECT_EQ(host_key, key(args, tensor_args));
    args.kv_cache_num_layers = 8;
    EXPECT_EQ(host_key, key(args, tensor_args));

    tensor_args.slot_id = make_scalar();
    tensor_args.kv_actual_isl = make_scalar();
    args.kv_cache_num_layers = 4;
    args.kv_cache_layer_idx = 0;
    const auto metadata_key = key(args, tensor_args);
    args.kv_cache_layer_idx = 1;
    EXPECT_NE(metadata_key, key(args, tensor_args));
    args.kv_cache_layer_idx = 0;
    args.kv_cache_num_layers = 8;
    EXPECT_NE(metadata_key, key(args, tensor_args));
}

TEST(RingJointSDPAKey, KeysOutputMemoryConfig) {
    auto args = make_params();
    const auto tensor_args = make_inputs();
    const auto dram_key = key(args, tensor_args);
    args.output_memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1};
    EXPECT_NE(dram_key, key(args, tensor_args));
}

}  // namespace
