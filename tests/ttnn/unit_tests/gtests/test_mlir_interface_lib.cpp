#include <gtest/gtest.h>

#include <cstddef>

#include "mlir_interface_api.hpp"

static void compare_cb_allocations(
    const std::vector<uint32_t>& expected, const std::optional<std::vector<uint32_t>>& output) {
    EXPECT_TRUE(output.has_value());
    for (auto x : output.value()) {
        std::cout << "CB " << x << std::endl;
    }
    EXPECT_EQ(expected.size(), output.value().size());
    for (int i = 0; i < expected.size(); i++) {
        EXPECT_EQ(expected[i], output.value()[i]);
    }
}

TEST(MLIR_INTERFACE_API, binary_op) {
    std::vector<uint32_t> shape = {1, 1, 32, 32};
    ttnn::mlir_interface::memory_config_tuple memory_config = {"interleaved", "l1", std::nullopt};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096, 4096},
        ttnn::mlir_interface::get_binary_circular_buffers_l1_allocations(
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout));
    EXPECT_TRUE(ttnn::mlir_interface::get_binary_tensor_buffers_l1_allocations(
                    shape,
                    memory_config,
                    data_type,
                    layout,
                    shape,
                    memory_config,
                    data_type,
                    layout,
                    shape,
                    memory_config,
                    data_type,
                    layout)
                    .has_value());
}

TEST(MLIR_INTERFACE_API, binary_op_batch_broadcast) {
    std::vector<uint32_t> shape_a = {2, 1, 32, 32};
    std::vector<uint32_t> shape_b = {1, 1, 32, 32};
    std::vector<uint32_t> shape_b_invalid = {5, 1, 32, 32};
    ttnn::mlir_interface::memory_config_tuple memory_config = {"interleaved", "dram", std::nullopt};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape_a, memory_config, data_type, shape_b, memory_config, data_type, memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096, 4096, 4096},
        ttnn::mlir_interface::get_binary_circular_buffers_l1_allocations(
            shape_a,
            memory_config,
            data_type,
            layout,
            shape_b,
            memory_config,
            data_type,
            layout,
            shape_a,
            memory_config,
            data_type,
            layout));

    EXPECT_FALSE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape_a, memory_config, data_type, shape_b_invalid, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_width_sharded) {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 64 * 5};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
    ttnn::mlir_interface::memory_config_tuple memory_config = {"width_sharded", "l1", shard_spec};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
    compare_cb_allocations(
        {0, 0, 0},
        ttnn::mlir_interface::get_binary_circular_buffers_l1_allocations(
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout));
}

TEST(MLIR_INTERFACE_API, binary_op_height_sharded) {
    std::vector<uint32_t> shape = {1, 64, 32 * 5, 32};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32 * 5, 32}, "col_major", false};
    ttnn::mlir_interface::memory_config_tuple memory_config = {"height_sharded", "l1", shard_spec};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
    compare_cb_allocations(
        {0, 0, 0},
        ttnn::mlir_interface::get_binary_circular_buffers_l1_allocations(
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout));
}

TEST(MLIR_INTERFACE_API, binary_op_block_sharded) {
    std::vector<uint32_t> shape = {1, 1, 32 * 5 * 2, 32 * 3 * 2};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 2, 2}}, {32 * 5, 32 * 3}, "row_major", false};
    ttnn::mlir_interface::memory_config_tuple memory_config = {"block_sharded", "l1", shard_spec};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
        shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
    compare_cb_allocations(
        {0, 0, 0},
        ttnn::mlir_interface::get_binary_circular_buffers_l1_allocations(
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout,
            shape,
            memory_config,
            data_type,
            layout));
}

TEST(MLIR_INTERFACE_API, unary_op) {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
    ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
    ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"width_sharded", "l1", shard_spec};
    ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"interleaved", "dram", std::nullopt};
    std::string data_type = "bf16";
    std::string layout = "tile";

    EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096},
        ttnn::mlir_interface::get_unary_circular_buffers_l1_allocations(
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout));
    EXPECT_TRUE(ttnn::mlir_interface::get_unary_tensor_buffers_l1_allocations(
                    shape,
                    l1_interleaved_memory_config,
                    data_type,
                    layout,
                    shape,
                    l1_interleaved_memory_config,
                    data_type,
                    layout)
                    .has_value());

    EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_sharded_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
    compare_cb_allocations(
        {0, 0},
        ttnn::mlir_interface::get_unary_circular_buffers_l1_allocations(
            shape, l1_sharded_memory_config, data_type, layout, shape, l1_sharded_memory_config, data_type, layout));

    EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
    EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_sharded_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));

    EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, dram_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096},
        ttnn::mlir_interface::get_unary_circular_buffers_l1_allocations(
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout));

    EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_sharded_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, dram_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));

    EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, dram_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096},
        ttnn::mlir_interface::get_unary_circular_buffers_l1_allocations(
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout));

    EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
        "RELU", shape, l1_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {4096, 4096},
        ttnn::mlir_interface::get_unary_circular_buffers_l1_allocations(
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout));
}

TEST(MLIR_INTERFACE_API, softmax_op) {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
    ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
    ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"width_sharded", "l1", shard_spec};
    ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"interleaved", "dram", std::nullopt};
    std::string data_type = "bf16";
    std::string layout = "tile";
    const int dim_arg = -1;

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));
    EXPECT_TRUE(ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
                    shape,
                    l1_interleaved_memory_config,
                    data_type,
                    layout,
                    shape,
                    l1_interleaved_memory_config,
                    data_type,
                    layout,
                    dim_arg)
                    .has_value());

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_sharded_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_sharded_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, dram_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_sharded_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, dram_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_sharded_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, dram_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));

    EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
        shape, l1_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
    compare_cb_allocations(
        {32768, 32768, 2048, 2048, 655360, 2048},
        ttnn::mlir_interface::get_softmax_circular_buffers_l1_allocations(
            shape,
            l1_interleaved_memory_config,
            data_type,
            layout,
            shape,
            dram_interleaved_memory_config,
            data_type,
            layout,
            dim_arg));
}

TEST(MLIR_INTERFACE_API, matmul_multicast) {
    // [n x k], [k x m] -> [n x m]
    // size_t n = 2 * 64 * 32, k = 64 * 32, m = 5 * 64 * 32;
    // std::vector<uint32_t> shape_a = ttnn::Shape(tt::tt_metal::Array4D{1, 1, n, k});
    // std::vector<uint32_t> shape_b = ttnn::Shape(tt::tt_metal::Array4D{1, 1, k, m});
    // std::vector<uint32_t> shape_o = ttnn::Shape(tt::tt_metal::Array4D{1, 1, n, m});

    // ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};

    // ttnn::mlir_interface::matmul_multicore_reuse_config_tuple matmul_config = {{8,8}, 1, 1, 4, 8, 16, false, false}

    // ttnn::mlir_interface::does_matmul_multicore_reuse_multicast_support_input_output_constraints()
}

TEST(MLIR_INTERFACE_API, matmul_multicast_1d) {
    // ttnn::mlir_interface::does_matmul_multicore_reuse_multicast_1d_op_support_input_output_constraints()
}
