#include <gtest/gtest.h>
#include <cstddef>

#include "mlir_interface_api.hpp"


TEST(MLIR_INTERFACE_API, binary_op)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"interleaved", "l1", std::nullopt};
  std::string data_type = "bf16";
  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_batch_broadcast)
{
  std::vector<uint32_t> shape_a = {2, 1, 32, 32};
  std::vector<uint32_t> shape_b = {1, 1, 32, 32};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"interleaved", "dram", std::nullopt};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape_a, memory_config, data_type, shape_b, memory_config, data_type, memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape_b, memory_config, data_type, shape_a, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_width_sharded)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32 * 64 * 5};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"width_sharded", "l1", shard_spec};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_height_sharded)
{
  std::vector<uint32_t> shape = {1, 64, 32 * 5, 32};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32 * 5, 32}, "col_major", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"height_sharded", "l1", shard_spec};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_block_sharded)
{
  std::vector<uint32_t> shape = {1, 1, 32 * 5 * 2, 32 * 3 * 2};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 2, 2}}, {32 * 5, 32 * 3}, "row_major", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"block_sharded", "l1", shard_spec};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, unary_op)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
  ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
  ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"width_sharded", "l1", shard_spec};
  ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"interleaved", "dram", std::nullopt};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_sharded_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_sharded_memory_config, data_type, shape, l1_sharded_memory_config, data_type));

  EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, dram_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_sharded_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, dram_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));

  EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, dram_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_unary_op_support_input_output_constraints("RELU", shape, l1_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, softmax_op)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
  ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
  ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"width_sharded", "l1", shard_spec};
  ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"interleaved", "dram", std::nullopt};
  std::string data_type = "bf16";

  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_sharded_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_sharded_memory_config, data_type, shape, l1_sharded_memory_config, data_type));

  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, dram_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_sharded_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, dram_interleaved_memory_config, data_type, shape, l1_sharded_memory_config, data_type));

  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, dram_interleaved_memory_config, data_type, shape, l1_interleaved_memory_config, data_type));
  EXPECT_TRUE(ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(shape, l1_interleaved_memory_config, data_type, shape, dram_interleaved_memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, matmul_multicast)
{
  // [n x k], [k x m] -> [n x m]
  // size_t n = 2 * 64 * 32, k = 64 * 32, m = 5 * 64 * 32;
  // std::vector<uint32_t> shape_a = ttnn::Shape(tt::tt_metal::Array4D{1, 1, n, k});
  // std::vector<uint32_t> shape_b = ttnn::Shape(tt::tt_metal::Array4D{1, 1, k, m});
  // std::vector<uint32_t> shape_o = ttnn::Shape(tt::tt_metal::Array4D{1, 1, n, m});

  // ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"interleaved", "l1", std::nullopt};

  // ttnn::mlir_interface::matmul_multicore_reuse_config_tuple matmul_config = {{8,8}, 1, 1, 4, 8, 16, false, false}

  // ttnn::mlir_interface::does_matmul_multicore_reuse_multicast_support_input_output_constraints()
}

TEST(MLIR_INTERFACE_API, matmul_multicast_1d)
{
  // ttnn::mlir_interface::does_matmul_multicore_reuse_multicast_1d_op_support_input_output_constraints()

}
