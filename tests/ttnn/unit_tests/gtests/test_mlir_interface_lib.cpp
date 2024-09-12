#include <gtest/gtest.h>
#include <cstddef>

#include "mlir_interface_api.hpp"

// Demonstrate some basic assertions.
TEST(MLIR_INTERFACE_API, add) {
  EXPECT_EQ(12 + 21, ttnn::mlir_interface::add(12, 21));
  EXPECT_EQ(-1 + 2, ttnn::mlir_interface::add(-1, 2));
}

TEST(MLIR_INTERFACE_API, subtract) {
  EXPECT_EQ(12 - 21, ttnn::mlir_interface::subtract(12, 21));
  EXPECT_EQ(-1 - 2, ttnn::mlir_interface::subtract(-1, 2));
}

TEST(MLIR_INTERFACE_API, dummy_check) {
  EXPECT_TRUE(ttnn::mlir_interface::dummy_check("INTERLEAVED", "DRAM"));
  EXPECT_TRUE(ttnn::mlir_interface::dummy_check("WIDTH_SHARDED", "L1"));
  EXPECT_TRUE(ttnn::mlir_interface::dummy_check("HEIGHT_SHARDED", "L1"));
  EXPECT_TRUE(ttnn::mlir_interface::dummy_check("BLOCK_SHARDED", "L1"));

  EXPECT_FALSE(ttnn::mlir_interface::dummy_check("INTERLEAVED", "L1"));
  EXPECT_FALSE(ttnn::mlir_interface::dummy_check("WIDTH_SHARDED", "DRAM"));
  EXPECT_FALSE(ttnn::mlir_interface::dummy_check("HEIGHT_SHARDED", "DRAM"));
  EXPECT_FALSE(ttnn::mlir_interface::dummy_check("BLOCK_SHARDED", "DRAM"));
  EXPECT_FALSE(ttnn::mlir_interface::dummy_check("INVALID", "INVALID"));
}

TEST(MLIR_INTERFACE_API, binary_op)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"INTERLEAVED", "L1", std::nullopt};
  std::string data_type = "BFLOAT16";
  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_batch_broadcast)
{
  std::vector<uint32_t> shape_a = {2, 1, 32, 32};
  std::vector<uint32_t> shape_b = {1, 1, 32, 32};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"INTERLEAVED", "DRAM", std::nullopt};
  std::string data_type = "BFLOAT16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape_a, memory_config, data_type, shape_b, memory_config, data_type, memory_config, data_type));
  EXPECT_FALSE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape_b, memory_config, data_type, shape_a, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_width_sharded)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32 * 64 * 5};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 8, 8}}, {32, 32 * 5}, "COL_MAJOR", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"WIDTH_SHARDED", "L1", shard_spec};
  std::string data_type = "BFLOAT16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_height_sharded)
{
  std::vector<uint32_t> shape = {1, 64, 32 * 5, 32};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 8, 8}}, {32 * 5, 32}, "COL_MAJOR", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"HEIGHT_SHARDED", "L1", shard_spec};
  std::string data_type = "BFLOAT16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, binary_op_block_sharded)
{
  std::vector<uint32_t> shape = {1, 1, 32 * 5 * 2, 32 * 3 * 2};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 2, 2}}, {32 * 5, 32 * 3}, "ROW_MAJOR", false};
  ttnn::mlir_interface::memory_config_tuple memory_config = {"BLOCK_SHARDED", "L1", shard_spec};
  std::string data_type = "BFLOAT16";

  EXPECT_TRUE(ttnn::mlir_interface::does_binary_op_support_input_output_constraints(shape, memory_config, data_type, shape, memory_config, data_type, memory_config, data_type));
}

TEST(MLIR_INTERFACE_API, unary_op)
{
  std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
  ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"INTERLEAVED", "L1", std::nullopt};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 8, 8}}, {32, 32 * 5}, "COL_MAJOR", false};
  ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"WIDTH_SHARDED", "L1", shard_spec};
  ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"INTERLEAVED", "DRAM", std::nullopt};
  std::string data_type = "BFLOAT16";

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
  ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {"INTERLEAVED", "L1", std::nullopt};
  ttnn::mlir_interface::shard_spec_tuple shard_spec = {{{0, 0, 8, 8}}, {32, 32 * 5}, "COL_MAJOR", false};
  ttnn::mlir_interface::memory_config_tuple l1_sharded_memory_config = {"WIDTH_SHARDED", "L1", shard_spec};
  ttnn::mlir_interface::memory_config_tuple dram_interleaved_memory_config = {"INTERLEAVED", "DRAM", std::nullopt};
  std::string data_type = "BFLOAT16";

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
