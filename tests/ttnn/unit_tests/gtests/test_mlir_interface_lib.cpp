#include <gtest/gtest.h>

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
