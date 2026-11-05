// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// GGUF dtypes on the host: the format table against the Blackhole decoder's, page geometry and sharding, the
// from_gguf / to_gguf round trip, and GGUF tensors failing cleanly outside from_gguf / to_gguf.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"

namespace tt::tt_metal {
namespace {

std::vector<DataType> gguf_dtypes() {
    std::vector<DataType> dtypes;
    for (int i = static_cast<int>(DataType::GGUF_Q8_0); i < static_cast<int>(DataType::INVALID); ++i) {
        dtypes.push_back(static_cast<DataType>(i));
    }
    return dtypes;
}

// The smallest K whose GGUF rows split into whole pages and whole blocks.
uint32_t k_unit(DataType dtype) {
    const auto& f = gguf_traits(dtype);
    return std::lcm(f.page_height(), f.block_elems);
}

size_t gguf_bytes(DataType dtype, uint32_t k, uint32_t n) {
    const auto& f = gguf_traits(dtype);
    return size_t(n) * k * f.block_bytes / f.block_elems;
}

TensorSpec gguf_spec(DataType dtype, uint32_t k, uint32_t n, const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(Shape({k, n}), TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

std::vector<std::byte> random_blocks(size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::byte> blocks(bytes);
    for (auto& b : blocks) {
        b = std::byte(rng() & 0xFF);
    }
    return blocks;
}

template <std::uint32_t... Ids>
ckernel::gguf::PageLayout decoder_layout(uint32_t id, std::integer_sequence<std::uint32_t, Ids...>) {
    ckernel::gguf::PageLayout layout{};
    ((id == Ids + 1 ? (layout = ckernel::gguf::page_layout<static_cast<ckernel::gguf::Format>(Ids + 1)>(), 0) : 0),
     ...);
    return layout;
}

TEST(GgufDtypesTest, TraitsMatchDecoderPageLayout) {
    const auto dtypes = gguf_dtypes();
    ASSERT_EQ(dtypes.size(), 25u);
    for (size_t i = 0; i < dtypes.size(); ++i) {
        const DataType dtype = dtypes[i];
        SCOPED_TRACE(fmt::format("{}", dtype));
        EXPECT_TRUE(is_gguf(dtype));
        const auto& f = gguf_traits(dtype);
        EXPECT_EQ(f.format_id, i + 1);
        const auto layout = decoder_layout(f.format_id, std::make_integer_sequence<std::uint32_t, 25>{});
        EXPECT_EQ(f.group_tiles, layout.group_tiles);
        EXPECT_EQ(f.page_tiles, layout.page_tiles);
        EXPECT_EQ(f.code_bytes, layout.code_bytes);
        EXPECT_EQ(f.scale_rows, layout.scale_rows);
    }
    EXPECT_FALSE(is_gguf(DataType::MXINT2));
    EXPECT_FALSE(is_gguf(DataType::INVALID));
}

TEST(GgufDtypesTest, PageIsOneGgufRecord) {
    for (const DataType dtype : gguf_dtypes()) {
        SCOPED_TRACE(fmt::format("{}", dtype));
        const auto& f = gguf_traits(dtype);
        const uint32_t k = 2 * k_unit(dtype), n = 96;
        const auto spec = gguf_spec(dtype, k, n);
        EXPECT_EQ(spec.compute_page_size_bytes(), f.page_bytes());
        EXPECT_EQ(f.page_bytes(), gguf_bytes(dtype, f.page_height(), 32));
        EXPECT_EQ(f.page_bytes() % 64, 0u);
        EXPECT_EQ(spec.compute_packed_buffer_size_bytes(), gguf_bytes(dtype, k, n));
        EXPECT_EQ(spec.physical_shape(), Shape2D(k, n));
        EXPECT_EQ(datatype_to_dataformat_converter(dtype), tt::DataFormat::Invalid);
    }
}

TEST(GgufDtypesTest, WidthShardedDramPages) {
    const DataType dtype = DataType::GGUF_Q4_K;                       // 256-row pages
    const uint32_t k = 512, n = 32 * 5, banks = 2, shard_n = 3 * 32;  // the last shard holds 2 of 3 N tiles
    const CoreRangeSet grid(CoreRange({0, 0}, {banks - 1, 0}));
    const MemoryConfig memory_config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        ShardSpec(grid, {k, shard_n}, ShardOrientation::ROW_MAJOR));
    const auto spec = gguf_spec(dtype, k, n, memory_config);
    const auto args = spec.compute_buffer_sharding_args();
    ASSERT_TRUE(args.shard_spec().has_value());
    EXPECT_EQ(args.shard_spec()->page_shape, (std::array<uint32_t, 2>{256, 32}));
    EXPECT_EQ(args.shard_spec()->tensor2d_shape_in_pages, (std::array<uint32_t, 2>{2, 5}));
    EXPECT_EQ(spec.compute_packed_buffer_size_bytes(), gguf_bytes(dtype, k, n));

    const MemoryConfig bad_height(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        ShardSpec(grid, {128, shard_n}, ShardOrientation::ROW_MAJOR));
    EXPECT_ANY_THROW(gguf_spec(dtype, k, n, bad_height));
}

TEST(GgufDtypesTest, RoundTripIsBitExact) {
    for (const DataType dtype : gguf_dtypes()) {
        SCOPED_TRACE(fmt::format("{}", dtype));
        const uint32_t k = 2 * k_unit(dtype), n = 96;
        const auto blocks = random_blocks(gguf_bytes(dtype, k, n), static_cast<uint32_t>(dtype));
        const HostTensor tensor = from_gguf(blocks, gguf_spec(dtype, k, n));
        EXPECT_EQ(tensor.dtype(), dtype);
        EXPECT_EQ(to_gguf(tensor), blocks);
    }
}

// Page (g, nt) holds the GGUF bytes of K group g of N tile nt, at page index g * N / 32 + nt.
TEST(GgufDtypesTest, PagesAreInKGroupMajorOrder) {
    const DataType dtype = DataType::GGUF_Q8_0;  // one 34-byte block per column per page
    const uint32_t k = 64, n = 96, nt = 1, g = 1, n_tiles = n / 32, row_blocks = k / 32;
    std::vector<std::byte> blocks(gguf_bytes(dtype, k, n));
    for (uint32_t col = 32 * nt; col < 32 * nt + 32; ++col) {
        for (uint32_t i = 0; i < 34; ++i) {
            blocks[(col * row_blocks + g) * 34 + i] = std::byte(0xA5);
        }
    }
    const HostTensor tensor = from_gguf(blocks, gguf_spec(dtype, k, n));
    const auto pages = host_buffer::get_host_buffer(tensor);
    const auto bytes = pages.view_bytes();
    const size_t page_bytes = gguf_traits(dtype).page_bytes();
    for (size_t page = 0; page < bytes.size() / page_bytes; ++page) {
        size_t set = 0;
        for (size_t i = 0; i < page_bytes; ++i) {
            set += bytes[page * page_bytes + i] != std::byte(0);
        }
        EXPECT_EQ(set != 0, page == g * n_tiles + nt) << "page " << page;
    }
}

TEST(GgufDtypesTest, RejectsShapesThePagesCannotHold) {
    const DataType dtype = DataType::GGUF_Q6_K;
    EXPECT_ANY_THROW(from_gguf(random_blocks(gguf_bytes(dtype, 256, 48), 0), gguf_spec(dtype, 256, 48)));
    EXPECT_ANY_THROW(from_gguf(random_blocks(gguf_bytes(dtype, 256, 32) - 1, 0), gguf_spec(dtype, 256, 32)));
    EXPECT_ANY_THROW(from_gguf(random_blocks(gguf_bytes(dtype, 512, 32), 0), gguf_spec(dtype, 384, 32)));
    EXPECT_ANY_THROW(TensorSpec(Shape({256, 32}), TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
}

TEST(GgufDtypesTest, UnawareHostOpsFail) {
    const DataType dtype = DataType::GGUF_Q4_0;
    const HostTensor tensor = from_gguf(random_blocks(gguf_bytes(dtype, 32, 32), 1), gguf_spec(dtype, 32, 32));
    EXPECT_ANY_THROW(to_dtype(tensor, DataType::BFLOAT16));
    EXPECT_ANY_THROW(to_layout(tensor, Layout::ROW_MAJOR));
    EXPECT_ANY_THROW(tensor.to_vector<float>());
    EXPECT_ANY_THROW(tile_size(dtype));
}

}  // namespace
}  // namespace tt::tt_metal
