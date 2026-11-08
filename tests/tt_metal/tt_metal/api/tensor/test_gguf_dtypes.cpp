// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// GGUF dtypes on the host: the format table, page geometry and sharding, the from_gguf / to_gguf round trip (also of
// the expanded layouts), and GGUF tensors failing cleanly outside from_gguf / to_gguf.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {
namespace {

std::vector<DataType> gguf_dtypes() {
    std::vector<DataType> dtypes;
    for (int i = static_cast<int>(DataType::GGUF_Q8_0); i <= static_cast<int>(DataType::GGUF_TQ1_0); ++i) {
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

// The formats in DataType order are the decoder's format numbers 1 .. 25 (the traits take its page layouts).
TEST(GgufDtypesTest, FormatIdsInDataTypeOrder) {
    const auto dtypes = gguf_dtypes();
    ASSERT_EQ(dtypes.size(), 25u);
    for (size_t i = 0; i < dtypes.size(); ++i) {
        const DataType dtype = dtypes[i];
        SCOPED_TRACE(fmt::format("{}", dtype));
        EXPECT_TRUE(is_gguf(dtype));
        EXPECT_EQ(gguf_traits(dtype).format_id, i + 1);
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

// TQ2_0 blocks hold codes 0, 1, 2 only (from_gguf refuses 3): each 2-bit code of the 64 code bytes taken mod 3.
void valid_tq2_0_codes(std::vector<std::byte>& blocks) {
    for (size_t block = 0; block < blocks.size(); block += 66) {
        for (size_t j = 0; j < 64; ++j) {
            uint32_t b = std::to_integer<uint32_t>(blocks[block + j]), v = 0;
            for (int f = 0; f < 4; ++f) {
                v |= (b >> 2 * f & 3) % 3 << 2 * f;
            }
            blocks[block + j] = std::byte(v);
        }
    }
}

TEST(GgufDtypesTest, RoundTripIsBitExact) {
    for (const DataType dtype : gguf_dtypes()) {
        SCOPED_TRACE(fmt::format("{}", dtype));
        const uint32_t k = 2 * k_unit(dtype), n = 96;
        auto blocks = random_blocks(gguf_bytes(dtype, k, n), static_cast<uint32_t>(dtype));
        if (dtype == DataType::GGUF_TQ2_0) {
            valid_tq2_0_codes(blocks);
        }
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

TEST(GgufDtypesTest, Tq2_0RefusesCode3) {
    const DataType dtype = DataType::GGUF_TQ2_0;
    auto blocks = random_blocks(gguf_bytes(dtype, 512, 32), 2);
    valid_tq2_0_codes(blocks);
    EXPECT_NO_THROW(from_gguf(blocks, gguf_spec(dtype, 512, 32)));
    blocks[3 * 66 + 10] |= std::byte(0x30);  // block 3: column 1, K block 1
    EXPECT_ANY_THROW(from_gguf(blocks, gguf_spec(dtype, 512, 32)));
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

std::vector<DataType> expanded_dtypes() {
    std::vector<DataType> dtypes;
    for (int i = static_cast<int>(DataType::GGUF_TQ1_0) + 1; i < static_cast<int>(DataType::INVALID); ++i) {
        dtypes.push_back(static_cast<DataType>(i));
    }
    return dtypes;
}

// TQ1_0 blocks as TQ1_0 quantization writes them: each base-3 byte ceil(256 v / 243) of a 5-trit value v (the last
// trit 0 in the 4 bytes after the 48), then d.
std::vector<std::byte> tq1_0_blocks(size_t blocks, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::byte> out(54 * blocks);
    for (size_t b = 0; b < blocks; ++b) {
        for (int i = 0; i < 52; ++i) {
            const uint32_t v = i < 48 ? rng() % 243 : rng() % 81 * 3;
            out[54 * b + i] = std::byte((v * 256 + 242) / 243);
        }
        out[54 * b + 52] = std::byte(rng() & 0xFF);
        out[54 * b + 53] = std::byte(0x20);
    }
    return out;
}

TEST(GgufDtypesTest, ExpandedLayouts) {
    const auto dtypes = expanded_dtypes();
    ASSERT_EQ(dtypes.size(), 8u);
    for (const DataType dtype : dtypes) {
        SCOPED_TRACE(fmt::format("{}", dtype));
        const DataType source = gguf_source_dtype(dtype);
        EXPECT_TRUE(is_gguf(dtype) && is_gguf_expanded(dtype) && !is_gguf_expanded(source));
        const auto& f = gguf_traits(dtype);
        const auto& s = gguf_traits(source == DataType::GGUF_TQ1_0 ? DataType::GGUF_TQ2_0 : source);
        EXPECT_EQ(f.format_id, s.format_id);
        EXPECT_EQ(f.l1_page_bytes(), s.l1_page_bytes());
        EXPECT_GE(f.block_bytes, s.block_bytes);
        EXPECT_EQ(f.page_bytes() % 64, 0u);

        const uint32_t k = 2 * k_unit(source), n = 96;
        const auto blocks = source == DataType::GGUF_TQ1_0
                                ? tq1_0_blocks(gguf_bytes(source, k, n) / 54, 7)
                                : random_blocks(gguf_bytes(source, k, n), static_cast<uint32_t>(dtype));
        const HostTensor tensor = from_gguf(blocks, gguf_spec(dtype, k, n));
        EXPECT_EQ(gguf_spec(dtype, k, n).compute_packed_buffer_size_bytes(), gguf_bytes(dtype, k, n));
        EXPECT_EQ(to_gguf(tensor), blocks);
    }
    // A TQ1_0 byte that quantization never writes has no TQ2_0 form that gives it back.
    auto blocks = tq1_0_blocks(96, 8);
    blocks[5] = std::byte(1);  // ceil(256 v / 243) is never 1
    EXPECT_ANY_THROW(from_gguf(blocks, gguf_spec(DataType::GGUF_TQ1_0_EXPANDED, 256, 96)));
}

}  // namespace
}  // namespace tt::tt_metal
