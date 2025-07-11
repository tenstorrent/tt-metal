// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

std::array<std::tuple<TensorSpec, BufferType>, 19> tensor_specs = {
    // Basic sanity configs
    std::tuple{
        TensorSpec(
            ttnn::Shape({3, 2, 32, 128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({3, 2, 32, 128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({3, 2, 512, 1024}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    // Multiple pages per packet with mixed layouts
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 152}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 152}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 152}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 152}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 160}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 160}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 160}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 160}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::L1},
    // Multiple pages per packet with mixed layouts
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3128}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3136}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3136}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
        BufferType::L1},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3136}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::DRAM},
    std::tuple{
        TensorSpec(
            ttnn::Shape({1, 1, 112, 3136}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1))),
        BufferType::L1},
};

}
