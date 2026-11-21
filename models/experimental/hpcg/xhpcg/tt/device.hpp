// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The card and the TT-NN helpers the ops share (the C++ counterparts of tt/layout.py and the module-level helpers of
// tt/custom_ops/sparse_rows/op.py).
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace hpcg_tt {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using ttnn::Tensor;

struct Dev {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh;
    tt::tt_metal::distributed::MeshDevice* d = nullptr;
    int gx = 0, gy = 0, nc = 0;      // worker grid, cores
    mutable double bytes = 0;        // device memory allocated through alloc_tile / alloc_rows
    std::vector<int64_t> noc_xy;     // NoC (x, y) of logical core i = x + y gx, flattened
    std::vector<CoreCoord> logical;  // CoreCoord of logical core i

    static Dev& open();  // opens the card on first use (device 0, a 512 MB trace region)
    static Dev* get();   // the open card, or nullptr
    void close();

    // fp32 DRAM-interleaved tensors, not filled
    Tensor alloc_tile(int64_t P) const;                   // a TILE vector [P / 32, 32]
    Tensor alloc_rows(int64_t rows, int64_t cols) const;  // ROW_MAJOR [rows, cols]
    // raw page bytes to / from a device tensor (no layout conversion: a TILE vector's pages hold storage order q)
    void write(Tensor& t, const void* src, int64_t offset_bytes = 0, int64_t size_bytes = -1) const;
    void read(const Tensor& t, void* dst, int64_t offset_bytes = 0, int64_t size_bytes = -1) const;
    // a ROW_MAJOR [rows, w] tensor of 32-bit words uploaded from src (tt/layout.py's u32_rows)
    Tensor upload_rows(const void* src, int64_t rows, int64_t w) const;
    void sync() const;
};

uint32_t addr(const Tensor& t);                        // buffer address (Python's buffer_address())
std::vector<uint32_t> accessor_args(const Tensor& t);  // TensorAccessorArgs(t).get_compile_time_args()

CoreRangeSet core_ranges(int nc, int gx);  // the first nc logical cores, row by row
// (core ranges, reader config, writer config) of each NoC assignment: reader NoC 0 / writer NoC 1, swapped inside
// rect (x0, x1, y0, y1); empty rect: no swap
struct KernelGroup {
    CoreRangeSet cores;
    tt::tt_metal::DataMovementConfigDescriptor reader, writer;
};
const std::vector<KernelGroup>& kernel_groups(int nc, int gx, const std::vector<int>& rect);
tt::tt_metal::CBDescriptor cb(const CoreRangeSet& cores, uint8_t idx, uint32_t nbytes, uint32_t page, bool face);

}  // namespace hpcg_tt
