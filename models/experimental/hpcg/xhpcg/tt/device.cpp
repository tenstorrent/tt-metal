// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <cstddef>
#include <cstdlib>
#include <map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/tensor_spec.hpp>

namespace hpcg_tt {

namespace {
std::unique_ptr<Dev> g_dev;

tt::tt_metal::TensorSpec spec(int64_t rows, int64_t cols, tt::tt_metal::Layout layout) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig{}));
}
}  // namespace

Dev& Dev::open() {
    if (!g_dev) {
        g_dev = std::make_unique<Dev>();
        Dev& D = *g_dev;
        D.mesh = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, 512ull << 20);
        D.d = D.mesh.get();
        const CoreCoord g = D.d->compute_with_storage_grid_size();
        D.gx = (int)g.x;
        D.gy = (int)g.y;
        D.nc = D.gx * D.gy;
        for (int i = 0; i < D.nc; ++i) {
            const CoreCoord lc(i % D.gx, i / D.gx);
            const CoreCoord w = D.d->worker_core_from_logical_core(lc);
            D.logical.push_back(lc);
            D.noc_xy.push_back((int64_t)w.x);
            D.noc_xy.push_back((int64_t)w.y);
        }
    }
    return *g_dev;
}

Dev* Dev::get() { return g_dev.get(); }

void Dev::close() {
    if (mesh) {
        ttnn::close_device(*mesh);
        mesh.reset();
        d = nullptr;
    }
}

Tensor Dev::alloc_tile(int64_t P) const {
    bytes += 4.0 * P;
    return ttnn::create_device_tensor(spec(P / 32, 32, tt::tt_metal::Layout::TILE), d);
}

Tensor Dev::alloc_rows(int64_t rows, int64_t cols) const {
    bytes += 4.0 * rows * cols;
    return ttnn::create_device_tensor(spec(rows, cols, tt::tt_metal::Layout::ROW_MAJOR), d);
}

void Dev::write(Tensor& t, const void* src, int64_t offset_bytes, int64_t size_bytes) const {
    std::optional<tt::tt_metal::BufferRegion> region;
    if (size_bytes >= 0) {
        region.emplace((tt::tt_metal::DeviceAddr)offset_bytes, (tt::tt_metal::DeviceAddr)size_bytes);
    }
    ttnn::copy_to_device(d->mesh_command_queue(), static_cast<const std::byte*>(src), t, region);
}

void Dev::read(const Tensor& t, void* dst, int64_t offset_bytes, int64_t size_bytes) const {
    std::optional<tt::tt_metal::BufferRegion> region;
    if (size_bytes >= 0) {
        region.emplace((tt::tt_metal::DeviceAddr)offset_bytes, (tt::tt_metal::DeviceAddr)size_bytes);
    }
    ttnn::copy_to_host(d->mesh_command_queue(), t, static_cast<std::byte*>(dst), region, true);
}

Tensor Dev::upload_rows(const void* src, int64_t rows, int64_t w) const {
    Tensor t = alloc_rows(rows, w);
    write(t, src);
    return t;
}

void Dev::sync() const { tt::tt_metal::distributed::Synchronize(*d, std::nullopt); }

uint32_t addr(const Tensor& t) { return (uint32_t)t.mesh_buffer().address(); }

std::vector<uint32_t> accessor_args(const Tensor& t) {
    return tt::tt_metal::TensorAccessorArgs(*t.buffer()).get_compile_time_args();
}

CoreRangeSet core_ranges(int nc, int gx) {
    static std::map<std::pair<int, int>, CoreRangeSet> made;  // descriptors copy what they are given
    auto it = made.find({nc, gx});
    if (it != made.end()) {
        return it->second;
    }
    const int full = nc / gx, part = nc % gx;
    std::vector<CoreRange> rs;
    if (full) {
        rs.emplace_back(CoreCoord(0, 0), CoreCoord(gx - 1, full - 1));
    }
    if (part) {
        rs.emplace_back(CoreCoord(0, full), CoreCoord(part - 1, full));
    }
    return made.emplace(std::make_pair(nc, gx), CoreRangeSet(rs)).first->second;
}

const std::vector<KernelGroup>& kernel_groups(int nc, int gx, const std::vector<int>& rect) {
    // tt/custom_ops/sparse_rows/op.py's _flip_ranges + _kernel_groups: the first nc cores split by whether they lie in
    // rect, each row's runs merged with the rows above when they span the same x range
    static std::map<std::tuple<int, int, std::vector<int>>, std::vector<KernelGroup>> made;
    const auto key = std::make_tuple(nc, gx, rect);
    auto it = made.find(key);
    if (it != made.end()) {
        return it->second;
    }
    std::map<bool, std::vector<std::array<int, 4>>> out;  // flipped -> (x0, x1, y0, y1) rectangles
    for (int y = 0; y < (nc + gx - 1) / gx; ++y) {
        const int n = std::min(gx, nc - y * gx);
        const bool f0 = !rect.empty() && rect[2] <= y && y <= rect[3];
        std::vector<std::tuple<bool, int, int>> runs;
        if (!f0) {
            runs.emplace_back(false, 0, n - 1);
        } else {
            const int a = rect[0], b = std::min(rect[1], n - 1);
            runs = {{false, 0, std::min(a, n) - 1}, {true, a, b}, {false, b + 1, n - 1}};
        }
        for (auto [f, x0, x1] : runs) {
            if (x1 < x0) {
                continue;
            }
            auto& rs = out[f];
            if (!rs.empty() && rs.back()[0] == x0 && rs.back()[1] == x1 && rs.back()[3] == y - 1) {
                rs.back()[3] = y;
            } else {
                rs.push_back({x0, x1, y, y});
            }
        }
    }
    std::vector<KernelGroup> groups;
    for (const auto& [f, rects] : out) {  // map order: false first, as Python's insertion order here
        KernelGroup g;
        if (out.size() > 1) {
            std::vector<CoreRange> rs;
            for (const auto& r : rects) {
                rs.emplace_back(CoreCoord(r[0], r[2]), CoreCoord(r[1], r[3]));
            }
            g.cores = CoreRangeSet(rs);
        } else {
            g.cores = core_ranges(nc, gx);
        }
        g.reader.processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        g.reader.noc = f ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;
        g.writer.processor = tt::tt_metal::DataMovementProcessor::RISCV_1;
        g.writer.noc = f ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1;
        groups.push_back(g);
    }
    return made.emplace(key, std::move(groups)).first->second;
}

tt::tt_metal::CBDescriptor cb(const CoreRangeSet& cores, uint8_t idx, uint32_t nbytes, uint32_t page, bool face) {
    tt::tt_metal::CBFormatDescriptor fd{.buffer_index = idx, .data_format = tt::DataFormat::Float32, .page_size = page};
    if (face) {
        fd.tile = tt::tt_metal::TileDescriptor(16, 16, false);
    }
    tt::tt_metal::CBDescriptor c;
    c.total_size = nbytes;
    c.core_ranges = cores;
    c.format_descriptors.push_back(fd);
    return c;
}

}  // namespace hpcg_tt
