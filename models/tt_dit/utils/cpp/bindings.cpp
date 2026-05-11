// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// nanobind glue between Python and the C++ planar concat kernel.

#include "planar_concat.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace nb = nanobind;
using namespace tt_dit_planar;

namespace {

// Per-input-list shape requirements.  Each shard must be uint8, 4D, with
// shape (1, h_per, w_per, T) for CHWT or (1, T, h_per, w_per) for CTHW.
//
// Templated on the ndarray flag set so callers can pass either the unflagged
// or the ``nb::c_contig`` variant — both expose the same shape/dtype/data
// query interface.
template <typename ArrT>
ShardView extract_shard(const ArrT& arr, DimOrder dim_order, int& out_h_per, int& out_w_per, int& out_T, bool first) {
    if (arr.ndim() != 4) {
        throw std::invalid_argument("shard must be 4D (1, h, w, T) for CHWT or (1, T, h, w) for CTHW");
    }
    if (arr.shape(0) != 1) {
        throw std::invalid_argument("shard dim 0 must be 1");
    }

    int h_per_local, w_per_local, T_local;
    if (dim_order == DimOrder::CHWT) {
        h_per_local = static_cast<int>(arr.shape(1));
        w_per_local = static_cast<int>(arr.shape(2));
        T_local = static_cast<int>(arr.shape(3));
    } else {
        T_local = static_cast<int>(arr.shape(1));
        h_per_local = static_cast<int>(arr.shape(2));
        w_per_local = static_cast<int>(arr.shape(3));
    }

    if (first) {
        out_h_per = h_per_local;
        out_w_per = w_per_local;
        out_T = T_local;
    } else {
        if (h_per_local != out_h_per || w_per_local != out_w_per || T_local != out_T) {
            throw std::invalid_argument("all shards within a component must have matching shape");
        }
    }

    // Require contiguous so the kernel can treat the per-shard pointer as a
    // packed (h_per, w_per, T) or (T, h_per, w_per) buffer.
    // (nanobind already validates `nb::c_contig` if requested in the
    // signature; this is a defensive check.)
    return ShardView{arr.data(), 0, 0};
}

DimOrder parse_dim_order(std::string_view s) {
    if (s == "CHWT") {
        return DimOrder::CHWT;
    }
    if (s == "CTHW") {
        return DimOrder::CTHW;
    }
    throw std::invalid_argument("dim_order must be 'CHWT' or 'CTHW'");
}

// Build per-shard mesh coordinates from the (TP, SP) shape, in row-major
// order (matches the Python convention `[(r, c) for r in TP for c in SP]`).
std::vector<std::pair<int, int>> build_mesh_coords(int TP, int SP) {
    std::vector<std::pair<int, int>> out;
    out.reserve(static_cast<size_t>(TP) * SP);
    for (int r = 0; r < TP; ++r) {
        for (int c = 0; c < SP; ++c) {
            out.emplace_back(r, c);
        }
    }
    return out;
}

}  // namespace

NB_MODULE(_planar_concat, m) {
    m.doc() = "Vectorized YUV 4:2:0 planar concat (AVX2 + std::thread pool).";

    m.def(
        "set_thread_pool_size",
        &set_thread_pool_size,
        nb::arg("n_threads"),
        "Set the size of the static C++ thread pool.  No-op after first scatter call.");

    m.def(
        "planar_concat",
        [](nb::list py_y_shards,
           nb::list py_cb_shards,
           nb::list py_cr_shards,
           std::string_view dim_order,
           std::pair<int, int> mesh_shape,
           nb::ndarray<uint8_t, nb::numpy, nb::c_contig> out_arr) -> void {
            const int TP = mesh_shape.first;
            const int SP = mesh_shape.second;
            if (TP <= 0 || SP <= 0) {
                throw std::invalid_argument("mesh_shape must be positive");
            }
            const size_t expected = static_cast<size_t>(TP) * SP;
            if (py_y_shards.size() != expected || py_cb_shards.size() != expected || py_cr_shards.size() != expected) {
                throw std::invalid_argument("y/cb/cr shard lists must each have len == TP*SP");
            }

            const DimOrder dim = parse_dim_order(dim_order);
            const auto coords = build_mesh_coords(TP, SP);

            // Extract shard pointers + verify shapes.  Lifetime: the
            // ndarrays are kept alive by the input Python lists for the
            // duration of this call.
            int y_h_per = 0, y_w_per = 0, y_T = 0;
            int uv_h_per = 0, uv_w_per = 0, uv_T = 0;
            std::vector<ShardView> y_shards, cb_shards, cr_shards;
            // Same type the `nb::cast<...>` calls below return — the
            // ``c_contig`` flag is part of the type, so the keep-alive
            // vectors must match for the moves to compile.
            using KeptArray = nb::ndarray<const uint8_t, nb::c_contig>;
            std::vector<KeptArray> y_keep, cb_keep, cr_keep;

            y_shards.reserve(expected);
            cb_shards.reserve(expected);
            cr_shards.reserve(expected);
            y_keep.reserve(expected);
            cb_keep.reserve(expected);
            cr_keep.reserve(expected);

            for (size_t i = 0; i < expected; ++i) {
                auto y_arr = nb::cast<nb::ndarray<const uint8_t, nb::c_contig>>(py_y_shards[i]);
                ShardView sv = extract_shard(y_arr, dim, y_h_per, y_w_per, y_T, i == 0);
                sv.r = coords[i].first;
                sv.c = coords[i].second;
                y_shards.push_back(sv);
                y_keep.push_back(std::move(y_arr));
            }
            for (size_t i = 0; i < expected; ++i) {
                auto cb_arr = nb::cast<nb::ndarray<const uint8_t, nb::c_contig>>(py_cb_shards[i]);
                ShardView sv = extract_shard(cb_arr, dim, uv_h_per, uv_w_per, uv_T, i == 0);
                sv.r = coords[i].first;
                sv.c = coords[i].second;
                cb_shards.push_back(sv);
                cb_keep.push_back(std::move(cb_arr));
            }
            for (size_t i = 0; i < expected; ++i) {
                auto cr_arr = nb::cast<nb::ndarray<const uint8_t, nb::c_contig>>(py_cr_shards[i]);
                int cr_h = uv_h_per, cr_w = uv_w_per, cr_T = uv_T;
                ShardView sv = extract_shard(cr_arr, dim, cr_h, cr_w, cr_T, false);
                if (cr_h != uv_h_per || cr_w != uv_w_per || cr_T != uv_T) {
                    throw std::invalid_argument("Cb and Cr shard shapes must match");
                }
                sv.r = coords[i].first;
                sv.c = coords[i].second;
                cr_shards.push_back(sv);
                cr_keep.push_back(std::move(cr_arr));
            }

            if (y_T != uv_T) {
                throw std::invalid_argument("Y and UV shards must have matching T");
            }
            if (y_h_per != 2 * uv_h_per || y_w_per != 2 * uv_w_per) {
                throw std::invalid_argument("UV per-shard dims must be exactly Y per-shard / 2 (4:2:0)");
            }

            const int T = y_T;
            const int H = y_h_per * TP;
            const int W = y_w_per * SP;

            if (H % 2 != 0 || W % 2 != 0) {
                throw std::invalid_argument("global H and W must be even for 4:2:0");
            }

            const size_t Hu = H / 2;
            const size_t Wu = W / 2;
            const size_t row_stride = static_cast<size_t>(H) * W + 2 * Hu * Wu;

            // The caller (Python wrapper) provides a pre-allocated output
            // buffer.  We do this rather than allocating in C++ because
            // numpy's `np.empty` ends up faster on first-touch than a C++
            // `new uint8_t[N]` for reasons that vary by libc / page-promo
            // settings.  Punting allocation to numpy keeps this path
            // apples-to-apples with the torch_threaded reference.
            if (out_arr.ndim() != 2 || out_arr.shape(0) != static_cast<size_t>(T) || out_arr.shape(1) != row_stride) {
                throw std::invalid_argument("out array must have shape (T, H*W + 2*(H/2 * W/2))");
            }
            uint8_t* out_buf = out_arr.data();

            // Drop the Python GIL during the C++ scatter — the kernel only
            // touches the raw uint8 buffers and the static thread pool.
            {
                nb::gil_scoped_release release;
                planar_concat(
                    y_shards, y_h_per, y_w_per, cb_shards, uv_h_per, uv_w_per, cr_shards, dim, T, H, W, out_buf);
            }
        },
        nb::arg("y_shards"),
        nb::arg("cb_shards"),
        nb::arg("cr_shards"),
        nb::arg("dim_order"),
        nb::arg("mesh_shape"),
        nb::arg("out"),
        "Planar YUV 4:2:0 concat from per-shard uint8 inputs.\n\n"
        "Args:\n"
        "  y_shards, cb_shards, cr_shards: list of TP*SP uint8 numpy arrays.\n"
        "    Per-shard shape:\n"
        "      CHWT: (1, h_per, w_per, T) — T innermost in memory.\n"
        "      CTHW: (1, T, h_per, w_per) — w_per innermost in memory.\n"
        "    UV shards have h_per/2 and w_per/2 of the Y shard's dims.\n"
        "  dim_order: 'CHWT' or 'CTHW'.\n"
        "  mesh_shape: (TP, SP) — height and width sharding factors.\n\n"
        "Returns:\n"
        "  numpy uint8 array of shape (T, H*W + 2*(H/2 * W/2)), per-frame\n"
        "  [Y plane | Cb plane | Cr plane] in row-major.");
}
