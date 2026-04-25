/*
 * Fused BGR-HWC uint8 → RGB-CHW bfloat16/255 tile conversion.
 *
 * Replaces the Python pipeline:
 *   1. numpy channel-wise slice (BGR→RGB, HWC→CHW into uint8 buffer)
 *   2. bf16_buf.copy_(torch.from_numpy(uint8))
 *   3. bf16_buf.mul_(1/255)
 *
 * Single-pass: reads each source pixel once, writes bf16 output once.
 * Memory traffic per tile: ~1.2 MB read + ~2.4 MB write = ~3.6 MB
 * vs Python: ~10.8 MB per tile (3 passes).
 *
 * Two parallelism modes:
 *   - fused_slice_bgr_to_bf16: OMP parallel across tiles (batch call)
 *   - fused_convert_tile_range: process [start,end) tiles in one call,
 *     releases the GIL so Python ThreadPoolExecutor can dispatch 8 of
 *     these concurrently (no OMP, no spin-wait interference)
 */
#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

namespace {

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

constexpr float INV_255 = 1.0f / 255.0f;

// Pre-computed LUT: uint8 → bf16 bits (256 entries, 512 bytes → fits L1).
alignas(64) uint16_t g_lut[256];

void init_lut() {
    for (int i = 0; i < 256; ++i) {
        g_lut[i] = f32_to_bf16(static_cast<float>(i) * INV_255);
    }
}

// Process one row of one tile.
template <bool use_nt>
inline void convert_row(
    const uint8_t* __restrict__ row_in,
    uint16_t* __restrict__ r_out,
    uint16_t* __restrict__ g_out,
    uint16_t* __restrict__ b_out,
    int src_w) {
    int x = 0;
#ifdef __AVX512BW__
    for (; x + 16 <= src_w; x += 16) {
        alignas(32) uint16_t r_tmp[16], g_tmp[16], b_tmp[16];
        for (int k = 0; k < 16; ++k) {
            int off = (x + k) * 3;
            b_tmp[k] = g_lut[row_in[off + 0]];
            g_tmp[k] = g_lut[row_in[off + 1]];
            r_tmp[k] = g_lut[row_in[off + 2]];
        }
        __m256i r_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(r_tmp));
        __m256i g_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(g_tmp));
        __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b_tmp));
        if constexpr (use_nt) {
            _mm256_stream_si256(reinterpret_cast<__m256i*>(r_out + x), r_vec);
            _mm256_stream_si256(reinterpret_cast<__m256i*>(g_out + x), g_vec);
            _mm256_stream_si256(reinterpret_cast<__m256i*>(b_out + x), b_vec);
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(r_out + x), r_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(g_out + x), g_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_out + x), b_vec);
        }
    }
#endif
    for (; x < src_w; ++x) {
        int off = x * 3;
        b_out[x] = g_lut[row_in[off + 0]];
        g_out[x] = g_lut[row_in[off + 1]];
        r_out[x] = g_lut[row_in[off + 2]];
    }
}

// Process one tile.
template <bool use_nt>
inline void convert_tile(
    const uint8_t* __restrict__ frame,
    int frame_stride,
    uint16_t* __restrict__ out,
    int tile_H,
    int tile_W,
    int row_start,
    int col_start,
    int src_h,
    int src_w,
    bool needs_pad,
    uint16_t pad_bf16) {
    const int plane_size = tile_H * tile_W;
    if (needs_pad) {
        __m256i fill = _mm256_set1_epi16(static_cast<int16_t>(pad_bf16));
        const int total = 3 * plane_size;
        int j = 0;
        for (; j + 16 <= total; j += 16) {
            if constexpr (use_nt) {
                _mm256_stream_si256(reinterpret_cast<__m256i*>(out + j), fill);
            } else {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), fill);
            }
        }
        for (; j < total; ++j) {
            out[j] = pad_bf16;
        }
    }
    uint16_t* r_plane = out;
    uint16_t* g_plane = out + plane_size;
    uint16_t* b_plane = out + 2 * plane_size;
    for (int y = 0; y < src_h; ++y) {
        const uint8_t* row = frame + (row_start + y) * frame_stride + col_start * 3;
        convert_row<use_nt>(row, r_plane + y * tile_W, g_plane + y * tile_W, b_plane + y * tile_W, src_w);
    }
    if constexpr (use_nt) {
        _mm_sfence();
    }
}

}  // namespace

// ---- Batch OMP version ----
torch::Tensor fused_slice_bgr_to_bf16(
    torch::Tensor frame_tensor, torch::Tensor bf16_out, torch::Tensor tile_specs, int n_threads, bool use_nt) {
    TORCH_CHECK(frame_tensor.dtype() == torch::kUInt8);
    TORCH_CHECK(bf16_out.dtype() == torch::kBFloat16);
    TORCH_CHECK(tile_specs.dtype() == torch::kInt32);

    static bool lut_ready = false;
    if (!lut_ready) {
        init_lut();
        lut_ready = true;
    }

    const uint8_t* frame = frame_tensor.data_ptr<uint8_t>();
    const int frame_stride = frame_tensor.size(1) * 3;
    const int N = bf16_out.size(0);
    const int tile_H = bf16_out.size(2);
    const int tile_W = bf16_out.size(3);
    const int n_specs = tile_specs.size(0);
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(bf16_out.data_ptr<at::BFloat16>());
    const int32_t* specs = tile_specs.data_ptr<int32_t>();
    const int tile_elems = 3 * tile_H * tile_W;
    const uint16_t zero_bf16 = f32_to_bf16(0.0f);

    omp_set_num_threads(n_threads);
    if (use_nt) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            uint16_t* t = out_ptr + i * tile_elems;
            if (i >= n_specs) {
                __m256i z = _mm256_set1_epi16(0);
                for (int j = 0; j + 16 <= tile_elems; j += 16) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(t + j), z);
                }
                _mm_sfence();
                continue;
            }
            convert_tile<true>(
                frame,
                frame_stride,
                t,
                tile_H,
                tile_W,
                specs[i * 6],
                specs[i * 6 + 1],
                specs[i * 6 + 2],
                specs[i * 6 + 3],
                specs[i * 6 + 4] != 0,
                g_lut[(uint8_t)specs[i * 6 + 5]]);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            uint16_t* t = out_ptr + i * tile_elems;
            if (i >= n_specs) {
                std::memset(t, 0, tile_elems * 2);
                continue;
            }
            convert_tile<false>(
                frame,
                frame_stride,
                t,
                tile_H,
                tile_W,
                specs[i * 6],
                specs[i * 6 + 1],
                specs[i * 6 + 2],
                specs[i * 6 + 3],
                specs[i * 6 + 4] != 0,
                g_lut[(uint8_t)specs[i * 6 + 5]]);
        }
    }
    return bf16_out;
}

// ---- Range version for ThreadPoolExecutor (releases GIL) ----
void fused_convert_tile_range(
    torch::Tensor frame_tensor, torch::Tensor bf16_out, torch::Tensor tile_specs, int start, int end, bool use_nt) {
    static bool lut_ready = false;
    if (!lut_ready) {
        init_lut();
        lut_ready = true;
    }

    // Extract all pointers before releasing GIL
    const uint8_t* frame = frame_tensor.data_ptr<uint8_t>();
    const int frame_stride = frame_tensor.size(1) * 3;
    const int tile_H = bf16_out.size(2);
    const int tile_W = bf16_out.size(3);
    const int n_specs = tile_specs.size(0);
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(bf16_out.data_ptr<at::BFloat16>());
    const int32_t* specs = tile_specs.data_ptr<int32_t>();
    const int tile_elems = 3 * tile_H * tile_W;
    const uint16_t zero_bf16 = f32_to_bf16(0.0f);

    // Release GIL — allows 8 Python threads to run concurrently
    py::gil_scoped_release release;

    for (int i = start; i < end; ++i) {
        uint16_t* t = out_ptr + i * tile_elems;
        if (i >= n_specs) {
            if (use_nt) {
                __m256i z = _mm256_set1_epi16(0);
                for (int j = 0; j + 16 <= tile_elems; j += 16) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(t + j), z);
                }
                _mm_sfence();
            } else {
                std::memset(t, 0, tile_elems * 2);
            }
            continue;
        }
        if (use_nt) {
            convert_tile<true>(
                frame,
                frame_stride,
                t,
                tile_H,
                tile_W,
                specs[i * 6],
                specs[i * 6 + 1],
                specs[i * 6 + 2],
                specs[i * 6 + 3],
                specs[i * 6 + 4] != 0,
                g_lut[(uint8_t)specs[i * 6 + 5]]);
        } else {
            convert_tile<false>(
                frame,
                frame_stride,
                t,
                tile_H,
                tile_W,
                specs[i * 6],
                specs[i * 6 + 1],
                specs[i * 6 + 2],
                specs[i * 6 + 3],
                specs[i * 6 + 4] != 0,
                g_lut[(uint8_t)specs[i * 6 + 5]]);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_slice_bgr_to_bf16",
        &fused_slice_bgr_to_bf16,
        "Batch OMP tile conversion",
        py::arg("frame_tensor"),
        py::arg("bf16_out"),
        py::arg("tile_specs"),
        py::arg("n_threads") = 8,
        py::arg("use_nt") = false);
    m.def(
        "fused_convert_tile_range",
        &fused_convert_tile_range,
        "Range tile conversion (GIL-free for ThreadPoolExecutor)",
        py::arg("frame_tensor"),
        py::arg("bf16_out"),
        py::arg("tile_specs"),
        py::arg("start"),
        py::arg("end"),
        py::arg("use_nt") = false);
}
