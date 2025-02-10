// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "hostdevcommon/kernel_structs.h"
#include "assert.hpp"
#include "base_types.hpp"
#include "tt_backend_api_types.hpp"
#include "utils.hpp"
#include "circular_buffer_constants.h"  // for NUM_CIRCULAR_BUFFERS

namespace tt {
/**
 * @brief A descriptor of the high-level kernel. Contains circular buffer formats, HLK filename, HLK args ptr/size.
 */

class tt_hlk_desc {
private:
    // data formats spec for the I/O operands (i.e., buffers)
    MathFidelity math_fidelity;
    bool approximation_mode;

    void* hlk_args;        // void ptr to user-defined hlk_args_t struct (user writes)
    size_t hlk_args_size;  // size of hlk_args_t in bytes (result of sizeof())

public:
    DataFormat buf_dataformat_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_num_faces_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_partial_face_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_face_r_dim_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_narrow_tile_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_tile_r_dim_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_tile_c_dim_arr[NUM_CIRCULAR_BUFFERS];
    uint32_t buf_tile_size_arr[NUM_CIRCULAR_BUFFERS];

    tt_hlk_desc() {
        math_fidelity = MathFidelity::Invalid;
        hlk_args = nullptr;
        hlk_args_size = 0;
        approximation_mode = true;

        for (int i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
            buf_dataformat_arr[i] = DataFormat::Invalid;
            buf_num_faces_arr[i] = constants::TILE_HW / constants::FACE_HW;
            buf_partial_face_arr[i] = 0;
            buf_face_r_dim_arr[i] = constants::FACE_HEIGHT;
            buf_narrow_tile_arr[i] = 0;
            buf_tile_r_dim_arr[i] = constants::TILE_HEIGHT;
            buf_tile_c_dim_arr[i] = constants::TILE_WIDTH;
            buf_tile_size_arr[i] = constants::BFLOAT8_B_TILE_HW;
        }
    }

    tt_hlk_desc(tt_hlk_desc& in) {
        for (int i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
            buf_dataformat_arr[i] = in.buf_dataformat_arr[i];
            buf_num_faces_arr[i] = in.buf_num_faces_arr[i];
            buf_partial_face_arr[i] = in.buf_partial_face_arr[i];
            buf_face_r_dim_arr[i] = in.buf_face_r_dim_arr[i];
            buf_narrow_tile_arr[i] = in.buf_narrow_tile_arr[i];
            buf_tile_r_dim_arr[i] = in.buf_tile_r_dim_arr[i];
            buf_tile_c_dim_arr[i] = in.buf_tile_c_dim_arr[i];
            buf_tile_size_arr[i] = in.buf_tile_size_arr[i];
        }

        math_fidelity = in.math_fidelity;
        hlk_args = in.hlk_args;
        hlk_args_size = in.hlk_args_size;
        approximation_mode = in.approximation_mode;
    }

    DataFormat get_buf_dataformat(int buf_idx) const { return buf_dataformat_arr[buf_idx]; }

    void set_buf_dataformat(int buf_idx, DataFormat data_format) { buf_dataformat_arr[buf_idx] = data_format; }

    uint32_t get_buf_num_faces(int buf_idx) const { return buf_num_faces_arr[buf_idx]; }

    void set_buf_num_faces(int buf_idx, uint32_t num_faces) { buf_num_faces_arr[buf_idx] = num_faces; }

    uint32_t get_buf_partial_face(int buf_idx) const { return buf_partial_face_arr[buf_idx]; }

    void set_buf_partial_face(int buf_idx, uint32_t partial_face) { buf_partial_face_arr[buf_idx] = partial_face; }

    uint32_t get_buf_face_r_dim(int buf_idx) const { return buf_face_r_dim_arr[buf_idx]; }

    void set_buf_face_r_dim(int buf_idx, uint32_t face_r_dim) { buf_face_r_dim_arr[buf_idx] = face_r_dim; }

    uint32_t get_buf_narrow_tile(int buf_idx) const { return buf_narrow_tile_arr[buf_idx]; }

    void set_buf_narrow_tile(int buf_idx, uint32_t narrow_tile) { buf_narrow_tile_arr[buf_idx] = narrow_tile; }

    uint32_t get_buf_tile_r_dim(int buf_idx) const { return buf_tile_r_dim_arr[buf_idx]; }

    void set_buf_tile_r_dim(int buf_idx, uint32_t tile_r_dim) { buf_tile_r_dim_arr[buf_idx] = tile_r_dim; }

    uint32_t get_buf_tile_c_dim(int buf_idx) const { return buf_tile_c_dim_arr[buf_idx]; }

    void set_buf_tile_c_dim(int buf_idx, uint32_t tile_c_dim) { buf_tile_c_dim_arr[buf_idx] = tile_c_dim; }

    uint32_t get_buf_tile_size(int buf_idx) const { return buf_tile_size_arr[buf_idx]; }

    void set_buf_tile_size(int buf_idx, uint32_t tile_size) { buf_tile_size_arr[buf_idx] = tile_size; }

    void set_hlk_args(void* args, size_t size) {
        hlk_args = args;
        hlk_args_size = size;
    }

    void* get_hlk_args() const { return hlk_args; }

    void set_hlk_math_fidelity(MathFidelity math_fi) { math_fidelity = math_fi; }

    MathFidelity get_hlk_math_fidelity() const { return math_fidelity; }

    void set_hlk_math_approx_mode(bool approx_mode) { approximation_mode = approx_mode; }

    bool get_hlk_math_approx_mode() const { return approximation_mode; }

    // rk: added by fw-dma-test-2 team
    size_t get_hlk_args_size() const { return hlk_args_size; }

    const DataFormat* get_buf_dataformats() const { return buf_dataformat_arr; }

};  // tt_hlk_desc
}  // namespace tt

// Hash for hlk_args
inline void hash_hlk_args(size_t& seed, void* hlk_args, size_t hlk_args_size) {
    char buffer[hlk_args_size];
    memcpy(buffer, hlk_args, hlk_args_size);

    for (int i = 0; i < hlk_args_size; i++) {
        tt::utils::hash_combine(seed, std::hash<char>{}(buffer[i]));
    }
}

template <>
struct std::hash<tt::tt_hlk_desc> {
    std::size_t operator()(tt::tt_hlk_desc const& obj) const noexcept {
        std::size_t hash_value = 0;
        for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
            tt::utils::hash_combine(hash_value, hash<tt::DataFormat>{}(obj.get_buf_dataformat(i)));
            tt::utils::hash_combine(hash_value, hash<uint32_t>{}(obj.get_buf_tile_r_dim(i)));
            tt::utils::hash_combine(hash_value, hash<uint32_t>{}(obj.get_buf_tile_c_dim(i)));
        }
        tt::utils::hash_combine(hash_value, hash<MathFidelity>{}(obj.get_hlk_math_fidelity()));
        tt::utils::hash_combine(hash_value, hash<bool>{}(obj.get_hlk_math_approx_mode()));

        // Get hash for hlk_args here
        void* hlk_args = obj.get_hlk_args();
        size_t hlk_args_size = obj.get_hlk_args_size();
        if (hlk_args != nullptr and hlk_args_size > 0) {
            hash_hlk_args(hash_value, hlk_args, hlk_args_size);
        } else if (hlk_args == nullptr and hlk_args_size == 0) {
        } else {
            TT_THROW(
                "Mismatching values, either hlk_args == nullptr and hlk_args_size == 0 or hlk_args != nullptr and "
                "hlk_args_size > 0!");
        }

        return hash_value;
    }
};
