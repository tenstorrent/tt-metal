// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <array>
#include <random>
#include <functional>
#include <iostream>

using SHAPE = std::array<std::uint32_t, 4>;

namespace tt {

namespace deprecated {

enum class Initialize { ZEROS = 0, ONES = 1, INCREMENT = 2, RANDOM = 3 };

template <class T>
class Tensor {
public:
    Tensor(std::vector<T>& values, std::array<uint32_t, 4>& shape) {
        this->shape = shape;
        this->values = values;
        this->strides = {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
    }
    Tensor(std::array<uint32_t, 4>& shape) {
        this->shape = shape;
        auto volume = shape[0] * shape[1] * shape[2] * shape[3];
        this->values.resize(volume);
        this->strides = {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
    }
    const std::vector<T>& get_values() const { return this->values; }
    const std::array<uint32_t, 4>& get_padded_shape() const { return this->shape; }
    const std::array<uint32_t, 4>& get_strides() const { return this->strides; }
    uint32_t get_volume() const { return shape[0] * shape[1] * shape[2] * shape[3]; }

private:
    std::vector<T> values;
    std::array<uint32_t, 4> shape;    // outer-most dimension first
    std::array<uint32_t, 4> strides;  // outer-most dimension first
};

template <class T>
void print(const Tensor<T>& tensor) {
    auto tensor_shape = tensor.padded_shape();
    auto tensor_strides = tensor.get_strides();
    auto tensor_data = tensor.get_values();

    std::cout << "Shape = [" << tensor_shape[0] << "," << tensor_shape[1] << "," << tensor_shape[2] << ","
              << tensor_shape[3] << "]" << std::endl;
    std::cout << "Strides = [" << tensor_strides[0] << "," << tensor_strides[1] << "," << tensor_strides[2] << ","
              << tensor_strides[3] << "]" << std::endl;
    std::cout << "Values = [";
    for (auto w = 0; w < tensor_shape[0]; w++) {
        std::cout << "[";
        for (auto z = 0; z < tensor_shape[1]; z++) {
            std::cout << "[";
            for (auto y = 0; y < tensor_shape[2]; y++) {
                std::cout << "[";
                for (auto x = 0; x < tensor_shape[3]; x++) {
                    auto idx = x + tensor_shape[3] * y + tensor_shape[3] * tensor_shape[2] * z +
                               tensor_shape[3] * tensor_shape[2] * tensor_shape[1] * w;
                    std::cout << tensor_data[idx] << ",";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

template <class T>
Tensor<T> initialize_tensor(
    std::array<uint32_t, 4>& shape, Initialize init_type, int rand_min_val = 0, int rand_max_val = 100, int seed = 0) {
    std::vector<T> values;
    auto rand_float = std::bind(std::uniform_real_distribution<float>(rand_min_val, rand_max_val), std::mt19937(seed));
    for (auto w = 0; w < shape[0]; w++) {
        for (auto z = 0; z < shape[1]; z++) {
            for (auto y = 0; y < shape[2]; y++) {
                for (auto x = 0; x < shape[3]; x++) {
                    float val;
                    switch (init_type) {
                        case Initialize::ZEROS: val = 0; break;
                        case Initialize::ONES: val = 1; break;
                        case Initialize::INCREMENT:
                            val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                            break;
                        case Initialize::RANDOM: val = rand_float(); break;
                        default: val = 0; break;
                    }
                    values.push_back(static_cast<T>(val));
                }
            }
        }
    }

    return Tensor<T>(values, shape);
}

// NCHW
// {0, 1, 2, 3} -> no change
// {0, 2, 3, 1} -> NHWC
template <class T>
Tensor<T> permute(const Tensor<T>& input, std::array<int, 4> dims) {
    auto in_shape = input.padded_shape();
    std::array<uint32_t, 4> out_shape = {in_shape[dims[0]], in_shape[dims[1]], in_shape[dims[2]], in_shape[dims[3]]};
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<T> out = std::vector<T>(output_volume);

    auto input_values = input.get_values();
    for (auto w = 0; w < in_shape[0]; w++) {              // N
        for (auto z = 0; z < in_shape[1]; z++) {          // Z
            for (auto y = 0; y < in_shape[2]; y++) {      // Y
                for (auto x = 0; x < in_shape[3]; x++) {  // X
                    auto in_idx = x + y * in_shape[3] + z * in_shape[3] * in_shape[2] +
                                  w * in_shape[3] * in_shape[2] * in_shape[1];
                    auto out_idx = z + x * out_shape[3] + y * out_shape[3] * out_shape[2] +
                                   w * out_shape[3] * out_shape[2] * out_shape[1];
                    out[out_idx] = input_values[in_idx];
                }
            }
        }
    }

    return Tensor<T>(out, out_shape);
}

template <class T>
Tensor<T> permute_nhwc_to_nchw(const Tensor<T>& input) {
    std::array<uint32_t, 4> in_shape = input.padded_shape();
    std::array<uint32_t, 4> out_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
    auto output_volume = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    std::vector<T> out = std::vector<T>(output_volume, 0);
    auto input_values = input.get_values();
    for (auto w = 0; w < out_shape[0]; w++) {              // N
        for (auto z = 0; z < out_shape[1]; z++) {          // Z
            for (auto y = 0; y < out_shape[2]; y++) {      // Y
                for (auto x = 0; x < out_shape[3]; x++) {  // X
                    auto out_idx = x + y * out_shape[3] + z * out_shape[3] * out_shape[2] +
                                   w * out_shape[3] * out_shape[2] * out_shape[1];
                    auto in_idx = z + x * in_shape[3] + y * in_shape[3] * in_shape[2] +
                                  w * in_shape[3] * in_shape[2] * in_shape[1];
                    out[out_idx] = input_values[in_idx];
                }
            }
        }
    }

    return Tensor<T>(out, out_shape);
}

template <class T>
Tensor<T> pad(Tensor<T>& input, std::array<std::array<uint32_t, 2>, 4> pad_size, T val = 0) {
    auto in_shape = input.padded_shape();
    std::array<uint32_t, 4> out_shape = {
        in_shape[0] + pad_size[0][0] + pad_size[0][1],
        in_shape[1] + pad_size[1][0] + pad_size[1][1],
        in_shape[2] + pad_size[2][0] + pad_size[2][1],
        in_shape[3] + pad_size[3][0] + pad_size[3][1]};
    Tensor<T> output(out_shape);

    auto output_strides = output.get_strides();
    auto input_values = input.get_values();
    std::vector<T> out;
    for (auto i = 0; i < pad_size[0][0] * output_strides[0]; i++) {
        out.push_back(val);
    }
    for (auto dim0 = 0; dim0 < in_shape[0]; dim0++) {
        for (auto i = 0; i < pad_size[1][0] * output_strides[1]; i++) {
            out.push_back(val);
        }
        for (auto dim1 = 0; dim1 < in_shape[1]; dim1++) {
            for (auto i = 0; i < pad_size[2][0] * output_strides[2]; i++) {
                out.push_back(val);
            }
            for (auto dim2 = 0; dim2 < in_shape[2]; dim2++) {
                for (auto i = 0; i < pad_size[3][0] * output_strides[3]; i++) {
                    out.push_back(val);
                }
                for (auto dim3 = 0; dim3 < in_shape[3]; dim3++) {
                    auto idx = dim3 + in_shape[3] * dim2 + in_shape[3] * in_shape[2] * dim1 +
                               in_shape[3] * in_shape[2] * in_shape[1] * dim0;
                    out.push_back(input_values[idx]);
                }
                for (auto i = 0; i < pad_size[3][1] * output_strides[3]; i++) {
                    out.push_back(val);
                }
            }
            for (auto i = 0; i < pad_size[2][1] * output_strides[2]; i++) {
                out.push_back(val);
            }
        }
        for (auto i = 0; i < pad_size[1][1] * output_strides[1]; i++) {
            out.push_back(val);
        }
    }
    for (auto i = 0; i < pad_size[0][1] * output_strides[0]; i++) {
        out.push_back(val);
    }

    // output.values = out;
    return Tensor<T>(out, out_shape);
}

}  // end namespace deprecated

}  // end namespace tt
