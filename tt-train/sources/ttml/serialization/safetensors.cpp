// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "fmt/format.h"

// POSIX mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tt_stl/cleanup.hpp>

#include "safetensors.hpp"

namespace ttml::serialization {
uint64_t le64(const unsigned char* p) {
    // little-endian decode (portable)
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

void SafetensorSerialization::visit_safetensors_file(const std::filesystem::path& path, const TensorCallback& cb) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("file does not exist: " + path.string());
    }

    auto fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("open failed: {}: {}", path.string(), std::strerror(errno)));
    }
    auto _1 = ttsl::make_cleanup([fd]() { ::close(fd); });
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        throw std::system_error(errno, std::generic_category(), "fstat");
    }
    const size_t file_size = size_t(st.st_size);
    constexpr size_t header_size = 8;
    if (file_size < header_size) {
        throw std::runtime_error("file too small for safetensors");
    }

    auto map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        throw std::runtime_error(fmt::format("mmap failed: {}", std::strerror(errno)));
    }
    auto _2 = ttsl::make_cleanup([map, file_size]() { munmap(map, file_size); });

    auto* base = reinterpret_cast<const unsigned char*>(map);

    const uint64_t header_len = le64(base);
    if (header_size + header_len > file_size) {
        throw std::runtime_error("header length out of range");
    }

    const char* header_begin = reinterpret_cast<const char*>(base + 8);
    nlohmann::json headers = nlohmann::json::parse(header_begin, header_begin + header_len);

    const size_t data_offset = 8 + size_t(header_len);
    if (data_offset > file_size) {
        throw std::runtime_error(fmt::format("bad data offset: {} > {}", data_offset, file_size));
    }

    const size_t data_size = file_size - data_offset;
    const std::byte* data_base = reinterpret_cast<const std::byte*>(base + data_offset);

    for (auto it = headers.begin(); it != headers.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        const auto& obj = it.value();
        const auto& dtype = obj.at("dtype").get_ref<const std::string&>();
        const auto& shape_json = obj.at("shape");
        std::vector<uint32_t> shape;
        shape.reserve(shape_json.size());
        for (const auto& d : shape_json) {
            shape.push_back(d.get<uint32_t>());
        }

        const auto offs = obj.at("data_offsets");
        uint64_t begin = offs.at(0).get<uint64_t>();
        uint64_t end = offs.at(1).get<uint64_t>();
        if (begin > end) {
            throw std::runtime_error(fmt::format("Offsets are invalid: end < begin ({} < {})", end, begin));
        }
        const uint64_t len = end - begin;

        if (end > data_size) {
            throw std::runtime_error(fmt::format("data_offsets out of range: {} > {}", end, data_size));
        }

        TensorInfo info{it.key(), dtype, ttnn::Shape(std::span(shape.data(), shape.size()))};
        std::span<const std::byte> bytes{data_base + begin, static_cast<size_t>(len)};

        if (!cb(info, bytes)) {
            break;
        }
    }
}

std::vector<float> SafetensorSerialization::bytes_to_floats_copy(std::span<const std::byte> bytes) {
    if (bytes.size_bytes() % sizeof(float) != 0) {
        throw std::runtime_error("bytes_to_floats_copy: size not multiple of sizeof(float)");
    }
    const std::size_t n = bytes.size_bytes() / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), bytes.data(), n * sizeof(float));
    return out;
}
std::vector<float> SafetensorSerialization::bytes_to_float_vec(
    const std::span<const std::byte>& bytes, const std::string& dtype) {
    std::vector<float> float_vec;
    if (dtype == "BF16" || dtype == "BFLOAT16") {
        if (bytes.size_bytes() % 2 != 0)
            throw std::runtime_error("BF16 data size must be even");
        const std::size_t n = bytes.size_bytes() / 2;
        float_vec.reserve(n);
        const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t tmp = static_cast<uint32_t>(bf16_data[i]) << 16;
            float value;
            std::memcpy(&value, &tmp, sizeof(value));
            float_vec.push_back(value);
        }
    } else if (dtype == "F16" || dtype == "FLOAT16") {
        if (bytes.size_bytes() % 2 != 0)
            throw std::runtime_error("F16 data size must be even");
        const std::size_t n = bytes.size_bytes() / 2;
        float_vec.resize(n);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(bytes.data());
        auto half2float = [](uint16_t h) -> float {
            uint16_t he = (h & 0x7C00u) >> 10;  // exp
            uint16_t hm = (h & 0x03FFu);        // mant
            uint32_t s = (h & 0x8000u) << 16;
            uint32_t e, m;
            if (he == 0) {  // subnorm/zero
                if (hm == 0) {
                    e = 0;
                    m = 0;
                } else {
                    int shift = 0;
                    while ((hm & 0x0400u) == 0) {
                        hm <<= 1;
                        ++shift;
                    }
                    hm &= 0x03FFu;
                    e = 127 - 15 - shift;
                    m = (uint32_t)hm << 13;
                }
            } else if (he == 0x1Fu) {  // inf/NaN
                e = 255;
                m = (uint32_t)hm << 13;
            } else {
                e = (uint32_t)(he - 15 + 127);
                m = (uint32_t)hm << 13;
            }
            uint32_t u = s | (e << 23) | m;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        };
        for (size_t i = 0; i < n; ++i) float_vec[i] = half2float(p[i]);
    } else if (dtype == "F32" || dtype == "FLOAT32") {
        float_vec = serialization::SafetensorSerialization::bytes_to_floats_copy(bytes);
    } else {
        throw std::runtime_error(fmt::format("Unsupported dtype: {}", dtype));
    }
    return float_vec;
};
}  // namespace ttml::serialization
