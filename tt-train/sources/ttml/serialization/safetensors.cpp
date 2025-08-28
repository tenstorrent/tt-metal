// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

    auto mmap_deleter = [file_size](void* addr) {
        if (addr != MAP_FAILED) {
            munmap(addr, file_size);
        }
    };

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
}  // namespace ttml::serialization
