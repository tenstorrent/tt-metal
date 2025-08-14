// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <memory>
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

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "safetensors.hpp"

namespace ttml::serialization {
uint64_t le64(const unsigned char* p) {
    // little-endian decode (portable)
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

struct UniqueFd {
    int fd{-1};

    UniqueFd() = default;
    explicit UniqueFd(int f) noexcept : fd(f) {
    }
    ~UniqueFd() {
        if (fd >= 0)
            ::close(fd);
    }

    UniqueFd(UniqueFd&& other) noexcept : fd(std::exchange(other.fd, -1)) {
    }
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) {
            reset();
            fd = std::exchange(other.fd, -1);
        }
        return *this;
    }

    void reset(int f = -1) noexcept {
        if (fd >= 0)
            ::close(fd);
        fd = f;
    }
    int get() const noexcept {
        return fd;
    }
    explicit operator bool() const noexcept {
        return fd >= 0;
    }
    int release() noexcept {
        return std::exchange(fd, -1);
    }
};

void SafetensorSerialization::visit_safetensors_file(const std::filesystem::path& path, const TensorCallback& cb) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("file does not exist: " + path.string());
    }

    UniqueFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    struct stat st{};
    if (fstat(fd.get(), &st) != 0) {
        throw std::system_error(errno, std::generic_category(), "fstat");
    }
    const size_t file_size = size_t(st.st_size);

    if (file_size < 8) {
        throw std::runtime_error("file too small for safetensors");
    }

    auto mmap_deleter = [file_size](void* addr) {
        if (addr != MAP_FAILED) {
            munmap(addr, file_size);
        }
    };

    std::unique_ptr<void, decltype(mmap_deleter)> map(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd.get(), 0), mmap_deleter);

    auto* base = reinterpret_cast<const unsigned char*>(map.get());

    const uint64_t header_len = le64(base);
    if (8 + header_len > file_size) {
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

        if (begin > data_size || len > data_size || begin + len > data_size) {
            throw std::runtime_error(fmt::format("data_offsets out of range: {} > {}", begin + len, data_size));
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
    if (n) {
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
    }
    return out;
}
}  // namespace ttml::serialization
