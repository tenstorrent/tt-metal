// minimal_safetensors_sax.hpp (single TU demo, C++20)
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// JSON: use any you like; nlohmann/json for brevity here
#include <nlohmann/json.hpp>

#include "fmt/format.h"

// POSIX mmap
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ttml::serialization {
struct TensorInfo {
    std::string_view name;
    std::string dtype;           // "F16","BF16","F32","I64",...
    std::vector<int64_t> shape;  // dims
};

// Return false from callback to stop early.
using TensorCallback = std::function<bool(const TensorInfo& info, std::span<const std::byte> bytes)>;

static uint64_t le64(const unsigned char* p) {
    // little-endian decode (portable)
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

inline void visit_safetensors_file(const std::filesystem::path& path, const TensorCallback& cb) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("file does not exist: " + path.string());
    }
    // use unique_ptr for RAII, but here we use raw file descriptor for simplicity
    auto file_closer = [](int fd) {
        if (fd) {
            ::close(fd);
        }
    };
    std::unique_ptr<int, decltype(file_closer)> fd(::open(path.c_str(), "r"), file_closer);
    struct stat st{};
    if (fstat(*fd.get(), &st) != 0) {
        throw std::system_error(errno, std::generic_category(), "fstat");
    }
    const size_t file_size = size_t(st.st_size);
    if (file_size < 8) {
        throw std::runtime_error("file too small for safetensors");
    }
    auto mmap_deleter = [&file_size](void* addr) {
        if (addr != MAP_FAILED) {
            munmap(addr, file_size);
        }
    };

    // Create a unique_ptr with the custom deleter
    std::unique_ptr<void, decltype(mmap_deleter)> map(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, *fd.get(), 0), mmap_deleter);

    auto* base = reinterpret_cast<const unsigned char*>(map.get());
    const uint64_t header_len = le64(base);
    if (8 + header_len > file_size) {
        throw std::runtime_error("header length out of range");
    }

    const char* header_begin = reinterpret_cast<const char*>(base + 8);
    nlohmann::json j = nlohmann::json::parse(header_begin, header_begin + header_len);

    const size_t data_offset = 8 + size_t(header_len);
    if (data_offset > file_size) {
        throw std::runtime_error(fmt::format("bad data offset: {} > {}", data_offset, file_size));
    }

    const size_t data_size = file_size - data_offset;
    const std::byte* data_base = reinterpret_cast<const std::byte*>(base + data_offset);

    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        const auto& obj = it.value();
        const auto& dtype = obj.at("dtype").get_ref<const std::string&>();
        const auto& shape_json = obj.at("shape");
        std::vector<int64_t> shape;
        shape.reserve(shape_json.size());
        for (const auto& d : shape_json) shape.push_back(d.get<int64_t>());

        const auto offs = obj.at("data_offsets");
        uint64_t begin = offs.at(0).get<uint64_t>();
        uint64_t end = offs.at(1).get<uint64_t>();
        if (end < begin) {
            throw std::runtime_error("end < begin");
        }
        const uint64_t len = end - begin;

        if (begin > data_size || len > data_size || begin + len > data_size) {
            throw std::runtime_error("data_offsets out of range");
        }

        TensorInfo info{std::string_view(it.key()), dtype, std::move(shape)};
        std::span<const std::byte> bytes{data_base + begin, static_cast<size_t>(len)};

        if (!cb(info, bytes)) {
            break;
        }
    }
}

}  // namespace ttml::serialization
