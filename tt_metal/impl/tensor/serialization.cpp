// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/tensor/serialization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/cleanup.hpp>

#include "tt-metalium/tensor/tensor_spec.hpp"
#include "tt-metalium/tensor/flatbuffer/tensor_flatbuffer.hpp"

namespace tt::tt_metal {

Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device) {
    constexpr std::uint32_t kFlatbufferAlignment = alignof(std::uint64_t);
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\"", file_name);
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    size_t file_size = file_stat.st_size;

    // Mmap the file to read tensor data lazily.
    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TT_FATAL(mmap_addr != MAP_FAILED, "Failed to mmap file \"{}\": {}", file_name, strerror(errno));

    std::shared_ptr<void> mmap_ptr(mmap_addr, [file_size](void* addr) { munmap(addr, file_size); });
    MemoryPin memory_pin(mmap_ptr);

    auto* file_data = static_cast<std::byte*>(mmap_addr);
    uint64_t header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));

    const auto* header_start = reinterpret_cast<const std::uint8_t*>(file_data) + sizeof(header_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header size is too large; this most likely indicates data corruption.");
    flatbuffers::Verifier verifier(header_start, header_size);
    TT_FATAL(
        tt::tt_metal::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor data; this most likely indicates data corruption.");
    auto fb_tensor = tt::tt_metal::flatbuffer::GetTensor(header_start);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = file_size - data_offset;

    std::byte* data_region = file_data + data_offset;
    TT_FATAL(
        (reinterpret_cast<uintptr_t>(data_region) & (kFlatbufferAlignment - 1)) == 0,
        "Tensor data pointer must be 8-byte aligned!");

    Tensor tensor =
        tt::tt_metal::from_flatbuffer(fb_tensor, tt::stl::Span<std::byte>(data_region, data_size), memory_pin);
    if (device != nullptr) {
        tensor = tensor.to_device(device, tensor.tensor_spec().memory_config());
    }
    return tensor;
}

}  // namespace tt::tt_metal
