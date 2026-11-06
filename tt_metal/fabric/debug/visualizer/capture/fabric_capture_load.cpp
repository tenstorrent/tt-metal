// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_load.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::tt_fabric::capture {

namespace {

// Maps all of `path` read-only, returning null for an empty file. The mapping outlives the file descriptor.
std::pair<const std::byte*, size_t> map_file(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd >= 0, "Fabric capture: can't open {}: {}", path.string(), std::strerror(errno));
    struct stat status{};
    if (::fstat(fd, &status) != 0) {
        const int error = errno;
        ::close(fd);
        TT_THROW("Fabric capture: can't stat {}: {}", path.string(), std::strerror(error));
    }
    const auto size_bytes = static_cast<size_t>(status.st_size);
    if (size_bytes == 0) {
        ::close(fd);
        return {nullptr, 0};
    }
    void* mapping = ::mmap(nullptr, size_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    const int error = errno;
    ::close(fd);
    TT_FATAL(mapping != MAP_FAILED, "Fabric capture: can't map {}: {}", path.string(), std::strerror(error));
    return {static_cast<const std::byte*>(mapping), size_bytes};
}

void unmap_file(const std::byte* bytes, size_t size_bytes) {
    if (bytes != nullptr) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): munmap takes the address map_file got as void*.
        ::munmap(const_cast<std::byte*>(bytes), size_bytes);
    }
}

}  // namespace

// Map the .bin file and validate that it matches the record.
LoadedCapture::LoadedCapture(CaptureRecord record, const std::filesystem::path& bin_path) : record_(std::move(record)) {
    std::tie(bin_bytes_, bin_size_bytes_) = map_file(bin_path);
    try {
        TT_FATAL(
            bin_size_bytes_ == record_.bin_size_bytes,
            "Fabric capture: {} is {} bytes, but its record says {}",
            bin_path.string(),
            bin_size_bytes_,
            record_.bin_size_bytes);
        const std::string blake3 = compute_blake3_hex({bin_bytes_, bin_size_bytes_});
        TT_FATAL(
            blake3 == record_.bin_blake3,
            "Fabric capture: {} has BLAKE3 {}, but its record says {}; it isn't the .bin the record was written with",
            bin_path.string(),
            blake3,
            record_.bin_blake3);
        for (const ChipRecord& chip : record_.chips) {
            for (const RouterRecord& router : chip.routers) {
                for (const ReadRecord& read : router.reads) {
                    TT_FATAL(
                        !read.bin_offset.has_value() || (*read.bin_offset <= bin_size_bytes_ &&
                                                         read.size_bytes <= bin_size_bytes_ - *read.bin_offset),
                        "Fabric capture: {}'s read at {:#x} runs past the end of {}",
                        router.target.router.path,
                        read.address,
                        bin_path.string());
                }
            }
        }
    } catch (...) {
        unmap_file(bin_bytes_, bin_size_bytes_);
        throw;
    }
}

// Unmaps the .bin file from memory on obj destruction.
LoadedCapture::~LoadedCapture() { unmap_file(bin_bytes_, bin_size_bytes_); }

LoadedCapture::LoadedCapture(LoadedCapture&& other) noexcept :
    record_(std::move(other.record_)),
    bin_bytes_(std::exchange(other.bin_bytes_, nullptr)),
    bin_size_bytes_(std::exchange(other.bin_size_bytes_, 0)) {}

LoadedCapture& LoadedCapture::operator=(LoadedCapture&& other) noexcept {
    if (this != &other) {
        unmap_file(bin_bytes_, bin_size_bytes_);
        record_ = std::move(other.record_);
        bin_bytes_ = std::exchange(other.bin_bytes_, nullptr);
        bin_size_bytes_ = std::exchange(other.bin_size_bytes_, 0);
    }
    return *this;
}

std::span<const std::byte> LoadedCapture::get_read_bytes(const ReadRecord& read) const {
    TT_FATAL(
        read.bin_offset.has_value(),
        "Fabric capture: the read at {:#x} failed ({}), so it has no bytes",
        read.address,
        read.error.value_or("no error recorded"));
    return {bin_bytes_ + *read.bin_offset, read.size_bytes};
}

LoadedCapture load_capture(const std::filesystem::path& record_path) {
    std::ifstream in(record_path);
    TT_FATAL(in.is_open(), "Fabric capture: can't open the capture record {}", record_path.string());
    nlohmann::ordered_json document;
    try {
        document = nlohmann::ordered_json::parse(in);
    } catch (const nlohmann::ordered_json::parse_error& error) {
        TT_THROW("Fabric capture: the capture record {} isn't JSON: {}", record_path.string(), error.what());
    }
    CaptureRecord record = parse_capture_record(document);
    // The .bin is named, not pathed, so it's always the one beside its record.
    TT_FATAL(
        std::filesystem::path(record.bin_file).filename() == record.bin_file,
        "Fabric capture: the capture record {} names its .bin \"{}\", which isn't a file name",
        record_path.string(),
        record.bin_file);
    const std::filesystem::path bin_path = record_path.parent_path() / record.bin_file;
    return LoadedCapture(std::move(record), bin_path);
}

}  // namespace tt::tt_fabric::capture
