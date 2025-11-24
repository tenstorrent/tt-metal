// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tar_writer.hpp"

#include <fcntl.h>
#include <fmt/format.h>
#include <tar.h>
#include <unistd.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

template <typename T>
concept NoThrowInvokable = std::is_nothrow_invocable_v<T>;

template <NoThrowInvokable T>
struct ScopeExit {
    ScopeExit(T&& dtor_func) : m_dtor_func(std::forward<decltype(dtor_func)>(dtor_func)) {
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&& other) = delete;
    ScopeExit& operator=(ScopeExit&& other) = delete;

    ~ScopeExit() {
        m_dtor_func();
    }

    T m_dtor_func;
};

}  // namespace

namespace ttml::serialization {

TarWriter::TarWriter() = default;

void TarWriter::add_file(std::string_view filename, std::vector<uint8_t>&& data) {
    FileEntry entry;
    entry.filename = std::string(filename);
    entry.data = std::move(data);
    m_files.push_back(std::move(entry));
}

std::string TarWriter::to_octal(uint64_t value, size_t width) const {
    std::ostringstream oss;
    oss << std::oct << std::setfill('0') << std::setw(width) << value;
    return oss.str();
}

void TarWriter::write_header(std::vector<uint8_t>& tarball, const FileEntry& entry) const {
    std::vector<uint8_t> header(tar_format::header::HEADER_SIZE, 0);

    std::string filename = entry.filename;
    if (filename.size() > tar_format::header::FILENAME_SIZE) {
        throw std::runtime_error("Filename too long for tar format: " + filename);
    }
    std::memcpy(header.data() + tar_format::header::FILENAME_OFFSET, filename.c_str(), filename.size());

    size_t mode_len = std::min(strlen(tar_format::defaults::MODE), tar_format::header::MODE_SIZE - 1);
    std::memcpy(header.data() + tar_format::header::MODE_OFFSET, tar_format::defaults::MODE, mode_len);
    header[tar_format::header::MODE_OFFSET + tar_format::header::MODE_SIZE - 1] = '\0';

    size_t uid_len = std::min(strlen(tar_format::defaults::UID), tar_format::header::UID_SIZE - 1);
    std::memcpy(header.data() + tar_format::header::UID_OFFSET, tar_format::defaults::UID, uid_len);
    header[tar_format::header::UID_OFFSET + tar_format::header::UID_SIZE - 1] = '\0';

    size_t gid_len = std::min(strlen(tar_format::defaults::GID), tar_format::header::GID_SIZE - 1);
    std::memcpy(header.data() + tar_format::header::GID_OFFSET, tar_format::defaults::GID, gid_len);
    header[tar_format::header::GID_OFFSET + tar_format::header::GID_SIZE - 1] = '\0';

    std::string size_str = to_octal(entry.data.size(), tar_format::header::SIZE_SIZE - 1);
    size_t size_len = std::min(size_str.size(), tar_format::header::SIZE_SIZE - 1);
    std::memcpy(header.data() + tar_format::header::SIZE_OFFSET, size_str.c_str(), size_len);
    header[tar_format::header::SIZE_OFFSET + tar_format::header::SIZE_SIZE - 1] = '\0';

    std::time_t current_time = std::time(nullptr);
    std::string mtime_str = to_octal(static_cast<uint64_t>(current_time), tar_format::header::MTIME_SIZE - 1);
    size_t mtime_len = std::min(mtime_str.size(), tar_format::header::MTIME_SIZE - 1);
    std::memcpy(header.data() + tar_format::header::MTIME_OFFSET, mtime_str.c_str(), mtime_len);
    header[tar_format::header::MTIME_OFFSET + tar_format::header::MTIME_SIZE - 1] = '\0';

    std::memcpy(header.data() + tar_format::header::MAGIC_OFFSET, TMAGIC, TMAGLEN - 1);
    header[tar_format::header::MAGIC_OFFSET + TMAGLEN - 1] = '\0';

    std::memcpy(header.data() + tar_format::header::VERSION_OFFSET, TVERSION, TVERSLEN);

    header[tar_format::header::TYPE_FLAG_OFFSET] = REGTYPE;

    uint32_t checksum = 0;
    for (size_t i = 0; i < tar_format::header::HEADER_SIZE; ++i) {
        if (i >= tar_format::header::CHECKSUM_OFFSET &&
            i < tar_format::header::CHECKSUM_OFFSET + tar_format::header::CHECKSUM_SIZE) {
            checksum += static_cast<uint8_t>(' ');
        } else {
            checksum += static_cast<uint8_t>(header[i]);
        }
    }

    std::string checksum_str = to_octal(checksum, tar_format::header::CHECKSUM_SIZE - 2);
    size_t checksum_len = std::min(checksum_str.size(), tar_format::header::CHECKSUM_SIZE - 2);
    std::memcpy(header.data() + tar_format::header::CHECKSUM_OFFSET, checksum_str.c_str(), checksum_len);
    header[tar_format::header::CHECKSUM_OFFSET + tar_format::header::CHECKSUM_SIZE - 2] = '\0';
    header[tar_format::header::CHECKSUM_OFFSET + tar_format::header::CHECKSUM_SIZE - 1] = ' ';

    tarball.insert(tarball.end(), header.begin(), header.end());
}

void TarWriter::pad_to_block(std::vector<uint8_t>& data) const {
    size_t remainder = data.size() % tar_format::BLOCK_SIZE;
    if (remainder != 0) {
        size_t padding = tar_format::BLOCK_SIZE - remainder;
        data.insert(data.end(), padding, 0);
    }
}

std::vector<uint8_t> TarWriter::get_tarball() const {
    std::vector<uint8_t> tarball;

    for (const auto& entry : m_files) {
        write_header(tarball, entry);
        tarball.insert(tarball.end(), entry.data.begin(), entry.data.end());
        pad_to_block(tarball);
    }

    tarball.insert(tarball.end(), tar_format::BLOCK_SIZE * tar_format::END_BLOCKS, 0);

    return tarball;
}

void TarWriter::write_to_file(std::string_view filename, bool use_tarball, bool compress) const {
    if (!use_tarball) {
        std::filesystem::path dir_path(filename);
        std::filesystem::create_directories(dir_path);

        for (const auto& entry : m_files) {
            std::filesystem::path file_path = dir_path / entry.filename;

            const std::string file_path_str = file_path.string();
            int fd = open(file_path_str.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                throw std::runtime_error("Failed to open file for writing: " + file_path.string());
            }
            ScopeExit close_fd([&fd]() noexcept {
                if (fd >= 0) {
                    close(fd);
                    fd = -1;
                }
            });

            const size_t file_data_size = entry.data.size();
            size_t total_written = 0;
            const uint8_t* data_ptr = entry.data.data();

            while (total_written < file_data_size) {
                const size_t remaining = file_data_size - total_written;
                ssize_t written = write(fd, data_ptr + total_written, remaining);
                if (written < 0) {
                    throw std::runtime_error("Failed to write data to file: " + file_path.string());
                }
                if (written == 0) {
                    throw std::runtime_error("write() returned 0 for file: " + file_path.string());
                }
                total_written += static_cast<size_t>(written);
            }

            if (fsync(fd) != 0) {
                throw std::runtime_error("Failed to sync file: " + file_path.string());
            }
        }

        return;
    }

    constexpr size_t chunk_size = 32 * 1024 * 1024;
    const int compression_level = 1;

    int fd = open(std::string(filename).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd < 0) {
        throw std::runtime_error("Unable to open file for writing: " + std::string(filename));
    }

    ScopeExit close_fd([&fd]() noexcept {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    });

    off_t file_offset = 0;

    if (compress) {
        std::vector<uint8_t> tarball;
        for (const auto& entry : m_files) {
            std::vector<uint8_t> header;
            write_header(header, entry);
            tarball.insert(tarball.end(), header.begin(), header.end());
            tarball.insert(tarball.end(), entry.data.begin(), entry.data.end());

            const size_t remainder = entry.data.size() % tar_format::BLOCK_SIZE;
            if (remainder != 0) {
                const size_t padding_size = tar_format::BLOCK_SIZE - remainder;
                tarball.insert(tarball.end(), padding_size, 0);
            }
        }

        const size_t end_blocks_size = tar_format::BLOCK_SIZE * tar_format::END_BLOCKS;
        tarball.insert(tarball.end(), end_blocks_size, 0);

        const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t num_chunks = (tarball.size() + chunk_size - 1) / chunk_size;

        struct ChunkResult {
            std::vector<uint8_t> compressed_data;
            size_t compressed_size = 0;
            std::exception_ptr exception = nullptr;
        };

        std::vector<ChunkResult> chunk_results(num_chunks);

        std::vector<std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>> contexts;
        contexts.reserve(num_threads);
        for (unsigned i = 0; i < num_threads; ++i) {
            auto cctx = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>(ZSTD_createCCtx(), ZSTD_freeCCtx);
            if (cctx == nullptr) {
                throw std::runtime_error("Failed to create ZSTD compression context");
            }

            size_t zresult = ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, compression_level);
            if (ZSTD_isError(zresult)) {
                throw std::runtime_error("Failed to set compression level: " + std::string(ZSTD_getErrorName(zresult)));
            }

            ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_strategy, ZSTD_fast);
            contexts.push_back(std::move(cctx));
        }

        auto compress_chunk = [](ZSTD_CCtx* cctx, const uint8_t* data, size_t size, ChunkResult& result) {
            try {
                const size_t compressed_bound = ZSTD_compressBound(size);
                result.compressed_data.resize(compressed_bound);

                const size_t compressed_size =
                    ZSTD_compress2(cctx, result.compressed_data.data(), compressed_bound, data, size);
                if (ZSTD_isError(compressed_size)) {
                    throw std::runtime_error(
                        "ZSTD compression failed: " + std::string(ZSTD_getErrorName(compressed_size)));
                }

                result.compressed_data.resize(compressed_size);
                result.compressed_size = compressed_size;
            } catch (...) {
                result.exception = std::current_exception();
            }
        };

        std::atomic<size_t> next_chunk_idx{0};
        std::vector<std::jthread> threads;

        for (unsigned t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                ZSTD_CCtx* cctx = contexts[t].get();
                while (true) {
                    const size_t chunk_idx = next_chunk_idx.fetch_add(1);
                    if (chunk_idx >= num_chunks) {
                        break;
                    }

                    const size_t offset = chunk_idx * chunk_size;
                    const size_t size = std::min(chunk_size, tarball.size() - offset);
                    compress_chunk(cctx, tarball.data() + offset, size, chunk_results[chunk_idx]);
                }
            });
        }

        threads.clear();
        contexts.clear();

        for (size_t i = 0; i < num_chunks; ++i) {
            if (chunk_results[i].exception) {
                std::rethrow_exception(chunk_results[i].exception);
            }
        }

        std::vector<size_t> offsets(num_chunks);
        size_t current_offset = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            offsets[i] = current_offset;
            current_offset += chunk_results[i].compressed_size;
        }

        std::mutex error_mutex;
        std::exception_ptr write_exception = nullptr;

        for (unsigned t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t, fd, num_chunks]() {
                for (size_t i = t; i < num_chunks; i += num_threads) {
                    try {
                        const ssize_t written = pwrite(
                            fd, chunk_results[i].compressed_data.data(), chunk_results[i].compressed_size, offsets[i]);
                        if (written < 0 || static_cast<size_t>(written) != chunk_results[i].compressed_size) {
                            throw std::runtime_error(fmt::format("Failed to write chunk {} to file", i));
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!write_exception) {
                            write_exception = std::current_exception();
                        }
                        break;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (write_exception) {
            std::rethrow_exception(write_exception);
        }
    } else {
        auto write_direct = [&](const void* data, size_t size) {
            ssize_t written = pwrite(fd, data, size, file_offset);
            if (written < 0 || static_cast<size_t>(written) != size) {
                throw std::runtime_error("Failed to write data to file");
            }
            file_offset += size;
        };

        for (const auto& entry : m_files) {
            std::vector<uint8_t> header;
            write_header(header, entry);
            write_direct(header.data(), header.size());

            const uint8_t* file_data = entry.data.data();
            size_t file_data_size = entry.data.size();
            size_t file_data_offset = 0;

            while (file_data_offset < file_data_size) {
                const size_t write_size = std::min(chunk_size, file_data_size - file_data_offset);
                write_direct(file_data + file_data_offset, write_size);
                file_data_offset += write_size;
            }

            const size_t remainder = entry.data.size() % tar_format::BLOCK_SIZE;
            if (remainder != 0) {
                const size_t padding_size = tar_format::BLOCK_SIZE - remainder;
                std::vector<uint8_t> padding(padding_size, 0);
                write_direct(padding.data(), padding_size);
            }
        }

        const size_t end_blocks_size = tar_format::BLOCK_SIZE * tar_format::END_BLOCKS;
        std::vector<uint8_t> end_blocks(end_blocks_size, 0);
        write_direct(end_blocks.data(), end_blocks_size);
    }
}

}  // namespace ttml::serialization
