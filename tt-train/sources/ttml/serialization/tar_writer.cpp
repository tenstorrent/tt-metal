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
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

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

void TarWriter::write_to_file(std::string_view filename, bool compress) const {
    const auto start_time = std::chrono::high_resolution_clock::now();

    constexpr size_t chunk_size = 20 * 1024 * 1024;
    const int compression_level = 3;

    const int fd = open(std::string(filename).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Unable to open file for writing: " + std::string(filename));
    }

    off_t file_offset = 0;

    if (compress) {
        auto cctx = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>(ZSTD_createCCtx(), ZSTD_freeCCtx);
        if (cctx == nullptr) {
            close(fd);
            throw std::runtime_error("Failed to create ZSTD compression context");
        }

        size_t zresult = ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, compression_level);
        if (ZSTD_isError(zresult)) {
            close(fd);
            throw std::runtime_error("Failed to set compression level: " + std::string(ZSTD_getErrorName(zresult)));
        }

        std::vector<uint8_t> output_buffer(chunk_size);

        auto compress_and_write = [&](const void* data, size_t size, ZSTD_EndDirective end_op = ZSTD_e_continue) {
            ZSTD_inBuffer input = {data, size, 0};

            while (input.pos < input.size) {
                ZSTD_outBuffer output = {output_buffer.data(), output_buffer.size(), 0};
                size_t remaining = ZSTD_compressStream2(cctx.get(), &output, &input, ZSTD_e_continue);

                if (ZSTD_isError(remaining)) {
                    close(fd);
                    throw std::runtime_error("ZSTD compression failed: " + std::string(ZSTD_getErrorName(remaining)));
                }

                if (output.pos > 0) {
                    ssize_t written = pwrite(fd, output_buffer.data(), output.pos, file_offset);
                    if (written < 0 || static_cast<size_t>(written) != output.pos) {
                        close(fd);
                        throw std::runtime_error("Failed to write compressed data to file");
                    }
                    file_offset += output.pos;
                }
            }

            if (end_op == ZSTD_e_end) {
                ZSTD_inBuffer final_input = {nullptr, 0, 0};
                while (true) {
                    ZSTD_outBuffer output = {output_buffer.data(), output_buffer.size(), 0};
                    size_t remaining = ZSTD_compressStream2(cctx.get(), &output, &final_input, ZSTD_e_end);

                    if (ZSTD_isError(remaining)) {
                        close(fd);
                        throw std::runtime_error(
                            "ZSTD compression failed: " + std::string(ZSTD_getErrorName(remaining)));
                    }

                    if (output.pos > 0) {
                        ssize_t written = pwrite(fd, output_buffer.data(), output.pos, file_offset);
                        if (written < 0 || static_cast<size_t>(written) != output.pos) {
                            close(fd);
                            throw std::runtime_error("Failed to write compressed data to file");
                        }
                        file_offset += output.pos;
                    }

                    if (remaining == 0) {
                        break;
                    }
                }
            }
        };

        for (const auto& entry : m_files) {
            std::vector<uint8_t> header;
            write_header(header, entry);
            compress_and_write(header.data(), header.size());

            compress_and_write(entry.data.data(), entry.data.size());

            const size_t remainder = entry.data.size() % tar_format::BLOCK_SIZE;
            if (remainder != 0) {
                const size_t padding_size = tar_format::BLOCK_SIZE - remainder;
                std::vector<uint8_t> padding(padding_size, 0);
                compress_and_write(padding.data(), padding_size);
            }
        }

        const size_t end_blocks_size = tar_format::BLOCK_SIZE * tar_format::END_BLOCKS;
        std::vector<uint8_t> end_blocks(end_blocks_size, 0);
        compress_and_write(end_blocks.data(), end_blocks_size, ZSTD_e_end);
    } else {
        auto write_direct = [&](const void* data, size_t size) {
            ssize_t written = pwrite(fd, data, size, file_offset);
            if (written < 0 || static_cast<size_t>(written) != size) {
                close(fd);
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

    close(fd);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = end_time - start_time;
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    constexpr int64_t ns_per_second = 1'000'000'000LL;
    constexpr int64_t ns_per_minute = 60LL * ns_per_second;
    constexpr int64_t ns_per_hour = 60LL * ns_per_minute;

    const int64_t hours = total_ns / ns_per_hour;
    const int64_t minutes = (total_ns % ns_per_hour) / ns_per_minute;
    const int64_t seconds = (total_ns % ns_per_minute) / ns_per_second;
    const int64_t milliseconds = (total_ns % ns_per_second) / 1'000'000LL;
    const int64_t microseconds = (total_ns % 1'000'000LL) / 1'000LL;
    const int64_t nanoseconds = total_ns % 1'000LL;

    fmt::print(
        "Serialization time: {:02d}:{:02d}:{:02d}.{:03d}{:03d}{:03d}\n",
        hours,
        minutes,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds);
}

}  // namespace ttml::serialization
