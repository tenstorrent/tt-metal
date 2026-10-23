// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Host-only BFLOAT4_B tile packer for the Qwen3.8 routed-weight staging tool.
//
// The TTNN host conversion reaches MetalContext::instance() to obtain the L1
// alignment even when no device is requested.  That is inappropriate for the
// CPU-only staging lane.  This small program implements the exact 32x32,
// Bfp4_b, exponent-b encoding used by blockfloat_common.cpp without linking
// TT-Metal or performing device discovery.  Input is row-major BF16 on stdin;
// output is the packed tile payload on stdout.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kTile = 32;
constexpr std::size_t kFace = 16;
constexpr std::size_t kTileElements = kTile * kTile;
constexpr std::size_t kExponentBytes = 64;
constexpr std::size_t kMantissaBytes = kTileElements / 2;
constexpr std::size_t kPackedTileBytes = kExponentBytes + kMantissaBytes;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "bf4_host_packer: %s\n", message);
    std::exit(2);
}

std::size_t parse_size(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        fail(label);
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > std::numeric_limits<std::size_t>::max()) {
        fail(label);
    }
    return static_cast<std::size_t>(value);
}

std::uint8_t bfp4_mantissa(std::uint32_t bits, std::uint8_t shared_exp) {
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    if (exponent == 0U) {
        return 0;
    }

    std::uint32_t mantissa = (1U << 23U) | (bits & 0x007fffffU);
    if (shared_exp > exponent) {
        const std::uint32_t shift = shared_exp - exponent;
        mantissa = shift >= 32U ? 0U : mantissa >> shift;
    }

    // Match convert_u32_to_bfp<Bfp4_b, false>: round to nearest, ties to even.
    constexpr std::uint32_t shift = 21U;
    constexpr std::uint32_t mask = (1U << shift) - 1U;
    constexpr std::uint32_t tie = 1U << (shift - 1U);
    const std::uint32_t round_value = mantissa & mask;
    mantissa >>= shift;
    const std::uint32_t guard_bit = mantissa & 1U;
    if (round_value > tie || (round_value == tie && guard_bit == 1U)) {
        ++mantissa;
    }
    mantissa = std::min(mantissa, 7U);
    if (mantissa == 0U) {
        return 0;
    }
    const std::uint32_t sign = bits >> 31U;
    return static_cast<std::uint8_t>((sign << 3U) | mantissa);
}

void pack_tile(
    const std::vector<std::uint16_t>& input,
    std::size_t rows,
    std::size_t cols,
    std::size_t tile_index,
    std::uint8_t* output) {
    const std::size_t tiles_per_row = cols / kTile;
    const std::size_t tile_row = tile_index / tiles_per_row;
    const std::size_t tile_col = tile_index % tiles_per_row;
    const std::size_t row_base = tile_row * kTile;
    const std::size_t col_base = tile_col * kTile;
    (void)rows;

    std::array<std::uint8_t, kExponentBytes> exponents{};
    std::array<std::uint8_t, kMantissaBytes> mantissas{};
    std::size_t row_index = 0;
    std::size_t mantissa_index = 0;

    for (std::size_t face_row = 0; face_row < 2; ++face_row) {
        for (std::size_t face_col = 0; face_col < 2; ++face_col) {
            for (std::size_t row = 0; row < kFace; ++row) {
                std::array<std::uint32_t, kFace> values{};
                std::uint8_t shared_exp = 0;
                for (std::size_t col = 0; col < kFace; ++col) {
                    const std::size_t source_row = row_base + face_row * kFace + row;
                    const std::size_t source_col = col_base + face_col * kFace + col;
                    const std::uint32_t bits = static_cast<std::uint32_t>(input[source_row * cols + source_col]) << 16U;
                    values[col] = bits;
                    shared_exp = std::max(shared_exp, static_cast<std::uint8_t>((bits >> 23U) & 0xffU));
                }
                exponents[row_index++] = shared_exp;
                for (std::size_t col = 0; col < kFace; col += 2) {
                    const std::uint8_t low = bfp4_mantissa(values[col], shared_exp);
                    const std::uint8_t high = bfp4_mantissa(values[col + 1], shared_exp);
                    mantissas[mantissa_index++] = static_cast<std::uint8_t>(low | (high << 4U));
                }
            }
        }
    }

    std::memcpy(output, exponents.data(), exponents.size());
    std::memcpy(output + exponents.size(), mantissas.data(), mantissas.size());
}

void read_exact(void* destination, std::size_t bytes) {
    auto* cursor = static_cast<std::uint8_t*>(destination);
    std::size_t offset = 0;
    while (offset < bytes) {
        const std::size_t count = std::fread(cursor + offset, 1, bytes - offset, stdin);
        if (count == 0) {
            if (std::ferror(stdin)) {
                fail("stdin read failed");
            }
            fail("stdin ended before the declared BF16 tensor");
        }
        offset += count;
    }
    if (std::fgetc(stdin) != EOF) {
        fail("stdin contains bytes after the declared BF16 tensor");
    }
}

void write_exact(const void* source, std::size_t bytes) {
    const auto* cursor = static_cast<const std::uint8_t*>(source);
    std::size_t offset = 0;
    while (offset < bytes) {
        const std::size_t count = std::fwrite(cursor + offset, 1, bytes - offset, stdout);
        if (count == 0) {
            fail("stdout write failed");
        }
        offset += count;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string_view(argv[1]) != "--rows" || std::string_view(argv[3]) != "--cols") {
        fail("usage: bf4_host_packer --rows ROWS --cols COLS");
    }
    const std::size_t rows = parse_size(argv[2], "invalid --rows");
    const std::size_t cols = parse_size(argv[4], "invalid --cols");
    if (rows == 0 || cols == 0 || rows % kTile != 0 || cols % kTile != 0) {
        fail("rows and cols must be positive multiples of 32");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        fail("input element count overflows size_t");
    }
    const std::size_t elements = rows * cols;
    std::vector<std::uint16_t> input(elements);
    read_exact(input.data(), elements * sizeof(std::uint16_t));

    const std::size_t tiles = elements / kTileElements;
    std::vector<std::uint8_t> packed(tiles * kPackedTileBytes);
    unsigned workers = std::thread::hardware_concurrency();
    workers = workers == 0 ? 1U : workers;
    constexpr unsigned kDefaultWorkerLimit = 16;
    unsigned worker_limit = kDefaultWorkerLimit;
    if (const char* configured = std::getenv("QWEN38_BF4_PACKER_THREADS")) {
        const std::size_t parsed = parse_size(configured, "invalid QWEN38_BF4_PACKER_THREADS");
        if (parsed == 0 || parsed > std::numeric_limits<unsigned>::max()) {
            fail("QWEN38_BF4_PACKER_THREADS is outside the unsigned range");
        }
        worker_limit = static_cast<unsigned>(parsed);
    }
    workers = std::min(workers, worker_limit);
    workers = std::min<unsigned>(workers, static_cast<unsigned>(std::max<std::size_t>(1, tiles / 256)));

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        const std::size_t begin = tiles * worker / workers;
        const std::size_t end = tiles * (worker + 1) / workers;
        threads.emplace_back([&, begin, end] {
            for (std::size_t tile = begin; tile < end; ++tile) {
                pack_tile(input, rows, cols, tile, packed.data() + tile * kPackedTileBytes);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    write_exact(packed.data(), packed.size());
    return std::fflush(stdout) == 0 ? 0 : 2;
}
