// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared scaffolding for the doorbell and coherent-alias device tests: the
// argument block their images serve (src/x280_hw_probe.h), and a one-core Tensix
// kernel (kernels/x280_noc_probe.cpp) that reads and writes the tile and DRAM
// beside the X280 session. The caller orders the two sides: every probe call
// blocks until its kernel and its result read have finished.

#pragma once

#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_deadline.hpp"

#include "src/x280_hw_probe.h"
#include "x280_address_map.h"
#include "x280_test_images.hpp"

#include <gtest/gtest.h>
#include <mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::test {

namespace distributed = tt::tt_metal::distributed;

/** The argument block in the tile's available LIM, at its first line. */
class ProbeBlock {
public:
    explicit ProbeBlock(X280Tile& tile) : tile_(&tile), base_(tile.available_lim().begin) {
        if (tile.available_lim().size() < X280_PROBE_BLOCK_BYTES || base_ % X280_PROBE_LINE_BYTES != 0) {
            throw std::runtime_error("available LIM cannot hold the probe block");
        }
    }

    uint64_t address() const { return base_; }
    uint64_t bytes() const { return X280_PROBE_BLOCK_BYTES; }

    /** Writes the whole block: zeros, then the header. Every line the image
     * stores into afterwards has been written whole, so its ECC is initialised. */
    void reset(uint64_t run_value, uint64_t ddr_offset = 0, uint64_t region_lines = 0) {
        std::vector<uint8_t> block(X280_PROBE_BLOCK_BYTES, 0);
        put(block, X280_PROBE_RUN_VALUE, run_value);
        put(block, X280_PROBE_DDR_OFFSET, ddr_offset);
        put(block, X280_PROBE_REGION_LINES, region_lines);
        tile_->write_available_lim(base_, block);
        seq_ = 0;
    }

    uint64_t read_u64(uint32_t offset) const {
        const auto bytes = tile_->read_available_lim(base_ + offset, sizeof(uint64_t));
        uint64_t value = 0;
        std::memcpy(&value, bytes.data(), sizeof(value));
        return value;
    }

    /** Polls the word at `offset` until `pred` holds or `timeout` passes. */
    template <typename Pred>
    bool wait_u64(uint32_t offset, std::chrono::milliseconds timeout, Pred pred) const {
        const Deadline deadline(timeout);
        return poll_until(deadline, std::chrono::milliseconds(1), [&] { return pred(read_u64(offset)); }).satisfied;
    }

    /** Publishes one command (line 1, then its sequence on line 2) and waits
     * for the image's response. Returns the response status, or 0 on timeout. */
    uint64_t command(uint64_t op, uint64_t first, uint64_t count, uint64_t seed, std::chrono::milliseconds timeout) {
        std::vector<uint8_t> line(X280_PROBE_LINE_BYTES, 0);
        put(line, X280_PROBE_CMD_OP - X280_PROBE_CMD_OP, op);
        put(line, X280_PROBE_CMD_FIRST - X280_PROBE_CMD_OP, first);
        put(line, X280_PROBE_CMD_COUNT - X280_PROBE_CMD_OP, count);
        put(line, X280_PROBE_CMD_SEED - X280_PROBE_CMD_OP, seed);
        tile_->write_available_lim(base_ + X280_PROBE_CMD_OP, line);
        const uint64_t seq = ++seq_;
        std::vector<uint8_t> doorbell(X280_PROBE_LINE_BYTES, 0);
        put(doorbell, 0, seq);
        tile_->write_available_lim(base_ + X280_PROBE_CMD_SEQ, doorbell);
        if (!wait_u64(X280_PROBE_RESP_SEQ, timeout, [&](uint64_t done) { return done == seq; })) {
            return 0;
        }
        return read_u64(X280_PROBE_RESP_STATUS);
    }

    /** Line `index` of the data area, as sixteen u32 words. */
    std::array<uint32_t, 16> data_line(uint32_t index) const {
        const auto bytes =
            tile_->read_available_lim(base_ + X280_PROBE_DATA + index * X280_PROBE_LINE_BYTES, X280_PROBE_LINE_BYTES);
        std::array<uint32_t, 16> words{};
        std::memcpy(words.data(), bytes.data(), X280_PROBE_LINE_BYTES);
        return words;
    }

private:
    static void put(std::vector<uint8_t>& bytes, uint32_t offset, uint64_t value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    X280Tile* tile_;
    uint64_t base_;
    uint64_t seq_ = 0;
};

/** The pattern line `line` of `seed`, as the image and the kernel write it. */
inline std::array<uint32_t, 16> pattern_line(uint32_t seed, uint32_t line) {
    std::array<uint32_t, 16> words{};
    for (uint32_t w = 0; w < words.size(); ++w) {
        words[w] = x280_probe_pattern(seed, line, w);
    }
    return words;
}

/**
 * Runs kernels/x280_noc_probe.cpp on logical Tensix core (0, 0), RISCV_0 on NoC 0, one blocking
 * workload per call. Coordinates passed in are translated for NoC 0.
 */
class NocProbe {
public:
    static constexpr uint32_t kOpWriteRaw = 0;
    static constexpr uint32_t kOpReadRaw = 1;
    static constexpr uint32_t kOpWriteBank = 2;
    static constexpr uint32_t kOpReadBank = 3;
    static constexpr uint32_t kOpDoorbell = 4;
    static constexpr uint32_t kMaxLines = X280_PROBE_DATA_LINES;
    static constexpr tt::tt_metal::NOC kNoc = tt::tt_metal::NOC::NOC_0;

    explicit NocProbe(const std::shared_ptr<distributed::MeshDevice>& mesh) : mesh_(mesh) {
        constexpr uint32_t bytes = kMaxLines * X280_PROBE_LINE_BYTES;
        const distributed::ReplicatedBufferConfig size{.size = bytes};
        out_ = distributed::MeshBuffer::create(
            size,
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
            mesh_.get());
        l1_ = distributed::MeshBuffer::create(
            size,
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = tt::tt_metal::BufferType::L1},
            mesh_.get());
    }

    void write_raw(TranslatedCoord at, uint64_t addr, uint32_t lines, uint32_t first, uint32_t seed) {
        (void)run({kOpWriteRaw, at.x(), at.y(), hi(addr), lo(addr), lines, first, seed});
    }

    std::vector<std::array<uint32_t, 16>> read_raw(TranslatedCoord at, uint64_t addr, uint32_t lines) {
        return as_lines(run({kOpReadRaw, at.x(), at.y(), hi(addr), lo(addr), lines, 0, 0}), lines);
    }

    void write_bank(uint32_t bank, uint32_t addr, uint32_t lines, uint32_t first, uint32_t seed) {
        (void)run({kOpWriteBank, bank, 0, 0, addr, lines, first, seed});
    }

    std::vector<std::array<uint32_t, 16>> read_bank(uint32_t bank, uint32_t addr, uint32_t lines) {
        return as_lines(run({kOpReadBank, bank, 0, 0, addr, lines, 0, 0}), lines);
    }

    struct DoorbellResult {
        bool woke = false;
        uint64_t ticks = 0;
        uint32_t counter = 0;
        uint32_t polls = 0;
    };

    /** Pushes `value` into the MSI catcher at `at`, then reads the LIM line at
     * `poll_line` until the u32 at `offset` in it leaves `base`, for at most
     * `budget_ticks` of the Tensix wall clock. */
    DoorbellResult doorbell(
        TranslatedCoord at, uint32_t value, uint64_t poll_line, uint32_t offset, uint32_t base, uint32_t budget_ticks) {
        const auto words = run(
            {kOpDoorbell,
             at.x(),
             at.y(),
             0,
             static_cast<uint32_t>(X280_MSI_CATCHER_BASE),
             1,
             0,
             0,
             0,
             0,
             value,
             static_cast<uint32_t>(poll_line),
             offset,
             base,
             budget_ticks});
        return {.woke = words[0] == 1, .ticks = words[1], .counter = words[2], .polls = words[3]};
    }

private:
    static uint32_t hi(uint64_t v) { return static_cast<uint32_t>(v >> 32); }
    static uint32_t lo(uint64_t v) { return static_cast<uint32_t>(v); }

    static std::vector<std::array<uint32_t, 16>> as_lines(const std::vector<uint32_t>& words, uint32_t lines) {
        std::vector<std::array<uint32_t, 16>> out(lines);
        for (uint32_t i = 0; i < lines; ++i) {
            std::memcpy(out[i].data(), words.data() + i * 16, X280_PROBE_LINE_BYTES);
        }
        return out;
    }

    std::vector<uint32_t> run(std::vector<uint32_t> args) {
        if (args.size() > 5 && args[5] > kMaxLines) {
            throw std::invalid_argument("NocProbe: more lines than the scratch holds");
        }
        args.resize(15, 0);
        args[8] = static_cast<uint32_t>(l1_->address());
        args[9] = static_cast<uint32_t>(out_->address());

        constexpr tt::tt_metal::CoreCoord core{0, 0};
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
        const auto kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/experimental/x280/tests/device/kernels/x280_noc_probe.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        tt::tt_metal::SetRuntimeArgs(program, kernel, core, args);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/true);

        std::vector<uint32_t> words;
        distributed::EnqueueReadMeshBuffer(mesh_->mesh_command_queue(), words, out_, /*blocking=*/true);
        return words;
    }

    std::shared_ptr<distributed::MeshDevice> mesh_;
    std::shared_ptr<distributed::MeshBuffer> out_;
    std::shared_ptr<distributed::MeshBuffer> l1_;
};

/**
 * Mesh, prepared runtime and the first tile in resident idle, as the active
 * fixtures set them up; skips where they skip.
 */
class X280ProbeHwBase : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = distributed::MeshDevice::create_unit_mesh(0);
        auto opened =
            X280Runtime::open(mesh_, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
        if (!opened.runtime) {
            GTEST_SKIP() << "prepared X280 runtime unavailable: " << unprepared(opened);
        }
        runtime_ = std::make_unique<X280Runtime>(std::move(*opened.runtime));
        const auto ids = runtime_->tile_ids();
        if (ids.empty()) {
            GTEST_SKIP() << "no enabled X280 tiles";
        }
        boot_resident_idle(*runtime_, ids.front());
        if (IsSkipped() || HasFatalFailure()) {
            return;
        }
        tile_ = &runtime_->tile(ids.front());
        const auto state = tile_->inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (!state.resident_idle_live) {
            GTEST_SKIP() << "requires a live resident idle";
        }
        block_ = std::make_unique<ProbeBlock>(*tile_);
        probe_ = std::make_unique<NocProbe>(mesh_);
    }

    void TearDown() override {
        probe_.reset();
        block_.reset();
        tile_ = nullptr;
        runtime_.reset();
        mesh_.reset();
    }

    /** Enqueues `image` on the tile with the probe block as its arguments. */
    SessionId enqueue(const std::string& image, uint64_t run_value) {
        ActiveRunRequest request;
        request.image_path = image;
        request.run_value = run_value;
        request.args_addr = block_->address();
        request.args_bytes = block_->bytes();
        request.cleanup_timeout = std::chrono::seconds(3);
        const auto enqueued = runtime_->enqueue(*tile_, request);
        if (!enqueued.session) {
            throw std::runtime_error("enqueue did not return an active session: " + describe(enqueued));
        }
        return enqueued.session;
    }

    std::shared_ptr<distributed::MeshDevice> mesh_;
    std::unique_ptr<X280Runtime> runtime_;
    X280Tile* tile_ = nullptr;
    std::unique_ptr<ProbeBlock> block_;
    std::unique_ptr<NocProbe> probe_;
};

}  // namespace x280::test
