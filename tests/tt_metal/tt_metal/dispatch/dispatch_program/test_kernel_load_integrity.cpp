// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-load integrity stress.
//
// Motivation: an eager model workload dispatches millions of ops whose kernels are large
// enough to nearly fill the default 69 KB kernel config buffer, so almost every launch forces
// a ring-buffer wrap and a sync. Nothing in the existing suites checks that the kernel binary
// actually arrives intact -- the kernel-size tests run with nullified kernels and only assert
// that dispatch does not throw, and the random-program tests validate runtime args and CB
// config, never the binary.
//
// This test closes that gap: it dispatches large, real (non-nullified) kernels back to back,
// and each one hashes its own .text pad where it landed in L1 and publishes the result. The
// host requires the hash to be identical on every core, on every iteration -- any truncated,
// torn, or corrupted load shows up immediately, attributed to a core and an iteration.
//
// Knobs (env):
//   KERNEL_LOAD_ITERS  iterations per size    (default 100)
//   KERNEL_LOAD_SIZES  comma-separated bytes  (default 69632 = 68 KB, the max that fits)
//   KERNEL_LOAD_FILLS  comma-separated fill words (default 0xAAAAAAAA,0x55555555)
//   KERNEL_LOAD_GRID   NxM core grid          (default 4x4)

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <fmt/format.h>
#include <sstream>
#include <memory>
#include <algorithm>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "command_queue_fixture.hpp"

namespace tt::tt_metal::kernel_load_integrity {

namespace {

constexpr uint32_t DEFAULT_KERNEL_CONFIG_BUFFER_SIZE = 69 * 1024;
constexpr uint32_t RESULT_WORDS = 16;     // see the kernel for the layout
constexpr uint32_t MARKER_STRIDE = 256u;  // keep in sync with the kernel
constexpr uint32_t MARKER_MAGIC = 0xC0DE0000u;

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr) {
        return fallback;
    }
    return static_cast<uint32_t>(std::stoul(v));
}

std::vector<uint32_t> env_sizes() {
    const char* v = std::getenv("KERNEL_LOAD_SIZES");
    if (v == nullptr) {
        // 69632 B (68 KB) is the largest kernel that fits the 69 KB config buffer -- 70656 B
        // is rejected at program.cpp:2789 (state.offset <= max_size). This is the size a
        // ~69 KB production op kernel actually lands on.
        return {69632};
    }
    std::vector<uint32_t> out;
    std::string s(v);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            out.push_back(static_cast<uint32_t>(std::stoul(tok)));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

// Mirrors the kernel's FNV-1a exactly (little-endian byte order within each word), so the
// host knows the correct hash before the kernel ever runs.
uint32_t expected_word(uint32_t i, uint32_t fill) {
    if ((i % MARKER_STRIDE) == 0u) {
        return MARKER_MAGIC ^ i;
    }
    return fill;
}

uint32_t fnv1a_fill(uint32_t fill, uint32_t words) {
    uint32_t h = 0x811C9DC5u;
    for (uint32_t i = 0; i < words; i++) {
        const uint32_t w = expected_word(i, fill);
        h ^= (w & 0xFFu);
        h *= 16777619u;
        h ^= ((w >> 8) & 0xFFu);
        h *= 16777619u;
        h ^= ((w >> 16) & 0xFFu);
        h *= 16777619u;
        h ^= ((w >> 24) & 0xFFu);
        h *= 16777619u;
    }
    return h;
}

std::vector<uint32_t> env_fills() {
    const char* v = std::getenv("KERNEL_LOAD_FILLS");
    if (v == nullptr) {
        return {0xAAAAAAAAu, 0x55555555u};  // exact bit-complements
    }
    std::vector<uint32_t> out;
    std::string s(v);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            out.push_back(static_cast<uint32_t>(std::stoul(tok, nullptr, 0)));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

}  // namespace

// Same L1 partition as production: worker_l1_size chosen so the kernel config buffer is 69 KB.
class KernelLoadIntegrityFixture : public UnitMeshCQFixture {
protected:
    uint32_t unreserved_base_{};
    uint32_t kernel_config_size_{};
    const Hal& hal_{MetalContext::instance().hal()};
    const uint32_t max_worker_l1_size_{hal::get_max_worker_l1_unreserved_size()};
    const uint32_t kernel_config_base_{
        hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG)};

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        if (arch_ == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Only supported on Wormhole and Blackhole";
        }
        create_devices(DEFAULT_TRACE_REGION_SIZE, max_worker_l1_size_ - DEFAULT_KERNEL_CONFIG_BUFFER_SIZE);
        auto* dev = devices_[0]->get_devices()[0];
        unreserved_base_ = dev->allocator()->get_base_allocator_addr(HalMemType::L1);
        kernel_config_size_ = unreserved_base_ - kernel_config_base_;
        log_info(tt::LogTest, "Kernel config buffer: {} B", kernel_config_size_);
    }
};

TEST_F(KernelLoadIntegrityFixture, TensixLargeKernelLoadsAreIntact) {
    const uint32_t iters = env_u32("KERNEL_LOAD_ITERS", 100);
    const std::vector<uint32_t> sizes = env_sizes();
    const std::vector<uint32_t> fills = env_fills();
    const uint32_t grid_x = env_u32("KERNEL_LOAD_GRID_X", 4);
    const uint32_t grid_y = env_u32("KERNEL_LOAD_GRID_Y", 4);

    auto& mesh = devices_[0];
    auto* dev = mesh->get_devices()[0];
    const CoreCoord compute_grid = dev->compute_with_storage_grid_size();
    const uint32_t gx = std::min<uint32_t>(grid_x, compute_grid.x);
    const uint32_t gy = std::min<uint32_t>(grid_y, compute_grid.y);
    CoreRange cr({0, 0}, {gx - 1, gy - 1});
    CoreRangeSet cr_set({cr});

    const uint32_t out_addr = unreserved_base_;

    log_info(
        tt::LogTest,
        "Kernel load integrity: {} iters x {} sizes on a {}x{} grid, config buffer {} B",
        iters,
        sizes.size(),
        gx,
        gy,
        kernel_config_size_);

    // Build one workload per size ONCE. Rebuilding a program per iteration cost ~110 s per
    // iteration (the large kernel is re-finalized each time) and would have made an overnight
    // run worthless. Re-enqueueing a prebuilt workload is also closer to what an eager model
    // does: the same program dispatched over and over.
    //
    // A size that does not fit the 69 KB config buffer throws from finalize_program_offsets();
    // that is a legitimate configuration limit, not a failure, so drop it and carry on.
    struct Variant {
        uint32_t size;
        uint32_t fill;
        uint32_t pad_words;
        uint32_t expected_hash;
        std::shared_ptr<distributed::MeshWorkload> workload;
    };
    std::vector<Variant> variants;

    for (uint32_t size : sizes) {
        for (uint32_t fill : fills) {
            const uint32_t pad_words = (size - 64) / 4;
            try {
                Program program = CreateProgram();
                auto kid = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/kernel_load_integrity.cpp",
                    cr_set,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .defines = {
                            {"KERNEL_BYTES", std::to_string(size)}, {"PAD_FILL", fmt::format("0x{:08X}u", fill)}}});
                SetRuntimeArgs(program, kid, cr_set, {out_addr, pad_words, fill});

                auto workload = std::make_shared<distributed::MeshWorkload>();
                workload->add_program(device_range_, std::move(program));
                distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), *workload, false);
                distributed::Finish(mesh->mesh_command_queue());

                variants.push_back({size, fill, pad_words, fnv1a_fill(fill, pad_words), std::move(workload)});
                log_info(
                    tt::LogTest,
                    "size {} B fill 0x{:08X} fits the {} B config buffer (expected hash 0x{:08X})",
                    size,
                    fill,
                    kernel_config_size_,
                    variants.back().expected_hash);
            } catch (const std::exception& e) {
                log_info(tt::LogTest, "size {} B fill 0x{:08X} does not fit ({}); skipping", size, fill, e.what());
            }
        }
    }
    ASSERT_FALSE(variants.empty()) << "no kernel variant fits the config buffer";

    // Two 69632 B programs cannot co-reside in a 70656 B config buffer, so alternating the
    // complement pair guarantees an eviction and a genuine re-load on every single iteration,
    // with every bit of the payload toggling between loads.
    if (variants.size() > 1) {
        log_info(
            tt::LogTest,
            "{} variants alternating; 2 x {} B vs {} B config buffer -> every load evicts the previous",
            variants.size(),
            variants[0].size,
            kernel_config_size_);
    }
    uint64_t checks = 0;
    uint64_t bytes_loaded = 0;
    const std::vector<uint8_t> zeros(RESULT_WORDS * sizeof(uint32_t), 0);

    for (uint32_t it = 0; it < iters; it++) {
        // Alternate variants so consecutive programs differ: with the 0xAA/0x55 complement
        // pair at the same size, that means every bit of ~68 KB flips on every reload.
        const uint32_t idx = it % variants.size();
        const Variant& var = variants[idx];
        const uint32_t size = var.size;
        const uint32_t pad_words = var.pad_words;
        const uint32_t tag = var.fill;

        // Wipe the result window first, so a stale value from the previous iteration cannot
        // be mistaken for this iteration's result.
        for (const CoreCoord& core : cr) {
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, out_addr, zeros);
        }

        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), *var.workload, false);
        distributed::Finish(mesh->mesh_command_queue());

        bool iter_ok = true;
        for (const CoreCoord& core : cr) {
            std::vector<uint32_t> res;
            tt::tt_metal::detail::ReadFromDeviceL1(dev, core, out_addr, RESULT_WORDS * sizeof(uint32_t), res);

            // The tag proves this dispatch actually ran on this core; without it a stale
            // result from the previous iteration would read as a pass.
            if (res[4] != tag) {
                ADD_FAILURE() << "iter " << it << " size " << size << " core " << core.str()
                              << ": kernel did not run or result not written (tag " << res[4] << " != " << tag << ")";
                iter_ok = false;
                continue;
            }
            if (res[1] != pad_words) {
                ADD_FAILURE() << "iter " << it << " size " << size << " core " << core.str() << ": hashed " << res[1]
                              << " words, expected " << pad_words;
                iter_ok = false;
                continue;
            }

            // The kernel already compared every word on device and attributed faults to byte
            // lanes; res[5] is the authoritative verdict. The hash is a second, independent
            // check against a value the host computed before the run, so a corruption present
            // even on the very first load is caught (a first-observation golden would bless it).
            if (res[5] != 0 || res[0] != var.expected_hash) {
                std::ostringstream lanes;
                for (int l = 0; l < 4; l++) {
                    lanes << "lane" << l << "=" << res[12 + l] << " (bits 0x" << std::hex
                          << ((res[9] >> (l * 8)) & 0xFFu) << std::dec << ")  ";
                }
                ADD_FAILURE() << "KERNEL LOAD CORRUPTION: iter " << it << " size " << size << " fill 0x" << std::hex
                              << var.fill << std::dec << " core " << core.str() << "\n"
                              << "    bad words " << res[5] << " of " << res[1] << ", bad markers " << res[10]
                              << (res[10] ? "  <-- OFFSET/DUPLICATION" : "") << "\n"
                              << "    first bad word #" << res[6] << ": got 0x" << std::hex << res[7] << " expected 0x"
                              << res[8] << " (diff 0x" << (res[7] ^ res[8]) << ")" << std::dec << "\n"
                              << "    per-lane bad word counts: " << lanes.str() << "\n"
                              << "    hash 0x" << std::hex << res[0] << " expected 0x" << var.expected_hash << std::dec;
                iter_ok = false;
                continue;
            }
            checks++;
            bytes_loaded += size;
        }

        if ((it % 25) == 0 || !iter_ok) {
            log_info(
                tt::LogTest,
                "iter {}/{} size {} B fill 0x{:08X}: {} cores verified, {} checks so far, {:.2f} GB loaded",
                it,
                iters,
                size,
                var.fill,
                gx * gy,
                checks,
                bytes_loaded / 1e9);
        }
    }

    log_info(
        tt::LogTest,
        "Kernel load integrity done: {} verifications, {:.2f} GB of kernel binaries loaded",
        checks,
        bytes_loaded / 1e9);
    EXPECT_GT(checks, 0u);
}

}  // namespace tt::tt_metal::kernel_load_integrity
