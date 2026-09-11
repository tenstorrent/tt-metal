// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Measures the 64 B same-offset rule for Tensix <-> L2CPU NOC transfers, for WRITES and
// READS separately, over a matrix of (source offset, destination offset) pairs. Prints,
// for each case, where the pattern actually landed. Env: DEVICE_ID, L2CPU_X/Y,
// PROBE_BASE (default 0x30300000, an unused uncached GDDR alias region).

#include "../l2cpu_fabric_forward/l2cpu_host_utils.hpp"

using namespace tt::tt_metal;
using namespace l2cpu_host;

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int device_id = static_cast<int>(env_or("DEVICE_ID", 0));
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t base = env_or("PROBE_BASE", 0x30300000u);
    const uint32_t rb = base + 0x1000;
    constexpr uint32_t size = 64;

    auto dev = distributed::MeshDevice::create_unit_mesh(device_id);
    X280Mem m(dev, l2cpu_x, l2cpu_y);
    distributed::DeviceLocalBufferConfig l1_cfg{.page_size = 2048, .buffer_type = BufferType::L1};
    auto l1 = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 2048}, l1_cfg, dev.get());
    const uint32_t l1_base = static_cast<uint32_t>(l1->address());
    fmt::print("L1 scratch 0x{:x} (mod 64 = {}), L2CPU probe region 0x{:08x}\n", l1_base, l1_base % 64, base);

    auto run = [&](uint32_t src_off, uint32_t dst_off, bool do_read) {
        Program p = CreateProgram();
        KernelHandle k = CreateKernel(
            p,
            OVERRIDE_KERNEL_PREFIX "l2cpu_noc_transfer/kernels/l2cpu_align_probe.cpp",
            CoreCoord{0, 0},
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
        SetRuntimeArgs(
            p, k, CoreCoord{0, 0}, {l1_base, l2cpu_x, l2cpu_y, base, size, src_off, dst_off, do_read ? 1u : 0u, rb});
        m.run(p);
    };

    // Describe where pattern words 0xA5000000|i are found in a 256 B window.
    auto describe = [&](const std::vector<uint32_t>& w, uint32_t expect_off) {
        int first = -1, count = 0;
        bool contiguous = true;
        for (size_t i = 0; i < w.size(); i++) {
            if ((w[i] & 0xFF000000u) == 0xA5000000u) {
                if (first < 0) {
                    first = static_cast<int>(i);
                }
                if (w[i] != (0xA5000000u | static_cast<uint32_t>(i - first))) {
                    contiguous = false;
                }
                count++;
            }
        }
        if (first < 0) {
            return std::string("pattern NOT found");
        }
        return fmt::format(
            "pattern at +0x{:x} ({} words{}){}",
            first * 4,
            count,
            contiguous ? "" : ", NOT contiguous",
            first * 4 == static_cast<int>(expect_off) ? " = expected" : fmt::format(" ≠ expected +0x{:x}", expect_off));
    };

    const uint32_t offs[][2] = {
        {0, 0}, {16, 0}, {0, 16}, {16, 16}, {48, 0}, {0, 48}, {16, 48}, {48, 48}, {32, 0}, {0, 32}};
    fmt::print("\nWRITE Tensix L1 (+src_off) -> L2CPU (+dst_off), 64 B:\n");
    for (auto& o : offs) {
        m.write(base, std::vector<uint32_t>(64, 0));  // clear 256 B (aligned)
        run(o[0], o[1], false);
        auto w = m.read(base, 256);
        fmt::print("  src+{:<3} dst+{:<3}: {}\n", o[0], o[1], describe(w, o[1]));
    }

    fmt::print("\nREAD L2CPU (+dst_off) -> Tensix L1 landing (+src_off), 64 B; landing zone shown:\n");
    // Stage a 256 B pattern at base (aligned) for the reads.
    {
        std::vector<uint32_t> pat(64);
        for (uint32_t i = 0; i < 64; i++) {
            pat[i] = 0xA5000000u | i;
        }
        m.write(base, pat);
    }
    for (auto& o : offs) {
        m.write(rb, std::vector<uint32_t>(64, 0));
        run(o[0], o[1], true);
        auto w = m.read(rb, 256);
        // Expected: landing at +src_off holds pattern words starting at index dst_off/4.
        int first = -1;
        for (size_t i = 0; i < w.size(); i++) {
            if ((w[i] & 0xFF000000u) == 0xA5000000u) {
                first = static_cast<int>(i);
                break;
            }
        }
        std::string got = first < 0
                              ? "pattern NOT found"
                              : fmt::format(
                                    "landed at +0x{:x}, first word index {} (expected landing +0x{:x}, first index {})",
                                    first * 4,
                                    w[first] & 0xFFFF,
                                    o[0],
                                    o[1] / 4);
        fmt::print("  src(L2CPU)+{:<3} -> L1 landing+{:<3}: {}\n", o[1], o[0], got);
    }
    dev->close();
    return 0;
}
