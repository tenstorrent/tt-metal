// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// l2cpu_mux — the fabric mux, hosted on the L2CPU (x280) instead of a Tensix core
// ============================================================================
//
// N worker kernels on chip A connect to per-channel slot rings in the L2CPU tile's
// memory with the V1 mux client call sequence (build / wait ready / connect /
// fabric_async_write / disconnect / terminate). The x280 firmware (x280/fw_mux.c)
// forwards every committed slot into its own router connection and returns credits
// into each worker's L1. Packets land in a Tensix core's L1 on chip B; the host reads
// them back and checks every word.
//
// Env: FF_CHIP_A/FF_CHIP_B (0/1), L2CPU_X/L2CPU_Y (8,3), LM_NUM_CLIENTS (4, <= 8),
// LM_NUM_PACKETS (8, per client), LM_PAYLOAD (2048 B), LM_ROUNDS (2: client programs per
// mux run; later rounds reconnect and adopt the persisted cursor), LM_ATOMIC (1: each client
// also sends a header-only atomic-inc packet), FF_SKIP_BOOT (0), FF_BOOT_TOOL,
// FF_FW_BIN (l2cpu_mux/x280/build/fw_mux.bin), FF_TIMEOUT_S (20).

#include "../l2cpu_fabric_forward/l2cpu_host_utils.hpp"
#include "l2cpu_mux_layout.h"

using namespace tt::tt_metal;
using namespace l2cpu_host;

namespace {
const char* lm_status_name(uint32_t s) {
    switch (s) {
        case LM_STATUS_STARTED: return "STARTED";
        case LM_STATUS_READY_FOR_TRAFFIC: return "READY_FOR_TRAFFIC";
        case LM_STATUS_TERMINATED: return "TERMINATED";
        case 0: return "0 (not written)";
        default: return "?";
    }
}

bool wait_mux_status(X280Mem& m, uint32_t want, double timeout_s) {
    auto t0 = std::chrono::steady_clock::now();
    uint32_t s = 0;
    for (;;) {
        s = m.read(LM_STATUS, 32)[0];
        if (s == want || std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    fmt::print("  mux status: {} (0x{:08x})\n", lm_status_name(s), s);
    return s == want;
}

// Copy `size` bytes of a Tensix core (0,0) L1 region to DRAM with the receiver kernel and
// return them. A single-page interleaved L1 MeshBuffer occupies ONE bank, i.e. one core's
// L1, so EnqueueRead/WriteMeshBuffer on it reach that core, not core (0,0) where the
// fabric delivers; the buffer only reserves the address range on every core.
std::vector<uint32_t> read_l1_via_kernel(
    std::shared_ptr<distributed::MeshDevice> dev, uint32_t l1_addr, uint32_t size) {
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = size, .buffer_type = BufferType::DRAM};
    auto dram = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = size}, dram_cfg, dev.get());
    Program p = CreateProgram();
    std::vector<uint32_t> ct;
    TensorAccessorArgs(*dram->get_backing_buffer()).append_to(ct);
    KernelHandle k = CreateKernel(
        p,
        OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/receiver.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
    SetRuntimeArgs(p, k, CoreCoord{0, 0}, {l1_addr, static_cast<uint32_t>(dram->address()), size});
    distributed::MeshWorkload w;
    w.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p));
    distributed::EnqueueMeshWorkload(dev->mesh_command_queue(), w, /*blocking=*/false);
    distributed::Finish(dev->mesh_command_queue());
    std::vector<uint32_t> out;
    distributed::EnqueueReadMeshBuffer(dev->mesh_command_queue(), out, dram, /*blocking=*/true);
    return out;
}

// Fill a Tensix core (0,0) L1 region through a kernel (see read_l1_via_kernel for why
// host-side L1 buffer writes are not used).
void fill_l1_via_kernel(std::shared_ptr<distributed::MeshDevice> dev, uint32_t l1_addr, uint32_t size, uint32_t value) {
    Program p = CreateProgram();
    KernelHandle k = CreateKernel(
        p,
        OVERRIDE_KERNEL_PREFIX "l2cpu_mux/kernels/l1_fill.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    SetRuntimeArgs(p, k, CoreCoord{0, 0}, {l1_addr, size, value});
    distributed::MeshWorkload w;
    w.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p));
    distributed::EnqueueMeshWorkload(dev->mesh_command_queue(), w, /*blocking=*/false);
    distributed::Finish(dev->mesh_command_queue());
}

void print_mux_stats(X280Mem& m, uint32_t num_channels) {
    auto st = m.read(LM_STATS(0), num_channels * 16);
    for (uint32_t ch = 0; ch < num_channels; ch++) {
        const uint32_t* r = st.data() + ch * 4;
        if (r[0] == 0 && r[1] == 0 && r[2] == 0) {
            continue;
        }
        fmt::print(
            "  channel {}: forwarded={} connects={} teardowns={} last_packet_bytes={}\n", ch, r[0], r[1], r[2], r[3]);
    }
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // keep progress visible when logging to a file
    const tt::ChipId chip_a = static_cast<tt::ChipId>(env_or("FF_CHIP_A", 0));
    const tt::ChipId chip_b = static_cast<tt::ChipId>(env_or("FF_CHIP_B", 1));
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t num_clients = env_or("LM_NUM_CLIENTS", 4);
    const uint32_t num_packets = env_or("LM_NUM_PACKETS", 8);
    const uint32_t payload_size = env_or("LM_PAYLOAD", 2048);
    const uint32_t rounds = env_or("LM_ROUNDS", 2);       // client programs per mux run (tests cursor persistence)
    const uint32_t atomic_mode = env_or("LM_ATOMIC", 1);  // 1: each client also sends one header-only atomic inc
    const double timeout_s = env_or("FF_TIMEOUT_S", 20);
    const std::string boot_tool = env_str("FF_BOOT_TOOL", "./build/programming_examples/metal_example_l2cpu_x280_boot");
    const std::string fw_bin = env_str("FF_FW_BIN", "tt_metal/programming_examples/l2cpu_mux/x280/build/fw_mux.bin");
    const uint32_t seq = (static_cast<uint32_t>(time(nullptr)) << 4) | 1u;

    if (num_clients == 0 || num_clients > LM_MAX_CHANNELS || payload_size % 16 != 0 || payload_size == 0) {
        fmt::print(stderr, "LM_NUM_CLIENTS must be 1..{} and LM_PAYLOAD a nonzero multiple of 16\n", LM_MAX_CHANNELS);
        return 1;
    }

    bool pass = true;
    try {
        if (GetNumAvailableDevices() < 2) {
            fmt::print(stderr, "ABORT: need 2 chips, found {}.\n", GetNumAvailableDevices());
            return 1;
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        if (!boot_x280(boot_tool, fw_bin, chip_a)) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            return 1;
        }

        auto meshes = distributed::MeshDevice::create_unit_meshes({static_cast<int>(chip_a), static_cast<int>(chip_b)});
        auto dev_a = meshes.at(static_cast<int>(chip_a));
        auto dev_b = meshes.at(static_cast<int>(chip_b));

        const uint32_t hdr_size = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
        const uint32_t slot_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
        if (payload_size + hdr_size > slot_bytes) {
            throw std::runtime_error(
                fmt::format("payload {} + header {} exceeds the slot size {}", payload_size, hdr_size, slot_bytes));
        }
        fmt::print(
            "fabric: header {} B, slot {} B (mux stride {} B) | mux: {} channels x {} slots in L2CPU memory @0x{:08x}, "
            "{} clients x {} packets x {} B\n",
            hdr_size,
            slot_bytes,
            LM_SLOT_STRIDE(slot_bytes),
            LM_MAX_CHANNELS,
            LM_NUM_BUFFERS,
            LM_CHANNELS_BASE,
            num_clients,
            num_packets,
            payload_size);

        X280Mem ma(dev_a, l2cpu_x, l2cpu_y);
        fmt::print("x280 status before setup:\n");
        if (!wait_alive("A", ma, timeout_s)) {
            throw std::runtime_error("x280 heartbeat not advancing (FF_SKIP_BOOT set without a booted firmware?)");
        }
        // A previous run may have left the mux TERMINATED; clearing the termination word re-arms it.
        ma.write_u32(LM_TERMINATION, LM_TERM_KEEP_RUNNING);

        // Router connection for the mux itself (same setup kernel as the fabric_forward example).
        const uint32_t nonce = 0xC0FF0000u | (seq & 0xFFFFu);
        setup_connection(ma, chip_a, chip_b, hdr_size, PeerInfo{}, 0, nonce);
        fmt::print("waiting for the mux's router connection:\n");
        if (!wait_state("A", ma, FF_STATE_OPENED, timeout_s, nonce)) {
            print_diag("A", read_status(ma));
            throw std::runtime_error("x280 did not open its router connection");
        }
        print_diag("A", read_status(ma));
        fmt::print("waiting for the mux to be ready:\n");
        if (!wait_mux_status(ma, LM_STATUS_READY_FOR_TRAFFIC, timeout_s)) {
            throw std::runtime_error("mux never reached READY_FOR_TRAFFIC");
        }

        // ---- destination on chip B: one L1 region, [client][packet] strided ----
        const uint32_t region_bytes = num_clients * num_packets * payload_size;
        if (region_bytes > 1024 * 1024) {
            throw std::runtime_error(fmt::format(
                "destination region {} B exceeds the 1 MiB budget for a single core's L1; lower "
                "LM_NUM_PACKETS/LM_PAYLOAD",
                region_bytes));
        }
        distributed::DeviceLocalBufferConfig l1_cfg{.page_size = region_bytes, .buffer_type = BufferType::L1};
        auto dst_l1_b = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = region_bytes}, l1_cfg, dev_b.get());
        fill_l1_via_kernel(dev_b, static_cast<uint32_t>(dst_l1_b->address()), region_bytes, 0);
        const CoreCoord rx = dev_b->worker_core_from_logical_core({0, 0});

        // ---- client program on chip A: one core per client ----
        // Local scratch, 64 B aligned, laid out so every transfer to/from L2CPU memory keeps
        // the same offset within 64 B on both sides (see the client header):
        //   +0x00 status read landing        (LM_STATUS      is +0    mod 64)
        //   +0x40 cursor landing             (LM_CURSOR(ch)  is +0    mod 64)
        //   +0x70 flow-control word          (conn_info+0x30 is +0x30 mod 64)
        //   +0x80 teardown word              (written by the x280, any alignment)
        //   +0xC0 packet header              (slot+0         is +0    mod 64)
        //   +0x100 + (hdr_size mod 64) payload (slot+hdr_size)
        const uint32_t slot_stride = LM_SLOT_STRIDE(slot_bytes);
        const uint32_t payload_off = 0x100 + (hdr_size % 64);
        const uint32_t scratch_bytes = round_up(payload_off + payload_size, 64);
        distributed::DeviceLocalBufferConfig scratch_cfg{.page_size = scratch_bytes, .buffer_type = BufferType::L1};
        auto scratch_a = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = scratch_bytes}, scratch_cfg, dev_a.get());
        const uint32_t S = static_cast<uint32_t>(scratch_a->address());
        if (S % 64 != 0) {
            throw std::runtime_error(fmt::format("L1 scratch 0x{:x} is not 64 B aligned", S));
        }
        const uint32_t hdr_buf = S + 0xC0;
        const uint32_t payload_buf = S + payload_off;

        // Semaphore on chip B that header-only atomic-inc packets target (64 B, zeroed).
        distributed::DeviceLocalBufferConfig sem_cfg{.page_size = 64, .buffer_type = BufferType::L1};
        auto sem_l1_b =
            distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 64}, sem_cfg, dev_b.get());
        fill_l1_via_kernel(dev_b, static_cast<uint32_t>(sem_l1_b->address()), 64, 0);
        const uint32_t atomic_addr = atomic_mode ? static_cast<uint32_t>(sem_l1_b->address()) : 0u;

        const CoreCoord master_core{0, 0};
        const CoreCoord master_noc = dev_a->worker_core_from_logical_core(master_core);
        std::vector<uint32_t> ct = {LM_NUM_BUFFERS, slot_stride, LM_STATUS, LM_TERMINATION, num_clients};
        const uint32_t words = payload_size / 4;
        std::vector<uint32_t> want(region_bytes / 4);
        for (uint32_t c = 0; c < num_clients; c++) {
            for (uint32_t p = 0; p < num_packets; p++) {
                for (uint32_t w = 0; w < words; w++) {
                    want[(c * num_packets + p) * words + w] = 0xC0000000u | (c << 20) | (p << 12) | (w & 0xFFFu);
                }
            }
        }

        for (uint32_t round = 0; round < rounds; round++) {
            const bool last = (round + 1 == rounds);
            if (round > 0) {
                fill_l1_via_kernel(dev_b, static_cast<uint32_t>(dst_l1_b->address()), region_bytes, 0);
            }
            Program clients = CreateProgram();
            for (uint32_t c = 0; c < num_clients; c++) {
                const CoreCoord core{c, 0};
                KernelHandle k = CreateKernel(
                    clients,
                    OVERRIDE_KERNEL_PREFIX "l2cpu_mux/kernels/mux_sender_client.cpp",
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
                const uint32_t sem_sync = CreateSemaphore(clients, core, 0);
                SetRuntimeArgs(
                    clients,
                    k,
                    core,
                    {l2cpu_x,
                     l2cpu_y,
                     LM_CHANNEL_BASE(c, slot_bytes),
                     LM_CONN_INFO(c),
                     LM_HANDSHAKE(c),
                     LM_WRITE_COUNTER(c),
                     LM_CURSOR(c),
                     c,
                     S + 0x00,  // (unused) status scratch — the client derives its own
                     S + 0x70,  // flow-control word (+0x30 mod 64, honoured by the client)
                     S + 0x80,  // teardown word
                     S + 0x00,  // local scratch base (>= 192 B, 64 B aligned): cursor +0, status +0x40, flow +0x70
                     sem_sync,
                     hdr_buf,
                     payload_buf,
                     payload_size,
                     num_packets,
                     1u,  // hops
                     static_cast<uint32_t>(rx.x),
                     static_cast<uint32_t>(rx.y),
                     static_cast<uint32_t>(dst_l1_b->address()) + c * num_packets * payload_size,
                     c == 0 ? 1u : 0u,
                     static_cast<uint32_t>(master_noc.x),
                     static_cast<uint32_t>(master_noc.y),
                     c,
                     last ? 1u : 0u,
                     atomic_addr,
                     num_packets});
            }
            fmt::print(
                "round {}/{}: {} mux clients on chip {} -> L2CPU mux ({},{}) -> router -> chip {} Tensix ({},{}){}{}\n",
                round + 1,
                rounds,
                num_clients,
                chip_a,
                l2cpu_x,
                l2cpu_y,
                chip_b,
                rx.x,
                rx.y,
                atomic_mode ? " + 1 atomic-inc packet each" : "",
                last ? ", then terminate" : "");
            auto t0 = std::chrono::steady_clock::now();
            ma.run(clients);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fmt::print("  clients finished in {:.1f} ms wall ({} B through the mux)\n", ms, region_bytes);

            // Clients return once the mux has ACCEPTED their packets; delivery at chip B
            // completes later. Each client's last packet is the atomic inc, so the semaphore
            // reaching this round's total is the completion barrier for all data before it.
            if (atomic_mode) {
                const uint32_t want_sem = (round + 1) * num_clients * num_packets;
                auto t1 = std::chrono::steady_clock::now();
                uint32_t semv = 0;
                for (;;) {
                    semv = read_l1_via_kernel(dev_b, static_cast<uint32_t>(sem_l1_b->address()), 64)[0];
                    if (semv == want_sem) {
                        break;
                    }
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() > timeout_s) {
                        fmt::print(
                            stderr,
                            "  TIMEOUT: chip B semaphore {} (expected {} after round {})\n",
                            semv,
                            want_sem,
                            round + 1);
                        pass = false;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                fmt::print(
                    "  delivery barrier: chip B semaphore = {} (expected {}) after {:.1f} ms\n",
                    semv,
                    want_sem,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            std::vector<uint32_t> got =
                read_l1_via_kernel(dev_b, static_cast<uint32_t>(dst_l1_b->address()), region_bytes);
            pass = compare(fmt::format("round {} chip B payload", round + 1).c_str(), got, want) == 0 && pass;

            // Cursor persistence: after this round every channel's cursor must read
            // (write_counter = (round+1)*num_packets, write_index = write_counter mod slots).
            auto cur = ma.read(LM_CURSOR(0), num_clients * 64);
            for (uint32_t c = 0; c < num_clients; c++) {
                const uint32_t wc = cur[c * 16], idx = cur[c * 16 + 1];
                const uint32_t want_wc = (round + 1) * (num_packets + (atomic_mode ? 1u : 0u));
                if (wc != want_wc || idx != want_wc % LM_NUM_BUFFERS) {
                    fmt::print(
                        stderr,
                        "  channel {} persisted cursor {{wc={}, idx={}}}, expected {{{}, {}}}\n",
                        c,
                        wc,
                        idx,
                        want_wc,
                        want_wc % LM_NUM_BUFFERS);
                    pass = false;
                }
            }
            fmt::print(
                "  persisted cursors after round {}: wc={} idx={} on all {} channels{}\n",
                round + 1,
                cur[0],
                cur[1],
                num_clients,
                pass ? "" : " (MISMATCH above)");
        }

        if (atomic_mode) {
            std::vector<uint32_t> semv = read_l1_via_kernel(dev_b, static_cast<uint32_t>(sem_l1_b->address()), 64);
            const uint32_t want_sem = rounds * num_clients * num_packets;
            fmt::print(
                "  chip B semaphore after {} header-only atomic-inc packets: {} (expected {}) {}\n",
                rounds * num_clients,
                semv[0],
                want_sem,
                semv[0] == want_sem ? "OK" : "MISMATCH");
            pass = (semv[0] == want_sem) && pass;
        }

        fmt::print("mux state after the run:\n");
        wait_mux_status(ma, LM_STATUS_TERMINATED, timeout_s);
        print_mux_stats(ma, LM_MAX_CHANNELS);
        print_status("A", read_status(ma));

        for (auto& [id, dev] : meshes) {
            if (!dev->close()) {
                pass = false;
            }
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed with exception: {}\n", e.what());
        try {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        } catch (...) {
        }
        throw;
    }
    fmt::print("{}\n", pass ? "Test Passed" : "Test FAILED");
    return pass ? 0 : 1;
}
