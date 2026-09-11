// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// l2cpu_fabric_forward — L2CPU (x280) as a fabric worker, host orchestrator
// ============================================================================
//
// Two Blackhole chips linked by ethernet with FABRIC_1D routers. The x280 on chip A
// runs fw_fabric.c, opens a connection to its local fabric router like a Tensix
// worker would, and pushes payloads to chip B: either into chip B's own L2CPU
// memory (default, FF_DEST=l2cpu) or into a Tensix core's L1 (FF_DEST=tensix).
// With FF_TWO_WAY=1 (default) chip B's x280 also runs the firmware and echoes every
// message it receives straight back to chip A's inbox over its own connection —
// L2CPU-to-L2CPU traffic in both directions with no host or Tensix in the data path.
//
// The host never touches x280 memory directly: two tiny Tensix kernels
// (kernels/l2cpu_mem_{write,read}.cpp) move bytes between DRAM buffers and the L2CPU
// tile, and kernels/conn_setup.cpp resolves the router connection parameters with
// the real WorkerToFabricEdmSender code path and drops them into the x280 mailbox.
// The x280 hart is booted by the sibling tool metal_example_l2cpu_x280_boot before
// the devices are opened (hart release is one-shot per chip reset: `tt-smi -r`).
//
// Env knobs: FF_CHIP_A (0), FF_CHIP_B (1), L2CPU_X/L2CPU_Y (8,3), FF_PAYLOAD_SIZE
// (4096), FF_DEST (l2cpu|tensix), FF_TWO_WAY (1), FF_CLOSE (0: leave the connection
// open; 1: close it at the end), FF_SKIP_BOOT (0: boot the x280s; 1: reuse running
// firmware — it reconfigures itself for this run's routers), FF_PROBE_NOC0 (0),
// FF_BOOT_TOOL, FF_FW_BIN, FF_TIMEOUT_S (20).

#include "l2cpu_host_utils.hpp"

using namespace tt::tt_metal;
using namespace l2cpu_host;

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // keep progress visible when logging to a file
    const tt::ChipId chip_a = static_cast<tt::ChipId>(env_or("FF_CHIP_A", 0));
    const tt::ChipId chip_b = static_cast<tt::ChipId>(env_or("FF_CHIP_B", 1));
    const uint32_t l2cpu_x = env_or("L2CPU_X", 8);
    const uint32_t l2cpu_y = env_or("L2CPU_Y", 3);
    const uint32_t payload_size = env_or("FF_PAYLOAD_SIZE", 4096);
    const std::string dest = env_str("FF_DEST", "l2cpu");
    const bool two_way = env_or("FF_TWO_WAY", 1) != 0;
    const bool do_close = env_or("FF_CLOSE", 0) != 0;
    const double timeout_s = env_or("FF_TIMEOUT_S", 20);
    const std::string boot_tool = env_str("FF_BOOT_TOOL", "./build/programming_examples/metal_example_l2cpu_x280_boot");
    const std::string fw_bin =
        env_str("FF_FW_BIN", "tt_metal/programming_examples/l2cpu_fabric_forward/x280/build/fw_fabric.bin");
    uint32_t seq = (static_cast<uint32_t>(time(nullptr)) << 4) | 1u;

    if (payload_size == 0 || payload_size % 16 != 0 || payload_size > FF_OUTBOX_MAX ||
        payload_size > FF_INBOX_DATA_MAX) {
        fmt::print(
            stderr,
            "FF_PAYLOAD_SIZE must be a nonzero multiple of 16 and <= {}\n",
            std::min<uint32_t>(FF_OUTBOX_MAX, FF_INBOX_DATA_MAX));
        return 1;
    }

    bool pass = true;
    try {
        if (GetNumAvailableDevices() < 2) {
            fmt::print(stderr, "ABORT: need 2 chips, found {}.\n", GetNumAvailableDevices());
            return 1;
        }

        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);

        // Boot the x280 harts before tt-metal opens the devices.
        if (!boot_x280(boot_tool, fw_bin, chip_a) || (two_way && !boot_x280(boot_tool, fw_bin, chip_b))) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            return 1;
        }

        auto meshes = distributed::MeshDevice::create_unit_meshes({static_cast<int>(chip_a), static_cast<int>(chip_b)});
        auto dev_a = meshes.at(static_cast<int>(chip_a));
        auto dev_b = meshes.at(static_cast<int>(chip_b));
        if (dev_a->arch() != tt::ARCH::BLACKHOLE) {
            throw std::runtime_error("this example targets Blackhole (L2CPU tiles only exist there)");
        }

        const uint32_t hdr_size = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
        fmt::print(
            "fabric: packet header {} B, channel slot {} B, max payload/packet {} B\n",
            hdr_size,
            tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

        X280Mem ma(dev_a, l2cpu_x, l2cpu_y);
        X280Mem mb(dev_b, l2cpu_x, l2cpu_y);

        // Firmware alive?
        fmt::print("x280 status before setup:\n");
        if (!wait_alive("A", ma, timeout_s) || (two_way && !wait_alive("B", mb, timeout_s))) {
            throw std::runtime_error("x280 heartbeat not advancing — is the firmware booted? (FF_SKIP_BOOT set?)");
        }

        // Connection parameters -> x280 mailboxes. The nonce makes a running firmware
        // reconfigure (re-probe + re-open) for this run's routers.
        const uint32_t nonce = 0xC0FF0000u | (seq & 0xFFFFu);
        PeerInfo peer_b{l2cpu_x, l2cpu_y, FF_MBOX_INBOX, 1};
        PeerInfo peer_a{l2cpu_x, l2cpu_y, FF_MBOX_INBOX, 1};
        setup_connection(ma, chip_a, chip_b, hdr_size, peer_b, FF_CFLAG_AUTO_ECHO, nonce);
        if (two_way) {
            setup_connection(mb, chip_b, chip_a, hdr_size, peer_a, FF_CFLAG_AUTO_ECHO, nonce);
        }

        // Wait for the firmware(s) to open their router connections for this nonce.
        fmt::print("waiting for x280 fabric connections to open:\n");
        bool opened = wait_state("A", ma, FF_STATE_OPENED, timeout_s, nonce);
        print_diag("A", read_status(ma));
        if (two_way) {
            opened = wait_state("B", mb, FF_STATE_OPENED, timeout_s, nonce) && opened;
            print_diag("B", read_status(mb));
        }
        if (!opened) {
            throw std::runtime_error("x280 firmware did not reach OPENED");
        }

        // Payload -> chip A outbox.
        const uint32_t num_words = payload_size / 4;
        std::vector<uint32_t> payload(num_words);
        for (uint32_t i = 0; i < num_words; i++) {
            payload[i] = 0xF00D0000u + i;
        }
        ma.write(FF_MBOX_OUTBOX, payload);

        // Destination on chip B.
        uint32_t dx, dy, daddr, flag_addr;
        std::shared_ptr<distributed::MeshBuffer> dst_l1_b;
        if (dest == "tensix") {
            distributed::DeviceLocalBufferConfig l1_cfg{.page_size = payload_size, .buffer_type = BufferType::L1};
            dst_l1_b = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = payload_size}, l1_cfg, dev_b.get());
            const CoreCoord rx = dev_b->worker_core_from_logical_core({0, 0});
            dx = rx.x;
            dy = rx.y;
            daddr = static_cast<uint32_t>(dst_l1_b->address());
            flag_addr = 0;
            // Clear the destination so stale data cannot pass.
            std::vector<uint32_t> zeros(num_words, 0);
            distributed::EnqueueWriteMeshBuffer(dev_b->mesh_command_queue(), dst_l1_b, zeros, /*blocking=*/true);
        } else {
            dx = l2cpu_x;
            dy = l2cpu_y;
            daddr = FF_MBOX_INBOX_DATA;
            flag_addr = FF_MBOX_INBOX;
            std::vector<uint32_t> zeros(16 + num_words, 0);
            mb.write(FF_MBOX_INBOX, zeros);
            if (two_way) {
                ma.write(FF_MBOX_INBOX, zeros);
            }
        }
        fmt::print(
            "send: chip {} x280 -> fabric (1 hop) -> chip {} {} ({},{}) addr 0x{:08x}, {} bytes, seq 0x{:x}\n",
            chip_a,
            chip_b,
            dest == "tensix" ? "Tensix L1" : "L2CPU inbox",
            dx,
            dy,
            daddr,
            payload_size,
            seq);

        // Fire.
        if (!post_request(
                "A", ma, seq, FF_MODE_SEND, FF_MBOX_OUTBOX, payload_size, dx, dy, daddr, 1, flag_addr, timeout_s)) {
            pass = false;
        }

        // Verify on chip B.
        if (dest == "tensix") {
            // Move the delivered L1 bytes to DRAM with the receiver kernel and compare.
            distributed::DeviceLocalBufferConfig dram_cfg{.page_size = payload_size, .buffer_type = BufferType::DRAM};
            auto dst_dram_b = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = payload_size}, dram_cfg, dev_b.get());
            Program p = CreateProgram();
            std::vector<uint32_t> ct;
            TensorAccessorArgs(*dst_dram_b->get_backing_buffer()).append_to(ct);
            KernelHandle k = CreateKernel(
                p,
                OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/receiver.cpp",
                CoreCoord{0, 0},
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
            SetRuntimeArgs(p, k, CoreCoord{0, 0}, {daddr, static_cast<uint32_t>(dst_dram_b->address()), payload_size});
            mb.run(p);
            std::vector<uint32_t> got;
            distributed::EnqueueReadMeshBuffer(dev_b->mesh_command_queue(), got, dst_dram_b, /*blocking=*/true);
            pass = compare("chip B Tensix L1 payload", got, payload) == 0 && pass;
        } else {
            auto inbox = mb.read(FF_MBOX_INBOX, 0x40 + payload_size);
            fmt::print(
                "  chip B inbox header: len={} seq=0x{:x} tag=0x{:x} (expect len={} seq=0x{:x} tag=0x{:x})\n",
                inbox[0],
                inbox[1],
                inbox[2],
                payload_size,
                seq,
                FF_TAG_ORIGINAL);
            std::vector<uint32_t> data(inbox.begin() + 16, inbox.begin() + 16 + num_words);
            pass = compare("chip B L2CPU inbox payload", data, payload) == 0 && pass;
            pass = (inbox[0] == payload_size && inbox[1] == seq && inbox[2] == FF_TAG_ORIGINAL) && pass;

            if (two_way) {
                // Chip B's x280 echoes the message back into chip A's inbox.
                auto t0 = std::chrono::steady_clock::now();
                std::vector<uint32_t> ainbox;
                for (;;) {
                    ainbox = ma.read(FF_MBOX_INBOX, 0x40 + payload_size);
                    if (ainbox[1] == seq) {
                        break;
                    }
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
                        fmt::print(stderr, "  TIMEOUT waiting for the echo in chip A's inbox\n");
                        print_status("B", read_status(mb));
                        print_diag("B", read_status(mb));
                        pass = false;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                fmt::print(
                    "  chip A inbox header (echo from chip B's x280): len={} seq=0x{:x} tag=0x{:x} (expect "
                    "tag=0x{:x})\n",
                    ainbox[0],
                    ainbox[1],
                    ainbox[2],
                    FF_TAG_ECHO);
                std::vector<uint32_t> adata(ainbox.begin() + 16, ainbox.begin() + 16 + num_words);
                pass = compare("chip A L2CPU inbox echo payload", adata, payload) == 0 && pass;
                pass = (ainbox[0] == payload_size && ainbox[1] == seq && ainbox[2] == FF_TAG_ECHO) && pass;
                print_diag("B", read_status(mb));
                fmt::print(
                    "  round trip A->B->A of {} bytes completed in {:.1f} ms wall (host-polled)\n",
                    payload_size,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }

        if (do_close) {
            fmt::print("closing x280 fabric connections:\n");
            seq += 2;
            pass = post_request("A", ma, seq, FF_MODE_CLOSE, 0, 0, 0, 0, 0, 1, 0, timeout_s) && pass;
            fmt::print(
                "  [A] router teardown ack (noc_semaphore_inc into the L2CPU) landed: {}\n",
                read_status(ma).diag[FF_DIAG_TEARDOWN_ACK / 4]);
            if (two_way) {
                pass = post_request("B", mb, seq, FF_MODE_CLOSE, 0, 0, 0, 0, 0, 1, 0, timeout_s) && pass;
                fmt::print(
                    "  [B] router teardown ack (noc_semaphore_inc into the L2CPU) landed: {}\n",
                    read_status(mb).diag[FF_DIAG_TEARDOWN_ACK / 4]);
            }
        }

        fmt::print("final x280 status:\n");
        print_status("A", read_status(ma));
        if (two_way) {
            print_status("B", read_status(mb));
        }

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
