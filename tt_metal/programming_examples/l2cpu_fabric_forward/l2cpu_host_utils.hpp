// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side helpers shared by the L2CPU fabric examples (l2cpu_fabric_forward,
// l2cpu_mux): x280 boot, L2CPU memory access through Tensix kernels, firmware status
// decoding, and router-connection setup via the conn_setup kernel.
#pragma once

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "x280/fabric_mbox.h"  // relative to this header

using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace l2cpu_host {

uint32_t env_or(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : fallback;
}
const char* env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}
uint32_t round_up(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }

// Blackhole eth channel -> NOC0 physical x (row y = 1), from blackhole_140_arch.yaml.
constexpr uint32_t kBhEthNoc0X[14] = {1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10};

const char* state_name(uint64_t s) {
    switch (s) {
        case FF_STATE_BOOT: return "BOOT";
        case FF_STATE_ALIVE: return "ALIVE";
        case FF_STATE_PARAMS_READY: return "PARAMS_READY";
        case FF_STATE_PROBED: return "PROBED";
        case FF_STATE_OPENED: return "OPENED";
        case FF_STATE_SENT: return "SENT";
        case FF_STATE_CLOSED: return "CLOSED";
        default: return "?";
    }
}
const char* fault_name(uint64_t f) {
    switch (f) {
        case FF_FAULT_NONE: return "none";
        case FF_FAULT_OPEN_TIMEOUT: return "OPEN_TIMEOUT";
        case FF_FAULT_SLOT_TIMEOUT: return "SLOT_TIMEOUT";
        case FF_FAULT_BAD_PARAMS: return "BAD_PARAMS";
        case FF_FAULT_CLOSE_TIMEOUT: return "CLOSE_TIMEOUT";
        case FF_FAULT_PROBE_FAILED: return "PROBE_FAILED";
        default: return "?";
    }
}
const char* rstatus_name(uint32_t s) {
    switch (s) {
        case FF_RSTATUS_OK: return "OK";
        case FF_RSTATUS_SLOT_TIMEOUT: return "SLOT_TIMEOUT";
        case FF_RSTATUS_BAD_REQ: return "BAD_REQ";
        case FF_RSTATUS_CLOSED: return "CLOSED";
        case FF_RSTATUS_CLOSE_TIMEOUT: return "CLOSE_TIMEOUT";
        case FF_RSTATUS_NOT_OPEN: return "NOT_OPEN";
        default: return "?";
    }
}
const char* probe_name(uint32_t p) {
    switch (p) {
        case FF_PROBE_NONE: return "not run";
        case FF_PROBE_TRANSLATED: return "translated coords match";
        case FF_PROBE_NOC0: return "NOC0 physical coords match";
        case FF_PROBE_NEITHER: return "NEITHER matched";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// X280Mem: read/write L2CPU tile memory through a Tensix core on the same chip.
// ---------------------------------------------------------------------------
class X280Mem {
public:
    X280Mem(std::shared_ptr<distributed::MeshDevice> dev, uint32_t x, uint32_t y) :
        dev_(std::move(dev)), x_(x), y_(y) {}

    std::vector<uint32_t> read(uint32_t addr, uint32_t size) {
        size = round_up(size, 32);
        auto [dram, l1] = buffers(size);
        Program p = CreateProgram();
        std::vector<uint32_t> ct;
        TensorAccessorArgs(*dram->get_backing_buffer()).append_to(ct);
        KernelHandle k = CreateKernel(
            p,
            OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/l2cpu_mem_read.cpp",
            core_,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
        SetRuntimeArgs(
            p,
            k,
            core_,
            {static_cast<uint32_t>(l1->address()), static_cast<uint32_t>(dram->address()), size, x_, y_, addr});
        run(p);
        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(dev_->mesh_command_queue(), out, dram, /*blocking=*/true);
        return out;
    }

    // Write `data` at addr (padded to 32 B), then optionally one word to flag_addr.
    void write(uint32_t addr, std::vector<uint32_t> data, uint32_t flag_addr = 0, uint32_t flag_val = 0) {
        const uint32_t size = data.empty() ? 0 : round_up(static_cast<uint32_t>(data.size() * 4), 32);
        data.resize(std::max<size_t>(size / 4, 8), 0);
        auto [dram, l1] = buffers(std::max<uint32_t>(size, 32));
        distributed::EnqueueWriteMeshBuffer(dev_->mesh_command_queue(), dram, data, /*blocking=*/false);
        Program p = CreateProgram();
        std::vector<uint32_t> ct;
        TensorAccessorArgs(*dram->get_backing_buffer()).append_to(ct);
        KernelHandle k = CreateKernel(
            p,
            OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/l2cpu_mem_write.cpp",
            core_,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
        SetRuntimeArgs(
            p,
            k,
            core_,
            {static_cast<uint32_t>(l1->address()),
             static_cast<uint32_t>(dram->address()),
             size,
             x_,
             y_,
             addr,
             flag_addr,
             flag_val});
        run(p);
    }

    void write_u32(uint32_t addr, uint32_t v) { write(addr, {}, addr, v); }

    distributed::MeshDevice& dev() { return *dev_; }
    uint32_t x() const { return x_; }
    uint32_t y() const { return y_; }

    void run(Program& p) {
        distributed::MeshWorkload w;
        w.add_program(distributed::MeshCoordinateRange(dev_->shape()), std::move(p));
        distributed::EnqueueMeshWorkload(dev_->mesh_command_queue(), w, /*blocking=*/false);
        distributed::Finish(dev_->mesh_command_queue());
    }

    std::pair<std::shared_ptr<distributed::MeshBuffer>, std::shared_ptr<distributed::MeshBuffer>> buffers(
        uint32_t size) {
        distributed::DeviceLocalBufferConfig dram_cfg{.page_size = size, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig l1_cfg{.page_size = size, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig cfg{.size = size};
        return {
            distributed::MeshBuffer::create(cfg, dram_cfg, dev_.get()),
            distributed::MeshBuffer::create(cfg, l1_cfg, dev_.get())};
    }

private:
    std::shared_ptr<distributed::MeshDevice> dev_;
    uint32_t x_, y_;
    CoreCoord core_{0, 0};
};

// Snapshot of the firmware's status + diag words.
struct FwStatus {
    uint64_t heartbeat = 0, state = 0, fault = 0, traps = 0, mcause = 0, boot_marker = 0;
    std::vector<uint32_t> diag;  // FF_MBOX_DIAG block
    std::vector<uint32_t> resp;  // FF_MBOX_RESP block
};

FwStatus read_status(X280Mem& m) {
    auto w = m.read(FF_MBOX, 0x200);
    FwStatus s;
    auto u64 = [&](uint32_t off) {
        return static_cast<uint64_t>(w[off / 4]) | (static_cast<uint64_t>(w[off / 4 + 1]) << 32);
    };
    s.heartbeat = u64(0x00);
    s.state = u64(0x08);
    s.traps = u64(0x18);
    s.mcause = u64(0x20);
    s.fault = u64(0x28);
    s.boot_marker = u64(0x30);
    s.diag.assign(w.begin() + 0x180 / 4, w.begin() + 0x1c0 / 4);
    s.resp.assign(w.begin() + 0xc0 / 4, w.begin() + 0x100 / 4);
    return s;
}

void print_status(const char* who, const FwStatus& s) {
    fmt::print(
        "  [{}] heartbeat={} state={} (0x{:x}) fault={} traps={} mcause=0x{:x} boot_marker=0x{:x}\n",
        who,
        s.heartbeat,
        state_name(s.state),
        s.state,
        fault_name(s.fault),
        s.traps,
        s.mcause,
        s.boot_marker);
}

void print_diag(const char* who, const FwStatus& s) {
    const auto& d = s.diag;
    fmt::print(
        "  [{}] coord probe: {} (read via translated=0x{:08x}, via NOC0=0x{:08x}); window aimed at ({},{})\n",
        who,
        probe_name(d[FF_DIAG_PROBE_RESULT / 4]),
        d[FF_DIAG_PROBE_TRANS / 4],
        d[FF_DIAG_PROBE_NOC0 / 4],
        d[FF_DIAG_WINDOW_X / 4],
        d[FF_DIAG_WINDOW_Y / 4]);
    fmt::print(
        "  [{}] open: cursor write_counter={} write_index={} edm_read_counter={} | router free-slots reg at "
        "open=0x{:08x} "
        "| handshake readback={} | free slots now={} | inbox seen={} echoes={}\n",
        who,
        d[FF_DIAG_OPEN_CTR / 4],
        d[FF_DIAG_OPEN_IDX / 4],
        d[FF_DIAG_OPEN_RDCTR / 4],
        d[FF_DIAG_SREG_OPEN / 4],
        d[FF_DIAG_HANDSHAKE_RB / 4],
        d[FF_DIAG_FREE_NOW / 4],
        d[FF_DIAG_INBOX_SEEN / 4],
        d[FF_DIAG_ECHOES / 4]);
}

// Wait for the firmware to reach `want`; if `gen` is nonzero also require that the
// connection was configured from that FF_CONN_VALID value (FF_DIAG_CONFIG_GEN).
bool wait_state(const char* who, X280Mem& m, uint64_t want, double timeout_s, uint32_t gen = 0) {
    auto t0 = std::chrono::steady_clock::now();
    FwStatus s;
    for (;;) {
        s = read_status(m);
        if ((s.state == want && (gen == 0 || s.diag[FF_DIAG_CONFIG_GEN / 4] == gen)) ||
            s.fault == FF_FAULT_BAD_PARAMS) {
            break;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    print_status(who, s);
    return s.state == want && (gen == 0 || s.diag[FF_DIAG_CONFIG_GEN / 4] == gen);
}

// Is the firmware running at all? (heartbeat advances)
bool wait_alive(const char* who, X280Mem& m, double timeout_s) {
    auto t0 = std::chrono::steady_clock::now();
    const uint64_t first = read_status(m).heartbeat;
    FwStatus s;
    for (;;) {
        s = read_status(m);
        if (s.heartbeat != first) {
            break;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    print_status(who, s);
    return s.heartbeat != first;
}

// ---------------------------------------------------------------------------
// Boot the x280 on one chip with the sibling boot tool (separate process, raw UMD),
// before tt-metal opens the devices.
// ---------------------------------------------------------------------------
bool boot_x280(const std::string& tool, const std::string& fw, uint32_t chip) {
    if (env_or("FF_SKIP_BOOT", 0)) {
        fmt::print("FF_SKIP_BOOT=1: assuming the x280 on chip {} already runs {}\n", chip, fw);
        return true;
    }
    const std::string cmd = fmt::format("{} --chip {} boot {}", tool, chip, fw);
    fmt::print("Booting x280 on chip {}: {}\n", chip, cmd);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        fmt::print(stderr, "ABORT: boot tool returned {} for chip {}. Recover with `tt-smi -r`.\n", rc, chip);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Resolve the router connection for chip -> peer and deliver it to that chip's x280.
// ---------------------------------------------------------------------------
struct PeerInfo {
    uint32_t noc_x = 0xFFFFFFFFu, noc_y = 0xFFFFFFFFu, inbox = 0, hops = 1;
};

std::vector<uint32_t> setup_connection(
    X280Mem& m,
    tt::ChipId chip,
    tt::ChipId peer,
    uint32_t hdr_size,
    const PeerInfo& pi,
    uint32_t flags,
    uint32_t nonce) {
    const auto node = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(chip);
    const auto peer_node = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(peer);
    const auto links = tt::tt_fabric::get_forwarding_link_indices(node, peer_node);
    if (links.empty()) {
        throw std::runtime_error(fmt::format(
            "no fabric forwarding link from chip {} (mesh {} chip {}) to chip {} (mesh {} chip {})",
            chip,
            *node.mesh_id,
            node.chip_id,
            peer,
            *peer_node.mesh_id,
            peer_node.chip_id));
    }
    fmt::print(
        "chip {} -> chip {}: fabric nodes (mesh {}, chip {}) -> (mesh {}, chip {}), link indices [{}], using {}\n",
        chip,
        peer,
        *node.mesh_id,
        node.chip_id,
        *peer_node.mesh_id,
        peer_node.chip_id,
        fmt::join(links, ","),
        links.front());

    auto [dram, l1] = m.buffers(128);
    Program p = CreateProgram();
    const CoreCoord core{0, 0};

    // Fabric connection args for a Tensix worker on `core`: [eth_channel, teardown sem, buffer-index sem].
    std::vector<uint32_t> fargs;
    tt::tt_fabric::append_fabric_connection_rt_args(
        node, peer_node, links.front(), p, core, fargs, tt::CoreType::WORKER);
    const uint32_t eth_channel = fargs.at(0);
    // NOC0-physical fallback for the window probe is opt-in (FF_PROBE_NOC0=1): the
    // L2CPU port translates coordinates (measured), so a physical eth coordinate is an
    // unmapped translated coordinate and reading it could stall the hart.
    const bool probe_noc0 = env_or("FF_PROBE_NOC0", 0) != 0;
    const uint32_t noc0_x = (probe_noc0 && eth_channel < 14) ? kBhEthNoc0X[eth_channel] : 0xFFFFFFFFu;
    const uint32_t noc0_y = (probe_noc0 && eth_channel < 14) ? 1u : 0xFFFFFFFFu;
    const uint32_t magic = 0x5EED0000u | (static_cast<uint32_t>(chip) << 8) | eth_channel;

    std::vector<uint32_t> args = {
        static_cast<uint32_t>(l1->address()),
        static_cast<uint32_t>(dram->address()),
        128,
        m.x(),
        m.y(),
        FF_MBOX_CONN,
        m.x(),  // self coords the router will push credits to (L2CPU translated == NOC0)
        m.y(),
        hdr_size,
        noc0_x,
        noc0_y,
        FF_MBOX_FREESLOTS_SINK,
        FF_MBOX_TEARDOWN_WORD,
        magic,
        pi.noc_x,
        pi.noc_y,
        pi.inbox,
        pi.hops,
        flags,
        0,  // reserved
        nonce};
    args.insert(args.end(), fargs.begin(), fargs.end());

    std::vector<uint32_t> ct;
    TensorAccessorArgs(*dram->get_backing_buffer()).append_to(ct);
    KernelHandle k = CreateKernel(
        p,
        OVERRIDE_KERNEL_PREFIX "l2cpu_fabric_forward/kernels/conn_setup.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = ct});
    SetRuntimeArgs(p, k, core, args);
    m.run(p);

    std::vector<uint32_t> blk;
    distributed::EnqueueReadMeshBuffer(m.dev().mesh_command_queue(), blk, dram, /*blocking=*/true);
    fmt::print(
        "  conn block delivered to x280 (chip {}): eth channel {} | EDM ({},{}) translated, NOC0 candidate ({},{}) | "
        "buffer_base=0x{:x} slots={} slot_bytes={} | handshake=0x{:x} loc_info=0x{:x} cursor=0x{:x} | "
        "stream id {} write=0x{:08x} read=0x{:08x} | hdr={} | valid=0x{:08x}\n",
        chip,
        eth_channel,
        blk[0x00 / 4],
        blk[0x04 / 4],
        blk[0x08 / 4],
        blk[0x0c / 4],
        blk[0x10 / 4],
        blk[0x14 / 4],
        blk[0x18 / 4],
        blk[0x1c / 4],
        blk[0x20 / 4],
        blk[0x24 / 4],
        blk[0x64 / 4],
        blk[0x28 / 4],
        blk[0x2c / 4],
        blk[0x38 / 4],
        blk[0x7c / 4]);
    return blk;
}

// Post a request and wait for the response.
bool post_request(
    const char* who,
    X280Mem& m,
    uint32_t seq,
    uint32_t mode,
    uint32_t src,
    uint32_t size,
    uint32_t dx,
    uint32_t dy,
    uint32_t daddr,
    uint32_t hops,
    uint32_t flag_addr,
    double timeout_s) {
    std::vector<uint32_t> req(16, 0);
    req[FF_REQ_SEQ / 4] = 0;
    req[FF_REQ_SRC_ADDR / 4] = src;
    req[FF_REQ_SIZE / 4] = size;
    req[FF_REQ_DST_NOC_X / 4] = dx;
    req[FF_REQ_DST_NOC_Y / 4] = dy;
    req[FF_REQ_DST_ADDR / 4] = daddr;
    req[FF_REQ_NUM_HOPS / 4] = hops;
    req[FF_REQ_FLAG_ADDR / 4] = flag_addr;
    req[FF_REQ_MODE / 4] = mode;
    m.write(FF_MBOX_REQ, req, FF_MBOX_REQ + FF_REQ_SEQ, seq);

    auto t0 = std::chrono::steady_clock::now();
    FwStatus s;
    for (;;) {
        s = read_status(m);
        if (s.resp[FF_RESP_SEQ / 4] == seq) {
            break;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            fmt::print(stderr, "  [{}] TIMEOUT waiting for response to seq 0x{:x}\n", who, seq);
            print_status(who, s);
            print_diag(who, s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto& r = s.resp;
    const uint64_t cyc =
        static_cast<uint64_t>(r[FF_RESP_CYCLES_LO / 4]) | (static_cast<uint64_t>(r[FF_RESP_CYCLES_HI / 4]) << 32);
    fmt::print(
        "  [{}] response seq 0x{:x}: {} | packets={} | free slots before={} after={} | router free-slots reg "
        "before=0x{:08x} after=0x{:08x} | {} cycles ({:.1f} us @200MHz)\n",
        who,
        seq,
        rstatus_name(r[FF_RESP_STATUS / 4]),
        r[FF_RESP_PACKETS / 4],
        r[FF_RESP_FREE_BEFORE / 4],
        r[FF_RESP_FREE_AFTER / 4],
        r[FF_RESP_SREG_BEFORE / 4],
        r[FF_RESP_SREG_AFTER / 4],
        cyc,
        cyc / 200.0);
    print_status(who, s);
    const uint32_t st = r[FF_RESP_STATUS / 4];
    return st == FF_RSTATUS_OK || st == FF_RSTATUS_CLOSED;
}

uint32_t compare(const char* what, const std::vector<uint32_t>& got, const std::vector<uint32_t>& want) {
    uint32_t mism = 0;
    for (size_t i = 0; i < want.size(); i++) {
        const uint32_t g = i < got.size() ? got[i] : 0xDEADDEADu;
        if (g != want[i]) {
            if (mism < 6) {
                fmt::print(stderr, "    {} word {:5d}: expected 0x{:08x}, got 0x{:08x}\n", what, i, want[i], g);
            }
            mism++;
        }
    }
    fmt::print("  {}: {} words, {}\n", what, want.size(), mism == 0 ? "all match" : fmt::format("{} MISMATCHES", mism));
    return mism;
}

}  // namespace l2cpu_host
