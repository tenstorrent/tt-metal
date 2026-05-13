// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "llrt/external_iface_sender.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <span>
#include <thread>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/tlb.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace tt::llrt {

// ---------------------------------------------------------------------------
// PostSendProfile — record / dump
// ---------------------------------------------------------------------------
void PostSendProfile::OpStat::record(uint64_t ns) {
    count++;
    total_ns += ns;
    if (ns < min_ns) {
        min_ns = ns;
    }
    if (ns > max_ns) {
        max_ns = ns;
    }
    static const uint64_t edges[8] = {100, 200, 500, 1000, 2000, 5000, 10000, 20000};
    int b = 0;
    for (; b < 8; ++b) {
        if (ns < edges[b]) {
            break;
        }
    }
    buckets[b]++;
}

namespace {
void dump_op(std::ostream& os, const char* name, const PostSendProfile::OpStat& s) {
    if (s.count == 0) {
        os << "  " << name << ": (no samples)\n";
        return;
    }
    os << "  " << name << ": n=" << s.count << " mean=" << s.mean_ns() << "ns"
       << " min=" << s.min_ns << "ns"
       << " max=" << s.max_ns << "ns"
       << " hist[<100,<200,<500,<1µ,<2µ,<5µ,<10µ,<20µ,≥20µ]={";
    for (int i = 0; i < 9; ++i) {
        os << s.buckets[i];
        if (i < 8) {
            os << ',';
        }
    }
    os << "}\n";
}
}  // namespace

void PostSendProfile::dump(std::ostream& os) const {
    os << "=== PostSendProfile ===\n";
    dump_op(os, "consumer_read    ", consumer_read);
    dump_op(os, "payload_write    ", payload_write);
    dump_op(os, "desc_pre_owned   ", desc_pre_owned);
    dump_op(os, "payload_drain    ", payload_drain);
    dump_op(os, "owned_publish    ", owned_publish);
    dump_op(os, "producer_write   ", producer_write);
    dump_op(os, "producer_drain   ", producer_drain);
    dump_op(os, "total_post       ", total_post);
}

// ---------------------------------------------------------------------------
// ProfileGuard — RAII scope timer. Records elapsed ns into op stat.
// Compiled out under !TT_METAL_CMAC_PROFILE so the production hot path stays
// clean (no std::chrono::now() calls, no stat updates).
// ---------------------------------------------------------------------------
#ifdef TT_METAL_CMAC_PROFILE
namespace {
class ProfileGuard {
public:
    explicit ProfileGuard(PostSendProfile::OpStat& s) : start_(std::chrono::steady_clock::now()), stat_(s) {}
    ~ProfileGuard() {
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_).count();
        stat_.record(static_cast<uint64_t>(ns));
    }

private:
    std::chrono::steady_clock::time_point start_;
    PostSendProfile::OpStat& stat_;
};
}  // namespace
#define PROFILE_OP(member) ProfileGuard _pg_##member(profile_.member)
#else
#define PROFILE_OP(member) ((void)0)
#endif

// ---------------------------------------------------------------------------
// Helper: return a const reference to the singleton Cluster.
// ---------------------------------------------------------------------------
static const ::tt::Cluster& cluster() { return ::tt::tt_metal::MetalContext::instance().get_cluster(); }

// ---------------------------------------------------------------------------
// Helper: read a single uint32_t from erisc L1.
// ---------------------------------------------------------------------------
static uint32_t read_word(ChipId chip_id, CoreCoord core, uint32_t addr) {
    auto vec = cluster().read_core(chip_id, core, addr, sizeof(uint32_t));
    return vec[0];
}

// ---------------------------------------------------------------------------
// Helper: write a single uint32_t to erisc L1 without write-combining
//         (important for doorbell semantics).
// ---------------------------------------------------------------------------
static void write_word_immediate(ChipId chip_id, CoreCoord core, uint32_t addr, uint32_t value) {
    std::vector<uint32_t> buf{value};
    cluster().write_core_immediate(chip_id, core, buf, addr);
}

// ---------------------------------------------------------------------------
// Helper: write a uint32_t via the WC TLB block path. Same semantics as
// write_core(span, addr) — but specialised for 4 bytes to avoid the vector
// allocation cost of write_word_immediate.
//
// Why both helpers exist: write_word_immediate goes through write_to_device_reg
// which always reconfigures the UC TLB (~380 ns/call). When a static WC TLB
// has been registered for this core via configure_tlb(), this block path
// hits write_block(offset, src, size) with no reconfigure — ~50-100 ns/call.
// ---------------------------------------------------------------------------
static void write_word_block(ChipId chip_id, CoreCoord core, uint32_t addr, uint32_t value) {
    cluster().write_core(&value, sizeof(value), tt_cxy_pair(chip_id, core), addr);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ExternalIfaceSender::ExternalIfaceSender(ChipId chip_id, CoreCoord virtual_eth_core) :
    chip_id_(chip_id), virtual_core_(virtual_eth_core) {}

// ---------------------------------------------------------------------------
// is_link_up
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::is_link_up() const {
    // Heartbeat must be non-zero — firmware actually running.
    // train_status==0 alone is ambiguous (also the post-memset uninitialised value).
    if (read_word(chip_id_, virtual_core_, kHeartbeatAddr) == 0u) {
        return false;
    }
    return read_word(chip_id_, virtual_core_, kTrainStatusAddr) == 0u;
}

// ---------------------------------------------------------------------------
// wait_for_link
//
// If wqe_ring_enabled_ was set by enable_wqe_ring() before this call, also
// poll fw_status (RCB+0x0C) for == 1 after writing the magic, to verify FW
// actually entered run_wqe_ring_loop. If FW falls back to legacy (e.g. mode
// bit didn't latch correctly), fw_status stays at 0 and we return false —
// the caller learns about the dispatch failure here, not silently at the
// first post_send().
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::wait_for_link(uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (!is_link_up()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // PCS lock achieved — switch firmware to data-path mode.
    write_word_immediate(chip_id_, virtual_core_, kModeAddr, kModeMagic);

    // If the caller enabled WQE-ring mode, verify the FW actually entered
    // run_wqe_ring_loop. fw_status is written to 1 on entry. Wait up to the
    // remaining timeout (or 100 ms, whichever is more) for it to become 1.
    if (wqe_ring_enabled_) {
        const auto verify_deadline =
            std::max(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
        while (read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbFwStatusOff) != 1u) {
            if (std::chrono::steady_clock::now() >= verify_deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::send(std::span<const uint8_t> buf) {
    if (buf.size() > kMaxFrameBytes) {
        return false;
    }

    // Check that the previous TX has been consumed (doorbell cleared).
    if (read_word(chip_id_, virtual_core_, kTxSizeAddr) != 0u) {
        return false;
    }

    // Select the ping-pong TX buffer.
    const uint32_t tx_addr = (tx_buf_sel_ == 0) ? kTxBuf0 : kTxBuf1;

    // Write frame bytes directly into erisc L1 (data path — no erisc copy).
    cluster().write_core(chip_id_, virtual_core_, std::span<const uint8_t>(buf), tx_addr);

    // Write buffer selector.
    write_word_immediate(chip_id_, virtual_core_, kTxBufSelAddr, static_cast<uint32_t>(tx_buf_sel_));

    // Write frame size — this is the doorbell that arms the TX DMA.
    write_word_immediate(chip_id_, virtual_core_, kTxSizeAddr, static_cast<uint32_t>(buf.size()));

    // Advance ping-pong index for next call.
    tx_buf_sel_ ^= 1;
    return true;
}

// ---------------------------------------------------------------------------
// wait_tx_consumed
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::wait_tx_consumed(uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (read_word(chip_id_, virtual_core_, kTxSizeAddr) != 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
}

// ---------------------------------------------------------------------------
// rx_watermark
// ---------------------------------------------------------------------------
uint32_t ExternalIfaceSender::rx_watermark() const { return read_word(chip_id_, virtual_core_, kRxWpAddr); }

// ---------------------------------------------------------------------------
// read_rx
// ---------------------------------------------------------------------------
void ExternalIfaceSender::read_rx(uint32_t buf_sel, uint32_t word_offset, std::span<uint8_t> out) const {
    const uint32_t base = (buf_sel == 0) ? kPacketBuf : kPacketBuf1;
    const uint32_t addr = base + word_offset * 4u;
    cluster().read_core(out.data(), static_cast<uint32_t>(out.size()), tt_cxy_pair(chip_id_, virtual_core_), addr);
}

// ---------------------------------------------------------------------------
// rx_consume
// ---------------------------------------------------------------------------
void ExternalIfaceSender::rx_consume(uint32_t consumed_wp) {
    write_word_immediate(chip_id_, virtual_core_, kRxRpAddr, consumed_wp);
}

// ===========================================================================
// WQE-ring (v1) implementation
// ===========================================================================
//
// Wire model:
//   - Host owns the producer cursor and the descriptor table.
//   - FW owns the consumer cursor, fw_status, and cq_head.
//   - Indices are 32-bit free-running; slot = idx & (kWqeRingN - 1).
//
// post_send write order (matches firmware reader):
//   1. Payload bytes → kWqePayloadBase + slot * kWqePayloadStride
//   2. Descriptor fields IN ORDER: size, seq, payload_off, cookie,
//      then flags-with-OWNED-bit LAST. FW polls flags first, so this
//      guarantees a fully-formed descriptor when OWNED_BY_FW is observed.
//   3. Bump producer_idx in RCB. No read-back fence: subsequent
//      poll_completion / wait_completion read_word naturally drains
//      the PCIe posted-write queue, and PCIe TLP ordering keeps the
//      payload→desc→producer_idx writes in order from FW's view.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// enable_wqe_ring
//
// MUST be called BEFORE wait_for_link(). FW reads the mode bit only at
// gw-mode entry (when wait_for_link writes GW_MODE_MAGIC). If wait_for_link
// has already run, FW is in the legacy loop and this call has no effect on
// FW dispatch; subsequent post_send() will silently no-op on the wire.
//
// Defensive check: if heartbeat is non-zero AND mode mailbox at 0x1F50 is
// already set to kModeMagic, FW has already dispatched. We can't recover —
// return false. Caller must reset and start over.
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::enable_wqe_ring() {
    // Detect "FW already dispatched" by checking if kModeAddr already holds
    // GW_MODE_MAGIC. wait_for_link() writes that; before then the value is 0.
    if (read_word(chip_id_, virtual_core_, kModeAddr) == kModeMagic) {
        // Too late: FW already entered legacy run_gateway_loop on the previous
        // wait_for_link() call. enable_wqe_ring() is a no-op now.
        return false;
    }

    // Register a static (long-lived) WC TLB window covering the erisc L1 region
    // we touch (0x0000 – 0x1FFFF, 128 KB). Once mapped, all cluster().write_core
    // and read_core calls in this range take the fast path in
    // LocalChip::write_to_device / read_from_device — `tlb_window->write_block(
    // offset, src, size)` direct memcpy through the mmap'd BAR2, no TLB
    // reconfiguration on every call. Without this, every write to L1 reconfigures
    // the WC TLB (~4.3 µs per 1500 B observed); with this, ~150-200 ns.
    //
    // Size: 1 MB is the WH cached_tlb_size and the smallest TLB option that
    // covers our entire 128 KB region with margin. Held for the lifetime of
    // the process (TLBManager owns the unique_ptr; freed on cluster shutdown).
    //
    // Failure handling: if the TLB resource pool is exhausted (other callers
    // hold all available TLBs), configure_tlb throws. We log and proceed — the
    // slow path still works correctly, just at v1 baseline performance.
    try {
        auto& umd_cluster = *tt::tt_metal::MetalContext::instance().get_cluster().get_driver();
        umd_cluster.configure_tlb(
            chip_id_,
            ::tt_xy_pair{virtual_core_.x, virtual_core_.y},
            /*tlb_size=*/1u << 20,
            /*address=*/0u,
            // Relaxed ordering: WC writes don't enforce per-write order on the
            // PCIe bus. Ordering for the FW handshake comes from the explicit
            // read-back drains in flush_pending (forces WC flush before OWNED
            // publishes) and at the end of each batch (forces OWNED + producer
            // commit before next batch). Strict was tested but penalises
            // reads — every consumer_idx read serialises against pending WC
            // writes (~6 µs each); with Relaxed, those reads are ~1 µs.
            ::tt::umd::tlb_data::Relaxed);
        log_debug(
            tt::LogLLRuntime,
            "ExternalIfaceSender: registered 1MB WC TLB for chip {} core ({},{})",
            chip_id_,
            virtual_core_.x,
            virtual_core_.y);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogLLRuntime,
            "ExternalIfaceSender: configure_tlb failed for chip {} core ({},{}): {} — "
            "post_send will use the slow reconfigure path",
            chip_id_,
            virtual_core_.x,
            virtual_core_.y,
            e.what());
    }

    // Pre-dispatch RCB + descriptor-table init.
    //
    // tt-smi -r does NOT zero L1; whatever was at 0x8000 (descriptor table) /
    // 0x8400 (RCB) from a prior boot persists. Two failure modes seen in the
    // wild before this guard landed:
    //   (a) Stale OWNED_BY_FW bits on descriptors → FW enters run_wqe_ring_loop
    //       and immediately starts firing the 64 stale descriptors in a loop,
    //       observable as multi-Gbps wire traffic with no host posting.
    //   (b) Stale consumer_idx value → host's ring-full check
    //       (producer_local_ - cons) >= N triggers on the very first post_send.
    //
    // We zero everything FW will read on entry. This is safe ONLY because we
    // run BEFORE wait_for_link writes GW_MODE_MAGIC — FW hasn't entered the
    // loop yet, so there's no concurrency.
    //
    // Zero descriptor table: 64 slots × 16 B = 1024 B at kWqeDescTableAddr.
    {
        std::vector<uint8_t> zeros(kWqeRingN * kWqeDescStride, 0);
        cluster().write_core(chip_id_, virtual_core_, std::span<const uint8_t>(zeros), kWqeDescTableAddr);
    }
    // Zero ALL RCB fields (host-owned + FW-owned) to a known-good baseline.
    // Order matters less here than the descriptor zeroing because FW will
    // re-init consumer_idx / cq_head / fw_status on entry — but having them
    // start at 0 makes the ring-full check correct on first post_send.
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbProducerOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbConsumerOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbAckedIdxOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbFwStatusOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbCqHeadOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbRetxPendingOff, 0u);
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbRingLog2NOff, kWqeRingLog2N);
    // Mode last — this is the latch FW samples on gw-entry. Read-back drains
    // the PCIe write buffer before returning.
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff, 1u);
    (void)read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff);

    next_seq_ = 0;
    producer_local_ = 0;
    wqe_ring_enabled_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// enable_fw_dma_pull (Phase 3.1 plumbing)
//
// Allocates a 2 MB hugepage, maps it for NoC DMA via UMD's
// PCIDevice::map_hugepage_to_noc, then publishes the resulting NoC base
// address into RCB+0x20/+0x24 and flips RCB.mode to kRcbModeDmaPull (=2).
//
// Must be called BEFORE wait_for_link() — same constraint as enable_wqe_ring.
// FW reads RCB.mode only at gw-mode entry. Calling after wait_for_link()
// leaves FW in whatever mode the previous call selected and this call
// becomes a no-op on the FW dispatch path (the RCB writes still happen but
// FW has already chosen its loop).
//
// Returns false on hugepage allocation failure or if the device does not
// support map_hugepage_to_noc. On failure dma_pull_enabled_ stays false and
// the RCB is left untouched so the caller can fall back to mode-1 wqe-ring.
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::enable_fw_dma_pull() {
    if (read_word(chip_id_, virtual_core_, kModeAddr) == kModeMagic) {
        log_warning(tt::LogLLRuntime, "ExternalIfaceSender::enable_fw_dma_pull: too late — gw-mode already entered");
        return false;
    }

    if (dma_pull_enabled_) {
        // Idempotent re-enable: re-publish RCB fields in case the chip was
        // reset out from under us. The hugepage mapping is still valid (UMD
        // pins via the kernel driver, survives chip reset).
        write_word_immediate(
            chip_id_,
            virtual_core_,
            kWqeRcbAddr + kRcbHostNocAddrLoOff,
            static_cast<uint32_t>(host_payload_noc_addr_ & 0xFFFFFFFFu));
        write_word_immediate(
            chip_id_,
            virtual_core_,
            kWqeRcbAddr + kRcbHostNocAddrHiOff,
            static_cast<uint32_t>(host_payload_noc_addr_ >> 32));
        write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff, kRcbModeDmaPull);
        (void)read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff);
        return true;
    }

    // 2 MB hugepage. Sized for the eventual 64-slot × 1536-B ring (= 96 KB)
    // with comfortable headroom for any future per-slot metadata / control
    // structures that may live in the same hugepage. Default x86_64 hugepage
    // size is 2 MB; map_hugepage_to_noc rejects sizes >1 GB so 2 MB is safe.
    const size_t kSize = 2u << 20;
    void* buf = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (buf == MAP_FAILED) {
        log_warning(
            tt::LogLLRuntime,
            "ExternalIfaceSender::enable_fw_dma_pull: mmap(MAP_HUGETLB) failed: {} — "
            "is /proc/sys/vm/nr_hugepages > 0?",
            std::strerror(errno));
        return false;
    }

    // Touch the first page so the kernel actually allocates a hugepage
    // (deferred-fault would otherwise mean the IOCTL pin sees a hole).
    std::memset(buf, 0, 4096);

    auto& umd = *cluster().get_driver();
    auto* tt_device = umd.get_tt_device(chip_id_);
    if (!tt_device) {
        log_warning(
            tt::LogLLRuntime, "ExternalIfaceSender::enable_fw_dma_pull: get_tt_device({}) returned null", chip_id_);
        munmap(buf, kSize);
        return false;
    }
    auto pci = tt_device->get_pci_device();
    if (!pci) {
        log_warning(
            tt::LogLLRuntime,
            "ExternalIfaceSender::enable_fw_dma_pull: get_pci_device() returned null for chip {}",
            chip_id_);
        munmap(buf, kSize);
        return false;
    }

    if (!tt::umd::PCIDevice::is_mapping_buffer_to_noc_supported()) {
        log_warning(
            tt::LogLLRuntime,
            "ExternalIfaceSender::enable_fw_dma_pull: KMD too old "
            "(map_buffer_to_noc unsupported); needs ≥ 2.0.0");
        munmap(buf, kSize);
        return false;
    }

    std::pair<uint64_t, uint64_t> mapping;
    try {
        mapping = pci->map_hugepage_to_noc(buf, kSize);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogLLRuntime, "ExternalIfaceSender::enable_fw_dma_pull: map_hugepage_to_noc failed: {}", e.what());
        munmap(buf, kSize);
        return false;
    }

    // The KMD returns a noc_addr that includes the PCIe BAR marker bit (35)
    // but NOT the PCIe NIU tile's X/Y. From an erisc tile, we have to OR
    // those bits in ourselves so the request routes to the PCIe NIU. On a
    // Tensix the NIU's local PCIe-TLB handles it; the erisc NIU does not.
    //
    // For WH B0, PCIE_CORES_NOC0 = {{0, 3}} (umd/arch/wormhole_implementation
    // .hpp:172). NOC params: LOCAL_BITS=36, NODE_ID_BITS=6, so
    //   NOC_XY_PCIE_ENCODING(x, y) = (y << 42) | (x << 36) | 0x800000000.
    // Query soc_desc for the PCIe core so we don't have to hard-code the
    // arch here.
    auto pcie_cores = cluster().get_soc_desc(chip_id_).get_cores(CoreType::PCIE, ::tt::CoordSystem::NOC0);
    TT_FATAL(!pcie_cores.empty(), "soc_desc has no PCIe cores on chip {}", chip_id_);
    const uint32_t pcie_x = static_cast<uint32_t>(pcie_cores[0].x);
    const uint32_t pcie_y = static_cast<uint32_t>(pcie_cores[0].y);
    const uint64_t pcie_xy_encoding =
        (static_cast<uint64_t>(pcie_y) << 42) | (static_cast<uint64_t>(pcie_x) << 36) | 0x800000000ULL;
    const uint64_t composed_noc_addr = pcie_xy_encoding | mapping.first;

    host_payload_buf_ = buf;
    host_payload_noc_addr_ = composed_noc_addr;
    host_payload_pa_ = mapping.second;
    host_payload_size_ = kSize;

    log_info(
        tt::LogLLRuntime,
        "ExternalIfaceSender::enable_fw_dma_pull: chip {} hugepage VA={:#x} size={:#x} "
        "→ kernel_noc={:#x} pcie_core=({},{}) → composed_noc_addr={:#x} pa={:#x}",
        chip_id_,
        reinterpret_cast<uintptr_t>(host_payload_buf_),
        host_payload_size_,
        mapping.first,
        pcie_x,
        pcie_y,
        host_payload_noc_addr_,
        host_payload_pa_);

    // Publish to FW: lo/hi halves first, then mode (= latch). Read-back the
    // mode to drain the PCIe write buffer before returning so FW will see
    // the writes by the time it samples mode at gw-mode entry.
    write_word_immediate(
        chip_id_,
        virtual_core_,
        kWqeRcbAddr + kRcbHostNocAddrLoOff,
        static_cast<uint32_t>(host_payload_noc_addr_ & 0xFFFFFFFFu));
    write_word_immediate(
        chip_id_,
        virtual_core_,
        kWqeRcbAddr + kRcbHostNocAddrHiOff,
        static_cast<uint32_t>(host_payload_noc_addr_ >> 32));
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff, kRcbModeDmaPull);
    (void)read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbModeOff);

    dma_pull_enabled_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// P7: register_mr_slot — synchronous CTRL_REG_MR doorbell.
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::register_mr_slot(
    uint32_t slot, uint64_t base_noc_addr, uint64_t length, uint32_t rkey, uint32_t access_flags, uint32_t timeout_us) {
    if (slot >= 16u) {
        return false;
    }

    // Stage the 24 B mr_entry_t prefix at kMrStagingAddr. FW copies these
    // into MR_TABLE[slot] and writes its own generation/state bytes.
    //   +0x00 u64 base_noc_addr
    //   +0x08 u64 length
    //   +0x10 u32 rkey
    //   +0x14 u32 access_flags
    //   +0x18 u32 pd (reserved; 0)
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x00, static_cast<uint32_t>(base_noc_addr));
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x04, static_cast<uint32_t>(base_noc_addr >> 32));
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x08, static_cast<uint32_t>(length));
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x0C, static_cast<uint32_t>(length >> 32));
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x10, rkey);
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x14, access_flags);
    write_word_immediate(chip_id_, virtual_core_, kMrStagingAddr + 0x18, 0u);

    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbCtrlSlotOff, slot);
    // Doorbell write goes last — FW samples it each WQE-ring iteration.
    write_word_immediate(chip_id_, virtual_core_, kWqeRcbAddr + kRcbCtrlDoorbellOff, kCtrlSubopRegMr);

    // Spin until FW clears the doorbell (writes 0). At the WQE-loop service
    // rate (~µs) this resolves in tens of µs; timeout caps pathological cases.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    while (read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbCtrlDoorbellOff) != 0u) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Destructor — release hugepage mapping if enable_fw_dma_pull() ran.
// ---------------------------------------------------------------------------
ExternalIfaceSender::~ExternalIfaceSender() {
    if (host_payload_buf_ == nullptr) {
        return;
    }
    try {
        auto& umd = *cluster().get_driver();
        if (auto* tt_device = umd.get_tt_device(chip_id_)) {
            if (auto pci = tt_device->get_pci_device()) {
                pci->unmap_for_dma(host_payload_buf_, host_payload_size_);
            }
        }
    } catch (...) {
        // Best-effort; the kernel will reap the pin on process exit anyway.
    }
    munmap(host_payload_buf_, host_payload_size_);
    host_payload_buf_ = nullptr;
    host_payload_size_ = 0;
    dma_pull_enabled_ = false;
}

// ---------------------------------------------------------------------------
// post_send
// ---------------------------------------------------------------------------
std::optional<uint32_t> ExternalIfaceSender::post_send(std::span<const uint8_t> buf, uint32_t cookie) {
    PROFILE_OP(total_post);

    if (buf.size() > kWqePayloadStride) {
        return std::nullopt;
    }

    // Ring-full check. Two flavors depending on reliability mode:
    //   - reliability_enabled_: window scope = (producer - acked_idx). Slots
    //     are reserved until receiver ACKs; can't be overwritten while still
    //     in-flight. Required for retx correctness.
    //   - else: window scope = (producer - consumer_idx). Slot is reusable
    //     once FW fires it. Faster (no waiting on receiver), no reliability.
    //
    // Cached value is monotonic-stale-safe (always ≤ true) so a stale read
    // only over-estimates fullness; we return nullopt and caller retries.
    if (reliability_enabled_) {
        if ((producer_local_ - acked_idx_cached_) >= (kWqeRingN - drain_every_)) {
            PROFILE_OP(consumer_read);  // re-using the same op slot for the RTT
            acked_idx_cached_ = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbAckedIdxOff);
        }
        if ((producer_local_ - acked_idx_cached_) >= kWqeRingN) {
            return std::nullopt;
        }
    } else {
        if ((producer_local_ - cons_cached_) >= (kWqeRingN - drain_every_)) {
            PROFILE_OP(consumer_read);
            cons_cached_ = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbConsumerOff);
        }
        if ((producer_local_ - cons_cached_) >= kWqeRingN) {
            return std::nullopt;
        }
    }

    const uint32_t slot = producer_local_ & (kWqeRingN - 1);
    const uint32_t desc_addr = kWqeDescTableAddr + slot * kWqeDescStride;
    const uint32_t seq = next_seq_;

    // payload_off encoding switches by mode:
    //   - L1-staged (Phase 1): kWqePayloadBase + slot*kWqePayloadStride.
    //     CMAC TX programs ETH_TXQ_TRANSFER_START_ADDR with this value.
    //   - DMA-pull  (Phase 3.2): slot*kWqePayloadStride — a byte offset into
    //     the host hugepage. FW issues a NoC PCIe read from host_noc_base +
    //     payload_off into TX_BUF0 before firing CMAC TX.
    const uint32_t payload_o =
        dma_pull_enabled_ ? (slot * kWqePayloadStride) : (kWqePayloadBase + slot * kWqePayloadStride);

    // 1) Payload bytes → either L1 slot (BAR write, Phase 1) or host
    //    hugepage slot (plain memcpy, Phase 3.2). The memcpy path avoids
    //    the PCIe BAR write entirely for payload bytes — only the descriptor
    //    + producer_idx (~24 B total) traverse the BAR per frame.
    {
        PROFILE_OP(payload_write);
        if (dma_pull_enabled_) {
            std::memcpy(static_cast<uint8_t*>(host_payload_buf_) + slot * kWqePayloadStride, buf.data(), buf.size());
        } else {
            cluster().write_core(chip_id_, virtual_core_, std::span<const uint8_t>(buf), payload_o);
        }
    }

    // 2) Descriptor pre-owned: size+flags=0, seq, payload_off, cookie.
    //    Single 16-byte block write through the cached WC TLB. FW polls word0's
    //    high-half (flags) so it won't act until step 4's OWNED publish.
    //
    //    Layout: +0x00 u16 size | u16 flags=0
    //            +0x04 u32 seq
    //            +0x08 u32 payload_off
    //            +0x0C u32 cookie
    {
        PROFILE_OP(desc_pre_owned);
        uint32_t desc[4];
        desc[0] = static_cast<uint32_t>(buf.size()) & 0xFFFFu;  // size, flags=0
        desc[1] = seq;
        desc[2] = payload_o;
        desc[3] = cookie;
        cluster().write_core(desc, sizeof(desc), tt_cxy_pair(chip_id_, virtual_core_), desc_addr);
    }

    // 3) Track this slot for batched flush. OWNED publish + producer_idx bump
    //    + drain happen once per drain_every_ posts in flush_pending().
    pending_slot_[pending_count_] = slot;
    pending_size_[pending_count_] = static_cast<uint16_t>(buf.size());
    pending_flags_[pending_count_] = 0;  // ordinary SEND/WRITE — no extra flags
    pending_count_++;
    last_pending_payload_off_ = payload_o;

    // Bump host's producer_local_ now (cheap; not visible to FW until flush).
    producer_local_++;
    next_seq_++;

    // 4) Auto-flush when batch is full.
    if (pending_count_ >= drain_every_) {
        flush_pending();
    }
    return seq;
}

// ---------------------------------------------------------------------------
// Phase I: post_send_read — host-initiated one-sided RDMA READ.
// ---------------------------------------------------------------------------
std::optional<uint32_t> ExternalIfaceSender::post_send_read(
    uint64_t local_mr_noc_addr,
    uint32_t local_offset,
    uint32_t remote_rkey,
    uint64_t remote_offset,
    uint16_t length,
    uint32_t cookie) {
    if (!dma_pull_enabled_) {
        return std::nullopt;  // legacy mode-1 doesn't carry correlation metadata
    }

    // Wire frame layout we stage in the hugepage slot (256 B total):
    //   +0x00..+0x1F : 32 B RDMA v1 READ_REQ header
    //                   opcode=0x20, version_flags=0x01, tag=seq&0xFFFF,
    //                   length=requested_bytes, seq, rkey, remote_offset,
    //                   imm=0, hdr_crc=0
    //   +0x20..+0x2F : 16 B FW correlation metadata
    //                   local_mr_noc_addr (8 B), local_offset (4 B), length (4 B)
    //   +0x30..+0xFF : 208 B zero padding to a safe wire size (>= Ethernet min)
    constexpr uint32_t kReadReqWireBytes = 256;
    static_assert(kReadReqWireBytes <= kWqePayloadStride, "READ_REQ wire size must fit in WQE payload stride");

    // Ring-full check (same as post_send).
    if (reliability_enabled_) {
        if ((producer_local_ - acked_idx_cached_) >= (kWqeRingN - drain_every_)) {
            acked_idx_cached_ = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbAckedIdxOff);
        }
        if ((producer_local_ - acked_idx_cached_) >= kWqeRingN) {
            return std::nullopt;
        }
    } else {
        if ((producer_local_ - cons_cached_) >= (kWqeRingN - drain_every_)) {
            cons_cached_ = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbConsumerOff);
        }
        if ((producer_local_ - cons_cached_) >= kWqeRingN) {
            return std::nullopt;
        }
    }

    const uint32_t slot = producer_local_ & (kWqeRingN - 1);
    const uint32_t desc_addr = kWqeDescTableAddr + slot * kWqeDescStride;
    const uint32_t seq = next_seq_;
    const uint32_t payload_o = slot * kWqePayloadStride;

    // Build the frame in a stack buffer, then memcpy to hugepage.
    uint8_t frame[kReadReqWireBytes];
    std::memset(frame, 0, sizeof(frame));

    // 32 B RDMA header
    frame[0] = 0x20;  // opcode READ_REQ
    frame[1] = 0x01;  // version_flags
    uint16_t tag = static_cast<uint16_t>(seq & 0xFFFFu);
    std::memcpy(frame + 2, &tag, 2);
    uint32_t plen32 = static_cast<uint32_t>(length);
    std::memcpy(frame + 4, &plen32, 4);          // length = bytes-requested
    std::memcpy(frame + 8, &seq, 4);             // seq
    std::memcpy(frame + 12, &remote_rkey, 4);    // rkey
    std::memcpy(frame + 16, &remote_offset, 8);  // remote_offset (u64 LE)
    // imm @ +24 and hdr_crc @ +28 are already zero from memset.

    // 16 B correlation metadata at +32..+47
    std::memcpy(frame + 32, &local_mr_noc_addr, 8);
    std::memcpy(frame + 40, &local_offset, 4);
    uint32_t length32 = static_cast<uint32_t>(length);
    std::memcpy(frame + 44, &length32, 4);

    // Stage in hugepage slot
    std::memcpy(static_cast<uint8_t*>(host_payload_buf_) + slot * kWqePayloadStride, frame, kReadReqWireBytes);

    // Pre-owned descriptor
    {
        uint32_t desc[4];
        desc[0] = kReadReqWireBytes & 0xFFFFu;  // size, flags=0 (OWNED set in flush)
        desc[1] = seq;
        desc[2] = payload_o;
        desc[3] = cookie;
        cluster().write_core(desc, sizeof(desc), tt_cxy_pair(chip_id_, virtual_core_), desc_addr);
    }

    // Track + auto-flush. Important: set the READ_REQ flag so FW records
    // correlation and defers cq_head bump until READ_RESP lands.
    pending_slot_[pending_count_] = slot;
    pending_size_[pending_count_] = kReadReqWireBytes;
    pending_flags_[pending_count_] = kWqeFlagReadReq;
    pending_count_++;
    last_pending_payload_off_ = payload_o;
    producer_local_++;
    next_seq_++;
    if (pending_count_ >= drain_every_) {
        flush_pending();
    }

    return seq;
}

// ---------------------------------------------------------------------------
// flush_pending — batch OWNED publish + producer_idx bump.
//
// Sequence:
//   1) ONE payload drain read-back. Forces the PCIe WC buffer to flush all
//      preceding payload writes for the mapped TLB region (any address in
//      that region works as the read target — we use the most-recently-written
//      payload start). Costs ~7 µs for 1500 B at this rig's PCIe rate, but
//      amortised across drain_every_ posts.
//   2) OWNED publish on each pending slot. With Strict TLB ordering these
//      become visible in program order; with cached-TLB block writes each
//      costs ~250 ns.
//   3) Single producer_idx bump in RCB — FW now sees all the new slots.
//   4) Producer drain read-back.
// ---------------------------------------------------------------------------
void ExternalIfaceSender::flush_pending() {
    if (pending_count_ == 0) {
        return;
    }

    if (dma_pull_enabled_) {
        // No PCIe WC buffer to drain — payload bytes were memcpy'd into the
        // hugepage on the host side. An sfence guarantees the memcpy stores
        // are globally visible before the OWNED publish below triggers FW
        // to issue its NoC PCIe read of the hugepage. x86 stores are TSO-
        // ordered, but for normal cacheable RAM there's no PCIe ordering
        // domain involvement, so the explicit fence is the safest portable
        // option. Cost: ~10 ns vs. ~7 µs payload_drain in Phase 1.
        __sync_synchronize();
    } else {
        PROFILE_OP(payload_drain);
        (void)read_word(chip_id_, virtual_core_, last_pending_payload_off_);
    }

    {
        PROFILE_OP(owned_publish);
        for (uint32_t i = 0; i < pending_count_; ++i) {
            const uint32_t slot = pending_slot_[i];
            const uint32_t desc_addr = kWqeDescTableAddr + slot * kWqeDescStride;
            const uint16_t flags = static_cast<uint16_t>(kWqeFlagOwnedByFw | pending_flags_[i]);
            const uint32_t word0_owned =
                (static_cast<uint32_t>(pending_size_[i]) & 0xFFFFu) | (static_cast<uint32_t>(flags) << 16);
            write_word_block(chip_id_, virtual_core_, desc_addr + 0x00, word0_owned);
        }
    }

    {
        PROFILE_OP(producer_write);
        write_word_block(chip_id_, virtual_core_, kWqeRcbAddr + kRcbProducerOff, producer_local_);
    }

    pending_count_ = 0;
}

void ExternalIfaceSender::set_drain_every(uint32_t n) { drain_every_ = (n == 0) ? 1u : std::min(n, kPendingMax); }

// ---------------------------------------------------------------------------
// poll_completion
// ---------------------------------------------------------------------------
uint32_t ExternalIfaceSender::poll_completion() {
    return read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbCqHeadOff);
}

// ---------------------------------------------------------------------------
// wait_completion
//
// Block until cq_head has advanced to >= seq, or timeout. cq_head is the
// last completed seq#, so condition is (cq_head + 1) > seq, i.e. cq_head >= seq
// once seq has been completed. Compare via signed wrap-safe difference.
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::wait_completion(uint32_t seq, uint32_t timeout_ms) {
    // Auto-flush any pending posts so FW can possibly see seq complete.
    if (pending_count_ > 0) {
        flush_pending();
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const uint32_t head = poll_completion();
        // Wrap-safe: treat as signed 32-bit difference.
        if (static_cast<int32_t>(head - seq) >= 0) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// ---------------------------------------------------------------------------
// inflight
// ---------------------------------------------------------------------------
uint32_t ExternalIfaceSender::inflight() const {
    const uint32_t cons = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbConsumerOff);
    return producer_local_ - cons;
}

// ===========================================================================
// Phase 2: reliability layer (sliding-window ARQ)
// ===========================================================================

void ExternalIfaceSender::enable_reliability(bool enabled) {
    reliability_enabled_ = enabled;
    if (enabled) {
        last_ack_progress_ = std::chrono::steady_clock::now();
    }
}

uint32_t ExternalIfaceSender::poll_acked() {
    acked_idx_cached_ = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbAckedIdxOff);
    return acked_idx_cached_;
}

bool ExternalIfaceSender::wait_acked(uint32_t seq, uint32_t timeout_ms) {
    if (pending_count_ > 0) {
        flush_pending();
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const uint32_t head = poll_acked();
        if (static_cast<int32_t>(head - seq) >= 0) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

bool ExternalIfaceSender::tick_retx() {
    if (!reliability_enabled_) {
        return false;
    }

    const uint32_t fresh = read_word(chip_id_, virtual_core_, kWqeRcbAddr + kRcbAckedIdxOff);
    const auto now = std::chrono::steady_clock::now();

    // Wrap-safe: did acked_idx make progress since last tick?
    if (static_cast<int32_t>(fresh - acked_idx_cached_) > 0) {
        acked_idx_cached_ = fresh;
        last_ack_progress_ = now;
        return false;
    }

    // No progress. Check if there's anything un-acked + timeout elapsed.
    if (acked_idx_cached_ == producer_local_) {
        // Nothing in flight — keep timer fresh so it doesn't fire spuriously.
        last_ack_progress_ = now;
        return false;
    }

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_ack_progress_).count();
    if (static_cast<uint32_t>(elapsed_us) < retx_timeout_us_) {
        return false;
    }

    // Timeout elapsed: trigger FW retx walk. FW will:
    //   1. Walk slots [acked_idx, producer_local_), re-set OWNED on each
    //   2. Rewind consumer_idx to acked_idx, so the TX loop re-fires them
    //   3. Clear retx_pending
    write_word_block(chip_id_, virtual_core_, kWqeRcbAddr + kRcbRetxPendingOff, producer_local_);

    // Reset the ack-progress clock — give it another retx_timeout_us_ before
    // triggering again. Avoids retx storms when receiver is genuinely down.
    last_ack_progress_ = now;
    return true;
}

}  // namespace tt::llrt
