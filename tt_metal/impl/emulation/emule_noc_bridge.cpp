// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Extern-C memory/NOC bridge + fiber-scheduler thunks, moved verbatim from
// emulated_program_runner.cpp. These are the dlsym kernel ABI: C linkage +
// global/unmangled symbols, resolved at dlopen via -rdynamic.

#include "emule_noc_bridge.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "emule_device_map.hpp"
#include "emule_fiber_scheduler.hpp"
#include "tt_emule/device.hpp"   // tt_emule::Core, CoreRole
#include "tt_emule/l1_pool.hpp"  // tt_emule::L1Pool
#include "umd/device/chip/sw_emule_chip.hpp"
#include "umd/device/chip_helpers/simulation_sysmem_manager.hpp"  // SimulationSysmemManager

// Defined in emule_asan_panic.cpp (same libtt_metal); the checks below and the JIT
// kernel .so files resolve it at link/dlopen.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

// Per-core NOC coordinates (declared extern in the header): set per fiber on swap-in,
// read by __emule_multicast_write.
thread_local uint8_t my_x[NUM_NOCS] = {};
thread_local uint8_t my_y[NUM_NOCS] = {};

// ===== bridge bodies (verbatim from emulated_program_runner.cpp) =====
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset) {
    emule_require_self(__func__);
    // ASAN out-of-bounds DRAM check (inert when TT_METAL_EMULE_ASAN off — ranges stay null). Range-test
    // in 32 bits to match the live-range registry (uint32_t start/end); the 64-bit offset is used only
    // for the backing-store address. Assumes DRAM addresses fit in 32 bits (true for every WH/BH config).
    if (__emule_self->san.dram_tensor_ranges != nullptr &&
        static_cast<uint32_t>(offset) >= __emule_self->san.dram_unreserved_base) {
        uint32_t addr = static_cast<uint32_t>(offset);
        bool in_tensor = false;
        for (uint32_t i = 0; i < __emule_self->san.dram_tensor_ranges_count; ++i) {
            uint64_t packed = __emule_self->san.dram_tensor_ranges[i];
            uint32_t r_start = static_cast<uint32_t>(packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(packed);
            if (addr >= r_start && addr < r_end) {
                in_tensor = true;
                break;
            }
        }
        if (!in_tensor) {
            __emule_asan_panic(
                "[ASAN ERROR] Out-of-Bounds Write: Attempted to access DRAM address 0x%x which is not part of any "
                "allocated tensor\n",
                addr);
        }
    }
    return __emule_self->bridge_dram ? __emule_self->bridge_dram + offset : nullptr;
}

extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset) {
    emule_require_self(__func__);
    // ASAN illegal-semaphore-region check (inert when ASAN off — range end stays 0).
    if (__emule_self->san.sem_l1_range_end > 0 && offset >= __emule_self->san.sem_l1_range_start &&
        offset < __emule_self->san.sem_l1_range_end) {
        __emule_asan_panic(
            "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
            offset,
            __emule_self->san.sem_l1_range_start,
            __emule_self->san.sem_l1_range_end);
    }
    return __emule_self->bridge_l1 ? __emule_self->bridge_l1 + offset : nullptr;
}

extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
    emule_require_self(__func__);
    if (__emule_self->core_map) {
        uint64_t key = (uint64_t(x) << 32) | y;
        auto it = __emule_self->core_map->find(key);
        if (it != __emule_self->core_map->end()) {
            return it->second->l1_ptr(static_cast<uint32_t>(addr));
        }
    }
    return nullptr;
}

// Fiber-scheduler bridge — the dlopen'd kernel .so calls these (declared in
// include/jit_hw/internal/emule_fiber_bridge.h) to park/wake/yield on the one
// scheduler instance. Resolved at dlopen via -rdynamic, like the resolvers above.
namespace efib = tt::tt_metal::emule_fiber;
extern "C" void __emule_fiber_lock(void) { efib::FiberScheduler::instance().lock(); }
extern "C" void __emule_fiber_unlock(void) { efib::FiberScheduler::instance().unlock(); }
extern "C" void __emule_fiber_park_locked(const void* key) { efib::FiberScheduler::instance().park_locked(key); }
extern "C" void __emule_fiber_park_locked_socket(const void* key) {
    efib::FiberScheduler::instance().park_locked_socket(key);
}
extern "C" void __emule_fiber_note_socket_poll_wait(int waiting, int host_fed) {
    efib::FiberScheduler::instance().note_socket_poll_wait(waiting != 0, host_fed != 0);
}
extern "C" void __emule_fiber_note_cb_poll_wait(unsigned cb_id, unsigned n) {
    efib::FiberScheduler::instance().note_cb_poll_wait(cb_id, n);
}
extern "C" void __emule_fiber_wake(const void* key) { efib::FiberScheduler::instance().wake(key); }
extern "C" void __emule_fiber_yield(void) { efib::FiberScheduler::instance().yield(); }
extern "C" void __emule_fiber_defer_to_quiescence(void) { efib::FiberScheduler::instance().quiescence_park(); }
extern "C" void __emule_fiber_note_publish(unsigned pages) { efib::FiberScheduler::instance().note_publish(pages); }

// Worker L1 slot size + mask: a worker's L1 field is a 0-based in-slot offset (< 2 MB), so masking the low
// bits is an idempotent guard. Applied ONLY for WORKER cores (DRAM banks are GB-scale — see the
// per-resolver comments). Used by every NOC-address resolver.
// Taken FROM the pool rather than restated: the mask is only an idempotent guard while it matches the
// allocator's actual stride, and a peer rank resolves into the same segment using the same constant.
static constexpr uint32_t L1_SLOT_SIZE = static_cast<uint32_t>(tt_emule::L1Pool::SLOT_SIZE);
static constexpr uint32_t L1_SLOT_MASK = L1_SLOT_SIZE - 1;  // 0x1FFFFF

// Resolve a NOC address (encoded 64-bit) to a host pointer.
// Real firmware encoding: y in bits [47:42], x in bits [41:36], addr in bits [35:0]
//
// The decoded offset is bounded by the target core's own size, not masked into range: a
// worker L1 field is a 0-based in-slot offset while a DRAM bank is GB-scale (2 GB on
// Wormhole views, 4 GB on Blackhole), so no single mask fits both, and an offset that fits
// neither belongs to no core.
// get_sw_emulated_chip / get_pcie_base_cached moved to emule_device_map.{hpp,cpp}.

extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr) {
    emule_require_self(__func__);

    // pcie_base alone cannot tell host-facing from on-chip: on Wormhole it is 0x8'0000'0000,
    // below the bit-36 coordinate field, so every on-chip address clears it. Registry
    // membership is the discriminator; the threshold is only a pre-filter.
    uint32_t device_id = __emule_self->chip_id;
    if (noc_addr >= tt::tt_metal::emule::get_pcie_base_cached(device_id)) {
        auto* sw_emu = tt::tt_metal::emule::get_sw_emulated_chip(static_cast<tt::ChipId>(device_id));
        auto* sysmem = sw_emu ? static_cast<tt::umd::SimulationSysmemManager*>(sw_emu->get_sysmem_manager()) : nullptr;
        // A host-facing address (>= pcie_base) is by construction on an emule chip that has a
        // SimulationSysmemManager, so a null manager is a contract violation, not a resolvable miss.
        TT_FATAL(
            sysmem != nullptr,
            "emule: host-facing NOC address 0x{:x} on chip {} has no SimulationSysmemManager.",
            noc_addr,
            device_id);
        if (auto* host_ptr = static_cast<uint8_t*>(sysmem->get_mapped_host_ptr(noc_addr))) {
            return host_ptr;
        }
        // Miss: decode as on-chip. The bounds check below is what keeps an unmapped
        // host-window address from landing on a core — it decodes to a real coord (the
        // window carries no coordinates) but with an offset no core is that big.
    }

    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t local_addr = noc_addr & NOC_LOCAL_MASK;  // 36 bits, raw

    if (__emule_self->core_map) {
        uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
        auto it = __emule_self->core_map->find(key);
        if (it != __emule_self->core_map->end()) {
            // An offset the target cannot hold is not an address on that core, so it is a
            // resolve miss like any other. Bounding it by the core's own size, rather than
            // masking it into range, is what keeps a bad address from silently landing on
            // real memory.
            if (local_addr < it->second->l1_size()) {
                return it->second->l1_ptr(local_addr);
            }
        }
    }
    return nullptr;
}

extern "C" bool __emule_noc_addr_is_dram(uint64_t noc_addr) {
    emule_require_self(__func__);
    if (!__emule_self->core_map) {
        return false;
    }
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
    auto it = __emule_self->core_map->find(key);
    if (it != __emule_self->core_map->end()) {
        return it->second->role() == tt_emule::CoreRole::DRAM;
    }
    return false;
}

// True only when (noc_x,noc_y) maps to a WORKER core in this chip's core_map.
// Used by the mcast semaphore walk to tell an *expected grid gap* (an ARC/DRAM/eth
// node with no worker NIU — false here) from a *genuine* worker-resolve miss.
// On the unharvested BH grid the worker columns are non-contiguous (ARC=x8,
// DRAM=x9), so a full-grid mcast bbox legitimately straddles non-worker nodes;
// silicon simply has no worker semaphore there. Mirrors __emule_noc_addr_is_dram.
extern "C" bool __emule_noc_addr_is_worker(uint64_t noc_addr) {
    emule_require_self(__func__);
    if (!__emule_self->core_map) {
        return false;
    }
    uint32_t noc_x = (noc_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t noc_y = (noc_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t key = (uint64_t(noc_x) << 32) | noc_y;
    auto it = __emule_self->core_map->find(key);
    return it != __emule_self->core_map->end() && it->second->role() == tt_emule::CoreRole::WORKER;
}

// Resolve multicast: iterate over rectangle of cores and memcpy to each.
// Real firmware encoding: x_start [53:48], y_start [59:54], x_end [41:36], y_end [47:42], addr [35:0]
//
// `include_self`: silicon's NOC_CMD_BRCST_SRC_INCLUDE bit. When the API is
// `noc_async_write_multicast_loopback_src` (or _set_multicast_loopback_src),
// silicon sets the bit and the sender NIU receives its own packet ->
// include_self=true. When the API is `noc_async_write_multicast` (non-loopback),
// silicon clears the bit and the sender NIU drops the packet at itself ->
// include_self=false. Sender coords come from the TLS that thread launch
// wires up (my_x[0], my_y[0]).
extern "C" void __emule_multicast_write(
    uint64_t mcast_addr, const uint8_t* src, uint32_t size, bool include_self, uint8_t noc) {
    uint32_t x_end = (mcast_addr >> NOC_LOCAL_BITS) & NOC_NODE_MASK;
    uint32_t y_end = (mcast_addr >> (NOC_LOCAL_BITS + NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t x_start = (mcast_addr >> (NOC_LOCAL_BITS + 2 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint32_t y_start = (mcast_addr >> (NOC_LOCAL_BITS + 3 * NOC_NODE_ID_BITS)) & NOC_NODE_MASK;
    uint64_t l1_offset = mcast_addr & NOC_LOCAL_MASK;

    // Left raw: the offset is a 0-based in-slot L1 offset (get_write_ptr() etc.), and an
    // out-of-range one is a kernel bug, so Core::l1_ptr's bounds check should surface it
    // rather than a mask hiding it. Multicast targets only WORKER cores; the delivery loop
    // below skips the rest.

    emule_require_self(__func__);
    if (!__emule_self->core_map) {
        return;
    }

    // Sender coordinates (from the TLS that thread launch wires up). Used to
    // skip self when include_self=false (non-loopback multicast).
    uint32_t self_x = my_x[0];
    uint32_t self_y = my_y[0];

    // NOC1 rectangles arrive with start<->end SWAPPED: silicon describes a NOC1
    // multicast in NOC1's reflected coordinate frame (paired with the DYNAMIC_NOC_X/Y
    // reflection). Emule models NOC coordinates as identity (DYNAMIC_NOC_X/Y is
    // identity), so no reflection happens and the swap alone leaves the rectangle
    // reversed. Undo it so the walk below runs on physical (NOC0-frame) coordinates
    // for both NOCs. Without this, canonical NOC1 in0/in1-mcast ops (matmul/linear,
    // multicore argmax) present start>end and the torus walk misreads it as a
    // wraparound → receivers never see their semaphore → quiescent deadlock.
    if (noc != 0) {
        uint32_t t;
        t = x_start;
        x_start = x_end;
        x_end = t;
        t = y_start;
        y_start = y_end;
        y_end = t;
    }

    // Torus-wraparound walk on physical coords. Silicon's NOC treats the rectangle on
    // a torus, so a rectangle whose cores straddle the worker-grid seam encodes
    // start > end and wraps around the NOC node space rather than covering the min..max
    // bounding box; SDPA S-block multicasts (NOC0) rely on this. Walk each axis
    // start->end stepping +1 mod the node space; coords with no core in the map are
    // skipped. For a non-wrapping rectangle (start <= end) this is identical to
    // min..max — the post-un-swap NOC1 case (matmul/argmax in0-mcast).
    auto axis_count = [](uint32_t s, uint32_t e) -> uint32_t {
        return (e >= s ? (e - s) : ((NOC_NODE_MASK + 1 - s) + e)) + 1;
    };
    const uint32_t nx = axis_count(x_start, x_end);
    const uint32_t ny = axis_count(y_start, y_end);
    uint32_t delivered = 0;
    for (uint32_t ix = 0; ix < nx; ix++) {
        const uint32_t x = (x_start + ix) & NOC_NODE_MASK;
        for (uint32_t iy = 0; iy < ny; iy++) {
            const uint32_t y = (y_start + iy) & NOC_NODE_MASK;
            if (!include_self && x == self_x && y == self_y) {
                continue;
            }
            uint64_t key = (uint64_t(x) << 32) | y;
            auto it = __emule_self->core_map->find(key);
            if (it != __emule_self->core_map->end() && it->second->role() == tt_emule::CoreRole::WORKER) {
                uint8_t* dst = it->second->l1_ptr(l1_offset);
                if (size == sizeof(uint32_t)) {
                    TT_FATAL(
                        reinterpret_cast<uintptr_t>(dst) % alignof(std::atomic<uint32_t>) == 0,
                        "multicast_write: L1 offset 0x{:x} is not 4-byte aligned for atomic store",
                        l1_offset);
                    // Atomic store for semaphore-sized writes (4 bytes)
                    uint32_t val;
                    std::memcpy(&val, src, sizeof(uint32_t));
                    reinterpret_cast<std::atomic<uint32_t>*>(dst)->store(val, std::memory_order_release);
                    efib::FiberScheduler::instance().wake(dst);  // wake the target core's sem waiter
                } else {
                    std::memcpy(dst, src, size);
                    std::atomic_thread_fence(std::memory_order_release);
                }
                delivered++;
            }
        }
    }
    static const bool mdbg = std::getenv("EMULE_DEBUG") != nullptr;
    if (delivered == 0 && mdbg) {
        fprintf(
            stderr,
            "EMULE WARN: multicast (%u,%u)->(%u,%u) offset=0x%lx size=%u: "
            "no worker cores found [from phys (%u,%u)]\n",
            x_start,
            y_start,
            x_end,
            y_end,
            (unsigned long)l1_offset,
            size,
            my_x[0],
            my_y[0]);
    }
}
