// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ping-pong latency between two Keraunos CCE harts (IDLE_ETH). The pair is two harts of one core, CCE0 of two HSIO
// tiles, or the two ends of an Ethernet link.
//
// Each participant polls only its own mailbox (uncached), and every exchange carries a distinct token
// (nonce + exchange + 1), so a stale or repeated value never satisfies a wait. Per exchange:
//   initiator: t0; put token in the peer's mailbox; wait for the token in its own; t1. RTT = t1 - t0.
//   responder: wait for the token in its own mailbox; t0; put it in the peer's; t1. TURN = t1 - t0.
// RTT includes the responder's TURN. Nothing here halves RTT into a one-way time.
//
// The first warm-up exchange is the rendezvous: it absorbs the launch skew between the two cores, so it gets its own
// result slot and is never a sample. Exchanges after warm-up are samples (at most kMaxSamples are stored).
//
// Transports (LatencyTransport):
//   CpuStore  32-bit store to the peer mailbox's CPU address, then fence.
//   Idma      the token goes to this hart's staging word, then an 8-byte cce_noc::copy() to the peer mailbox's iDMA
//             address. The copy is not waited for: the next token is written only after a full round trip, and the
//             kernel checks the previous copy is acked first.
//   Eth       optional payload of P bytes (cce_eth::send_bytes from the data region to the same offset on the link
//             peer), then the token as an inline 8-byte write to the peer mailbox's offset, both on TX queue q.
//
// Runtime args:
//   0 base, 1 hart (result/mailbox/sample slot), 2 role, 3 transport, 4 warm-up exchanges, 5 samples, 6 polls per
//   wait, 7 nonce, 8-9 peer mailbox CPU address, 10-11 peer mailbox iDMA address, 12 peer mailbox flat offset (Eth),
//   13-14 own staging iDMA address, 15 payload bytes (Eth; 0 = token only), 16 TX queue (Eth).
//
// Result: cycles[kMin/kMax/kSum] over the samples, cycles[kRendezvous]; samples at samples_offset(hart).

#include <cstdint>

#include "internal/ethernet/tunneling.h"  // eth_cce_send.h, which needs tunneling.h's postcode pointers
#include "internal/tt-2xx/quasar/cce_noc.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

using namespace keraunos_ubench;
namespace lat = keraunos_ubench::latency;
namespace cce_eth = internal_::cce_eth;
namespace cce_noc = internal_::cce_noc;

namespace {

struct Args {
    uint32_t base;
    uint32_t hart;
    LatencyRole role;
    LatencyTransport transport;
    uint32_t warmup;
    uint32_t samples;
    uint32_t polls;
    uint32_t nonce;
    uint64_t peer_cpu;
    uint64_t peer_idma;
    uint32_t peer_flat;
    uint64_t own_staging_idma;
    uint32_t payload_bytes;
    uint32_t txq;
};

// Puts @p token into the peer's mailbox. False when a precondition wait timed out.
bool put(const Args& a, uint32_t token) {
    switch (a.transport) {
        case LatencyTransport::CpuStore: {
            *reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(a.peer_cpu)) = token;
            asm volatile("fence iorw, iorw" ::: "memory");
            return true;
        }
        case LatencyTransport::Idma: {
            // The staging word is the previous copy's source: it must be read before it is overwritten.
            uint32_t p = 0;
            while (!cce_noc::copies_done()) {
                if (++p >= a.polls) {
                    return false;
                }
            }
            *uncached<uint64_t>(a.base + staging_offset(a.hart)) = token;
            asm volatile("fence iorw, iorw" ::: "memory");
            cce_noc::copy(a.own_staging_idma, a.peer_idma, sizeof(uint64_t));
            return true;
        }
        case LatencyTransport::Eth: {
            if (a.payload_bytes != 0) {
                while (cce_eth::txq_busy(a.txq)) {
                }
                const uint32_t data = a.base + src_offset(a.hart);
                cce_eth::send_bytes(a.txq, data, data, a.payload_bytes);
            }
            while (cce_eth::txq_busy(a.txq)) {
            }
            cce_eth::send_dword(a.txq, a.peer_flat, token);
            return true;
        }
    }
    return false;
}

}  // namespace

void kernel_main() {
    Args a{};
    a.base = get_arg_val<uint32_t>(0);
    a.hart = get_arg_val<uint32_t>(1);
    a.role = static_cast<LatencyRole>(get_arg_val<uint32_t>(2));
    a.transport = static_cast<LatencyTransport>(get_arg_val<uint32_t>(3));
    a.warmup = get_arg_val<uint32_t>(4);
    a.samples = get_arg_val<uint32_t>(5);
    a.polls = get_arg_val<uint32_t>(6);
    a.nonce = get_arg_val<uint32_t>(7);
    a.peer_cpu = arg_u64(8);
    a.peer_idma = arg_u64(10);
    a.peer_flat = get_arg_val<uint32_t>(12);
    a.own_staging_idma = arg_u64(13);
    a.payload_bytes = get_arg_val<uint32_t>(15);
    a.txq = get_arg_val<uint32_t>(16);

    volatile Result* r = result_block(a.base, a.hart);
    r->kernel = static_cast<uint32_t>(KernelId::Latency);
    r->hart = a.hart;
    r->status = kStatusRunning;

    uint32_t bad = 0;
    if (a.hart >= kMaxHarts) {
        bad = 1;
    } else if (a.warmup == 0) {
        bad = 2;  // the first warm-up exchange is the rendezvous
    } else if (a.samples == 0 || a.samples > kMaxSamples) {
        bad = 3;
    } else if (a.transport == LatencyTransport::Eth &&
               (a.txq >= cce_eth::kNumTxqs || a.payload_bytes % 16 != 0 || a.payload_bytes > kMaxTransferBytes ||
                a.peer_flat % 8 != 0)) {
        bad = 4;
    } else if (static_cast<uint32_t>(a.transport) > static_cast<uint32_t>(LatencyTransport::Eth) ||
               static_cast<uint32_t>(a.role) > static_cast<uint32_t>(LatencyRole::Responder)) {
        bad = 5;
    }
    if (bad != 0) {
        r->detail = bad;
        finish(r, kStatusBadArgs);
        return;
    }
    if (a.transport == LatencyTransport::Eth) {
        cce_eth::check_on_cce0();
    }

    const volatile uint32_t* mine = uncached(a.base + mailbox_offset(a.hart));
    volatile uint32_t* samples = uncached(a.base + samples_offset(a.hart));
    const bool initiator = a.role == LatencyRole::Initiator;
    const uint32_t total = a.warmup + a.samples;
    uint64_t lo = ~0ull, hi = 0, sum = 0;

    for (uint32_t i = 0; i < total; i++) {
        const uint32_t token = a.nonce + i + 1;
        uint64_t t0 = 0, t1 = 0;
        bool ok = true;
        if (initiator) {
            t0 = now();
            ok = put(a, token) && wait_eq<uint32_t>(mine, token, a.polls);
            t1 = now();
        } else {
            ok = wait_eq<uint32_t>(mine, token, a.polls);
            t0 = now();
            ok = ok && put(a, token);
            t1 = now();
        }
        if (!ok) {
            r->timeout_iteration = i + 1;
            r->detail = token;
            finish(r, kStatusTimeout);
            return;
        }
        const uint64_t d = t1 - t0;
        if (i == 0) {
            r->cycles[lat::kRendezvous] = d;
        } else if (i >= a.warmup) {
            samples[i - a.warmup] = static_cast<uint32_t>(d > 0xFFFFFFFFull ? 0xFFFFFFFFull : d);
            lo = d < lo ? d : lo;
            hi = d > hi ? d : hi;
            sum += d;
        }
        r->iterations_done = i + 1;
    }

    // The last token's copy must have read the staging word before the kernel returns.
    if (a.transport == LatencyTransport::Idma) {
        for (uint32_t p = 0; !cce_noc::copies_done() && p < a.polls; p++) {
        }
    }
    r->cycles[lat::kMin] = lo;
    r->cycles[lat::kMax] = hi;
    r->cycles[lat::kSum] = sum;
    finish(r, kStatusDone);
}
