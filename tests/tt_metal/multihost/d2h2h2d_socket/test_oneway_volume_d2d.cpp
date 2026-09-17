// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// THE D2DSocket VARIANT OF test_oneway_volume.cpp, AND IT MUST STAY EQUIVALENT TO IT.
//
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/sockets/internal/host_clock.hpp>
#include <tt-metalium/experimental/sockets/internal/host_deliver.hpp>
#include <tt-metalium/experimental/sockets/internal/host_region.hpp>
#include <tt-metalium/experimental/sockets/internal/host_scan.hpp>
#include <tt-metalium/experimental/sockets/d2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h2h2d_socket.hpp>

#if !defined(TT_METAL_HOST_BRIDGE)
#error "this test needs TT_METAL_HOST_BRIDGE: D2H2H2DSocket IS the middle hop, and it has no transport-less form."
#endif
#include <tt-metalium/experimental/sockets/internal/host_stats.hpp>
#include <tt-metalium/experimental/sockets/internal/host_transport.hpp>
#include <tt-metalium/experimental/sockets/internal/host_uva.hpp>
#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>
#include <tt-metalium/distributed_context.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal::experimental;

namespace {

const char* kProg = "test";

struct Options {
    int device_id = -1;
    uint32_t cores = 4;
    uint32_t bytes = 4096;
    // 4096 is both the default and a legitimate payload, so the steady block below cannot read
    // `bytes == 4096` as "unset" the way it reads `volume == 0`.
    bool bytes_set = false;
    uint32_t iters = 16;
    uint64_t volume = 0;
    bool steady = false;
    uint32_t steady_pct = 10;
    bool h2d_socket = true;
    uint32_t l1_lo = 0, l1_hi = 0, l1_signal = 0, l1_completion = 0, l1_stop = 0, l1_dest_word = 0;
    uint32_t warmup = 0;
    uint32_t workers = 0;
    bool pin = true;

    uint32_t send_window = 0;
    bool send_blocking = false;

    uint32_t host_ident = 0;
    uint32_t host_num = 1;
    uint32_t chips_per_host = 1;
    uint32_t chip = 0;

    bool use_transport = false;
    bool same_host = false;
    bool measure_retire = false;
    bool measure_credit = false;
    // Every core sends and receives. Needs delivery to own its L1 buffer, halving the largest
    // payload that fits. Off: rank 0 sends, the rest receive, one shared buffer per core.
    bool bidir = false;
    double ns_per_cycle = 0.0;

    std::string csv;
    std::string tag;

    bool csv_append = false;
};

void usage() {
    std::cout <<
        R"(test -- the D2H2H2DSocket path driven through D2DSocket.

Same flags, CSV schema and kernel as test_oneway_volume, so rows from the two are directly
comparable. Use --tag to tell them apart in one CSV.

MODE
  --device <umd id>        real run: Tensix cores push into host arenas

SHAPE
  --cores N                cores in use (default 4). Each costs 3 MiB of pinned arena.
  --bytes N                payload per message (default 4096, max 1572864).
                           --steady raises the DEFAULT to 14336; passing --bytes always
                           wins and the substitution is printed when it happens.
  --iters N                messages per core (default 16)
  --volume N[K|M|G]        total traffic target; derives --iters
  --steady N               discard the first N%% of traffic before recording
  --warmup N               iterations to discard before recording
  --workers N              scan threads (default: one per CPU, capped at --cores)
  --send-window N          cap RMAs in flight across the sender (default: --cores).
                           Refused above --cores.
  --send-blocking          post-and-wait, one in flight. NOT --send-window 1, which spins.
  --no-pin                 do not pin worker threads to CPUs
  --bidir                  every core on every host sends AND receives, in a ring
                           (host h -> h+1). Delivery gets its own L1 buffer, so the
                           largest --bytes that fits is halved.

TOPOLOGY
  Host identity is the MPI rank and world size, not configurable.
  --chips-per-host N       selector slot stride (default 1)
  --chip N                 which chip on this host (default 0)

HOST-TO-HOST
  There is no bootstrap to configure: connect_mesh() builds one endpoint per peer rank and
  takes identity from DistributedContext.
  --same-host              both processes on one box: skip clock sync
  --measure-credit         sample the credit round trip: h2h:credit-raw, h2h:net,
                           h2h:payload-at-peer. Only a LATENCY at --send-window 1. Costs
                           sender-thread work, so the run's bandwidth is not quotable.
  --measure-retire         time each payload write from POST to COMPLETION, reported as
                           diag:h2h-retire. Nothing waits. Needs more than one rank.
                           The row pools payload retires with 40-byte notice retires, so its
                           mean is the per-op cost of neither, and it is POST to LOCAL
                           completion under MPI -- not the transfer. See flush().

  Delivery is always H2DSocket DEVICE_PULL; there is no flag. Ring aliasing puts the peer's
  RMA straight into the ring, so the payload crosses host RAM once.

OUTPUT
  --csv FILE               append per-hop rows
  --csv-append             append instead of truncating
  --tag STR                label for the CSV rows

ENVIRONMENT (applied first; any flag overrides)
  TT_RDMA_CHIPS_PER_HOST   same as --chips-per-host
)";
}

void apply_env(Options& o) {
    auto u32 = [](const char* name, uint32_t& slot) {
        if (const char* v = std::getenv(name)) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(v, &end, 10);
            if (end && *end == '\0') {
                slot = static_cast<uint32_t>(parsed);
            } else {
                std::cerr << "warning: " << name << "='" << v << "' is not a number, ignoring\n";
            }
        }
    };
    u32("TT_RDMA_CHIPS_PER_HOST", o.chips_per_host);
}

void resolve_identity(Options& o) {
    namespace mh = tt::tt_metal::distributed::multihost;
    if (!mh::DistributedContext::is_initialized()) {
        std::cerr << "error: the distributed context is not initialized. Launch under mpirun.\n";
        std::exit(2);
    }
    const auto& ctx = mh::DistributedContext::get_current_world();
    o.host_ident = static_cast<uint32_t>(*ctx->rank());
    o.host_num = static_cast<uint32_t>(*ctx->size());
    o.use_transport = o.host_num > 1;
}

bool parse(int argc, char** argv, Options& o) {
    apply_env(o);
    auto number = [&](int& i, const char* what) -> long long {
        const std::string flag = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "error: " << flag << " needs " << what << "\n";
            std::exit(2);
        }
        const std::string v = argv[++i];
        try {
            size_t used = 0;
            const long long n = std::stoll(v, &used);
            if (used != v.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return n;
        } catch (const std::exception&) {
            std::cerr << "error: " << flag << " needs " << what << ", got '" << v << "'\n";
            std::exit(2);
        }
    };
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "error: " << argv[i] << " needs a value\n";
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--device") { o.device_id = std::stoi(next(i)); }
        else if (a == "--cores") { o.cores = std::stoul(next(i)); }
        else if (a == "--bidir") { o.bidir = true; }
        else if (a == "--bytes") { o.bytes = std::stoul(next(i)); o.bytes_set = true; }
        else if (a == "--iters") { o.iters = std::stoul(next(i)); }
        else if (a == "--warmup") { o.warmup = std::stoul(next(i)); }
        else if (a == "--steady") {
            o.steady = true;
            // OPTIONAL ARGUMENT. `--steady` alone keeps the historical 10%; `--steady 25`
            // sets it. Peeked rather than consumed with next(), because --steady has always
            // been a bare flag and swallowing the following token would turn every existing
            // `--steady --measure-retire` into a parse error.
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                o.steady_pct = static_cast<uint32_t>(number(i, "a percentage 0..99"));
                if (o.steady_pct > 99) {
                    std::cerr << "error: --steady percentage must be 0..99 (it is the share of "
                                 "--volume discarded before the counters start)\n";
                    return false;
                }
            }
        }
        else if (a == "--volume") {
            std::string v = next(i);
            uint64_t mult = 1;
            if (!v.empty()) {
                const char c = v.back();
                if (c == 'k' || c == 'K') { mult = 1024ull; v.pop_back(); }
                else if (c == 'm' || c == 'M') { mult = 1024ull * 1024; v.pop_back(); }
                else if (c == 'g' || c == 'G') { mult = 1024ull * 1024 * 1024; v.pop_back(); }
            }
            o.volume = std::stoull(v) * mult;
        }
        else if (a == "--workers") { o.workers = std::stoul(next(i)); }
        else if (a == "--no-pin") { o.pin = false; }
        // The command line wins over the environment, same precedence as every other pair.
        else if (a == "--send-window") {
            const unsigned long n = std::stoul(next(i));
            if (n == 0) {
                std::cerr << "error: --send-window must be positive; omit it for the default "
                             "(cores-in-use)\n";
                std::exit(2);
            }
            o.send_window = static_cast<uint32_t>(n);
        }
        else if (a == "--send-blocking") { o.send_blocking = true; }
        else if (a == "--chips-per-host") { o.chips_per_host = std::stoul(next(i)); }
        else if (a == "--chip") { o.chip = std::stoul(next(i)); }
        else if (a == "--same-host") { o.same_host = true; }
        else if (a == "--measure-retire") { o.measure_retire = true; }
        else if (a == "--measure-credit") { o.measure_credit = true; }
        else if (a == "--csv") { o.csv = next(i); }
        else if (a == "--csv-append") { o.csv_append = true; }
        else if (a == "--tag") { o.tag = next(i); }
        else { std::cerr << "error: unknown flag " << a << "\n"; usage(); return false; }
    }

    if (o.bytes == 0 || o.bytes > kArenaBytes) {
        std::cerr << "error: --bytes must be 1.." << kArenaBytes << " (one arena)\n";
        return false;
    }
    if (o.cores == 0 || o.cores > kProvisionedCores) {
        std::cerr << "error: --cores must be 1.." << kProvisionedCores << "\n";
        return false;
    }
    if (o.host_ident >= o.host_num) {
        // Not a flag error: both numbers come from the communicator, so this is a broken MPI
        // world rather than something the caller typed.
        std::cerr << "error: MPI rank " << o.host_ident << " is not < world size " << o.host_num
                  << ". Identity comes from the communicator, not from a flag.\n";
        return false;
    }
    if (o.measure_retire && !o.use_transport) {
        std::cerr << "error: --measure-retire needs a transport, which means a job of TWO OR MORE\n"
                     "  MPI RANKS -- launch with `mpirun -n 2`. This process is rank " << o.host_ident
                  << " of " << o.host_num
                  << ".\n"
                     "  Locally routed messages are delivered by memcpy, so there is no posted\n"
                     "  operation to time.\n";
        return false;
    }
    // One L1 buffer per core at a shared address means a core cannot be both a source and a
    // destination, so senders and receivers live on different hosts: rank 0 sends, the rest
    // receive. At three or more ranks that is the fan-out deal below.
    if (o.host_num < 2) {
        std::cerr << "error: --host-num must be at least 2 -- the path is chip->host->host->chip.\n"
                     "This process is rank " << o.host_ident << " of " << o.host_num
                  << ". Launch under mpirun with two or more ranks.\n";
        return false;
    }
    if (o.host_num > kMaxHosts) {
        std::cerr << "error: --host-num " << o.host_num << " exceeds the " << kMaxHosts
                  << " hosts the region layout can address -- a credit is indexed by the\n"
                     "receiver's host id inside one 64 B register line. See kMaxHosts.\n";
        return false;
    }
    if (o.device_id < 0) {
        std::cerr << "error: no mode -- pass --device N\n";
        return false;
    }
    if (o.steady) {
        // --steady changes what two other options default to, and reports both: a number
        // quietly becoming a different number is the failure this tree keeps producing. 14336
        // is a chosen operating point, not a layout bound -- kArenaBytes / 14336 is not integral.
        if (!o.bytes_set) {
            o.bytes = 14336;
            std::cout << "  --steady  =>  --bytes " << o.bytes
                      << " (the steady default; pass --bytes to override)\n";
        }
        if (o.volume == 0) {
            o.volume = 1ull << 30;
            std::cout << "  --steady  =>  --volume " << (o.volume >> 20)
                      << " MiB (the steady default; pass --volume to override)\n";
        }
    }
    if (o.volume > 0) {
        const uint64_t per_iter = static_cast<uint64_t>(o.cores) * o.bytes;
        const uint64_t measured = std::max<uint64_t>(1, o.volume / std::max<uint64_t>(1, per_iter));

        uint64_t total = measured;
        if (o.warmup > 0) {
            total = measured + o.warmup;  // explicit --warmup: pad by exactly what is discarded
        } else if (o.steady && o.steady_pct > 0 && o.steady_pct < 100) {
            const uint64_t keep = 100u - o.steady_pct;
            total = (measured * 100u + keep - 1u) / keep;
            while (total - (total * o.steady_pct) / 100u < measured) {
                ++total;
            }
        }
        o.iters = static_cast<uint32_t>(total);
        std::cout << "  --volume " << (o.volume >> 20) << " MiB over " << o.cores << " cores x " << o.bytes
                  << " B  =>  --iters " << o.iters << "  (" << ((per_iter * total) >> 20) << " MiB moved, "
                  << ((per_iter * measured) >> 20) << " MiB measured)\n";
    }
    if (o.steady && o.warmup == 0) {
        const uint64_t discard = (static_cast<uint64_t>(o.iters) * o.steady_pct) / 100u;
        o.warmup = static_cast<uint32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(discard, o.iters > 1 ? o.iters - 1 : 0)));
        // "of N MiB" is the volume MOVED, not --volume: the two differ now that the warmup is
        // padded on rather than carved out, and printing --volume here would say the counters
        // start after a share of a total that is not the one being sent.
        const uint64_t moved = static_cast<uint64_t>(o.iters) * o.cores * o.bytes;
        std::cout << "  --steady " << o.steady_pct << "%  =>  counters start after "
                  << ((static_cast<uint64_t>(o.warmup) * o.cores * o.bytes) >> 20) << " MiB of "
                  << (moved >> 20) << " MiB moved (--warmup " << o.warmup << ")\n";
    }
    if (o.warmup >= o.iters) {
        std::cerr << "error: --warmup " << o.warmup << " leaves no recorded iterations (--iters " << o.iters
                  << ")\n";
        return false;
    }

    if (o.bytes < kPayloadHeaderBytes) {
        std::cerr << "error: --bytes " << o.bytes << " is smaller than the " << kPayloadHeaderBytes
                  << " B payload header (iteration + destination selector)\n";
        return false;
    }

    if (o.send_blocking && o.send_window != 0 && o.send_window != 1) {
        std::cerr << "error: --send-blocking implies a window of 1, but --send-window "
                  << o.send_window << " was given.\n"
                  << "       The window caps CONCURRENCY, the shape decides whether the sender "
                     "parks or spins. Pick one.\n";
        return false;
    }
    if (o.send_window > o.cores) {
        std::cerr << "error: --send-window " << o.send_window << " exceeds --cores " << o.cores
                  << ".\n"
                  << "       A destination core has ONE RX control word, so the window is across "
                     "cores, never within one.\n";
        return false;
    }

    // Tagged distinctly by default. Two rows in one CSV that differ only in their numbers
    // and not in their label is how a comparison gets read backwards.
    if (o.tag.empty()) {
        o.tag = "socket-device";
        if (o.steady) {
            o.tag += "-steady";
        }
        // The shape, because the tag is how CSV rows are told apart.
        o.tag += o.bidir ? "-bidir" : "-ow";
    }
    return true;
}


// the fan-out deal. Rank 0 sends, ranks 1..N-1 receive, core i goes to core i on host
// 1 + (i % (N - 1)): every destination core gets one source core on one source host, and no
// core is both. At two hosts it is `dest_host = 1`. host_num >= 2 is refused in validate().
uint32_t fanout_dest_host(uint32_t core, uint32_t host_num) { return 1u + (core % (host_num - 1u)); }

// the ring, used by --bidir. Host h sends to h+1, so a destination host takes traffic from
// exactly one predecessor and every destination core keeps one source host -- the invariant
// D2H2H2DSocket latches. Core i sends to core i on h+1 and receives from core i on h-1.
uint32_t ring_dest_host(uint32_t host_ident, uint32_t host_num) { return (host_ident + 1u) % host_num; }

// counted, not divided: cores / (host_num - 1) drops the remainder, and a receiver owed one
// message too many times out on a run that delivered everything it was sent.
uint32_t fanout_cores_for(uint32_t host, uint32_t cores, uint32_t host_num) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < cores; ++i) {
        if (fanout_dest_host(i, host_num) == host) {
            ++n;
        }
    }
    return n;
}

// where core i's messages GO, and whether this rank receives into core i. The two must agree
// or a receiver waits for a core nobody sends to; both derive from the same flag.
uint32_t dest_host_for(uint32_t core, const Options& o) {
    return o.bidir ? ring_dest_host(o.host_ident, o.host_num) : fanout_dest_host(core, o.host_num);
}

bool receives_into(uint32_t core, const Options& o) {
    if (o.bidir) {
        return true;  // ring: every core receives from its predecessor
    }
    return o.host_ident != 0 && fanout_dest_host(core, o.host_num) == o.host_ident;
}

uint32_t rx_cores_for(const Options& o) {
    if (o.bidir) {
        return o.cores;
    }
    return o.host_ident == 0 ? 0u : fanout_cores_for(o.host_ident, o.cores, o.host_num);
}

bool verify_delivery(Deliverer& deliverer, const Options& o, std::string& detail) {
    const uint32_t last = o.iters - 1;
    uint32_t checked = 0;
    for (uint32_t core = 0; core < o.cores; ++core) {
        // Only the cores something is sent to: an unsent core's L1 holds the seed pattern,
        // which passes the byte checks below. Under --bidir that is every core.
        if (!receives_into(core, o)) {
            continue;
        }
        ++checked;
        const std::vector<uint8_t> got = deliverer.read_payload(core, o.bytes, 0u);
        if (got.size() < o.bytes) {
            detail = "core " + std::to_string(core) + ": short read from L1";
            return false;
        }
        uint32_t stamp = 0;
        std::memcpy(&stamp, got.data() + kPayloadStampOffset, sizeof(stamp));

        // One sender per destination at depth 1, so iters-1 is the only acceptable stamp.
        // See test_oneway_volume.cpp for why a wider window would need more than one source.
        if (stamp != last) {
            std::ostringstream m;
            m << "core " << core << " L1 holds iteration stamp " << stamp << ", expected " << last;
            detail = m.str();
            return false;
        }

        {
            uint32_t landed = 0;
            std::memcpy(&landed, got.data() + kPayloadDestOffset, sizeof(landed));
            const uint32_t mine =
                t6_global_selector(o.host_ident, o.chip, core, o.chips_per_host);
            if (landed != mine) {
                std::ostringstream m;
                m << "core " << core << " received a message addressed to selector " << landed
                  << " (host " << t6_selector_host(landed, o.chips_per_host) << " core "
                  << t6_selector_core(landed) << "), but this is selector " << mine
                  << " -- the payload was delivered to the wrong core";
                detail = m.str();
                return false;
            }
        }

        const uint32_t probes[] = {kPayloadHeaderBytes, o.bytes / 2, o.bytes - 1};
        bool have_ref = false;
        uint8_t ref = 0;
        for (uint32_t off : probes) {
            if (off >= o.bytes || off < kPayloadHeaderBytes) {
                continue;
            }
            const uint8_t b = got[off];
            if (b < 0x40 || b >= static_cast<uint8_t>(0x40 + o.cores)) {
                std::ostringstream m;
                m << "core " << core << " L1 byte " << off << " = 0x" << std::hex << int(b)
                  << " is no sender's pattern (0x40 .. 0x" << int(0x40 + o.cores - 1) << ")";
                detail = m.str();
                return false;
            }
            if (!have_ref) {
                ref = b;
                have_ref = true;
            } else if (b != ref) {
                std::ostringstream m;
                m << "core " << core << " L1 byte " << off << " = 0x" << std::hex << int(b)
                  << " but an earlier byte read 0x" << int(ref)
                  << " -- one payload holding two senders' bytes";
                detail = m.str();
                return false;
            }
        }
    }
    // A verifier that checked nothing must not answer "verified". run_common refuses an empty
    // rank earlier; this is the second guard on the same vacuous pass.
    if (checked == 0) {
        detail = "no cores on this rank were dealt a sender, so there is nothing here to verify";
        return false;
    }
    return true;
}

int run_common(D2DSocket& sock, Options& o, const std::string& provider_label_str,
               const std::function<void()>& after_open = {}) {
    HostRegion& region = sock.region();
    Transport* const transport = sock.primary_transport();
    Deliverer* const deliverer = sock.deliverer();
    const ClockSync& clock = sock.clock();

    std::cout << "  socket        D2DSocket -> D2H2H2DSocket\n";

    // What the sender owes vs what this receiver is owed. They differ once the deal spreads
    // the sender's cores over more than one receiver.
    const uint64_t msgs_sent = static_cast<uint64_t>(o.cores) * o.iters;

    // Under --bidir every rank is both. The socket services both directions on the same
    // threads regardless; it is one shared L1 buffer per core that forbids it, not the socket.
    const bool tx_side = o.bidir || o.host_ident == 0;
    const bool rx_side = o.bidir || o.host_ident != 0;
    const uint32_t my_cores = rx_cores_for(o);
    const uint64_t msgs_recv = static_cast<uint64_t>(my_cores) * o.iters;
    if (o.bidir) {
        std::cout << "  bidir       delivery has its own L1 buffer; every core SENDS and RECEIVES\n"
                  << "  ring        host " << o.host_ident << " -> host "
                  << ring_dest_host(o.host_ident, o.host_num) << ", core i to core i\n";
    } else {
        std::cout << "  symmetric   one L1 buffer per core, shared address; this side "
                  << (tx_side ? "SENDS only" : "RECEIVES only") << "\n";
    }
    if (!o.bidir && o.host_num > 2) {
        std::cout << "  fan-out     rank 0 -> ranks 1.." << (o.host_num - 1)
                  << ", core i to core i on host 1 + (i % " << (o.host_num - 1) << ")\n";
        if (rx_side) {
            std::cout << "              this rank is dealt " << my_cores << " of " << o.cores
                      << " cores => " << msgs_recv << " messages\n";
        }
        if (rx_side && my_cores == 0) {
            std::cerr << "error: the deal gave this rank no cores -- --cores " << o.cores << " over "
                      << (o.host_num - 1) << " receivers leaves rank " << o.host_ident
                      << " with nothing, and a rank that receives nothing cannot verify anything.\n"
                         "  Use at least --cores "
                      << (o.host_num - 1) << ".\n";
            return 2;
        }
    }

    std::string oerr;
    if (!sock.open(oerr)) {
        std::cerr << "socket open failed: " << oerr << "\n";
        return 1;
    }
    if (after_open) {
        after_open();
    }

    auto wait_for = [&](const std::atomic<uint64_t>& c, uint64_t want, uint64_t budget_ns, const char* what) {
        uint64_t seen = c.load(std::memory_order_acquire);
        uint64_t dl = now_ns() + budget_ns;
        uint64_t next_report = now_ns() + 5ull * 1000 * 1000 * 1000;
        while (c.load(std::memory_order_acquire) < want && now_ns() < dl) {
            if (sock.transport_failed()) {
                std::cerr << "  giving up on " << what << ": the transport has faulted\n";
                return;
            }
            // A refused message is not recoverable either, and the reason is on the peer.
            if (sock.peer_refused()) {
                std::cerr << "  giving up on " << what << ": a peer refused a message\n";
                return;
            }
            const uint64_t now_count = c.load(std::memory_order_acquire);
            if (now_count != seen) {
                seen = now_count;
                dl = now_ns() + budget_ns;
            }
            if (now_ns() > next_report) {
                std::cerr << "  waiting for " << what << ": " << c.load() << " of " << want
                          << sock.stall_dump(what) << std::flush;
                next_report = now_ns() + 5ull * 1000 * 1000 * 1000;
            }
            std::this_thread::yield();
        }
    };

    const uint64_t sent_target = tx_side ? msgs_sent : 0;
    if (sent_target) {
        wait_for(sock.counters().tx_done, sent_target, 30ull * 1000 * 1000 * 1000, "tx_done");
    }

    // The sender's total on both sides: the barrier completes only when every rank has.
    const uint32_t barrier_ms = static_cast<uint32_t>(60000 + std::min<uint64_t>(msgs_sent, 600000));
    if (transport != nullptr) {
        for (Transport* t : sock.peers_for_barrier()) {
            if (const std::string be = t->barrier(); !be.empty()) {
                std::cerr << "  end-of-send barrier failed (host " << t->peer().host_id << "): " << be
                          << "\n";
            }
        }
    }

    if (rx_side) {
        wait_for(sock.counters().delivered, msgs_recv, 15ull * 1000 * 1000 * 1000, "delivered");
    }

    // A sender must drain its credits before stamping, or its interval ends at the last post
    // and the bandwidth is over-reported. Keyed on tx_side, not on "not a receiver", because
    // under --bidir every rank is both.
    if (tx_side && transport != nullptr && o.host_num > 1) {
        uint32_t slow_core = 0;
        if (!drain_credits(region, o.cores, o.iters, static_cast<uint64_t>(barrier_ms) * 1000000ull,
                           slow_core)) {
            std::cerr << "  end-of-run credit drain timed out on core " << slow_core << " (wanted "
                      << o.iters
                      << "); the bandwidth interval is bounded by posts, not arrivals -- do not quote it\n";
        }
    }
    sock.stamp_timed_end();

    // Before stop(): our workers and the progress thread must still be live while the peer
    // drains.
    if (transport != nullptr) {
        for (Transport* t : sock.peers_for_barrier()) {
            if (const std::string be = t->barrier(); !be.empty()) {
                std::cerr << "  end-of-receive barrier failed (host " << t->peer().host_id << "): " << be
                          << "\n";
            }
        }
    }

    sock.stop();

    RunStats stats = sock.collect();
    stats.payload_bytes = o.bytes;
    // the cores this rank used, not --cores. The bandwidth cell divides a measured byte counter
    // so it is right either way, but `cores` is what a reader multiplies out to cross-check it.
    stats.cores = rx_side ? my_cores : o.cores;
    stats.iters = o.iters;
    stats.provider = provider_label_str;
    stats.mode = o.bidir ? "sym-bidir" : (tx_side ? "sym-tx" : "sym-rx");
    stats.host_clock_valid = clock.valid;
    stats.host_clock_uncertainty_ns = clock.uncertainty_ns;
    stats.device_clock_valid = o.ns_per_cycle > 0.0;
    stats.device_clock_uncertainty_ns = 0;

    stats.run_id = make_run_id();
    stats.run_started_utc = utc_now_iso();

    auto stats_role_fn = [transport, &o]() {
        if (transport == nullptr) {
            return "local";
        }
        if (o.host_ident == 0) {
            return "server";
        }
        return "peer";
    };

    stats.role = stats_role_fn();
    stats.host_ident = o.host_ident;
    stats.symmetric = true;
    stats.h2d = o.h2d_socket ? "socket" : "write";
    stats.tx_side = tx_side;
    stats.warmup = o.warmup;
    stats.warmup_applied = o.warmup > 0;
    stats.timed_iters = o.iters - o.warmup;
    stats.ns_per_cycle = o.ns_per_cycle;

    std::cout << format_table(stats);

    const SocketCounters& cn = sock.counters();
    std::cout << "\n=== routing and delivery ===\n\n";
    std::printf("  local     %llu\n", (unsigned long long)cn.routed_local.load());
    std::printf("  remote    %llu\n", (unsigned long long)cn.routed_remote.load());
    std::printf("  nowhere   %llu   (selector named no configured host)\n",
                (unsigned long long)cn.routed_nowhere.load());
    std::printf("  delivered %llu   (bytes written into a Tensix L1)\n", (unsigned long long)cn.delivered.load());
    {
        const unsigned long long fc = cn.flush_calls.load();
        const unsigned long long fs = cn.flush_slots.load();
        if (fc > 0) {
            std::printf("  flush     %llu calls, %llu payloads  =>  %.2f payloads per flush\n", fc, fs,
                        (double)fs / (double)fc);
        }
    }
    std::printf("  errors    %llu\n", (unsigned long long)cn.errors.load());
    if (transport != nullptr) {
        const TransportDiag d = transport->diag();
        std::printf(
            "  transport posted=%llu retired=%llu injected=%llu outstanding=%llu unmatched=%llu "
            "abandoned=%llu\n",
            (unsigned long long)d.posted, (unsigned long long)d.retired, (unsigned long long)d.injected,
            (unsigned long long)d.outstanding, (unsigned long long)d.unmatched, (unsigned long long)d.abandoned);
        if (!d.last_error.empty()) {
            std::printf("  transport last CQ error: %s\n", d.last_error.c_str());
        }
    }
    if (o.warmup > 0) {
        std::printf("  warmup    %u of %u iterations discarded (%s)\n", o.warmup, o.iters,
                    "NOT APPLIED: the device producer records from message 1");
    }
    const std::string fe = sock.first_error();
    if (!fe.empty()) {
        std::cout << "  first error: " << fe << "\n";
    }
    std::cout << "  clock: " << clock.describe() << "\n";
    if (deliverer) {
        std::cout << "  deliverer: " << deliverer->describe() << "\n";
    }

    if (!o.csv.empty()) {
        std::string path = o.csv;
        bool truncate = !o.csv_append;
        if (!truncate) {
            if (const std::string e = csv_schema_error(path, basic_csv_header()); !e.empty()) {
                path += ".new";
                std::cerr << "  " << e << "\n  writing to " << path << " instead\n";
            }
        }
        const bool fresh = truncate || !std::ifstream(path).good();
        std::ofstream f(path, truncate ? std::ios::trunc : std::ios::app);
        if (fresh) {
            f << basic_csv_header();
        }
        f << format_basic_csv(stats, o.tag);
        std::cout << "  csv " << (fresh ? "written to " : "appended to ") << path << "\n";
    }

    bool ok = cn.errors.load() == 0;
    std::ostringstream why;
    if (cn.errors.load()) {
        why << cn.errors.load() << " service errors. ";
    }
    const uint64_t want_delivered = !rx_side ? 0 : msgs_recv;
    if (cn.delivered.load() < want_delivered) {
        ok = false;
        why << "delivered " << cn.delivered.load() << " of " << want_delivered << " into L1. ";
    }
    if (tx_side && cn.tx_done.load() < msgs_sent) {
        ok = false;
        why << "sent " << cn.tx_done.load() << " of " << msgs_sent << " messages. ";
    }
    if (ok && cn.routed_remote.load() == 0 && deliverer) {
        std::string detail;
        if (!verify_delivery(*deliverer, o, detail)) {
            ok = false;
            why << "L1 payload check failed: " << detail << ". ";
        }
    } else if (cn.routed_remote.load() > 0) {
        std::cout << "\n  NOTE: " << cn.routed_remote.load()
                  << " messages were routed to a peer. Their arrival is the RECEIVING\n"
                     "        rank's to verify; this process cannot witness it and does not\n"
                     "        claim to.\n";
    }

    std::cout << "\n" << (ok ? "PASS" : "FAIL") << (ok ? "" : ": " + why.str()) << "\n\n";
    return ok ? 0 : 1;
}

int run_device(Options& o) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    std::cout << "\n=== " << kProg << " on device " << o.device_id << " ===\n\n";

    namespace mh = tt::tt_metal::distributed::multihost;
    // BY VALUE, NOT BY REFERENCE. get_current_world() returns a reference to the global
    // current_world_; binding a reference here would make `world` alias whatever
    // set_current_world(solo) installs below, and the restore at set_current_world(world) would
    // assign it to itself -- leaving the solo (size 1) context current.
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) -- the copy IS the point.
    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    {
        const mh::ContextPtr solo =
            world->split(mh::Color{static_cast<int>(o.host_ident)}, mh::Key{0});
        mh::DistributedContext::set_current_world(solo);
    }

    auto mesh_device = MeshDevice::create_unit_mesh(o.device_id);
    IDevice* device = mesh_device->get_devices().front();

    mh::DistributedContext::set_current_world(world);

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const HostRegion::Grid g{static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y)};
    if (o.cores > g.width * g.height) {
        std::cerr << "error: --cores " << o.cores << " exceeds the " << (g.width * g.height) << " cores on a "
                  << g.width << "x" << g.height << " grid\n";
        return 2;
    }

    const PinLimits limits = query_pin_limits(mesh_device);
    std::printf("  grid          %ux%u, %u cores in use\n", g.width, g.height, o.cores);
    std::printf("  memlock       %s\n",
                limits.rlimit_memlock == UINT64_MAX
                    ? "unlimited"
                    : (std::to_string(limits.rlimit_memlock >> 20) + " MiB").c_str());
    std::printf("  want pinned   %llu MiB\n", (unsigned long long)(pinned_bytes_for(o.cores) >> 20));

    D2DSocketConfig dc;
    dc.host_ident = o.host_ident;
    dc.host_num = o.host_num;
    dc.chips_per_host = o.chips_per_host;
    dc.chip = o.chip;
    dc.cores = o.cores;
    dc.grid_width = g.width;
    dc.grid_height = g.height;
    dc.payload_bytes = o.bytes;
    dc.workers = o.workers;
    dc.send_window = o.send_window;
    dc.send_blocking = o.send_blocking;
    dc.pin = o.pin;
    dc.bidirectional = o.bidir;

    std::string serr;
    std::unique_ptr<D2DSocket> sock = D2DSocket::create(mesh_device, device, dc, serr);
    if (!sock) {
        std::cerr << "socket bringup failed: " << serr << "\n";
        return 1;
    }

    // this program IS A measurement, so it always opts in. Collective and before open(), so
    // every rank reaches it here. --iters stays out: it bounds this program's kernels, and the
    // socket has no use for it.
    D2DMeasurementConfig mc;
    mc.warmup = o.warmup;
    mc.measure_retire = o.measure_retire;
    mc.measure_credit = o.measure_credit;
    mc.ns_per_cycle_override = o.ns_per_cycle;
    mc.same_host = o.same_host;
    if (const std::string me = sock->measure(mc); !me.empty()) {
        std::cerr << "measurement setup failed: " << me << "\n";
        return 1;
    }
    const L1Map& l1 = sock->l1();
    HostRegion& region = sock->region();

    std::cout << "  deliverer     " << sock->deliverer_describe() << "\n";
    std::printf("  region        base %p, %llu MiB pinned\n", static_cast<void*>(region.base()),
                (unsigned long long)(region.pinned_bytes() >> 20));
    std::printf("  device view   pcie_xy_enc 0x%08X, io_base 0x%016llx\n", region.device().pcie_xy_enc,
                (unsigned long long)region.device().io_base);
    std::cout << "  connected: " << sock->peer_count() << " peer(s)\n";
    std::cout << "  transport: " << sock->transport_describe() << "\n";
    std::cout << "  clock: " << sock->clock().describe() << "\n";
    std::cout << "  l1 map        " << l1.describe() << "\n";

    // Stores are not supported (see host_deliver.cpp), so the opcode is always kOpSendUva
    // and the store destination is unused.
    const uint32_t kernel_opcode = static_cast<uint32_t>(kOpSendUva);
    const uint32_t store_dest_addr = 0;

    o.l1_lo = l1.payload_addr;
    o.l1_hi = l1.l1_size;
    o.l1_signal = l1.signal_addr;
    o.l1_completion = l1.completion_addr;
    o.l1_stop = l1.stop_addr;
    o.l1_dest_word = l1.dest_word_addr;

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < o.cores; ++i) {
        const CoreCoord c{i % g.width, i / g.width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }



    Program program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            // this program's kernel, SO this program's argument order. Every value comes from
            // the socket -- l1() and region().device() -- but the order belongs to whoever owns
            // the kernel file it has to match. test_oneway_volume.cpp builds it the same way.
            .compile_args = {
                region.device().pcie_xy_enc,
                static_cast<uint32_t>(region.device().io_base & 0xFFFFFFFFull),
                static_cast<uint32_t>(region.device().io_base >> 32),
                g.width,
                l1.payload_addr,
                l1.stage_addr,
                l1.signal_addr,
                o.bytes,
                (o.bidir || o.host_ident == 0) ? o.iters : 0u,
                kernel_opcode,
                // unconditionally: the kernel always writes elapsed_pack(), so register 2 is
                // packed whether the flag is set or not and the flag only tells the host to
                // unpack it. Unset, t6->host is wrong by 2^32x the probe cost once one runs.
                static_cast<uint32_t>(kFlagStamped | kFlagElapsedSplit),
                1u,  // await_completion: the kernel's control word is a single slot
                l1.completion_addr,
                0u,  // verify_landing -- this program has no --verify-landing
                0u,  // landing_addr -- never read; the probe branch is if constexpr'd away
            }});

    for (uint32_t i = 0; i < o.cores; ++i) {
        const uint32_t sel = t6_global_selector(dest_host_for(i, o), o.chip, i, o.chips_per_host);
        SetRuntimeArgs(program, kernel, core_list[i], {sel, store_dest_addr});
    }

    for (uint32_t i = 0; i < o.cores; ++i) {
        const uint32_t b = 0x40u + (i & 0x1F);
        std::vector<uint32_t> page(o.bytes / sizeof(uint32_t), b * 0x01010101u);
        tt::tt_metal::detail::WriteToDeviceL1(device, core_list[i], l1.payload_addr, page, tt::CoreType::WORKER);
    }

    if (o.h2d_socket) {
        const std::vector<uint32_t> cfg_addrs = sock->deliverer()->socket_config_addresses();
        if (cfg_addrs.size() != o.cores) {
            std::cerr << "H2D socket deliverer published " << cfg_addrs.size()
                      << " config addresses for " << o.cores << " cores\n";
            return 1;
        }
        auto recv_kernel = CreateKernel(
            program,
            TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_pull.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_1,
                .compile_args = {l1.deliver_addr, o.bytes, l1.signal_addr, 1u, l1.stop_addr,
                                 l1.dest_word_addr}});
        for (uint32_t i = 0; i < o.cores; ++i) {
            const uint32_t enabled = (o.bidir || o.host_ident != 0) ? 1u : 0u;
            SetRuntimeArgs(program, recv_kernel, core_list[i], {cfg_addrs[i], enabled});
        }
        if (const std::string e = sock->deliverer()->arm_receivers(); !e.empty()) {
            std::cerr << "H2D socket: " << e << "\n";
            return 1;
        }
    }

    if (sock->ns_per_cycle() > 0.0) {
        std::printf("  device clock  %.4f ns/cycle (%s)\n", sock->ns_per_cycle(),
                    sock->clock_rate_detail().c_str());
    } else {
        std::printf("  device clock  UNMEASURED (%s) -- stage t6->host will report no samples\n",
                    sock->clock_rate_detail().c_str());
    }
    o.ns_per_cycle = sock->ns_per_cycle();

    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(mesh_device->shape()), std::move(program));

    std::thread launcher;
    const int rc = run_common(*sock, o, o.use_transport ? "mpi-rma" : "none", [&] {
        launcher = std::thread([&] {
            EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
            Finish(mesh_device->mesh_command_queue());
        });
    });
    if (const std::string e = sock->deliverer()->stop_receivers(); !e.empty()) {
        std::cerr << "warning: could not stop the receiver kernels: " << e << "\n";
    }
    if (launcher.joinable()) {
        launcher.join();
    }
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    // The context must exist before identity can be read, and identity before parse() validates
    // a topology -- so this runs first.
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    Options o;
    resolve_identity(o);
    if (!parse(argc, argv, o)) {
        return 2;
    }
    return run_device(o);
}
