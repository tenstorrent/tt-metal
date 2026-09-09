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

#include "tt_metal/distributed/host_clock.hpp"
#include "tt_metal/distributed/host_deliver.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include "tt_metal/distributed/host_scan.hpp"
#include <tt-metalium/experimental/sockets/D2DSocket.hpp>
#include <tt-metalium/experimental/sockets/D2H2H2DSocket.hpp>

#if !defined(TT_METAL_HOST_BRIDGE)
#error "this test needs TT_METAL_HOST_BRIDGE: D2H2H2DSocket IS the middle hop, and it has no transport-less form."
#endif
#include "tt_metal/distributed/host_stats.hpp"
#include "tt_metal/distributed/host_transport.hpp"
#include "tt_metal/distributed/host_uva.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"
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
    bool layout = false;
    int device_id = -1;
    uint32_t cores = 4;
    uint32_t bytes = 4096;
    uint32_t iters = 16;
    uint64_t volume = 0;
    bool steady = false;
    uint32_t steady_pct = 10;
    bool h2d_socket = true;
    uint32_t dest_offset = 0;
    bool store = false;
    uint32_t l1_lo = 0, l1_hi = 0, l1_signal = 0, l1_completion = 0, l1_stop = 0, l1_dest_word = 0;
    std::string csv_rotate;
    std::string volume_csv;
    std::string trace_csv;
    bool volume_quiesce = false;
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
    bool roundtrip = false;
    double ns_per_cycle = 0.0;

    std::string csv;
    std::string tag;

    bool csv_append = false;
};

void usage() {
    std::cout <<
        R"(test -- t6_host_uva's path driven through D2H2H2DSocket (finalized/)

Same flags, same CSV schema and the same kernel as t6_host_uva, so rows from the two are
directly comparable. Use --tag to tell them apart in one CSV.

MODES
  --layout                 print the region layout and exit (no device, no peer)
  --device <umd id>        real run: Tensix cores push into host arenas

SHAPE
  --cores N                cores in use (default 4). Each costs 3 MiB of pinned arena.
  --bytes N                payload per message (default 4096, max 1572864)
  --iters N                messages per core (default 16)
  --volume N[K|M|G]        total traffic target; derives --iters
  --steady                 14 KiB chunks, 1 GiB of traffic, 10%% warmup discarded
  --warmup N               iterations to discard before recording
  --workers N              scan threads (default: one per CPU, capped at --cores)
  --trace-csv PATH         cumulative-volume time series
  --send-window N          cap RMAs in flight across the sender (default: --cores).
                           Refused above --cores.
  --send-blocking          post-and-wait, credit waited, one in flight -- the revert path
                           to the pre-2026-08-28 sender.
                           NOT the same as --send-window 1, which still spins.
  --no-pin                 do not pin worker threads to CPUs
  --oneway                 t6 -> host -> [remote host] -> t6      (the only shape)

                           (--roundtrip is NOT listed because it does not parse: no
                           case sets o.roundtrip true. The return half went with
                           libfabric -- see TODO_D2H2H2D.md P6.)

ENVIRONMENT (applied first; any command-line flag overrides)
  TT_RDMA_CHIPS_PER_HOST   same as --chips-per-host

  The sender knobs are FLAGS ONLY -- --send-window and --send-blocking. Their TT_HOST_UVA_*
  twins were a liability rather than a convenience: four pair runs on 2026-08-28 set
  TT_HOST_UVA_SEND_WINDOW against binaries that had no such variable and silently measured
  the default instead. One spelling cannot go unread.

  Host identity is NOT configurable: --host-ident and --host-num are the MPI rank and world
  size. Two sources for one fact is how the peer table and the communicator end up disagreeing.

TOPOLOGY
  (host identity is the MPI rank; see ENVIRONMENT above)
  --chips-per-host N       selector slot stride (default 1)
  --chip N                 which chip on this host (default 0)

HOST-TO-HOST
  (there is no bootstrap to configure. --server, --peer, --peers, --port, --provider and
   --bind-addr are GONE, removed 2026-09-03: the parser never accepted any of them, so
   every one of them failed the run with "unknown flag" while this text advertised it.
   They date from the sockets bootstrap, which connect_mesh() replaced -- it builds one
   endpoint per peer rank and takes identity from DistributedContext, so there is no
   address list, no port, no listen/connect asymmetry and no provider to name. The last
   of those is the one that cost a measurement: `provider` was a dead Options field, and
   the CSV's provider column comes from a hardcoded "mpi-rma" at the run_* call site, so
   csv files carry provider=mpi-rma no matter what the tag or the flag said.
   make_transport() returns MpiRmaTransport unconditionally -- see host_transport.cpp.)
  --same-host              both processes on one box: skip clock sync
  --measure-retire         time each payload write from POST to COMPLETION, reported as
                           diag:h2h-retire. Nothing waits. Needs more than one rank.
                           NOTE the row pools payload retires with 40-byte notice retires,
                           so its mean is the per-op cost of neither. It is also POST to
                           LOCAL completion under MPI, which is not the transfer: see
                           host_transport.cpp flush() and MEASURING-BANDWIDTH.md.
  --h2d socket             ACCEPTED AND REDUNDANT. Host-to-device delivery is always
                           tt-metal's H2DSocket in DEVICE_PULL: the device reads the payload
                           out of pinned host memory, and ring aliasing puts the peer's RMA
                           straight into that ring, so the payload crosses host RAM once.
                           `--h2d write` -- host CPU stores into L1 over the PCIe BAR -- is
                           REFUSED: it plateaus at 314 MB/s where this reaches 2577, and at
                           110 cores it trips UMD's MMIO per-op timeout mid-run.
  (there is no --deliver flag: delivery is H2DSocket DEVICE_PULL, see --h2d above. The
   old --deliver push/pull selected between this and an unimplemented bespoke-kernel pull
   whose only artefact is kernels/test_kernel_pull.cpp; keeping a flag whose sole other
   value was refused made the real pull path -- --h2d socket -- look unimplemented.)

OUTPUT
  --csv FILE               append per-hop rows
  --tag STR                label for the CSV rows (default: socket-*)

NOT HERE
  --hh-pingpong            no device and no register file, so it exercises nothing in
                           D2H2H2DSocket. Run it from t6_host_uva.

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
        else if (a == "--layout") { o.layout = true; }
        else if (a == "--device") { o.device_id = std::stoi(next(i)); }
        else if (a == "--cores") { o.cores = std::stoul(next(i)); }
        else if (a == "--bytes") { o.bytes = std::stoul(next(i)); }
        else if (a == "--iters") { o.iters = std::stoul(next(i)); }
        else if (a == "--warmup") { o.warmup = std::stoul(next(i)); }
        else if (a == "--steady") {
            o.steady = true;
            // OPTIONAL ARGUMENT. `--steady` alone keeps the historical 10%; `--steady 25`
            // sets it. Peeked rather than consumed with next(), because --steady has always
            // been a bare flag and swallowing the following token would turn every existing
            // `--steady --oneway` into a parse error.
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                o.steady_pct = static_cast<uint32_t>(number(i, "a percentage 0..99"));
                if (o.steady_pct > 99) {
                    std::cerr << "error: --steady percentage must be 0..99 (it is the share of "
                                 "--volume discarded before the counters start)\n";
                    return false;
                }
            }
        }
        else if (a == "--csv-rotate") { o.csv_rotate = next(i); }
        else if (a == "--volume-csv") { o.volume_csv = next(i); }
        else if (a == "--trace-csv") { o.trace_csv = next(i); }
        else if (a == "--volume-quiesce") { o.volume_quiesce = true; }
        else if (a == "--store") {
            o.store = true;
        }
        else if (a == "--dest-offset") {
            // Byte offset from payload_addr; the allocator base is added where it is known.
            // Implies --store: an offset with no store to carry it would be silently ignored.
            o.dest_offset = static_cast<uint32_t>(std::stoul(next(i), nullptr, 0));
            o.store = true;
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
        else if (a == "--oneway") { o.roundtrip = false; }
        else if (a == "--chips-per-host") { o.chips_per_host = std::stoul(next(i)); }
        else if (a == "--chip") { o.chip = std::stoul(next(i)); }
        else if (a == "--same-host") { o.same_host = true; }
        else if (a == "--measure-retire") { o.measure_retire = true; }
        else if (a == "--csv") { o.csv = next(i); }
        else if (a == "--csv-append") { o.csv_append = true; }
        else if (a == "--tag") { o.tag = next(i); }
        else { std::cerr << "error: unknown flag " << a << "\n"; usage(); return false; }
    }

    if (!o.csv_rotate.empty()) {
        return true;  // rotate-and-exit: no mode, no device, nothing below applies
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
        // world rather than something the caller typed. Naming the flags that used to set them
        // sends the reader looking for a spelling that no longer exists.
        std::cerr << "error: MPI rank " << o.host_ident << " is not < world size " << o.host_num
                  << ". Identity comes from the communicator, not from a flag.\n";
        return false;
    }
    if (o.measure_retire && !o.use_transport) {
        std::cerr << "error: --measure-retire needs a transport, which means a job of TWO MPI\n"
                     "  RANKS -- launch with `mpirun -n 2`. This process is rank " << o.host_ident
                  << " of " << o.host_num
                  << ".\n"
                     "  Locally routed messages are delivered by memcpy, so there is no posted\n"
                     "  operation to time.\n";
        return false;
    }
    // SYMMETRIC OPERATION IS NATIVE AND ASSUMED, so this is not conditional on a flag: the
    // path is chip->host->host->chip, exactly two hosts. With one host the sender and the
    // receiver would be the same core and the shared L1 buffer would have two writers.
    // D2H2H2DSocket::open() enforces the same thing; this refuses earlier, before anything
    // has been provisioned or connected.
    if (o.host_num != 2) {
        std::cerr << "error: --host-num must be 2. One L1 buffer per core at a shared address is "
                     "native\n"
                     "here, and that is only coherent when every destination is on the other host.\n";
        return false;
    }
    if (!o.layout && o.device_id < 0) {
        std::cerr << "error: pick a mode -- --layout or --device N\n";
        return false;
    }
    if (o.steady) {
        if (o.bytes == 4096) {
            o.bytes = 14336;
        }
        if (o.volume == 0) {
            o.volume = 1ull << 30;
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
        o.tag += "-ow";  // was `o.roundtrip ? "-rt" : "-ow"` -- o.roundtrip cannot be true
    }
    return true;
}

void print_layout(const Options& o) {
    std::cout << "\n=== " << kProg << " region layout (identical to t6_host_uva) ===\n\n";
    std::printf("  register bank      %u registers x %u B = %u B\n", kRegistersPerBank, kRegisterBytes, kBankBytes);
    std::printf("                     %u data (0..%u), control TX=%u RX=%u\n", kDataRegisters, kDataRegisters - 1,
                kCtrlTx, kCtrlRx);
    std::printf("  arena              %llu B TX + %llu B RX = %llu B per core\n",
                (unsigned long long)kArenaBytes, (unsigned long long)kArenaBytes,
                (unsigned long long)kArenaStride);
    std::printf("  provisioned cores  %u\n", kProvisionedCores);
    std::printf("  header             %llu B at offset 0\n", (unsigned long long)kHeaderBytes);
    std::printf("  bank array         %llu B at offset %llu\n", (unsigned long long)kBankArrayBytes,
                (unsigned long long)kHeaderBytes);
    std::printf("  arena array        at offset %llu (2 MiB aligned)\n", (unsigned long long)kArenaArrayOffset);
    std::printf("  full region        %llu B (%llu MiB)\n", (unsigned long long)kRegionBytesMax,
                (unsigned long long)(kRegionBytesMax >> 20));
    std::printf("  pinned for %-4u    %llu B (%llu MiB)\n", o.cores,
                (unsigned long long)pinned_bytes_for(o.cores), (unsigned long long)(pinned_bytes_for(o.cores) >> 20));

    std::cout << "\n  core   bank offset    TX arena       RX arena\n";
    std::cout << "  ---- ------------ -------------- --------------\n";
    const uint32_t show = std::min<uint32_t>(o.cores, 8);
    for (uint32_t c = 0; c < show; ++c) {
        std::printf("  %4u %12llu %14llu %14llu\n", c, (unsigned long long)bank_offset(c),
                    (unsigned long long)tx_arena_offset(c), (unsigned long long)rx_arena_offset(c));
    }
    if (o.cores > show) {
        std::printf("  ... %u more\n", o.cores - show);
    }

    const uint64_t sample = ctrl_encode(kOpSendUva, 0, 2, kFlagStamped, 7);
    std::cout << "\n=== control word ===\n\n";
    std::printf("  example (op=send_uva base=0 count=2 flags=stamped seq=7) = 0x%016llx\n",
                (unsigned long long)sample);
    std::printf("    magic    0x%04X   version %u   seq %u\n", ctrl_magic(sample), ctrl_version(sample),
                ctrl_sequence(sample));
    std::printf("    opcode   %u        base %u   count %u   flags 0x%llx\n", ctrl_opcode(sample),
                ctrl_base(sample), ctrl_count(sample), (unsigned long long)ctrl_flags(sample));
    std::printf("    validate -> %s\n", ctrl_verdict_name(ctrl_validate(sample)));
    std::printf("    a zeroed bank   -> %s\n", ctrl_verdict_name(ctrl_validate(0)));
    std::printf("    a legacy v2 word -> %s\n", ctrl_verdict_name(ctrl_validate(0x57A7ull << 48)));

    const HostTopology t{o.host_ident, o.host_num, o.chips_per_host};
    std::cout << "\n=== routing ===\n\n";
    std::printf("  topology ident=%u num=%u chips_per_host=%u -> %s\n", t.ident, t.num, t.chips_per_host,
                host_topology_ok(t) ? "ok" : "REJECTED");
    for (uint32_t h = 0; h < std::min<uint32_t>(o.host_num + 1, 4); ++h) {
        const uint64_t u = uva_encode(kRegionT6, t6_global_selector(h, o.chip, 3, o.chips_per_host), 0, 0);
        std::printf("  uva 0x%016llx -> host %u chip %u core %u : %s\n", (unsigned long long)u,
                    uva_t6_host(u, o.chips_per_host), uva_t6_chip(u, o.chips_per_host), uva_t6_core(u),
                    host_reach_name(uva_host_reach(u, t)));
    }
    std::cout << "\n";
}

bool verify_delivery(Deliverer& deliverer, const Options& o, std::string& detail) {
    const uint32_t last = o.iters - 1;
    const uint32_t verify_at = o.store ? (o.l1_lo + o.dest_offset) : 0u;
    for (uint32_t core = 0; core < o.cores; ++core) {
        const std::vector<uint8_t> got = deliverer.read_payload(core, o.bytes, verify_at);
        if (got.size() < o.bytes) {
            detail = "core " + std::to_string(core) + ": short read from L1";
            return false;
        }
        uint32_t stamp = 0;
        std::memcpy(&stamp, got.data() + kPayloadStampOffset, sizeof(stamp));

        const uint32_t oldest_ok = last;
        if (stamp < oldest_ok || stamp > last) {
            std::ostringstream m;
            m << "core " << core << " L1 holds iteration stamp " << stamp << ", expected ";
            if (oldest_ok == last) {
                m << last;
            } else {
                m << oldest_ok << ".." << last << " (any of the last " << o.cores
                  << " iterations -- with a rotating destination the sources are unordered "
                     "against each other)";
            }
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
    return true;
}

int run_common(D2DSocket& sock, Options& o, const std::string& provider_label_str,
               const std::function<void()>& after_open = {}) {
    HostRegion& region = sock.region();
    Transport* const transport = sock.primary_transport();
    Deliverer* const deliverer = sock.deliverer();
    const ClockSync& clock = sock.clock();

    if (sock.ladder().enabled) {
        std::cout << "  volume ladder " << sock.ladder().marks.size() << " checkpoints, " << o.bytes
                  << " B chunks over " << (sock.ladder().total_bytes >> 20) << " MiB recorded -> "
                  << o.volume_csv << "\n";
    }
    std::cout << "  socket        D2DSocket -> D2H2H2DSocket\n";

    const uint64_t msgs = static_cast<uint64_t>(o.cores) * o.iters;

    const bool tx_side = o.host_ident == 0;
    const bool rx_side = o.host_ident != 0;
    std::cout << "  symmetric   one L1 buffer per core, shared address; this side "
              << (tx_side ? "SENDS only" : "RECEIVES only") << "\n";

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

    const uint64_t sent_target = tx_side ? msgs : 0;
    if (sent_target) {
        wait_for(sock.counters().tx_done, sent_target, 30ull * 1000 * 1000 * 1000, "tx_done");
    }

    const uint32_t barrier_ms = static_cast<uint32_t>(60000 + std::min<uint64_t>(msgs, 600000));
    if (transport != nullptr) {
        for (Transport* t : sock.peers_for_barrier()) {
            if (const std::string be = t->barrier(); !be.empty()) {
                std::cerr << "  end-of-send barrier failed (host " << t->peer().host_id << "): " << be
                          << "\n";
            }
        }
    }

    if (rx_side) {
        wait_for(sock.counters().delivered, msgs, 15ull * 1000 * 1000 * 1000, "delivered");
    }

    if (rx_side) {
        sock.stamp_timed_end();
    } else if (transport != nullptr && o.host_num > 1) {
        uint32_t slow_core = 0;
        if (!drain_credits(region, o.cores, o.iters, static_cast<uint64_t>(barrier_ms) * 1000000ull,
                           slow_core)) {
            std::cerr << "  end-of-run credit drain timed out on core " << slow_core << " (wanted "
                      << o.iters
                      << "); the bandwidth interval is bounded by posts, not arrivals -- do not quote it\n";
        }
        sock.stamp_timed_end();
    }

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
    if (!o.volume_csv.empty() && stats.ladder.enabled) {
        stats.ladder_seal_final();
        const bool fresh = !std::ifstream(o.volume_csv).good();
        std::ofstream lf(o.volume_csv, std::ios::app);
        if (!lf) {
            std::cerr << "warning: could not open " << o.volume_csv << " for the volume ladder\n";
        } else {
            if (fresh) {
                lf << ladder_csv_header();
            }
            lf << ladder_csv_rows(stats, o.tag);
            std::cout << "  volume ladder -> " << o.volume_csv << " (" << stats.ladder_points()
                      << " checkpoints reached of " << stats.ladder.marks.size() << ")\n";
        }
    }
    stats.payload_bytes = o.bytes;
    stats.cores = o.cores;
    stats.iters = o.iters;
    stats.provider = provider_label_str;
    stats.mode = tx_side ? "sym-tx" : "sym-rx";
    stats.host_clock_valid = clock.valid;
    stats.host_clock_uncertainty_ns = clock.uncertainty_ns;
    stats.device_clock_valid = o.ns_per_cycle > 0.0;
    stats.device_clock_uncertainty_ns = 0;

    stats.run_id = make_run_id();
    stats.run_started_utc = utc_now_iso();
    stats.role = transport == nullptr ? "local" : (o.host_ident == 0 ? "server" : "peer");
    stats.host_ident = o.host_ident;
    stats.symmetric = true;
    stats.h2d = o.h2d_socket ? "socket" : "write";
    stats.tx_side = tx_side;
    stats.warmup = o.warmup;
    stats.warmup_applied = o.warmup > 0;
    stats.timed_iters = o.iters - o.warmup;
    stats.ns_per_cycle = o.ns_per_cycle;

    if (!o.trace_csv.empty()) {
        std::ofstream f(o.trace_csv, std::ios::trunc);
        f << format_trace_csv(stats, o.tag);
        const uint64_t clamped = stats.total_trace_clamped();
        std::cout << "  trace written to " << o.trace_csv << " (bucket " << (1ull << stats.trace_shift)
                  << " ns";
        if (clamped > 0) {
            std::cout << "; " << clamped << " samples FOLDED into the last bucket -- raise "
                      << "the run outlasted the trace span";
        }
        std::cout << ")\n";
    }

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
        if (truncate) {
            std::string err;
            const std::string archived = rotate_csv(path, err);
            if (!err.empty()) {
                path += "." + make_run_id() + ".csv";
                std::cerr << "  " << err << "\n  writing to " << path << " instead\n";
            } else if (!archived.empty()) {
                std::cout << "  rotated previous csv to " << archived << "\n";
            }
        } else if (const std::string e = csv_schema_error(path, basic_csv_header()); !e.empty()) {
            path += ".new";
            std::cerr << "  " << e << "\n  writing to " << path << " instead\n";
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
    const uint64_t want_delivered = !rx_side ? 0 : msgs;
    if (cn.delivered.load() < want_delivered) {
        ok = false;
        why << "delivered " << cn.delivered.load() << " of " << want_delivered << " into L1. ";
    }
    if (tx_side && cn.tx_done.load() < msgs) {
        ok = false;
        why << "sent " << cn.tx_done.load() << " of " << msgs << " messages. ";
    }
    if (ok && cn.routed_remote.load() == 0 && deliverer) {
        std::string detail;
        if (!verify_delivery(*deliverer, o, detail)) {
            ok = false;
            why << "L1 payload check failed: " << detail << ". ";
        }
    } else if (cn.routed_remote.load() > 0) {
        std::cout << "\n  NOTE: " << cn.routed_remote.load()
                  << " messages were routed to the peer. Their arrival is the PEER's to verify;\n"
                     "        this process cannot witness it and does not claim to.\n";
    }

    std::cout << "\n" << (ok ? "PASS" : "FAIL") << (ok ? "" : ": " + why.str()) << "\n\n";
    return ok ? 0 : 1;
}

int run_device(Options& o) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    std::cout << "\n=== " << kProg << " on device " << o.device_id << " ===\n\n";

    namespace mh = tt::tt_metal::distributed::multihost;
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
    dc.iters = o.iters;
    dc.warmup = o.warmup;
    dc.workers = o.workers;
    dc.send_window = o.send_window;
    dc.send_blocking = o.send_blocking;
    dc.pin = o.pin;
    dc.h2d_socket = o.h2d_socket;
    dc.measure_retire = o.measure_retire;
    dc.same_host = o.same_host;
    dc.ladder_enabled = !o.volume_csv.empty();
    dc.ladder_quiesce = o.volume_quiesce;

    std::string serr;
    std::unique_ptr<D2DSocket> sock = D2DSocket::create(mesh_device, device, dc, serr);
    if (!sock) {
        std::cerr << "socket bringup failed: " << serr << "\n";
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

    const uint32_t kernel_opcode =
        !o.store ? static_cast<uint32_t>(kOpSendUva)
                 : (o.bytes <= kCtrlImmMax ? static_cast<uint32_t>(kOpRdmaWriteImm)
                                           : static_cast<uint32_t>(kOpRdmaWrite));
    const uint32_t store_dest_addr = l1.store_dest(o.dest_offset, o.store);

    o.l1_lo = l1.payload_addr;
    o.l1_hi = l1.l1_size;
    o.l1_signal = l1.signal_addr;
    o.l1_completion = l1.completion_addr;
    o.l1_stop = l1.stop_addr;
    o.l1_dest_word = l1.dest_word_addr;

    if (o.store) {
        std::printf("  store         %s, dest 0x%08X (payload_addr 0x%08X + 0x%X)\n",
                    (kernel_opcode == static_cast<uint32_t>(kOpRdmaWriteImm))
                        ? "rdma_write_imm (length in the instruction)"
                        : "rdma_write (length in a register)",
                    store_dest_addr, l1.payload_addr, o.dest_offset);
    }

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < o.cores; ++i) {
        const CoreCoord c{i % g.width, i / g.width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }

    const uint32_t dest_host = (o.host_num > 1) ? ((o.host_ident + 1) % o.host_num) : o.host_ident;

    Program program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = sock->sender_compile_args(
                /*iterations=*/(o.host_ident != 0) ? 0u : o.iters, kernel_opcode,
                static_cast<uint32_t>(kFlagStamped), /*await_completion=*/true)});

    for (uint32_t i = 0; i < o.cores; ++i) {
        const uint32_t sel = t6_global_selector(dest_host, o.chip, i, o.chips_per_host);
        const uint32_t rnd_hosts = 0u;  // fixed destination: the kernel does not walk
        const uint32_t seed = 0x9E3779B9u ^ (i * 2654435761u) ^ (o.host_ident * 40503u);
        SetRuntimeArgs(program, kernel, core_list[i],
                       {sel, store_dest_addr, rnd_hosts, o.chips_per_host, o.cores, seed,
                        o.host_ident, o.chip});
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
                .compile_args = sock->receiver_compile_args()});
        for (uint32_t i = 0; i < o.cores; ++i) {
            const uint32_t enabled = (o.host_ident == 0) ? 0u : 1u;
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
    if (!o.csv_rotate.empty()) {
        std::string err;
        const std::string archived = rotate_csv(o.csv_rotate, err);
        if (!err.empty()) {
            std::cerr << "error: " << err << "\n";
            return 2;
        }
        if (archived.empty()) {
            std::cout << "no existing " << o.csv_rotate << " to rotate\n";
        } else {
            std::cout << "rotated " << o.csv_rotate << " -> " << archived << "\n";
        }
        return 0;
    }
    if (o.layout) {
        print_layout(o);
        return 0;
    }
    return run_device(o);
}
