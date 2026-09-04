// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

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
#include <tt-metalium/experimental/sockets/D2H2H2DSocket.hpp>

// THIS PROGRAM REQUIRES A HOST-TO-HOST TRANSPORT, and the requirement comes from the class
// rather than from the driver: D2H2H2DSocket takes `Transport&`, so there is no version of it
// -- and therefore no version of this -- without one. The original replica could fall back to a
// transport-less D2H2DSocket and so still built without libfabric; this cannot, and saying so
// here is better than several hundred lines of errors about an undeclared type.
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
    // Share of --volume discarded before the counters start. The DECLARATION had been
    // swallowed by this comment since the initial import, so the file never compiled --
    // eight live use sites against a member that did not exist. Default 10 matches the
    // --help text and the bare `--steady` case in the parser.
    uint32_t steady_pct = 10;
    // THE H2D LEG'S IMPLEMENTATION. This program exists
    // to be compared against it row for row, so an option that changes what stage 3 measures
    // has to exist on both sides or the pair stops being a check on anything.
    // SOCKET (H2DSocket, DEVICE_PULL) IS ALWAYS ON, exactly like ring aliasing, and for the
    // same reason: there is no run for which the other path is the right answer.
    // Measured 2026-09-02, 110 cores, 8 GiB per point, timed_mb_per_s: the write path plateaus
    // at 314 MB/s from 128 KiB up while socket keeps climbing to 2577 MB/s at 1 MiB -- 8.2x at
    // the top, and 5x by 128 KiB. The write path is host CPU stores into device L1 through a
    // PCIe BAR; at 110 cores UMD's own guard aborts the run over it ("MMIO per-op timeout: 256B
    // store took 19604 us"). A default that cannot survive the program's own core count is not
    // a default. So this is not configurable -- `--h2d write` is REFUSED BY NAME rather than
    // honoured, and rather than deleted, so a script carrying it is told instead of silently
    // measuring a path it did not ask for.
    bool h2d_socket = true;
    // THE STORE.  these two exist to be compared row for
    // row, so a flag one has and the other lacks is a divergence CLAUDE.md says means one of
    // them is wrong. Until now this program could EXECUTE a store (the executor lives in the
    // shared host_socket.cpp) but could not ISSUE one, so half the pair was untestable.
    uint32_t dest_offset = 0;
    bool store = false;
    // The L1 map as the executor needs to see it, to bound a store's address.
    uint32_t l1_lo = 0, l1_hi = 0, l1_signal = 0, l1_completion = 0, l1_stop = 0, l1_dest_word = 0;
    // --csv-rotate PATH: archive an existing CSV and exit. No mode, no device. Present
    // because run_sweep_*.sh calls it once per phase before any point, so a binary without
    // it cannot be dropped into those scripts at all.
    std::string csv_rotate;
    // --volume-csv PATH: the volume ladder. See host_stats.hpp.
    std::string volume_csv;
    // --trace-csv PATH: the cumulative-volume time series. format_trace_csv() has been
    // compiled into this binary all along with no flag and no recorder to feed it -- bug 11's
    // shape, logged as TODO.md T7c-2.
    std::string trace_csv;
    // --volume-quiesce: hold every worker at each ladder checkpoint so the boundary is exact
    // regardless of load imbalance. OFF BY DEFAULT -- it pauses servicing, which is the shape
    // host_socket.hpp:192 documents as deadlocking a pair, and it is bounded rather than
    // trusted: a worker that misses the budget is abandoned and the checkpoint is recorded as
    // degraded. Unquiesced boundaries are already exact when the pool is balanced.
    bool volume_quiesce = false;
    uint32_t warmup = 0;
    uint32_t workers = 0;
    bool pin = true;

    // THE TWO SENDER KNOBS. `send_window` 0 means unset ->
    // cores-in-use, the behaviour before the knob existed; `send_blocking` selects
    // post-and-wait and implies a window of 1. Independent: the window caps CONCURRENCY,
    // the shape decides whether the thread parks or spins.
    uint32_t send_window = 0;
    bool send_blocking = false;

    //
    // This is the GUPS shape, and it is the first mode where the ADDRESS varies. Every other
    // mode computes dest_uva once outside the kernel's loop, so they measure the transport
    // and never the addressing -- the same numbers would come from a hardcoded destination.
    // Needs no wire change: register 0 was already written every iteration and simply held
    // a constant.
    //
    // REMOTE ONLY. A random draw hits this host 1/N of the time and real GUPS counts those,
    // but a local destination is not deliverable here (one shared L1 slot per core, and the
    // aliased-ring refusal). Not GUPS-conformant, and the results must
    // say so.

    uint32_t host_ident = 0;
    uint32_t host_num = 1;
    uint32_t chips_per_host = 1;
    uint32_t chip = 0;

    bool use_transport = false;
    // REMOVED 2026-09-03: server, peers, provider, port, bind_addr. Five fields with no
    // reader anywhere -- no parser case, so no flag could set them, and nothing downstream
    // consulted them. `provider` was the harmful one: it defaulted to "tcp" and looked like
    // it selected a backend, while make_transport() has returned MpiRmaTransport
    // unconditionally since libfabric was removed and the CSV's provider column is written
    // from a literal at the run_* call site. use_transport survives because :294 derives it
    // from host_num, which is the only bootstrap decision left.
    bool same_host = false;
    // Time the payload write post -> completion on the progress thread, without waiting.
    // Adds the diag:h2h-retire row. See ../MEASURING-BANDWIDTH.md.
    bool measure_retire = false;
    bool roundtrip = false;
    double ns_per_cycle = 0.0;

    std::string csv;
    std::string tag;

    // ONE FILE PER RUN by default (rotate_csv). --csv-append opts back into piling runs
    // into one file, for loops that invoke this binary once per payload.
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

// IDENTITY COMES FROM THE COMMUNICATOR, read once so every later use -- the peer table, UVA
// routing, the CSV's host_ident column -- sees the same two numbers connect_mesh() will use.
// It cannot be overridden by a flag: a flag that disagreed with the rank would produce a run in
// which the socket and the transport hold different opinions about which host this process is.
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
    // Checked numeric parse: an unset shell variable leaves the
    // NEXT FLAG where a number should be, and std::stoi would abort on it with a core dump
    // naming neither the option nor the value.
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

        // --volume IS THE MEASURED VOLUME: the warmup is PADDED ON TOP of it rather than taken
        // out of it. Before this, `--volume 2G --steady 10` moved 2048 MiB and reported 1843 --
        // the number in the flag was not the number in the CSV, which is how two runs get
        // compared as though they had moved the same bytes. The warmup is a share of the TOTAL,
        // so the inverse is total = ceil(measured / (1 - pct/100)); the loop settles the
        // integer-floor case rather than trusting the closed form to land on the right side.
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

    // --random-dest NEEDS SOMEWHERE TO SEND. At --host-num 1 every draw is this host, and
    // a local destination is not deliverable on this path -- so the mode would silently
    // degenerate to the fixed destination and report a "random" measurement that was not.
    // Refused by name instead.
    // The payload carries an 8-byte header (iteration, then the destination selector), and
    // the verifier reads both. A payload shorter than that would overlap them with data and
    // the routing check would compare against bytes that are payload, not address.
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

// The verdict is a byte comparison of independently written buffers -- what the sender
// staged against what actually reached L1 -- not either side reporting success.
bool verify_delivery(Deliverer& deliverer, const Options& o, std::string& detail) {
    const uint32_t last = o.iters - 1;
    // READ WHERE THE STORE AIMED. A store names its own destination, so verifying at the
    // fixed payload_addr would report a perfectly delivered message as a failure -- observed
    // exactly once, at --dest-offset 0x1000: 160 delivered, 0 errors, and a verifier looking
    // at an address the bytes had deliberately not gone to.
    const uint32_t verify_at = o.store ? (o.l1_lo + o.dest_offset) : 0u;
    for (uint32_t core = 0; core < o.cores; ++core) {
        const std::vector<uint8_t> got = deliverer.read_payload(core, o.bytes, verify_at);
        if (got.size() < o.bytes) {
            detail = "core " + std::to_string(core) + ": short read from L1";
            return false;
        }
        uint32_t stamp = 0;
        std::memcpy(&stamp, got.data() + kPayloadStampOffset, sizeof(stamp));

        // WHICH ITERATION MAY BE THE LAST ONE IN THIS L1?
        //
        // With a fixed destination, exactly one: core c receives only from core c, and that
        // source is depth 1, so its messages land in order and the last is iters-1.
        //
        // With a rotating destination, core d is targeted at EVERY iteration but by a
        // DIFFERENT source each time -- (d - i) mod cores. Each source is still depth 1, so
        // its own messages are ordered, but there is no ordering BETWEEN sources: they take
        // different receive slots and are serviced by different workers. So the final content
        // is whichever source's last message to d completed last, and that source's last
        // message is somewhere in the final `cores` iterations.
        //
        // Measured exactly there: 2 cores, 65536 iterations, core 1 holding stamp 65534 while
        // 65535 was expected. 65535 was core 0's message to core 1 and 65534 was core 1's own;
        // both are legitimately last, and which one wins is a race the protocol never promised
        // to settle. Demanding iters-1 asserts an ordering that does not exist.
        //
        // The check is not dropped -- it still catches a STALE arena, which is what it is for.
        // It is loosened to the window that ordering actually permits.
        // The newest stamp is the only acceptable one; with a fixed destination per core there
        // is exactly one sender. This widened to a window of --cores for --random-dest.
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
        // DID IT LAND WHERE THE ADDRESS SAID? This is the only check here that can answer
        // that. The memset pattern identifies the SENDER (0x40 + source core), so comparing
        // it against the destination core only worked while every mode sent core c to core c
        // -- it proved "bytes are intact", never "bytes arrived at the named core", and a
        // genuine misroute passed silently in every fixed-destination run ever made.
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

        // THE SENDER'S PATTERN, which under a rotating destination is NOT this core's. All it
        // can prove is that the bytes are intact and came from a legitimate sender; the
        // routing claim rests entirely on the selector checked above.
        // EVERY PROBE MUST SHOW THE SAME SENDER, and it must be a sender that exists.
        //
        // Two conditions, and both are needed. "In range" alone would accept a payload torn
        // together from two senders; "all equal" alone would accept a buffer of any single
        // wrong value, including one never written. Comparing a probe against ITSELF -- which
        // is what falls out of relaxing this carelessly -- is a check that cannot fail.
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

// `after_open` runs once the socket is live and the pool is scanning. The device path uses
// it to enqueue the workload: the kernel arms its control word within microseconds of
// launch, and a pool that has not started yet would see the sequence already advanced and
// treat those messages as duplicates.
// `mesh` is every peer transport beyond `transport` -- one per rank in the world past this
// one, built by connect_mesh(); empty at two ranks. Registered on the socket before open()
// so its PeerTable sees all of them.
int run_common(HostRegion& region, Options& o, Transport* transport, Deliverer* deliverer, const ClockSync& clock,
               const std::string& provider_label_str, const std::function<void()>& after_open = {},
               const std::vector<Transport*>& mesh = {}) {
    const HostTopology topo{o.host_ident, o.host_num, o.chips_per_host};

    VolumeLadder ladder;
    const uint32_t ladder_workers =
        o.workers != 0 ? o.workers : std::max(1u, std::thread::hardware_concurrency());
    if (!o.volume_csv.empty()) {
        const uint64_t recorded = static_cast<uint64_t>(o.iters - o.warmup) * o.cores * o.bytes;
        ladder.build(o.bytes, recorded, ladder_workers);
        ladder.quiesced = o.volume_quiesce;
        ladder.discarded_bytes = static_cast<uint64_t>(o.warmup) * o.cores * o.bytes;
        std::cout << "  volume ladder " << ladder.marks.size() << " checkpoints, " << o.bytes
                  << " B chunks over " << (recorded >> 20) << " MiB recorded -> " << o.volume_csv
                  << "\n";
    }

    LadderSync ladder_sync;
    SocketConfig sc;
    sc.ladder = ladder.enabled ? &ladder : nullptr;
    sc.ladder_sync = (ladder.enabled && ladder.quiesced) ? &ladder_sync : nullptr;
    sc.payload_bytes = o.bytes;
    sc.chip = o.chip;
    sc.cores = o.cores;
    sc.workers = o.workers;
    sc.pin = o.pin;
    sc.roundtrip = o.roundtrip;
    sc.send_window = o.send_window;
    sc.send_blocking = o.send_blocking;
    sc.ns_per_cycle = o.ns_per_cycle;

    sc.record_from_start = o.warmup == 0;
    sc.warmup_msgs = static_cast<uint64_t>(o.warmup) * static_cast<uint64_t>(o.cores);

    // ONE CONSTRUCTION PATH, BECAUSE THE CLASS HAS ONE. host_socket.hpp's pair let the driver
    // fall back to a transport-less D2H2DSocket; this class takes `Transport&`, so the
    // no-transport case cannot be built rather than being built and then refusing at the first
    // remote UVA. Refused by name here instead of turning into a null dereference three layers
    // down.
    //
    // WHAT THAT COSTS, STATED PLAINLY: single-process local-only mode is gone from this
    // program. There is now exactly one way in -- launch it under mpirun with two or more
    // ranks. :294 derives use_transport from host_num for that reason, and it is the whole of
    // the bootstrap decision; connect_mesh() does the rest from DistributedContext.
    if (transport == nullptr) {
        std::cerr << "error: D2H2H2DSocket requires a host-to-host transport, and none was "
                     "configured.\n"
                     "  Launch under mpirun with two or more ranks. A single-process local run "
                     "has no\n"
                     "  middle hop to exercise -- the path is chip->host->host->chip.\n";
        return 2;
    }
    auto sock = std::make_unique<D2H2H2DSocket>(region, deliverer, topo, clock, sc, *transport);
    // BEFORE open(): start_transport() builds the peer table from these and the sender
    // thread then reads it without synchronisation.
    for (Transport* t : mesh) {
        sock->add_peer(t);
    }

    // THE STORE FAULT DOMAIN. A store's offset arrives from another machine, so an executor
    // that trusts it is an arbitrary-write primitive. In the two-construction-path version this
    // was set on the transport branch only, which left the local one with lo == hi == 0 and
    // every store faulting as "runs past the end of this core's L1" -- 160 service errors, 0
    // delivered. With one path there is nowhere for it to be missed.
    sock->set_store_guard(D2H2H2DSocket::StoreGuard{o.l1_lo, o.l1_hi, o.l1_signal,
                                                    o.l1_completion, o.l1_stop});
    std::cout << "  socket        D2H2H2DSocket\n";

    const uint64_t msgs = static_cast<uint64_t>(o.cores) * o.iters;

    // WHO DOES WHAT. ident 0 only sends and ident 1 only receives -- the precondition that
    // makes one shared L1 buffer per core safe, and the reason a core is never simultaneously
    // a source and a destination. Every wait and every verdict below has to follow, because a
    // side that never receives cannot be asked for `delivered`.
    //
    // THIS IS THE UNIDIRECTIONAL SHAPE, and it is the only one this program drives. The socket
    // itself services both directions on the same threads and does not impose the split.
    const bool tx_side = o.host_ident == 0;
    const bool rx_side = o.host_ident != 0;
    std::cout << "  symmetric   one L1 buffer per core, shared address; this side "
              << (tx_side ? "SENDS only" : "RECEIVES only") << "\n";

    std::string oerr;
    if (!sock->open(oerr)) {
        std::cerr << "socket open failed: " << oerr << "\n";
        return 1;
    }
    if (after_open) {
        after_open();
    }

    // THE KERNEL IS THE ONLY PRODUCER. Nothing on the host arms a TX control word: the
    // device does it, so every run needs real silicon. There is no host stand-in to fall
    // back to, which is why open_transport() failing is fatal rather than degrading.

    // --- shutdown, in the only order that does not truncate a peer -------------
    //
    // 1. wait until WE have sent everything we owe
    // 2. rendezvous, so the PEER has also finished sending before either of us stops
    // 3. only then wait for our own inbound tail
    // 4. rendezvous again, so nobody tears down until BOTH sides have received
    //
    // Skipping step 4 is what lost the last few messages: measured on a two-host tcp run as
    // A reporting 160/160 PASS and B reporting 145/160 FAIL from the SAME run, because A had
    // finished, exited, and taken the remaining 15 transfers with it. An asymmetric result
    // across one run is the signature.
    //
    // A DEADLINE ON PROGRESS, NOT ON DURATION. An absolute budget sized for 160 messages is
    // silently wrong for 149,792: a receiver reported `delivered 138036 of 149792` and FAILED
    // while still delivering at full rate. Progress resets the deadline, so the same number
    // is correct at both scales and a real stall is still caught in the same time.
    auto wait_for = [&](const std::atomic<uint64_t>& c, uint64_t want, uint64_t budget_ns, const char* what) {
        uint64_t seen = c.load(std::memory_order_acquire);
        uint64_t dl = now_ns() + budget_ns;
        uint64_t next_report = now_ns() + 5ull * 1000 * 1000 * 1000;
        while (c.load(std::memory_order_acquire) < want && now_ns() < dl) {
            // A faulted transport does not recover, so spinning to the deadline would report
            // a timeout where the actual error is already known.
            if (sock->transport_failed()) {
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
                          << sock->stall_dump(what) << std::flush;
                next_report = now_ns() + 5ull * 1000 * 1000 * 1000;
            }
            std::this_thread::yield();
        }
    };

    // The sending side has no inbound traffic to wait for afterwards, so tx_done IS its
    // completion condition.
    const uint64_t sent_target = tx_side ? msgs : 0;
    if (sent_target) {
        wait_for(sock->counters().tx_done, sent_target, 30ull * 1000 * 1000 * 1000, "tx_done");
    }

    // Budget proportional to what the peer may still owe: a constant 60 s is a false failure
    // on any run big enough to matter.
    const uint32_t barrier_ms = static_cast<uint32_t>(60000 + std::min<uint64_t>(msgs, 600000));
    if (transport != nullptr) {
        for (Transport* t : sock->peers_for_barrier()) {
            if (const std::string be = t->barrier(); !be.empty()) {
                std::cerr << "  end-of-send barrier failed (host " << t->peer().host_id << "): " << be
                          << "\n";
            }
        }
    }

    if (rx_side) {
        // was `o.roundtrip ? ...home_done : ...delivered` -- home_done is never incremented
        // by anything, so the round-trip arm would have waited out its whole deadline.
        wait_for(sock->counters().delivered, msgs, 15ull * 1000 * 1000 * 1000, "delivered");
    }

    //   rx-side   our own delivered counter IS the arrival evidence, and it is local, so no
    //             barrier round trip enters the interval.
    //   tx-only   (the sending side) we receive nothing, so our own books say only "posted".
    //             The peer's credit registers say "consumed", which is the same evidence
    //             fabtests takes from its per-window ack.
    //
    // Before stop(): the workers and the progress thread are still live, which is what lets
    // the credit drain make progress at all.
    if (rx_side) {
        sock->stamp_timed_end();
    } else if (transport != nullptr && o.host_num > 1) {
        uint32_t slow_core = 0;
        if (!drain_credits(region, o.cores, o.iters, static_cast<uint64_t>(barrier_ms) * 1000000ull,
                           slow_core)) {
            std::cerr << "  end-of-run credit drain timed out on core " << slow_core << " (wanted "
                      << o.iters
                      << "); the bandwidth interval is bounded by posts, not arrivals -- do not quote it\n";
        }
        sock->stamp_timed_end();
    }

    // Before stop(): our workers and the progress thread must still be live while the peer
    // drains.
    if (transport != nullptr) {
        for (Transport* t : sock->peers_for_barrier()) {
            if (const std::string be = t->barrier(barrier_ms); !be.empty()) {
                std::cerr << "  end-of-receive barrier failed (host " << t->peer().host_id << "): " << be
                          << "\n";
            }
        }
    }

    sock->stop();

    RunStats stats = sock->collect();
    stats.ladder = ladder;
    // Harvested after the workers have joined, so the counts are final.
    stats.ladder.quiesce_clean = ladder_sync.clean.load(std::memory_order_relaxed);
    stats.ladder.quiesce_degraded = ladder_sync.degraded.load(std::memory_order_relaxed);
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

    // IDENTITY AND PROVENANCE, matching t6_host_uva. These two programs exist to be compared
    // against each other, so a field one of them records and the other does not is a field
    // that cannot be used in the comparison.
    stats.run_id = make_run_id();
    stats.run_started_utc = utc_now_iso();
    stats.role = transport == nullptr ? "local" : (o.host_ident == 0 ? "server" : "peer");
    stats.host_ident = o.host_ident;
    stats.symmetric = true;
    stats.h2d = o.h2d_socket ? "socket" : "write";
    stats.tx_side = tx_side;
    stats.warmup = o.warmup;
    // Was `o.warmup > 0 && !o.device_producer`, honestly reporting that the gate did not
    // apply to a kernel-driven run. It applies to every run shape now -- the socket opens it
    // by message count when no producer loop will -- so the condition matches t6_host_uva's
    // (`warmup_msgs > 0`) and the two files' columns mean the same thing again.
    stats.warmup_applied = o.warmup > 0;
    // THE POPULATION BEHIND timed_ns, and it was never assigned -- the field defaulted to 0,
    // so format_table()'s `xfers = timed_iters * cores * xfers_per_iter` came out zero and
    // THREE of the bandwidth block's seven columns printed 0: iters, usec/xfer and
    // Mxfers/sec. timed_mb_per_s was unaffected, which is why it went unnoticed.
    //
    // usec/xfer is the column that matters: timed_ns / (this * cores) is a message's whole
    // RESIDENCE in the path -- 408.68 us at 16 KiB/x1 against 25.64 us of measured legs. That
    // ratio is the queueing this program cannot otherwise show, because ONEWAY_TOTAL sums the
    // three legs and omits diag:sendq-wait and the credit stall between them.
    //
    // PER-CORE, because the formula multiplies by `cores`. `iters` is already per-core (the
    // kernel's loop count) and so is `warmup`, so the difference is the right quantity.
    stats.timed_iters = o.iters - o.warmup;
    // Left at its default of 1 deliberately. It is fabtests' show_perf() argument -- 1 one-way,
    // 2 round trip -- and the round-trip path is refused at parse time, so 2 is unreachable.
    stats.ns_per_cycle = o.ns_per_cycle;

    // AFTER the identity fields, not before. Placed above the ladder block first, which is
    // ahead of stats.payload_bytes / stats.cores / stats.run_id -- the trace then carried
    // real buckets under a header reading payload_bytes=0, cores=0 and an EMPTY run_id. The
    // data was right and unattributable, which is the shape recovery.md records as costing a
    // comparison twice: same filename is not the same run, and a row with no run_id cannot
    // be told apart from any other.
    if (!o.trace_csv.empty()) {
        std::ofstream f(o.trace_csv, std::ios::trunc);
        f << format_trace_csv(stats, o.tag);
        const uint64_t clamped = stats.total_trace_clamped();
        std::cout << "  trace written to " << o.trace_csv << " (bucket " << (1ull << stats.trace_shift)
                  << " ns";
        if (clamped > 0) {
            // NAMED, because a folded tail plots as a spike in the last bucket and would
            // otherwise read as a burst at the end of the run rather than as lost resolution.
            std::cout << "; " << clamped << " samples FOLDED into the last bucket -- raise "
                      << "the run outlasted the trace span";
        }
        std::cout << ")\n";
    }

    std::cout << format_table(stats);

    const SocketCounters& cn = sock->counters();
    std::cout << "\n=== routing and delivery ===\n\n";
    std::printf("  local     %llu\n", (unsigned long long)cn.routed_local.load());
    std::printf("  remote    %llu\n", (unsigned long long)cn.routed_remote.load());
    std::printf("  nowhere   %llu   (selector named no configured host)\n",
                (unsigned long long)cn.routed_nowhere.load());
    std::printf("  delivered %llu   (bytes written into a Tensix L1)\n", (unsigned long long)cn.delivered.load());
    // REPLIES / HOME REMOVED: nothing in the tree increments either counter, so both
    // printed a confident 0 next to real numbers. The round-trip half went with libfabric.
    //   std::printf("  replies   %llu\n", (unsigned long long)cn.replies.load());
    //   std::printf("  home      %llu   (replies delivered back to the originating core)\n",
    //               (unsigned long long)cn.home_done.load());
    std::printf("  errors    %llu\n", (unsigned long long)cn.errors.load());
    if (transport != nullptr) {
        // THE TRANSPORT'S FINAL STATE, not just its state 5 s into a stall. `retired` versus
        // `abandoned` separates "the completion arrived at 31 s" from "it never arrived".
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
    const std::string fe = sock->first_error();
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
            // ONE FILE PER RUN. Archive whatever is there, then write a fresh file with its
            // own header. A schema change then cannot produce a ragged file, and no reader
            // has to guess where one run ends and the next begins.
            std::string err;
            const std::string archived = rotate_csv(path, err);
            if (!err.empty()) {
                // Could not archive, so truncating would destroy data. Divert instead.
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
        // THE WIDE CSV IS OFF. Sixty columns of per-hop distribution, uncertainty bounds and
        // rate flags, none of which can be checked by hand, and one of which (mb_per_s_mean)
        // was a bandwidth derived from a latency histogram. Restore by swapping the two calls
        // above back to csv_header() / format_csv() -- both still build.
        //
        //   f << csv_header();
        //   f << format_csv(stats, o.tag);
        std::cout << "  csv " << (fresh ? "written to " : "appended to ") << path << "\n";
    }

    bool ok = cn.errors.load() == 0;
    std::ostringstream why;
    if (cn.errors.load()) {
        why << cn.errors.load() << " service errors. ";
    }
    // A round trip delivers TWICE per message -- outbound into the destination core, and the
    // reply into the originator. Checking only the outbound half is what let a run with
    // `replies 0` report PASS.
    // was `o.roundtrip ? msgs * 2 : msgs`
    const uint64_t want_delivered = !rx_side ? 0 : msgs;
    if (cn.delivered.load() < want_delivered) {
        ok = false;
        why << "delivered " << cn.delivered.load() << " of " << want_delivered << " into L1. ";
    }
    // if (o.roundtrip && cn.replies.load() < msgs) {          <- unreachable, both halves
    //     ok = false;                                            dead: o.roundtrip cannot be
    //     why << "only " << cn.replies.load() << " of " ...;      true and replies never moves
    // }
    if (tx_side && cn.tx_done.load() < msgs) {
        ok = false;
        why << "sent " << cn.tx_done.load() << " of " << msgs << " messages. ";
    }
    // Only an all-local run can be verified from this process. A remote-routed run's bytes
    // land on the PEER, so its verdict is the peer's to give -- claiming PASS here from our
    // own state would be exactly the self-witnessing this tree forbids.
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

// Bootstraps a transport from the options. Shared by both modes so they cannot drift in
// which fields they fill.
// `mesh_owned`/`mesh_table` are filled by connect_mesh() -- one endpoint per peer rank in the
// world. `out` is the FIRST peer and the rest are registered on the socket with add_peer().
// At two ranks there is exactly one, which is the D2H2H2DSocket path (open() refuses topo.num
// != 2 today).
// ONE ENDPOINT PER PEER RANK. Nothing to configure: no provider to name, no listen/connect
// asymmetry, no address list, no port. connect_mesh() takes identity from DistributedContext,
// so this cannot tell it a different story than the communicator can.
bool open_transport(const Options& o, uint32_t grid_width, HostRegion& region, std::unique_ptr<Transport>& out,
                    ClockSync& clock, int& rc, std::vector<std::unique_ptr<Transport>>& mesh_owned,
                    PeerTable& mesh_table) {
    namespace mh = tt::tt_metal::distributed::multihost;

    TransportConfig tc;
    tc.chips_per_host = o.chips_per_host;
    tc.grid_width = grid_width;
    tc.cores_in_use = o.cores;
    tc.measure_retire = o.measure_retire;

    if (const std::string e = connect_mesh(region.base(), region.pinned_bytes(), tc, mesh_owned, mesh_table);
        !e.empty()) {
        std::cerr << "mesh bringup failed: " << e << "\n";
        rc = 1;
        return false;
    }
    if (mesh_owned.empty()) {
        std::cerr << "no peers: this is a one-rank job and the path is chip->host->host->chip\n";
        rc = 1;
        return false;
    }
    std::cout << "  connected: " << mesh_owned.size() << " peer(s)\n";
    std::cout << "  transport: " << mesh_owned.front()->describe() << "\n";

    // SYNCED AGAINST RANK 0'S PARTNER, with the lower rank initiating, so both sides pick the
    // same roles without negotiating them.
    const auto& ctx = mh::DistributedContext::get_current_world();
    const uint32_t self = static_cast<uint32_t>(*ctx->rank());
    const uint32_t peer = (self == 0) ? 1u : 0u;
    clock = sync_clocks(ctx, mh::Rank{static_cast<int>(peer)}, /*initiator=*/self < peer, o.same_host);
    std::cout << "  clock: " << clock.describe() << "\n";
    if (!clock.valid) {
        std::cerr << "refusing to report cross-host hop timings without a clock offset\n";
        rc = 1;
        return false;
    }

    out = std::move(mesh_owned.front());
    mesh_owned.erase(mesh_owned.begin());
    return true;
}




int run_device(Options& o) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    std::cout << "\n=== " << kProg << " on device " << o.device_id << " ===\n\n";

    // ---- THE DEVICE IS OPENED UNDER A SINGLE-HOST CONTEXT ----------------------------
    //
    // This path joins two hosts over MPI, through host memory -- that is what
    // the name spells out, and it is the whole reason the class exists: the destination is not
    // in any chip's NOC address space. There is no TT-ethernet fabric between the two devices
    // and the test does not want one.
    //
    // tt-metal's control plane does not know that. Seeing a world of size 2 it demands a mesh
    // graph descriptor spanning both hosts, and then tries to MAP that graph onto discovered
    // silicon. On two uncabled galaxies it reports `intermesh degree histogram {0:1}` -- zero
    // links between them -- and refuses, correctly: the descriptor asserts cabling that is not
    // there. Handing it dual_bh_galaxy_torus_xy produces exactly that failure.
    //
    // So the device is opened while the current world is a context of size ONE, split per rank.
    // Every rank then drives its own local silicon with single-host semantics, which is the
    // truth, and auto-discovery's one-host mesh becomes correct rather than a limitation. The
    // full world is restored before the transport is opened, because the H2H leg is the one
    // thing that really is multi-host.
    //
    // This also makes the test runnable on ANY two nodes that each have a device, rather than
    // only on a pair wired into one fabric -- which is what a CI job needs.
    namespace mh = tt::tt_metal::distributed::multihost;
    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    {
        const mh::ContextPtr solo =
            world->split(mh::Color{static_cast<int>(o.host_ident)}, mh::Key{0});
        mh::DistributedContext::set_current_world(solo);
    }

    auto mesh_device = MeshDevice::create_unit_mesh(o.device_id);
    IDevice* device = mesh_device->get_devices().front();

    // THE CONTROL PLANE HAS ITS VIEW NOW, and it holds the context it was built with; restoring
    // here does not reach back into it. From this line on, get_current_world() is the real
    // 2-rank world again, which is what connect_mesh() and the clock sync need.
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

    const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t payload_addr = (l1_base + 0x3F) & ~0x3Fu;
    const uint32_t stage_addr = payload_addr + ((o.bytes + 0x3F) & ~0x3Fu);
    const uint32_t signal_addr = stage_addr + 5 * 16;
    const uint32_t completion_addr = signal_addr + 64;  // own cache line, own meaning
    const uint32_t stop_addr = completion_addr + 64;
    const uint32_t dest_word_addr = stop_addr + 64;
    const uint32_t deliver_addr = payload_addr;

    const uint32_t l1_size = static_cast<uint32_t>(device->l1_size_per_core());

    const uint32_t kernel_opcode =
        !o.store ? static_cast<uint32_t>(kOpSendUva)
                 : (o.bytes <= kCtrlImmMax ? static_cast<uint32_t>(kOpRdmaWriteImm)
                                           : static_cast<uint32_t>(kOpRdmaWrite));
    // The effective address the kernel will name -- absolute in the far core's L1. The caller
    // gives an offset from payload_addr; the allocator base is added here, where it is known.
    const uint32_t store_dest_addr = o.store ? (payload_addr + o.dest_offset) : 0u;

    // Published to the executor, which bounds a store's address against it. Taken from the
    // same values the kernels are compiled with, so the check cannot drift from the map.
    o.l1_lo = payload_addr;
    o.l1_hi = l1_size;
    o.l1_signal = signal_addr;
    o.l1_completion = completion_addr;
    o.l1_stop = stop_addr;
    o.l1_dest_word = dest_word_addr;

    if (o.store) {
        std::printf("  store         %s, dest 0x%08X (payload_addr 0x%08X + 0x%X)\n",
                    (kernel_opcode == static_cast<uint32_t>(kOpRdmaWriteImm))
                        ? "rdma_write_imm (length in the instruction)"
                        : "rdma_write (length in a register)",
                    store_dest_addr, payload_addr, o.dest_offset);
    }
    const uint32_t copies = 1u;
    if (payload_addr + copies * o.bytes + (deliver_addr - payload_addr - (copies - 1) * o.bytes) > l1_size ||
        deliver_addr + o.bytes > l1_size) {
        const uint32_t overhead =
            stage_addr - payload_addr - static_cast<uint32_t>(o.bytes) + 5 * 16 + 128;
        const uint32_t ceiling = ((l1_size - payload_addr - overhead) / copies) & ~0x3Fu;
        std::cerr << "error: --bytes " << o.bytes << " does not fit L1 on this core.\n"
                  << "  L1 per core        " << l1_size << " B\n"
                  << "  allocator base     " << payload_addr << " B\n"
                  << "  needed             " << copies << " x " << o.bytes << " B ("
                  << "one shared buffer) + " << overhead
                  << " B of control words\n"
                  << "  largest --bytes    " << ceiling << " B\n"
                  << "The 1.5 MiB arena is the HOST-side buffer; L1 has to hold what the device holds.\n"
                  ;
        return 2;
    }

    std::string derr;
    const L1Layout l1_layout{deliver_addr, signal_addr, completion_addr, stop_addr, dest_word_addr};
    std::unique_ptr<Deliverer> deliverer;
    if (o.h2d_socket) {
        H2DSocketConfig scfg;
        scfg.page_size = o.bytes;  // one page per message; see kernels/test_kernel_pull.cpp
        // ALWAYS. Ring aliasing was a #define already fixed at 1; the import resolved it.
        const bool alias_requested = true;
        // THE RING IS THE L1 MIRROR, so under aliasing it is sized to a whole arena rather
        // than to the payload.
        //
        // A store writes at VARYING
        // offsets, so we address the ring explicitly instead -- the receive SCR names the
        // offset -- and the socket's pointer bookkeeping leaves the addressing path entirely.
        // With that gone, the constraint that produced `fifo_size == payload` goes too.
        //
        // The pages are the arena's own (that is what aliasing means), so a 1.5 MiB ring per
        // core is not 1.5 MiB of additional pinned memory.
        // ONE PAGE SHORT OF THE ARENA, and the page is not slack -- it is tt-metal's.
        //
        // init_host_buffer() allocates align(fifo_size + 4 + sizeof(HDSocketConnectorState),
        // page), so a ring sized to a whole arena produces an shm one page LARGER than the
        // arena slot -- and map_rings() then refuses it, correctly, because the overlay would
        // reach into the next core's TX arena. Measured the hard way: sizing this at
        // kArenaBytes made every aliased run fall back to the memcpy path with only a stderr
        // line to say so.
        //
        // The cost is that the top page of the L1 mirror is not addressable by a store. That
        // is checked, not assumed -- see the store fault domain.
        // TWO ALIASED RING SIZES, BECAUSE THE TWO OPCODES FIND THEIR BYTES DIFFERENTLY.
        //
        //   kOpSendUva + alias : the device locates the payload through the socket's own
        //                        read_ptr, so the write pointer has to be back at 0 every
        //                        message -- h2d_socket.cpp:663-667 wraps it only on the
        //                        exact-fill case. fifo_size == payload, as before.
        //   store     + alias : the receive SCR names the offset explicitly, so the socket's
        //                        pointer is not in the addressing path and the ring is the
        //                        L1 mirror.
        //
        // ONE PAGE SHORT OF THE ARENA in the mirror case, and the page is tt-metal's, not
        // slack: init_host_buffer() allocates align(fifo_size + 4 + sizeof(ConnectorState),
        // page), so a ring sized to a WHOLE arena yields an shm one page larger than the
        // arena slot -- and map_rings() then refuses it, correctly, because the overlay would
        // reach the next core's TX arena.
        scfg.fifo_size = alias_requested
                             ? (o.bytes /* this program cannot issue a store yet -- TODO P1 */)
                             : 17u * o.bytes;
        // reserved_base(), not region.base(): there is no region yet, which is the point.
        // The storage is a static array, so this is already the pointer base() will return.
        // Checked immediately after provision() below.
        scfg.alias_region_base = HostRegion::reserved_base();
        deliverer = make_h2d_socket_deliverer(mesh_device, g.width, o.cores, l1_layout, scfg, derr);
    } else {
        deliverer = make_device_deliverer(device, g.width, o.cores, l1_layout, derr);
    }
    if (!deliverer) {
        std::cerr << "H2D delivery unavailable: " << derr << "\n";
        return 1;
    }
    std::cout << "  deliverer     " << deliverer->describe() << "\n";

    HostRegion& region = HostRegion::provision(
        mesh_device, o.chip, o.cores, HostTopology{o.host_ident, o.host_num, o.chips_per_host}, g);
    if (const std::string e = region.verify_header(); !e.empty()) {
        std::cerr << "region header check failed: " << e << "\n";
        return 1;
    }
    // The one assumption the hoist introduces, checked rather than trusted: the deliverer was
    // handed reserved_base() before there was a region to ask.
    if (region.base() != HostRegion::reserved_base()) {
        std::cerr << "region base " << static_cast<void*>(region.base()) << " is not the reserved base "
                  << static_cast<void*>(HostRegion::reserved_base())
                  << "; the deliverer was built against the wrong address\n";
        return 1;
    }
    std::printf("  region        base %p, %llu MiB pinned\n", static_cast<void*>(region.base()),
                (unsigned long long)(region.pinned_bytes() >> 20));
    std::printf("  device view   pcie_xy_enc 0x%08X, io_base 0x%016llx\n", region.device().pcie_xy_enc,
                (unsigned long long)region.device().io_base);

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < o.cores; ++i) {
        const CoreCoord c{i % g.width, i / g.width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }

    std::unique_ptr<Transport> transport;
    // Mesh peers beyond the primary. These must outlive the socket -- its PeerTable holds raw
    // pointers and does not own. `mesh_table` is connect_mesh's scratch: it does the identity
    // and duplicate checks at bringup; the socket builds its own from transport_ + add_peer().
    std::vector<std::unique_ptr<Transport>> mesh_owned;
    PeerTable mesh_table;
    ClockSync clock;
    clock.same_host = true;
    clock.valid = true;
    if (o.use_transport) {
        int rc = 0;
        if (!open_transport(o, g.width, region, transport, clock, rc, mesh_owned, mesh_table)) {
            return rc;
        }
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
            .compile_args = {
                region.device().pcie_xy_enc,
                static_cast<uint32_t>(region.device().io_base & 0xFFFFFFFFull),
                static_cast<uint32_t>(region.device().io_base >> 32),
                g.width,
                payload_addr,
                stage_addr,
                signal_addr,
                o.bytes,
                // Zero iterations on the receiving side: the kernel is still built and
                // launched so both chips keep an identical L1 map, but its loop never runs and
                // it never writes the buffer the host delivers into.
                (o.host_ident != 0) ? 0u : o.iters,
                kernel_opcode,
                static_cast<uint32_t>(kFlagStamped),
                // AWAIT THE DOORBELL. The kernel's control word is a single slot; without
                // this it arms iteration i+1 before a host worker has read iteration i and
                // the duplicate filter drops the skipped message.
                1u,
                completion_addr,
            }});

    for (uint32_t i = 0; i < o.cores; ++i) {
        const uint32_t sel = t6_global_selector(dest_host, o.chip, i, o.chips_per_host);
        // ARGS 2..7 ARM THE RANDOM DESTINATION; host_num 0 leaves the kernel on its fixed
        // target, so a run without --random-dest is bit-for-bit what it always was.
        //
        // SEEDED PER CORE. One seed for all of them would make every core walk the same
        // address stream in lockstep and hammer one destination at a time -- that measures
        // contention, not random access. The seed is derived, not random, so a run is
        // reproducible: same cores, same stream, same traffic pattern.
        const uint32_t rnd_hosts = 0u;  // fixed destination: the kernel does not walk
        const uint32_t seed = 0x9E3779B9u ^ (i * 2654435761u) ^ (o.host_ident * 40503u);
        SetRuntimeArgs(program, kernel, core_list[i],
                       {sel, store_dest_addr, rnd_hosts, o.chips_per_host, o.cores, seed,
                        o.host_ident, o.chip});
    }

    // Seed each core's payload in L1. BYTE pattern, not a word value: filling uint32
    // elements with 0x40 produces the bytes 40 00 00 00 repeating, and the verifier -- which
    // checks bytes -- then reports byte 4095 as 0x00 when the transfer was perfect.
    for (uint32_t i = 0; i < o.cores; ++i) {
        const uint32_t b = 0x40u + (i & 0x1F);
        std::vector<uint32_t> page(o.bytes / sizeof(uint32_t), b * 0x01010101u);
        tt::tt_metal::detail::WriteToDeviceL1(device, core_list[i], payload_addr, page, tt::CoreType::WORKER);
    }

    if (o.h2d_socket) {
        const std::vector<uint32_t> cfg_addrs = deliverer->socket_config_addresses();
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
                .compile_args = {deliver_addr, o.bytes, signal_addr, 1u, stop_addr, dest_word_addr}});
        for (uint32_t i = 0; i < o.cores; ++i) {
            // MIRROR of the post kernel's symmetric rule, not a copy: ident 0 sends and never
            // receives, so its receivers idle. Backwards and the receiving host has no
            // receiver at all, and every delivery times out in wait_delivered.
            const uint32_t enabled = (o.host_ident == 0) ? 0u : 1u;
            SetRuntimeArgs(program, recv_kernel, core_list[i], {cfg_addrs[i], enabled});
        }
        // BEFORE THE ENQUEUE. L1 still holds whatever the previous point's process left in
        // the stop word, including the 1 its own stop_receivers() wrote.
        if (const std::string e = deliverer->arm_receivers(); !e.empty()) {
            std::cerr << "H2D socket: " << e << "\n";
            return 1;
        }
    }

    // Stage 1 arrives from the kernel in Tensix cycles. Measure the rate now, once -- never
    // an epoch, only a scale. Failure is not fatal: stage 1 reports no samples rather than
    // being converted with a guessed rate.
    std::string rate_detail;
    const double ns_per_cycle = measure_ns_per_cycle(*deliverer, 0, 50, rate_detail);
    if (ns_per_cycle > 0.0) {
        std::printf("  device clock  %.4f ns/cycle (%s)\n", ns_per_cycle, rate_detail.c_str());
    } else {
        std::printf("  device clock  UNMEASURED (%s) -- stage t6->host will report no samples\n",
                    rate_detail.c_str());
    }
    o.ns_per_cycle = ns_per_cycle;

    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(mesh_device->shape()), std::move(program));

    // ENQUEUE AFTER THE POOL IS SCANNING, not racing it. The kernel arms its control word
    // within microseconds of launch, and a pool that has not started would see the sequence
    // already advanced and treat those messages as duplicates. run_common calls this hook
    // once open() has returned, so the ordering is enforced rather than merely likely.
    std::thread launcher;
    const int rc = run_common(region, o, transport.get(), deliverer.get(), clock,
                              o.use_transport ? "mpi-rma" : "none", [&] {
                                  launcher = std::thread([&] {
                                      EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload,
                                                          /*blocking=*/false);
                                      Finish(mesh_device->mesh_command_queue());
                                  });
                              },
                              [&] {
                                  std::vector<Transport*> v;
                                  for (auto& up : mesh_owned) {
                                      v.push_back(up.get());
                                  }
                                  return v;
                              }());
    // BEFORE THE JOIN. The receiver kernels loop until this word is set and the launcher
    // thread is sitting in Finish() waiting for them to retire, so joining first would wait
    // on a thread waiting on a kernel waiting on this write.
    if (const std::string e = deliverer->stop_receivers(); !e.empty()) {
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
    // BEFORE ANYTHING ELSE, and it shares rotate_csv() with t6_host_uva rather than
    // reimplementing the archive naming -- two implementations of a filename convention is
    // two things free to drift, with nothing to detect it.
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
