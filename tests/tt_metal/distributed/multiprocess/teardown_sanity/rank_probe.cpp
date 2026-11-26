// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The C++ rank program of the teardown sanity, the twin of tests/ttnn/distributed/teardown_sanity/probe_common.py:
// one tt-metal rank on the DistributedContext with one injected fault, so the rank side of a job stop can be checked
// from C++ as it is from Python. Built twice (tests/tt_metal/distributed/CMakeLists.txt):
//   teardown_context_probe   level 1: the context alone, no device (PROBE_MESH off)
//   teardown_mesh_probe      level 2: a mock MeshDevice open around the collectives (PROBE_MESH on)
// Same knobs as the Python probe (PROBE_SCENARIO, PROBE_VICTIM, PROBE_PHASE, PROBE_COLL, PROBE_DELAY_MS,
// PROBE_ON_ERROR, PROBE_SLOW_S, PROBE_MESH_ROWS/COLS, PROBE_MESH_SHAPE_VICTIM, PROBE_FABRIC_RANK, PROBE_PIDDIR),
// same markers ("[probe r=R pid=P t=T] ...") and marker files, so the harness in sanity_common.py reads both alike.
//   scenarios: clean | mesh_open | throw | exit1 | exit0 | exit_nofinalize | abort_n | abort | sigkill | segv | sigint
//              | hang | ignore_sigterm | slow_exit
//   collectives: barrier | barrier_skip | barrier_slow | barrier_dead_peer | recv_dead_peer | send_no_recv
//                | recv_no_send | allgather_skip | split | split_slow | split_peer_throws

#include <tt-metalium/distributed_context.hpp>
#ifdef PROBE_MESH
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#endif

#include <cxxabi.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace tt::tt_metal::distributed::multihost;

namespace {
const auto T0 = std::chrono::steady_clock::now();
int g_rank = -1;  // the launch rank until the context exists, then the context's
int g_size = 0;
int g_file_rank = -1;  // the launch rank, for the marker files

std::string env(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::string(v) : d;
}

void mark(const std::string& msg) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    std::printf("[probe r=%d pid=%d t=%.3f] %s\n", g_rank, static_cast<int>(::getpid()), t, msg.c_str());
    std::fflush(stdout);
}

void touch(const std::string& name) {
    const std::string d = env("PROBE_PIDDIR", "");
    if (!d.empty()) {
        if (FILE* f = std::fopen((d + "/rank_" + std::to_string(g_file_rank) + "." + name).c_str(), "w")) {
            std::fclose(f);
        }
    }
}

const std::string SCENARIO = env("PROBE_SCENARIO", "clean");
const int VICTIM = std::stoi(env("PROBE_VICTIM", "1"));
const std::string PHASE = env("PROBE_PHASE", "after_open");
const std::string COLL = env("PROBE_COLL", "barrier");
const int DELAY_MS = std::stoi(env("PROBE_DELAY_MS", "300"));
const std::string ON_ERROR = env("PROBE_ON_ERROR", "exit");
const std::string FABRIC_RANK = env("PROBE_FABRIC_RANK", "-1");

bool is_victim() { return VICTIM == -1 || g_rank == VICTIM; }

void sleep_s(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }

[[noreturn]] void never_returns() {
    for (;;) {
        sleep_s(3600);
    }
}

void do_fault() {
    sleep_s(DELAY_MS / 1000.0);
    mark("victim acting: " + SCENARIO + " at " + PHASE);
    if (SCENARIO == "throw") {
        throw std::runtime_error("injected failure at " + PHASE);  // uncaught: std::terminate, the TT_FATAL shape
    }
    if (SCENARIO == "exit1") {
        std::exit(1);  // main returning 1, a caught error: on_exit sees the status
    }
    if (SCENARIO == "exit0") {
        std::exit(0);  // a logic bug's early return while the peers still work (P27)
    }
    if (SCENARIO == "exit_nofinalize") {
        ::_exit(1);  // no hook runs
    }
    if (SCENARIO == "abort_n") {
        DistributedContext::get_current_world()->abort(2);
    }
    if (SCENARIO == "abort") {
        std::abort();  // SIGABRT, a core
    }
    if (SCENARIO == "sigkill") {
        ::raise(SIGKILL);
    }
    if (SCENARIO == "segv") {
        ::raise(SIGSEGV);
    }
    if (SCENARIO == "sigint") {
        ::raise(SIGINT);  // a user interrupts one pid: the default disposition, 130
        sleep_s(5);
    }
    if (SCENARIO == "hang") {
        mark("hang: sleeping forever outside any context call");
        never_returns();
    }
    if (SCENARIO == "ignore_sigterm") {
        std::signal(SIGTERM, SIG_IGN);  // replaces the context's handler too: nothing reacts
        mark("ignoring SIGTERM");
    }
}

void maybe_fault(const std::string& phase) {
    static const std::vector<std::string> faults = {
        "throw",
        "exit1",
        "exit0",
        "exit_nofinalize",
        "abort_n",
        "abort",
        "sigkill",
        "segv",
        "sigint",
        "hang",
        "ignore_sigterm"};
    for (const auto& f : faults) {
        if (SCENARIO == f && is_victim() && PHASE == phase) {
            do_fault();
        }
    }
}

// One call and what it returned; on error either propagate (the process ends non-zero) or record and go on.
void timed(const std::string& label, const std::function<void()>& fn) {
    const auto t = std::chrono::steady_clock::now();
    mark(label + "_enter");
    auto dt = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    try {
        fn();
        mark(label + "_rc=SUCCESS dt=" + std::to_string(dt()));
    } catch (const std::exception& e) {
        std::string what = e.what();
        what = what.substr(0, what.find('\n')).substr(0, 120);
        int status = 0;
        char* demangled = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
        const std::string type = (status == 0 && demangled != nullptr) ? demangled : typeid(e).name();
        std::free(demangled);
        mark(label + "_rc=" + type + ": " + what + " dt=" + std::to_string(dt()));
        if (ON_ERROR == "exit") {
            throw;
        }
    }
}

void collectives(const DistributedContext& ctx) {
    const int rank = g_rank;
    auto barrier = [&] { ctx.barrier(); };
    if (SCENARIO == "ignore_sigterm" && is_victim()) {
        mark("stuck: sleeping 600 s with SIGTERM ignored");
        sleep_s(600);
    }
    if (COLL == "barrier") {
        timed("coll_barrier", barrier);
    } else if (COLL == "barrier_skip") {
        if (is_victim()) {
            mark("coll_skip: victim skips the barrier, sleeps 3 s");
            sleep_s(3);
        } else {
            timed("coll_barrier", barrier);
        }
    } else if (COLL == "barrier_slow") {
        if (is_victim()) {
            const double slow = std::stod(env("PROBE_SLOW_S", "4"));
            mark("coll_slow: victim joins the barrier " + std::to_string(slow) + " s late (a slow peer, not a hang)");
            sleep_s(slow);
        }
        timed("coll_barrier", barrier);
    } else if (COLL == "barrier_dead_peer") {
        if (is_victim()) {
            mark("victim acting: sigkill before the peers' barrier");
            ::raise(SIGKILL);
        }
        sleep_s(0.5);
        timed("coll_barrier", barrier);
    } else if (COLL == "recv_dead_peer") {
        if (is_victim()) {
            mark("victim acting: sigkill before the peer's recv");
            ::raise(SIGKILL);
        } else if (rank == 0) {
            sleep_s(0.5);
            std::vector<std::byte> buf(16);
            timed("coll_recv_16B_from_" + std::to_string(VICTIM), [&] {
                ctx.recv(ttsl::Span<std::byte>(buf.data(), buf.size()), Rank{VICTIM}, Tag{7});
            });
        }
    } else if (COLL == "send_no_recv") {
        if (is_victim()) {
            std::vector<std::byte> buf(1000000, std::byte{'x'});
            timed("coll_send_1MB_to_0", [&] {
                ctx.send(ttsl::Span<std::byte>(buf.data(), buf.size()), Rank{0}, Tag{7});
            });
        } else {
            timed("coll_barrier", barrier);
        }
    } else if (COLL == "recv_no_send") {
        if (rank == 0) {
            std::vector<std::byte> buf(16);
            timed("coll_recv_16B_from_" + std::to_string(VICTIM), [&] {
                ctx.recv(ttsl::Span<std::byte>(buf.data(), buf.size()), Rank{VICTIM}, Tag{7});
            });
        } else if (is_victim()) {
            mark("coll_recv_no_send: victim never sends, waits 10 s, goes to close");
            sleep_s(10);  // so the receiver's timeout fires before this rank's own close barrier does
        }
    } else if (COLL == "allgather_skip") {
        if (is_victim()) {
            mark("coll_skip: victim skips the all_gather, sleeps 30 s");
            sleep_s(30);
        } else {
            int mine = rank;
            std::vector<int> all(static_cast<size_t>(g_size));
            timed("coll_allgather_int", [&] {
                ctx.all_gather(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&mine), sizeof mine),
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(all.data()), all.size() * sizeof(int)));
            });
        }
    } else if (COLL == "split" || COLL == "split_slow" || COLL == "split_peer_throws") {
        if (COLL == "split_peer_throws" && is_victim()) {
            sleep_s(1);
            mark("victim acting: throw while the peers are in split");
            throw std::runtime_error("injected failure while the peers are in split");
        }
        if (COLL == "split_slow" && is_victim()) {
            const double slow = std::stod(env("PROBE_SLOW_S", "4"));
            mark("coll_slow: victim joins the split " + std::to_string(slow) + " s late");
            sleep_s(slow);
        }
        timed("coll_split", [&] { (void)ctx.split(Color{0}, Key{rank}); });
    }
}
}  // namespace

int main(int argc, char** argv) {
    g_rank = std::stoi(env("TT_RUN_RANK", env("OMPI_COMM_WORLD_RANK", "-1")));
    g_file_rank = g_rank;
    const std::string piddir = env("PROBE_PIDDIR", "");
    if (!piddir.empty()) {
        ::mkdir(piddir.c_str(), 0777);
        if (FILE* f = std::fopen((piddir + "/rank_" + std::to_string(g_file_rank) + ".pid").c_str(), "w")) {
            std::fprintf(f, "%d", static_cast<int>(::getpid()));
            std::fclose(f);
        }
    }
    mark(
        "start scenario=" + SCENARIO + " victim=" + std::to_string(VICTIM) + " phase=" + PHASE + " coll=" + COLL +
        " on_error=" + ON_ERROR + " ppid=" + std::to_string(::getppid()));
    const double sd = std::stod(env("PROBE_START_DELAY_S", "0"));
    if (sd > 0 && g_rank == std::stoi(env("PROBE_START_DELAY_RANK", "1"))) {
        mark("start_delay " + std::to_string(sd) + "s");
        sleep_s(sd);
    }
    mark("import_done");
    touch("import_done");
    maybe_fault("before_init");
    DistributedContext::create(argc, argv);
    const auto& ctx = *DistributedContext::get_current_world();
    g_rank = *ctx.rank();
    g_size = *ctx.size();
    mark("ctx_init rank=" + std::to_string(g_rank) + " size=" + std::to_string(g_size));
    timed("barrier_init", [&] { ctx.barrier(); });
    maybe_fault("after_init");

    // ---- open: the control plane's distributed init (the mesh probe only) ----
    int rows = std::stoi(env("PROBE_MESH_ROWS", "8"));
    int cols = std::stoi(env("PROBE_MESH_COLS", "4"));
    const std::string victim_shape = env("PROBE_MESH_SHAPE_VICTIM", "");
    if (is_victim() && !victim_shape.empty()) {
        rows = std::stoi(victim_shape.substr(0, victim_shape.find('x')));
        cols = std::stoi(victim_shape.substr(victim_shape.find('x') + 1));
    }
#ifdef PROBE_MESH
    using namespace tt::tt_metal::distributed;
    const bool fabric_on = FABRIC_RANK != "-1" && g_rank == std::stoi(FABRIC_RANK);
    if (fabric_on) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
        mark("fabric_config_set FABRIC_2D");
    }
    std::thread in_open_fault;
    if ((SCENARIO == "throw" || SCENARIO == "abort" || SCENARIO == "exit_nofinalize") && is_victim() &&
        PHASE == "in_open") {
        in_open_fault = std::thread([] { do_fault(); });  // a throw on this thread is std::terminate too
        in_open_fault.detach();
    }
    auto t_open = std::chrono::steady_clock::now();
    mark("open_enter shape=" + std::to_string(rows) + "x" + std::to_string(cols));
    touch("open_enter");
    std::shared_ptr<MeshDevice> mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(rows, cols)));
    mark(
        "open_done dt=" +
        std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_open).count()));
#else
    mark("open_enter shape=" + std::to_string(rows) + "x" + std::to_string(cols) + " (no mesh)");
    touch("open_enter");
    mark("open_done dt=0.000");
#endif
    timed("barrier_open", [&] { ctx.barrier(); });

    // ---- the collectives section: the fault lands here by default ----
    maybe_fault("after_open");
    touch("coll_ready");
    collectives(ctx);

    auto t_close = std::chrono::steady_clock::now();
    mark("close_enter");
#ifdef PROBE_MESH
    mesh->close();
    mesh.reset();
    if (fabric_on) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
#endif
    mark(
        "close_done dt=" +
        std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_close).count()));
    timed("barrier_close", [&] { ctx.barrier(); });
    if (SCENARIO == "slow_exit" && is_victim()) {
        const double slow = std::stod(env("PROBE_SLOW_S", "8"));
        mark("slow_exit: still working " + std::to_string(slow) + " s after the others finished");
        sleep_s(slow);
    }
    mark("main_exit");
    return 0;
}
