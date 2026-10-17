// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Breaks cables, sends frames across each bridged one, checks they land (usage: --help).

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <mpi.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/distributed/experimental/virtual_cables.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/fabric/bridged_cables.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using tt::tt_fabric::FabricNodeId;

namespace {

constexpr std::uint32_t kPayloadTag = 0xC0DE0000u;  // what erisc_bridge_t6_sender.cpp writes past word 2
constexpr std::uint32_t kStatusOffset = 0x4000;     // {stage, frames sent}, as erisc_bridge_bench_producer.hpp
constexpr std::uint32_t kReceiveOffset = 0x8000;    // where each sender's frames land on the far chip
constexpr std::uint32_t kPayload = 14 * 1024;       // per frame; fabric's packet size is raised to fit (default 4352)
constexpr std::uint32_t kVerifySlots = 32;  // --verify: receive ring depth, checked frame by frame on the far core
constexpr std::uint32_t kVerifyStatus = kStatusOffset + 0x10;  // {checked, corrupt, overrun, ready, abort}

constexpr const char* kUsage = R"(usage: test_virtual_cables [options]
Breaks ethernet cables, bridges them through the hosts, sends frames across each one and checks they land.
Under MPI, give every rank the same options.

options:
  --break=<spec>      cables to treat as broken: list | all | cable:<i>[,<j>...] | random:<n>[:<seed>]
                      (list prints every cable with its index and bridges none)
  --downed            bridge cables the factory descriptor expects but the system lacks;
                      with --break, simulates unplugging those cables instead
  --frames=N          frames per direction per cable (default 1000)
  --mib=N             measured MiB per direction instead of --frames; prints MB/s
  --warmup=P          send P% extra frames before the MB/s clock starts (default 0, max 1000)
  --verify            check every frame, in order, on the receiving core
  --timeout=S         give up after S seconds (default 120)
  --relaxed           RELAXED system-health mode (needed for split or multi-mesh setups)
  --fabric=<cfg>      1d | 1d_ring | 2d | 2d_torus_x | 2d_torus_y | 2d_torus_xy (default 2d)
  -h, --help          this text

environment:
  TT_METAL_FACTORY_SYSTEM_DESCRIPTOR_PATH   factory descriptor (the cables the system should have); needed by --downed
  TT_METAL_E2H_BRIDGE_RING_PAGES            frames in flight per channel; pinned memory grows with it. Unset, it
                                            starts at 32 and halves until the pinned region fits ulimit -l
  TT_METAL_E2H_BRIDGE_STALL_SECONDS         abort if frames stop moving this long (default 60, <= 0 disables)
  TT_METAL_E2H_BRIDGE_TEARDOWN_SECONDS      abort if shutting the bridge down takes longer (default 60)
  TT_METAL_E2H_BRIDGE_DOWNED_LINKS=1        same as --downed (for any program, not just this test)
  TT_METAL_E2H_BRIDGE_BREAK_LINKS=<spec>    same as --break (test only)
  TT_MESH_GRAPH_DESC_PATH, TT_MESH_ID, TT_VISIBLE_DEVICES   multi-rank runs: mesh layout, this rank's mesh, its chips

exit status: 0 pass (or list), 1 fail, 2 bad options, 3 timed out; the job ends with 75 if the bridge stalls,
76 if its shutdown hangs.

example: test_virtual_cables --break=cable:0 --mib=256 --warmup=10 --verify
)";

struct Args {
    std::optional<std::string> break_spec;
    bool downed = false;
    std::uint32_t frames = 1000;
    std::uint32_t timeout_s = 120;
    std::uint32_t mib = 0;         // nonzero: measured MiB per lane, which sets frames
    std::uint32_t warmup_pct = 0;  // extra frames sent before the throughput clock starts
    std::uint32_t warmup_frames = 0;
    tt::tt_fabric::FabricConfig fabric = tt::tt_fabric::FabricConfig::FABRIC_2D;
    bool relaxed = false;
    bool verify = false;
};

std::optional<std::string> value_of(std::string_view arg, std::string_view name) {
    return arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '='
               ? std::optional<std::string>(arg.substr(name.size() + 1))
               : std::nullopt;
}

bool parse(int argc, char** argv, Args& a, std::string& err) {
    static const std::vector<std::pair<std::string_view, std::uint32_t Args::*>> kNumbers = {
        {"--frames", &Args::frames},
        {"--timeout", &Args::timeout_s},
        {"--mib", &Args::mib},
        {"--warmup", &Args::warmup_pct}};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (auto v = value_of(arg, "--break")) {
            a.break_spec = *v;
        } else if (arg == "--downed") {
            a.downed = true;
        } else if (const auto num = std::find_if(
                       kNumbers.begin(),
                       kNumbers.end(),
                       [&](const auto& n) { return value_of(arg, n.first).has_value(); });
                   num != kNumbers.end()) {
            const auto v = *value_of(arg, num->first);
            const auto r = std::from_chars(v.data(), v.data() + v.size(), a.*(num->second));
            if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) {
                err = "bad number: " + std::string(arg);
                return false;
            }
        } else if (auto v = value_of(arg, "--fabric")) {
            using tt::tt_fabric::FabricConfig;
            static const std::map<std::string, FabricConfig> kNames = {
                {"1d", FabricConfig::FABRIC_1D},
                {"1d_ring", FabricConfig::FABRIC_1D_RING},
                {"2d", FabricConfig::FABRIC_2D},
                {"2d_torus_x", FabricConfig::FABRIC_2D_TORUS_X},
                {"2d_torus_y", FabricConfig::FABRIC_2D_TORUS_Y},
                {"2d_torus_xy", FabricConfig::FABRIC_2D_TORUS_XY}};
            if (!kNames.contains(*v)) {
                err = "unknown --fabric=" + *v;
                return false;
            }
            a.fabric = kNames.at(*v);
        } else if (arg == "--relaxed") {
            a.relaxed = true;
        } else if (arg == "--verify") {
            a.verify = true;
        } else {
            err = "unknown argument: " + std::string(arg);
            return false;
        }
    }
    const std::uint64_t measured = a.mib != 0 ? (std::uint64_t{a.mib} << 20) / kPayload : a.frames;
    const std::uint64_t total = measured + measured * a.warmup_pct / 100;
    if (measured == 0 || total > 0xFFFFFFFFu || a.warmup_pct > 1000) {
        err = "--frames/--mib must give 1 to 2^32-1 frames (--mib wins over --frames), --warmup at most 1000";
        return false;
    }
    a.frames = static_cast<std::uint32_t>(total);
    a.warmup_frames = static_cast<std::uint32_t>(total - measured);
    return true;
}

using Key = std::pair<FabricNodeId, tt::tt_fabric::chan_id_t>;

// One direction of one bridged cable: this end's router sends, the far end's receives.
struct Lane {
    Key end;
    tt::tt_fabric::experimental::BridgedCableEnd info;
    CoreCoord sender{0, 0};    // on end's chip
    CoreCoord receiver{0, 0};  // on the far chip
};

CoreCoord core_at(std::uint32_t slot, const CoreCoord& grid) {
    return CoreCoord(slot % grid.x, (slot / grid.x) % grid.y);
}

bool tags_ok(const std::vector<std::uint32_t>& w, std::uint32_t words) {
    for (std::uint32_t k = 3; k < words && k < w.size(); ++k) {
        if (w[k] != (kPayloadTag | (k & 0xFFFFu))) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    std::string perr;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "-h" || std::string_view(argv[i]) == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        }
    }
    if (!parse(argc, argv, args, perr)) {
        std::fprintf(stderr, "test_virtual_cables: %s\n\n%s", perr.c_str(), kUsage);
        return 2;
    }

    // Before MeshDevice::create, which picks the cables, builds routers for them and starts the bridge.
    auto& rt = MetalContext::instance().rtoptions();
    if (args.break_spec.has_value()) {
        rt.set_e2h_bridge_break_links(*args.break_spec);
    }
    if (args.downed) {
        rt.set_e2h_bridge_downed_links(true);
    }
    const bool list_only = rt.get_e2h_bridge_break_links() == "list" && !args.downed;
    tt::tt_fabric::FabricRouterConfig router_config;
    router_config.max_packet_payload_size_bytes = kPayload;
    tt::tt_fabric::SetFabricConfig(
        args.fabric,
        args.relaxed ? tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE
                     : tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricManagerMode::DEFAULT,
        router_config);

    auto mesh = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const auto& cp = mesh->impl().metal_env().get_control_plane();
    const auto& ends = cp.bridged_cables().ends;
    int result = 0;

    if (list_only || ends.empty()) {
        if (rank == 0) {
            std::printf(
                list_only ? "VIRTUAL CABLES: listed the cables (see the log above); nothing bridged.\n"
                          : "VIRTUAL CABLES: nothing is bridged (no downed cables found, or no --break/--downed).\n");
        }
        result = list_only ? 0 : 1;
    } else {
        auto* bridge = mesh->impl().virtual_cable_bridge();
        const CoreCoord grid = mesh->compute_with_storage_grid_size();

        // Every lane, in the same order on every rank; each takes the next free core on its chip.
        std::vector<Lane> lanes;
        std::map<FabricNodeId, std::uint32_t> senders_on;
        std::map<FabricNodeId, std::uint32_t> receivers_on;
        for (const auto& [key, info] : ends) {
            lanes.push_back(Lane{
                .end = key,
                .info = info,
                .sender = core_at(senders_on[key.first]++, grid),
                .receiver = core_at(receivers_on[info.peer_node]++, grid)});
        }
        std::vector<int> routed(lanes.size(), 0);  // set by the sending rank, then shared

        std::map<FabricNodeId, MeshCoordinate> local;
        for (const auto& coord : MeshCoordinateRange(mesh->shape())) {
            if (mesh->is_local(coord)) {
                local.emplace(mesh->get_fabric_node_id(coord), coord);
            }
        }

        // Clear every receive slot this rank hosts before any rank sends.
        const auto base = static_cast<std::uint32_t>(mesh->allocator()->get_base_allocator_addr(HalMemType::L1));
        const std::uint32_t words = kPayload / sizeof(std::uint32_t);
        const std::uint32_t slots = args.verify ? kVerifySlots : 1;
        std::vector<std::uint32_t> unwritten(words * slots, 0xFFFFFFFFu);
        for (auto& lane : lanes) {
            if (const auto it = local.find(lane.info.peer_node); it != local.end()) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(it->second), lane.receiver, base + kReceiveOffset, unwritten);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // One workload: a program per chip that owns a sending end, a kernel per end.
        MeshWorkload workload;
        std::map<MeshCoordinate, Program> programs;
        std::vector<std::pair<IDevice*, CoreCoord>> local_senders;
        std::vector<std::pair<IDevice*, CoreCoord>> local_checkers;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            auto& lane = lanes[i];
            const auto it = local.find(lane.end.first);
            const auto fwd = it == local.end()
                                 ? std::vector<tt::tt_fabric::chan_id_t>{}
                                 : cp.get_forwarding_eth_chans_to_chip(lane.end.first, lane.info.peer_node);
            const auto pos = std::find(fwd.begin(), fwd.end(), lane.end.second);
            if (pos == fwd.end()) {
                continue;  // not this rank's, or fabric does not forward to the far chip on this channel
            }
            routed[i] = 1;
            IDevice* dev = mesh->get_device(it->second);
            auto& prog = programs.try_emplace(it->second, CreateProgram()).first->second;
            const std::vector<std::uint32_t> ct{kPayload, args.frames, base, base + kStatusOffset};
            auto k = CreateKernel(
                prog,
                "tests/tt_metal/distributed/kernels/erisc_bridge_t6_sender.cpp",
                lane.sender,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::NOC_0,
                    .compile_args = ct,
                    .defines = args.verify
                                   ? std::map<std::string, std::string>{{"VC_VERIFY_SLOTS", std::to_string(slots)}}
                                   : std::map<std::string, std::string>{}});
            const auto dst = dev->worker_core_from_logical_core(lane.receiver);
            std::vector<std::uint32_t> rta{
                static_cast<std::uint32_t>(dst.x),
                static_cast<std::uint32_t>(dst.y),
                base + kReceiveOffset,
                static_cast<std::uint32_t>(*lane.info.peer_node.mesh_id),
                static_cast<std::uint32_t>(lane.info.peer_node.chip_id)};
            tt::tt_fabric::append_fabric_connection_rt_args(
                lane.end.first,
                lane.info.peer_node,
                static_cast<std::uint32_t>(std::distance(fwd.begin(), pos)),
                prog,
                lane.sender,
                rta);
            SetRuntimeArgs(prog, k, lane.sender, rta);
            std::vector<std::uint32_t> zero{0u, 0u, 0u};  // {stage, sent, --verify go}
            tt::tt_metal::detail::WriteToDeviceL1(dev, lane.sender, base + kStatusOffset, zero);
            local_senders.emplace_back(dev, lane.sender);
        }
        MPI_Allreduce(MPI_IN_PLACE, routed.data(), static_cast<int>(routed.size()), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // --verify: a checker on each routed lane's receiving core (its second RISC-V), once routed[] is agreed.
        for (std::size_t i = 0; i < lanes.size() && args.verify; ++i) {
            const auto it = local.find(lanes[i].info.peer_node);
            if (routed[i] == 0 || it == local.end()) {
                continue;
            }
            auto& prog = programs.try_emplace(it->second, CreateProgram()).first->second;
            const std::vector<std::uint32_t> ct{
                kPayload, args.frames, base + kReceiveOffset, slots, base + kVerifyStatus};
            CreateKernel(
                prog,
                "tests/tt_metal/distributed/kernels/virtual_cable_verify_receiver.cpp",
                lanes[i].receiver,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = ct});
            std::vector<std::uint32_t> zero(5, 0u);
            tt::tt_metal::detail::WriteToDeviceL1(
                mesh->get_device(it->second), lanes[i].receiver, base + kVerifyStatus, zero);
            local_checkers.emplace_back(mesh->get_device(it->second), lanes[i].receiver);
        }
        for (auto& [coord, prog] : programs) {
            workload.add_program(MeshCoordinateRange(coord, coord), std::move(prog));
        }

        // A stuck bridge would leave Finish() waiting forever.
        std::atomic<bool> done{false};
        std::thread watchdog([&] {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(args.timeout_s + 10);  // lanes first
            while (!done.load()) {
                if (std::chrono::steady_clock::now() > deadline) {
                    std::fprintf(
                        stderr,
                        "VIRTUAL CABLES rank %d: timed out after %u s.\n%s\n",
                        rank,
                        args.timeout_s,
                        bridge->describe().c_str());
                    std::fflush(stderr);
                    std::_Exit(3);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });

        const auto t0 = std::chrono::steady_clock::now();
        const auto timed_out = [&] {
            return std::chrono::steady_clock::now() - t0 > std::chrono::seconds(args.timeout_s);
        };
        auto word = [&](const std::pair<IDevice*, CoreCoord>& at, std::uint32_t addr) {
            std::vector<std::uint32_t> w;
            tt::tt_metal::detail::ReadFromDeviceL1(at.first, at.second, addr, sizeof(std::uint32_t), w);
            return w.empty() ? 0u : w[0];
        };
        if (!programs.empty()) {
            EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);
        }
        if (args.verify) {  // every rank: release the senders only once every checker on every rank is up
            for (const auto& c : local_checkers) {
                while (word(c, base + kVerifyStatus + 12) == 0 && !timed_out()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            std::vector<std::uint32_t> go{1u};
            for (const auto& [dev, core] : local_senders) {
                tt::tt_metal::detail::WriteToDeviceL1(dev, core, base + kStatusOffset + 8, go);
            }
        }
        // Steady-state rate: bytes each sender sent after the slowest passed its warm-up, over that time.
        double steady_s = 0.0;
        std::uint64_t steady_frames = 0;
        if (args.mib != 0 || args.warmup_pct != 0 || args.verify) {
            std::optional<std::chrono::steady_clock::time_point> t_warm;
            std::vector<std::uint32_t> sent(local_senders.size(), 0), at_warm;
            for (;;) {
                std::uint32_t least = args.frames;
                for (std::size_t j = 0; j < local_senders.size(); ++j) {
                    sent[j] = word(local_senders[j], base + kStatusOffset + 4);
                    least = std::min(least, sent[j]);
                }
                const auto now = std::chrono::steady_clock::now();
                if (!t_warm && least >= args.warmup_frames) {
                    t_warm = now;
                    at_warm = sent;
                }
                bool all_done = least >= args.frames;
                for (const auto& c : local_checkers) {
                    all_done = all_done && word(c, base + kVerifyStatus) >= args.frames;
                }
                if (all_done) {
                    steady_s = std::chrono::duration<double>(now - *t_warm).count();
                    for (std::size_t j = 0; j < sent.size(); ++j) {
                        steady_frames += args.frames - at_warm[j];
                    }
                    break;
                }
                if (timed_out()) {
                    std::vector<std::uint32_t> stop{1u};  // lets each checker return, so Finish() does
                    for (const auto& [dev, core] : local_checkers) {
                        tt::tt_metal::detail::WriteToDeviceL1(dev, core, base + kVerifyStatus + 16, stop);
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (!programs.empty()) {
            Finish(mesh->mesh_command_queue());
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Each rank judges its side of each lane: all frames sent, and the last one landed whole (waiting for it).
        std::uint64_t mine[3] = {0, 0, 0};  // lane sides judged, failures, bridge errors
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const auto& lane = lanes[i];
            const auto sender = local.find(lane.end.first);
            const auto receiver = local.find(lane.info.peer_node);
            if (routed[i] == 0 || (sender == local.end() && receiver == local.end())) {
                continue;
            }
            std::string verdict = "ok";
            const std::uint32_t last_at = base + kReceiveOffset + ((args.frames - 1) % slots) * kPayload;
            const std::uint32_t tagged = args.verify ? words - 1 : words;  // --verify puts the index in the last word
            for (std::vector<std::uint32_t> got; receiver != local.end();) {
                tt::tt_metal::detail::ReadFromDeviceL1(
                    mesh->get_device(receiver->second), lane.receiver, last_at, kPayload, got);
                if (!got.empty() && got[0] == args.frames - 1 && tags_ok(got, tagged)) {
                    break;
                }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(args.timeout_s)) {
                    verdict = got.empty() ? "FAIL nothing read back"
                                          : "FAIL last frame landed was " + std::to_string(got[0]) +
                                                (tags_ok(got, tagged) ? "" : " (payload corrupt)");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (sender != local.end()) {
                std::vector<std::uint32_t> st;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    mesh->get_device(sender->second), lane.sender, base + kStatusOffset, 2 * sizeof(std::uint32_t), st);
                if (st.size() != 2 || st[1] != args.frames) {
                    verdict =
                        "FAIL sent " + std::to_string(st.size() == 2 ? st[1] : 0) + "/" + std::to_string(args.frames);
                }
            }
            if (args.verify && receiver != local.end() && verdict == "ok") {
                std::vector<std::uint32_t> v;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    mesh->get_device(receiver->second),
                    lane.receiver,
                    base + kVerifyStatus,
                    3 * sizeof(std::uint32_t),
                    v);
                if (v.size() != 3 || v[0] != args.frames || v[1] != 0 || v[2] != 0) {
                    verdict = v.size() != 3 ? "FAIL verify status unreadable"
                                            : "FAIL verify checked " + std::to_string(v[0]) + "/" +
                                                  std::to_string(args.frames) + ", corrupt " + std::to_string(v[1]) +
                                                  ", overwritten before checked " + std::to_string(v[2]) +
                                                  (v[1] == 0 ? " (checker lagged, not data corruption)" : "");
                }
            }
            ++mine[0];
            mine[1] += verdict == "ok" ? 0 : 1;
            std::printf(
                "VIRTUAL CABLE rank %d: %u:%u chan %u -> %u:%u chan %u (rank %u%s): %s\n",
                rank,
                *lane.end.first.mesh_id,
                lane.end.first.chip_id,
                static_cast<unsigned>(lane.end.second),
                *lane.info.peer_node.mesh_id,
                lane.info.peer_node.chip_id,
                static_cast<unsigned>(lane.info.peer_chan),
                lane.info.peer_rank,
                lane.info.downed ? ", downed" : ", injected",
                verdict.c_str());
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        done = true;
        watchdog.join();
        std::printf("%s\n", bridge->describe().c_str());
        if (steady_s > 0.0 && args.mib + args.warmup_pct != 0 &&
            !local_senders.empty()) {  // payload bytes, MB = 1e6 B as the benchmark
            const double mbps = static_cast<double>(steady_frames) * kPayload / 1e6 / steady_s;
            std::printf(
                "VIRTUAL CABLES rank %d throughput: %.0f MB/s per lane, %.0f MB/s over %zu lane(s), %u warm-up frames, "
                "%.2f s\n",
                rank,
                mbps / static_cast<double>(local_senders.size()),
                mbps,
                local_senders.size(),
                args.warmup_frames,
                steady_s);
        }
        std::fflush(stdout);

        mine[2] = bridge->first_error().empty() ? 0 : 1;
        std::uint64_t total[3] = {0, 0, 0};
        MPI_Allreduce(mine, total, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        const bool pass = total[0] > 0 && total[1] == 0 && total[2] == 0;
        if (rank == 0) {
            std::printf(
                "VIRTUAL CABLES %s: %zu cable(s) bridged, %llu lane check(s), %llu failed, %llu unrouted, %u frames x "
                "%u B per lane, %.2f s, %d rank(s).\n",
                pass ? "PASS" : "FAIL",
                cp.bridged_cables().cables.size(),
                static_cast<unsigned long long>(total[0]),
                static_cast<unsigned long long>(total[1]),
                static_cast<unsigned long long>(std::count(routed.begin(), routed.end(), 0)),
                args.frames,
                kPayload,
                secs,
                world);
        }
        result = pass ? 0 : 1;
    }

    mesh->close();
    mesh.reset();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    return result;  // no MPI_Finalize: tt-metal registered its own
}
