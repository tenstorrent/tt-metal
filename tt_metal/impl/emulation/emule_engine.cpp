// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The emule engine — see emule_engine.hpp. Machinery extracted verbatim from
// emulated_program_runner.cpp (the runner is now a thin forwarder over the 6 public symbols).
#include "emule_engine.hpp"
#include "emule_descriptor_builder.hpp"
#include "emule_multi_rank_runtime.hpp"
#include "emule_live_ranges.hpp"
#include "host_sanitizers.hpp"
#include "emule_sanitizers.hpp"
#include "emule_jit.hpp"
#include "emule_program_model.hpp"
#include "emule_cb_dfb_setup.hpp"
#include "emule_device_map.hpp"
#include "emule_noc_bridge.hpp"  // extern-C mem/NOC bridge + fiber thunks (moved out); emule_require_self, my_x/my_y
#include "emule_fabric.hpp"
#include "emule_diagnostics.hpp"
#include "emule_tile_geometry.hpp"
#include "emule_kernel_defines.hpp"
#include "emule_metal2_emit.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/resource.h>  // getrlimit(RLIMIT_NOFILE) — bound JIT compile fan-out under the fd limit
#include <csignal>
#if defined(__x86_64__) && defined(__linux__)
#include <ucontext.h>
#include <sys/ucontext.h>
#endif

#include <bit>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <tt_stl/assert.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <future>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>

#ifndef TT_EMULE_CXX_COMPILER
#error "TT_EMULE_CXX_COMPILER must be defined by CMake"
#endif
#ifndef TT_EMULE_CXX_STANDARD
#error "TT_EMULE_CXX_STANDARD must be defined by CMake"
#endif

#include "impl/kernels/kernel.hpp"
#include "jit_build/jit_build_settings.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_utils.hpp"  // format_named_ct_arg_map (shared with the silicon JIT path)
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/face_geometry.hpp>  // FaceGeometry (per-CB unpack override)
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/context/metal_context.hpp"
#include "hostdevcommon/fabric_common.h"  // routing_l1_info_t — identity field layout
#include "llrt/metal_soc_descriptor.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "umd/device/chip_helpers/simulation_sysmem_manager.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>  // fabric route table (multi-chip dst resolve)
#include <tt-metalium/experimental/fabric/fabric_types.hpp>   // FabricNodeId, MeshId, FabricConfig
#include <tt-metalium/experimental/fabric/fabric.hpp>         // is_2d_fabric_config
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>     // RoutingDirection
#include "tt_emule/chip_store.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/l1_pool.hpp"
#include "tt_emule/rank_state.hpp"
#include "tt_emule/kernel_patcher.hpp"  // tt::emule::patch_kernel_source (the extracted JIT patch pass)
#include "tt_emule/tile_counter.hpp"
#include "jit_hw/internal/emule_thread_ctx.h"
#include "emule_fiber_scheduler.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/common/stable_hash.hpp"

#ifndef TT_EMULE_JIT_INCLUDE_DIR
#error "TT_EMULE_JIT_INCLUDE_DIR must be defined by CMake (path to tt-emule's include/jit_hw)"
#endif
#ifndef TT_EMULE_INCLUDE_DIR
#error "TT_EMULE_INCLUDE_DIR must be defined by CMake (path to tt-emule's include/)"
#endif

////////////////////////////////////////////////////////////
// Blaze-only experimental named args
// Removal is tracked by issue #50953
namespace tt::tt_metal::experimental::blaze {
bool emit_named_args_header(
    const std::string& dir, const NamedCTArgNamespaces& ct_namespaces, const NamedRuntimeArgNamespaces& rt_namespaces);
}  // namespace tt::tt_metal::experimental::blaze
////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------
// Thread-local context for JIT kernels.
// Exported via -rdynamic so dlopen'd .so files can resolve them at load time.
// ---------------------------------------------------------------------------

// __emule_cb_state is an alias for tt_emule::CBSyncState (see emule_cb_state.h).
// We use the real type directly here.
using __emule_cb_state = tt_emule::CBSyncState;

// The per-RISC identity / handles — rt_args, common_rt_args, core_obj, device,
// bridge_l1/dram, cbs, dfbs, tc_array, processor_id, neo_id, trisc_id,
// num_threads, my_thread_id, core_map — are now fields of the per-thread
// ThreadCommonCtx, reached via __emule_self (defined just below; see
// emule_thread_ctx.h). The runner sets them in the launch lambda; the JIT kernel
// and the extern-C resolvers above read them through __emule_self->X. (The
// mhartid regex now emits `__emule_self->processor_id`; CSR/get_num_threads/
// get_arg shims read the corresponding ctx fields.)

// Per-thread execution context — the single source of truth for an emulated
// RISC's thread-local state, specialized by RISC type (see emule_thread_ctx.h).
// Defined here, exported via -rdynamic so the JIT .so resolves it at dlopen; set
// per kernel thread in the launch lambda below.
thread_local ThreadCommonCtx* __emule_self = nullptr;

// Core execution state (bridge_l1/dram, cbs, dfbs, tc_array, num_threads, my_thread_id, core_map)
// now lives in ThreadCommonCtx, reached via __emule_self and set per-fiber in launch_cores.
//
// These three Quasar identity signals are ALSO kept as -rdynamic globals: the JIT kernel reads them
// from the ctx (__emule_self->{processor_id,neo_id,trisc_id}), but the ASAN sanitizer
// (emule_sanitizers.cpp) reads the globals, so the launch lambda sets both.
//   __processor_id   — RISC-V mhartid analogue (DM index / Neo engine index).
//   __emule_neo_id   — Quasar NEO_ID CSR (0xBC2).
//   __emule_trisc_id — Quasar TRISC_ID CSR (0xBC3); iterated 0..3 across ki.variants for compute.
thread_local uint8_t __processor_id = 0;
thread_local uint8_t __emule_neo_id = 0;
thread_local uint8_t __emule_trisc_id = 0;

// The sanitizer per-launch range/counter state (semaphore region, L1/DRAM OOB +
// padding + host-poke extents, CB window counters + Dirty-CB flags/sites, the
// NoC-read counter, and the Object-Intent resolved-log pointer) now lives in the
// per-fiber context (__emule_self->san, EmuleSanitizerState) — moved out of these
// worker-thread-locals so a kernel that yields under the fiber engine can't have
// it clobbered by a co-scheduled fiber (the all_reduce/global-sem OOB false
// positive). It is armed per launch by set_sanitizer_thread_locals and consumed
// by the kernel-side checks + the two host resolvers below. See docs/ASAN.md
// "Per-fiber sanitizer state". __emule_kernel_name stays a -rdynamic global (the
// launch path owns the backing string); the ASAN trace reads kernel identity
// from __emule_self->san (armed alongside the ranges).
thread_local const char* __emule_kernel_name = nullptr;

// Bank-mapping constants (NUM_NOCS, MAX_NUM_BANKS, NOC_NODE_ID_BITS) and the four
// -rdynamic bank arrays (dram_bank_to_noc_xy, bank_to_dram_offset, l1_bank_to_noc_xy,
// bank_to_l1_offset) now live in emule_device_map.{hpp,cpp} at global scope, unmangled
// for JIT dlsym (included via emule_device_map.hpp above).

thread_local uint32_t __emule_logical_x = 0;
thread_local uint32_t __emule_logical_y = 0;
////////////////////////////////////////////////////////////
// Blaze-only experimental firmware-global shim
// Removal is tracked by issue #50953
// Silicon-named per-core LOGICAL coords (firmware globals `my_logical_x_/y_`,
// mirroring hw/firmware/src/tt-1xx/brisc.cc; declared extern by
// blaze/kernels/kernel_utils.hpp). Defined here so compute (TRISC) kernels that
// reference them link; restored per fiber swap-in by the scheduler's
// install_fiber. The dataflow (NCRISC/BRISC) senders that must read a CORRECT
// per-fiber value instead resolve `my_logical_x_/y_` through the
// dataflow_utils.hpp shadow's per-fiber accessor (__emule_self->core->logical_*),
// so this definition is only a link/fallback anchor on other RISCs.
// These MUST stay at global scope with these exact unmangled names (NOT inside a
// namespace): under emule there is no firmware, and JIT'd kernels are x86-compiled
// and resolve these symbols against libtt_metal via dlopen(-rdynamic) — any
// mangling/rename breaks that lookup.
thread_local uint8_t my_logical_x_ = 0;
thread_local uint8_t my_logical_y_ = 0;
////////////////////////////////////////////////////////////

namespace tt::tt_metal::emule {
namespace engine {
namespace {

// Set while THIS thread holds the dispatch mutex through a MeshDispatchLock. run_mesh_dispatch is
// also reachable from the deferred-flush path with no lock held.
thread_local bool t_holds_dispatch_lock = false;

}  // namespace

// ---------------------------------------------------------------------------
// setup_core_state: Configure CBs and semaphores per core, build CoreSetup list.
// ---------------------------------------------------------------------------
// Initialize a core's CB-sync state: configure each cb_id from the CB that owns
// it locally (a CB whose core_ranges() contain this core).
// CB/DFB/semaphore setup moved to emule_cb_dfb_setup.{hpp,cpp}.

// EmuleOobTensorState, the Object-Intent tracker, the per-kernel sanitizer
// thread-local set/clear, the Dirty-CB sweep, and build_oob_tensor_state now
// live in emule_sanitizers.{hpp,cpp}. See SANITIZER_CHECKS.md.

// ---------------------------------------------------------------------------
// launch_cores: Spawn concurrent threads per core, each runs its kernels.
// SIGFPE -> RISC-V divide/overflow recovery moved to emule_diagnostics.{hpp,cpp}.

// [MESH] Register/run split for concurrent multi-device dispatch: in defer mode each
// execute_program_emulated REGISTERS its fibers (spawn, no run); run_mesh_dispatch then drives ONE
// run_until_idle so all chips' fibers run concurrently. See tt-emule docs/fiber-engine.md.
static bool g_emule_mesh_defer = false;
// Tagged with the dispatch's spawn generation; untagged, RSS grows monotonically with dispatch count.
using MeshDfbKeep = std::vector<std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>>>;
static std::vector<std::pair<uint64_t, MeshDfbKeep>> g_mesh_dfb_keep;
// [MESH] ASAN snapshot keepalive. In defer mode the deferred fibers read the per-launch
// live-range snapshot (via oob.state's pointers) only later, in run_mesh_dispatch — long
// after dispatch_to_device's local OobStateOwner would have been destroyed. Without this
// the range pointers dangle and the checks read freed/reused heap (the all_reduce global-
// semaphore OOB false positive). Holds each device's OobStateOwner alive until the run
// completes; cleared alongside g_mesh_dfb_keep. std::move preserves the heap buffers the
// pointers reference, so the captured views stay valid across the vector's own growth.
static std::vector<std::pair<uint64_t, tt::tt_metal::emule::OobStateOwner>> g_mesh_oob_keep;

// Per generation, not "older than the oldest live": a parked relay pins any age bound for the sequence.
static uint64_t g_mesh_keep_gen = 0;
static void reclaim_dead_mesh_keepalives() {
    // ONE registry scan: the per-candidate query is quadratic in a full mesh's cores x RISCs fibers.
    const auto live_gens = tt::tt_metal::emule_fiber::FiberScheduler::instance().live_spawn_generations();
    const std::unordered_set<uint64_t> live(live_gens.begin(), live_gens.end());
    auto dead = [&live](const auto& kv) { return live.count(kv.first) == 0; };
    g_mesh_dfb_keep.erase(std::remove_if(g_mesh_dfb_keep.begin(), g_mesh_dfb_keep.end(), dead), g_mesh_dfb_keep.end());
    g_mesh_oob_keep.erase(std::remove_if(g_mesh_oob_keep.begin(), g_mesh_oob_keep.end(), dead), g_mesh_oob_keep.end());
}

// Whose programs are in the registry, so one mesh's Finish cannot spend its budget on another's run.
static std::mutex g_emule_run_device_ids_mu;
static std::unordered_set<int> g_emule_run_device_ids;
static void run_device_ids_insert(int id) {
    std::lock_guard<std::mutex> g(g_emule_run_device_ids_mu);
    g_emule_run_device_ids.insert(id);
}
static void run_device_ids_erase(int id) {
    std::lock_guard<std::mutex> g(g_emule_run_device_ids_mu);
    g_emule_run_device_ids.erase(id);
}
static void run_device_ids_clear() {
    std::lock_guard<std::mutex> g(g_emule_run_device_ids_mu);
    g_emule_run_device_ids.clear();
}
// True when the set is empty, or when any of `ids` is in it (i.e. this caller owns the parked run).
static bool run_device_ids_owns(const std::vector<int>& ids) {
    std::lock_guard<std::mutex> g(g_emule_run_device_ids_mu);
    if (g_emule_run_device_ids.empty()) {
        return true;
    }
    for (int id : ids) {
        if (g_emule_run_device_ids.count(id) != 0) {
            return true;
        }
    }
    return false;
}

// Process-lifetime storage for FiberIdentity::kernel_src; node-based, so addresses survive rehash.
static const char* intern_kernel_name(const std::string& name) {
    if (name.empty()) {
        return nullptr;
    }
    static std::mutex mu;
    static std::unordered_set<std::string> table;
    std::lock_guard<std::mutex> g(mu);
    return table.insert(name).first->c_str();
}

// Separate flags preserve the suspension cause; neither set means running. Quiescence is
// published only inside the peer probe, never while the host owns control.
static std::atomic<bool> g_emule_host_wait{false};
static std::atomic<bool> g_emule_peer_wait{false};
static std::atomic<bool> g_emule_pump_in_flight{false};
static std::atomic<uint64_t> g_emule_run_sequence{0};

static void notify_peer_wait_driver();
static void ensure_peer_wait_driver();

static bool emule_run_suspended() { return g_emule_host_wait || g_emule_peer_wait; }

// A PeerWait is itself reason to pump: the scheduler already concluded a peer may still deliver, and
// a cross-process delivery cannot wake this process's scheduler by itself. A HostWait with a live
// peer-fed poller also needs autonomous progress while its host is inside a distributed barrier.
static bool emule_run_needs_peer_pump();

static bool emule_run_has_peer_fed_waiter() {
    return emule_run_suspended() && tt::tt_metal::emule_fiber::FiberScheduler::instance().has_peer_fed_waiter();
}

static void resume_emule_run() {
    tt::tt_metal::emule::multi_rank::rank_state().publish_quiesced(false);
    g_emule_host_wait = false;
    g_emule_peer_wait = false;
    notify_peer_wait_driver();
}

static bool emule_run_needs_peer_pump() {
    return g_emule_peer_wait.load(std::memory_order_acquire) || emule_run_has_peer_fed_waiter();
}

static void suspend_emule_run(tt::tt_metal::emule_fiber::RunOutcome outcome) {
    const bool host = outcome == tt::tt_metal::emule_fiber::RunOutcome::HostWait;
    // Only a rank parked on a PEER is a fixed-point participant. One that went back to its host is
    // running as far as peers are concerned, and counting it as parked would fabricate a fixed point.
    if (host) {
        tt::tt_metal::emule::multi_rank::rank_state().publish_quiesced(false);
    }
    g_emule_host_wait = host;
    g_emule_peer_wait = outcome == tt::tt_metal::emule_fiber::RunOutcome::PeerWait;
    notify_peer_wait_driver();
}

// Serializes worker-pool drivers: pump_device() + MeshDispatchLock; run_persistent() returns at quiescence.
static std::mutex g_emule_run_mu;

// Resolved-program cache — emule's analogue of silicon's is_compiled(): collect + JIT compile + resolve
// run ONCE per program (keyed by ProgramId); every device dispatches against the shared read-only result.
// LRU-bounded as a safety net. See tt-emule docs/fiber-engine.md.
//
// Lock-free by invariant: written only from prepare_program on the sequential mesh-register path — single
// writer, never a fiber. prepare_program asserts this.
struct ResolvedProgram {
    std::map<CoreCoord, std::vector<KernelInfo>> core_kernels;
    uint32_t emule_sem_base = 0;
};
// shared_ptr: every fiber holds a ref, so LRU eviction must drop only the CACHE's or a KernelInfo* dangles.
static std::unordered_map<ProgramId, std::shared_ptr<ResolvedProgram>> g_resolved_programs;
static std::deque<ProgramId> g_resolved_lru;
static constexpr size_t kMaxResolvedPrograms = 256;

// Per-program fabric routing, keyed by ProgramId. record_conn populates the globals g_conn_route/g_mux_dir
// during host program construction, but ttnn's program cache SKIPS construction on a cache hit, so an
// intervening op leaves the globals holding ITS routes. Routing is a property of the program (like the
// compiled kernels), so capture it once (first resolve) and restore it into the globals at each dispatch.
// Deliberately NOT LRU-bounded (unlike g_resolved_programs): tiny, and must outlive kernel-cache eviction so
// a re-resolved program keeps its own routes. g_worker_dir is a run-time cache, re-derived per op.
// See docs/fabric-ccl-emulation.md.
struct ProgramRoutes {
    std::unordered_map<uint32_t, std::vector<ConnRoute>> conn_route;
    std::unordered_map<uint64_t, std::vector<ConnRoute>> worker_conns;
    std::unordered_map<uint64_t, uint32_t> mux_dir;
    std::unordered_map<uint32_t, std::set<uint32_t>> ring_adj;
};
static std::unordered_map<ProgramId, ProgramRoutes> g_program_routes;

static void launch_cores(
    std::vector<CoreSetup>& core_setups,
    uint8_t* dram_data,
    std::unordered_map<uint64_t, tt_emule::Core*>* core_map_ptr,
    ChipId device_id,
    bool defer_run,
    const EmuleOobTensorState& oob_state,
    // What the CoreSetups' KernelInfo pointers point into; each fiber below captures a copy of the owner.
    const std::shared_ptr<ResolvedProgram>& resolved_owner) {
#if defined(__x86_64__) && defined(__linux__)
    EmuleSigfpeGuard sigfpe_guard;
#endif
    // Fiber engine: one cooperatively-scheduled fiber per (core, RISC), multiplexed
    // onto a runtime-sized worker pool (TT_EMULE_FIBER_WORKERS). A blocked fiber parks
    // (yields its worker) instead of blocking an OS thread — no thread ceiling, no spin.
    // See docs/fiber-engine.md.
    auto& sched = tt::tt_metal::emule_fiber::FiberScheduler::instance();

    // The fibers borrow the per-core DFB interface arrays; own them here so they
    // outlive run_until_idle.
    std::vector<std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>>> dfb_keepalive;
    dfb_keepalive.reserve(core_setups.size());

    // Object-Intent (ASAN §12): one tracker per core, owned here so it outlives the
    // fiber run (run_until_idle below, non-deferred single-device path only). The
    // deferred mesh path skips OI — per-fiber ASAN state is single-device-scoped and
    // the trackers must not outlive this frame across a deferred run_mesh_dispatch.
    // Empty (no snapshot/verify cost) when ASAN is off. See tt-emule #241 / docs/ASAN.md.
    std::vector<std::unique_ptr<tt::tt_metal::emule::ObjectIntentTracker>> intent_trackers;
    const bool object_intent_active = !defer_run && oob_state.object_intent_strict;
    if (object_intent_active) {
        intent_trackers.reserve(core_setups.size());
    }

    for (size_t core_idx = 0; core_idx < core_setups.size(); ++core_idx) {
        auto& cs = core_setups[core_idx];
        auto* core = cs.core;
        uint8_t* l1_data = core->l1_data();
        tt_emule::CBSyncState* cb_array = core->cb_sync_array();
        tt_emule::TileCounterArray* tc_array = cs.has_tc_dfbs ? core->tile_counters() : nullptr;
        const uint8_t px = cs.phys_x;
        const uint8_t py = cs.phys_y;
        const uint32_t lx = cs.logical_core.x;
        const uint32_t ly = cs.logical_core.y;

        // Per-core logical coords (shared by all RISC fibers on this core).
        auto& cstate = core->core_state();
        cstate.logical_x = lx;
        cstate.logical_y = ly;

        std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> per_thread_dfbs;
        if (cs.has_tc_dfbs) {
            per_thread_dfbs = build_per_thread_dfb_interfaces(*cs.ki_list, cs.dfb_allocs);
        }

        // Object-Intent: snapshot this core's non-I/O live buffers BEFORE its kernel
        // runs (on the dispatch thread, so L1 still holds the pre-kernel bytes). The
        // fiber(s) below record resolved extents, then verify at exit. Self-gates to a
        // no-op unless single-kernel (the attribution rule). See tt-emule #241.
        tt::tt_metal::emule::ObjectIntentTracker* intent_tracker = nullptr;
        if (object_intent_active) {
            intent_trackers.push_back(std::make_unique<tt::tt_metal::emule::ObjectIntentTracker>());
            intent_tracker = intent_trackers.back().get();
            static const std::vector<uint32_t> kEmptyRtArgs;
            intent_tracker->pre_launch_snapshot(
                oob_state,
                cs.ki_list->size(),
                cs.ki_list->size() == 1 ? (*cs.ki_list)[0].rt_arg_values : kEmptyRtArgs,
                l1_data,
                cs.persistent_cb_ranges,
                lx,
                ly);
        }

        for (size_t kidx = 0; kidx < cs.ki_list->size(); ++kidx) {
            KernelInfo* ki_ptr = &(*cs.ki_list)[kidx];
            auto& ki = *ki_ptr;
            tt_emule::EmuleDFBInterface* dfb_array = cs.has_tc_dfbs ? per_thread_dfbs[kidx].get() : nullptr;

            // Build + populate the fiber-owned ctx (set-once identity). The scheduler
            // repoints __emule_self to this ctx on swap-in; my_x/my_y are restored from
            // the FiberIdentity (they cannot move into the ctx — silicon-named globals).
            std::unique_ptr<ThreadCommonCtx> ctx = ki.is_tensix
                                                       ? std::unique_ptr<ThreadCommonCtx>(new ComputeThreadCtx())
                                                       : std::unique_ptr<ThreadCommonCtx>(new DatamovementThreadCtx());
            ctx->rt_args = (ki.rta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                               ? reinterpret_cast<uint32_t*>(core->l1_ptr(ki.kernel_config_base + ki.rta_offset_in_kc))
                               : nullptr;
            ctx->common_rt_args =
                (ki.crta_offset_in_kc != kRtaCrtaNoArgsSentinel)
                    ? reinterpret_cast<uint32_t*>(core->l1_ptr(ki.kernel_config_base + ki.crta_offset_in_kc))
                    : nullptr;
            // Bounds so out-of-range per-core/common arg reads return 0 (silicon zero-pads
            // the RTA region; emule's mock L1 keeps stale bytes).
            ctx->rt_args_count = ki.num_unique_rt_args;
            ctx->common_rt_args_count = static_cast<uint32_t>(ki.rt_arg_values.size()) - ki.num_unique_rt_args;
            ctx->bridge_l1 = l1_data;
            ctx->l1_size = static_cast<uint32_t>(core->l1_size());
            ctx->bridge_dram = dram_data;
            ctx->cbs = cb_array;
            ctx->dfbs = dfb_array;
            ctx->tc_array = tc_array;
            ctx->processor_id = ki.processor_id;
            ctx->core_obj = core;
            ctx->device = nullptr;
            ctx->chip_id = static_cast<uint32_t>(device_id);
            ctx->core_map = core_map_ptr;
            ctx->neo_id = ki.is_tensix ? ki.processor_id : 0;
            ctx->trisc_id = 0;
            ctx->num_threads = ki.num_threads;
            ctx->my_thread_id = ki.thread_idx;
            ctx->core = &cstate;

            tt::tt_metal::emule_fiber::FiberIdentity id;
            id.phys_x = px;
            id.phys_y = py;
            id.logical_x = lx;
            id.logical_y = ly;
            id.proc_id = ki.processor_id;
            // Interned, not ki.kernel_name.c_str(): the hang dump reads this after the closure is gone.
            id.kernel_src = intern_kernel_name(ki.kernel_name);

            // The fiber entry is the kernel body. __emule_self is set by the scheduler
            // on swap-in; the no-op start-barrier of the OS-thread model is gone (a
            // blocked fiber parks rather than spins, so start order is irrelevant).
            //
            // ASAN sanitizer (#44848) is armed per-kernel here: set_sanitizer_thread_locals /
            // sweep_per_kernel_dirty_cbs / clear all write/read __emule_self->san (the per-fiber
            // sanitizer state), so a kernel that yields mid-body can't have its ranges clobbered by
            // a co-scheduled fiber — the fix for the fiber-engine false positives (the all_reduce /
            // global-semaphore OOB). All inert when TT_METAL_EMULE_ASAN is off — set_sanitizer_thread_locals
            // arms null/zero and the chokepoint early-outs. The by-value oob_state view's snapshot
            // vectors (owned by dispatch_to_device's OobStateOwner) are kept alive past this deferred
            // spawn by g_mesh_oob_keep so the armed range pointers don't dangle before the mesh run.
            sched.spawn(
                [ki_ptr,
                 lx,
                 ly,
                 cb_array,
                 l1_data,
                 intent_tracker,
                 oob_state,
                 // ki_ptr points into it, and the LRU can evict it while this fiber is parked.
                 resolved_owner,
                 sem_base = cs.sem_base,
                 sem_size = cs.sem_size]() {
                    auto& ki = *ki_ptr;
                    __processor_id = ki.processor_id;
                    __emule_neo_id = ki.is_tensix ? ki.processor_id : 0;
                    __emule_trisc_id = 0;
                    __emule_kernel_name = ki.kernel_name.empty() ? nullptr : ki.kernel_name.c_str();
                    // ASAN identity on the fiber's OWN ctx (read by the [ASAN ERROR] trace) —
                    // per-fiber so a co-scheduled fiber can't overwrite it across a yield. lx/ly
                    // also fix the previously-always-zero logical-coord report.
                    __emule_self->san.kernel_name = __emule_kernel_name;
                    __emule_self->san.logical_x = lx;
                    __emule_self->san.logical_y = ly;
                    __emule_self->san.processor_id = ki.processor_id;
                    set_sanitizer_thread_locals(oob_state, sem_base, sem_size);
                    // Arm the Object-Intent resolved-range log in THIS fiber's ctx (reset
                    // count, enable recording). The kernel-side OOB check appends resolved
                    // extents to __emule_self->san_resolved_log; accumulate/verify below.
                    if (intent_tracker != nullptr) {
                        __emule_self->san_resolved_active = true;
                        __emule_self->san_resolved_count = 0;
                    }
                    try {
                        for (size_t t = 0; t < ki.variants.size(); ++t) {
                            if (ki.run_all_variants) {
                                __emule_self->trisc_id = static_cast<uint8_t>(t);
                                __emule_trisc_id = static_cast<uint8_t>(t);
                            }
                            ki.variants[t]();
                        }
                        sweep_per_kernel_dirty_cbs(oob_state, cb_array, ki.processor_id, lx, ly);
                    } catch (...) {
                        if (intent_tracker != nullptr) {
                            intent_tracker->accumulate_resolved(
                                oob_state, __emule_self->san_resolved_log, __emule_self->san_resolved_count);
                            __emule_self->san_resolved_active = false;
                        }
                        clear_sanitizer_thread_locals();
                        std::throw_with_nested(std::runtime_error(
                            "EMULE: kernel on core (" + std::to_string(lx) + "," + std::to_string(ly) + ") failed"));
                    }
                    if (intent_tracker != nullptr) {
                        // Fold this kernel's resolved extents into the core's resolved set, then
                        // verify: any non-I/O live buffer whose bytes changed but was never
                        // resolved is an Object-Intent violation (aborts). No-op on multi-kernel
                        // cores (nothing snapshotted). Runs in-fiber, after the kernel wrote L1.
                        intent_tracker->accumulate_resolved(
                            oob_state, __emule_self->san_resolved_log, __emule_self->san_resolved_count);
                        __emule_self->san_resolved_active = false;
                        // Use the per-fiber kernel name (survives a mid-kernel yield) rather than
                        // the worker-thread_local __emule_kernel_name, which a co-scheduled fiber
                        // could have overwritten — otherwise an OI violation can misattribute.
                        intent_tracker->verify_post_launch(l1_data, lx, ly, __emule_self->san.kernel_name);
                    }
                    __emule_kernel_name = nullptr;
                    clear_sanitizer_thread_locals();
                },
                std::move(ctx),
                id);
        }

        dfb_keepalive.push_back(std::move(per_thread_dfbs));
    }

    if (defer_run) {
        // Mesh register phase: fibers are spawned but not run yet. Keep the DFB arrays they
        // borrow alive until run_mesh_dispatch (the spawned ctx is already owned by the
        // scheduler; core_kernels is kept by execute_program_emulated). The SIGFPE guard
        // above is a no-op here since no kernel runs; run_mesh_dispatch installs its own.
        g_mesh_dfb_keep.emplace_back(g_mesh_keep_gen, std::move(dfb_keepalive));
        return;
    }

    // Run all registered fibers to completion; rethrows the first kernel exception,
    // throws on a quiescent deadlock, aborts with a dump on livelock/hang.
    sched.run_until_idle();
}

// ---------------------------------------------------------------------------
// prepare_program: resolve a program's kernels ONCE (collect + JIT-compile + resolve), memoized by
// ProgramId — emule's analogue of silicon's CompileProgram. The first mesh device resolves; the rest
// reuse, taking its (homogeneous-chip-identical) compile defines. See tt-emule docs/metal-integration.md.
// ---------------------------------------------------------------------------
static std::shared_ptr<ResolvedProgram> prepare_program(IDevice* device, Program& program) {
    // Single-writer invariant for g_resolved_programs/g_resolved_lru: this runs only on the
    // sequential dispatch thread (register phase), never inside a fiber. __emule_self is the
    // running fiber (set on worker threads, null on the dispatch thread), so off-fiber == null.
    TT_FATAL(__emule_self == nullptr, "prepare_program must run on the dispatch path, not a fiber");
    auto& impl = program.impl();
    const ProgramId pid = impl.get_id();
    if (auto it = g_resolved_programs.find(pid); it != g_resolved_programs.end()) {
        return it->second;  // already resolved (peer mesh device or repeated invocation)
    }

    auto device_id = device->id();
    auto* sw_emu = get_sw_emulated_chip(device_id);

    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    uint32_t num_l1_banks = 0;
    const auto emule_soc = tt_emule::build_soc_view(device);
    populate_bank_mapping(sw_emu, emule_soc, dram_core, num_dram_channels, num_l1_banks);
    const auto emule_desc = tt_emule::build_emule_descriptor(program, device);

    std::string worker_col_map_str, worker_row_map_str;
    build_worker_coord_maps(device, worker_col_map_str, worker_row_map_str);

    std::string extra_inc = get_extra_include_flags();

    const auto& hal = MetalContext::instance().hal();
    uint32_t tensix_pct_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    uint32_t kernel_config_base =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG));
    const auto& prog_config = impl.get_program_config(tensix_pct_index);
    uint32_t emule_sem_base = kernel_config_base + prog_config.sem_offset;

    std::map<CoreCoord, std::vector<PendingKernelInfo>> pending_core_kernels;
    std::map<std::string, DeferredCompile> deferred_compiles;
    std::unordered_map<std::string, std::function<void()>> resolved_fns;
    std::vector<std::string> inline_src_temps;
    collect_kernels(
        num_dram_channels,
        num_l1_banks,
        worker_col_map_str,
        worker_row_map_str,
        emule_sem_base,
        extra_inc,
        pending_core_kernels,
        deferred_compiles,
        resolved_fns,
        inline_src_temps,
        emule_desc,
        emule_soc);
    jit_compile_pending(deferred_compiles, resolved_fns, inline_src_temps);

    ResolvedProgram resolved;
    resolved.emule_sem_base = emule_sem_base;
    for (auto& [logical_core, pending_list] : pending_core_kernels) {
        for (auto& pk : pending_list) {
            KernelInfo ki{
                {},
                pk.run_all_variants,
                pk.processor_id,
                pk.thread_idx,
                pk.is_tensix,
                pk.num_threads,
                pk.kernel_config_base,
                pk.rta_offset_in_kc,
                pk.crta_offset_in_kc};
            ki.variants.reserve(pk.variant_cache_keys.size());
            for (const auto& key : pk.variant_cache_keys) {
                ki.variants.push_back(resolved_fns.at(key));
            }
            ki.rt_arg_values = std::move(pk.rt_arg_values);
            ki.num_unique_rt_args = pk.num_unique_rt_args;
            ki.kernel_name = std::move(pk.kernel_name);
            resolved.core_kernels[logical_core].push_back(std::move(ki));
        }
    }
    log_info(
        tt::LogMetal,
        "execute_program_emulated: program {} resolved ({} logical cores)",
        pid,
        resolved.core_kernels.size());

    // Capture once, keyed by pid: on first resolve the globals still hold what record_conn recorded for this
    // program. A later re-resolve (cache-hit, construction skipped) finds the entry present and keeps it.
    {
        std::lock_guard<std::mutex> lk(g_conn_route_mu);
        if (g_program_routes.find(pid) == g_program_routes.end()) {
            g_program_routes[pid] = ProgramRoutes{g_conn_route, g_worker_conns, g_mux_dir, g_ring_adj};
        }
    }

    // LRU-bound the cache; erasing drops only the cache's ref, so a parked run's entry survives eviction.
    if (g_resolved_programs.size() >= kMaxResolvedPrograms && !g_resolved_lru.empty()) {
        g_resolved_programs.erase(g_resolved_lru.front());
        g_resolved_lru.pop_front();
    }
    g_resolved_lru.push_back(pid);
    auto entry = std::make_shared<ResolvedProgram>(std::move(resolved));
    return g_resolved_programs.emplace(pid, std::move(entry)).first->second;
}

// ---------------------------------------------------------------------------
// dispatch_to_device: per-device setup + launch, reusing the program's resolved kernels.
// emule's analogue of dispatching the already-compiled program to one chip. dram_core (the
// chip's DRAM backing) and the bank-table globals are per device.
// ---------------------------------------------------------------------------
static void dispatch_to_device(
    IDevice* device, Program& program, const std::shared_ptr<ResolvedProgram>& resolved_owner, bool defer_run) {
    ResolvedProgram& resolved = *resolved_owner;
    auto device_id = device->id();
    auto* sw_emu = get_sw_emulated_chip(device_id);

    tt_emule::Core* dram_core = nullptr;
    uint32_t num_dram_channels = 0;
    uint32_t num_l1_banks = 0;
    const auto emule_soc = tt_emule::build_soc_view(device);
    const auto emule_desc = tt_emule::build_emule_descriptor(program, device);
    populate_bank_mapping(sw_emu, emule_soc, dram_core, num_dram_channels, num_l1_banks);

    auto* core_map_ptr = build_core_map(sw_emu, device, device_id, emule_soc);
    std::vector<CoreSetup> core_setups;
    setup_core_state(
        device, sw_emu, resolved.core_kernels, resolved.emule_sem_base, emule_soc, emule_desc, core_setups);

    uint8_t* dram_data = dram_core ? dram_core->l1_data() : nullptr;

    OobStateOwner oob = build_oob_tensor_state(device, device_id);
    launch_cores(core_setups, dram_data, core_map_ptr, device_id, defer_run, oob.state, resolved_owner);
    if (defer_run) {
        // Deferred fibers run later (run_mesh_dispatch); keep the snapshot vectors that
        // oob.state's ASAN range pointers reference alive until then. See g_mesh_oob_keep.
        g_mesh_oob_keep.emplace_back(g_mesh_keep_gen, std::move(oob));
    }
}

// ---------------------------------------------------------------------------
// execute_program_emulated: Main entry point. Mirrors silicon's compile-once / dispatch-
// reuse: prepare_program resolves once (memoized by program id), dispatch_to_device runs
// per device against the shared resolved kernels.
// ---------------------------------------------------------------------------
void execute_program_emulated(IDevice* device, Program& program) {
    auto device_id = device->id();
    log_debug(tt::LogMetal, "execute_program_emulated: device {} starting", device_id);

    // STAGE 1: build + discard the full descriptor (validation); its consumers land in later 2b units.
    // SocView is now consumed for real by populate_bank_mapping (prepare_program / dispatch_to_device).
    auto _emule_desc = tt_emule::build_emule_descriptor(program, device);
    (void)_emule_desc;
    // Mark the fabric connection-route table stale: the next op's first connection record clears it, so
    // routes stay scoped to the current op (this op's builds already recorded before this launch).
    g_conn_route_dirty.store(true, std::memory_order_relaxed);

    std::shared_ptr<ResolvedProgram> resolved = prepare_program(device, program);  // compile-once (memoized)

    // Restore this program's routing into the globals before any 1D send resolves, so a program-cache hit
    // reinstates its own directions over an intervening op's. Keyed by pid (never evicted); g_worker_dir is a
    // run-time cache, cleared here to re-derive per op. See ProgramRoutes / docs/fabric-ccl-emulation.md.
    {
        const ProgramId pid = program.impl().get_id();
        std::lock_guard<std::mutex> lk(g_conn_route_mu);
        if (auto rit = g_program_routes.find(pid); rit != g_program_routes.end()) {
            g_conn_route = rit->second.conn_route;
            g_worker_conns = rit->second.worker_conns;
            g_mux_dir = rit->second.mux_dir;
            g_ring_adj = rit->second.ring_adj;
        }
        g_worker_dir.clear();
    }

    const bool defer = g_emule_mesh_defer;  // mesh register phase (the run is deferred)
    // Remember whose fibers are in the registry, so a later Finish knows if the run is its own.
    run_device_ids_insert(static_cast<int>(device_id));
    // The non-deferred path finishes inside dispatch_to_device, so its id must not outlive the call.
    struct DeviceIdScope {
        int id;
        bool armed;
        ~DeviceIdScope() {
            if (armed) {
                run_device_ids_erase(id);
            }
        }
    } id_scope{static_cast<int>(device_id), !defer};
    dispatch_to_device(device, program, resolved, defer);

    if (defer) {
        log_debug(tt::LogMetal, "execute_program_emulated: device {} registered (deferred mesh run)", device_id);
        return;
    }
    log_debug(tt::LogMetal, "execute_program_emulated: device {} done", device_id);
}

// ---------------------------------------------------------------------------
// Mesh register/run split (see header). begin_mesh_dispatch puts execute_program_emulated
// into defer mode; run_mesh_dispatch drives the single concurrent run + frees kept state.
// ---------------------------------------------------------------------------
// MeshDispatchLock ctor/dtor logic (the public RAII type lives in emulated_program_runner.cpp).
void mesh_lock_acquire() {
    g_emule_run_mu.lock();
    t_holds_dispatch_lock = true;
}
void mesh_lock_release() {
    t_holds_dispatch_lock = false;
    g_emule_run_mu.unlock();
}

void begin_mesh_dispatch() {
    g_emule_mesh_defer = true;
    // Tag FROM the scheduler (a parallel counter drifts): a blanket clear frees arrays under live fibers.
    g_mesh_keep_gen = tt::tt_metal::emule_fiber::FiberScheduler::instance().begin_spawn_generation();
    reclaim_dead_mesh_keepalives();
    // Ids from a register phase that threw before launch belong to no run; a parked run keeps its ids.
    if (!emule_run_suspended()) {
        run_device_ids_clear();
    }
}

// Drop a parked run's state; pre: the registry is gone. A stale flag lets a feeder kill the next dispatch.
static void clear_suspended_run_state() {
    resume_emule_run();
    g_emule_mesh_defer = false;
    // Generation-scoped reclamation states the rule even though no generation is live here.
    reclaim_dead_mesh_keepalives();
    run_device_ids_clear();
}

void run_mesh_dispatch() {
    std::unique_lock<std::mutex> dispatch_lock(g_emule_run_mu, std::defer_lock);
    if (!t_holds_dispatch_lock) {
        dispatch_lock.lock();
    }
#if defined(__x86_64__) && defined(__linux__)
    EmuleSigfpeGuard sigfpe_guard;  // the actual kernel run happens here, across all chips
#endif
    // Reset defer + free the kept per-device state even if the run throws. Disarmed while
    // either suspension keeps the state alive across the return to the host.
    struct Cleanup {
        bool armed = true;
        ~Cleanup() {
            // A suspension disarms cleanup because the rank is not finished yet.
            if (armed) {
                clear_suspended_run_state();
            }
            if (std::uncaught_exceptions() > 0) {
                tt::tt_metal::emule::multi_rank::rank_state().note_faulted();
            }
        }
    } cleanup;
    // All devices' fibers were registered (spawned) during the per-device register phase; run them
    // concurrently on the worker pool in one pass. Each fiber's ctx carries its device's
    // core_map/bridge_dram, so cross-chip NOC resolution stays correct. run_persistent (vs
    // run_until_idle) lets a host-interleaved socket program quiesce back to the host mid-run. On a
    // throw the RAII Cleanup above frees the kept state during unwind.
    // Force the rank state up before the run: it is what installs the scheduler's peer probe, and
    // without it a quiescence could never be classified as a PeerWait in the first place.
    auto& scheduler = tt::tt_metal::emule_fiber::FiberScheduler::instance();
    tt::tt_metal::emule::multi_rank::begin_dispatch();
    ensure_peer_wait_driver();
    g_emule_run_sequence.fetch_add(1, std::memory_order_release);
    tt::tt_metal::emule_fiber::RunOutcome oc = scheduler.run_persistent();
    if (oc != tt::tt_metal::emule_fiber::RunOutcome::Completed) {
        cleanup.armed = false;
        suspend_emule_run(oc);
        return;
    }
    // Completed synchronously (no host-fed socket wait): the RAII Cleanup frees the kept state.
}

// pre: g_emule_run_mu held. See pump_device().
static void pump_device_locked() {
    if (!emule_run_suspended()) {
        return;
    }
    struct PumpFlight {
        PumpFlight() { g_emule_pump_in_flight.store(true, std::memory_order_release); }
        ~PumpFlight() { g_emule_pump_in_flight.store(false, std::memory_order_release); }
    } pump_flight;
#if defined(__x86_64__) && defined(__linux__)
    // pump() re-enters kernel bodies; run_mesh_dispatch's guard was destroyed at its HostWait return,
    // so reinstall the SIGFPE->RISC-V divide/overflow handler for the resumed execution.
    EmuleSigfpeGuard sigfpe_guard;
#endif
    try {
        resume_emule_run();
        auto oc = tt::tt_metal::emule_fiber::FiberScheduler::instance().pump();
        if (oc == tt::tt_metal::emule_fiber::RunOutcome::Completed) {
            clear_suspended_run_state();
        } else {
            suspend_emule_run(oc);
        }
    } catch (...) {
        // pump() threw (kernel exception / host-wait stall deadlock) — the scheduler registry is torn
        // down; drop the mesh keepalives + host-wait/defer flags so a later dispatch starts clean.
        //
        // Tell the peers too. This rank is neither done nor quiesced from here on, so without the
        // fault they hold a PeerWait against a rank that has already given up and only escape on the
        // driver's wall-clock timeout. The dispatch path and the driver both publish it; this path
        // was the gap.
        tt::tt_metal::emule::multi_rank::rank_state().note_faulted();
        clear_suspended_run_state();
        throw;
    }
}

static void ensure_peer_wait_driver() {
    tt::tt_metal::emule::multi_rank::ensure_peer_wait_driver({
        .needs_peer_pump = &emule_run_needs_peer_pump,
        .run_sequence = [] { return g_emule_run_sequence.load(std::memory_order_acquire); },
        .invalidate_run_sequence = [] { g_emule_run_sequence.fetch_add(1, std::memory_order_release); },
        .run_mutex = &g_emule_run_mu,
        .pump_locked = &pump_device_locked,
        .clear_suspended_state = &clear_suspended_run_state,
    });
}

static void notify_peer_wait_driver() { tt::tt_metal::emule::multi_rank::notify_peer_wait_driver(); }

void pump_device() {
    // One scheduler quantum. pump() re-polls: the host's credit store was a raw L1 write with no wake.
    std::lock_guard<std::mutex> g(g_emule_run_mu);
    pump_device_locked();
}

// Drain backstop, defaulted above the engine's bound so a wedged run escalates there first. 0 = unbounded.
static uint64_t drain_max_pumps() {
    // Strict: strtoull's unsigned wrap would turn "-1" into ~1.8e19 pumps inside Finish().
    auto parse_u64 = [](const char* name, uint64_t fallback, bool* present) -> uint64_t {
        const char* v = std::getenv(name);
        if (present != nullptr) {
            *present = (v != nullptr && v[0] != '\0');
        }
        if (v == nullptr || v[0] == '\0') {
            return fallback;
        }
        errno = 0;
        char* end = nullptr;
        const unsigned long long n = std::strtoull(v, &end, 10);
        if (end == v || *end != '\0' || errno == ERANGE || std::strchr(v, '-') != nullptr) {
            log_warning(tt::LogMetal, "{}='{}' is not a non-negative integer; using {}", name, v, fallback);
            if (present != nullptr) {
                *present = false;
            }
            return fallback;
        }
        return static_cast<uint64_t>(n);
    };

    // From the engine, not a re-parse: env_size and parse_u64 disagree on 0 and on trailing garbage.
    const uint64_t engine_limit = tt::tt_metal::emule_fiber::FiberScheduler::host_wait_stall_limit();
    // Saturating: a wrapping `engine_limit + 64` inverts the floor and abandons a healthy run.
    const uint64_t floor_pumps = engine_limit > UINT64_MAX - 64 ? UINT64_MAX : engine_limit + 64;

    bool present = false;
    const uint64_t n = parse_u64("TT_EMULE_DRAIN_MAX_PUMPS", floor_pumps, &present);
    if (!present) {
        return floor_pumps;
    }
    if (n == 0) {
        return UINT64_MAX;  // explicit "unbounded" — the engine's own limits still terminate it
    }
    if (n < floor_pumps) {
        log_warning(
            tt::LogMetal,
            "TT_EMULE_DRAIN_MAX_PUMPS={} is below the engine's host-wait stall limit ({}), so a "
            "wedged run will be abandoned still-parked instead of escalating; using it anyway",
            n,
            engine_limit);
    }
    return n;
}

void drain_device(const std::vector<int>& device_ids) {
    if (!emule_run_suspended() && !g_emule_pump_in_flight.load(std::memory_order_acquire)) {
        return;
    }
    // Finish means idle on return, and a run past its termination signal has no credit loop driving it.
    static const uint64_t max_pumps = drain_max_pumps();
    for (uint64_t i = 0; i < max_pumps; ++i) {
        // Re-read every pass under the pumping lock: the run can complete and another mesh park a new one.
        std::lock_guard<std::mutex> g(g_emule_run_mu);
        if (!emule_run_suspended()) {
            break;
        }
        if (!device_ids.empty() && !run_device_ids_owns(device_ids)) {
            return;  // not ours (any more) — leave it to its own Finish
        }
        pump_device_locked();
    }
    // Bound hit, run still parked: hand it to the engine. Locked — this tears down the pump's registry.
    std::lock_guard<std::mutex> g(g_emule_run_mu);
    if (!emule_run_suspended()) {
        return;
    }
    try {
        tt::tt_metal::emule_fiber::FiberScheduler::instance().abandon_host_wait(fmt::format(
            "EMULE fiber engine: drain_device gave up after {} pumps with the run still parked. "
            "The engine's host-wait liveness bound ({} no-progress pumps) never escalated, so some "
            "global progress kept resetting it while the awaited socket never advanced. The usual "
            "cause is host ordering: a Finish/synchronize_device or a device read issued BEFORE the "
            "socket data the parked kernels are waiting for was streamed. On silicon that ordering "
            "hangs; here it is bounded and reported. If the feed is merely slow, raise "
            "TT_EMULE_DRAIN_MAX_PUMPS; to get the engine's own escalation (and its dump) first, "
            "lower TT_EMULE_HOST_WAIT_STALL_LIMIT.",
            max_pumps,
            tt::tt_metal::emule_fiber::FiberScheduler::host_wait_stall_limit()));
    } catch (...) {
        clear_suspended_run_state();
        throw;
    }
    // abandon_host_wait found no registry to tear down, so the flags outlived the run they described.
    clear_suspended_run_state();
}

}  // namespace engine
}  // namespace tt::tt_metal::emule
