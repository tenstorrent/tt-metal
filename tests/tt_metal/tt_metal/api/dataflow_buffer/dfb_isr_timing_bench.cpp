// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dfb_isr_timing_bench: workloads for characterising DM0's DFB implicit-sync ISR on Quasar (issue #55788).
//
// This binary only RUNS the workloads. It carries no device-side timing: every number is measured by
// post-processing emulator waveforms (quasar-isr-latency/ next to the tt-metal checkout). To make that
// possible it writes a manifest (--manifest) describing, per configuration and launch, which txn ids,
// thresholds, tile counters, DMs and Neos each DFB uses, in launch order.
//
// Design: dfb_isr_timing_bench_design.md (next to the tt-metal checkout).
//
// Usage:
//   export TT_METAL_SLOW_DISPATCH_MODE=1
//   ./build/test/tt_metal/dfb_isr_timing_bench --list --suite b2            # print configurations, no device
//   ./build/test/tt_metal/dfb_isr_timing_bench --suite b1 --manifest b1.json [--filter s] [--launches 2]
//
// Configurations that the host rejects (txn-id divisibility, TC / txn-id / DM budgets, L1) are logged as
// SKIP with the reason and do not stop the run.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {
namespace {

namespace ex = experimental;

constexpr const char* kProducerSrc = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_isr_bench_dm_producer.cpp";
constexpr const char* kConsumerSrc = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_isr_bench_dm_consumer.cpp";
constexpr const char* kTriscSrc = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_isr_bench_trisc.cpp";

constexpr uint32_t kMaxBenchDfbs = 12;         // matches isr_bench::kMaxBenchDfbs in the kernels
constexpr uint32_t kWorkerDms = 6;             // DM2..DM7 (DM0/DM1 are reserved)
constexpr uint32_t kNeos = 4;                  // TRISC threads per Tensix
constexpr uint32_t kDfbL1BudgetBytes = 3u << 20;  // all DFBs on a core; free L1 ~3.6 MB, top kept clear
constexpr uint32_t kNocMaxBurstBytes = 64u * 1024u;  // NOC_MAX_BURST_SIZE on Quasar (256 words x 256 B)
// NOC_OVERLAY_MAX_BYTES_IN_PACKET: the overlay splits larger transactions into 8 KB packets. Implicit-sync
// transactions larger than this hung on the emulator (2026-11-03: 16 KB hung with 1, 4 and 8 KB passing; the
// ISR never credited the first section), so implicit flows stay at or below one packet.
constexpr uint32_t kOverlayPacketBytes = 8u * 1024u;
constexpr tt::DataFormat kBenchDataFormat = tt::DataFormat::Float16_b;

// Producer kernel modes (dfb_isr_bench_dm_producer.cpp)
constexpr uint32_t kProdReadImplicit = 0, kProdFillExplicit = 1, kProdRawRead = 2;
// Consumer kernel modes (dfb_isr_bench_dm_consumer.cpp)
constexpr uint32_t kConsWriteImplicit = 0, kConsSinkExplicit = 1, kConsIdle = 2;
// TRISC kernel roles (dfb_isr_bench_trisc.cpp)
constexpr uint32_t kTriscConsumer = 0, kTriscProducer = 1;

enum class Flow {
    READ,   // DRAM -> DM producer (implicit, ISR posts) -> DFB -> sink (TRISC unpacker or DM explicit)
    WRITE,  // source (TRISC packer or DM explicit) -> DFB -> DM consumer (implicit, ISR acks) -> DRAM
    LOOP,   // DRAM -> DM producer (implicit) -> DFB -> DM consumer (implicit) -> DRAM: both ISR paths
    RAW,    // B0: plain NoC reads DRAM -> L1 with trid-tagged sections, no tile counters, no ISR
};
enum class Peer { DM, TRISC };
enum class Pat { STRIDED, ALL };  // BLOCKED is deliberately not exercised by this bench

const char* to_str(Flow f) {
    switch (f) {
        case Flow::READ: return "read";
        case Flow::WRITE: return "write";
        case Flow::LOOP: return "loop";
        case Flow::RAW: return "raw";
    }
    return "?";
}
const char* to_str(Peer p) { return p == Peer::DM ? "dm" : "trisc"; }
const char* to_str(Pat p) {
    switch (p) {
        case Pat::STRIDED: return "strided";
        case Pat::ALL: return "all";
    }
    return "?";
}

struct BenchCfg {
    std::string suite;
    std::string name;
    Flow flow = Flow::READ;
    Peer peer = Peer::DM;  // READ: the sink. WRITE: the source. LOOP/RAW: unused (DM).
    uint32_t entry_size = 1024;
    uint32_t sections = 2;             // txn ids per implicit side (forced); 0 = host default
    uint32_t entries_per_section = 4;  // num_entries = sections * entries_per_section (per producer thread x TCs)
    Pat pap = Pat::STRIDED;
    Pat cap = Pat::STRIDED;
    uint32_t producer_threads = 1;
    uint32_t consumer_threads = 1;
    uint32_t dfbs_per_core = 1;
    uint32_t dfbs_per_producer_kernel = 1;
    uint32_t dfbs_per_consumer_kernel = 1;
    uint32_t num_cores = 1;
    uint32_t min_laps = 8;              // stream at least this many laps of the ring...
    uint32_t min_stream_bytes = 256 * 1024;  // ...and at least this many bytes per DFB,
    uint32_t max_stream_bytes = 2u << 20;    // but no more than this (big rings then get fewer laps, >= 2)
    // RAW (B0) only: sections of the L1 ring and NoC transactions per section (in flight = product).
    uint32_t raw_sections = 2;
    uint32_t raw_ops_per_section = 1;

    uint32_t num_entries() const {
        const uint32_t s = sections == 0 ? 2 : sections;
        return s * entries_per_section;
    }
    uint32_t dfb_bytes() const { return num_entries() * entry_size; }
};

// Entries each DFB streams per launch: whole laps, at least min_stream_bytes, and divisible by the
// thread counts so every thread does a whole number of ops.
uint32_t stream_entries(const BenchCfg& c) {
    const uint32_t ring = c.num_entries();
    uint32_t laps = std::max(c.min_laps, (c.min_stream_bytes + c.dfb_bytes() - 1) / c.dfb_bytes());
    laps = std::min(laps, std::max(2u, c.max_stream_bytes / c.dfb_bytes()));
    const uint32_t g = std::lcm(
        std::lcm(c.producer_threads, c.consumer_threads), ring);
    uint32_t n = laps * ring;
    n = ((n + g - 1) / g) * g;
    return n;
}

std::string describe(const BenchCfg& c) {
    return fmt::format(
        "flow={} peer={} entry={} sections={} eps={} num_entries={} pap={} cap={} P={} C={} "
        "dfbs={} dfbs/pk={} dfbs/ck={} cores={} stream={} raw_sec={} raw_ops={}",
        to_str(c.flow),
        to_str(c.peer),
        c.entry_size,
        c.sections,
        c.entries_per_section,
        c.num_entries(),
        to_str(c.pap),
        to_str(c.cap),
        c.producer_threads,
        c.consumer_threads,
        c.dfbs_per_core,
        c.dfbs_per_producer_kernel,
        c.dfbs_per_consumer_kernel,
        c.num_cores,
        stream_entries(c),
        c.raw_sections,
        c.raw_ops_per_section);
}

// Host-side checks the device API does not do for us (the rest is left to the DFB host code, whose
// TT_FATALs are caught and reported as SKIP).
std::optional<std::string> precheck(const BenchCfg& c) {
    if (c.dfbs_per_core == 0 || c.dfbs_per_core > kMaxBenchDfbs) {
        return "dfbs_per_core must be 1..12";
    }
    if (c.dfbs_per_core % c.dfbs_per_producer_kernel != 0 || c.dfbs_per_core % c.dfbs_per_consumer_kernel != 0) {
        return "dfbs_per_core must be a multiple of the DFBs per kernel";
    }
    const uint32_t prod_kernels = c.dfbs_per_core / c.dfbs_per_producer_kernel;
    const uint32_t cons_kernels = c.dfbs_per_core / c.dfbs_per_consumer_kernel;
    const bool producer_dm = c.flow != Flow::WRITE || c.peer == Peer::DM;
    // READ and RAW use the peer as the consumer (RAW's consumer only lends the ring and does nothing).
    const bool consumer_dm = c.flow == Flow::WRITE || c.flow == Flow::LOOP || c.peer == Peer::DM;
    const uint32_t dm_threads = (producer_dm ? prod_kernels * c.producer_threads : 0) +
                                (consumer_dm ? cons_kernels * c.consumer_threads : 0);
    const uint32_t neo_threads = (producer_dm ? 0 : prod_kernels * c.producer_threads) +
                                 (consumer_dm ? 0 : cons_kernels * c.consumer_threads);
    if (dm_threads > kWorkerDms) {
        return fmt::format("needs {} DM threads, only {} worker DMs", dm_threads, kWorkerDms);
    }
    if (neo_threads > kNeos) {
        return fmt::format("needs {} TRISC threads, only {} Neos", neo_threads, kNeos);
    }
    if (static_cast<uint64_t>(c.dfb_bytes()) * c.dfbs_per_core > kDfbL1BudgetBytes) {
        return fmt::format("DFBs need {} B of L1 per core, budget {}", c.dfb_bytes() * c.dfbs_per_core, kDfbL1BudgetBytes);
    }
    if (c.entry_size > kNocMaxBurstBytes) {
        return "entry (= one NoC transaction) exceeds one NoC burst (64 KB)";
    }
    if (false && c.flow != Flow::RAW && c.entry_size > kOverlayPacketBytes) {  // PKT_THRESH experiment: host threshold now counts packets
        return "implicit-sync transaction exceeds one overlay packet (8 KB): the ISR never credits it (hangs)";
    }
    if (c.flow == Flow::RAW && c.dfbs_per_core != 1) {
        return "RAW uses one ring per core";
    }
    if (c.pap == Pat::ALL) {
        return "ALL is a consumer pattern";
    }
    if (c.cap == Pat::ALL && (c.flow == Flow::WRITE || c.flow == Flow::LOOP)) {
        return "ALL consumers writing to DRAM would each write every page";
    }
    // main does not support DM producer -> DM ALL consumers with implicit sync; its own tests skip it
    // (test_dataflow_buffer_multinode.cpp "legacy parity"). It hung here on the emulator (2026-11-03).
    if (c.cap == Pat::ALL && c.flow == Flow::READ && c.peer == Peer::DM) {
        return "DM -> DM ALL with implicit sync is unsupported on main";
    }
    return std::nullopt;
}

ex::KernelSpec::DFBBinding binding(const ex::DFBSpecName& dfb, const std::string& accessor, bool producer, const BenchCfg& c) {
    const Pat p = producer ? c.pap : c.cap;
    if (producer) {
        return ex::ProducerOf(dfb, accessor);
    }
    return p == Pat::ALL ? ex::AllConsumerOf(dfb, accessor) : ex::StridedConsumerOf(dfb, accessor);
}

tt::tt_metal::TensorSpec flat_dram_spec(uint32_t page_bytes, uint32_t pages) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{pages, page_bytes / static_cast<uint32_t>(sizeof(uint32_t))}, layout);
}

struct BenchContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    IDevice* device = nullptr;
};

struct BuiltProgram {
    Program program;
    std::vector<CoreCoord> cores;
};

// Everything a launch needs that outlives the Program (tensors, the expected output).
struct Workload {
    std::optional<MeshTensor> in_tensor;
    std::optional<MeshTensor> out_tensor;
    std::vector<uint32_t> input;
    bool verify_output = false;
};

std::vector<CoreCoord> pick_cores(const BenchContext& ctx, uint32_t n) {
    const CoreCoord grid = ctx.device->compute_with_storage_grid_size();
    std::vector<CoreCoord> cores;
    for (uint32_t y = 0; y < grid.y && cores.size() < n; y++) {
        for (uint32_t x = 0; x < grid.x && cores.size() < n; x++) {
            cores.emplace_back(x, y);
        }
    }
    TT_FATAL(cores.size() == n, "need {} worker cores, grid is {}x{}", n, grid.x, grid.y);
    return cores;
}

Workload make_workload(BenchContext& ctx, const BenchCfg& c, bool compile_only) {
    Workload w;
    const uint32_t n_stream = stream_entries(c);
    const uint32_t page = c.entry_size;  // one STRIDED NoC transaction = one entry = one DRAM page
    const uint32_t dfbs_total = c.dfbs_per_core * c.num_cores;
    const bool producer_reads = c.flow == Flow::READ || c.flow == Flow::LOOP || c.flow == Flow::RAW;
    const bool consumer_writes = c.flow == Flow::WRITE || c.flow == Flow::LOOP;
    // Kernels always bind their tensor; a side that does not touch DRAM gets a one-page dummy.
    const uint32_t in_pages = producer_reads ? dfbs_total * n_stream : 1;
    const uint32_t out_pages = consumer_writes ? dfbs_total * n_stream : 1;
    w.in_tensor = MeshTensor::allocate_on_device(*ctx.mesh_device, flat_dram_spec(page, in_pages));
    w.out_tensor = MeshTensor::allocate_on_device(*ctx.mesh_device, flat_dram_spec(page, out_pages));
    // LOOP copies page for page, so the output is checkable (ALL consumers would each write every page).
    w.verify_output = !compile_only && c.flow == Flow::LOOP && c.cap != Pat::ALL;
    if (w.verify_output) {
        w.input.resize(static_cast<size_t>(in_pages) * page / sizeof(uint32_t));
        for (size_t i = 0; i < w.input.size(); i++) {
            w.input[i] = 0x15A00000u + static_cast<uint32_t>(i);
        }
        slow_dispatch::WriteToBuffer(w.in_tensor->mesh_buffer(), w.input);
        // TODO #38042: the write barrier is not uplifted for Quasar; wait for the DRAM write to land.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::vector<uint32_t> written;
        slow_dispatch::ReadFromBuffer(w.in_tensor->mesh_buffer(), written);
        TT_FATAL(written == w.input, "input tensor did not land in DRAM before launch");
    }
    return w;
}

BuiltProgram build_program(BenchContext& ctx, const BenchCfg& c, Workload& w) {
    const std::vector<CoreCoord> cores = pick_cores(ctx, c.num_cores);
    const uint32_t n_stream = stream_entries(c);
    const uint32_t K = c.dfbs_per_core;
    const uint32_t Mp = c.dfbs_per_producer_kernel;
    const uint32_t Mc = c.dfbs_per_consumer_kernel;
    const bool producer_dm = c.flow != Flow::WRITE || c.peer == Peer::DM;
    // READ and RAW use the peer as the consumer (RAW's consumer only lends the ring and does nothing).
    const bool consumer_dm = c.flow == Flow::WRITE || c.flow == Flow::LOOP || c.peer == Peer::DM;
    const ex::TensorParamName IN{"in_tensor"};
    const ex::TensorParamName OUT{"out_tensor"};

    // Per-thread entries: STRIDED threads split the stream, ALL consumers each see all of it.
    const uint32_t prod_entries_per_thread = n_stream / c.producer_threads;
    const uint32_t cons_entries_per_thread =
        c.flow == Flow::RAW ? 0u : (c.cap == Pat::ALL ? n_stream : n_stream / c.consumer_threads);

    std::vector<ex::DataflowBufferSpec> dfb_specs;
    for (uint32_t d = 0; d < K; d++) {
        dfb_specs.push_back(ex::DataflowBufferSpec{
            .unique_id = ex::DFBSpecName{fmt::format("dfb{}", d)},
            .entry_size = c.entry_size,
            .num_entries = c.num_entries(),
            .data_format_metadata = kBenchDataFormat,
        });
    }

    std::vector<ex::KernelSpec> kernels;
    std::vector<ex::KernelSpecName> kernel_names;
    std::vector<ex::KernelSpecName> dm_kernel_names;  // kernels that take the core_idx runtime arg

    auto add_kernel = [&](bool producer, uint32_t k) {
        const uint32_t per_kernel = producer ? Mp : Mc;
        const bool is_dm = producer ? producer_dm : consumer_dm;
        ex::KernelSpec ks;
        ks.unique_id = ex::KernelSpecName{fmt::format("{}{}", producer ? "prod" : "cons", k)};
        ks.num_threads = producer ? c.producer_threads : c.consumer_threads;
        for (uint32_t j = 0; j < per_kernel; j++) {
            const uint32_t d = k * per_kernel + j;
            ks.dfb_bindings.push_back(binding(dfb_specs[d].unique_id, fmt::format("d{:02}", j), producer, c));
        }
        const uint32_t entries_per_thread = producer ? prod_entries_per_thread : cons_entries_per_thread;
        if (is_dm) {
            ks.source = producer ? kProducerSrc : kConsumerSrc;
            ks.tensor_bindings = {{.tensor_parameter_name = producer ? IN : OUT, .accessor_name = producer ? "src" : "dst"}};
            uint32_t mode = 0;
            bool implicit = true;
            if (producer) {
                mode = c.flow == Flow::RAW ? kProdRawRead : (c.flow == Flow::WRITE ? kProdFillExplicit : kProdReadImplicit);
                implicit = mode == kProdReadImplicit;
            } else {
                mode = c.flow == Flow::RAW ? kConsIdle : (c.flow == Flow::READ ? kConsSinkExplicit : kConsWriteImplicit);
                implicit = mode == kConsWriteImplicit;
            }
            ks.compile_time_args = {
                {"mode", mode},
                {"entries_per_thread", entries_per_thread},
                {"pages_per_dfb", n_stream},
                {"dfb_base", k * per_kernel},
                {"dfbs_per_core", K},
            };
            if (producer) {
                ks.compile_time_args["raw_sections"] = c.raw_sections;
                ks.compile_time_args["raw_ops_per_section"] = c.raw_ops_per_section;
                ks.compile_time_args["raw_txn_bytes"] = c.entry_size;
            }
            ks.runtime_arg_schema = {.runtime_arg_names = {"core_idx"}};
            ex::DataMovementHardwareConfig hw{};
            if (!implicit) {
                hw.config_2xx = ex::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true};
            }
            ks.hw_config = hw;
            dm_kernel_names.push_back(ks.unique_id);
        } else {
            ks.source = kTriscSrc;
            ks.compile_time_args = {
                {"role", producer ? kTriscProducer : kTriscConsumer},
                {"entries_per_thread", entries_per_thread},
            };
            ks.hw_config = ex::ComputeHardwareConfig{};
        }
        kernel_names.push_back(ks.unique_id);
        kernels.push_back(std::move(ks));
    };
    for (uint32_t k = 0; k < K / Mp; k++) {
        add_kernel(/*producer=*/true, k);
    }
    for (uint32_t k = 0; k < K / Mc; k++) {
        add_kernel(/*producer=*/false, k);
    }

    ex::NodeRangeSet nodes;
    {
        std::vector<CoreRange> ranges;
        for (const auto& core : cores) {
            ranges.emplace_back(core, core);
        }
        nodes = CoreRangeSet(ranges);
    }
    ex::ProgramSpec spec{
        .name = c.name,
        .kernels = kernels,
        .dataflow_buffers = dfb_specs,
        .work_units = {{.name = c.name + "_wu", .kernels = kernel_names, .target_nodes = nodes}},
    };

    // Declare only the tensors some kernel binds (a TRISC peer has no DRAM tensor); Metal 2.0 rejects unused ones.
    if (producer_dm) {
        spec.tensor_parameters.push_back({.unique_id = IN, .spec = w.in_tensor->tensor_spec()});
    }
    if (consumer_dm) {
        spec.tensor_parameters.push_back({.unique_id = OUT, .spec = w.out_tensor->tensor_spec()});
    }

    // The section count is read by the DFB host code when the program is finalized.
    MetalContext::instance().rtoptions().set_dfb_force_num_txn_ids(c.sections);
    Program program = ex::MakeProgramFromSpec(*ctx.mesh_device, spec);

    ex::ProgramRunArgs run_args;
    for (const auto& name : kernel_names) {
        ex::ProgramRunArgs::KernelRunArgs kra{.kernel = name};
        if (std::find(dm_kernel_names.begin(), dm_kernel_names.end(), name) != dm_kernel_names.end()) {
            for (uint32_t i = 0; i < cores.size(); i++) {
                ex::AddRuntimeArgsForNode(kra.runtime_arg_values, cores[i], {{"core_idx", i}});
            }
        }
        run_args.kernel_run_args.push_back(std::move(kra));
    }
    if (producer_dm) {
        run_args.tensor_args.emplace(IN, ex::TensorArgument{*w.in_tensor});
    }
    if (consumer_dm) {
        run_args.tensor_args.emplace(OUT, ex::TensorArgument{*w.out_tensor});
    }
    ex::SetProgramRunArgs(program, run_args);

    // Assign TCs and txn ids now (normally done at launch) so the manifest can record them; this is
    // also where an illegal layout throws.
    program.impl().finalize_dataflow_buffer_configs();
    return BuiltProgram{.program = std::move(program), .cores = cores};
}

// ---------------------------------------------------------------------------------------------
// Manifest: everything post-processing needs to map waveform events to this configuration.
// ---------------------------------------------------------------------------------------------
std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            o += '\\';
        }
        o += ch;
    }
    return o + "\"";
}

std::string manifest_entry(
    BenchContext& ctx, const BenchCfg& c, Program& program, const std::vector<CoreCoord>& cores, uint32_t launch,
    uint32_t seq) {
    std::ostringstream o;
    o << "{\"seq\":" << seq << ",\"suite\":" << json_str(c.suite) << ",\"name\":" << json_str(c.name)
      << ",\"launch\":" << launch << ",\"config\":" << json_str(describe(c)) << ",\"flow\":" << json_str(to_str(c.flow))
      << ",\"peer\":" << json_str(to_str(c.peer)) << ",\"entry_size\":" << c.entry_size
      << ",\"num_entries\":" << c.num_entries() << ",\"stream_entries\":" << stream_entries(c) << ",\"cores\":[";
    for (size_t i = 0; i < cores.size(); i++) {
        const CoreCoord v = ctx.device->worker_core_from_logical_core(cores[i]);
        o << (i ? "," : "") << "{\"logical\":[" << cores[i].x << "," << cores[i].y << "],\"virtual\":[" << v.x << ","
          << v.y << "],\"dfbs\":[";
        bool first_dfb = true;
        for (const auto& dfb : program.impl().dataflow_buffers_on_core(cores[i])) {
            o << (first_dfb ? "" : ",") << "{\"id\":" << dfb->id << ",\"slot\":" << dfb->device_slot
              << ",\"producer_risc_mask\":" << dfb->config.producer_risc_mask
              << ",\"consumer_risc_mask\":" << dfb->config.consumer_risc_mask << ",\"capacity\":" << dfb->capacity;
            first_dfb = false;
            auto txn = [&](const char* side, const dfb_txn_id_descriptor_t& t) {
                o << ",\"" << side << "_txn\":{\"txn_ids\":[";
                for (uint32_t i2 = 0; i2 < t.num_txn_ids; i2++) {
                    o << (i2 ? "," : "") << static_cast<uint32_t>(t.txn_ids[i2]);
                }
                o << "],\"threshold\":" << static_cast<uint32_t>(t.num_entries_to_process_threshold)
                  << ",\"entries_per_txn_id\":" << static_cast<uint32_t>(t.num_entries_per_txn_id)
                  << ",\"entries_per_txn_id_per_tc\":" << static_cast<uint32_t>(t.num_entries_per_txn_id_per_tc) << "}";
            };
            txn("producer", dfb->producer_txn_descriptor);
            txn("consumer", dfb->consumer_txn_descriptor);
            o << ",\"risc_tcs\":[";
            bool first_rc = true;
            for (const auto& rc : dfb->compute_per_core_risc_configs(cores[i])) {
                o << (first_rc ? "" : ",") << "{\"risc\":" << static_cast<uint32_t>(rc.risc_id)
                  << ",\"producer\":" << (rc.is_producer ? "true" : "false") << ",\"tcs\":[";
                for (uint32_t t = 0; t < rc.config.num_tcs_to_rr; t++) {
                    const auto ptc = rc.config.packed_tile_counter[t];
                    o << (t ? "," : "") << "[" << static_cast<uint32_t>(::dfb::get_tensix_id(ptc)) << ","
                      << static_cast<uint32_t>(::dfb::get_counter_id(ptc)) << "]";
                }
                o << "]}";
                first_rc = false;
            }
            o << "]}";
        }
        o << "]}";
    }
    o << "]}";
    return o.str();
}

// ---------------------------------------------------------------------------------------------
// Suites (see the design doc). Every list is a starting grid; filter with --filter.
// ---------------------------------------------------------------------------------------------
std::vector<BenchCfg> suite_b0() {
    std::vector<BenchCfg> v;
    const std::vector<std::pair<uint32_t, uint32_t>> inflight = {{1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}};
    for (uint32_t txn : {64u, 256u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 65536u}) {
        for (uint32_t readers : {1u, 2u, 4u, 6u}) {
            for (auto [sec, ops] : inflight) {
                BenchCfg c;
                c.flow = Flow::RAW;
                c.peer = Peer::TRISC;  // idle TRISC consumer, so 6 readers fit the worker DMs
                c.entry_size = txn;
                c.producer_threads = readers;
                c.consumer_threads = 1;
                c.raw_sections = sec;
                c.raw_ops_per_section = ops;
                // Ring = every thread's sections; the kernel splits it per thread.
                c.sections = 0;
                c.entries_per_section = std::max(1u, readers * sec * ops / 2);
                c.name = fmt::format("b0_raw_txn{}_r{}_inflight{}x{}", txn, readers, sec, ops);
                v.push_back(c);
            }
        }
    }
    // Explicit-sync DFB reference is the B2 grid with --explicit (TODO if needed: separate kernel mode).
    return v;
}

std::vector<BenchCfg> suite_b1() {
    std::vector<BenchCfg> v;
    auto base = [] {
        BenchCfg c;
        c.flow = Flow::READ;
        c.peer = Peer::DM;
        c.entry_size = 64;
        c.sections = 2;
        c.entries_per_section = 4;
        c.min_laps = 8;
        c.min_stream_bytes = 0;
        return c;
    };
    // B1 base + B1a: k DFBs fed by one reader thread, drained by one DM sink -> k trids pending together.
    for (uint32_t k : {1u, 2u, 4u, 8u, 12u}) {
        BenchCfg c = base();
        c.dfbs_per_core = k;
        c.dfbs_per_producer_kernel = k;
        c.dfbs_per_consumer_kernel = k;
        c.name = fmt::format("b1a_pending{}", k);
        v.push_back(c);
    }
    // B1b: tile counters per txn descriptor (producer side round-robins / broadcasts over consumer TCs).
    for (uint32_t cons : {1u, 2u, 4u}) {
        for (Peer p : {Peer::DM, Peer::TRISC}) {
            for (Pat cap : {Pat::STRIDED, Pat::ALL}) {
                BenchCfg c = base();
                c.peer = p;
                c.cap = cap;
                c.consumer_threads = cons;
                c.entries_per_section = 4 * cons;
                c.name = fmt::format("b1b_{}_{}_c{}", to_str(p), to_str(cap), cons);
                v.push_back(c);
            }
        }
    }
    // B1c: path. Post (READ) vs ack (WRITE), DM vs TRISC peer, and both in one ISR (LOOP).
    for (Flow f : {Flow::READ, Flow::WRITE}) {
        for (Peer p : {Peer::DM, Peer::TRISC}) {
            BenchCfg c = base();
            c.flow = f;
            c.peer = p;
            c.name = fmt::format("b1c_{}_{}", to_str(f), to_str(p));
            v.push_back(c);
        }
    }
    {
        BenchCfg c = base();
        c.flow = Flow::LOOP;
        c.name = "b1c_loop_dm";
        v.push_back(c);
    }
    return v;
}

std::vector<BenchCfg> suite_b2() {
    std::vector<BenchCfg> v;
    for (Peer p : {Peer::TRISC, Peer::DM}) {
        for (uint32_t entry : {256u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u}) {  // PKT_THRESH
            for (uint32_t sec : {2u, 3u, 4u}) {
                for (uint32_t eps : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 255u}) {
                    BenchCfg c;
                    c.flow = Flow::READ;
                    c.peer = p;
                    c.entry_size = entry;
                    c.sections = sec;
                    c.entries_per_section = eps;
                    c.name = fmt::format("b2_{}_e{}_s{}_eps{}", to_str(p), entry, sec, eps);
                    v.push_back(c);
                }
            }
        }
    }
    return v;
}

// B3 runs around the B2 knee; these defaults bracket it until the knee is known (override with --entry/--eps).
std::vector<BenchCfg> suite_b3() {
    std::vector<BenchCfg> v;
    const uint32_t entry = 2048;
    for (uint32_t eps : {2u, 4u, 8u, 16u}) {
        // B3a: threads on one DFB.
        for (uint32_t t : {1u, 2u, 3u, 6u}) {
            BenchCfg c;
            c.flow = Flow::READ;
            c.peer = t == 6 ? Peer::TRISC : Peer::DM;
            c.entry_size = entry;
            c.producer_threads = t;
            c.entries_per_section = eps * t;
            c.name = fmt::format("b3a_threads{}_{}_eps{}", t, to_str(c.peer), eps * t);
            v.push_back(c);
        }
        // B3b: independent DFBs on one core, all serviced by the one DM0 ISR.
        for (uint32_t k : {1u, 2u, 4u, 6u, 12u}) {
            BenchCfg c;
            c.flow = Flow::READ;
            c.peer = Peer::TRISC;
            c.entry_size = entry;
            c.entries_per_section = eps;
            c.dfbs_per_core = k;
            c.dfbs_per_producer_kernel = k <= 6 ? 1 : k / 6;  // one reader per DFB up to the 6 worker DMs
            c.dfbs_per_consumer_kernel = k;                   // one Neo drains them all
            c.name = fmt::format("b3b_dfbs{}_eps{}", k, eps);
            v.push_back(c);
        }
        // B3c: cores (each has its own DM0 ISR; isolates NoC/DRAM contention).
        for (uint32_t cores : {1u, 2u, 4u, 6u}) {
            BenchCfg c;
            c.flow = Flow::READ;
            c.peer = Peer::TRISC;
            c.entry_size = entry;
            c.entries_per_section = eps;
            c.num_cores = cores;
            c.name = fmt::format("b3c_cores{}_eps{}", cores, eps);
            v.push_back(c);
        }
    }
    return v;
}

std::vector<BenchCfg> suite_b4() {
    std::vector<BenchCfg> v;
    for (uint32_t entry : {1024u, 4096u, 8192u, 16384u, 32768u, 8888u}) {  // PKT_THRESH (8888: odd size test)
        for (uint32_t sec : {2u, 4u}) {
            for (uint32_t eps : {1u, 4u, 16u, 64u}) {
                for (Peer p : {Peer::TRISC, Peer::DM}) {
                    BenchCfg c;
                    c.flow = Flow::WRITE;
                    c.peer = p;
                    c.entry_size = entry;
                    c.sections = sec;
                    c.entries_per_section = eps;
                    c.name = fmt::format("b4_write_{}_e{}_s{}_eps{}", to_str(p), entry, sec, eps);
                    v.push_back(c);
                }
                BenchCfg c;
                c.flow = Flow::LOOP;
                c.entry_size = entry;
                c.sections = sec;
                c.entries_per_section = eps;
                c.name = fmt::format("b4_loop_e{}_s{}_eps{}", entry, sec, eps);
                v.push_back(c);
            }
        }
        // Writer scaling (DM consumers, TRISC packer source).
        for (uint32_t w : {2u, 4u}) {
            BenchCfg c;
            c.flow = Flow::WRITE;
            c.peer = Peer::TRISC;
            c.entry_size = entry;
            c.consumer_threads = w;
            c.entries_per_section = 4 * w;
            c.name = fmt::format("b4_writers{}_e{}", w, entry);
            v.push_back(c);
        }
    }
    return v;
}

// PKT_THRESH proof capture: the shortest implicit-sync stream with a multi-packet transaction. One 16 KB entry
// (2 NoC packets at 8 KB) per section, 2 sections, 2 laps: 4 reads, the second lap reuses each txn id.
std::vector<BenchCfg> suite_min() {
    BenchCfg c;
    c.flow = Flow::READ;
    c.peer = Peer::DM;
    c.entry_size = 16384;
    c.sections = 2;
    c.entries_per_section = 1;
    c.min_laps = 2;
    c.min_stream_bytes = 0;
    c.max_stream_bytes = 0;
    c.name = "min_read_dm_e16384_s2_eps1";
    return {c};
}

// Same, one lap (2 reads): no txn id is reused, so an early interrupt cannot stall the producer on reuse.
std::vector<BenchCfg> suite_min1() {
    BenchCfg c = suite_min()[0];
    c.min_laps = 1;
    c.name = "min1_read_dm_e16384_s2_eps1_1lap";
    return {c};
}

std::vector<BenchCfg> suite_b5() {
    // Access patterns at fixed bytes per section. Varying the entry (= NoC transaction) size at fixed
    // section bytes separates "bigger transactions" from "fewer interrupts": the threshold and the
    // interrupt count per byte stay the same, only the transactions per section change.
    std::vector<BenchCfg> v;
    const uint32_t section_bytes = 32 * 1024;
    for (Pat cap : {Pat::STRIDED, Pat::ALL}) {
        for (uint32_t entry : {1024u, 4096u, 8192u, 16384u, 32768u}) {  // PKT_THRESH
            for (uint32_t readers : {1u, 4u}) {
                for (Peer p : {Peer::DM, Peer::TRISC}) {
                    BenchCfg c;
                    c.flow = Flow::READ;
                    c.peer = p;
                    c.cap = cap;
                    c.entry_size = entry;
                    c.producer_threads = readers;
                    c.consumer_threads = cap == Pat::ALL ? 2 : 1;
                    c.entries_per_section = std::max(readers * c.consumer_threads, section_bytes / entry);
                    c.name = fmt::format("b5_strided_{}_e{}_r{}_{}", to_str(cap), entry, readers, to_str(p));
                    v.push_back(c);
                }
            }
        }
    }
    return v;
}

std::vector<BenchCfg> suite_b6() {
    std::vector<BenchCfg> v;
    for (uint32_t entry : {64u, 256u}) {
        for (uint32_t eps : {1u, 2u, 4u}) {
            for (uint32_t t : {1u, 2u, 3u, 6u}) {
                BenchCfg c;
                c.flow = Flow::READ;
                c.peer = t == 6 ? Peer::TRISC : Peer::DM;
                c.entry_size = entry;
                c.producer_threads = t;
                c.entries_per_section = eps * t;
                c.min_stream_bytes = 0;
                c.min_laps = 32;
                c.name = fmt::format("b6_threads{}_e{}_eps{}", t, entry, eps * t);
                v.push_back(c);
            }
            for (uint32_t k : {2u, 4u, 6u, 12u}) {
                BenchCfg c;
                c.flow = Flow::READ;
                c.peer = Peer::TRISC;
                c.entry_size = entry;
                c.entries_per_section = eps;
                c.dfbs_per_core = k;
                c.dfbs_per_producer_kernel = k <= 6 ? 1 : k / 6;
                c.dfbs_per_consumer_kernel = k;
                c.min_stream_bytes = 0;
                c.min_laps = 32;
                c.name = fmt::format("b6_dfbs{}_e{}_eps{}", k, entry, eps);
                v.push_back(c);
            }
        }
    }
    return v;
}

std::vector<BenchCfg> suite_b7() {
    std::vector<BenchCfg> v;
    for (uint32_t entry : {4096u, 8192u}) {
        for (uint32_t sec : {2u, 4u}) {
            for (uint32_t readers : {1u, 6u}) {
                for (Peer p : {Peer::TRISC, Peer::DM}) {
                    if (readers == 6 && p == Peer::DM) {
                        continue;  // 6 readers + a DM sink exceeds the worker DMs
                    }
                    BenchCfg c;
                    c.flow = Flow::READ;
                    c.peer = p;
                    c.entry_size = entry;
                    c.sections = sec;
                    c.producer_threads = readers;
                    // Largest ring under the L1 budget, capped at 255 entries per section.
                    const uint32_t max_entries = kDfbL1BudgetBytes / entry;
                    uint32_t eps = std::min(255u, max_entries / sec);
                    eps -= eps % readers;
                    c.entries_per_section = eps;
                    c.min_laps = 2;
                    c.min_stream_bytes = 8u << 20;
                    c.max_stream_bytes = 8u << 20;
                    c.name = fmt::format("b7_huge_e{}_s{}_r{}_{}", entry, sec, readers, to_str(p));
                    v.push_back(c);
                }
            }
        }
    }
    return v;
}

const std::map<std::string, std::function<std::vector<BenchCfg>()>>& suites() {
    static const std::map<std::string, std::function<std::vector<BenchCfg>()>> m = {
        {"b0", suite_b0},
        {"b1", suite_b1},
        {"b2", suite_b2},
        {"b3", suite_b3},
        {"b4", suite_b4},
        {"b5", suite_b5},
        {"min", suite_min},
        {"min1", suite_min1},
        {"b6", suite_b6},
        {"b7", suite_b7},
    };
    return m;
}

BenchContext open_device() {
    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE"), "Set TT_METAL_SLOW_DISPATCH_MODE=1");
    TT_FATAL(MetalContext::instance().get_cluster().arch() == ARCH::QUASAR, "dfb_isr_timing_bench requires Quasar");
    std::vector<ChipId> ids;
    for (ChipId id : MetalContext::instance().get_cluster().mmio_chip_ids()) {
        ids.push_back(id);
    }
    TT_FATAL(!ids.empty(), "No MMIO devices available");
    const auto& dispatch_core_config = MetalContext::instance().resolve_dispatch_core_config();
    auto meshes = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
    BenchContext ctx;
    ctx.mesh_device = meshes.begin()->second;
    ctx.device = ctx.mesh_device->impl().get_devices()[0];
    return ctx;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " --suite b0|b1|...|b7 [--list] [--filter SUBSTR] [--launches N] [--manifest PATH] [--control]\n"
                 "  --list       print the suite's configurations (and host prechecks) without opening a device\n"
                 "  --filter     run only configurations whose name contains SUBSTR (repeatable: any match)\n"
                 "  --launches   launches per configuration (default 2: launch 0 is the cold one)\n"
                 "  --manifest   append one JSON line per launch (default dfb_isr_bench_manifest.jsonl)\n"
                 "  --control    repeat the first selected configuration at the end (drift check)\n"
                 "  --compile-only  build and JIT-compile every selected program without launching it\n"
                 "Device code is uninstrumented; capture a waveform of the run and post-process it.\n";
}

}  // namespace
}  // namespace tt::tt_metal

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    std::string suite;
    std::vector<std::string> filters;
    bool list_only = false;
    bool control = false;
    bool compile_only = false;
    uint32_t launches = 2;
    std::string manifest_path = "dfb_isr_bench_manifest.jsonl";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            TT_FATAL(i + 1 < argc, "{} needs a value", a);
            return argv[++i];
        };
        if (a == "--suite") {
            suite = next();
        } else if (a == "--filter") {
            filters.push_back(next());
        } else if (a == "--launches") {
            launches = static_cast<uint32_t>(std::stoul(next()));
        } else if (a == "--manifest") {
            manifest_path = next();
        } else if (a == "--list") {
            list_only = true;
        } else if (a == "--control") {
            control = true;
        } else if (a == "--compile-only") {
            compile_only = true;
        } else {
            print_usage(argv[0]);
            return a == "--help" || a == "-h" ? 0 : 1;
        }
    }
    const auto it = suites().find(suite);
    if (it == suites().end()) {
        print_usage(argv[0]);
        return 1;
    }
    std::vector<BenchCfg> cfgs;
    for (BenchCfg c : it->second()) {
        c.suite = suite;
        const bool keep = filters.empty() || std::any_of(filters.begin(), filters.end(), [&](const std::string& f) {
                              return c.name.find(f) != std::string::npos;
                          });
        if (keep) {
            cfgs.push_back(c);
        }
    }
    if (control && !cfgs.empty()) {
        BenchCfg c = cfgs.front();
        c.name += "_control";
        cfgs.push_back(c);
    }

    if (list_only) {
        uint32_t ok = 0;
        for (const auto& c : cfgs) {
            const auto bad = precheck(c);
            std::cout << (bad ? "SKIP " : "OK   ") << c.name << "  " << describe(c) << (bad ? "  -- " + *bad : "")
                      << "\n";
            ok += bad ? 0 : 1;
        }
        std::cout << ok << " of " << cfgs.size() << " pass the host prechecks (the DFB host code may still reject some)\n";
        return 0;
    }

    BenchContext ctx = open_device();
    std::ofstream manifest(manifest_path, std::ios::app);
    TT_FATAL(manifest.good(), "cannot open manifest {}", manifest_path);
    uint32_t seq = 0, ran = 0, skipped = 0, failed = 0;
    for (const auto& c : cfgs) {
        if (auto bad = precheck(c)) {
            log_info(tt::LogTest, "SKIP {}: {}", c.name, *bad);
            skipped++;
            continue;
        }
        bool cfg_ok = true;
        for (uint32_t l = 0; l < (compile_only ? 1u : launches) && cfg_ok; l++) {
            try {
                Workload w = make_workload(ctx, c, compile_only);
                BuiltProgram bp = build_program(ctx, c, w);
                if (compile_only) {
                    bp.program.impl().compile(ctx.device);
                    log_info(tt::LogTest, "COMPILED {}", c.name);
                    continue;
                }
                manifest << manifest_entry(ctx, c, bp.program, bp.cores, l, seq) << "\n";
                manifest.flush();
                log_info(tt::LogTest, "RUN  seq={} {} launch {}: {}", seq, c.name, l, describe(c));
                LaunchProgram(*ctx.mesh_device, std::move(bp.program));
                seq++;
                if (w.verify_output) {
                    std::vector<uint32_t> output;
                    slow_dispatch::ReadFromBuffer(w.out_tensor->mesh_buffer(), output);
                    const bool match = output == w.input;
                    log_info(tt::LogTest, "DONE {} launch {}: output {}", c.name, l, match ? "matches input" : "MISMATCH");
                    if (!match) {
                        failed++;
                        cfg_ok = false;
                    }
                } else {
                    log_info(tt::LogTest, "DONE {} launch {}", c.name, l);
                }
            } catch (const std::exception& e) {
                log_info(tt::LogTest, "SKIP {}: host rejected the configuration: {}", c.name, e.what());
                skipped++;
                cfg_ok = false;
            }
        }
        ran += cfg_ok ? 1 : 0;
    }
    MetalContext::instance().rtoptions().set_dfb_force_num_txn_ids(0);
    log_info(
        tt::LogTest, "dfb_isr_timing_bench: {} configurations ran, {} skipped, {} mismatched; manifest {}", ran, skipped,
        failed, manifest_path);
    ctx.mesh_device->close();
    return failed == 0 ? 0 : 2;
}
