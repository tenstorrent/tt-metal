// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Directed op-to-op hazard tests (RAW / WAR / WAW), Metal 2.0 Host API path. These are the
// ground-truth scenarios for the future trace-mode barrier-relaxation optimizer: each is a minimal
// producer/consumer pair with a KNOWN hazard structure, expressed on the ProgramSpec API where the
// hazard detection surface is explicit -- each kernel's buffer usage is declared BY NAME via tensor
// bindings (tensor::src reads / tensor::dst writes), staging goes through a framework-allocated
// scratchpad (no addresses), and scalars are named runtime args. There are no ProgramDescriptors,
// no emplace_runtime_args(Buffer*), no manual MeshBuffer scratch, and no raw addresses (except the
// deliberate bail kernel, whose whole point is to be un-analyzable). All PASS today because full
// op-boundary barriers are in effect (each op is a separate program, enqueued blocking).
//
// Methodology (self-checking, no golden file): a tensor is "doped" with a known pattern D from the
// host, then the producer writes a distinct pattern W and the consumer copies the tensor out for
// host readback. With the barrier a hazard cannot manifest, so the result equals the correct value.
// Once relaxation exists, removing the barrier while the producer is stalled makes the consumer
// observe the wrong pattern -> deterministic corruption (never a hang: the sole inter-op sync is the
// op-boundary barrier; reader on NOC0 / writer on NOC1).

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

#include "command_queue_fixture.hpp"

namespace tt::tt_metal {
namespace {

namespace exp = tt::tt_metal::experimental;

// Inlined from the api-test-local test_helpers.hpp so this dispatch test is self-contained: push a
// TensorBinding{param_name, accessor_name} onto the kernel (the kernel then reaches the bound tensor
// via TensorAccessor(tensor::<accessor_name>)).
void BindTensorParameterToKernel(
    exp::KernelSpec& kernel, std::string tensor_parameter_name, std::string accessor_name) {
    kernel.tensor_bindings.push_back(exp::TensorBinding{
        .tensor_parameter_name = exp::TensorParamName{std::move(tensor_parameter_name)},
        .accessor_name = std::move(accessor_name),
    });
}

// Default tensor: 1x32 BFLOAT16 ROW_MAJOR interleaved = 64 bytes = one page. The scratchpad matches.
constexpr uint32_t kBufBytes = 64;
constexpr uint32_t kWords = kBufBytes / sizeof(uint32_t);  // 16
constexpr uint32_t kDoped = 0xDEADBEEFu;
constexpr uint32_t kWritten = 0xA5A5A5A5u;
constexpr uint32_t kWritten2 = 0x5A5A5A5Au;
constexpr uint32_t kStallSmall = 1000000;  // ~1ms @1GHz: producer holds the buffer long enough that,
                                           // with the barrier gone, the reorder/race is observable

// Writer and reader run on DIFFERENT nodes so they can execute concurrently (same node => same core =>
// serial). Trace replay pipelines the two programs; with the op-boundary barrier relaxed the reader races.
const exp::NodeCoord wNode{0, 0};
const exp::NodeCoord rNode{1, 0};
const exp::NodeCoord mNode{2, 0};  // 3rd distinct core for the middle op of 3-program (WAW / transitive) chains

const char* kWriterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/hazard_writer.cpp";
const char* kReaderKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/hazard_reader.cpp";
const char* kRawWriterKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/hazard_writer_raw.cpp";

MeshTensor alloc(distributed::MeshDevice& md, BufferType bt) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, bt};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    TensorSpec spec(Shape{1, 32}, tensor_layout);
    return MeshTensor::allocate_on_device(md, spec);
}

void dope(const MeshTensor& t, uint32_t pattern) {
    std::vector<uint32_t> v(kWords, pattern);
    detail::WriteToBuffer(*t.mesh_buffer().get_reference_buffer(), v);
}

std::vector<uint32_t> readback(distributed::MeshCommandQueue& cq, const MeshTensor& t) {
    std::vector<uint32_t> v;
    distributed::Finish(cq);
    detail::ReadFromBuffer(*t.mesh_buffer().get_reference_buffer(), v);
    return v;
}

// Analyzable WRITER: scratchpad staging + tensor::dst binding, on NOC1.
Program build_writer(
    distributed::MeshDevice& md, exp::NodeCoord node, const MeshTensor& dst, uint32_t pattern, uint32_t stall) {
    exp::KernelSpec k{
        .unique_id = exp::KernelSpecName{"writer"},
        .source = kWriterKernel,
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"pattern", "stall"}},
        .hw_config = exp::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1},
    };
    k.scratchpad_bindings.push_back(exp::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = exp::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    BindTensorParameterToKernel(k, "dst", "dst");

    exp::ProgramSpec spec{
        .name = "hazard_writer",
        .kernels = {k},
        .scratchpads = {exp::ScratchpadSpec{.unique_id = exp::ScratchpadSpecName{"pad"}, .size_per_node = kBufBytes}},
        .work_units = std::vector<exp::WorkUnitSpec>{exp::WorkUnitSpec{
            .name = "wu", .kernels = {exp::KernelSpecName{"writer"}}, .target_nodes = node}},
    };
    spec.tensor_parameters = {
        exp::TensorParameter{.unique_id = exp::TensorParamName{"dst"}, .spec = dst.tensor_spec()}};

    Program program = exp::MakeProgramFromSpec(md, spec);
    exp::ProgramRunArgs params;
    params.kernel_run_args = {exp::ProgramRunArgs::KernelRunArgs{
        .kernel = exp::KernelSpecName{"writer"},
        .runtime_arg_values = exp::MakeRuntimeArgsForSingleNode(node, {{"pattern", pattern}, {"stall", stall}})}};
    params.tensor_args = {{exp::TensorParamName{"dst"}, exp::TensorArgument{dst}}};
    exp::SetProgramRunArgs(program, params);
    return program;
}

// Analyzable READER: scratchpad staging + tensor::src (read) / tensor::dst (write) bindings, on NOC0.
Program build_reader(
    distributed::MeshDevice& md, exp::NodeCoord node, const MeshTensor& src, const MeshTensor& dst, uint32_t stall) {
    exp::KernelSpec k{
        .unique_id = exp::KernelSpecName{"reader"},
        .source = kReaderKernel,
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"stall"}},
        .hw_config = exp::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
    };
    k.scratchpad_bindings.push_back(exp::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = exp::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    BindTensorParameterToKernel(k, "src", "src");
    BindTensorParameterToKernel(k, "dst", "dst");

    exp::ProgramSpec spec{
        .name = "hazard_reader",
        .kernels = {k},
        .scratchpads = {exp::ScratchpadSpec{.unique_id = exp::ScratchpadSpecName{"pad"}, .size_per_node = kBufBytes}},
        .work_units = std::vector<exp::WorkUnitSpec>{exp::WorkUnitSpec{
            .name = "wu", .kernels = {exp::KernelSpecName{"reader"}}, .target_nodes = node}},
    };
    spec.tensor_parameters = {
        exp::TensorParameter{.unique_id = exp::TensorParamName{"src"}, .spec = src.tensor_spec()},
        exp::TensorParameter{.unique_id = exp::TensorParamName{"dst"}, .spec = dst.tensor_spec()},
    };

    Program program = exp::MakeProgramFromSpec(md, spec);
    exp::ProgramRunArgs params;
    params.kernel_run_args = {exp::ProgramRunArgs::KernelRunArgs{
        .kernel = exp::KernelSpecName{"reader"},
        .runtime_arg_values = exp::MakeRuntimeArgsForSingleNode(node, {{"stall", stall}})}};
    params.tensor_args = {
        {exp::TensorParamName{"src"}, exp::TensorArgument{src}},
        {exp::TensorParamName{"dst"}, exp::TensorArgument{dst}},
    };
    exp::SetProgramRunArgs(program, params);
    return program;
}

// UN-ANALYZABLE WRITER (bail case): NO tensor binding; dst is a raw address via a named RTA + AllocatorBank.
Program build_raw_writer(
    distributed::MeshDevice& md, exp::NodeCoord node, const MeshTensor& dst, uint32_t pattern, uint32_t stall) {
    exp::KernelSpec k{
        .unique_id = exp::KernelSpecName{"raw_writer"},
        .source = kRawWriterKernel,
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "pattern", "stall"}},
        .hw_config = exp::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1},
    };
    k.scratchpad_bindings.push_back(exp::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = exp::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    // NOTE: deliberately NO BindTensorParameterToKernel -> the detector cannot see dst.

    exp::ProgramSpec spec{
        .name = "hazard_raw_writer",
        .kernels = {k},
        .scratchpads = {exp::ScratchpadSpec{.unique_id = exp::ScratchpadSpecName{"pad"}, .size_per_node = kBufBytes}},
        .work_units = std::vector<exp::WorkUnitSpec>{exp::WorkUnitSpec{
            .name = "wu", .kernels = {exp::KernelSpecName{"raw_writer"}}, .target_nodes = node}},
    };

    Program program = exp::MakeProgramFromSpec(md, spec);
    const uint32_t dst_addr = static_cast<uint32_t>(dst.mesh_buffer().get_reference_buffer()->address());
    exp::ProgramRunArgs params;
    params.kernel_run_args = {exp::ProgramRunArgs::KernelRunArgs{
        .kernel = exp::KernelSpecName{"raw_writer"},
        .runtime_arg_values =
            exp::MakeRuntimeArgsForSingleNode(node, {{"dst_addr", dst_addr}, {"pattern", pattern}, {"stall", stall}})}};
    exp::SetProgramRunArgs(program, params);
    return program;
}

// Warm up (compile + cache the kernel binaries, taking the cold-compile sync/stall out of the hot path),
// dope the buffers, then capture the workloads (in order) into ONE trace and replay them pipelined at full
// device speed. Trace replay is the regime where the op-to-op barrier is the only serializer, so with the
// barrier in place a hazard cannot manifest, and with it removed the consumer races. Returns the trace id;
// the caller reads back the result and calls release_mesh_trace(tid).
template <typename DopeFn>
auto warmup_and_replay(
    distributed::MeshDevice& md,
    distributed::MeshCommandQueue& cq,
    std::vector<distributed::MeshWorkload*> wls,
    DopeFn dope_fn) {
    for (auto* wl : wls) {
        distributed::EnqueueMeshWorkload(cq, *wl, /*blocking=*/true);  // warm-up: compile + cache
    }
    dope_fn();  // dope AFTER warm-up (warm-up executed the programs), then land it before capture
    distributed::Finish(cq);
    auto tid = md.begin_mesh_trace(cq);
    for (auto* wl : wls) {
        distributed::EnqueueMeshWorkload(cq, *wl, /*blocking=*/false);  // capture (cache-hit, no new binary)
    }
    md.end_mesh_trace(cq, tid);
    md.replay_mesh_trace(cq, tid, /*blocking=*/true);
    return tid;
}

void expect_all(const std::vector<uint32_t>& v, uint32_t val) {
    ASSERT_EQ(v.size(), kWords);
    for (uint32_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], val) << "mismatch at word " << i;
    }
}

// Gen1 (WH/BH) only, mirroring test_scratchpad_hw.cpp's guard.
bool skip_if_not_gen1(IDevice* device) {
    return device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE;
}

// Confirm a kernel's ELF buffer-R/W info (op-to-op R/W inference). This reads what the loader harvested
// from the compiled .tt.BUF_RW section (see Kernel::query_buf_rw), keyed off the KernelSpec name, so it
// must run after the kernels are compiled (i.e. after warmup_and_replay). The expected R/W set is a
// property of each kernel source + its tensor-binding order, so it is encoded once here per kernel:
//   writer     -> WRITES tensor::dst (slot 0)
//   reader     -> READS tensor::src (slot 0), WRITES tensor::dst (slot 1)
//   raw_writer -> OPAQUE (raw NoC, no binding -> a detector must keep the barrier)
void expect_buf_rw(
    distributed::MeshWorkload& wl,
    const distributed::MeshCoordinateRange& range,
    distributed::MeshDevice& md,
    const std::string& kernel_name) {
    auto kernel = wl.get_programs()[range].impl().get_kernel_by_spec_name(kernel_name);
    ASSERT_NE(kernel, nullptr) << "no kernel '" << kernel_name << "' in workload";
    const ll_api::BufRwInfo rw = kernel->query_buf_rw(*md.get_devices()[0]);
    if (kernel_name == "writer") {
        EXPECT_TRUE(rw.reads.empty()) << "writer reads";
        EXPECT_EQ(rw.writes, (std::set<uint32_t>{0})) << "writer writes tensor::dst";
        EXPECT_FALSE(rw.opaque) << "writer is analyzable";
    } else if (kernel_name == "reader") {
        EXPECT_EQ(rw.reads, (std::set<uint32_t>{0})) << "reader reads tensor::src";
        EXPECT_EQ(rw.writes, (std::set<uint32_t>{1})) << "reader writes tensor::dst";
        EXPECT_FALSE(rw.opaque) << "reader is analyzable";
    } else if (kernel_name == "raw_writer") {
        EXPECT_TRUE(rw.reads.empty()) << "raw_writer reads";
        EXPECT_TRUE(rw.writes.empty()) << "raw_writer writes";
        EXPECT_TRUE(rw.opaque) << "raw_writer is un-analyzable (bail)";
    } else {
        FAIL() << "unexpected kernel name '" << kernel_name << "'";
    }
}

}  // namespace

// RAW hazard, DRAM: producer writes W to DRAM X (stalled); consumer reads X -> Y; barrier => Y == W.
TEST_F(UnitMeshCQSingleCardFixture, RawHazardDram) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    auto y = alloc(*md, BufferType::DRAM);
    distributed::MeshWorkload wwl, rwl;
    wwl.add_program(range, build_writer(*md, wNode, x, kWritten, kStallSmall));
    rwl.add_program(range, build_reader(*md, rNode, x, y, 0));
    auto tid = warmup_and_replay(*md, cq, {&wwl, &rwl}, [&] {
        dope(x, kDoped);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kWritten);
    expect_buf_rw(wwl, range, *md, "writer");
    expect_buf_rw(rwl, range, *md, "reader");
    md->release_mesh_trace(tid);
}

// RAW hazard, L1: producer writes W to L1 X (stalled); consumer reads X -> Y; barrier => Y == W.
TEST_F(UnitMeshCQSingleCardFixture, RawHazardL1) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::L1);
    auto y = alloc(*md, BufferType::DRAM);
    distributed::MeshWorkload wwl, rwl;
    wwl.add_program(range, build_writer(*md, wNode, x, kWritten, kStallSmall));
    rwl.add_program(range, build_reader(*md, rNode, x, y, 0));
    auto tid = warmup_and_replay(*md, cq, {&wwl, &rwl}, [&] {
        dope(x, kDoped);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kWritten);
    expect_buf_rw(wwl, range, *md, "writer");
    expect_buf_rw(rwl, range, *md, "reader");
    md->release_mesh_trace(tid);
}

// RAW-free (disjoint): producer writes X, consumer reads a DISJOINT Z (doped W) -> Y => Y == W; plus a
// serial/overlap timing probe (long stalls on two different nodes) used later to confirm concurrency.
TEST_F(UnitMeshCQSingleCardFixture, RawFreeDisjointAndOverlapProbe) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    auto z = alloc(*md, BufferType::DRAM);
    auto y = alloc(*md, BufferType::DRAM);
    // No hazard: the reader reads a DISJOINT z (doped W), so it stays correct WITH or WITHOUT the barrier
    // (this is the one test that must NOT flip when the barrier is removed).
    distributed::MeshWorkload wwl, rwl;
    wwl.add_program(range, build_writer(*md, wNode, x, kWritten2, kStallSmall));
    rwl.add_program(range, build_reader(*md, rNode, z, y, 0));
    auto tid = warmup_and_replay(*md, cq, {&wwl, &rwl}, [&] {
        dope(z, kWritten);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kWritten);
    expect_buf_rw(wwl, range, *md, "writer");
    expect_buf_rw(rwl, range, *md, "reader");
    md->release_mesh_trace(tid);
    // TODO(op2op): once relaxation lands, add an overlap assertion (trace replay should overlap the two).
}

// WAR hazard: consumer reads X (stalled) -> Y; producer then writes W to X; barrier => Y == D (original).
TEST_F(UnitMeshCQSingleCardFixture, WarHazard) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    auto y = alloc(*md, BufferType::DRAM);
    // Order: reader (stalled) THEN writer. Barrier => reader reads the original D before the writer
    // overwrites x. Without the barrier the writer races ahead and clobbers x -> reader sees W.
    distributed::MeshWorkload rwl, wwl;
    rwl.add_program(range, build_reader(*md, rNode, x, y, kStallSmall));
    wwl.add_program(range, build_writer(*md, wNode, x, kWritten, 0));
    auto tid = warmup_and_replay(*md, cq, {&rwl, &wwl}, [&] {
        dope(x, kDoped);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kDoped);
    expect_buf_rw(rwl, range, *md, "reader");
    expect_buf_rw(wwl, range, *md, "writer");
    md->release_mesh_trace(tid);
}

// WAR-free (disjoint): consumer reads Z (doped W) -> Y; producer writes a DISJOINT X => Y == W.
TEST_F(UnitMeshCQSingleCardFixture, WarFreeDisjoint) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto z = alloc(*md, BufferType::DRAM);
    auto y = alloc(*md, BufferType::DRAM);
    auto x = alloc(*md, BufferType::DRAM);
    // No hazard: reader reads disjoint z, writer writes disjoint x -> correct WITH or WITHOUT the barrier.
    distributed::MeshWorkload rwl, wwl;
    rwl.add_program(range, build_reader(*md, rNode, z, y, 0));
    wwl.add_program(range, build_writer(*md, wNode, x, kWritten2, 0));
    auto tid = warmup_and_replay(*md, cq, {&rwl, &wwl}, [&] {
        dope(z, kWritten);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kWritten);
    expect_buf_rw(rwl, range, *md, "reader");
    expect_buf_rw(wwl, range, *md, "writer");
    md->release_mesh_trace(tid);
}

// WAW: two producers write X (W1 then W2); consumer reads X -> Y; barrier => Y == W2 (last write wins).
TEST_F(UnitMeshCQSingleCardFixture, WawLastWriteWins) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    // WAW is a write-ORDERING hazard on one buffer -- no reader needed. Two writers to x on DIFFERENT
    // cores: w1 (program order first) is SLOW, w2 (second) is fast. Barrier => ordered, last write w2
    // wins (x == W2). Without the barrier w1's stall makes its write land AFTER w2 -> x == W1 (reordered).
    distributed::MeshWorkload w1, w2;
    w1.add_program(range, build_writer(*md, wNode, x, kWritten, kStallSmall));
    w2.add_program(range, build_writer(*md, rNode, x, kWritten2, 0));
    auto tid = warmup_and_replay(*md, cq, {&w1, &w2}, [&] { dope(x, kDoped); });
    expect_all(readback(cq, x), kWritten2);
    expect_buf_rw(w1, range, *md, "writer");
    expect_buf_rw(w2, range, *md, "writer");
    md->release_mesh_trace(tid);
}

// Transitive N/N+1/N+2: N writes X=W (stalled), N+1 touches a disjoint buffer, N+2 reads X -> Z => Z == W.
TEST_F(UnitMeshCQSingleCardFixture, TransitiveSkipDependency) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    auto unrelated = alloc(*md, BufferType::DRAM);
    auto z = alloc(*md, BufferType::DRAM);
    // N writes x (stalled), N+1 writes a disjoint buffer, N+2 reads x. Barrier => N+2 reads W. Without it,
    // N+2 races past the still-in-flight N (skip dependency: N+1 didn't clear N) and reads D.
    distributed::MeshWorkload nwl, midwl, rwl;
    nwl.add_program(range, build_writer(*md, wNode, x, kWritten, kStallSmall));
    midwl.add_program(range, build_writer(*md, mNode, unrelated, kWritten2, 0));
    rwl.add_program(range, build_reader(*md, rNode, x, z, 0));
    auto tid = warmup_and_replay(*md, cq, {&nwl, &midwl, &rwl}, [&] {
        dope(x, kDoped);
        dope(z, kDoped);
    });
    expect_all(readback(cq, z), kWritten);
    expect_buf_rw(nwl, range, *md, "writer");
    expect_buf_rw(midwl, range, *md, "writer");
    expect_buf_rw(rwl, range, *md, "reader");
    md->release_mesh_trace(tid);
}

// Bail: producer has NO tensor binding (raw address + AllocatorBank) -> the framework cannot infer its
// buffer usage, so a future detector must conservatively KEEP the barrier. Passes today.
TEST_F(UnitMeshCQSingleCardFixture, RawHazardFreeFunctionKernelBail) {
    auto md = devices_.at(0);
    if (skip_if_not_gen1(md->get_devices()[0])) {
        GTEST_SKIP() << "requires Wormhole B0 or Blackhole";
    }
    auto& cq = md->mesh_command_queue();
    distributed::MeshCoordinateRange range(md->shape());
    auto x = alloc(*md, BufferType::DRAM);
    auto y = alloc(*md, BufferType::DRAM);
    // Real RAW hazard, but the writer is un-analyzable (raw address, no tensor binding) so a future
    // detector must conservatively KEEP the barrier. Behaves like a hazard here: flips when the barrier
    // is removed (reader races the stalled raw writer and reads D).
    distributed::MeshWorkload wwl, rwl;
    wwl.add_program(range, build_raw_writer(*md, wNode, x, kWritten, kStallSmall));
    rwl.add_program(range, build_reader(*md, rNode, x, y, 0));
    auto tid = warmup_and_replay(*md, cq, {&wwl, &rwl}, [&] {
        dope(x, kDoped);
        dope(y, kDoped);
    });
    expect_all(readback(cq, y), kWritten);
    expect_buf_rw(wwl, range, *md, "raw_writer");
    expect_buf_rw(rwl, range, *md, "reader");
    md->release_mesh_trace(tid);
}

}  // namespace tt::tt_metal
