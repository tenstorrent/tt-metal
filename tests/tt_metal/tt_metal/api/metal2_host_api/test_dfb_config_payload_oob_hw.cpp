// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal reproducer: the per-node dataflow-buffer config payload is sized by a COUNT but
// addressed by the buffer's program-global id, so a program whose nodes carry different buffer
// subsets serializes its high-id buffers out of bounds.
//
// The two disagreeing calculations
// --------------------------------
//   Sizing   (finalize_dfbs, impl/dataflow_buffer/dataflow_buffer.cpp): for each kernel group, sum
//            serialized_size() over the buffers whose range intersects that group, then take the
//            max over groups. That is a COUNT of the buffers present on the busiest group.
//
//   Indexing (assemble_device_commands, impl/program/dispatch.cpp): on Gen1 the write offset is
//            dfb->id * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t), and
//            dfb->id is assigned program-globally at registration (dataflow_buffers_.size()).
//
// The two agree only when some kernel group carries every buffer in the program, which is the case
// for every program in the existing test suite. They diverge as soon as node sets carry different
// subsets, and then any buffer whose global id is >= the busiest group's count is written past the
// end of the payload vector.
//
// The legacy circular-buffer path computes the same quantity correctly, from the highest slot index
// in use rather than a count: see finalize_cbs in impl/program/dispatch.cpp, which folds a per-group
// local_cb_mask and takes the position of its highest set bit.
//
// Shape of this repro
// -------------------
// Two nodes, one data-movement kernel each, one WorkUnitSpec each, and a set of buffers split
// between them. Each kernel self-loops its own buffers (bound PRODUCER and CONSUMER), which is legal
// on Gen1 and is the least machinery needed to give every buffer the producer/consumer pair the
// validator requires. The kernels are empty: the failure is entirely in host-side dispatch-command
// assembly and happens before anything executes on the device.
//
// kNumDfbs = 2 (one per node) is the smallest triggering configuration: the busiest group holds 1
// buffer, so the payload is one slot long, and buffer id 1 is written to slot 1. This test uses more
// than that so the overrun is large enough to fault rather than silently corrupt the heap; see
// kNumDfbs below.
//
// Expected behaviour
// ------------------
//   Debug build   (TT_ASSERT live): dies on the bounds check that already guards the write,
//                 "dfb_byte_offset + serialized.size() <= payload.size()" in dispatch.cpp. This is
//                 the deterministic signal.
//   Release build (TT_ASSERT compiled to (void)(condition)): the out-of-bounds std::copy proceeds
//                 and corrupts whatever heap block follows the payload. Where the process then dies
//                 is allocator-dependent, so the signal varies: observed here as glibc "corrupted
//                 size vs. prev_size" aborting in free() during device teardown, and in the
//                 originally-reported ttnn::sort case as SIGSEGV in the memcpy inside
//                 DeviceCommand::add_dispatch_write_packed_large. Under ASAN it reports at the write
//                 itself.
//
// Because the Release failure point is allocator-dependent, treat a clean Release run as
// inconclusive and the Debug assert as authoritative. A Debug run that does not assert means the
// defect is fixed.
//
// Observed on Wormhole B0 at kNumDfbs = 32: the runtime logs "Finalize dfb: ... dfb size: 256",
// a 16-slot payload, while ids 0-31 address up to byte 512.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "command_queue_fixture.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

// Total buffers, split evenly between the two nodes. The overrun is
// (kNumDfbs - kNumDfbs / 2) slots past a (kNumDfbs / 2)-slot payload, so a larger even value makes
// the fault more reliable without changing the shape being tested. 32 is the Gen1 ceiling on
// buffers per program.
constexpr uint32_t kNumDfbs = 32;
constexpr uint32_t kEntrySize = 32;  // bytes; kept small, the contents are never read
constexpr uint32_t kNumEntries = 2;

class DfbConfigPayloadOobHWTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        auto mesh_device = devices_.at(0);
        IDevice* device = mesh_device->get_devices()[0];
        if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
        if (device->compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Skipping: test requires at least two nodes in a row";
        }
    }
};

// Build a spec whose two nodes carry disjoint halves of the buffer set.
ProgramSpec MakeSplitBufferSpec() {
    const NodeCoord node_a{0, 0};
    const NodeCoord node_b{1, 0};

    auto kernel_a = MakeMinimalGen1DMKernel("kernel_a", DataMovementProcessor::RISCV_0);
    auto kernel_b = MakeMinimalGen1DMKernel("kernel_b", DataMovementProcessor::RISCV_0);

    ProgramSpec spec;
    spec.name = "dfb_config_payload_oob";

    for (uint32_t i = 0; i < kNumDfbs; i++) {
        const std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name, kEntrySize, kNumEntries);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);

        // Low half to node A, high half to node B. Each owning kernel takes both endpoint roles,
        // which is the minimum that satisfies "every buffer needs a producer and a consumer" when
        // only one kernel runs on the node.
        auto& owner = (i < kNumDfbs / 2) ? kernel_a : kernel_b;
        owner.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFBSpecName{name},
            .accessor_name = name,
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        owner.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DFBSpecName{name},
            .accessor_name = name,
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    spec.kernels = {kernel_a, kernel_b};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_a", node_a, {"kernel_a"}),
        MakeMinimalWorkUnit("work_unit_b", node_b, {"kernel_b"}),
    };
    return spec;
}

// The spec itself is valid: this passes, which localizes the defect to dispatch-command assembly
// rather than to spec validation.
TEST_F(DfbConfigPayloadOobHWTest, SplitBufferSetsPassSpecValidation) {
    auto mesh_device = devices_.at(0);
    EXPECT_NO_THROW(
        { distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, MakeSplitBufferSpec()); });
}

// Enqueue is where it breaks. Expected to die (Debug: bounds assert; Release: fault in the packed
// write) until the sizing basis in finalize_dfbs matches the id-based addressing in
// assemble_device_commands.
TEST_F(DfbConfigPayloadOobHWTest, SplitBufferSetsSurviveEnqueue) {
    auto mesh_device = devices_.at(0);
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, MakeSplitBufferSpec());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
