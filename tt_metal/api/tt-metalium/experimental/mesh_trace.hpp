#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::distributed::experimental {

using tt_metal::experimental::KernelSpecName;
using tt_metal::experimental::NodeCoord;
using tt_metal::experimental::ProgramRunArgs;
using tt_metal::experimental::Table;
using tt_metal::experimental::TensorParamName;

// Trace-level names are distinct from the Program parameter names to
// prevent accidentally using one in place of the other.
using TraceTensorArgName = ttsl::StrongType<std::string, struct TraceTensorArgNameTag>;
using TraceRuntimeArgName = ttsl::StrongType<std::string, struct TraceRuntimeArgNameTag>;
using TraceCommonRuntimeArgName = ttsl::StrongType<std::string, struct TraceCommonRuntimeArgNameTag>;

struct TraceTensorArgPath {
    // Identifies a field in a workload program. MeshTraceBuilder::add() maps
    // this program reference to the staged trace-node instance it creates.
    std::reference_wrapper<const Program> program;
    TensorParamName param_name;
};

struct TraceRuntimeArgPath {
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
    // Link a series of nodes to the same trace runtime argument.
    // If empty, a warning is logged.
    std::vector<NodeCoord> nodes;
};

// Common runtime arguments have one value across all nodes, so
// their path does not include a NodeCoord.
struct TraceCommonRuntimeArgPath {
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
};

// Builder-time mappings from trace parameter names to fields in workload
// programs. Multiple paths can map to a single trace parameter, so long as
// they share the same underlying type. add() resolves programs to staged trace
// nodes; build() resolves those node fields to locations in its MeshTrace's
// final DRAM buffer.
struct TraceParameters {
    Table<TraceTensorArgName, std::vector<TraceTensorArgPath>> tensor_parameters;
    Table<TraceRuntimeArgName, std::vector<TraceRuntimeArgPath>> runtime_parameters;
    Table<TraceCommonRuntimeArgName, std::vector<TraceCommonRuntimeArgPath>> common_runtime_parameters;
};

// Used to pass in arguments for updating the values of trace parameters
struct TraceArgPatch {
    Table<TraceTensorArgName, ProgramRunArgs::TensorArgument> tensor_args;
    Table<TraceRuntimeArgName, uint32_t> runtime_args;
    Table<TraceCommonRuntimeArgName, uint32_t> common_runtime_args;
};

// Host-only recorder of workload sequences. Recording never touches the
// device, so any number of builders may record concurrently.
// Throughout this API, validation failures throw and no call partially applies.
class MeshTraceBuilder {
public:
    // Pins the active mesh device and sub-device configurations; builds are
    // rejected if the device's configuration changes before build() is called.
    //
    // Places a lock within the active SubDeviceManager that enforces a single
    // MeshTraceBuilder instance per SubDeviceManager.
    explicit MeshTraceBuilder(MeshDevice& device);

    // movable. A copy is an independent snapshot of the recording;
    // each can be extended separately (e.g. shared prefix, divergent tails).
    MeshTraceBuilder(const MeshTraceBuilder&) = delete;             // TODO: Maybe copyable?
    MeshTraceBuilder& operator=(const MeshTraceBuilder&) = delete;  // TODO: Maybe copyable?
    MeshTraceBuilder(MeshTraceBuilder&&) noexcept;
    MeshTraceBuilder& operator=(MeshTraceBuilder&&) noexcept;
    // Release the lock on the SubDeviceManager.
    ~MeshTraceBuilder();

    // Record one workload iteration. Only enqueues exist here: reads, writes,
    // and events are unrepresentable during recording by construction.
    // The program references within parameters are mapped to the trace-node
    // instances staged for this workload, removing any dependency on the
    // lifetime of the program object.
    // Tensor allocations remain externally managed: the trace records the
    // device addresses used by each program but does not retain tensor buffers.
    // Allocations and frees between recorded workloads are tracked as metadata
    // so their addresses can be validated for safe replay.
    void add(MeshWorkload& workload, const TraceParameters& parameters = {});

    // Assemble the recording, commit it to device DRAM, and bind it to cq_id.
    // Repeatable: each call produces an independent MeshTrace.
    // Trace-parameter mappings are resolved from staged node fields to offsets
    // in the new trace's DRAM buffer. The resulting registry belongs to that
    // MeshTrace, so the builder may be cleared or destroyed independently.
    // No tensor buffer ownership is transferred to the MeshTrace. The allocator
    // may reuse storage between programs, but each recorded or patched address
    // must refer to valid backing storage when its command executes. Raw
    // addresses smuggled in as plain runtime arg values are invisible to
    // allocation tracking. The trace itself lives in the reserved trace region
    // if one is configured, otherwise in regular DRAM.
    //
    // Should build the program binaries and load them onto DRAM as well,
    // removing the need for a warm up.
    MeshTrace build(MeshCommandQueue& cq) const;

    MeshDevice& device() const;

    // Release the lock on the SubDeviceManager, invalidates the MeshTraceBuilder.
    // Any usage of this object afterwards throws. This is to support Python
    // bindings that cannot support RAII patterns.
    void deallocate();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// A replayable trace resident in device DRAM. It owns a registry mapping trace
// parameter names to patch locations in that DRAM buffer, along with allocation-
// safety metadata, but does not own referenced tensor buffers. Move-only RAII
// handle: destruction releases the trace's device buffer and registry.
class MeshTrace {
public:
    // Cannot be copied
    MeshTrace(const MeshTrace&) = delete;
    MeshTrace& operator=(const MeshTrace&) = delete;
    // Movable
    MeshTrace(MeshTrace&&) noexcept;
    MeshTrace& operator=(MeshTrace&&) noexcept;
    // RAII deallocation
    ~MeshTrace();

    // Replay on the CQ this trace was built for. blocking=true waits for the
    // device to finish; blocking=false returns once the replay is issued.
    // void replay(bool blocking) const; // NOTE: Separating out as free function for now to mirror EnqueueMeshWorkload

    // Patch registered parameters for future replays. The trace resolves each
    // name through its own registry and writes the value to every corresponding
    // location in its DRAM buffer. Transactional: unknown names, kind
    // mismatches, or a replay in flight reject the whole patch. Patching records
    // tensor addresses but does not extend tensor-buffer lifetimes. The backing
    // allocation must be valid when replay accesses it.
    void update_args(const TraceArgPatch& patch);

    MeshDevice& device() const;
    uint8_t cq_id() const;  // NOTE: this is still needed, but might be able to move into the impl

    // Manually deallocate the trace from device DRAM. The MeshTrace becomes
    // invalid after this call, and all other functions will throw. This is
    // to support Python bindings that cannot support RAII patterns.
    void deallocate();

private:
    // Only MeshTraceBuilder::build() creates instances.
    friend class MeshTraceBuilder;
    class Impl;
    explicit MeshTrace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Parody with EnqueueMeshWorkload
void EnqueueMeshTrace(MeshCommandQueue& mesh_cq, MeshTrace& mesh_trace, bool blocking);

}  // namespace tt::tt_metal::distributed::experimental
