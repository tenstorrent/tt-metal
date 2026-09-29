#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

using distributed::MeshCommandQueue;
using distributed::MeshDevice;
using distributed::MeshWorkload;

using tt_metal::experimental::KernelSpecName;
using tt_metal::experimental::NodeCoord;
using tt_metal::experimental::ProgramRunArgs;
using tt_metal::experimental::Table;
using tt_metal::experimental::TensorParamName;

class CommandList;

// Command-list parameter names are distinct from the Program parameter names to
// prevent accidentally using one in place of the other.
using CmdListTensorArgName = ttsl::StrongType<std::string, struct CmdListTensorArgNameTag>;
using CmdListRuntimeArgName = ttsl::StrongType<std::string, struct CmdListRuntimeArgNameTag>;
using CmdListCommonRuntimeArgName = ttsl::StrongType<std::string, struct CmdListCommonRuntimeArgNameTag>;

struct CmdListTensorArgInfo {
    // Identifies a field in a workload program. CommandListBuilder::add()
    // resolves the reference so the builder can outlive the Program handle.
    std::reference_wrapper<const Program> program;
    TensorParamName param_name;
};

struct CmdListRuntimeArgInfo {
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
    // Link a series of nodes to the same command list runtime argument.
    // Must not be empty.
    std::vector<NodeCoord> nodes;
};

// Common runtime arguments have one value across all nodes, so
// their info does not include a NodeCoord.
struct CmdListCommonRuntimeArgInfo {
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
};

// Builder-time mappings from command list parameter names to fields in workload
// programs. Multiple info objects can map to a single command list parameter, so
// long as they share the same underlying type. add() resolves programs to staged
// command-list nodes; build() resolves those fields to locations in its final
// DRAM buffer.
struct CmdListParameters {
    Table<CmdListTensorArgName, std::vector<CmdListTensorArgInfo>> tensor_parameters;
    Table<CmdListRuntimeArgName, std::vector<CmdListRuntimeArgInfo>> runtime_parameters;
    Table<CmdListCommonRuntimeArgName, std::vector<CmdListCommonRuntimeArgInfo>> common_runtime_parameters;
};

// Used to pass in arguments for updating the values of command list parameters
struct CmdListArgPatch {
    Table<CmdListTensorArgName, ProgramRunArgs::TensorArgument> tensor_args;
    Table<CmdListRuntimeArgName, uint32_t> runtime_args;
    Table<CmdListCommonRuntimeArgName, uint32_t> common_runtime_args;
};

// Recorder of workload sequences. Throughout this API, validation failures
// throw and no call partially applies.
class CommandListBuilder {
public:
    // Captures the active sub-device manager. add() and build() require it to
    // be active.
    explicit CommandListBuilder(MeshDevice& device);

    CommandListBuilder(const CommandListBuilder&) = delete;
    CommandListBuilder& operator=(const CommandListBuilder&) = delete;
    CommandListBuilder(CommandListBuilder&&) noexcept;
    CommandListBuilder& operator=(CommandListBuilder&&) noexcept;
    ~CommandListBuilder();

    // Record one workload iteration. Only enqueues exist here: reads, writes,
    // and events are unrepresentable during recording by construction.
    // Compiles/finalizes the workload if needed, commits its kernel binaries
    // through the current thread's default CQ, and waits for those writes. It
    // does not launch the workload.
    // The program references within parameters are mapped to the command list-node
    // instances staged for this workload, removing any dependency on the
    // lifetime of the program object.
    // Tensor allocations remain externally managed: the command list records raw
    // device addresses but retains neither tensor objects nor their allocations.
    // Releasing an allocation after add() does not alter its recorded address.
    void add(MeshWorkload& workload, const CmdListParameters& parameters = {});

    // Assemble the recording, commit it to device DRAM, and bind it to cq_id.
    // Repeatable: each call produces an independent CommandList.
    // Command list-parameter mappings are resolved from staged node fields to offsets
    // in the new command list's DRAM buffer. The resulting registry belongs to that
    // CommandList, so the builder may be cleared or destroyed independently.
    // No tensor buffer ownership is transferred to the CommandList. Replay issues
    // recorded addresses without consulting allocator ownership, so tensor objects
    // and allocations may be released after recording. Subsequent allocations may
    // reuse or overwrite that storage without updating the CommandList; managing
    // recorded address contents remains the caller's responsibility. The same applies
    // to raw addresses supplied as plain runtime argument values.
    // The command list itself lives in the reserved command list region if one
    // is configured, otherwise in regular DRAM.
    CommandList build(MeshCommandQueue& cq) const;

    MeshDevice& device() const;

    // Clears all recorded workloads.
    void clear();

    // Releases the builder lock and invalidates the builder. Repeated calls
    // have no effect.
    void deallocate();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// A replayable command list resident in device DRAM. It owns a registry mapping
// command list parameter names to patch locations in that DRAM buffer, but does
// not own referenced tensor buffers. Move-only RAII handle: destruction releases
// the command list's device buffer and registry.
class CommandList {
public:
    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList(CommandList&&) noexcept;
    CommandList& operator=(CommandList&&) noexcept;
    ~CommandList();

    // Replays on the command queue used by build(). The recorded sub-device
    // manager must be active. If blocking, waits for completion.
    void replay(bool blocking) const;

    // Patch registered parameters for future replays. The command list resolves each
    // name through its own registry and writes the value to every corresponding
    // location in its DRAM buffer. Transactional: every patch is validated before
    // any write is issued. The bound CQ is synchronized before patching. Patching
    // records raw tensor addresses but retains neither the tensor object nor its
    // allocation. Releasing a patched tensor does not alter the recorded address.
    void update_args(const CmdListArgPatch& patch);

    MeshDevice& device() const;

    uint8_t cq_id() const;

    // Releases device resources and invalidates the handle. Repeated calls
    // have no effect.
    void deallocate();

private:
    friend class CommandListBuilder;
    class Impl;
    explicit CommandList(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Replays command_list on cq, which must match the device and queue used to
// build it.
void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking);

}  // namespace tt::tt_metal::experimental
