// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tt_stl/span.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <unordered_map>

#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_metal::distributed {

class MeshWorkload;
class MeshWorkloadImpl;

class MeshCommandQueue;
class FDMeshCommandQueue;
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

class MeshWorkload {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
    // The current iteration supports the following compute paradigms:
    //  - Single Program Multi Device (Completely Homogeneous MeshWorkload)
    //  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
    // Runtime args can be set per device for a program: see SetRuntimeArgsPerDevice below.
public:
    // Main User-Facing API building blocks
    MeshWorkload();
    ~MeshWorkload();
    MeshWorkload(MeshWorkload&& other) noexcept;
    MeshWorkload& operator=(MeshWorkload&& other) noexcept;
    // Requirement: In one MeshWorkload, device ranges for different programs must not share devices; each device can
    // belong to only one program. Not all devices need a program—leaving some devices unassigned is allowed.
    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs();
    const std::unordered_map<MeshCoordinateRange, Program>& get_programs() const;

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    MeshWorkloadImpl& impl() { return *pimpl_; }

private:
    std::unique_ptr<MeshWorkloadImpl> pimpl_;

    friend void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);
    friend FDMeshCommandQueue;
};

// Per-device runtime args. A Program added to a MeshWorkload on a range runs on every device of the range with the
// same runtime args; this sets chosen words of a kernel's runtime args per device instead (for example a ring rank or
// a peer chip's id), so one program can serve a range where each chip needs a few different values.
//
// - A field is words [first_word, first_word + num_words) of the kernel's unique runtime args on `core` (or of its
//   common runtime args when `core` is nullopt). It must lie within the args the program set there (set them first).
//   Its length is fixed by its first call; a call with the same first_word and another length, or one that overlaps
//   another field of the same args, fails.
// - `values_in_range_order` holds `num_words` words for every coordinate of `program_range`, in the range's iteration
//   order (size == range size * num_words). Calling again replaces the values; setting the same values again does not
//   by itself rebuild the per-device commands at the next enqueue.
// - Words outside every field keep the program's own args, including later changes to them (SetRuntimeArgs /
//   GetRuntimeArgs on the Program).
// - Setting values must be serialized with enqueue / trace capture of the workload.
// - `program_range` names the program by the range it was added with. Values are kept for every coordinate of the
//   range; each host uses its local ones.
// - Supported in fast dispatch (eager and trace) and slow dispatch; not for Metal 2.0 (ProgramSpec) programs (see
//   experimental::SetProgramRunArgsPerDevice), service workloads, emulated devices or light-metal capture.
void SetRuntimeArgsPerDevice(
    MeshWorkload& mesh_workload,
    const MeshCoordinateRange& program_range,
    KernelHandle kernel,
    const std::optional<CoreCoord>& core,
    uint32_t first_word,
    uint32_t num_words,
    ttsl::Span<const uint32_t> values_in_range_order);
// Whether per-device runtime args can be used for a program on `program_range` of `mesh_device`: false for a
// wraparound range, during light-metal capture and on emulated devices. Callers that can fall back (one program per
// device) check it first; SetRuntimeArgsPerDevice and enqueue fail otherwise.
bool PerDeviceRuntimeArgsSupported(MeshDevice& mesh_device, const MeshCoordinateRange& program_range);
}  // namespace tt::tt_metal::distributed
