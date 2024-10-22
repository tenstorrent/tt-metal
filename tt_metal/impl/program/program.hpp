// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <memory>
#include <optional>

#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/dispatch/program_command_sequence.hpp"
#include "tt_metal/impl/program/program_device_map.hpp"
#include "dev_msgs.h"

namespace tt {

namespace tt_metal {

// Fwd declares
inline namespace v0 {

class Buffer;
class Kernel;
class CircularBuffer;
class Device;
class Program;
class CircularBufferConfig;

}  // namespace v0

class EnqueueProgramCommand;
class HWCommandQueue;
class JitBuildOptions;
namespace detail{
    class Program_;

    void ValidateCircularBufferRegion(const Program &program, const Device *device);
    KernelHandle AddKernel (Program &program, std::shared_ptr<Kernel> kernel, const HalProgrammableCoreType core_type);
    std::shared_ptr<Kernel> GetKernel(const Program &program, KernelHandle kernel_id);
    std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program &program, CBHandle id);
    void AddConfigBuffer(Program &program, std::shared_ptr<Buffer> config_buffer);

    class Internal_;
}

typedef std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX> kernel_id_array_t;

struct KernelGroup {
    uint32_t programmable_core_type_index;
    CoreRangeSet core_ranges;
    kernel_id_array_t kernel_ids;
    uint32_t rta_sizes[DISPATCH_CLASS_MAX];
    uint32_t total_rta_size;
    uint32_t kernel_text_offsets[MaxProcessorsPerCoreType];
    uint32_t kernel_bin_sizes[MaxProcessorsPerCoreType];
    launch_msg_t launch_msg;
    go_msg_t go_msg;

    KernelGroup();
    KernelGroup(
        const detail::Program_ &program,
        uint32_t programmable_core_type_index,
        kernel_id_array_t kernel_ids,
        bool erisc_is_idle,
        int last_cb_index,
        const CoreRangeSet &new_ranges);

    uint32_t get_programmable_core_type_index() const;

    CoreType get_core_type() const;
};

// Contains the program's worker memory map
struct ProgramConfig {
    uint32_t rta_offset;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_offsets;
    std::array<uint32_t, DISPATCH_CLASS_MAX> crta_sizes;
    uint32_t sem_offset;
    uint32_t sem_size;
    uint32_t cb_offset;
    uint32_t cb_size;
    uint32_t kernel_text_offset; // offset of first kernel bin
    uint32_t kernel_text_size;   // max size of all kernel bins across all kernel groups
};

inline namespace v0 {

class Program {
   public:
    Program();

    Program(const Program &other) = delete;
    Program& operator=(const Program &other) = delete;

    Program(Program &&other) noexcept;
    Program& operator=(Program &&other) noexcept;

    void set_runtime_id(uint64_t id);
    ~Program() noexcept;

    uint64_t get_id() const;
    uint64_t get_runtime_id() const;

    size_t num_kernels() const;

    const std::vector<std::shared_ptr<CircularBuffer>> &circular_buffers() const;

    const std::vector< Semaphore > & semaphores() const;

    KernelGroup * kernels_on_core(const CoreCoord &core, uint32_t programmable_core_type_index);
    std::vector<KernelGroup>& get_kernel_groups(uint32_t programmable_core_type_index);
    void add_buffer(std::shared_ptr<Buffer> buf);
    void release_buffers();
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_core(const CoreCoord &core) const;

    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers_on_corerange(const CoreRange &cr) const;

    std::vector<CoreRange> circular_buffers_unique_coreranges() const;

    std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(const CoreCoord &core) const;

    size_t num_semaphores ( const CoreCoord & core ) const;
    size_t num_semaphores () const;
    void init_semaphores ( const Device & device, const CoreCoord &logical_core, uint32_t programmable_core_type_index) const;
    // XXXXX TODO: this should return a const reference
    std::vector<std::vector<CoreCoord>> logical_cores() const;

    void compile(Device * device, bool fd_bootloader_mode = false);

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers(const Device *device);

    bool is_finalized() const;
    bool is_cached() const;
    void set_cached();
    void finalize(Device *device);
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    ProgramConfig& get_program_config(uint32_t programmable_core_type_index);

    // debug/test
    uint32_t get_sem_base_addr(Device *device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(Device *device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(Device *device, CoreCoord logical_core, CoreType core_type) const;
    uint32_t get_cb_size(Device *device, CoreCoord logical_core, CoreType core_type) const;
    void set_last_used_command_queue_for_testing(HWCommandQueue *queue);

    const std::vector<uint32_t> &determine_sub_device_ids(const Device *device);

   private:
    std::unique_ptr<detail::Program_> pimpl_;

    friend CBHandle CreateCircularBuffer(Program &program, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, const CircularBufferConfig &config);
    friend std::shared_ptr<CircularBuffer> detail::GetCircularBuffer(const Program &program, CBHandle id);
    friend void detail::ValidateCircularBufferRegion(const Program &program, const Device *device);

    friend KernelHandle detail::AddKernel(Program &program, std::shared_ptr<Kernel> kernel, const HalProgrammableCoreType core_type);
    friend std::shared_ptr<Kernel> detail::GetKernel(const Program &program, KernelHandle kernel_id);

    friend uint32_t CreateSemaphore(Program &program, const std::variant<CoreRange,CoreRangeSet> &core_spec, uint32_t initial_value, CoreType core_type);

    CBHandle add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);

    void add_semaphore(const CoreRangeSet & crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    friend void detail::AddConfigBuffer(Program &program, std::shared_ptr<Buffer> config_buffer);

    bool runs_on_noc_unicast_only_cores();
    bool runs_on_noc_multicast_only_cores();
    bool kernel_binary_always_stored_in_ringbuffer();

    friend HWCommandQueue;
    friend EnqueueProgramCommand;
    friend detail::Internal_;

    const ProgramTransferInfo &get_program_transfer_info() const noexcept;
    const std::shared_ptr<Buffer> &get_kernels_buffer() const noexcept;
    const std::vector<uint32_t> &get_program_config_sizes() const noexcept;
    std::unordered_map<uint64_t, ProgramCommandSequence> &get_cached_program_command_sequences() noexcept;
};

}  // namespace v0
}  // namespace tt_metal

}  // namespace tt
