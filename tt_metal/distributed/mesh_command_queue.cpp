// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_command_queue.hpp>
#include <mesh_device.hpp>
#include <tt-metalium/dispatch_settings.hpp>

#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal::distributed {

MeshCommandQueue::MeshCommandQueue(MeshDevice* mesh_device, uint32_t id) {
    this->mesh_device_ = mesh_device;
    this->id_ = id;
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        config_buffer_mgr_, expected_num_workers_completed_, DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
    this->populate_virtual_program_dispatch_core();
    this->populate_dispatch_core_type();
}

void MeshCommandQueue::populate_virtual_program_dispatch_core() {
    int device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_ == device->virtual_program_dispatch_core(this->id_),
                "Expected Dispatch Cores to match across devices in a Mesh");
        } else {
            this->dispatch_core_ = device->virtual_program_dispatch_core(this->id_);
        }
        device_idx++;
    }
}

void MeshCommandQueue::populate_dispatch_core_type() {
    uint32_t device_idx = 0;
    for (auto device : this->mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_type_ == dispatch_core_manager::instance().get_dispatch_core_type(device->id()),
                "Expected the Dispatch Core Type to match across device in a Mesh");
        } else {
            this->dispatch_core_type_ = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        }
        device_idx++;
    }
}

CoreCoord MeshCommandQueue::virtual_program_dispatch_core() const { return this->dispatch_core_; }

CoreType MeshCommandQueue::dispatch_core_type() const { return this->dispatch_core_type_; }

void MeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    std::unordered_set<SubDeviceId> sub_device_ids = mesh_workload.determine_sub_device_ids(mesh_device_);
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    auto sub_device_id = *(sub_device_ids.begin());
    auto sub_device_index = sub_device_id.to_index();
    auto mesh_device_id = this->mesh_device_->id();
    auto& sysmem_manager = mesh_device_->get_device(0, 0)->sysmem_manager();

    TT_FATAL(
        mesh_workload.get_program_binary_status(mesh_device_id) != ProgramBinaryStatus::NotSent,
        "Expected program binaries to be written to the MeshDevice.");

    // Compute number of workers being used for this workload.
    uint32_t num_workers = 0;
    bool unicast_go_signals = mesh_workload.runs_on_noc_unicast_only_cores();
    bool mcast_go_signals = mesh_workload.runs_on_noc_multicast_only_cores();
    TT_FATAL(!unicast_go_signals, "Running a MeshWorkload on Ethernet Cores is not supported!");
    TT_ASSERT(
        mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id) == 0,
        "MeshDevice should not report Ethernet Cores.");

    if (mcast_go_signals) {
        num_workers += mesh_device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }

    program_dispatch::ProgramDispatchMetadata dispatch_metadata;
    // Reserve space in the L1 Kernel Config Ring Buffer for this workload.
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->get_config_buffer_mgr(sub_device_index),
        mesh_workload.get_program_config_sizes(),
        mesh_workload.get_program_binary_status(mesh_device_id),
        num_workers,
        expected_num_workers_completed_[sub_device_index],
        dispatch_metadata);

    std::unordered_set<uint32_t> chip_ids_in_workload = {};
    // Iterate over all programs. Update dispatch commands per program to reflect
    // current device state. Write the finalized program command sequence to each
    // physical device tied to the program.
    for (const auto& device_range : mesh_workload.get_logical_device_ranges()) {
        auto& program = mesh_workload.get_program_on_device_range(device_range);
        auto& program_cmd_seq = mesh_workload.get_dispatch_cmds_for_program(program);

        program_dispatch::update_program_dispatch_commands(
            program,
            program_cmd_seq,
            sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].get_mcast_wptr(),
            sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].get_unicast_wptr(),
            expected_num_workers_completed_[sub_device_index],
            this->virtual_program_dispatch_core(),
            this->dispatch_core_type(),
            sub_device_id,
            dispatch_metadata,
            mesh_workload.get_program_binary_status(mesh_device_id),
            std::pair<bool, int>(
                unicast_go_signals,
                mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id)));

        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x + 1;
             logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y + 1;
                 logical_y++) {
                experimental::write_program_commands(
                    this->mesh_device_->get_device(logical_y, logical_x)->command_queue(this->id_),
                    program_cmd_seq,
                    num_workers,
                    sub_device_id,
                    dispatch_metadata.stall_first,
                    dispatch_metadata.stall_before_program,
                    false);
                chip_ids_in_workload.insert(this->mesh_device_->get_device(logical_y, logical_x)->id());
            }
        }
    }
    // Send go signals to devices not running a program to ensure consistent global state
    for (auto& device : this->mesh_device_->get_devices()) {
        if (chip_ids_in_workload.find(device->id()) == chip_ids_in_workload.end()) {
            experimental::write_go_signal(
                device->command_queue(this->id_),
                expected_num_workers_completed_[sub_device_index],
                this->virtual_program_dispatch_core(),
                mcast_go_signals,
                unicast_go_signals,
                mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        }
    }
    // Increment Launch Message Buffer Write Pointers
    if (mcast_go_signals) {
        sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].inc_mcast_wptr(1);
    }
    if (unicast_go_signals) {
        sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].inc_unicast_wptr(1);
    }
    // Update the expected number of workers dispatch must wait on
    expected_num_workers_completed_[sub_device_index] += num_workers;
    // From the dispatcher's perspective, binaries are now committed to DRAM
    mesh_workload.set_program_binary_status(mesh_device_id, ProgramBinaryStatus::Committed);
    mesh_workload.set_last_used_command_queue_for_testing(this);

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::finish() {
    for (auto device : this->mesh_device_->get_devices()) {
        Finish(device->command_queue(this->id_));
    }
}

void MeshCommandQueue::write_shard_to_device(
    std::shared_ptr<Buffer>& shard_view, const void* src, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto device = shard_view->device();
    BufferRegion region(0, shard_view->size());
    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    buffer_dispatch::write_to_device_buffer(
        src, *shard_view, region, id_, expected_num_workers_completed_, this->dispatch_core_type(), sub_device_ids);
}

void MeshCommandQueue::read_shard_from_device(
    std::shared_ptr<Buffer>& shard_view, void* dst, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto device = shard_view->device();
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);

    bool exit_condition = false;

    BufferRegion region(0, shard_view->size());
    if (is_sharded(shard_view->buffer_layout())) {
        auto dispatch_params = buffer_dispatch::initialize_sharded_buf_read_dispatch_params(
            *shard_view, id_, expected_num_workers_completed_, region);
        auto cores = buffer_dispatch::get_cores_for_sharded_buffer(
            dispatch_params.width_split, dispatch_params.buffer_page_mapping, *shard_view);
        for (uint32_t core_id = 0; core_id < shard_view->num_cores(); ++core_id) {
            buffer_dispatch::copy_sharded_buffer_from_core_to_completion_queue(
                core_id, *shard_view, dispatch_params, sub_device_ids, cores[core_id], this->dispatch_core_type());
            if (dispatch_params.pages_per_txn > 0) {
                auto read_descriptor = std::get<tt::tt_metal::ReadBufferDescriptor>(
                    *buffer_dispatch::generate_sharded_buffer_read_descriptor(dst, dispatch_params, *shard_view));
                buffer_dispatch::copy_completion_queue_data_into_user_space(
                    read_descriptor, mmio_device_id, channel, id_, device->sysmem_manager(), exit_condition);
            }
        }
    } else {
        auto dispatch_params = buffer_dispatch::initialize_interleaved_buf_read_dispatch_params(
            *shard_view, id_, expected_num_workers_completed_, region);
        buffer_dispatch::copy_interleaved_buffer_to_completion_queue(
            dispatch_params, *shard_view, sub_device_ids, this->dispatch_core_type());
        if (dispatch_params.pages_per_txn > 0) {
            auto read_descriptor = std::get<tt::tt_metal::ReadBufferDescriptor>(
                *buffer_dispatch::generate_interleaved_buffer_read_descriptor(dst, dispatch_params, *shard_view));
            buffer_dispatch::copy_completion_queue_data_into_user_space(
                read_descriptor, mmio_device_id, channel, id_, device->sysmem_manager(), exit_condition);
        }
    }
}

void MeshCommandQueue::enqueue_write_shard(
    std::shared_ptr<MeshBuffer>& mesh_buffer, const void* host_data, const Coordinate& coord, bool blocking) {
    auto shard = mesh_buffer->get_device_buffer(coord);
    this->write_shard_to_device(shard, host_data);

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_read_shard(
    void* host_data, const std::shared_ptr<MeshBuffer>& mesh_buffer, const Coordinate& coord, bool blocking) {
    TT_FATAL(blocking, "Only blocking reads are currently supported from MeshBuffer shards.");
    auto shard = mesh_buffer->get_device_buffer(coord);
    this->read_shard_from_device(shard, host_data);
}

void MeshCommandQueue::write_sharded_buffer(const MeshBuffer& buffer, const void* src) {
    auto global_buffer_shape = buffer.global_shard_spec().global_buffer_shape;
    auto global_buffer_size = buffer.global_shard_spec().global_size;

    auto shard_shape = buffer.physical_shard_shape();
    auto datum_size_bytes = buffer.datum_size_bytes();

    auto stride_size_bytes = datum_size_bytes * global_buffer_shape.width();
    auto single_read_size = datum_size_bytes * shard_shape.width();
    auto total_read_size_per_shard = single_read_size * shard_shape.height();

    auto num_shards_x = global_buffer_shape.width() / shard_shape.width();
    auto num_shards_y = global_buffer_shape.height() / shard_shape.height();

    uint32_t num_devices_x = buffer.device()->num_cols();
    uint32_t num_devices_y = buffer.device()->num_rows();

    uint32_t device_x = 0;
    uint32_t device_y = 0;
    std::vector<uint32_t> shard_data = std::vector<uint32_t>(total_read_size_per_shard / sizeof(uint32_t), 0);
    const auto& [height_replicated, width_replicated] = buffer.replicated_dims();
    for (std::size_t shard_y = 0; shard_y < num_shards_y; shard_y++) {
        for (std::size_t shard_x = 0; shard_x < num_shards_x; shard_x++) {
            auto read_offset = shard_x * single_read_size + shard_y * stride_size_bytes * shard_shape.height();
            uint32_t size_to_read = total_read_size_per_shard;
            uint32_t local_offset = 0;
            while (size_to_read) {
                std::memcpy(
                    shard_data.data() + local_offset * (single_read_size / sizeof(uint32_t)),
                    (uint8_t*)(src) + read_offset + local_offset * stride_size_bytes,
                    single_read_size);
                size_to_read -= single_read_size;
                local_offset++;
            }

            if (height_replicated and width_replicated) {
                for (std::size_t replicated_device_x = 0; replicated_device_x < num_devices_x; replicated_device_x++) {
                    for (std::size_t replicated_device_y = 0; replicated_device_y < num_devices_y;
                         replicated_device_y++) {
                        auto device_shard_view =
                            buffer.get_device_buffer(Coordinate(replicated_device_y, replicated_device_x));
                        this->write_shard_to_device(device_shard_view, shard_data.data());
                    }
                }
            } else if (height_replicated or width_replicated) {
                if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                    for (auto replicated_device_y = 0; replicated_device_y < num_devices_y; replicated_device_y++) {
                        auto device_shard_view = buffer.get_device_buffer(Coordinate(replicated_device_y, device_x));
                        this->write_shard_to_device(device_shard_view, shard_data.data());
                    }
                    device_x++;
                } else {
                    for (auto replicated_device_x = 0; replicated_device_x < num_devices_x; replicated_device_x++) {
                        auto device_shard_view = buffer.get_device_buffer(Coordinate(device_y, replicated_device_x));
                        this->write_shard_to_device(device_shard_view, shard_data.data());
                    }
                    device_y++;
                }
            } else {
                auto device_shard_view = buffer.get_device_buffer(Coordinate(device_y, device_x));
                this->write_shard_to_device(device_shard_view, shard_data.data());
                if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                    if (++device_x == num_devices_x) {
                        device_x = 0;
                        ++device_y;
                    }
                } else {
                    if (++device_y == num_devices_y) {
                        device_y = 0;
                        ++device_x;
                    }
                }
            }
        }
    }
}

void MeshCommandQueue::read_sharded_buffer(MeshBuffer& buffer, void* dst) {
    const auto& [height_replicated, width_replicated] = buffer.replicated_dims();
    TT_FATAL(
        not(height_replicated or width_replicated), "Cannot read a MeshBuffer that is replicated along any dimension.");
    auto global_buffer_shape = buffer.global_shard_spec().global_buffer_shape;
    auto shard_shape = buffer.physical_shard_shape();
    auto datum_size_bytes = buffer.datum_size_bytes();

    auto stride_size_bytes = datum_size_bytes * global_buffer_shape.width();
    auto single_write_size = datum_size_bytes * shard_shape.width();
    auto total_write_size_per_shard = single_write_size * shard_shape.height();
    auto num_shards_x = global_buffer_shape.width() / shard_shape.width();
    auto num_shards_y = global_buffer_shape.height() / shard_shape.height();
    uint32_t num_devices_x = buffer.device()->num_cols();
    uint32_t num_devices_y = buffer.device()->num_rows();

    uint32_t device_x = 0;
    uint32_t device_y = 0;

    std::vector<uint32_t> shard_data = std::vector<uint32_t>(total_write_size_per_shard / sizeof(uint32_t), 0);
    for (std::size_t shard_y = 0; shard_y < num_shards_y; shard_y++) {
        for (std::size_t shard_x = 0; shard_x < num_shards_x; shard_x++) {
            auto device_shard_view = buffer.get_device_buffer(Coordinate(device_y, device_x));
            this->read_shard_from_device(device_shard_view, shard_data.data());
            uint32_t write_offset = shard_x * single_write_size + shard_y * stride_size_bytes * shard_shape.height();
            uint32_t size_to_write = total_write_size_per_shard;
            uint32_t local_offset = 0;
            while (size_to_write) {
                std::memcpy(
                    (uint8_t*)(dst) + write_offset + local_offset * stride_size_bytes,
                    shard_data.data() + local_offset * (single_write_size / sizeof(uint32_t)),
                    single_write_size);
                local_offset++;
                size_to_write -= single_write_size;
            }
            if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                if (++device_x == num_devices_x) {
                    device_x = 0;
                    ++device_y;
                }
            } else {
                if (++device_y == num_devices_y) {
                    device_y = 0;
                    ++device_x;
                }
            }
        }
    }
}

void MeshCommandQueue::enqueue_write_shard_to_sub_grid(
    const MeshBuffer& buffer, const void* host_data, const LogicalDeviceRange& device_range, bool blocking) {
    if (buffer.global_layout() == MeshBufferLayout::REPLICATED) {
        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x + 1;
             logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y + 1;
                 logical_y++) {
                auto device_shard_view = buffer.get_device_buffer(Coordinate(logical_y, logical_x));
                this->write_shard_to_device(device_shard_view, host_data);
            }
        }
    } else {
        this->write_sharded_buffer(buffer, host_data);
    }
    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_write_mesh_buffer(
    const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) {
    LogicalDeviceRange mesh_device_extent({0, 0}, {buffer->device()->num_cols() - 1, buffer->device()->num_rows() - 1});
    this->enqueue_write_shard_to_sub_grid(*buffer, host_data, mesh_device_extent, blocking);
}

void MeshCommandQueue::enqueue_read_mesh_buffer(
    void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) {
    TT_FATAL(
        buffer->global_layout() == MeshBufferLayout::SHARDED, "Can only read a Sharded MeshBuffer from a MeshDevice.");
    this->read_sharded_buffer(*buffer, host_data);
}

void MeshCommandQueue::reset_worker_state(
    bool reset_launch_msg_state, uint32_t num_sub_devices, const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) {
    for (auto device : mesh_device_->get_devices()) {
        program_dispatch::reset_worker_dispatch_state_on_device(
            mesh_device_,
            device->sysmem_manager(),
            id_,
            this->virtual_program_dispatch_core(),
            expected_num_workers_completed_,
            reset_launch_msg_state);
        program_dispatch::set_num_worker_sems_on_dispatch(mesh_device_, device->sysmem_manager(), id_, num_sub_devices);
        program_dispatch::set_go_signal_noc_data_on_dispatch(
            mesh_device_, go_signal_noc_data, device->sysmem_manager(), id_);
    }
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        config_buffer_mgr_, expected_num_workers_completed_, mesh_device_->num_sub_devices());
    for (auto device : mesh_device_->get_devices()) {
        for (int i = 0; i < mesh_device_->num_sub_devices(); i++) {
            device->command_queue(id_).set_expected_num_workers_completed_for_sub_device(i, 0);
        }
    }
    if (reset_launch_msg_state) {
        auto& sysmem_manager = mesh_device_->get_device(0, 0)->sysmem_manager();
        sysmem_manager.reset_worker_launch_message_buffer_state(num_sub_devices);
    }
}

}  // namespace tt::tt_metal::distributed
