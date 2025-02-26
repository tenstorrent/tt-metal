// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_command_queue.hpp>
#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <optional>
#include <tt-metalium/dispatch_settings.hpp>

#include "buffer.hpp"
#include "mesh_coord.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal::distributed {

struct MeshReadEventDescriptor {
    ReadEventDescriptor single_device_descriptor;
    MeshCoordinateRange device_range;
};

MeshCommandQueue::MeshCommandQueue(MeshDevice* mesh_device, uint32_t id, std::shared_ptr<ThreadPool> thread_pool) :
    thread_pool_(thread_pool) {
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
    auto& sysmem_manager = this->reference_sysmem_manager();
    auto dispatch_core_config = DispatchQueryManager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();

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
    uint32_t expected_num_workers_completed = sysmem_manager.get_bypass_mode()
                                                  ? trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores
                                                  : expected_num_workers_completed_[sub_device_index];
    // Reserve space in the L1 Kernel Config Ring Buffer for this workload.
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->get_config_buffer_mgr(sub_device_index),
        mesh_workload.get_program_config_sizes(),
        mesh_workload.get_program_binary_status(mesh_device_id),
        num_workers,
        expected_num_workers_completed,
        dispatch_metadata);

    std::unordered_set<uint32_t> chip_ids_in_workload = {};
    std::vector<MeshCoordinateRange> active_sub_grids = {};
    // Iterate over all programs. Update dispatch commands per program to reflect
    // current device state. Write the finalized program command sequence to each
    // physical device tied to the program.
    for (auto& [device_range, program] : mesh_workload.get_programs()) {
        auto& program_cmd_seq = mesh_workload.get_dispatch_cmds_for_program(program);
        program_dispatch::update_program_dispatch_commands(
            program,
            program_cmd_seq,
            sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].get_mcast_wptr(),
            sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].get_unicast_wptr(),
            expected_num_workers_completed,
            this->virtual_program_dispatch_core(),
            dispatch_core_type,
            sub_device_id,
            dispatch_metadata,
            mesh_workload.get_program_binary_status(mesh_device_id),
            std::pair<bool, int>(
                unicast_go_signals,
                mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id)));

        if (sysmem_manager.get_bypass_mode()) {
            this->capture_program_trace_on_subgrid(
                device_range, program_cmd_seq, dispatch_metadata.stall_first, dispatch_metadata.stall_before_program);
            active_sub_grids.push_back(device_range);
        } else {
            this->write_program_cmds_to_subgrid(
                device_range,
                program_cmd_seq,
                dispatch_metadata.stall_first,
                dispatch_metadata.stall_before_program,
                chip_ids_in_workload);
        }
    }
    // Send go signals to devices not running a program to ensure consistent global state
    if (not sysmem_manager.get_bypass_mode()) {
        this->write_go_signal_to_unused_sub_grids(
            chip_ids_in_workload, sub_device_id, expected_num_workers_completed, mcast_go_signals, unicast_go_signals);
    } else {
        MeshCoordinateRangeSet active_sub_grids_set;
        for (const auto& sub_grid : active_sub_grids) {
            active_sub_grids_set.merge(sub_grid);
        }
        TT_FATAL(active_sub_grids_set.size() == 1, "Cannot support non convex grids.");
        this->capture_go_signal_trace_on_unused_subgrids(
            active_sub_grids_set.ranges().front(),
            sub_device_id,
            expected_num_workers_completed,
            mcast_go_signals,
            unicast_go_signals);
    }
    // Increment Launch Message Buffer Write Pointers
    if (mcast_go_signals) {
        sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].inc_mcast_wptr(1);
    }
    if (unicast_go_signals) {
        sysmem_manager.get_worker_launch_message_buffer_state()[sub_device_index].inc_unicast_wptr(1);
    }

    if (sysmem_manager.get_bypass_mode()) {
        if (mcast_go_signals) {
            // The workload contains programs that required a go signal mcast. Capture this here
            // to accurately update the launch msg ring buffer state post trace execution on all
            // mcast cores.
            trace_ctx_->descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
        }
        // Update the expected number of workers dispatch must wait on
        trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores += num_workers;
    } else {
        expected_num_workers_completed_[sub_device_index] += num_workers;
    }
    // From the dispatcher's perspective, binaries are now committed to DRAM
    mesh_workload.set_program_binary_status(mesh_device_id, ProgramBinaryStatus::Committed);
    mesh_workload.set_last_used_command_queue_for_testing(this);

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    std::shared_ptr<MeshEvent> event = std::make_shared<MeshEvent>();
    this->enqueue_record_event_to_host(event, sub_device_ids);
    this->drain_events_from_completion_queue();
    this->verify_reported_events_after_draining(event);
}

void MeshCommandQueue::write_shard_to_device(
    std::shared_ptr<Buffer>& shard_view,
    const void* src,
    const BufferRegion& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto device = shard_view->device();
    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    buffer_dispatch::write_to_device_buffer(
        src, *shard_view, region, id_, expected_num_workers_completed_, this->dispatch_core_type(), sub_device_ids);
}

void MeshCommandQueue::read_shard_from_device(
    std::shared_ptr<Buffer>& shard_view,
    void* dst,
    const BufferRegion& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    this->drain_events_from_completion_queue();
    auto device = shard_view->device();
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);

    bool exit_condition = false;

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
                            buffer.get_device_buffer(MeshCoordinate(replicated_device_y, replicated_device_x));
                        const BufferRegion region(0, device_shard_view->size());
                        this->write_shard_to_device(device_shard_view, shard_data.data(), region);
                    }
                }
            } else if (height_replicated or width_replicated) {
                if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                    for (auto replicated_device_y = 0; replicated_device_y < num_devices_y; replicated_device_y++) {
                        auto device_shard_view =
                            buffer.get_device_buffer(MeshCoordinate(replicated_device_y, device_x));
                        const BufferRegion region(0, device_shard_view->size());
                        this->write_shard_to_device(device_shard_view, shard_data.data(), region);
                    }
                    device_x++;
                } else {
                    for (auto replicated_device_x = 0; replicated_device_x < num_devices_x; replicated_device_x++) {
                        auto device_shard_view =
                            buffer.get_device_buffer(MeshCoordinate(device_y, replicated_device_x));
                        const BufferRegion region(0, device_shard_view->size());
                        this->write_shard_to_device(device_shard_view, shard_data.data(), region);
                    }
                    device_y++;
                }
            } else {
                auto device_shard_view = buffer.get_device_buffer(MeshCoordinate(device_y, device_x));
                const BufferRegion region(0, device_shard_view->size());
                this->write_shard_to_device(device_shard_view, shard_data.data(), region);
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
            auto device_shard_view = buffer.get_device_buffer(MeshCoordinate(device_y, device_x));
            const BufferRegion region(0, device_shard_view->size());
            this->read_shard_from_device(device_shard_view, shard_data.data(), region);

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
    const MeshBuffer& buffer,
    const void* host_data,
    const MeshCoordinateRange& device_range,
    bool blocking,
    std::optional<BufferRegion> region) {
    if (buffer.global_layout() == MeshBufferLayout::REPLICATED) {
        // Multi-Threaded writes supported for Replicated buffers.
        // Currently not supported when doing TT-Mesh Native sharding, since we
        // rely on TTNN to perform sharding and call enqueue_write_shards
        auto dispatch_lambda =
            std::function<void(MeshCoordinate)>([this, &buffer, host_data, &region](MeshCoordinate&& coord) {
                auto device_shard_view = buffer.get_device_buffer(coord);
                const BufferRegion buffer_region = region.value_or(BufferRegion(0, device_shard_view->size()));
                this->write_shard_to_device(device_shard_view, host_data, buffer_region);
            });

        for (const auto& coord : device_range) {
            thread_pool_->enqueue([&dispatch_lambda, coord]() { dispatch_lambda(std::move(coord)); });
        }
        thread_pool_->wait();
    } else {
        this->write_sharded_buffer(buffer, host_data);
    }

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_write_mesh_buffer(
    const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) {
    MeshCoordinateRange mesh_device_extent(buffer->device()->shape());
    this->enqueue_write_shard_to_sub_grid(*buffer, host_data, mesh_device_extent, blocking);
}

void MeshCommandQueue::enqueue_read_mesh_buffer(
    void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) {
    TT_FATAL(
        buffer->global_layout() == MeshBufferLayout::SHARDED, "Can only read a Sharded MeshBuffer from a MeshDevice.");
    this->read_sharded_buffer(*buffer, host_data);
}

void MeshCommandQueue::enqueue_write_shards(
    const std::shared_ptr<MeshBuffer>& buffer,
    const std::vector<ShardDataTransfer>& shard_data_transfers,
    bool blocking) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.

    auto dispatch_lambda =
        std::function<void(uint32_t)>([&shard_data_transfers, &buffer, this](uint32_t shard_idx) {
            auto& shard_data_transfer = shard_data_transfers[shard_idx];
            auto device_shard_view = buffer->get_device_buffer(shard_data_transfer.shard_coord);
            this->write_shard_to_device(
                device_shard_view,
                shard_data_transfer.host_data,
                shard_data_transfer.region.value_or(BufferRegion(0, device_shard_view->size())));
        });

    for (std::size_t shard_idx = 0; shard_idx < shard_data_transfers.size(); shard_idx++) {
        thread_pool_->enqueue([&dispatch_lambda, shard_idx]() { dispatch_lambda(shard_idx); });
    }
    thread_pool_->wait();

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_read_shards(
    const std::vector<ShardDataTransfer>& shard_data_transfers,
    const std::shared_ptr<MeshBuffer>& buffer,
    bool blocking) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.
    auto dispatch_lambda =
        std::function<void(uint32_t)>([&shard_data_transfers, &buffer, this](uint32_t shard_idx) {
            auto& shard_data_transfer = shard_data_transfers[shard_idx];
            auto device_shard_view = buffer->get_device_buffer(shard_data_transfer.shard_coord);
            read_shard_from_device(
                device_shard_view,
                shard_data_transfer.host_data,
                shard_data_transfer.region.value_or(BufferRegion(0, device_shard_view->size())));
        });

    for (std::size_t shard_idx = 0; shard_idx < shard_data_transfers.size(); shard_idx++) {
        thread_pool_->enqueue([&dispatch_lambda, shard_idx]() { dispatch_lambda(shard_idx); });
    }
    thread_pool_->wait();

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::enqueue_record_event_helper(
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    bool notify_host,
    const std::optional<MeshCoordinateRange>& device_range) {
    auto& sysmem_manager = this->reference_sysmem_manager();
    event->cq_id = id_;
    event->event_id = sysmem_manager.get_next_event(id_);
    event->device = mesh_device_;
    event->device_range = device_range.value_or(MeshCoordinateRange(mesh_device_->shape()));

    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    for (const auto& coord : event->device_range) {
        event_dispatch::issue_record_event_commands(
            mesh_device_,
            event->event_id,
            id_,
            mesh_device_->num_hw_cqs(),
            mesh_device_->get_device(coord)->sysmem_manager(),
            sub_device_ids,
            expected_num_workers_completed_,
            notify_host);
    }
}

void MeshCommandQueue::enqueue_record_event(
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<MeshCoordinateRange>& device_range) {
    this->enqueue_record_event_helper(event, sub_device_ids, false, device_range);
}

void MeshCommandQueue::enqueue_record_event_to_host(
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<MeshCoordinateRange>& device_range) {
    this->enqueue_record_event_helper(event, sub_device_ids, true, device_range);
    event_descriptors_.push(std::make_shared<MeshReadEventDescriptor>(MeshReadEventDescriptor{
        .single_device_descriptor = ReadEventDescriptor(event->event_id), .device_range = event->device_range}));
}

void MeshCommandQueue::enqueue_wait_for_event(const std::shared_ptr<MeshEvent>& sync_event) {
    for (const auto& coord : sync_event->device_range) {
        event_dispatch::issue_wait_for_event_commands(
            id_, sync_event->cq_id, mesh_device_->get_device(coord)->sysmem_manager(), sync_event->event_id);
    }
}

void MeshCommandQueue::drain_events_from_completion_queue() {
    constexpr bool exit_condition = false;
    auto num_events = event_descriptors_.size();
    for (std::size_t event_idx = 0; event_idx < num_events; event_idx++) {
        auto& mesh_read_descriptor = event_descriptors_.front();
        auto& device_range = mesh_read_descriptor->device_range;
        for (const auto& coord : device_range) {
            auto device = mesh_device_->get_device(coord);
            chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
            uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
            bool exit_condition = false;
            device->sysmem_manager().completion_queue_wait_front(id_, exit_condition);

            event_dispatch::read_events_from_completion_queue(
                mesh_read_descriptor->single_device_descriptor, mmio_device_id, channel, id_, device->sysmem_manager());
        }
        event_descriptors_.pop();
    }
}

void MeshCommandQueue::verify_reported_events_after_draining(const std::shared_ptr<MeshEvent>& event) {
    auto& device_range = event->device_range;
    for (const auto& coord : device_range) {
        TT_FATAL(
            mesh_device_->get_device(coord)->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id,
            "Expected to see event id {} in completion queue",
            event->event_id);
    }
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
    if (reset_launch_msg_state) {
        auto& sysmem_manager = this->reference_sysmem_manager();
        sysmem_manager.reset_worker_launch_message_buffer_state(num_sub_devices);
    }
}

void MeshCommandQueue::write_program_cmds_to_subgrid(
    const MeshCoordinateRange& sub_grid,
    ProgramCommandSequence& program_cmd_seq,
    bool stall_first,
    bool stall_before_program,
    std::unordered_set<uint32_t>& chip_ids_in_workload) {
    auto dispatch_core_config = DispatchQueryManager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();

    for (const auto& coord : sub_grid) {
        program_dispatch::write_program_command_sequence(
            program_cmd_seq,
            this->mesh_device_->get_device(coord)->sysmem_manager(),
            id_,
            dispatch_core_type,
            stall_first,
            stall_before_program);
        chip_ids_in_workload.insert(this->mesh_device_->get_device(coord)->id());
    }
}

void MeshCommandQueue::write_go_signal_to_unused_sub_grids(
    std::unordered_set<uint32_t>& chip_ids_in_workload,
    const SubDeviceId& sub_device_id,
    uint32_t expected_num_workers_completed,
    bool mcast_go_signals,
    bool unicast_go_signals) {
    for (auto& device : this->mesh_device_->get_devices()) {
        if (chip_ids_in_workload.find(device->id()) == chip_ids_in_workload.end()) {
            write_go_signal(
                id_,
                mesh_device_,
                sub_device_id,
                device->sysmem_manager(),
                expected_num_workers_completed,
                this->virtual_program_dispatch_core(),
                mcast_go_signals,
                unicast_go_signals,
                mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        }
    }
}

void MeshCommandQueue::capture_program_trace_on_subgrid(
    const MeshCoordinateRange& sub_grid,
    ProgramCommandSequence& program_cmd_seq,
    bool stall_first,
    bool stall_before_program) {
    auto& sysmem_manager_for_trace = mesh_device_->get_device(sub_grid.start_coord())->sysmem_manager();
    uint32_t sysmem_manager_offset = sysmem_manager_for_trace.get_issue_queue_write_ptr(id_);

    auto dispatch_core_config = DispatchQueryManager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();

    program_dispatch::write_program_command_sequence(
        program_cmd_seq, sysmem_manager_for_trace, id_, dispatch_core_type, stall_first, stall_before_program);
    auto mesh_trace_md = MeshTraceStagingMetadata{
        sub_grid,
        sub_grid.start_coord(),
        sysmem_manager_offset,
        sysmem_manager_for_trace.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
    ordered_mesh_trace_md_.push_back(mesh_trace_md);
}

void MeshCommandQueue::capture_go_signal_trace_on_unused_subgrids(
    const MeshCoordinateRange& active_grid,
    const SubDeviceId& sub_device_id,
    uint32_t expected_num_workers_completed,
    bool mcast_go_signals,
    bool unicast_go_signals) {
    MeshCoordinateRange full_grid(mesh_device_->shape());
    MeshCoordinateRangeSet unused_grids = subtract(full_grid, active_grid);
    for (const auto& unused_grid : unused_grids.ranges()) {
        auto& sysmem_manager_for_trace = mesh_device_->get_device(unused_grid.start_coord())->sysmem_manager();
        uint32_t sysmem_manager_offset = sysmem_manager_for_trace.get_issue_queue_write_ptr(id_);
        write_go_signal(
            id_,
            mesh_device_,
            sub_device_id,
            sysmem_manager_for_trace,
            expected_num_workers_completed,
            this->virtual_program_dispatch_core(),
            mcast_go_signals,
            unicast_go_signals,
            mesh_device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id));
        auto mesh_trace_md = MeshTraceStagingMetadata{
            unused_grid,
            unused_grid.start_coord(),
            sysmem_manager_offset,
            sysmem_manager_for_trace.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
        ordered_mesh_trace_md_.push_back(mesh_trace_md);
    }
}

void MeshCommandQueue::enqueue_trace(const MeshTraceId& trace_id, bool blocking) {
    auto trace_inst = mesh_device_->get_mesh_trace(trace_id);
    auto descriptor = trace_inst->desc;
    auto buffer = trace_inst->mesh_buffer;
    uint32_t num_sub_devices = descriptor->sub_device_ids.size();

    auto cmd_sequence_sizeB = trace_dispatch::compute_trace_cmd_size(num_sub_devices);

    trace_dispatch::TraceDispatchMetadata dispatch_md(
        cmd_sequence_sizeB,
        descriptor->descriptors,
        descriptor->sub_device_ids,
        buffer->page_size(),
        buffer->num_pages(),
        buffer->address());

    for (auto device : mesh_device_->get_devices()) {
        trace_dispatch::issue_trace_commands(
            mesh_device_, device->sysmem_manager(), dispatch_md, id_, expected_num_workers_completed_, dispatch_core_);
    }
    trace_dispatch::update_worker_state_post_trace_execution(
        trace_inst->desc->descriptors,
        this->reference_sysmem_manager(),
        config_buffer_mgr_,
        expected_num_workers_completed_);

    if (blocking) {
        this->finish();
    }
}

void MeshCommandQueue::record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) {
    trace_dispatch::reset_host_dispatch_state_for_trace(
        mesh_device_->num_sub_devices(),
        this->reference_sysmem_manager(),
        expected_num_workers_completed_,
        config_buffer_mgr_,
        worker_launch_message_buffer_state_reset_,
        expected_num_workers_completed_reset_,
        config_buffer_mgr_reset_);

    trace_id_ = trace_id;
    trace_ctx_ = ctx;
    for (auto device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ true, /*clear*/ true);
    }
}

void MeshCommandQueue::record_end() {
    trace_ctx_->assemble_dispatch_commands(this->device(), ordered_mesh_trace_md_);
    trace_id_ = std::nullopt;
    trace_ctx_ = nullptr;

    trace_dispatch::load_host_dispatch_state(
        mesh_device_->num_sub_devices(),
        this->reference_sysmem_manager(),
        expected_num_workers_completed_,
        config_buffer_mgr_,
        worker_launch_message_buffer_state_reset_,
        expected_num_workers_completed_reset_,
        config_buffer_mgr_reset_);

    ordered_mesh_trace_md_.clear();
    for (auto device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ false, /*clear*/ true);
    }
}

SystemMemoryManager& MeshCommandQueue::reference_sysmem_manager() {
    return mesh_device_->get_device(0, 0)->sysmem_manager();
}

}  // namespace tt::tt_metal::distributed
