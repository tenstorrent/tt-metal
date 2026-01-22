# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TT-Metalium Stable API Headers
set(TT_METAL_STABLE_API
    api/tt-metalium/allocator.hpp
    api/tt-metalium/base_types.hpp
    api/tt-metalium/bfloat16.hpp
    api/tt-metalium/bfloat4.hpp
    api/tt-metalium/bfloat8.hpp
    api/tt-metalium/buffer.hpp
    api/tt-metalium/buffer_distribution_spec.hpp
    api/tt-metalium/buffer_page_mapping.hpp
    api/tt-metalium/buffer_types.hpp
    api/tt-metalium/circular_buffer.hpp
    api/tt-metalium/circular_buffer_config.hpp
    api/tt-metalium/circular_buffer_constants.h
    api/tt-metalium/cluster.hpp
    api/tt-metalium/constants.hpp
    api/tt-metalium/core_coord.hpp
    api/tt-metalium/data_types.hpp
    api/tt-metalium/device.hpp
    api/tt-metalium/dispatch_core_common.hpp
    api/tt-metalium/distributed.hpp
    api/tt-metalium/distributed_context.hpp
    api/tt-metalium/distributed_host_buffer.hpp
    api/tt-metalium/event.hpp
    api/tt-metalium/global_circular_buffer.hpp
    api/tt-metalium/global_semaphore.hpp
    api/tt-metalium/graph_tracking.hpp
    api/tt-metalium/hal.hpp
    api/tt-metalium/hal_types.hpp
    api/tt-metalium/host_api.hpp
    api/tt-metalium/host_buffer.hpp
    api/tt-metalium/kernel_types.hpp
    api/tt-metalium/math.hpp
    api/tt-metalium/maybe_remote.hpp
    api/tt-metalium/memory_pin.hpp
    api/tt-metalium/memory_reporter.hpp
    api/tt-metalium/mesh_buffer.hpp
    api/tt-metalium/mesh_command_queue.hpp
    api/tt-metalium/mesh_config.hpp
    api/tt-metalium/mesh_coord.hpp
    api/tt-metalium/mesh_device.hpp
    api/tt-metalium/mesh_device_view.hpp
    api/tt-metalium/mesh_event.hpp
    api/tt-metalium/mesh_trace_id.hpp
    api/tt-metalium/mesh_workload.hpp
    api/tt-metalium/profiler_optional_metadata.hpp
    api/tt-metalium/profiler_types.hpp
    api/tt-metalium/program.hpp
    api/tt-metalium/program_cache.hpp
    api/tt-metalium/program_descriptors.hpp
    api/tt-metalium/queue_id.hpp
    api/tt-metalium/runtime_args_data.hpp
    api/tt-metalium/shape.hpp
    api/tt-metalium/shape2d.hpp
    api/tt-metalium/shape_base.hpp
    api/tt-metalium/sub_device.hpp
    api/tt-metalium/sub_device_types.hpp
    api/tt-metalium/system_mesh.hpp
    api/tt-metalium/tensor_accessor_args.hpp
    api/tt-metalium/tile.hpp
    api/tt-metalium/tilize_utils.hpp
    api/tt-metalium/tt_align.hpp
    api/tt-metalium/tt_backend_api_types.hpp
    api/tt-metalium/tt_metal.hpp
    api/tt-metalium/tt_metal_profiler.hpp
    api/tt-metalium/vector_aligned.hpp
    api/tt-metalium/work_split.hpp
)

# Load experimental API headers from subdirectory
add_subdirectory(experimental)

# Combine all public API headers
set(TT_METAL_PUBLIC_API
    ${TT_METAL_STABLE_API}
    ${TT_METAL_EXPERIMENTAL_API}
    PARENT_SCOPE
)
