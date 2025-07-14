if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage flags for all tt_metal targets")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

add_library(tt_metal)
add_library(TT::Metalium ALIAS tt_metal)
add_library(Metalium::Metal ALIAS tt_metal) # For backwards compatibility

set_target_properties(
    tt_metal
    PROPERTIES
        EXPORT_NAME
            Metalium
)

include(flatbuffers)
GENERATE_FBS_HEADER(
    api/tt-metalium/serialized_descriptors/mesh_coordinate.fbs
    TARGET TT::Metalium
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/api/tt-metalium/serialized_descriptors/
)
set(API_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})

add_custom_target(
    metalium_GeneratedHeaders
    DEPENDS
        ${API_GENERATED_HEADERS}
    COMMENT "Generating all FlatBuffer headers"
)

target_sources(
    tt_metal
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
        hal.cpp
        ${API_GENERATED_HEADERS}
)

target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api ${CMAKE_CURRENT_BINARY_DIR}/api
        FILES
            api/tt-metalium/allocator.hpp
            api/tt-metalium/allocator_types.hpp
            api/tt-metalium/assert.hpp
            api/tt-metalium/base_types.hpp
            api/tt-metalium/bfloat16.hpp
            api/tt-metalium/bfloat4.hpp
            api/tt-metalium/bfloat8.hpp
            api/tt-metalium/blockfloat_common.hpp
            api/tt-metalium/buffer.hpp
            api/tt-metalium/buffer_distribution_spec.hpp
            api/tt-metalium/buffer_page_mapping.hpp
            api/tt-metalium/buffer_types.hpp
            api/tt-metalium/circular_buffer.hpp
            api/tt-metalium/circular_buffer_constants.h
            api/tt-metalium/circular_buffer_config.hpp
            api/tt-metalium/command_queue.hpp
            api/tt-metalium/command_queue_interface.hpp
            api/tt-metalium/constants.hpp
            api/tt-metalium/core_coord.hpp
            api/tt-metalium/core_descriptor.hpp
            api/tt-metalium/data_format.hpp
            api/tt-metalium/data_types.hpp
            api/tt-metalium/device.hpp
            api/tt-metalium/device_pool.hpp
            api/tt-metalium/dispatch_core_common.hpp
            api/tt-metalium/event.hpp
            api/tt-metalium/fabric_host_interface.h
            api/tt-metalium/fabric_edm_packet_header.hpp
            api/tt-metalium/global_circular_buffer.hpp
            api/tt-metalium/global_semaphore.hpp
            api/tt-metalium/graph_tracking.hpp
            api/tt-metalium/fabric_types.hpp
            api/tt-metalium/hal.hpp
            api/tt-metalium/hal_types.hpp
            api/tt-metalium/hlk_desc.hpp
            api/tt-metalium/host_api.hpp
            api/tt-metalium/kernel.hpp
            api/tt-metalium/kernel_types.hpp
            api/tt-metalium/lightmetal_binary.hpp
            api/tt-metalium/math.hpp
            api/tt-metalium/maybe_remote.hpp
            api/tt-metalium/memory_reporter.hpp
            api/tt-metalium/mesh_config.hpp
            api/tt-metalium/mesh_coord.hpp
            api/tt-metalium/mesh_device.hpp
            api/tt-metalium/mesh_device_view.hpp
            api/tt-metalium/mesh_trace_id.hpp
            api/tt-metalium/metal_soc_descriptor.h
            api/tt-metalium/persistent_kernel_cache.hpp
            api/tt-metalium/profiler_optional_metadata.hpp
            api/tt-metalium/profiler_types.hpp
            api/tt-metalium/program.hpp
            api/tt-metalium/program_cache.hpp
            api/tt-metalium/program_descriptors.hpp
            api/tt-metalium/profiler_optional_metadata.hpp
            api/tt-metalium/profiler_types.hpp
            api/tt-metalium/runtime_args_data.hpp
            api/tt-metalium/semaphore.hpp
            api/tt-metalium/shape.hpp
            api/tt-metalium/shape2d.hpp
            api/tt-metalium/shape_base.hpp
            api/tt-metalium/sub_device.hpp
            api/tt-metalium/sub_device_types.hpp
            api/tt-metalium/system_mesh.hpp
            api/tt-metalium/tile.hpp
            api/tt-metalium/tilize_utils.hpp
            api/tt-metalium/tt_align.hpp
            api/tt-metalium/tt_backend_api_types.hpp
            api/tt-metalium/tt_metal.hpp
            api/tt-metalium/util.hpp
            api/tt-metalium/utils.hpp
            api/tt-metalium/vector_aligned.hpp
            api/tt-metalium/work_split.hpp
            api/tt-metalium/distributed_context.hpp
)

# TODO(afuller): this should be self-describing modules.
#   For now just cherry-pick all the files I discovered empirally by trying to run a test.
add_library(jitapi INTERFACE)
# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    jitapi
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    jitapi
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            api/tt-metalium/circular_buffer_constants.h
            api/tt-metalium/fabric_host_interface.h
            api/tt-metalium/fabric_edm_packet_header.hpp
            api/tt-metalium/fabric_edm_types.hpp
            api/tt-metalium/edm_fabric_counters.hpp
            core_descriptors/blackhole_140_arch.yaml
            core_descriptors/wormhole_b0_80_arch.yaml
            core_descriptors/wormhole_b0_80_arch_eth_dispatch.yaml
            fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml
            fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml
            fabric/hw/inc/fabric_routing_mode.h
            fabric/hw/inc/tt_fabric_utils.h
            fabric/hw/inc/noc_addr.h
            fabric/hw/inc/tt_fabric_mux.hpp
            fabric/hw/inc/tt_fabric_mux_interface.hpp
            fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp
            fabric/hw/inc/edm_fabric/named_types.hpp
            fabric/hw/inc/edm_fabric/1d_fabric_constants.hpp
            fabric/hw/inc/edm_fabric/edm_handshake.hpp
            fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp
            fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp
            fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
            fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
            fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp
            fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp
            fabric/hw/inc/edm_fabric/edm_fabric_tmp_utils.hpp
            impl/dispatch/kernels/cq_commands.hpp
            impl/dispatch/kernels/cq_common.hpp
            impl/dispatch/kernels/cq_relay.hpp
            impl/dispatch/kernels/cq_helpers.hpp
            impl/dispatch/kernels/packet_queue.hpp
            impl/dispatch/kernels/packet_queue_ctrl.hpp
            include/compute_kernel_api.h
            include/compute_kernel_api/bcast.h
            include/compute_kernel_api/blank.h
            include/compute_kernel_api/cb_api.h
            include/compute_kernel_api/common.h
            include/compute_kernel_api/common_globals.h
            include/compute_kernel_api/compute_kernel_hw_startup.h
            include/compute_kernel_api/eltwise_binary.h
            include/compute_kernel_api/eltwise_unary/comp.h
            include/compute_kernel_api/eltwise_unary/eltwise_unary.h
            include/compute_kernel_api/eltwise_unary/exp.h
            include/compute_kernel_api/eltwise_unary/negative.h
            include/compute_kernel_api/eltwise_unary/recip.h
            include/compute_kernel_api/eltwise_unary/relu.h
            include/compute_kernel_api/eltwise_unary/sfpu_split_includes.h
            include/compute_kernel_api/eltwise_unary/sqrt.h
            include/compute_kernel_api/mask.h
            include/compute_kernel_api/matmul.h
            include/compute_kernel_api/pack.h
            include/compute_kernel_api/pack_untilize.h
            include/compute_kernel_api/reconfig_data_format.h
            include/compute_kernel_api/reduce.h
            include/compute_kernel_api/reg_api.h
            include/compute_kernel_api/softmax.h
            include/compute_kernel_api/tile_move_copy.h
            include/compute_kernel_api/tilize.h
            include/compute_kernel_api/transpose_wh.h
            include/compute_kernel_api/untilize.h
            include/compute_kernel_api/eltwise_binary.h
            include/compute_kernel_api/tile_move_copy.h
            soc_descriptors/blackhole_140_arch.yaml
            soc_descriptors/wormhole_b0_80_arch.yaml
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_addr_map.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_addrmod.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_debug.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_defs.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_globals.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_gpr_map.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_include.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_main.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_ops.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_pcbuf.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_sfpi.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_sfpu.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_structs.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_template.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_xmov.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/cmath_common.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/cunpack_common.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_abs.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_activations.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_add_int.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bitwise.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_cdf.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_converter.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_cumsum.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_load_config.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_max.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_max_int32.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_mul_int.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_polyval.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_negative.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_power.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_quant.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rounding_ops.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sigmoid.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sign.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_square.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sub_int.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_tanh.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_tanh_derivative.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_trigonometry.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h
            third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_defs.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_common.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpi.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_common.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_reduce.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_tilize.h
            third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_untilize.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_addr_map.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_addrmod.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_debug.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_globals.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_gpr_map.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_include.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_main.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_ops.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_pcbuf.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_sfpi.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_sfpu.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_structs.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_template.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_xmov.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cmath_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cpack_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_abs.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_activations.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_add_int.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary_bitwise.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_cdf.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_cumsum.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_load_config.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_log.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max_int32.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_mul_int.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_negative.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_power.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_quant.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_reshuffle_rows.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rounding_ops.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sigmoid.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sign.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_square.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sub_int.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_tanh.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_tanh_derivative.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_topk.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_trigonometry.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h
            third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_datacopy.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpi.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_transpose_dest.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_A.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB_matmul.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_common.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_reduce.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h
            third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_untilize.h
            tools/profiler/kernel_profiler.hpp
            # Kernel sources
            impl/dispatch/kernels/cq_dispatch.cpp
            impl/dispatch/kernels/cq_dispatch_subordinate.cpp
            impl/dispatch/kernels/cq_prefetch.cpp
            impl/dispatch/kernels/packet_demux.cpp
            impl/dispatch/kernels/packet_mux.cpp
            impl/dispatch/kernels/vc_eth_tunneler.cpp
            impl/dispatch/kernels/vc_packet_router.cpp
            fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp
            fabric/impl/kernels/tt_fabric_mux.cpp
            kernels/compute/blank.cpp
            kernels/compute/eltwise_binary.cpp
            kernels/compute/eltwise_sfpu.cpp
            kernels/dataflow/blank.cpp
            kernels/dataflow/reader_binary_diff_lengths.cpp
            kernels/dataflow/reader_unary.cpp
            kernels/dataflow/writer_unary.cpp
            kernels/dataflow/writer_unary_1.cpp
            jit_build/genfiles.cpp
            jit_build/genfiles.hpp
)

target_link_libraries(
    tt_metal
    PUBLIC
        umd::device
        magic_enum::magic_enum
        fmt::fmt-header-only
        TracyClient
        nlohmann_json::nlohmann_json
        TT::Metalium::HostDevCommon
        Reflect::Reflect
        TT::STL
        tt-logger::tt-logger
    PRIVATE
        Metalium::Metal::Impl
        metal_common_libs
        common
        jit_build
        llrt
        detail
        distributed
        fabric
        HAL::wormhole
        HAL::blackhole
        FlatBuffers::FlatBuffers
        TT::Metalium::Logging
)

target_precompile_headers(tt_metal REUSE_FROM TT::CommonPCH)

target_include_directories(
    tt_metal
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/api>"
    PRIVATE
        api/tt-metalium # FIXME: Re-home the tests and remove this
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "/opt/openmpi-v5.0.7-ulfm/lib;${PROJECT_BINARY_DIR}/lib;$ORIGIN"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()

# Allow internal files to access the public API "by default" and without the
# scoping that external consumers must use.  Scoping may still be used if desired.
include_directories(
    api
    api/tt-metalium
    ${CMAKE_CURRENT_BINARY_DIR}/api
)

add_subdirectory(logging)
add_subdirectory(hw)
add_subdirectory(hostdevcommon)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(fabric)
if(TT_METAL_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(TT_INSTALL)
    install(
        TARGETS
            tt_metal
        EXPORT Metalium
        LIBRARY
            COMPONENT metalium-runtime
        FILE_SET
        api
            DESTINATION COMPONENT
            metalium-dev
    )
    install(EXPORT Metalium NAMESPACE TT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium COMPONENT metalium-dev)

    install(
        TARGETS
            jitapi
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )

    # 3rd party dependencies we export in our public API.
    # We must ship not only these libraries, but their public headers as well
    # ... or refactor our public API.
    install(
        TARGETS
            reflect
            magic_enum
            TracyClient
            span
            small_vector
            tt-logger
        EXPORT Metalium
        FILE_SET
        api
            COMPONENT metalium-dev
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty
    )
endif()
