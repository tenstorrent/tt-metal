# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Metal 2.0 Host API test sources, include()'d from ../sources.cmake and compiled into unit_tests_api.
#
# Layout:
#   unit_tests/invariant_tests/<header>/  which specs are accepted, per public header (mock device)
#   unit_tests/                           other mock-device and host-only unit tests
#   kernel_compilation_tests/             JIT-compile kernels against a mock device
#   integration_tests/                    real Wormhole / Blackhole silicon, or the Quasar emulator
#
# Paths are absolute (${CMAKE_CURRENT_LIST_DIR}) because this list is consumed in
# the parent api/ scope, where a bare relative path would resolve against api/.
list(
    APPEND
    UNIT_TESTS_API_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/advanced_options/dfb_advanced_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/advanced_options/kernel_advanced_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/advanced_options/semaphore_advanced_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/data_movement_hardware_config/config_1xx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/dataflow_buffer_spec/borrowed_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/dataflow_buffer_spec/dataflow_buffer_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/basic_kernel_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/compiler_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/dfb_binding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/hardware_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/kernel_arguments.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/scratchpad_binding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/semaphore_binding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/kernel_spec/tensor_binding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/prefetcher_pipe_parameter/prefetcher_pipe_parameter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/declarations_used.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/dfb_aliasing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/dfb_endpoints.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/dfbs_per_node.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/gen1_dm_placement.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/gen2_dm_core_assignment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/prefetcher_pipe_lanes.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/prefetcher_pipe_relays.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/prefetcher_pipe_roles.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/program_spec_fields.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/references.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/semaphores.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/valid_program_specs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/work_unit_bindings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/work_unit_capacity.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/program_spec/work_unit_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/invariant_tests/scratchpad_spec/scratchpad_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/dataflow_buffer_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/kernel_advanced_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/kernel_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/scratchpad_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/tensor_parameter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/kernel_hash/tensor_spec_relaxations.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/compute_config_lowering.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/graph_tracking.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/kernel_args_layout.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/kernel_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/prefetcher_pipe_slots.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program/semaphore_scope.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/borrowed_dfb_attach.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/dfb_run_overrides.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/kernel_run_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/merge_program_run_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/named_runtime_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/prefetcher_pipe_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/runtime_varargs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/tensor_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/update_program_run_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/program_run_args/update_tensor_args.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/spec_type_properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/tensor_spec_relaxations/tensor_spec_relaxations.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/utility/table.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/resource_bindings/get_token_if_present.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/resource_bindings/llk_operand.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/resource_bindings/scratchpad_bindings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/resource_bindings/tensor_bindings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/kernel_args/compile_time_varargs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_compilation_tests/kernel_args/tt_kernel_shim.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/binding_loopbacks.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/compute_semaphore.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/kernel_args_loopbacks.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/llk_operand_mul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/mesh_workload_factories.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/scratchpad.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/scratchpad_fast_dispatch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/integration_tests/trisc0_rvv_vadd.cpp
)
