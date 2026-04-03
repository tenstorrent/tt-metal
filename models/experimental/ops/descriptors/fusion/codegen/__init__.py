# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Codegen subpackage: C++ parsing, source generation, and build orchestration.

Re-exports the public API from submodules so that external code can continue
to use ``from ...fusion.codegen import X``.
"""

import ttnn

from models.experimental.ops.descriptors.fusion.codegen.cpp_parser import (
    extract_kernel_body,
    inline_local_includes,
    collect_includes,
    collect_defines,
)
from models.experimental.ops.descriptors.fusion.codegen.source_gen import (
    _read_kernel_source,
    _offset_compile_time_args_in_source,
    _emit_rt_arg_wrapper,
    _emit_rt_arg_define,
    _emit_rt_arg_undef,
    _strip_include_lines,
    _strip_file_scope_defines,
    _clean_phase_source,
    _generate_phase_block,
    _generate_fused_source,
    _generate_coordinator_only_source,
)
from models.experimental.ops.descriptors.fusion.codegen.barrier import (
    _generate_barrier_namespace,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (
    _get_core_coords_from_ranges,
    _compute_and_concatenate_runtime_args,
    _append_barrier_runtime_args,
    _concatenate_common_runtime_args,
    _merge_named_compile_time_args,
    _merge_compile_time_args,
    _collect_phase_defines,
    _emit_define_lines,
    _emit_undef_lines,
    _validate_fp32_consistency,
)
from models.experimental.ops.descriptors.fusion.codegen.builder import (
    _build_fused_descriptor,
    _build_coordinator_only_kernel,
    _create_phase_info,
    _create_barrier_segment_config,
    _validate_and_get_compute_config_for_role,
)

__all__ = [
    "ttnn",
    # C++ parsing
    "extract_kernel_body",
    "inline_local_includes",
    "collect_includes",
    "collect_defines",
    # Source generation
    "_read_kernel_source",
    "_offset_compile_time_args_in_source",
    "_emit_rt_arg_wrapper",
    "_emit_rt_arg_define",
    "_emit_rt_arg_undef",
    "_strip_include_lines",
    "_strip_file_scope_defines",
    "_clean_phase_source",
    "_generate_phase_block",
    "_generate_fused_source",
    "_generate_coordinator_only_source",
    # Barrier
    "_generate_barrier_namespace",
    # Args
    "_get_core_coords_from_ranges",
    "_compute_and_concatenate_runtime_args",
    "_append_barrier_runtime_args",
    "_concatenate_common_runtime_args",
    "_merge_named_compile_time_args",
    "_merge_compile_time_args",
    "_collect_phase_defines",
    "_emit_define_lines",
    "_emit_undef_lines",
    "_validate_fp32_consistency",
    # Builder
    "_build_fused_descriptor",
    "_build_coordinator_only_kernel",
    "_create_phase_info",
    "_create_barrier_segment_config",
    "_validate_and_get_compute_config_for_role",
]
