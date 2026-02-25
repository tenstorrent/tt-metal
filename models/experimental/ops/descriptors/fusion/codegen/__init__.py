# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Codegen subpackage: C++ parsing, source generation, and build orchestration.

Re-exports the public API from submodules so that external code can continue
to use ``from ...fusion.codegen import X``.
"""

import ttnn  # noqa: F401 — exposed as package attribute for test compatibility

from models.experimental.ops.descriptors.fusion.codegen.cpp_parser import (  # noqa: F401
    extract_kernel_body,
    inline_local_includes,
    collect_includes,
    collect_defines,
)
from models.experimental.ops.descriptors.fusion.codegen.source_gen import (  # noqa: F401
    _read_kernel_source,
    _extract_pre_main_text,
    _dedent_ignoring_column_zero,
    _extract_phase_pre_main,
    _prefix_named_args_in_source,
    _offset_compile_time_args_in_source,
    _emit_rt_arg_wrapper,
    _emit_rt_arg_define,
    _emit_rt_arg_undef,
    _transform_phase_source,
    _generate_phase_namespace,
    _strip_include_lines,
    _strip_file_scope_defines,
    _clean_phase_source,
    _offset_tensor_accessor_in_source,
    _generate_phase_block,
    _generate_fused_source,
)
from models.experimental.ops.descriptors.fusion.codegen.barrier import (  # noqa: F401
    _generate_barrier_namespace,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (  # noqa: F401
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
from models.experimental.ops.descriptors.fusion.codegen.builder import (  # noqa: F401
    _build_fused_descriptor,
    _create_phase_info,
    _create_barrier_segment_config,
    _validate_and_get_compute_config_for_role,
)

__all__ = [
    # C++ parsing
    "extract_kernel_body",
    "inline_local_includes",
    "collect_includes",
    "collect_defines",
    # Build orchestration
    "_build_fused_descriptor",
    "_create_phase_info",
    "_create_barrier_segment_config",
]
