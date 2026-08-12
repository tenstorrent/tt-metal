# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_PATH = REPO_ROOT / "helper_design/mcast_pipe/migration/ledger.json"


def _migrated_sources():
    ledger = json.loads(LEDGER_PATH.read_text())
    kernels = [REPO_ROOT / entry["kernel"] for entry in ledger["entries"] if entry["status"] == "migrated"]
    factories = {
        REPO_ROOT / binding["factory"] for binding in ledger["host_bindings"] if binding["status"] == "migrated"
    }
    return kernels, sorted(factories)


def test_migrated_host_bindings_treat_helper_outputs_as_opaque_ranges():
    _, factories = _migrated_sources()
    declaration = re.compile(r"\b(?:const\s+)?auto\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.DOTALL)

    violations = []
    for factory in factories:
        source = factory.read_text()
        helper_outputs = {
            name
            for name, expression in declaration.findall(source)
            if "compile_time_args(" in expression or "runtime_args(" in expression
        }
        for name in helper_outputs:
            if re.search(rf"\b{re.escape(name)}\s*\[", source):
                violations.append(f"{factory.relative_to(REPO_ROOT)} indexes helper output `{name}`")

    assert not violations, "Helper argument vectors must be copied as complete ranges:\n" + "\n".join(violations)


def test_migrated_kernels_do_not_repeat_rotating_span_as_a_template_argument():
    kernels, _ = _migrated_sources()
    third_template_argument = re.compile(r"\bMcastArgs\s*<\s*[^,>]+\s*,\s*[^,>]+\s*,")

    violations = [
        str(kernel.relative_to(REPO_ROOT)) for kernel in kernels if third_template_argument.search(kernel.read_text())
    ]

    assert not violations, "McastArgs rotating span must come only from the v10 CT wire:\n" + "\n".join(violations)


def test_sort_row_start_readiness_is_pipe_owned():
    kernels, factories = _migrated_sources()
    sort_kernels = [path for path in kernels if "single_row_multi_core.cpp" in path.name]
    sort_factories = [path for path in factories if path.name == "sort_program_factory.cpp"]

    assert len(sort_factories) == 1
    violations = [
        str(path.relative_to(REPO_ROOT)) for path in sort_kernels if "cores_to_coordinator_ready" in path.read_text()
    ]
    factory_source = sort_factories[0].read_text()
    assert ".handshake = true" in factory_source
    assert "row_start_mcast" in factory_source and "substage_mcast" in factory_source
    assert not violations, "Sort row-start readiness must remain inside the handshaked Pipe:\n" + "\n".join(violations)


OPAQUE_BOUNDARY_RULES = {
    "reader_bmm_tile_layout_in1_sender_writer_padding.cpp": [
        (r"rt_args_idx\s*\+=\s*4", "manual runtime skip assumes the helper wire width"),
    ],
    "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp": [
        (r"get_compile_time_arg_val\((?:1[0-9]|2[0-2])\)", "fixed compile-time tail bypasses McastArgs"),
    ],
    "activation_reader_width_sharded.cpp": [
        (
            r"get_compile_time_arg_val\((?:17|18|20|21|22|24|25|26|27|28)\)",
            "fixed compile-time tail bypasses McastArgs",
        ),
        (r"load_config_tensor_if_in_dram<(?:26|27|28)", "fixed config tail bypasses McastArgs"),
    ],
    "reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-2])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|20)\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-5])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-3])\)", "fixed operation tail bypasses the mcast chain"),
    ],
}


@pytest.mark.parametrize("kernel,rules", OPAQUE_BOUNDARY_RULES.items(), ids=OPAQUE_BOUNDARY_RULES)
def test_migrated_kernels_resume_from_named_mcast_boundaries(kernel, rules):
    kernels, _ = _migrated_sources()
    matching_paths = [path for path in kernels if path.name == kernel]
    assert len(matching_paths) == 1, f"Expected one migrated ledger entry for {kernel}, found {matching_paths}"

    source = matching_paths[0].read_text()
    violations = [message for pattern, message in rules if re.search(pattern, source)]
    assert not violations, f"{matching_paths[0].relative_to(REPO_ROOT)}: " + "; ".join(violations)


MATMUL_MCAST_SOURCES = [
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp",
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp",
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
    REPO_ROOT
    / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
    REPO_ROOT
    / "ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp",
]


def test_matmul_migration_has_one_unconditional_mcast_abi():
    violations = [str(path.relative_to(REPO_ROOT)) for path in MATMUL_MCAST_SOURCES if "MCAST_ARGS" in path.read_text()]
    assert not violations, "Matmul must use McastArgs without a conditional legacy ABI:\n" + "\n".join(violations)


def test_matmul_in0_padding_appends_mcast_after_fixed_operation_args():
    kernel = MATMUL_MCAST_SOURCES[2].read_text()
    assert "McastArgs<24, 4>()" in kernel
    assert "TensorAccessorArgs<in0_post_mcast_ct_offset>()" in kernel

    factory_sources = "\n".join(path.read_text() for path in MATMUL_MCAST_SOURCES[:2])
    assert "begin() + 15" not in factory_sources


def test_matmul_1d_in1_bindings_use_helper_generated_blocks_even_when_inactive():
    source = MATMUL_MCAST_SOURCES[0].read_text()
    assert source.count("const auto in1_mcast_compile_time_args = in1_mcast.compile_time_args();") == 4
    assert source.count("const auto in1_mcast_runtime_args = in1_mcast.runtime_args(core);") == 6
    assert "in1_mcast_dest_noc_start_x" not in source
    assert "in1_mcast_num_dests" not in source


def test_sparse_matmul_bindings_preserve_fixed_abi_and_use_inactive_helper_block():
    source = MATMUL_MCAST_SOURCES[4].read_text()

    in0_ct_helper = source.index("const auto in0_mcast_compile_time_args")
    fixed_tail_matches = [
        match.start()
        for match in re.finditer(r"\(std::uint32_t\)false,\s*// fuse_op", source)
        if match.start() < in0_ct_helper
    ]
    assert fixed_tail_matches
    in0_ct_fixed_tail = fixed_tail_matches[-1]
    in0_ct_accessor = source.index("TensorAccessorArgs(*in0_buffer)")
    assert in0_ct_fixed_tail < in0_ct_helper < in0_ct_accessor

    in0_rt_fixed_tail = source.index("(std::uint32_t)sparsity_buffer->address()};")
    in0_rt_helper = source.index("in0_mcast_runtime_args.begin()")
    assert in0_rt_fixed_tail < in0_rt_helper

    assert "const auto in1_mcast_compile_time_args = in1_mcast.compile_time_args();" in source
    assert "const auto in1_mcast_runtime_args = in1_mcast.runtime_args(core);" in source
    assert "in1_mcast_dest_noc_start_x" not in source
    assert "in1_mcast_num_dests" not in source


def test_matmul_mcast_objects_reuse_their_kernel_descriptor_nocs():
    for path in (MATMUL_MCAST_SOURCES[0], MATMUL_MCAST_SOURCES[1], MATMUL_MCAST_SOURCES[4]):
        source = path.read_text()
        assert not re.search(
            r"McastConfig\s*\{\s*\.noc\s*=\s*tt::tt_metal::detail::preferred_noc_for_dram_",
            source,
        ), f"{path.relative_to(REPO_ROOT)} recomputes a NoC inside McastConfig"
        assert "McastConfig{.noc = in0_noc" in source or ".noc = in0_noc," in source
        assert "McastConfig{.noc = in1_noc" in source or ".noc = in1_noc," in source


def test_block_sharded_matmul_keeps_receiver_geometry_separate_from_sender_span():
    factory_1d = MATMUL_MCAST_SOURCES[0].read_text()
    factory_2d = MATMUL_MCAST_SOURCES[1].read_text()

    assert "CoreRange in0_mcast_rect = all_cores_with_work.bounding_box();" in factory_1d
    assert "in0_mcast_senders" in factory_1d
    assert factory_2d.count("device, output_work_grid, in0_tensor.shard_spec()->grid") == 2
    assert factory_2d.count("in0_mcast.participating_cores()") == 2
    assert factory_2d.count("in0_mcast.sender_only_cores()") == 2
    assert "in0_mcast_sender_lines" not in factory_2d
    assert not re.search(r"Mcast[12]D\([^;]+CoreRangeSet\(all_cores\)", factory_1d, re.DOTALL)
    assert not re.search(r"Mcast[12]D\([^;]+CoreRangeSet\(all_cores\)", factory_2d, re.DOTALL)


def test_sender_pipe_degenerate_copy_preserves_async_write_semantics():
    source = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl").read_text()
    start = source.index("    local_copy_(uint32_t src_l1")
    body = source[start : source.index("// ReceiverPipe", start)]
    assert "noc_.async_write(" in body
    assert "async_read" not in body
    assert "barrier" not in body
