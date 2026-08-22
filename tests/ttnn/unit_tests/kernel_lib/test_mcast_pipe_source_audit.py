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


def test_conv3d_mcast_preserves_fixed_abi_and_scoped_weight_share_modes():
    factory = (REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp").read_text()
    kernel = (REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp").read_text()

    assert "using WeightMcastArgs = McastArgs<bias_args.next_compile_time_args_offset(), 19>;" in kernel
    assert "argidx = WeightMcastArgs::next_runtime_args_offset();" in kernel
    assert "writer_args.append(weights_mcast_runtime_args);" in factory
    assert "weights_mcast_template.runtime_args(core)" in factory
    assert "mcast_bbox_start_x" not in factory + kernel
    assert "mcast_num_dests" not in factory + kernel


def test_sender_pipe_degenerate_copy_preserves_async_write_semantics():
    source = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl").read_text()
    start = source.index("    local_copy_(uint32_t src_l1")
    body = source[start : source.index("// ReceiverPipe", start)]
    assert "noc_.async_write(" in body
    assert "async_read" not in body
    assert "barrier" not in body


def test_interleaved_groupnorm_uses_three_opaque_helper_wires_and_preserves_legacy_phases():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/normalization/groupnorm/device"
    factories = [
        (base / "groupnorm_mcast_program_factory.cpp").read_text(),
        (base / "groupnorm_no_mcast_program_factory.cpp").read_text(),
    ]
    kernel_base = base / "kernels/dataflow"
    legacy_sender = (kernel_base / "reader_mcast_sender_unary_gn.cpp").read_text()
    legacy_receiver = (kernel_base / "reader_mcast_receiver_unary_gn.cpp").read_text()
    welford_sender = (kernel_base / "welford_reader_mcast_sender_unary_gn.cpp").read_text()
    welford_receiver = (kernel_base / "welford_reader_mcast_receiver_unary_gn.cpp").read_text()

    for source in (legacy_sender, legacy_receiver, welford_sender, welford_receiver):
        assert source.count("using ") >= 3
        assert "LastMcastArgs::next_compile_time_args_offset()" in source
        assert "LastMcastArgs::next_runtime_args_offset()" in source
        assert "mcast_dest_noc" not in source
        assert "num_mcast_cores_mid_group" not in source

    assert legacy_sender.count("_pipe.send_signal();") == 3
    assert legacy_sender.count("_pipe.send(l1_read_addr_ex") == 3
    assert "reduce_pipe.receive_signal();" in legacy_receiver
    assert "reduce_pipe.receive();" in legacy_receiver
    assert "reduce_receiver_sem.up(" in legacy_receiver
    assert welford_sender.count("_pipe.send(global_means_ptr") == 3
    assert "reduce_pipe.receive();" in welford_receiver

    mcast_factory, no_mcast_factory = factories
    assert "compile_time_args(/*pre_handshake=*/use_welford)" in mcast_factory
    assert mcast_factory.count("reader_args.append(mcast.runtime_args(core));") == 1
    assert "compile_time_args(/*pre_handshake=*/false)" in no_mcast_factory
    assert no_mcast_factory.count("reader_args.append(mcast.runtime_args(core));") == 1
    for source in factories:
        assert "std::vector<ttnn::kernel_lib::host::Mcast2D>" in source
        assert "mcast_dest_noc" not in source


def test_sdpa_decode_read_k_uses_opaque_fixed_star_and_keeps_blackhole_completion():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device"
    common = (base / "kernels/dataflow/dataflow_common.hpp").read_text()
    reader = (base / "kernels/dataflow/reader_decode_all.cpp").read_text()
    factory = (base / "sdpa_decode_program_factory.cpp").read_text()
    read_k = common[
        common.index("uint32_t read_k(") : common.index("// Non-multicast path", common.index("uint32_t read_k("))
    ]

    assert "async_write_multicast" not in read_k
    assert "SourceL1Guard::CallerManaged" in read_k
    assert "k_pipe.receive();" in read_k
    assert "noc.async_write_barrier();" in read_k
    assert "noc.async_atomic_barrier();" in read_k
    assert "McastArgs<32, 16>" in reader
    assert "KMcastArgs::next_runtime_args_offset()" in reader
    assert "mcast_x" not in reader + factory
    assert "std::vector<ttnn::kernel_lib::host::Mcast2D> k_mcasts" in factory
    assert "reader_rt_args.append(k_mcasts[mcast_index].runtime_args(core));" in factory


def test_argmax_multicore_composes_two_counter_wires_and_keeps_done_fanin():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/reduction/argmax/device"
    kernel = (base / "kernels/reader_argmax_interleaved_multicore.cpp").read_text()
    factory = (base / "argmax_multi_core_program_factory.cpp").read_text()

    assert "McastArgs<18, 7>" in kernel
    assert "group0_start_args.next_compile_time_args_offset()" in kernel
    assert "group0_start_args.next_runtime_args_offset()" in kernel
    assert kernel.count("send_signal();") == 2
    assert kernel.count("start_receiver->receive_signal();") == 2
    assert "set_multicast" not in kernel
    assert "done_sem.up(" in kernel and "done_sem.wait(num_cores)" in kernel
    assert factory.count("DataReadyMode::Counter") == 1
    assert factory.count("reader_runtime_args.append(group0_start_mcast.runtime_args(core));") == 2
    assert factory.count("reader_runtime_args.append(group1_start_mcast.runtime_args(core));") == 2


def test_move_overlap_composes_three_release_wires_and_keeps_return_counter():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/move/device"
    kernels = [
        (base / "kernels/dataflow/move_interleaved_with_overlap.cpp").read_text(),
        (base / "kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp").read_text(),
    ]
    factory = (base / "move_overlap_program_factory.cpp").read_text()
    cache_override = (base / "move_sharded_program_factory.cpp").read_text()

    for kernel in kernels:
        assert kernel.count("McastArgs<") == 3
        assert kernel.count("send_signal();") == 3
        assert kernel.count("receive_signal();") == 3
        assert "return_sem.up(" in kernel and "return_sem.wait(num_workers)" in kernel
        assert "set_multicast" not in kernel

    assert "std::vector<ttnn::kernel_lib::host::Mcast2D> release_mcasts" in factory
    assert factory.count("mcast.compile_time_args()") == 1
    assert factory.count("runtime_args.append(mcast.runtime_args(core));") == 1
    assert "mcast_dest_noc" not in factory
    assert "case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP" in cache_override
    assert "a[0] = src_addr;" in cache_override and "a[1] = dst_addr;" in cache_override
